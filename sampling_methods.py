import torch
import torch.nn.functional as F
import random
import math

# (logits,top_p=1.0, randomspace_rate=0.1, preserve_largest_prob=1, top_k=5,noise_weight=0.1, MaxEntropy=0.1, filter_value=-float("Inf"), min_tokens_to_keep=1)

def top_p_transform(logits,top_p=1.0, filter_value=0, min_tokens_to_keep=1):  # -float("Inf") don't need softmax
    """
    :param logits:  shape (batch size, vocabulary size)
    :param top_p: nucleus filtering
    :param filter_value: transform the probability
    :param min_tokens_to_keep:  maintain the min tokens
    :return: top_p logits
    """
    if top_p <= 1.0:  #  top_k==None and
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p  # total probability
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)  # 不足的地方自动置换为0
        logits[indices_to_remove] = filter_value
    return logits

def random_mask_transform(logits, randomspace_rate=0.1, top_k=None, preserve_largest_prob=1):  # don't need softmax
    """
    :param logits:  shape (batch size, vocabulary size)
    :param top_k: top-k filtering
    :param randomspace_rate: the least ratio to mask
    :param preserve_largest_prob:  maintain the large probability tokens
    :return: random_mask logits
    """
    tt = torch.sort(logits, descending=True)
    if top_k is not None:
        val, ind = tt[0][:, :top_k], tt[1][:, :top_k]
        assert (val.size(1) == top_k)
    else:
        val, ind = tt[0], tt[1]

    least_value = 0  # torch.min(val).item() - 1000
    logits.fill_(least_value)  # [[-999.xxx],[]]
    ra = torch.ones(val.size()).cuda().uniform_()
    if preserve_largest_prob == 1:  # preserve the max token
        ra[:, 0] = 2
    bo = ra < randomspace_rate
    val[bo] = least_value
    logits = logits.scatter(1, ind, val)
    return logits




def  noised_top_k_transform(logits,top_k=5,noise_weight=0.1, device='cuda'):
    """
    :param logits:  shape (batch size, vocabulary size)
    :param top_k: top-k filtering
    :param randomspace_rate: the least ratio to mask
    :param preserve_largest_prob:  maintain the large probability tokens
    :return: random_mask logits
    """
    noise_weight = float(noise_weight)
    assert (noise_weight < 1.0)
    #topk_logits, topk_indices = torch.sort(logits, descending=True)
    tt = torch.sort(logits, descending=True)
    val, ind = tt[0][:, :top_k], tt[1][:, :top_k]
    probs = torch.softmax(val, dim=-1) #.detach()
    #print(probs)
    unif = torch.ones(logits.size(0), top_k).uniform_().to(device)  # .cuda() # .pin_memory().to(torch.device) #.torch  #to(torch.device) 
    #print("unif is " + str(unif))
    #unif = torch.zeros(logits.size(0), top_k).uniform_().to(torch.device) #.cuda()
    unif[unif < 1e-09] = 1e-09
    unif[unif > 1 - 1e-09] = 1 - 1e-09
    log_unif = - torch.log(unif)
    unif_simplex = log_unif / torch.sum(log_unif, dim=-1).view(-1, 1)
    #print(unif_simplex)
    sort_s = torch.sort(unif_simplex, dim=-1, descending=True)[0]  # .cuda()  # different .cuda() torch.Tensor
    #print(sort_s)
    probs = probs *(1 - noise_weight) + (noise_weight) * sort_s
    probs = probs / probs.sum(dim=-1).view(-1, 1)
    topk_logits = torch.log(probs)
    least_value = logits.min().item() - 1000
    logits.fill_(least_value)
    #print(ind.dtype)
    #print(topk_logits.dtype)
    #topk_logits = topk_logits.type(torch.float16)
    #print(logits.dtype)
    #print(topk_logits.dtype)
    logits = logits.scatter(1, ind, topk_logits)
    return logits.detach()


'''
def  TargetEntropy_transform(logits, noise_weight=0.1, preserve_largest_prob=1, top_k=5):
'''

def compute_entropy(logits):
    distro = torch.softmax(logits, dim=-1)
    entropy = - torch.sum(distro * torch.log(distro + 1e-10), dim=- 1)
    return entropy

def  MaxEntropy_transform(logits, MaxEntropy=0.1, device='cuda'):
    max_entropy = float(MaxEntropy)
    e_max = torch.zeros(logits.size(0)).to(device)
    scale = torch.ones(logits.size(0)).to(device)
    e_max.fill_(max_entropy)
    e_cur = compute_entropy(logits).squeeze()
    for kk in range(30):
        scale.fill_(1)
        ss = 1.3
        if kk > 20: ss = 1.1
        scale[e_cur > e_max * ss] = ss
        logits = (logits * scale.unsqueeze(1)).detach()
        e_cur = compute_entropy(logits).squeeze()
        
    #if random.random() < 0.005:
        #print('random debug target entropy gap:', ((e_cur - e_max).abs() * (e_cur > e_max).float()).max())
    return logits.detach()


def TargetEntropy_transform(logits, TargetEntropy=0.1, device='cuda'):
    Target_Entropy = float(TargetEntropy)
    e_tar = torch.zeros(logits.size(0)).to(device)
    scale = torch.ones(logits.size(0)).to(device)
    e_tar.fill_(Target_Entropy)
    e_cur = compute_entropy(logits).squeeze()
    for kk in range(30):
        scale.fill_(1)
        ss = 1.2 if kk < 10 else 1.02
        if kk > 20: ss = 1.002
        scale[e_cur < e_tar] = 1 / ss
        scale[e_cur > e_tar] = ss
        logits = (logits * scale.unsqueeze(1)).detach()
        e_cur = compute_entropy(logits).squeeze()

    # if random.random() < 0.005:
    # print('random debug target entropy gap:', ((e_cur - e_max).abs() * (e_cur > e_max).float()).max())
    return logits.detach()

def  sin_transform(logits):

    logits = torch.sin(logits)

    return logits

def  tanh_transform(logits):
    logits = torch.tanh(logits)
    return logits

def  square_transform(logits):
    logits = torch.pow(logits, 2)
    return logits

def  cube_transform(logits):
    logits = torch.pow(logits, 3)
    return logits




