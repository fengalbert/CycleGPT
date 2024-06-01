import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from lion_macro_model import GPTConfig, GPT

from torch.nn import functional as F
from data.chembl_add_drugbank.prepare_smiles import encode, decode, indices_token, special_token, token_indices
import time
import warnings
from sampling_methods import top_p_transform, random_mask_transform, noised_top_k_transform, \
    MaxEntropy_transform, sin_transform, tanh_transform, square_transform, cube_transform
# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# out_dir = f'/data2/fhu/nanoGPT-master/out_sample'  # ignored if init_from is not 'resume'  
resume_checkpoint = './model/lion_macro/macro_pretrain.pt'

sampling_path = f'/data2/fhu/nanoGPT-master/out_sample/sampling/lion_macro_fine_tune_from_transform/'   # {macro_name}
os.makedirs(sampling_path, exist_ok=True)

model_name = resume_checkpoint.split('/')[-1]
epoch_name = model_name.split('_')[1]

start = 'G'  #  special_token['start_char']  # "\n" #   or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
end = 'E'  # special_token['end_char']

num_samples = 30000  

max_new_tokens = 142 

use_temperature = 1
temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions 
top_k = 5  #  200  retain only the top_k most likely tokens, clamp others to have 0 probability  
transform_sampling = 'tanh_temp'
top_p = 0.9
randomspace_rate = 0.1
preserve_largest_prob = 1
noise_weight = 0.1
MaxEntropy = 0.1


seed = 1337  
device = 'cuda:1'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)  

def save_obj(obj, name):
    """save obj with pickle"""
    name = name.replace('.pkl', '')  
    with open(name + '.pkl', 'wb') as f:  
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

start_time = time.time()

if init_from == 'resume':
    # init from a model saved in a specific directory

    checkpoint = torch.load(resume_checkpoint, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model  
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# encode the beginning of the prompt
if start.startswith('FILE:'):  # rompt.txt
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

if init_from == 'resume' and 'config' in checkpoint:  # and 'dataset' in checkpoint['config']
    print("we assuming trained GPT encodings...")
    #start_strs_list = encode(start)  # ['36']
    #start_ids = [int(i) for i in start_strs_list]#(i for i in start_strs_list)start_strs_list[:]
    #end_str_list = encode(end)
    #end_ids = [int(i) for i in end_str_list]
    encode = lambda s: [int(token_indices[c]) for c in s]  # [stoi[c] for c in s]indices_token  
    decode = lambda l: ''.join([indices_token[str(i)] for i in l])  # ''.join([itos[i] for i in l])token_indices


    #encode = lambda s: encode(s)
    #decode = lambda l: decode(l)
    #start_ids = int(encode(start))  # \n
    #end_ids = int(encode(end))

else:
    # ok let's assume gpt-2 encodings by default
    print("we assuming GPT-2 encodings...")  # gpt2
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    #start_ids = encode(start)
    #end_ids = encode(end)

start_ids = encode(start)
end_ids = encode(end)
print(start_ids)
print(end_ids)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])  # x
end_inds = (torch.tensor(end_ids, dtype=torch.long, device=device)[None, ...])


def sample(idx, max_new_tokens, transform_sampling, device):
    for _ in range(max_new_tokens):
        logits = model.generate(idx, end_inds, max_new_tokens, temperature)
        logits = logits[:, -1, :]
        logits = F.softmax(logits, dim=-1)  
        if transform_sampling is not None:
            if transform_sampling == 'top_p_transform':
                logits = top_p_transform(logits, top_p=top_p, filter_value=-float("Inf"), min_tokens_to_keep=1)
            elif transform_sampling == 'random_mask_transform': 
                logits = random_mask_transform(logits, randomspace_rate=randomspace_rate, top_k=top_k,
                                               preserve_largest_prob=1)
            elif transform_sampling == 'noised_top_k_transform':
                logits = noised_top_k_transform(logits, top_k=top_k, noise_weight=noise_weight, device=device)
            elif transform_sampling == 'MaxEntropy_transform':
                logits = MaxEntropy_transform(logits, MaxEntropy=MaxEntropy)
            elif transform_sampling == 'temp_topk':
                logits = torch.log(logits) / temperature
                logits = F.softmax(logits, dim=-1)
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = 0#-float('Inf')
            elif transform_sampling == 'tanh_temp_topk':
                logits = torch.log(logits) / temperature  
                logits = F.softmax(logits, dim=-1)
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = 0 #-float('Inf')
                logits = torch.tanh(logits)
            elif transform_sampling == 'tanh_temp':
                logits = torch.log(logits) / temperature  
                logits = F.softmax(logits, dim=-1)
                #v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                #logits[logits < v[:, [-1]]] = 0#-float('Inf')
                logits = torch.tanh(logits)
        #if use_temperature is not None:
            #logits = torch.log(logits) / temperature  # torch.log(logits) / temperature
            #print(logits)

        #probs = F.softmax(logits, dim=-1)
        
        #print(probs)
        probs = logits 
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next == end_inds:
            break
    return idx


if num_samples > 5000:
    warnings.warn('The macrocycle sampling number was set to more than 5000')
# run generation
with torch.no_grad():
    with ctx:
        generated_macro = []
        generated_percentage = 0
        start_sampling = time.time()
        print(f'Start sampling the macrocycle at transform_sampling : {transform_sampling}, temperature: {temperature}')
        for k in range(num_samples):
            y = sample(x, max_new_tokens, transform_sampling, device)
            #print(decode(y[0].tolist()))
            generated_macro.append(decode(y[0].tolist()))
            #print('---------------')

            if num_samples >= 100:
                if len(generated_macro) % int(0.1 * num_samples) == 0:
                    generated_percentage += 10
                    save_obj(generated_macro, os.path.join(sampling_path, f'{epoch_name}_{transform_sampling}_at_{temperature}_{generated_percentage}%'))
                    sampling_time = time.time() - start_sampling
                    start_sampling = start_sampling + sampling_time
                    print(f'Macro_GPT model {model_name}: {generated_percentage}% of the molecules sampled in {sampling_time:.2f} seconds')
        save_obj(generated_macro, os.path.join(sampling_path, f'{epoch_name}_{transform_sampling}_at_{temperature}'))


t_end = time.time()
print(f'Sampling of Macrocycle was done in {t_end - start_time:.05} seconds')

