"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from data.Macro_processing import indices_token
# from torch.nn.parallel import DistributedDataParallel as DDP 
# from torch.distributed import init_process_group, destroy_process_group  

from lion_macro_model import GPTConfig, GPT

t_begin = time.time()

input_name = 'drug_name'

eval_interval = 1
log_interval = 1


eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'resume'  # 'scratch' or 'resume' or 'gpt2*'

resume_checkpoint = './model/lion_macro/macro_pretrain.pt'

# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())


gradient_accumulation_steps = 1  # used to simulate larger batch sizes

batch_size = 128    # if gradient_accumulation_steps > 1, this is the micro-batch size

block_size = 141 

# model
n_layer = 12  

n_head = 12 

n_embd = 768
dropout = 0.0
# for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 2e-4 # 9e-5 # 3e-4  # max learning rate
max_iters = 30  # 100 # 2000                 total number of training iterations
weight_decay = 0.01  # 1e-1
beta1 = 0.95
beta2 = 0.98
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0 
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate

warmup_iters = 2000   # how many steps to warm up

lr_decay_iters = 600000  # 2000  
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings

backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda:2'      # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

# compile = True  # use PyTorch 2.0 to compile the model to be faster

master_process = True
seed_offset = 0  

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
if master_process:
    out_dir = f'./model/{input_name}'
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)  
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype] 
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

#data_dir = os.path.join('data', dataset)
train_data = (f'./data/{input_name}/train_array.npy')
print(f'train_data is:{train_data}')

val_data = (f'./data/{input_name}/val_array.npy')
print(f'val_data is:{val_data}')



class MacroDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir  
        #self.seq_len = seq_len  
        self.data = np.load(data_dir)

    def __len__(self):
        #return len(np.load(self.data_dir)) 
        return len(self.data)

    def __getitem__(self, idx):  
        #raw_data = np.load(self.data_dir)  #raw_data = torch.Tensor(np.load(self.data_dir)).to(torch.int64).pin_memory()  #
        raw_data = self.data
        one_data = raw_data[idx]

        x = one_data[:-1]
        y = one_data[1:]
        if device_type == 'cuda':
            x, y = torch.Tensor(x).to(torch.int64).pin_memory().to(device, non_blocking=True), torch.Tensor(y).to(torch.int64).pin_memory().to(device, non_blocking=True)
            #x, y = torch.Tensor(x).to(device), torch.Tensor(y).to(device)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

def get_data(split):
    data = train_data if split == 'train' else val_data
    dataset = MacroDataset(data)
    #torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_start_method('forkserver', force=True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0 )  # ,pin_memory=True
    return dataloader


iter_num = 0
best_val_loss = 1e9

meta_vocab_size = len(indices_token)  

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  

@torch.no_grad()
def estimate_loss(data): 
    model.eval()
    losses = []
    val_loader = get_data(data)
    for x, y in val_loader:
        if device_type == 'cuda':
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)
            loss_batch = loss.item()
            losses.append(loss_batch)
        else:
            with ctx:
                logits, loss = model(x, y)
                loss_batch = loss.item()
                losses.append(loss_batch)
    #out = losses.mean()
    out = sum(losses) / len(losses)
            #losses.append(loss.item())
    # model.train()
    return out


if init_from == 'scratch':  
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:  
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':  
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = resume_checkpoint

    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)  # GPT
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)  

if block_size < model.config.block_size:
    model.crop_block_size(block_size)  
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)  #
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# learning rate decay scheduler (cosine with warmup) 
def get_lr(it): 
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:  # 2000
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


t0 = time.time()
local_iter_num = 0
raw_model = model  # unwrap DDP container if needed
running_mfu = -1.0

loss_save = []
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    t_batch_first = time.time()
    model.train()

    train_loader = get_data('train')
    for x, y in train_loader:
        if device_type == 'cuda':
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)
        else:
            with ctx: 
                logits, loss = model(x, y)

        scaler.scale(loss).backward() 
        print('loss is ', loss)


        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) 
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t_batch_end = time.time()
        t_each_batch = t_batch_end - t_batch_first

        lossf = loss.item()
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, t_each_batch)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: mean loss {lossf:.4f}, batch_time {t_each_batch * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
    t1 = time.time()
    d_epoch = t1 - t0
    t0 = t1

    print(f"iter {iter_num}: mean loss {lossf:.4f}, epoch_time {d_epoch * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")  

    if iter_num % eval_interval == 0 : 
        # train_loss = estimate_loss('train')
        val_loss = estimate_loss('val') 
        print(f"step {iter_num}:  val loss {val_loss:.4f}") 
        best_val_loss = loss
        checkpoint = {
            'model': raw_model.state_dict(),  
            'optimizer': optimizer.state_dict(), 
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': val_loss,
            'config': config,
        }
        print(f"saving checkpoint to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}_{lossf:.2f}_{val_loss:.2f}.pt'))

    iter_num += 1
    local_iter_num += 1
    loss_save.append(lossf)

    if iter_num > max_iters-1:
        with open(f'{out_dir}/loss_log.txt','a') as f:
            i = 0
            for loss in loss_save:
                f.write('epoch'+ str(i) +' ,' + str(loss) + '\n')
                i = i+1
        break

t_end = time.time()

print(f'Training of Macrocycle was done in {t_end - t_begin:.05} seconds')




