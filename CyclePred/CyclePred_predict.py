import os
import math
import numpy as np
import pandas as pd
import json
import operator
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam
import wandb

from Macfrag_data import create_dataloader
from CyclePred_model import CyclePred as Model
from schedular import NoamLR
from utils import get_func,remove_nan_label

def evaluate(dataloader,model,device,metric_fn,metric_dtype,task):
    metric = 0
    for bg,labels in dataloader:
        bg,labels = bg.to(device),labels.type(metric_dtype)
        pred = model(bg).cpu().detach()
        if task == 'classification':
            pred = torch.sigmoid(pred)
        elif task == 'multiclass':
            pred = torch.softmax(pred,dim=1)
        num_task =  pred.size(1)
        if num_task >1:
            m = 0
            for i in range(num_task):
                try:
                    m += metric_fn(*remove_nan_label(pred[:,i],labels[:,i]))
                except:
                    print(f'only one class for task {i}')
            m = m/num_task
        else:
            m = metric_fn(pred,labels.reshape(pred.shape))
        metric += m.item()*len(labels)
    metric = metric/len(dataloader.dataset)
    
    return metric

def predict_macro(data_args,train_args,model_args, model_path_args):
    device = train_args['device'] if torch.cuda.is_available() else 'cpu'
    save_path = train_args['save_path']
    os.makedirs(save_path, exist_ok=True)

    results = []
    valloader = create_dataloader(data_args, data_args["file_name"], shuffle=False, train=False)  #
    print(f'macro set: {len(valloader.dataset)}')
    model = Model(model_args).to(device)
    state_dict = torch.load(model_path_args)
    model.load_state_dict(state_dict)
    model.eval()

    metric_fn = get_func(train_args['metric_fn'])
    if train_args['metric_fn'] in []:
        metric_dtype = torch.long
    else:
        metric_dtype = torch.float32
    val_metric = evaluate(valloader, model, device, metric_fn, metric_dtype, data_args['task'])
    print(f'macro {train_args["metric_fn"]}: {val_metric}')


if __name__=='__main__':

    import sys
    config_path = sys.argv[1]
    config = json.load(open(config_path,'r'))
    data_args = config['data'] 

    train_args = config['train']
    train_args['data_name'] = config_path.split('/')[-1].strip('.json')
    model_args = config['model']
    model_path_args = config['model_path']
    print(f'model_path_args is: {model_path_args}')

    print(config)
    # results = train(data_args,train_args,model_args,seed)
    predict_macro(data_args, train_args, model_args, model_path_args)
    # print(f'average performance: {np.mean(results)}+/-{np.std(results)}')