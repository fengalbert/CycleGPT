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

from torch.utils.data import Dataset
from Macfrag_data import Mol2HeteroGraph
from rdkit import Chem
import dgl
from dgl.dataloading import GraphDataLoader
import pandas as pd
import time

def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],
                       squeeze=True).astype(str).tolist()

class predGraphSet(Dataset):
    def __init__(self, df,  log=print):
        self.data = df
        self.smiles = []
        self.mols = []
        self.graphs = []
        for i, row in df.iterrows():
            smi = row['SMILES']  
            # label = row[target].values.astype(float)
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    log('invalid', smi)  
                else:
                    g = Mol2HeteroGraph(mol)
                    if g.num_nodes('a') == 0:
                        log('no edge in graph', smi)  
                    else:
                        self.mols.append(mol)
                        self.smiles.append(smi)
                        self.graphs.append(g)
                        # self.labels.append(label)
            except Exception as e:  # list assignment index out of range 
                log(e, 'invalid', smi)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):

        return self.graphs[idx], self.smiles[idx]  #


def pred_dataloader(args, shuffle=False):  #  file_path, , train=True
    dataset = predGraphSet(pd.read_csv(args['file_path']))  #  self.graphs[idx], self.smiles[idx]

    batch_size = min(2000, len(dataset))  # 4200

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def predict(dataloader, model, device, task):  # ,metric_fn,metric_dtype
    # metric = 0
    predict_result = []  # smiles,
    all_smi = []
    for bg, smis in dataloader:  # 
        bg = bg.to(device)
        # bg,labels = bg.to(device),labels.type(metric_dtype)
        pred = model(bg).cpu().detach()  
        if task == 'classification':  
            pred = torch.sigmoid(pred)
        elif task == 'multiclass':  # multiclass
            pred = torch.softmax(pred, dim=1)
        num_task = pred.size(1)  #
        if num_task > 1:
            print(f'This is a multi task predict')
        else:
            print(f'This is one task predict')
        predict_result.extend(pred)
        all_smi.extend(smis)
        # predict_result.append([smi, pred])  #

    return predict_result, all_smi

def convert_IC50(preds):
    convert = []
    for i in preds:
        predict_result_IC50 = (10 ** abs(np.array(i))).tolist()
        convert.append(predict_result_IC50)

    return convert


def main(data_args, pred_args, model_args, model_path_args):

    device = pred_args['device'] if torch.cuda.is_available() else 'cpu'
    model = Model(model_args).to(device) 

    # state_dict = torch.load(os.path.join(save_path, f'./best_fold{fold}.pt'))  
    state_dict = torch.load(model_path_args)  
    model.load_state_dict(state_dict)
    model.eval()

    model_name = model_path_args.split('/')[-1].replace('.pt','')
    # pd.read_csv(data_args['file_path'])
    #all_smi = read_smiles_csv(data_args['file_path'])

    predictloader = pred_dataloader(data_args, shuffle=True)  

    predict_result, all_smi = predict(predictloader, model, device, data_args['task'])  # [[smi, pred],   ]

    #predict_result_IC50 = (10 ** abs(np.array(predict_result))).tolist()  # IC50
    convert = convert_IC50(predict_result)  # IC50

    pre_data = pd.DataFrame(convert, columns=['Predict_IC50'], index=all_smi)  
    data_path = data_args['file_path']
    data_name = data_path.split('/')[-1].replace('.csv', '')
    os.makedirs(f'./predict/{data_name}', exist_ok=True)
    pre_data.to_csv(f'./predict/{data_name}/{model_name}_predict_IC50.csv')


if __name__=='__main__':
    start = time.time()
    import sys
    config_path = sys.argv[1] 
    config = json.load(open(config_path, 'r'))
    data_args = config['data'] 
    pred_args = config['pred']  # device
    model_args = config['model']  

    model_path_args = config['model_path']

    print(config)

    main(data_args, pred_args, model_args, model_path_args)

    end = time.time()
    print(f'decode the generated Macrocycle was done in {end - start:.2f} seconds')
