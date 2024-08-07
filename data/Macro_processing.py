from random import shuffle
import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
import pickle
import time

import sys
import os

def get_parser():
    parser = argparse.ArgumentParser(
        "do macrocycle processing")
    parser.add_argument(
        '--macro_path', '-a', type=str, default='config.csv',
        help='Path to the dataset with macrocycle smiles. ' )
    parser.add_argument(
        '--path_to', type=str, default='./',
        help='Path to the dataset with aug macrocycle smiles. ' )
    parser.add_argument(
        '--verbose', type=bool, default=True,
        help='verbose')
    parser.add_argument(
        '--augment', type=int, default='10',
        help='Controls the number of augmentation')
    parser.add_argument(
        '--min_len', type=int, default='12',
        help='The min lenth of Macrocycle')
    parser.add_argument(
        '--max_len', type=int, default='140',
        help='The max lenth of Macrocycle')
    parser.add_argument(
        '--split', type=float, default='0.8',
        help='The ratio to split Macrocycle data')

    return parser

special_token = {'start_char': 'G',
                    'end_char': 'E',
                    'pad_char': 'A'}
indices_token = {"0": 'H', "1": '9', "2": 'D', "3": 'r', "4": 'T', "5": 'R', "6": 'V', "7": '4',
                 "8": 'c', "9": 'l', "10": 'b', "11": '.', "12": 'C', "13": 'Y', "14": 's', "15": 'B',
                 "16": 'k', "17": '+', "18": 'p', "19": '2', "20": '7', "21": '8', "22": 'O',
                 "23": '%', "24": 'o', "25": '6', "26": 'N', "27": 'A', "28": 't', "29": '$',
                 "30": '(', "31": 'u', "32": 'Z', "33": '#', "34": 'M', "35": 'P', "36": 'G',
                 "37": 'I', "38": '=', "39": '-', "40": 'X', "41": '@', "42": 'E', "43": ':',
                 "44": '\\', "45": ')', "46": 'i', "47": 'K', "48": '/', "49": '{', "50": 'h',
                 "51": 'L', "52": 'n', "53": 'U', "54": '[', "55": '0', "56": 'y', "57": 'e',
                 "58": '3', "59": 'g', "60": 'f', "61": '}', "62": '1', "63": 'd', "64": 'W',
                 "65": '5', "66": 'S', "67": 'F', "68": ']', "69": 'a', "70": 'm'}  

token_indices = {v: k for k, v in indices_token.items()}  

def encode(s): 
    return [token_indices[c] for c in s]

def decode(l): 
    ''.join([indices_token[i] for i in l])

vocab_size = len(indices_token)

def str2ind_array(path):
    data_arrays = []
    for i in read_smiles_csv(path):
        smi_ind = encode(i)  
        ids = np.array(smi_ind, dtype=np.int64)  
        data_arrays.append(ids) 
    return data_arrays

def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],
                       squeeze=True).astype(str).tolist()

def save_data(path_to_file, data):  # list
    # print(total)
    name = ['SMILES']
    AUG = pd.DataFrame(columns=name, data=data)
    # print(AUG)
    AUG.to_csv(f'{path_to_file}.csv', encoding='utf-8', index=False)

def pad_smiles(smiles,max_len):
    all_list = ['G']
    all_list.append(smiles)
    all_list.append('E')
    all_list.append('A'*(max_len-len(smiles)))
    smiles_pad = (''.join(all_list))
    return smiles_pad

def write_in_file(path_to_file, data):
    with open(path_to_file, 'w+') as f: 
        for item in data:
            f.write("%s\n" % item)

def save_obj(obj, name):
    """save obj with pickle"""
    name = name.replace('.pkl', '')  
    with open(name + '.pkl', 'wb') as f:  
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

def load_macro(config):
    macro_smi = []
    macro_mol = []
    macro_all = read_smiles_csv(config.macro_path)
    for macro in macro_all:
        if len(macro) <= config.max_len and len(macro) >= config.min_len:
            mol = Chem.MolFromSmiles(macro)
            if mol is not None:
                macro_smi.append(macro)
                macro_mol.append(mol)
    if config.verbose: print(f'The macrocycle set was check using rdkit retain:{len(macro_smi)}')
    return macro_smi, macro_mol

def macro_random(smiles):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)

def augmentation(config,smi):  
    aug_each = []
    for i in range(1000):
        new = macro_random(smi)
        check = Chem.MolFromSmiles(new)
        if check is not None and len(new) <= config.max_len:
            aug_each.append(new)
            aug_each = list(set(aug_each))
            if len(aug_each) == config.augment:
                break
    return aug_each

def augment_dataset(data_ori, config):
    all_alter_macro = []
    for i,x in enumerate(data_ori):
        alternative_smi = augmentation(config,x)
        all_alter_macro.extend(alternative_smi)
        if config.verbose and i%50000:
            print(f'Macrocycle augmentation is at step {i}')
    if config.verbose:
        print(f'macrocycle augmentation done; number of new SMILES: {len(all_alter_macro)}')
    return all_alter_macro

def macro_process(config,path_to, name):
    macro_smi, macro_mol = load_macro(config)
    all_idx = np.arange(len(macro_smi))  
    idx_split = int(config.split*len(all_idx))  
    np.random.shuffle(all_idx)  
    if idx_split == 0:  
        idx_ori_tr = [0]
        idx_ori_val = [0]
    else:
        idx_ori_tr = all_idx[:idx_split]  
        idx_ori_val = all_idx[idx_split:]
        if config.verbose:
            print(f'Size of the training set after split: {len(idx_ori_tr)}')
            print(f'Size of the validation set after split: {len(idx_ori_val)}')
    d = dict(enumerate(macro_smi)) 
    smi_ori_tr = [d.get(item) for item in idx_ori_tr] 
    smi_ori_val = [d.get(item) for item in idx_ori_val]  

    save_data(f'{path_to}macro_ori_tr', smi_ori_tr)
    save_data(f'{path_to}macro_ori_val', smi_ori_val)
    if config.augment > 0 and config.verbose:
        print(f'Macrocycle of augmentation is set to {config.augment}-fold start')
        smi_aug_tr = augment_dataset(smi_ori_tr, config)  
        smi_aug_val = augment_dataset(smi_ori_val, config)
        full_aug_tr = list(set(smi_ori_tr + smi_aug_tr))  
        shuffle(full_aug_tr)
        full_aug_val = list(set(smi_ori_val + smi_aug_val)) 
        shuffle(full_aug_val)
        all_macro_aug = full_aug_tr + full_aug_val
        save_data(f'{path_to}full_aug_tr', full_aug_tr)
        save_data(f'{path_to}full_aug_val', full_aug_val)
        save_data(f'{path_to}all_macro_aug', all_macro_aug)

        full_aug_tr_pad = [pad_smiles(each, config.max_len) for each in full_aug_tr]
        full_aug_val_pad = [pad_smiles(each, config.max_len) for each in full_aug_val]
        full_ori_tr_pad = [pad_smiles(each, config.max_len) for each in smi_ori_tr]
        full_ori_val_pad = [pad_smiles(each, config.max_len) for each in smi_ori_val]

        save_data(f'{path_to}full_aug_tr_pad', full_aug_tr_pad)
        save_data(f'{path_to}full_aug_val_pad', full_aug_val_pad)
        save_data(f'{path_to}full_ori_tr_pad', full_ori_tr_pad)
        save_data(f'{path_to}full_ori_val_pad', full_ori_val_pad)

        train_array = str2ind_array(f'{path_to}full_aug_tr_pad.csv') 
        val_array = str2ind_array(f'{path_to}full_aug_val_pad.csv')
        np.save(os.path.join(f'{path_to}train_array'), train_array )
        np.save(os.path.join(f'{path_to}val_array'), val_array )

        if config.verbose:
            print(f'The number of training macrocycles after agumentation: {len(full_aug_tr)}')
            print(f'The number of validation macrocycles after agumentation: {len(full_aug_val)}')
    else:  
        full_ori_tr_pad = [pad_smiles(each, config.max_len) for each in smi_ori_tr]
        full_ori_val_pad = [pad_smiles(each, config.max_len) for each in smi_ori_val]
        save_data(f'{path_to}full_ori_tr_pad', full_ori_tr_pad)
        save_data(f'{path_to}full_ori_val_pad', full_ori_val_pad)
        
        train_array = str2ind_array(f'{path_to}full_ori_tr_pad.csv') 
        val_array = str2ind_array(f'{path_to}full_ori_val_pad.csv')
        np.save(os.path.join(f'{path_to}train_array'), train_array )
        np.save(os.path.join(f'{path_to}val_array'), val_array )



def main(config):
    print('\nThe macrocycle processing is beginning')
    name = config.macro_path.split('/')[-1].replace('.csv', '')
    print(f'\nThe processing dataset is : {name}')

    path_ori = config.path_to
    path_to = f'{path_ori}/{name}/'
    os.makedirs(path_to, exist_ok=True)
    macro_process(config, path_to, name)

if __name__ == '__main__':
    start = time.time()
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument " + unknown[0])
    main(config)
    end = time.time()
    print(f'Processing of Macrocycle  in {end - start:.04} seconds')
