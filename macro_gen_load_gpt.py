from rdkit import Chem
import os, sys
import time
import argparse
import configparser
import ast
from rdkit import Chem
from rdkit import rdBase
import pickle
import pandas as pd
rdBase.DisableLog('rdApp.*')


def get_parser():
    parser = argparse.ArgumentParser(
        "decode the macrocycle sampling")
    parser.add_argument(
        '--input_name', '-a', type=str, default='./data/macro_all.csv',
        help='Path to the dataset with macrocycle smiles. ' )
    parser.add_argument(
        '--verbose', type=bool, default=True,
        help='verbose')
    parser.add_argument(
        '--pkl_path', type=str, default='./out_sample/xxx.pkl',
        help='the pkl sampling from model_path')

    return parser


PROCESSING_FIXED = {'begin_char': 'G',
                    'end_char': 'E',
                    'pad_char': 'A'}


pad_char = PROCESSING_FIXED['pad_char']
start_char = PROCESSING_FIXED['begin_char']
end_char = PROCESSING_FIXED['end_char']


def load_obj(name):
    """load a pickle object"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'rb') as f:  
        return pickle.load(f)  


def save_data(aug_after, csv_name):
    # print(total)
    name = ['SMILES']
    AUG = pd.DataFrame(columns=name, data=aug_after)
    print(AUG)
    AUG.to_csv(f'{csv_name}.csv',encoding='utf-8',index=False)



def main(config):
    input_name = config.input_name
    sampling_path = config.pkl_path  # 
    data = load_obj(sampling_path)  

    sampling_name = sampling_path.split('/')[-1].replace('.pkl', '')
 
    input_name = input_name.split('/')[-1].replace('.csv', '')  

    os.makedirs(f'./before_macro/{input_name}', exist_ok=True)
    before_macro_path = f'./before_macro/{input_name}/{sampling_name}'


    all_macro_smi = []
    for macro_smi in data:  # all_data
        if len(macro_smi) != 0 and isinstance(macro_smi, str):
            macro_smi = macro_smi.replace(pad_char, '')
            macro_smi = macro_smi.replace(end_char, '')
            macro_smi = macro_smi.replace(start_char, '')
            all_macro_smi.append(macro_smi)
    save_data(all_macro_smi, before_macro_path)



if __name__ == '__main__':
    start = time.time()
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument " + unknown[0])
    main(config)
    end = time.time()
    print(f'decode the generated Macrocycle was done in {end - start:.2f} seconds')
