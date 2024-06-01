import pandas as pd
from rdkit import Chem
import argparse
import time
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path',
                        type=str,  default='./after_macro/xxx.csv',
                        help='Path to generated molecules csv')  #
    parser.add_argument('--train_path',
                        type=str,  default='./data/macro_all.csv',
                        help='Path to train molecules csv')  #
    parser.add_argument('--novel_path', type=str, default='./after_novelty/',
                        help='Store novelty smiles in this folder')
    return parser

def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],
                       squeeze=True).astype(str).tolist()

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def save_data(aug_after,data_name, csv_name):
    # print(total)
    name = ['SMILES']
    AUG = pd.DataFrame(columns=name,data=aug_after)
    print(AUG)
    AUG.to_csv(f'{data_name}_{csv_name}.csv',encoding='utf-8', index=False)


def novelty(gen, train):   
    gen_smiles_set = []
    train_smiles_set = []
    for smiles in gen:
        outcome = canonic_smiles(smiles)
        if outcome is not None:
            gen_smiles_set.append(outcome)
    gen_smiles_set = set(gen_smiles_set)
    for smiles in train:
        outcome = canonic_smiles(smiles)
        if outcome is not None:
            train_smiles_set.append(outcome)
    train_smiles_set = set(train_smiles_set)
    print("gen_smiles," + str(len(gen_smiles_set)))
    print("train_smiles," + str(len(train_smiles_set)))
    print("novelty," + str(len(gen_smiles_set - train_smiles_set) / len(gen_smiles_set)))
    novel = gen_smiles_set - train_smiles_set
    return novel


def main(config):
    gen = read_smiles_csv(config.gen_path)
    data_name = config.gen_path.split('/')[-1].replace('.csv', '')
    train = read_smiles_csv(config.train_path)
    novel = novelty(gen, train)
    novel_path = config.novel_path
    os.makedirs(novel_path, exist_ok=True)
    
    save_data(novel, f'{novel_path}{data_name}', 'novel')



if __name__ == '__main__':
    start = time.time()
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument " + unknown[0])
    main(config)
    end = time.time()
    print(f'\n The novelty of generated Macrocycle was computed in {end - start:.2f} seconds')


