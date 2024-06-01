from rdkit import Chem
import pandas as pd
import time
import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser("read file to get numbers of macrocycles and linear molecules. ")
    parser.add_argument('--gen_path','-d',type=str, required=True,
                        help='Path of the molecules csv to calculate the mocrocycle')
    parser.add_argument('--n_jobs',
                        type=int, default=1,
                        help='Number of processes to run metrics')
    parser.add_argument('--verbose', type=bool, default=True,
                        help='verbose')
    parser.add_argument('--device', type=str, default='cpu',
                        help='GPU device id (`cpu` or `cuda:n`)')
    parser.add_argument('--save_path', type=str, default='./after_macro/',
                        help='Store macro in this folder')
    return parser

def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],
                       squeeze=True).astype(str).tolist()

def after_save_data(aug_after, save_path, model, csv_name):
    # print(total)
    name = ['SMILES']
    AUG = pd.DataFrame(columns=name,data=aug_after)
    # print(AUG)
    # AUG.to_csv(f'{save_path}{model}_{aug_after}_macro.csv', encoding='utf-8')
    AUG.to_csv(f'{save_path}{model}_{csv_name}.csv', index=False, encoding='utf-8')

def cal_macro(data):
    invalid = []
    invalid_num = 0
    macro_rings = []
    macro_num = 0
    linear = []
    linear_num = 0
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid.append(smiles)
            invalid_num +=1
        else:
            ring = mol.GetRingInfo().AtomRings()
            ring_num_atom = [len(k) for k in ring]
            if len(ring_num_atom) != 0 and max(ring_num_atom) > 11: 
                # smi = Chem.MolToSmiles(mol)
                macro_rings.append(smiles)
                macro_num +=1
            else:
                linear.append(smiles)
                linear_num +=1
            
            
    return invalid, invalid_num, macro_rings, macro_num, linear, linear_num 

def main(config):
    data = read_smiles_csv(config.gen_path)
    invalid, invalid_num, macro_rings, macro_num, linear, linear_num = cal_macro(data)
    if config.verbose:
        print('-------------------------------------------------------')
        print('invalid_num'+','+str(invalid_num))
        print('macro_num' + ',' + str(macro_num))
        print('linear_num' + ',' + str(linear_num))
    model = config.gen_path.split('/')[-1].replace('_generated.csv', '')
    save_path = config.save_path
    os.makedirs(save_path, exist_ok=True)
    
    after_save_data(invalid, save_path, model, 'invalid')
    after_save_data(macro_rings, save_path, model, 'macro_rings')
    after_save_data(linear, save_path, model, 'linear')


if __name__ == '__main__':
    start = time.time()
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument " + unknown[0])
    main(config)
    end = time.time()
    print(f'\ndecode the generated Macrocycle was done in {end - start:.2f} seconds')











