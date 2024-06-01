import rdkit
from rdkit.Chem import AllChem as Chem
import umap
import pandas as pd
import numpy as np
import os,sys
import random  
import time
from collections import OrderedDict
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sklearn
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pylab as plt

def get_parser():
    parser = argparse.ArgumentParser(
        "do umap projection of macrocycle data" )
    parser.add_argument(
        '--all_path', '-a', type=str, default='config.csv',
        help='Path to the all csvs with macrocycle smiles. ' )
    parser.add_argument(
        '--verbose', type=bool, default=True,
        help='verbose')
    parser.add_argument(
        '--seed', type=int, default='20',
        help='Controls the randomness of selected macrocycle')
    parser.add_argument(
        '--n', type=int, default='2000',  # 4000
        help='The select number of Macrocycle to do UMAP')
    parser.add_argument(
        '--path_to', type=str, default='./UMAP/',
        help='Store img and csv in this folder')
    return parser

def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES'],
                       squeeze=True).astype(str).tolist()

def save_data_csv(smis,save_path,data_name):
    name = ['SMILES']
    AUG = pd.DataFrame(columns=name, data=smis)
    # print(AUG)
    AUG.to_csv(f'{save_path}{data_name}.csv', index=False, encoding='utf-8')

def random_n_data(path_data, config):
    random.seed(config.seed)
    n = config.n
    macro_data = read_smiles_csv(path_data)
    #with open(path_data, 'r') as f:
    #     macro_data = f.read().splitlines()
    if n < len(macro_data):
        return random.sample(macro_data, n)
    else:
        warnings.warn('There are less macrocycle than n')
        print(f'n = {n}')
        print(f'n macrocycles available = {len(macro_data)}')
        return macro_data

def cal_macro_fp(macro_smis):
    macro_fp = []
    mols = [Chem.MolFromSmiles(x) for x in macro_smis]
    idx_to_remove = []
    for idx, mol in enumerate(mols):
        try:
            fprint = Chem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False)
        except:
            idx_to_remove.append(idx)
        else:
            macro_fp.append(fprint)
    smi_keep = [smi for i, smi in enumerate(macro_smis) if i not in idx_to_remove]
    assert len(macro_fp)==len(smi_keep)
    return macro_fp, smi_keep

def macro_embedding(all_data, n_components=2):
    "Calculate the UMAP embedding of Macrocycle"
    macro_scaled = StandardScaler().fit_transform(all_data)
    embedding = umap.UMAP(n_neighbors=10, min_dist=0.8,  metric='correlation',  
                          n_components=n_components, random_state=16).fit_transform(macro_scaled) 
    return embedding

Color_Select = {'Ori_linear': '#CD5B45',  
                'Ori_linear_gen': '#F5DCBC', 
                'macro_train': '#66CC98',  
                'macro_train_gen': '',
                'gen_fp': '#F5DCBC',  
                'gen_fp': '#F5DCBC'}  

def embedding_plot(embedding, Ori_linear_fp, Ori_linear_gen_fp, macro_train_fp, macro_train_gen_fp, 
                   edgecolors='#444444', m_data="o", m_gen="o",
                   s_data=70, s_gen=55, alpha_gen=1.00, alpha_data=0.85,
                   linewidth_gen='1.40',legend=False):
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.xlim([np.min(embedding[:,0])-0.5, np.max(embedding[:,0])+1.5])
    plt.ylim([np.min(embedding[:,1])-0.5, np.max(embedding[:,1])+0.5])
    labelsize = 16
    plt.xlabel('UMAP 1', fontsize=labelsize)
    plt.ylabel('UMAP 2', fontsize=labelsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.scatter(embedding[:Ori_linear_fp, 0], embedding[:Ori_linear_fp, 1],   
                lw=0, c=Color_Select['Ori_linear'], label='macro dataset', alpha=alpha_data, s=s_data,
                marker=m_data)  
    plt.scatter(embedding[Ori_linear_fp:Ori_linear_gen_fp, 0], embedding[Ori_linear_fp:Ori_linear_gen_fp, 1],
                lw=0, c=Color_Select['Ori_linear_gen'], label='macro generates', alpha=alpha_gen, s=s_gen,
                marker=m_gen, edgecolors=edgecolors, linewidth=linewidth_gen)  

    plt.scatter(embedding[Ori_linear_gen_fp:macro_train_fp, 0], embedding[Ori_linear_gen_fp:macro_train_fp, 1],
                lw=0, c=Color_Select['macro_train'], label='fine_tune generates', alpha=alpha_data, s=s_data,
                marker=m_gen, edgecolors=edgecolors, linewidth=linewidth_gen)

    plt.scatter(embedding[macro_train_fp:macro_train_gen_fp, 0], embedding[macro_train_fp:macro_train_gen_fp, 1],
                lw=0, c='#5A5A5A', label=f'fine_tune',  alpha=1.0, s=150,
                marker=m_data, edgecolors='k', linewidth='2')


    if legend:
        leg = plt.legend(prop={'size': labelsize}, loc='upper left', markerscale=1.00)  # loc='upper right
        leg.get_frame().set_alpha(0.9)
    plt.setp(ax, xticks=[], yticks=[])
    return fig

def main(config):
    all_dataset = OrderedDict()
    data_sets = pd.read_csv(config.all_path)
    for path, name in zip(data_sets['path'], data_sets['name']):
        if name == 'Ori_linear':
            Ori_linear = random_n_data(path, config)
            Ori_linear = read_smiles_csv(path)

            Ori_linear_fp, Ori_linear_smi = cal_macro_fp(Ori_linear)
            save_data_csv(Ori_linear_smi, config.path_to,name)
        if name == 'Ori_linear_gen':
            Ori_linear_gen = random_n_data(path, config)
            Ori_linear_gen = read_smiles_csv(path)

            Ori_linear_gen_fp, Ori_linear_gen_smi = cal_macro_fp(Ori_linear_gen)
            save_data_csv(Ori_linear_gen_smi, config.path_to,name)
        if name == 'macro_train':
            macro_train = random_n_data(path, config)
            macro_train_fp, macro_train_smi = cal_macro_fp(macro_train)
            save_data_csv(macro_train_smi, config.path_to,name)

        if name == 'macro_train_gen':
            macro_train_gen = random_n_data(path, config)
            macro_train_gen_fp, macro_train_gen_smi = cal_macro_fp(macro_train_gen)
            save_data_csv(macro_train_gen_smi, config.path_to,name)

    path_projection = f'{config.path_to}macrocycle_umap_test.npy'
    if not os.path.isfile(path_projection):
        Ori_linear_fp = np.array(Ori_linear_fp)
        Ori_linear_gen_fp = np.array(Ori_linear_gen_fp)
        macro_train_fp = np.array(macro_train_fp)
        macro_train_gen_fp = np.array(macro_train_gen_fp)
        '''
        
        target_fp = np.array(target_fp)
        '''
        all_data = np.concatenate([Ori_linear_fp, Ori_linear_gen_fp,macro_train_fp, macro_train_gen_fp], axis=0)
                                   
                                     
        print(f'all_data shape: {all_data.shape}')

        embedding = macro_embedding(all_data)  
        assert embedding.shape[0]==all_data.shape[0]
        assert embedding.shape[1]!=all_data.shape[1]
        np.save(path_projection, embedding)
    else:
        embedding = np.load(path_projection)

    len_Ori_linear = len(Ori_linear_fp)  
    len_Ori_linear_gen = len_Ori_linear + len(Ori_linear_gen_fp) 
    len_macro_train = len_Ori_linear_gen + len(macro_train_fp)  
    len_macro_train_gen = len_macro_train + len(macro_train_gen_fp)
   
    common_emb = embedding[:len_macro_train,:]  
    ft_emb = embedding[len_macro_train:,:]  
    print(f'Embedding shapes, common: {common_emb.shape}, fine-tuning: {ft_emb.shape}')
    fig_S = embedding_plot(np.concatenate([common_emb, ft_emb]),
                              len_Ori_linear, len_Ori_linear_gen, len_macro_train, len_macro_train_gen,  legend=True)
    plt.savefig(f'{config.path_to}umap_test-10.png', dpi=300)


if __name__ == '__main__':
    start = time.time()
    parser = get_parser()
    config, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise ValueError("Unknown argument " + unknown[0])
    main(config)
    end = time.time()
    print(f'UMAP of Macrocycle  in {end - start:.04} seconds')

