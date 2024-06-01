# CycleGPT
This repo is using for 

## Environment
```
conda env create -f CycleGPT.yaml  
conda activate CycleGPT 
```
## Usage of CycleGPT
### Data processing
```
cd data/xxx
Macro_processing.py: split and pad the data, implement augmentation if necessary.  
prepare_smiles.py: convert chose data to .npy file 
```
### Training
```
python lion_macro_train.py --input_name=drug_name --device=cuda:2 --init_from=resume 
```
input_name: train dataset name

init_from: scratch--init a new model, resume--load a pretrain model. 
### Sampling
Tanh tempered sampling as example. You can switch sin tempered sampling or other sampling methods through sampling_methods.py. 
```
python lion_sample_tanh_temp_transform.py --temperature=0.7 --device=cuda:2  
python macro_gen_load_gpt.py --macro_path xxx  --pkl_path xxxx.pkl
```
Then valuate generated molecules:
```
macro_pick.py: seperate the macrocycles, linear molecules and invalid molecules.
macro_set.py: drop duplicate molecules.
macro_novelty.py: pick the novel molecules.
```
## Usage of CyclePred
### Data processing
```
cd CyclePred
python Macfrag_data.py
```
Using MacFrag to constract the Heterogeneous graph and Seperate the dataset. 
### Training
```
python CyclePred_train.py config/xxx.json
```
Json file can switch the hyperparameter and train data.
### Predicting
```
python CyclePred_predict_IC50.py config/xxx.json
```

