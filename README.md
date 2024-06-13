# CycleGPT
This repo  is using for “Exploring the macrocyclic chemical space for heuristic drug design with deep learning models”。

## Environment
```
conda env create -f CycleGPT.yaml  
conda activate CycleGPT 
```
## Usage of CycleGPT
### Data processing
```
cd data/xxx
Macro_processing.py --macro_path xxx.csv --augment 0
```
Macro_processing.py: split and pad the data, implement augmentation if necessary.




### Training
```
python lion_macro_train.py --input_name=xxx --device=cuda:2 --init_from=scratch --batch_size=128 --max_iters=30
```
input_name: train dataset name,  the same as processing data.

init_from: scratch--init a new model, resume--load a pretrain model, need assign "resume_checkpoint". 
### Sampling
Tanh-Tempered sampling as example. You can switch Sin-Tempered sampling or other sampling methods through sampling_methods.py. 

Pretrain model for generated macrocycles can acquire from https://huggingface.co/FengAlbert/CycleGPT
```
python lion_sample_tanh_temp_transform.py --resume_checkpoint=xxx --temperature=0.7 --device=cuda:2  
python macro_gen_load_gpt.py --input_name xxx  --pkl_path xxxx.pkl
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
python CyclePred_train.py ./config/CyclePred_train_Jak2.json
```
Json file can switch the hyperparameter and train data.
### Predicting
```
python CyclePred_predict_IC50.py ./config/CyclePred_predict_Jak2_IC50.json
```

