# CycleGPT
This repo  is using for “Exploring the macrocyclic chemical space for heuristic drug design with deep learning models”.

## Environment
```
conda env create -f CycleGPT.yaml  
conda activate CycleGPT 
```
After testing many times on many different machines, we recommended to install pytorch separately to accommodate compatibility issues between installation packages.

The following pytorch versions are used and tested at code time:
```
pip3 install numpy --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
```
## Usage of CycleGPT
### Data processing
```
cd data
Macro_processing.py --macro_path xxx.csv --augment 0
Macro_processing.py --macro_path macro_all.csv --augment 0
```
Macro_processing.py: split and pad the data, implement augmentation if necessary(augment: switch to number you want). 

xxx.csv can switch to your own data, such as: macro_all.csv. 


### Training
```
python lion_macro_train.py --input_name=xxx --device=cuda:2 --init_from=scratch --batch_size=128 --max_iters=30
python lion_macro_train.py --input_name=macro_all --device=cuda:2 --init_from=scratch --batch_size=128 --max_iters=30
```
input_name: train dataset name,  the same as processing data,such as: macro_all.

init_from: choose whether to train the network from scratch or use the pre-trained network for transfer learning training.

init_from:scratch--init a new model.  
init_from:resume--load a pretrain model. This need assign "resume_checkpoint"such as: --resume_checkpoint=xxx(pretrain model).
If resume, max_iters may need to adjust.

### Sampling
HyperTemp sampling(Tanh-Tempered) as example. You can switch SinusoTemp sampling(Sin-Tempered) or other sampling methods through sampling_methods.py. 

Pretrain model for generated macrocycles can acquire from https://huggingface.co/FengAlbert/CycleGPT
```
python lion_sample_tanh_temp_transform.py --resume_checkpoint=xxx --temperature=0.7 --device=cuda:2
python lion_sample_tanh_temp_transform.py --resume_checkpoint=./ckpt_22_cycle_gpt.pt --temperature=0.7 --device=cuda:2
```
A trained model must be assigned to sample. --resume_checkpoint=xxx/xxx/xxx.pt (model file path).

This will generated a pkl file which contain generated molecules. Employing following script to decode molecules.
```
python macro_gen_load_gpt.py --input_name xxx  --pkl_path xxxx.pkl
python macro_gen_load_gpt.py --input_name macro_all  --pkl_path xxxx.pkl
```

Then valuate generated molecules:
```
macro_pick.py: seperate the macrocycles, linear molecules and invalid molecules.
macro_set.py: drop duplicate molecules.
macro_novelty.py: pick the novel molecules.
```
## Usage of CyclePred
If you only use pre-trained CyclePred for prediction, you can skip the data processing and the training process. We provide pre-trained models in the ckpt folder.

We used dgl(1.1.1+cuda) version during code development, and now dgl maintains version 2.x. If the GPU is not available, the user can change devide to the cpu or update the dgl dependency package.
### Data processing
```
cd CyclePred
python Macfrag_data.py
```
Using MacFrag to constract the Heterogeneous graph and Seperate the dataset for training. 
### Training
```
python CyclePred_train.py ./config/CyclePred_train_Jak2.json
```
Json file can switch the hyperparameter and train data.
### Predicting
```
python CyclePred_predict_IC50.py ./config/CyclePred_predict_Jak2_IC50.json
```
Json file can switch the predicted model and predict molecules data.

Json file can switch the hyperparameter and train data.
### Predicting
```
python CyclePred_predict_IC50.py ./config/CyclePred_predict_Jak2_IC50.json
```
Json file can switch the predicted model and predict molecules data.
