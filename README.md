# Disentangling-before-Composing

This repository provides dataset splits and code for Paper:

Disentangling before Composing: Learning Invariant Disentangled Features for Compositional Zero-Shot Learning

## Usage 

1. Clone the repo.

2. We recommend using Anaconda for environment setup. To create the environment and activate it, please run:
```
    conda env create --file environment.yml
    conda activate czsl
```

3. The dataset and splits can be downloaded from: [CZSL-dataset](https://drive.google.com/drive/folders/1ZSw4uL8bjxKxBhrEFVeG3rgewDyDVIWj).


4. To run DBC for UT-Zappos dataset:
```
    Training:
    python train.py --config configs/zappos.yml

    Testing:
    python test.py --logpath LOG_DIR
```         
5. Add complex experiments, first create new folders and change:
```
    val_pairs.txt
    train_pairs.txt
    test_pairs.txt
```
Then change 2 paths (create deep folders):    
```      
    process.py --split_name --DATASET_ROOT      
    configs/your_experiment.yml --name --splitname
```      


**Note:** Most of the code is an improvement based on https://github.com/ExplainableML/czsl.
