m11115013 資工所 Min Sheng Tsai
# ADL Homework 1 - Chinese Question Answering

## Environment
- GPU: NVIDIA GeForce RTX 3090 

## Preparation
Download the pretrained weights:
```bash
bash ./download.sh
```
## Setup the virtual environment
```bash
conda env create -f environment.yml
```

## Inference the Best model
Run the following command:
```bash
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

## Training the model
1. Execute the training script:
```bash
bash ./train.sh
```
2. Modify hyperparameters and file path in the `.sh` file as needed.

## Testing the model 

Executing the testing script: 
```bash 
bash ./test.sh 
```



