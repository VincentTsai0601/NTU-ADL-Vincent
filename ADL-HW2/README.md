m11115013 資工所  Min Sheng Tsai
# ADL Homework 2 - Instruction Tuning (Classical Chinese) 2025 

## Preparation
1. Download and unzip the pretrained Qwen and Taiwan llama LoRA weights:
```bash
bash ./download.sh
```

## Setup the environments

Python 3.10.22

```bash
pip install -r requirements.txt
```

## Testing the Best Model

1. Modify hyperparameters and file path in the `.sh` file as needed.


2. Execute the testing script:
```bash
bash run.sh /path/to/model-folder /path/to/adapter_checkpoint /path/to/input /path/to/output
```
## Qwen Example:
```bash
bash run.sh Qwen/Qwen3-4B ./adapter_checkpoint ./data/private_test.json ./prediction.json
```


## Testing the Perplexity of Model

```bash
bash eval.sh /path/to/model-folder /path/to/adapter_checkpoint /path/to/input
```

Qwen3 Example:
```bash
bash eval.sh  Qwen/Qwen3-4B  output_Qwen/Qwen3-4B/4bits/checkpoint-2000/adapter_model  ./data/public_test.json
```


## Training the Model
1. Modify hyperparameters, model type and file path in the `train.py` file (for Qwen).

2. Execute the training script:
```bash
bash  train.sh
```
  
## Visualization

Refer to the `plot` folder for visualization resources:

- `plot_ppl.py`: Python script that plots the relationship between model training steps and perplexity (PPL) values.

To generate plots:
```bash
python3 plot_ppl.py
```

## Additional Notes
- Ensure all paths in the scripts are correctly set before running.
