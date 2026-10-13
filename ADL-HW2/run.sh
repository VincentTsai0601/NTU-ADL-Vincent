python3 inference.py \
  --base_model_path ${1} \
  --peft_path ${2} \
  --test_data_path ${3} \
  --output ${4}



# ${1}= path to the model checkpoint folder
# ${2}= path to the adapter_checkpoint downloaded under your folder
# ${3}= path to the input file (.json)
# ${4}= path to the output file (.json)

