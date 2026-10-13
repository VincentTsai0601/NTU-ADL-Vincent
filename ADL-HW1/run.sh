python3 test_mc.py \
  --model_path model/chinese-lert-base_mc/ \
  --context_file ${1} \
  --test_file ${2} \
  --output_file ./test/test_mc.json \
  --max_seq_length 128 \
  --per_device_test_batch_size 32


python3 test_qa.py \
  --model_path model/chinese-lert-base_qa \
  --context_file ${1} \
  --test_file ./test/test_mc.json \
  --output_file ${3} \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_eval_batch_size 32


#test/predictions.csv 