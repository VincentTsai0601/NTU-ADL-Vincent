python3 ppl.py \
    --base_model_path $1 \
    --peft_path $2 \
    --test_data_path $3 \

# bash eval.sh  Qwen/Qwen3-4B  output_Qwen/Qwen3-4B/4bits/checkpoint-2000/adapter_model  ./data/public_test.json



# for i in {500..4000..500}
# do
#     echo "Evaluating checkpoint $i"
#     python3 ppl.py \
#         --base_model_path Qwen/Qwen3-4B \
#         --peft_path output_Qwen/Qwen3-4B/4bits/checkpoint-$i/adapter_model \
#         --test_data_path ./data/public_test.json \
  

# done


# for i in {500..4000..500}
# do
#     echo "Evaluating checkpoint $i"
#     python3 ppl.py \
#         --base_model_path yentinglin/Llama-3.1-Taiwan-8B \
#         --peft_path output_Qwen/Llama-3.1-Taiwan-8B/4bits/checkpoint-$i/adapter_model \
#         --test_data_path ./data/public_test.json \
  
# done
