#!/usr/bin/env python
# coding: utf-8
"""Generate JSON answers with a base causal LM and a trained LoRA adapter.

Place this file beside the utils.py used during training. Input must be a
JSON array of objects with an instruction string and an optional id.

Example:
    python generate.py --peft_path ./output/checkpoint-1000/adapter_model \
        --test_data_path ./data/test.json --output ./res.json

The base checkpoint is read from adapter_config.json unless explicitly supplied.
Use the same base checkpoint, tokenizer, and get_prompt function as training.
"""
# Set these before importing Transformers or PEFT (this script uses PyTorch).
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse
import json

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftConfig, PeftModel

   




def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", default=None,
                        help="Exact training base checkpoint; otherwise read from adapter config.")
    parser.add_argument("--peft_path", required=True,
                        help="Adapter folder containing adapter_config.json and adapter weights.")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Saved training tokenizer folder; defaults to the base checkpoint.")
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--output", default="./res.json")
    parser.add_argument("--source_max_len", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--bits", type=int, choices=[4, 8, 16], default=4,
                        help="4 uses your training get_bnb_config(); 16 disables quantization.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    for name in ("source_max_len", "max_new_tokens", "num_beams"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.do_sample and (args.temperature <= 0 or not 0 < args.top_p <= 1):
        parser.error("Sampling requires temperature > 0 and 0 < top_p <= 1")
    return args


def read_data(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Test data must be a JSON array of instruction objects.")
    for index, example in enumerate(data):
        if not isinstance(example, dict) or not isinstance(example.get("instruction"), str):
            raise ValueError(f"Example {index} must contain an 'instruction' string.")
    return data


def generate_answers(model, tokenizer, data, args, get_prompt):
    import torch
    from tqdm import tqdm

    device = model.get_input_embeddings().weight.device
    results = []
    model.eval()
    model.config.use_cache = True
    for index, example in enumerate(tqdm(data, desc="Generating")):
        prompt = get_prompt(example["instruction"])
        if not isinstance(prompt, str):
            raise TypeError("utils.get_prompt() must return a string.")
        # Match the training collator's BOS + prompt convention where BOS exists.
        # Qwen tokenizers may have no BOS token; do not construct a tensor of None.
        if tokenizer.bos_token is not None:
            prompt = tokenizer.bos_token + prompt
        inputs = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
            truncation=True, max_length=args.source_max_len,
        ).to(device)
        input_length = inputs["input_ids"].shape[1]
        if input_length == 0:
            raise ValueError(f"Example {index} produced an empty token sequence.")
        generation_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        # Keep checkpoint generation defaults, including model-specific EOS IDs.
        if args.do_sample:
            generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)
        answer = tokenizer.decode(
            outputs[0, input_length:], skip_special_tokens=True,
        ).strip()
        results.append({"id": example.get("id", index), "output": answer})
    return results


def main():
    args = parse_args()
    data = read_data(args.test_data_path)

    try:
        from utils import get_prompt
    except ImportError as error:
        raise ImportError(
            "Place generate.py beside the training utils.py containing get_prompt."
        ) from os.error

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU. Check your PyTorch CUDA installation.")
    set_seed(args.seed)
    adapter_config = PeftConfig.from_pretrained(args.peft_path)
    base_path = args.base_model_path or adapter_config.base_model_name_or_path
    if not base_path:
        raise ValueError("Adapter config has no base checkpoint; supply --base_model_path.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = dict(
        torch_dtype=dtype, device_map={"": 0},
        trust_remote_code=args.trust_remote_code,
    )
    if args.bits == 4:
        try:
            from utils import get_bnb_config
        except ImportError as error:
            raise ImportError("4-bit loading requires get_bnb_config in training utils.py.") from error
        model_kwargs["quantization_config"] = get_bnb_config()
    elif args.bits == 8:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading base model: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or base_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="right",
    )
    model = AutoModelForCausalLM.from_pretrained(base_path, **model_kwargs)
    # A tokenizer saved during training may contain newly added special tokens.
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither PAD nor EOS; pass the saved training tokenizer.")
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter: {args.peft_path}")
    model = PeftModel.from_pretrained(model, args.peft_path)
    results = generate_answers(model, tokenizer, data, args, get_prompt)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} answers to {output}")


if __name__ == "__main__":
    main()