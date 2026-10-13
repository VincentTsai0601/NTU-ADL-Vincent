#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select a relevant paragraph with a trained multiple-choice checkpoint.

Input: a JSON array of {id, question, paragraphs: [four paragraph IDs]}.
Context: a JSON list indexed by integer IDs, or a dictionary with string IDs.
Output: the original examples with a predicted integer `relevant` field.
This script does not extract answer text or compute test accuracy.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context_file", default="data/context.json")
    parser.add_argument("--model_path", required=True,
                        help="Trained multiple-choice model and tokenizer directory.")
    parser.add_argument("--test_file", default="data/test.json")
    parser.add_argument("--output_file", default="test/test_mc.json")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--per_device_test_batch_size", type=int, default=32)

    args = parser.parse_args()
    
    if args.max_seq_length <= 0 or args.per_device_test_batch_size <= 0:
        parser.error("Sequence length and batch size must be positive.")
    if Path(args.test_file).resolve() == Path(args.output_file).resolve():
        parser.error("Use a separate output file to preserve the input.")
    return args


def get_paragraph(context, paragraph_id):
    if isinstance(context, list):
        index = int(paragraph_id)
        if index < 0 or index >= len(context):
            raise ValueError(f"Paragraph ID out of range: {paragraph_id}")
        text = context[index]
    elif isinstance(context, dict):
        text = context[str(paragraph_id)]
    else:
        raise ValueError("context.json must contain a list or dictionary.")
    if not isinstance(text, str):
        raise ValueError(f"Paragraph {paragraph_id} must contain text.")
    return text


@dataclass
class DataCollatorForMultipleChoice:
    """Pad candidate pairs without requiring dummy labels."""
    tokenizer: object
    pad_to_multiple_of: int | None = None

    def __call__(self, features):
        indices = torch.tensor([f["row_index"] for f in features], dtype=torch.long)
        flattened = []
        for feature in features:
            for choice in range(4):
                flattened.append({
                    key: value[choice]
                    for key, value in feature.items()
                    if key != "row_index"
                })
        padded = self.tokenizer.pad(
            flattened, padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt",
        )
        batch = {key: value.reshape(len(features), 4, -1)
                 for key, value in padded.items()}
        batch["row_index"] = indices
        return batch


def main():
    args = parse_args()
    accelerator = Accelerator()
    with open(args.test_file, encoding="utf-8") as handle:
        examples = json.load(handle)
    with open(args.context_file, encoding="utf-8") as handle:
        context = json.load(handle)
    if not isinstance(examples, list):
        raise ValueError("test.json must contain a JSON array of examples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_path)
    # A tokenizer with extra tokens should be repaired at training time.
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer vocabulary exceeds the saved model embeddings.")

    rows = []
    for index, example in enumerate(examples):
        candidates = example["paragraphs"]
        if len(candidates) != 4:
            raise ValueError(f"Example {index} must have exactly four candidates.")
        encoded = tokenizer(
            [example["question"]] * 4,
            [get_paragraph(context, candidate) for candidate in candidates],
            max_length=args.max_seq_length, truncation=True, padding=False,
        )
        rows.append({**dict(encoded), "row_index": index})

    if rows:
        multiple = (16 if accelerator.mixed_precision == "fp8" else
                    8 if accelerator.mixed_precision != "no" else None)
        loader = DataLoader(
            Dataset.from_list(rows), shuffle=False,
            batch_size=args.per_device_test_batch_size,
            collate_fn=DataCollatorForMultipleChoice(tokenizer, multiple),
        )
        model, loader = accelerator.prepare(model, loader)
        model.eval()
        predictions_by_index = {}
        for batch in tqdm(loader, desc="Selecting paragraphs",
                          disable=not accelerator.is_local_main_process):
            indices = batch.pop("row_index")
            with torch.inference_mode():
                choices = model(**batch).logits.argmax(dim=-1)
            indices, choices = accelerator.gather_for_metrics((indices, choices))
            for index, choice in zip(indices.cpu().tolist(), choices.cpu().tolist()):
                predictions_by_index[index] = choice
        if len(predictions_by_index) != len(examples):
            raise RuntimeError("The number of predictions does not match the input.")
        for index, example in enumerate(examples):
            example["relevant"] = int(example["paragraphs"][predictions_by_index[index]])

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output = Path(args.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(examples, handle, ensure_ascii=False, indent=2)
        accelerator.print(f"Saved {len(examples)} predictions to {output}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
