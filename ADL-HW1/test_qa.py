#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract answers from paragraphs selected by test_mc.py.

Input JSON: [{"id": "...", "question": "...", "relevant": 123, ...}].
Output CSV contains id,answer. A .json output preserves fields and adds prediction_text.
No answer labels or utils_qa.py are required. Supports overlapping context windows.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", "--model_name_or_path", dest="model_path", required=True, help="Trained QA model/tokenizer directory.")
    parser.add_argument("--context_file", default="data/context.json")
    parser.add_argument("--test_file", default="test/test_mc.json")
    parser.add_argument("--output_file", default="test/predictions.csv")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--per_device_test_batch_size", "--per_device_eval_batch_size",
                        dest="per_device_test_batch_size", type=int, default=16)
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--max_answer_length", type=int, default=30,
                        help="Maximum answer length in tokens, not characters.")
    parser.add_argument("--version_2_with_negative", action="store_true",
                        help="Allow empty answers; requires a model trained for no-answer cases.")
    parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)

    args = parser.parse_args()

    for key in ("max_seq_length", "per_device_test_batch_size", "n_best_size", "max_answer_length"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive.")
    if args.doc_stride < 0:
        parser.error("--doc_stride must be nonnegative.")
    if Path(args.test_file).resolve() == Path(args.output_file).resolve():
        parser.error("Use a separate output file to preserve the input.")
    if Path(args.output_file).suffix.lower() not in (".csv", ".json"):
        parser.error("--output_file must end in .csv or .json.")
    return args


def get_paragraph(contexts, paragraph_id):
    if isinstance(contexts, list):
        index = int(paragraph_id)
        if not 0 <= index < len(contexts):
            raise ValueError(f"Paragraph ID out of range: {paragraph_id}")
        text = contexts[index]
    elif isinstance(contexts, dict):
        text = contexts[str(paragraph_id)]
    else:
        raise ValueError("context.json must contain a list or dictionary.")
    if not isinstance(text, str) or not text:
        raise ValueError(f"Paragraph {paragraph_id} must contain nonempty text.")
    return text


def best_span(start_logits, end_logits, offsets, n_best_size, max_answer_length):
    """Return (score, start character, exclusive end character).

    Only context tokens are eligible; question, special, and padding tokens
    have None offsets. The shortlist bounds computation for long windows.
    """
    valid = [i for i, offset in enumerate(offsets)
             if offset is not None and offset[1] > offset[0]]
    starts = sorted(valid, key=lambda i: start_logits[i], reverse=True)[:n_best_size]
    ends = sorted(valid, key=lambda i: end_logits[i], reverse=True)[:n_best_size]
    best = (-math.inf, 0, 0)
    for start in starts:
        for end in ends:
            if end < start or end - start + 1 > max_answer_length:
                continue
            score = start_logits[start] + end_logits[end]
            if offsets[end][1] > offsets[start][0] and score > best[0]:
                best = (score, offsets[start][0], offsets[end][1])
    return best



def write_predictions(examples, output_file):
    """Write CSV submissions or JSON details, creating the parent directory."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
            writer.writeheader()
            for example in examples:
                writer.writerow({"id": example["id"], "answer": example["prediction_text"]})
    elif output.suffix.lower() == ".json":
        with output.open("w", encoding="utf-8") as handle:
            json.dump(examples, handle, ensure_ascii=False, indent=2)
    else:
        raise ValueError("Output must end in .csv or .json.")
    return output


def main():
    import torch
    from accelerate import Accelerator
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    args = parse_args()
    accelerator = Accelerator()
    with open(args.test_file, encoding="utf-8") as handle:
        examples = json.load(handle)
    with open(args.context_file, encoding="utf-8") as handle:
        contexts = json.load(handle)
    if not isinstance(examples, list):
        raise ValueError("test.json must contain a JSON array.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("QA offset mapping requires a fast tokenizer.")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        raise ValueError("Tokenizer vocabulary exceeds the saved model embeddings.")
    max_length = min(args.max_seq_length, tokenizer.model_max_length)
    right = tokenizer.padding_side == "right"
    rows, metadata, paragraphs = [], [], []
    for row_index, example in enumerate(examples):
        if "relevant" not in example:
            raise ValueError(f"Example {row_index} has no relevant paragraph; run test_mc.py first.")
        paragraph = get_paragraph(contexts, example["relevant"])
        paragraphs.append(paragraph)
        question = example["question"].lstrip()
        question_length = len(tokenizer(question, add_special_tokens=False)["input_ids"])
        capacity = max_length - question_length - tokenizer.num_special_tokens_to_add(pair=True)
        if capacity <= args.doc_stride:
            raise ValueError(f"Example {row_index}: context token capacity {capacity} must exceed "
                             f"doc_stride {args.doc_stride}. Shorten the question, increase "
                             "max_seq_length within the model limit, or reduce doc_stride.")
        encoded = tokenizer(
            question if right else paragraph, paragraph if right else question,
            truncation="only_second" if right else "only_first",
            max_length=max_length, stride=args.doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding=False,
        )
        for window in range(len(encoded["input_ids"])):
            sequence_ids = encoded.sequence_ids(window)
            offsets = [offset if sequence_ids[i] == (1 if right else 0) else None
                       for i, offset in enumerate(encoded["offset_mapping"][window])]
            ids = encoded["input_ids"][window]
            null_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
            if args.version_2_with_negative and null_id not in ids:
                raise ValueError("No CLS/BOS token available for no-answer scoring.")
            cls_index = ids.index(null_id) if null_id in ids else 0
            feature_index = len(rows)
            rows.append({key: value[window] for key, value in encoded.items()
                         if key not in ("offset_mapping", "overflow_to_sample_mapping")}
                        | {"feature_index": feature_index})
            metadata.append((row_index, offsets, cls_index, len(ids)))

    if rows:
        multiple = (16 if accelerator.mixed_precision == "fp8" else
                    8 if accelerator.mixed_precision != "no" else None)

        def collate(features):
            indices = torch.tensor([f["feature_index"] for f in features], dtype=torch.long)
            inputs = [{k: v for k, v in f.items() if k != "feature_index"} for f in features]
            batch = tokenizer.pad(inputs, padding=True, pad_to_multiple_of=multiple, return_tensors="pt")
            batch["feature_index"] = indices
            return batch

        loader = DataLoader(Dataset.from_list(rows), shuffle=False, collate_fn=collate,
                            batch_size=args.per_device_test_batch_size)
        model, loader = accelerator.prepare(model, loader)
        model.eval()
        collected = {}
        for batch in tqdm(loader, desc="Extracting answers",
                          disable=not accelerator.is_local_main_process):
            indices = batch.pop("feature_index")
            with torch.inference_mode():
                outputs = model(**batch)
            starts = outputs.start_logits.float().cpu().tolist()
            ends = outputs.end_logits.float().cpu().tolist()
            records, scores = [], []
            for j, feature_index in enumerate(indices.cpu().tolist()):
                _, offsets, cls_index, length = metadata[feature_index]
                shift = 0 if right else len(starts[j]) - length
                start = starts[j][shift:shift + length]
                end = ends[j][shift:shift + length]
                score, char_start, char_end = best_span(
                    start, end, offsets, args.n_best_size, args.max_answer_length)
                records.append([feature_index, char_start, char_end])
                scores.append([score, start[cls_index] + end[cls_index]])
            records, scores = accelerator.gather_for_metrics((
                torch.tensor(records, dtype=torch.long, device=accelerator.device),
                torch.tensor(scores, dtype=torch.float32, device=accelerator.device),
            ))
            for record, score in zip(records.cpu().tolist(), scores.cpu().tolist()):
                collected[record[0]] = (score[0], record[1], record[2], score[1])
        if len(collected) != len(metadata):
            raise RuntimeError("Missing predictions for one or more context windows.")
        best = [(-math.inf, 0, 0) for _ in examples]
        null_scores = [math.inf for _ in examples]
        for feature_index, (score, start, end, null_score) in collected.items():
            row_index = metadata[feature_index][0]
            if score > best[row_index][0]:
                best[row_index] = (score, start, end)
            null_scores[row_index] = min(null_scores[row_index], null_score)
        for index, example in enumerate(examples):
            score, start, end = best[index]
            no_answer = (args.version_2_with_negative and
                         null_scores[index] - score > args.null_score_diff_threshold)
            example["prediction_text"] = "" if no_answer else paragraphs[index][start:end]

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output = write_predictions(examples, args.output_file)
        accelerator.print(f"Saved {len(examples)} answers to {output}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
