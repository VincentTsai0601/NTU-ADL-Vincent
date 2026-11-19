"""Collect real rewards using the existing retriever, reranker, index and Qwen."""
import argparse
import json
from pathlib import Path
from .data import RewardCache
from .rag_backend import add_backend_arguments, backend_from_args, load_split_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(parser)
    parser.add_argument("--train-path", default="data/train.txt")
    parser.add_argument("--split-path", default="runs/retriever/split.json")
    parser.add_argument("--output-dir", default="bonus/cache/real")
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--val-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actions", nargs="+", type=int, default=[1, 3, 5])
    args = parser.parse_args()
    if min(args.actions) < 1 or len(set(args.actions)) != len(args.actions) or max(args.actions) > args.top_k:
        raise ValueError("Actions must be distinct positive counts no larger than top_k")
    train, val = load_split_queries(args.train_path, args.split_path, args.train_limit, args.val_limit, args.seed)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh cache output directory")
    out.mkdir(parents=True, exist_ok=True)
    backend = backend_from_args(args)
    (out / "metadata.json").write_text(json.dumps(backend.metadata, indent=2) + "\n")
    (out / "selected_queries.json").write_text(json.dumps({"training": [r["qid"] for r in train],
                    "validation": [r["qid"] for r in val], "seed": args.seed}, indent=2) + "\n")
    try:
        for name, rows in (("train", train), ("val", val)):
            records = []
            with (out / f"{name}_records.jsonl").open("w", encoding="utf-8") as stream:
                for i, row in enumerate(rows):
                    query = row["rewrite"]
                    label = row.get("answer")
                    reference = label.get("text") if isinstance(label, dict) else None
                    if not isinstance(reference, str):
                        raise ValueError(f"Query {row['qid']} has no reference answer")
                    retrieval = backend.retrieve(query)
                    answers, rewards = [], []
                    for m in args.actions:
                        answer = backend.generate(query, retrieval["texts"][:m])
                        answers.append(answer)
                        rewards.append(backend.score(answer, reference))
                    record = {"qid": row["qid"], "features": retrieval["features"].tolist(),
                              "rewards": rewards, "actions": args.actions, "answers": answers,
                              "reference_answer": reference, "candidate_count": len(retrieval["pids"]),
                              "reranked_pids": retrieval["pids"],
                              "context_tokens": [sum(retrieval["token_lengths"][:m]) for m in args.actions]}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    records.append(record)
                    print(f"{name} {i+1}/{len(rows)}: {row['qid']} rewards={rewards}", flush=True)
            RewardCache([r["qid"] for r in records], [r["features"] for r in records],
                        [r["rewards"] for r in records], args.actions, backend.metadata).save(out / f"{name}.npz")
    finally:
        backend.close()
    print(f"Real reward caches saved under {out}")


if __name__ == "__main__":
    main()
