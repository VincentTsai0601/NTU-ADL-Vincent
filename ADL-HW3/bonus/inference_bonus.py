"""Run original RAG components with fixed-M or PPO passage selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .inference import PassageSelector
from .rag_backend import add_backend_arguments, backend_from_args, read_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(parser)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", help="PPO run directory")
    mode.add_argument("--fixed-m", type=int, help="Matched fixed-M baseline")
    parser.add_argument("--test-data-path", default="data/test_open.txt")
    parser.add_argument("--qrels-path", default="data/qrels.txt")
    parser.add_argument("--output", default="results/rl_bonus.json")
    parser.add_argument("--limit", type=int, default=0, help="0 means all query rows")
    args = parser.parse_args()
    
    if args.limit < 0 or (args.fixed_m is not None and not 1 <= args.fixed_m <= args.top_k):
        raise ValueError("Invalid limit or fixed passage count")
    rows = read_queries(args.test_data_path)
    if args.limit:
        rows = rows[:args.limit]
    selector = PassageSelector(args.run_dir) if args.run_dir else None
    if selector and max(selector.actions) > args.top_k:
        raise ValueError("Policy can select more passages than top_k")
    if selector and set(r["qid"] for r in rows) & set(selector.config["training_qids"]):
        raise ValueError("Inference/evaluation set overlaps policy-training queries")
    qrels = json.loads(Path(args.qrels_path).read_text()) if Path(args.qrels_path).is_file() else {}
    backend = backend_from_args(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if selector:
        trained = selector.config.get("cache_metadata", {})
        if trained.get("synthetic_demo") or trained.get("pipeline_id") != backend.metadata["pipeline_id"]:
            backend.close()
            raise ValueError("Policy was trained on synthetic rewards or a different RAG pipeline; rebuild its real caches")
    try:
        with output.with_suffix(".records.jsonl").open("w", encoding="utf-8") as stream:
            for i, row in enumerate(rows):
                result = backend.retrieve(row["rewrite"])
                m = selector.select_top_m(result["features"]) if selector else args.fixed_m
                contexts = result["texts"][:m]
                answer = backend.generate(row["rewrite"], contexts)
                label = row.get("answer")
                reference = label.get("text") if isinstance(label, dict) else None
                similarity = backend.score(answer, reference) if isinstance(reference, str) else None
                gold = {str(pid) for pid, label in qrels.get(row["qid"], {}).items() if float(label) > 0}
                rank = next((n for n, pid in enumerate(result["pids"], 1) if pid in gold), None)
                record = {"qid": row["qid"], "generated": answer, "reference_answer": reference,
                          "top_m": m, "context_tokens": sum(result["token_lengths"][:m]),
                          "reranked_pids": result["pids"], "answer_similarity": similarity,
                          "recall_at_k": len(gold & set(result["retrieved_pids"])) / len(gold) if gold else None,
                          "mrr_at_k": (1.0 / rank if rank else 0.0) if gold else None}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{i+1}/{len(rows)} {row['qid']} M={m}", flush=True)
    finally:
        backend.close()
    def mean_present(key):
        values = [r[key] for r in records if r[key] is not None]
        return float(np.mean(values)) if values else None
    metrics = {"queries": len(records), "policy": args.run_dir or f"fixed_m_{args.fixed_m}",
               "answer_similarity": mean_present("answer_similarity"),
               f"recall@{args.top_k}": mean_present("recall_at_k"),
               f"mrr@{args.top_k}": mean_present("mrr_at_k"),
               "average_top_m": mean_present("top_m"), "average_context_tokens": mean_present("context_tokens"),
               "pipeline_metadata": backend.metadata,
               "comparison_note": "Compare PPO with --fixed-m in this bonus script: both use identical completion-only parsing and greedy generation.",
               "records": records}
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"records", "pipeline_metadata"}}, indent=2))


if __name__ == "__main__":
    main()
