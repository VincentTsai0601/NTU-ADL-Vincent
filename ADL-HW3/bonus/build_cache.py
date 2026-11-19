"""Convert real, precomputed RAG outcome records from JSONL into an NPZ cache."""
import argparse
import json
from .data import RewardCache, build_features


def features_from_record(record, top_k=10):
    if "features" in record:
        return record["features"]
    return build_features(record["query_embedding"], record["reranker_scores"],
                          record["passage_token_lengths"], top_k)


def from_records(records, actions=(1, 3, 5), metadata=None, top_k=10):
    rows = list(records)
    if not rows:
        raise ValueError("No records found")
    for row in rows:
        candidate_count = row.get("candidate_count")
        if candidate_count is None and "reranker_scores" in row:
            candidate_count = len(row["reranker_scores"])
        if candidate_count is not None and candidate_count < max(actions):
            raise ValueError(f"Query {row['qid']} has too few candidates for the action set")
    return RewardCache([r["qid"] for r in rows],
                       [features_from_record(r, top_k) for r in rows],
                       [r["rewards"] for r in rows], actions, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL with qid, features, rewards")
    parser.add_argument("--output", required=True, help="Destination .npz")
    parser.add_argument("--actions", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--metadata", help="JSON file describing checkpoints and generation settings")
    args = parser.parse_args()
    metadata = {"synthetic_demo": False}
    if args.metadata:
        with open(args.metadata, encoding="utf-8") as stream:
            metadata.update(json.load(stream))
    with open(args.input, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    raw_vectors = all("features" in row for row in rows)
    metadata.setdefault("feature_schema", "provided-vector-v1" if raw_vectors else f"embedding-scores-lengths-k{args.top_k}-v1")
    cache = from_records(rows, args.actions, metadata, args.top_k)
    cache.save(args.output)
    print(f"Saved {len(cache.qids)} queries to {args.output}")


if __name__ == "__main__":
    main()
