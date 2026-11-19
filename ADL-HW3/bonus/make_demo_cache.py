"""Generate clearly labeled synthetic rewards for learning the PPO workflow."""
import argparse
from pathlib import Path
import numpy as np
from .data import RewardCache


def make_demo(output_dir, train_queries=300, val_queries=150, seed=42):
    if train_queries < 1 or val_queries < 1:
        raise ValueError("Both splits must have at least one query")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for split, count in (("train", train_queries), ("val", val_queries)):
        category = rng.integers(0, 3, size=count)
        features = np.concatenate([np.eye(3)[category], rng.normal(0, .1, size=(count, 3))], axis=1)
        rewards = np.full((count, 3), .15, dtype=np.float32)
        rewards[np.arange(count), category] = .9
        rewards += rng.uniform(-.01, .01, rewards.shape).astype(np.float32)
        cache = RewardCache([f"SYNTHETIC-{split}-{i}" for i in range(count)], features, rewards,
                            metadata={"synthetic_demo": True, "feature_schema": "synthetic-onehot-v1",
                                      "pipeline_id": "synthetic-demo-not-rag"})
        path = out / f"{split}.npz"
        cache.save(path)
        paths.append(path)
    return tuple(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="bonus/cache/demo")
    parser.add_argument("--train-queries", type=int, default=300)
    parser.add_argument("--val-queries", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = make_demo(args.output_dir, args.train_queries, args.val_queries, args.seed)
    print("SYNTHETIC DEMO ONLY: these are not assignment scores.")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
