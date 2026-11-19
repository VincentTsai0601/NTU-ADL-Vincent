"""Load a policy and choose passage counts using observations only."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .data import FeatureNormalizer
from .build_cache import features_from_record


class PassageSelector:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        with np.load(self.run_dir / "feature_stats.npz", allow_pickle=False) as obj:
            self.normalizer = FeatureNormalizer(obj["mean"], obj["std"])
            self.actions = obj["actions"].astype(np.int64)
        self.model = PPO.load(str(self.run_dir / "best_policy.zip"), device="cpu")
        self.config = json.loads((self.run_dir / "config.json").read_text())

    def predict_many(self, raw_features):
        x = self.normalizer.transform(raw_features)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Use a nonempty [N, D] feature matrix")
        indices, _ = self.model.predict(x, deterministic=True)
        return self.actions[np.asarray(indices, dtype=np.int64).reshape(-1)]

    def select_top_m(self, raw_features):
        x = np.asarray(raw_features, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("select_top_m expects one 1-D observation")
        return int(self.predict_many(x[None, :])[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True, help="JSONL with qid and inference-time features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    
    selector = PassageSelector(args.run_dir)
    with open(args.input, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    choices = selector.predict_many([features_from_record(r, args.top_k) for r in rows])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row, m in zip(rows, choices):
            stream.write(json.dumps({"qid": str(row["qid"]), "top_m": int(m)}) + "\n")
    print(f"Saved {len(rows)} passage-count decisions to {output}")


if __name__ == "__main__":
    main()
