"""Compare deterministic policy choices with fixed-M baselines."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import RewardCache
from .inference import PassageSelector


def evaluate_policy(selector, cache):
    if tuple(selector.actions.tolist()) != cache.actions:
        raise ValueError("Evaluation cache has a different action mapping")
    for key in ("feature_schema", "synthetic_demo", "pipeline_id"):
        if selector.config.get("cache_metadata", {}).get(key) != cache.metadata.get(key):
            raise ValueError(f"Evaluation metadata mismatch: {key}")
    chosen = selector.predict_many(cache.features)
    columns = np.asarray([cache.actions.index(int(m)) for m in chosen])
    rewards = cache.rewards[np.arange(len(chosen)), columns]
    return {"queries": len(chosen), "synthetic_demo": bool(cache.metadata.get("synthetic_demo", False)),
            "policy_mean_reward": float(rewards.mean()), "average_top_m": float(chosen.mean()),
            "fixed_m_mean_rewards": {str(m): float(cache.rewards[:, i].mean()) for i, m in enumerate(cache.actions)},
            "action_counts": {str(m): int((chosen == m).sum()) for m in cache.actions},
            "oracle_mean_reward": float(cache.rewards.max(axis=1).mean()),
            "oracle_note": "Label-aware diagnostic upper bound; not a deployable policy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default="results/ppo_evaluation.json")
    args = parser.parse_args()
    
    selector = PassageSelector(args.run_dir)
    cache = RewardCache.load(args.cache)
    used = set(selector.config["training_qids"])
    if used & set(cache.qids):
        raise ValueError("Evaluation queries overlap the policy-training queries")
    metrics = evaluate_policy(selector, cache)
    metrics["evaluation_role"] = ("validation_used_for_checkpoint_selection"
                                   if set(cache.qids) & set(selector.config["validation_qids"])
                                   else "held_out_from_policy_training_and_selection")
    

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
