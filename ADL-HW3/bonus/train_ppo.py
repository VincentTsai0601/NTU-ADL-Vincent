"""Train a CPU PPO controller on cached single-step RAG rewards."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from .data import RewardCache, FeatureNormalizer, check_compatible
from .rag_env import RAGPassageEnv
import gymnasium, stable_baselines3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timesteps", type=int, default=10240)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-freq", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=.01)
    args = parser.parse_args()
    train_policy(args.train_cache, args.val_cache, args.output_dir,
                 args.timesteps, args.n_steps, args.batch_size, args.eval_freq,
                 args.seed, args.learning_rate, args.entropy_coef)

class HeldoutCallback(BaseCallback):
    def __init__(self, features, rewards, actions, output, eval_freq):
        super().__init__()
        self.features, self.rewards, self.actions = features, rewards, np.asarray(actions)
        self.output, self.eval_freq = Path(output), eval_freq
        self.best = -np.inf
        self.rows = []

    def record(self, phase):
        choices, _ = self.model.predict(self.features, deterministic=True)
        choices = np.asarray(choices, dtype=np.int64).reshape(-1)
        reward = float(self.rewards[np.arange(len(choices)), choices].mean())
        row = {"timesteps": self.num_timesteps, "phase": phase,
               "validation_reward": reward, "average_top_m": float(self.actions[choices].mean())}
        if reward > self.best:
            self.best = reward
            self.model.save(str(self.output / "best_policy"))
            (self.output / "selection.json").write_text(json.dumps(row, indent=2) + "\n")
        self.rows.append(row)
        with (self.output / "evaluation.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)

    def _on_training_start(self):
        self.record("initial")

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            self.record("rollout")
        return True

    def _on_training_end(self):
        self.record("final")


def train_policy(train_cache, val_cache, output_dir, total_timesteps=10240,
                 n_steps=256, batch_size=64, eval_freq=1024, seed=42,
                 learning_rate=3e-4, ent_coef=.01, verbose=1):
    if min(total_timesteps, n_steps, batch_size, eval_freq) < 1 or n_steps < 2 or batch_size < 2:
        raise ValueError("Use positive timesteps/eval frequency and n_steps/batch_size >= 2")
    if n_steps % batch_size:
        raise ValueError("Choose a batch size that divides n_steps")
    train, val = RewardCache.load(train_cache), RewardCache.load(val_cache)
    check_compatible(train, val)
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is nonempty; use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    normalizer = FeatureNormalizer.fit(train.features)
    train_x, val_x = normalizer.transform(train.features), normalizer.transform(val.features)
    normalizer.save(out / "feature_stats.npz", train.actions)
    config = {"algorithm": "PPO", "problem": "one-step contextual bandit", "seed": seed,
              "requested_timesteps": total_timesteps, "n_steps": n_steps, "batch_size": batch_size,
              "learning_rate": learning_rate, "ent_coef": ent_coef, "gamma": 0.0,
              "gae_lambda": 1.0, "n_epochs": 10, "net_arch": [64, 64], "eval_freq": eval_freq,
              "training_qids": train.qids, "validation_qids": val.qids,
              "actions": train.actions, "feature_dimension": train.features.shape[1],
              "cache_metadata": train.metadata,
              "train_cache_sha256": hashlib.sha256(Path(train_cache).read_bytes()).hexdigest(),
              "validation_cache_sha256": hashlib.sha256(Path(val_cache).read_bytes()).hexdigest(),
              "versions": {"torch": str(torch.__version__)}}
    
    config["versions"].update(gymnasium=gymnasium.__version__, stable_baselines3=stable_baselines3.__version__)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if train.metadata.get("synthetic_demo"):
        print("SYNTHETIC DEMO: these rewards are not RAG/assignment performance.")
    env = RAGPassageEnv(train_x, train.rewards, train.actions)
    check_env(env, warn=True)
    env = Monitor(env, filename=str(out / "training"))
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, n_steps=n_steps,
                batch_size=batch_size, n_epochs=10, gamma=0.0, gae_lambda=1.0,
                ent_coef=ent_coef, policy_kwargs={"net_arch": [64, 64]},
                seed=seed, device="cpu", verbose=verbose)
    callback = HeldoutCallback(val_x, val.rewards, train.actions, out, eval_freq)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(str(out / "final_policy"))
    from .inference import PassageSelector
    from .evaluate import evaluate_policy
    metrics = evaluate_policy(PassageSelector(out), val)
    metrics["evaluation_role"] = "validation_used_for_checkpoint_selection"
    metrics["actual_timesteps"] = model.num_timesteps
    (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    env.close()
    print(json.dumps(metrics, indent=2))
    return metrics





if __name__ == "__main__":
    main()
