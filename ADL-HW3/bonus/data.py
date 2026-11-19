"""Validated reward tables and train-only feature normalization."""
import json
from pathlib import Path

import numpy as np


def build_features(query_embedding, reranker_scores, passage_token_lengths, top_k=10):
    embedding = np.asarray(query_embedding, dtype=np.float32)
    scores = np.asarray(reranker_scores, dtype=np.float32)
    lengths = np.asarray(passage_token_lengths, dtype=np.float32)
    if top_k < 1 or embedding.ndim != 1 or embedding.size == 0:
        raise ValueError("Use a nonempty query embedding and positive top_k")
    if scores.ndim != 1 or lengths.ndim != 1 or scores.shape != lengths.shape:
        raise ValueError("Scores and passage lengths must be aligned 1-D vectors")
    if (lengths < 0).any():
        raise ValueError("Token lengths cannot be negative")
    n = min(len(scores), top_k)
    padded_scores = np.zeros(top_k, dtype=np.float32)
    padded_lengths = np.zeros(top_k, dtype=np.float32)
    padded_scores[:n], padded_lengths[:n] = scores[:n], lengths[:n]
    result = np.concatenate([embedding, padded_scores, padded_lengths])
    if not np.isfinite(result).all():
        raise ValueError("Features must be finite")
    return result


class RewardCache:
    def __init__(self, qids, features, rewards, actions=(1, 3, 5), metadata=None):
        self.qids = [str(qid) for qid in qids]
        self.features = np.asarray(features, dtype=np.float32)
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.actions = tuple(int(a) for a in actions)
        self.metadata = dict(metadata or {})
        if not self.actions or len(set(self.actions)) != len(self.actions) or min(self.actions) < 1:
            raise ValueError("Actions must be distinct positive passage counts")
        if not self.qids or len(set(self.qids)) != len(self.qids):
            raise ValueError("Cache needs nonempty, unique query IDs")
        if self.features.ndim != 2 or self.features.shape[0] != len(self.qids) or self.features.shape[1] == 0:
            raise ValueError("features must have shape [number of queries, D], D > 0")
        if self.rewards.shape != (len(self.qids), len(self.actions)):
            raise ValueError("rewards must have one column per action and one row per query")
        if not np.isfinite(self.features).all() or not np.isfinite(self.rewards).all():
            raise ValueError("Cache contains NaN or infinity")

    def save(self, path):
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("New caches must use the .npz extension")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, qids=np.asarray(self.qids), features=self.features,
                            rewards=self.rewards, actions=np.asarray(self.actions),
                            metadata=np.asarray(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix == ".pt":
            import torch
            obj = torch.load(path, map_location="cpu", weights_only=True)
            return cls(obj["qids"], obj["features"], obj["rewards"],
                       obj.get("actions", (1, 3, 5)), obj.get("metadata"))
        with np.load(path, allow_pickle=False) as obj:
            metadata = json.loads(str(obj["metadata"].item())) if "metadata" in obj else {}
            return cls(obj["qids"].tolist(), obj["features"], obj["rewards"],
                       obj["actions"].tolist(), metadata)


class FeatureNormalizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.ndim != 1 or self.mean.shape != self.std.shape or not (self.std > 0).all():
            raise ValueError("Invalid normalizer statistics")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Normalizer statistics must be finite")

    @classmethod
    def fit(cls, training_features):
        x = np.asarray(training_features, dtype=np.float32)
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("Need finite training observations with shape [N, D]")
        return cls(x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6))

    def transform(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim not in (1, 2) or x.shape[-1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Feature dimension mismatch or nonfinite input")
        result = ((x - self.mean) / self.std).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite normalized features")
        return result

    def save(self, path, actions):
        np.savez(path, mean=self.mean, std=self.std, actions=np.asarray(actions))


def check_compatible(training, validation):
    if set(training.qids) & set(validation.qids):
        raise ValueError("Training and evaluation query IDs overlap")
    if training.actions != validation.actions:
        raise ValueError("Action mappings differ")
    if training.features.shape[1] != validation.features.shape[1]:
        raise ValueError("Feature dimensions differ")
    for key in ("feature_schema", "synthetic_demo", "pipeline_id"):
        if training.metadata.get(key) != validation.metadata.get(key):
            raise ValueError(f"Cache metadata differs for {key}")
