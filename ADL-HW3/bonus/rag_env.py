"""Gymnasium contextual bandit: each episode is one RAG query."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import RewardCache


class RAGPassageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features, rewards, actions=(1, 3, 5)):
        super().__init__()
        cache = RewardCache([str(i) for i in range(len(features))], features, rewards, actions)
        self.features, self.rewards, self.actions = cache.features, cache.rewards, cache.actions
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.features.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))
        self.current_index = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = int(self.np_random.integers(len(self.features)))
        return self.features[self.current_index].copy(), {}

    def step(self, action):
        if self.current_index is None:
            raise RuntimeError("Call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action index: {action}")
        index, action = self.current_index, int(action)
        reward = float(self.rewards[index, action])
        self.current_index = None
        terminal = np.zeros(self.observation_space.shape, dtype=np.float32)
        return terminal, reward, True, False, {"query_index": index, "top_m": self.actions[action]}
