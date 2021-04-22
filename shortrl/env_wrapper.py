import gym
import torch
import numpy as np

class ShortMDP(gym.Wrapper):
    """Short-RL wrapper for gym.Env.

    This wrapper reshapes the reward using a heuristic.

    Args:
        env: The environment to be wrapped.
    """

    def __init__(self, env, heuristic, lambd, gamma):
        super().__init__(env)
        self._heuristic = heuristic
        self._lambd = lambd
        self._gamma = gamma

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        info['orig_reward'] = reward
        if not done:
            reward += (1-self._lambd) * self._gamma * self._heuristic(np.array([obs]))
        return obs, reward, done, info
