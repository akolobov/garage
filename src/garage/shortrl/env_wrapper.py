import gym
import torch
import numpy as np

class ShortMDP(gym.Wrapper):
    """Short-RL wrapper for gym.Env.

    This wrapper reshapes the reward using a heuristic.

    Args:
        env: The environment to be wrapped.
    """

    def __init__(self, env, heuristic=None, lambd=1.0, gamma=1.0, scale=1.0):
        super().__init__(env)
        self._heuristic = heuristic
        self._lambd = lambd
        self._gamma = gamma
        self._scale = scale

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        info['orig_reward'] = reward
        info['lambda'] = self._lambd
        info['gamma'] = self._gamma
        info['scale'] = self._scale

        if (not done or 'TimeLimit.truncated' in info) and self._heuristic is not None:
            reward += (1-self._lambd) * self._gamma * self._heuristic(np.array([obs]))

        reward *= self._scale
        return obs, reward, done, info
