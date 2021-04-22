import gym
import torch

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
        if not done:
            with torch.no_grad():
                # BUG: We need to package "obs" correctly (the way the value function object expects it to be packaged...)
                reward += (1-self._lambd) * self._gamma * self._heuristic(torch.Tensor(obs))
        return obs, reward, done, info
