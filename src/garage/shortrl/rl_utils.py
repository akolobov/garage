# Some helper functions for using garage


import numpy as np
import torch

from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy, DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.sampler import FragmentWorker, LocalSampler, RaySampler
from garage.torch.optimizers import OptimizerWrapper


def get_mlp_policy(*,
                   env,
                   stochastic=True,
                   clip_output=False,
                   hidden_sizes=(64, 64),
                   hidden_nonlinearity=torch.tanh,
                   min_std=np.exp(-20.),
                   max_std=np.exp(2.)):

    if stochastic and clip_output:
        return TanhGaussianMLPPolicy(
                    env_spec=env.spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=None,
                    min_std=min_std,
                    max_std=max_std)

    if stochastic and not clip_output:
        return GaussianMLPPolicy(env.spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=None)

    if not stochastic:
        return DeterministicMLPPolicy(
                    env_spec=env.spec,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=torch.tanh if use_tanh else None)


def get_mlp_value(form='Q',*,
                  env,
                  hidden_sizes=(64, 64),
                  hidden_nonlinearity=torch.tanh
                  ):
    if form=='Q':
        return ContinuousMLPQFunction(
                env_spec=env.spec,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None)
    if form=='V':
        return GaussianMLPValueFunction(
                env_spec=env.spec,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None)


def get_sampler(policy, *,
                env,
                sampler_mode='ray',
                n_workers=4):

    if sampler_mode=='ray':
        return RaySampler(agents=policy,
                          envs=env,
                          max_episode_length=env.spec.max_episode_length,
                          n_workers=n_workers)

    elif n_workers==1:
        return LocalSampler(agents=policy,
                            envs=env,
                            max_episode_length=env.spec.max_episode_length,
                            worker_class=FragmentWorker)
    else:
        raise ValueError('Required sampler is unavailable.')



from garage.replay_buffer import PathBuffer

def get_replay_buferr(capacity=int(1e6)):
    return PathBuffer(capacity_in_transitions=capacity)

def get_optimizer(obj, lr, *,
                  max_optimization_epochs=1,
                  minibatch_size=128):

    return OptimizerWrapper((torch.optim.Adam, dict(lr=lr)),
                             obj,
                             max_optimization_epochs=max_optimization_epochs,
                             minibatch_size=minibatch_size)