#!/usr/bin/env python3
"""This is an example to train a task with PPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch
import gym

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.experiment import Snapshotter
from shortrl.env_wrapper import ShortMDP


def torch_value_wrapper(torch_value):
    def wrapped_fun(x):
        with torch.no_grad():
            # BUG: We need to package "obs" correctly (the way the value function object expects it to be packaged...)
            return torch_value(torch.Tensor(x))
    return wrapped_fun

# Before running this, make sure to run `python -m shortrl.ppo_pendulum_train_heuristics.py`
def ppo_pendulum(ctxt=None, seed=1, lambd=0.9):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    snapshotter = Snapshotter()
    # Load the value function from one of the snapshots generated by shortrl.ppo_pendulum_train_heuristics.
    data = snapshotter.load('data/local/experiment/shortrl/heuristics/ppo_pendulum/', itr=19)
    heuristic = torch_value_wrapper(data['algo']._value_function)
    discount = data['algo']._discount

    set_seed(seed)
    # Wrap the gym env into our *gym* wrapper first and then into the standard garage wrapper.
    env = ShortMDP(gym.make('InvertedDoublePendulum-v2'), heuristic, lambd=lambd, gamma=discount)
    env = GymEnv(env)

    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[64, 64],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length)
                         # n_workers = 48)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=lambd*discount,
               center_adv=False)

    trainer.setup(algo, env, lambd)
    trainer.train(n_epochs=50, batch_size=10000)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lambd', type=float, default=0.9)
    parser.add_argument('-d', '--seed', type=int, default=1)
    args = parser.parse_args()

    name = 'ppo_pendulum'
    lambd = args.lambd
    seed = args.seed

    ppo_pendulum = wrap_experiment(ppo_pendulum,
                    prefix='experiment/shortrl/agents',
                    snapshot_mode='last',
                    name='ppo_pendulum_{}_{}'.format(lambd, seed))

    ppo_pendulum(seed=seed, lambd=lambd)
