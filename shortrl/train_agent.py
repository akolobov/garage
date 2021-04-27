import torch
import gym

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.trainer import Trainer

from shortrl.env_wrapper import ShortMDP
from shortrl.algorithms import get_algo
from shortrl.heuristics import get_snapshot_values


def train_agent(ctxt=None,
                env_name='InvertedDoublePendulum-v2', # gym env identifier
                discount=1.0,  # oirginal discount
                heuristic=None,  # a python function
                lambd=0.9,  # extra discount
                seed=1,  # random seed
                n_epochs=50,  # number of updates
                batch_size=10000,  # number of samples collected per update
                **kwargs,
                ):

    assert discount is not None
    if heuristic is None:
        heuristic = lambda x : 0.

    set_seed(seed)
    # Wrap the gym env into our *gym* wrapper first and then into the standard garage wrapper.
    env = ShortMDP(gym.make(env_name), heuristic, lambd=lambd, gamma=discount)
    env = GymEnv(env)

    algo = get_algo(env=env,
                    discount=discount*lambd,  #  algorithm sees a shorter horizon,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    **kwargs)

    trainer = Trainer(ctxt)
    trainer.setup(algo, env, lambd)
    trainer.train(n_epochs=n_epochs, batch_size=batch_size)


# Run this.
def run_exp(exp_name=None,
            snapshot_frequency=1,
            log_prefix='agents',
            **kwargs):
    snapshot_gap = snapshot_frequency if snapshot_frequency>0 else 1
    snapshot_mode = 'gap_and_last' if snapshot_frequency>0 else 'last'
    wrapped_train_agent = wrap_experiment(train_agent,
                            prefix='experiment/shortrl/'+log_prefix,
                            snapshot_mode=snapshot_mode,
                            snapshot_gap=snapshot_gap,
                            name=exp_name,
                            use_existing_dir=True)  # overwrites existing directory
    return wrapped_train_agent(**kwargs)


if __name__ == '__main__':
    # Parse command line inputs.
    import argparse
    from shortrl.utils import str2bool
    parser = argparse.ArgumentParser()

    parser.add_argument('-u', '--use_heuristic', type=str2bool, default=False)
    # arguments for run_exp
    parser.add_argument('--snapshot_frequency', type=int, default=0)
    parser.add_argument('--log_prefix', type=str, default='agents')
    # arguments for train_agent
    parser.add_argument('-e', '--env_name', type=str, default='InvertedDoublePendulum-v2')
    parser.add_argument('-d', '--discount', type=float, default=1.0)
    parser.add_argument('-l', '--lambd', type=float, default=0.9)
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-N', '--n_epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=10000)
    # arguments for get_algo
    parser.add_argument('-a', '--algo_name', type=str, default='PPO')
    parser.add_argument('-n', '--n_workers', type=int, default=4)
    parser.add_argument('--value_lr', type=float, default=5e-3)
    parser.add_argument('--policy_lr', type=float, default=2e-4)

    args = parser.parse_args()

    # Run experiment.
    args_dict = vars(args)
    heuristic = get_snapshot_values(
                'data/local/experiment/shortrl/heuristics/PPO_Inver_0.99_False_1/',
                itr=30) if args.use_heuristic else None
    exp_name = args.algo_name+'_'+args.env_name[:min(len(args.env_name),5)]+\
                '_{}_{}_{}'.format(args.lambd, args.use_heuristic, args.seed)
    del args_dict['use_heuristic']

    run_exp(exp_name=exp_name,
            heuristic=heuristic,
            **args_dict,
            )
