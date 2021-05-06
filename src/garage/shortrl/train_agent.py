import torch
import gym
import os
import numpy as np


from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.shortrl.trainer import Trainer
from garage.shortrl.env_wrapper import ShortMDP
from garage.shortrl.algorithms import get_algo
from garage.shortrl import lambda_schedulers
from garage.shortrl.heuristics import load_policy_from_snapshot, load_heuristic_from_snapshot
from garage.shortrl.pretrain import init_policy_from_baseline


def pretrain_agent(ctxt=None,
                   baseline_policy=None,  # when provided, it will be use to warmstart the learner policy
                   warmstart_mode='bc', # how the warmstart is done; 'bc' or 'copy';
                   warmstart_batch_size=None,
                   save_mode='light',
                   **kwargs,  # kwargs used by the train_agent
                   ):

    assert baseline_policy is not None

    # Set the random seed
    seed = kwargs.get('seed',1)
    set_seed(seed)
    # Wrap the gym env into our *gym* wrapper first and then into the standard garage wrapper.
    env = GymEnv(gym.make(kwargs['env_name']))
    # Get an algorithm template to get a compatible intial policy
    algo_ = get_algo(env=env, n_epochs=1, **kwargs)
    batch_size = warmstart_batch_size or kwargs.get('batch_size',10000)
    init_policy = init_policy_from_baseline(
                        algo_.policy,
                        baseline_policy=baseline_policy,
                        # bc parameters
                        mode=warmstart_mode,
                        env=env,
                        n_epochs=1,
                        policy_lr=1e-3,
                        batch_size=batch_size,
                        opt_n_grad_steps=batch_size,
                        n_workers=kwargs.get('n_workers',4),
                        ctxt=ctxt)
    return init_policy

def train_agent(ctxt=None,
                env_name='InvertedDoublePendulum-v2', # gym env identifier
                discount=0.99,  # oirginal discount
                heuristic=None,  # a python function
                init_policy=None,
                lambd=1.0,  # extra discount
                ls_n_epochs=None, # n_epoch for lambd to converge to 1.0 (default: n_epoch)
                ls_cls='TanhLS', # class of LambdaScheduler
                seed=1,  # random seed
                total_n_samples=50*10000,  # total number of samples
                batch_size=10000,  # number of samples collected per update
                ignore_shutdown=False,  # do not shutdown workers after training
                save_mode='light',
                **kwargs,
                ):

    assert discount is not None
    heuristic = heuristic or (lambda x : 0.)
    n_epochs = np.ceil(total_n_samples/batch_size)

    # Set the random seed
    set_seed(seed)

    # Wrap the gym env into our *gym* wrapper first and then into the standard garage wrapper.
    env = ShortMDP(gym.make(env_name), heuristic, lambd=lambd, gamma=discount)
    env = GymEnv(env)

    # Initialize the algorithm
    algo = get_algo(env=env,
                    discount=discount*lambd,  #  algorithm sees a shorter horizon,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    init_policy=init_policy,
                    **kwargs)

    ls_n_epochs = ls_n_epochs or n_epochs
    ls = getattr(lambda_schedulers, ls_cls)(init_lambd=lambd, n_epochs=ls_n_epochs)
    trainer = Trainer(ctxt)
    trainer.setup(algo, env, lambd=ls, discount=discount, save_mode=save_mode)

    return trainer.train(n_epochs=n_epochs,
                         batch_size=batch_size,
                         ignore_shutdown=ignore_shutdown)

# Run this.
def run_exp(*,
            exp_name,
            snapshot_frequency=0,  # 0 means only taking the last snapshot
            log_root=None,
            log_prefix='agents',
            log_dir=None,
            seed=1,
            save_mode='light',
            # pretain
            baseline_policy=None,  # when provided, it will be use to warmstart the learner policy
            warmstart_mode='bc', # how the warmstart is done; 'bc' or 'copy';
            warmstart_batch_size=None,
            **kwargs):

    snapshot_gap = snapshot_frequency if snapshot_frequency>0 else 1
    snapshot_mode = 'gap_and_last' if snapshot_frequency>0 else 'last'
    prefix= os.path.join('shortrl',log_prefix,exp_name)
    name=str(seed)
    log_root = log_root or '.'
    # mimic garage's logging behavior
    log_dir = os.path.join(log_root,'data','local','experiment',prefix, name)


    # Pretrain agent
    if baseline_policy is None:
        init_policy = None
    else:
        pretrain_log_dir = os.path.join(log_dir,'pretrain_log')
        wrapped_pretrain_agent = wrap_experiment(pretrain_agent,
                                    log_dir=pretrain_log_dir,  # overwrite
                                    snapshot_mode=snapshot_mode,
                                    snapshot_gap=snapshot_gap,
                                    archive_launch_repo=save_mode!='light',
                                    use_existing_dir=True)  # overwrites existing directory
        init_policy = wrapped_pretrain_agent(seed=seed, save_mode=save_mode,
                        baseline_policy=baseline_policy,
                        warmstart_mode=warmstart_mode,
                        warmstart_batch_size=warmstart_batch_size,
                        **kwargs)

    # Train agent
    wrapped_train_agent = wrap_experiment(train_agent,
                            log_dir=log_dir,  # overwrite
                            prefix=prefix,
                            snapshot_mode=snapshot_mode,
                            snapshot_gap=snapshot_gap,
                            archive_launch_repo=save_mode!='light',
                            name=name,
                            use_existing_dir=True)  # overwrites existing directory
    wrapped_train_agent(seed=seed, save_mode=save_mode, init_policy=init_policy, **kwargs)


def simple_run_exp(*,
                   use_heuristic=False,
                   warmstart_policy=False,
                   data_path=None,
                   data_itr=None,
                   **kwargs):
    """ A wrapper of run_exp that takes basic dtypes as inputs, so that run_exp
        can be more easily called.

        run_exp, which runs train_agent, requires baseline_policy and heuristic
        to be provided as python functions. This method wraps
    """
    if use_heuristic or warmstart_policy:
        assert data_itr is not None and data_path is not None

    heuristic = load_heuristic_from_snapshot(data_path, data_itr) if use_heuristic else None
    baseline_policy = load_policy_from_snapshot(data_path, data_itr) if warmstart_policy else None
    algo_name = kwargs['algo_name']
    env_name = kwargs['env_name']
    lambd = kwargs['lambd']

    exp_name = algo_name+'_'+ env_name[:min(len(env_name),5)]+\
                '_{}_{}_{}'.format(lambd, str(use_heuristic)[0], str(warmstart_policy)[0])

    return  run_exp(exp_name=exp_name,
            heuristic=heuristic,
            baseline_policy=baseline_policy,
            **kwargs,
            )

if __name__ == '__main__':
    # Parse command line inputs.
    import argparse
    from garage.shortrl.utils import str2bool
    parser = argparse.ArgumentParser()

    # arguments for simple_run_exp
    parser.add_argument('-u', '--use_heuristic', type=str2bool, default=False)
    parser.add_argument('-w', '--warmstart_policy', type=str2bool, default=False)
    parser.add_argument('--data_path', type=str, default='data/local/experiment/shortrl/heuristics/PPO_Inver_1.0_F/1/')
    parser.add_argument('--data_itr', type=int, default=15)
    # arguments for run_exp
    parser.add_argument('--snapshot_frequency', type=int, default=0)
    parser.add_argument('--log_root', type=str, default=None)
    parser.add_argument('--log_prefix', type=str, default='agents')
    parser.add_argument('--save_mode', type=str, default='light')
    # arguments for train_agent
    parser.add_argument('-e', '--env_name', type=str, default='InvertedDoublePendulum-v2')
    parser.add_argument('-d', '--discount', type=float, default=0.99)
    parser.add_argument('-l', '--lambd', type=float, default=1.0)
    parser.add_argument('--ls_n_epochs', type=int, default=None)
    parser.add_argument('--ls_cls', type=str, default='TanhLS')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-N', '--total_n_samples', type=int, default=500000)
    parser.add_argument('-b', '--batch_size', type=int, default=10000)
    parser.add_argument('--warmstart_mode', type=str, default='bc')
    # arguments for get_algo
    parser.add_argument('-a', '--algo_name', type=str, default='PPO')
    parser.add_argument('--value_lr', type=float, default=5e-3)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('-n', '--n_workers', type=int, default=4)
    parser.add_argument('--use_gpu', type=str2bool, default=False)
    parser.add_argument('--sampler_mode', type=str, default='ray')

    args = parser.parse_args()

    # Run experiment.
    args_dict = vars(args)
    simple_run_exp(**args_dict)
