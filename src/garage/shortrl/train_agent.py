import torch
import gym

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.shortrl.trainer import Trainer
from garage.shortrl.env_wrapper import ShortMDP
from garage.shortrl.algorithms import get_algo
from garage.shortrl import lambda_schedulers
from garage.shortrl.heuristics import load_policy_and_heuristic_from_snapshot
from garage.shortrl.pretrain import init_policy_from_baseline

def train_agent(ctxt=None,
                env_name='InvertedDoublePendulum-v2', # gym env identifier
                discount=0.99,  # oirginal discount
                heuristic=None,  # a python function
                lambd=1.0,  # extra discount
                ls_n_epochs=None, # n_epoch for lambd to converge to 1.0 (default: n_epoch)
                ls_cls='TanhLS', # class of LambdaScheduler
                seed=1,  # random seed
                n_epochs=50,  # number of updates
                batch_size=10000,  # number of samples collected per update
                ignore_shutdown=False,  # do not shutdown workers after training
                baseline_policy=None,  # when provided, it will be use to initialize the learner policy
                **kwargs,
                ):

    assert discount is not None
    if heuristic is None:
        heuristic = lambda x : 0.

    set_seed(seed)
    # Wrap the gym env into our *gym* wrapper first and then into the standard garage wrapper.
    env = ShortMDP(gym.make(env_name), heuristic, lambd=lambd, gamma=discount)
    env = GymEnv(env)
    algo_ = get_algo(env=env,
                     discount=discount*lambd,  #  algorithm sees a shorter horizon,
                     n_epochs=n_epochs,
                     batch_size=batch_size,
                     **kwargs)

    # Warm start by behavior cloning
    if baseline_policy is not None:
        init_policy = init_policy_from_baseline(
                        algo_.policy,
                        baseline_policy=baseline_policy,
                        # bc parameters
                        use_bc=True,
                        env=env,
                        n_epochs=1,
                        policy_lr=1e-3,
                        batch_size=batch_size,
                        opt_n_grad_steps=batch_size,
                        n_workers=kwargs.get('n_workers',4),
                        ctxt=ctxt)

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
    trainer.setup(algo, env, lambd=ls, discount=discount)
    return trainer.train(n_epochs=n_epochs,
                         batch_size=batch_size,
                         ignore_shutdown=ignore_shutdown)

# Run this.
def run_exp(*,
            exp_name,
            snapshot_frequency=0,
            log_prefix='agents',
            seed=1,
            **kwargs):
    snapshot_gap = snapshot_frequency if snapshot_frequency>0 else 1
    snapshot_mode = 'gap_and_last' if snapshot_frequency>0 else 'last'
    wrapped_train_agent = wrap_experiment(train_agent,
                            prefix='experiment/shortrl/'+log_prefix+'/'+exp_name,
                            snapshot_mode=snapshot_mode,
                            snapshot_gap=snapshot_gap,
                            name=str(seed),
                            use_existing_dir=True)  # overwrites existing directory
    return wrapped_train_agent(seed=seed, **kwargs)


if __name__ == '__main__':
    # Parse command line inputs.
    import argparse
    from garage.shortrl.utils import str2bool
    parser = argparse.ArgumentParser()

    # arguments for __main__
    parser.add_argument('-u', '--use_heuristic', type=str2bool, default=False)
    parser.add_argument('--init_policy', type=str2bool, default=True)
    # arguments for run_exp
    parser.add_argument('--snapshot_frequency', type=int, default=0)
    parser.add_argument('--log_prefix', type=str, default='agents')
    # arguments for train_agent
    parser.add_argument('-e', '--env_name', type=str, default='InvertedDoublePendulum-v2')
    parser.add_argument('-d', '--discount', type=float, default=0.99)
    parser.add_argument('-l', '--lambd', type=float, default=1.0)
    parser.add_argument('--ls_n_epochs', type=int, default=None)
    parser.add_argument('--ls_cls', type=str, default='TanhLS')
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

    if args.use_heuristic:
        data_path = 'data/local/experiment/shortrl/heuristics/PPO_Inver_1.0_F/1/'
        data_itr = 30
        baseline_policy, heuristic = load_policy_and_heuristic_from_snapshot(data_path, data_itr)
        if not args.init_policy:
            baseline_policy = None
    else:
        baseline_policy=heuristic=None

    exp_name = args.algo_name+'_'+args.env_name[:min(len(args.env_name),5)]+\
                '_{}_{}'.format(args.lambd, str(args.use_heuristic)[0])

    # Delete keywords not used by run_exp
    del args_dict['use_heuristic']
    del args_dict['init_policy']

    run_exp(exp_name=exp_name,
            heuristic=heuristic,
            baseline_policy=baseline_policy,
            **args_dict,
            )
