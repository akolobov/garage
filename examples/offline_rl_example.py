
import gym, d4rl, torch, os
import numpy as np
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer
from garage.torch.algos import SAC
from garage.tools.algos import CQL, CAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.tools.rl_utils import train_agent, get_sampler, setup_gpu, get_algo, get_log_dir_name

from garage.tools.trainer import Trainer

def load_d4rl_data_as_buffer(dataset, replay_buffer):
    assert isinstance(replay_buffer, PathBuffer)
    replay_buffer.add_path(
        dict(observation=dataset['observations'],
            action=dataset['actions'],
            reward=dataset['rewards'].reshape(-1, 1),
            next_observation=dataset['next_observations'],
            terminal=dataset['terminals'].reshape(-1, 1))
    )


def train_func(ctxt=None,
               *,
               algo,
               # Environment parameters
               env_name,
               # Trainer parameters
               n_epochs=3000,  # number of training epochs
               batch_size=0,  # number of samples collected per update
               replay_buffer_size=int(2e6),
               # Network parameters
               policy_hidden_sizes=(256, 256, 256),
               policy_hidden_nonlinearity=torch.nn.ReLU,
               policy_init_std=1.0,
               value_hidden_sizes=(256, 256, 256),
               value_hidden_nonlinearity=torch.nn.ReLU,
               # Algorithm parameters
               discount=0.99,
               policy_lr=1e-4,  # optimization stepsize for policy update
               value_lr=3e-4,  # optimization stepsize for value regression
               target_update_tau=5e-3, # for target network
               minibatch_size=256,  # optimization/replaybuffer minibatch size
               n_grad_steps=1000,  # number of gradient updates per epoch
               steps_per_epoch=1,  # number of internal epochs steps per epoch
               n_bc_steps=10000,
               fixed_alpha=None,
               use_two_qfs=True,
               use_deterministic_evaluation=True,
               num_evaluation_episodes=10, # number of episodes to evaluate (only affect off-policy algorithms)
               # CQL parameters
               lagrange_thresh=5.0,
               min_q_weight=1.0,
               # CAC parameters
               policy_update_version=1,
               kl_constraint=0.1,
               policy_update_tau=5e-3, # for the policy.
               # Compute parameters
               seed=0,
               n_workers=1,  # number of workers for data collection
               gpu_id=-1,  # try to use gpu, if implemented
               force_cpu_data_collection=True,  # use cpu for data collection.
               # Logging parameters
               save_mode='light',
               ignore_shutdown=False,  # do not shutdown workers after training
               return_mode='average', # 'full', 'average', 'last'
               return_attr='Evaluation/AverageReturn',  # the log attribute
               ):

    """ Train an agent in batch mode. """

    # Set the random seed
    set_seed(seed)

    # Initialize gym env
    _env = gym.make(env_name)
    env = GymEnv(_env)

    # Initialize replay buffer
    dataset = d4rl.qlearning_dataset(_env)
    replay_buffer = PathBuffer(capacity_in_transitions=int(replay_buffer_size))
    load_d4rl_data_as_buffer(dataset, replay_buffer)

    # Initialize the algorithm
    env_spec = env.spec

    policy = TanhGaussianMLPPolicy(
                env_spec=env_spec,
                hidden_sizes=policy_hidden_sizes,
                hidden_nonlinearity=policy_hidden_nonlinearity,
                init_std=policy_init_std)

    qf1 = ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=value_hidden_sizes,
                hidden_nonlinearity=value_hidden_nonlinearity,
                output_nonlinearity=None)

    qf2 = ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=value_hidden_sizes,
                hidden_nonlinearity=value_hidden_nonlinearity,
                output_nonlinearity=None)

    sampler = get_sampler(policy, env, n_workers=n_workers)


    Algo = globals()[algo]

    algo_config = dict(  # union of all algorithm configs
                env_spec=env_spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                sampler=sampler,
                replay_buffer=replay_buffer,
                discount=discount,
                policy_lr=policy_lr,
                qf_lr=value_lr,
                target_update_tau=target_update_tau,
                buffer_batch_size=minibatch_size,
                gradient_steps_per_itr=n_grad_steps,
                steps_per_epoch=steps_per_epoch,
                use_deterministic_evaluation=use_deterministic_evaluation,
                min_buffer_size=int(0),
                num_evaluation_episodes=num_evaluation_episodes,
                n_bc_steps=n_bc_steps,
                fixed_alpha=fixed_alpha
    )
    extra_algo_config = dict()
    if algo=='CQL':
        extra_algo_config = dict(
            lagrange_thresh=lagrange_thresh,
            min_q_weight=min_q_weight,
        )
    elif algo=='CAC':
        extra_algo_config = dict(
            min_q_weight=min_q_weight,
            policy_update_version=policy_update_version,
            kl_constraint=kl_constraint,
            policy_update_tau=policy_update_tau,
            use_two_qfs=use_two_qfs,
        )
    algo_config.update(extra_algo_config)


    algo = Algo(**algo_config)

    setup_gpu(algo, gpu_id=gpu_id)

    # Initialize the trainer
    from garage.tools.trainer import BatchTrainer as Trainer
    trainer = Trainer(ctxt)
    trainer.setup(algo=algo,
                  env=env,
                  force_cpu_data_collection=force_cpu_data_collection,
                  save_mode=save_mode,
                  return_mode=return_mode,
                  return_attr=return_attr)

    return trainer.train(n_epochs=n_epochs,
                         batch_size=batch_size,
                         ignore_shutdown=ignore_shutdown)


def run(log_root='.',
        torch_n_threads=2,
        **train_kwargs):
    torch.set_num_threads(torch_n_threads)
    if train_kwargs['algo']=='CQL':
        log_dir = get_log_dir_name(train_kwargs, ['policy_lr', 'value_lr', 'lagrange_thresh', 'seed'])
    if train_kwargs['algo']=='CAC':
        log_dir = get_log_dir_name(train_kwargs, ['policy_update_version', 'policy_lr', 'value_lr', 'target_update_tau', 'policy_update_tau',
                                                  'use_two_qfs', 'kl_constraint', 'fixed_alpha', 'seed'])

    train_kwargs['return_mode'] = 'full'
    full_score =  train_agent(train_func,
                    log_dir=os.path.join(log_root,'testdata','Offline'+train_kwargs['algo']+'_'+train_kwargs['env_name'], log_dir),
                    train_kwargs=train_kwargs,
                    x_axis='Epoch')
    return {'score': np.median(full_score[-min(len(full_score),50):]),  # last 50 epochs
            'mean': np.mean(full_score)}

if __name__=='__main__':
    import argparse
    from garage.tools.utils import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='CQL')
    parser.add_argument('-e', '---env_name',  type=str, default='hopper-medium-v0')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gpu_id', type=int, default=-1)  # use cpu by default
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--force_cpu_data_collection', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lagrange_thresh', type=float, default=5.0)
    parser.add_argument('--n_bc_steps', type=int, default=10000)  # 40000
    parser.add_argument('--fixed_alpha', type=float, default=None)
    parser.add_argument('--min_q_weight', type=float, default=1.0)
    parser.add_argument('--policy_lr', type=float, default=1e-4)
    parser.add_argument('--value_lr', type=float, default=3e-4)
    parser.add_argument('--target_update_tau', type=float, default=5e-3)
    parser.add_argument('--policy_update_tau', type=float, default=5e-3)
    parser.add_argument('--use_deterministic_evaluation', type=str2bool, default=True)
    parser.add_argument('--policy_update_version', type=int, default=1)
    parser.add_argument('--kl_constraint', type=float, default=0.1)
    parser.add_argument('--use_two_qfs', type=str2bool, default=True)

    train_kwargs = vars(parser.parse_args())
    run(**train_kwargs)