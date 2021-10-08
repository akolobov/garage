
import gym, d4rl, torch, os
import numpy as np
from urllib.error import HTTPError
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer
from garage.torch.algos import SAC
from garage.tools.algos import CQL, CAC, CAC0
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.tools.rl_utils import train_agent, get_sampler, setup_gpu, get_algo, get_log_dir_name, load_algo

from garage.tools.trainer import Trainer
from garage import StepType

def load_d4rl_data_as_buffer(dataset, replay_buffer):
    assert isinstance(replay_buffer, PathBuffer)
    # Determine whether timeout or absorbing happens
    terminals = np.zeros(dataset['terminals'].shape).reshape(-1,1)
    true_terminals = dataset['terminals']
    terminals[true_terminals] = StepType.TERMINAL

    observation=dataset['observations']
    next_observation=dataset['next_observations']
    diff = np.sum(np.abs(observation[1:]-next_observation[:-1]),axis=1)
    timeout = np.logical_not(np.isclose(diff,0.))
    timeout = np.concatenate((timeout, [not true_terminals[-1]]))
    terminals[timeout] = StepType.TIMEOUT

    # Make sure timeouts are not true terminal
    assert np.sum(timeout*true_terminals)<=0

    replay_buffer.add_path(
        dict(observation=dataset['observations'],
             action=dataset['actions'],
             reward=dataset['rewards'].reshape(-1, 1),
             next_observation=dataset['next_observations'],
             terminal=terminals,
    ))

def load_d4rl_data_as_buffer_basic(dataset, replay_buffer):
    assert isinstance(replay_buffer, PathBuffer)
    replay_buffer.add_path(
        dict(observation=dataset['observations'],
             action=dataset['actions'],
             reward=dataset['rewards'].reshape(-1, 1),
             next_observation=dataset['next_observations'],
             terminal=dataset['terminals'].reshape(-1,1),
    ))

def train_func(ctxt=None,
               *,
               algo,
               # Environment parameters
               env_name,
               # Evaluation mode
               evaluation_mode=False,
               policy_path=None,
               # Trainer parameters
               n_epochs=3000,  # number of training epochs
               batch_size=0,  # number of samples collected per update
               replay_buffer_size=int(2e6),
               normalize_reward=True,  # normalize the reawrd to be in [-10, 10]
               # Network parameters
               policy_hidden_sizes=(256, 256, 256),
               policy_hidden_nonlinearity=torch.nn.ReLU,
               policy_init_std=1.0,
               value_hidden_sizes=(256, 256, 256),
               value_hidden_nonlinearity=torch.nn.ReLU,
               min_std=1e-5,
               # Algorithm parameters
               discount=0.99,
               policy_lr=5e-5,  # optimization stepsize for policy update
               value_lr=5e-4,  # optimization stepsize for value regression
               target_update_tau=5e-3, # for target network
               minibatch_size=256,  # optimization/replaybuffer minibatch size
               n_grad_steps=1000,  # number of gradient updates per epoch
               steps_per_epoch=1,  # number of internal epochs steps per epoch
               n_bc_steps=20000,
               fixed_alpha=None,
               use_two_qfs=True,
               use_deterministic_evaluation=True,
               num_evaluation_episodes=5, # number of episodes to evaluate (only affect off-policy algorithms)
               # CQL parameters
               lagrange_thresh=5.0,
               min_q_weight=1.0,
               # CAC parameters
               version=0,
               kl_constraint=0.05,
               alpha_lr=None,  # stepsize for controlling the entropy
               bc_policy_lr=None, # stepsize of bc
               policy_lr_decay_rate=0, # decay rate of policy_lr in CAC
               policy_update_tau=None, # for the policy.
               penalize_time_out=False,
               decorrelate_actions=False,
               terminal_value=0,
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
    dataset = None
    while dataset is None:
        try:
            _env = gym.make(env_name)  # d4rl env
            dataset = d4rl.qlearning_dataset(_env)
        except (HTTPError, OSError):
            pass

    # Initialize replay buffer and gymenv
    env = GymEnv(_env)
    replay_buffer = PathBuffer(capacity_in_transitions=int(replay_buffer_size))

    if algo=='CAC':
        load_d4rl_data_as_buffer(dataset, replay_buffer)
    else:
        load_d4rl_data_as_buffer_basic(dataset, replay_buffer)

    # Normalize the rewards to be in [-10, 10]
    if normalize_reward:
        r_max = np.abs(np.max(dataset['rewards']))
        r_min = np.abs(np.min(dataset['rewards']))
        reward_scale = 10./(max(r_min, r_max) + 1e-6)
    else:
        reward_scale = 1.0

    # Initialize the algorithm
    env_spec = env.spec

    policy = TanhGaussianMLPPolicy(
                env_spec=env_spec,
                hidden_sizes=policy_hidden_sizes,
                hidden_nonlinearity=policy_hidden_nonlinearity,
                init_std=policy_init_std,
                min_std=min_std)

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

    """ Overwrite the parameters for setting up the policy evaluation mode. """
    if evaluation_mode:
        assert policy_path is not None
        policy_path, itr = policy_path.split(':')
        policy = load_algo(policy_path, itr=itr).policy
        policy_lr = 0
        n_epochs = int(n_bc_steps/max(1,n_grad_steps))

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
                fixed_alpha=fixed_alpha,
                reward_scale=reward_scale,
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
            version=version,
            kl_constraint=kl_constraint,
            policy_update_tau=policy_update_tau,
            use_two_qfs=use_two_qfs,
            penalize_time_out=penalize_time_out,
            alpha_lr=alpha_lr,
            bc_policy_lr=bc_policy_lr,
            policy_lr_decay_rate=policy_lr_decay_rate,
            decorrelate_actions=decorrelate_actions,
            terminal_value=terminal_value
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
        snapshot_frequency=0,
        **train_kwargs):
    torch.set_num_threads(torch_n_threads)
    if train_kwargs['algo']=='CQL':
        log_dir = get_log_dir_name(train_kwargs, ['policy_lr', 'value_lr', 'lagrange_thresh', 'min_q_weight', 'seed'])
    if train_kwargs['algo']=='CAC':
        log_dir = get_log_dir_name(train_kwargs, ['version', 'min_q_weight', 'discount',
                                                  'policy_lr', 'value_lr', 'target_update_tau',
                                                   #'terminal_value', 'penalize_time_out',
                                                   # 'alpha_lr', 'bc_policy_lr',
                                                   # 'policy_update_tau', 'fixed_alpha', 'kl_constraint',
                                                  'use_two_qfs', 'n_bc_steps', 'seed'])
    train_kwargs['return_mode'] = 'full'

    # Offline training
    log_dir_path = os.path.join(log_root,'testdata','Offline'+train_kwargs['algo']+'_'+train_kwargs['env_name'], log_dir)
    full_score =  train_agent(train_func,
                    log_dir=log_dir_path,
                    train_kwargs=train_kwargs,
                    snapshot_frequency=snapshot_frequency,
                    x_axis='Epoch')

    # Extra policy evaluation
    if snapshot_frequency>0:
        eval_kwargs = train_kwargs.copy()
        eval_kwargs['evaluation_mode'] = True
        n_trainin_epochs = len(full_score)
        min_qf_losses = []
        policy_returns = []
        for n in range(0, n_trainin_epochs, snapshot_frequency):
            eval_kwargs['policy_path'] = log_dir_path+':'+str(n)
            log_dir_path_eval = os.path.join(log_dir_path, 'policy_'+str(n))
            full_score_eval =  train_agent(train_func,
                            log_dir=log_dir_path_eval,
                            train_kwargs=eval_kwargs,
                            snapshot_frequency=0,  # don't need extra logging
                            x_axis='Epoch')

            from garage.tools.utils import read_attr_from_csv
            min_qf1_loss = read_attr_from_csv(os.path.join(log_dir_path_eval,'progress.csv'), 'Algorithm/min_qf1_loss')
            min_qf2_loss = read_attr_from_csv(os.path.join(log_dir_path_eval,'progress.csv'), 'Algorithm/min_qf2_loss')
            min_qf_losses.append(min(min_qf1_loss[-1], min_qf2_loss[-1]))
            policy_returns.append(np.mean(full_score_eval))

        score = policy_returns[np.argmax(min_qf_losses)]
        best_score = max(policy_returns)
        print('Estimated best score', score, '\n', 'True best score', best_score)
        return {'score': score,
                'best_score': best_score}

    else:
        window = 50
        score = np.median(full_score[-min(len(full_score),window):])
        print('Median of performance of last {} epochs'.format(window), score)
        return {'score': score,  # last 50 epochs
                'mean': np.mean(full_score)}

if __name__=='__main__':
    import argparse
    from garage.tools.utils import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='CQL')
    parser.add_argument('-e', '---env_name',  type=str, default='hopper-medium-v0')
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gpu_id', type=int, default=-1)  # use cpu by default
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--force_cpu_data_collection', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lagrange_thresh', type=float, default=5.0)
    parser.add_argument('--n_bc_steps', type=int, default=20000)  # 40000
    parser.add_argument('--fixed_alpha', type=float, default=None)
    parser.add_argument('--min_q_weight', type=float, default=1.0)
    parser.add_argument('--policy_lr', type=float, default=5e-5)
    parser.add_argument('--value_lr', type=float, default=5e-4)
    parser.add_argument('--alpha_lr', type=float, default=None)
    parser.add_argument('--bc_policy_lr', type=float, default=None)
    parser.add_argument('--policy_lr_decay_rate', type=float, default=0.)
    parser.add_argument('--target_update_tau', type=float, default=5e-3)
    parser.add_argument('--policy_update_tau', type=float, default=None)
    parser.add_argument('--use_deterministic_evaluation', type=str2bool, default=True)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--kl_constraint', type=float, default=0.05)
    parser.add_argument('--use_two_qfs', type=str2bool, default=True)
    parser.add_argument('--penalize_time_out', type=str2bool, default=False)
    parser.add_argument('--decorrelate_actions', type=str2bool, default=False)
    parser.add_argument('--terminal_value', type=float, default=0)

    train_kwargs = vars(parser.parse_args())
    run(**train_kwargs)