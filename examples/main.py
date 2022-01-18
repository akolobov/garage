
import gym, d4rl, torch, os

import numpy as np
from urllib.error import HTTPError
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


from garage.offline_rl.algos import CAC
from garage.offline_rl.rl_utils import train_agent, get_sampler, setup_gpu, get_algo, get_log_dir_name, load_algo
from garage.offline_rl.trainer import Trainer


class PRB(PathBuffer):


    def add_path(self, path):
        path_len = self._get_path_length(path)
        path['priority'] = np.ones((path_len,1))*10
        super().add_path(path)

    def sample_transitions(self, batch_size):
        ## A naive version but too slow.
        # priority = self._buffer['priority'][:self._transitions_stored].flatten()
        # p = priority/priority.sum()
        # idx = np.random.choice(self._transitions_stored, size=batch_size, p=p)
        # samples = {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}
        # samples['_idx'] = idx
        # samples['_w'] = 1/p[idx]/len(p)

        idx0 = np.random.randint(self._transitions_stored, size=min(self._transitions_stored, batch_size*100))
        priority = self._buffer['priority'][idx0].flatten()
        p = priority/priority.sum()
        idx_ = np.random.choice(len(priority), size=batch_size, p=p)
        idx = idx0[idx_]

        samples = {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}
        samples['_idx'] = idx
        # samples['_w'] = 1/p[idx_]/len(p)
        is_weights = 1/p[idx_]/len(p)
        samples['_w'] = is_weights/is_weights.sum()*len(p)  # normalized IS
        # multiply with extra len(p) because it takes mean in the code later on
        return samples

    def update_priority(self, td_error, idx):
        # self._buffer['priority'][idx] = td_error[:,None]
        self._buffer['priority'][idx] = (td_error[:,None]+1e-3)**0.5



def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    # Add timeout and timestep keys
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    timestep_ = []
    timeout_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        timestep = episode_step

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        timestep_.append(timestep)
        timeout_.append(final_timestep)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'timesteps': np.array(timestep_),
        'timeouts': np.array(timeout_)
    }

def load_d4rl_data_as_buffer(dataset, replay_buffer):
    assert isinstance(replay_buffer, PathBuffer)
    replay_buffer.add_path(
        dict(observation=dataset['observations'],
             action=dataset['actions'],
             reward=dataset['rewards'].reshape(-1, 1),
             next_observation=dataset['next_observations'],
             terminal=dataset['terminals'].reshape(-1,1),
             timestep=dataset['timesteps'].reshape(-1,1),
             timeout=dataset['timeouts'].reshape(-1,1),
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
               policy_activation='ReLU',
               policy_init_std=1.0,
               value_hidden_sizes=(256, 256, 256),
               value_activation='ReLU',
               min_std=1e-5,
               # Algorithm parameters
               discount=0.99,
               policy_lr=5e-6,  # optimization stepsize for policy update
               value_lr=5e-4,  # optimization stepsize for value regression
               target_update_tau=5e-3, # for target network
               minibatch_size=256,  # optimization/replaybuffer minibatch size
               n_grad_steps=2000,  # number of gradient updates per epoch
               steps_per_epoch=1,  # number of internal epochs steps per epoch
               n_warmstart_steps=100000,  # number of warm-up steps
               max_n_warmstart_steps=200000,
               fixed_alpha=None,  # whether to fix the temperate parameter
               use_deterministic_evaluation=True,  # do evaluation based on the deterministic policy
               num_evaluation_episodes=5, # number of episodes to evaluate (only affect off-policy algorithms)
               # CQL parameters
               lagrange_thresh=5.0,
               min_q_weight=1.0,
               # CAC parameters
               beta=1.0,  # weight on the Bellman error
               norm_constraint=100,
               use_two_qfs=True,  # whether to use two q function
               optimizer='Adam',
               q_eval_mode='0.5_0.5',
               q_eval_loss='MSELoss',
               init_q_eval_mode=None, # XXX deprecated
               bellman_surrogate='td', # 'td', 'target', None
               lambd=0.0, # XXX deprecated
               clip_qfs=False,
               shift_reward=False,
               use_prb=False,
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
    d4rl_env = gym.make(env_name)  # d4rl env
    while dataset is None:
        try:
            dataset = qlearning_dataset(d4rl_env)
        except (HTTPError, OSError):
            print('Unable to download dataset. Retry.')
            pass

    # Initialize replay buffer and gymenv
    env = GymEnv(d4rl_env)

    if use_prb:
        replay_buffer = PRB(capacity_in_transitions=int(replay_buffer_size))
    else:
        replay_buffer = PathBuffer(capacity_in_transitions=int(replay_buffer_size))

    # Set Vmin and Vmax, if known
    Vmin = max(dataset['rewards'].min()/(1-discount), d4rl.infos.REF_MIN_SCORE[env_name]) if clip_qfs else -float('inf')
    Vmax = dataset['rewards'].max()/(1-discount)if clip_qfs else float('inf')

    if 'kitchen' in env_name:
        # remove wrongly labeled terminal states
        if shift_reward:
            good_indices = np.logical_not(dataset['terminals']*(dataset['rewards']!=4))
            for k in dataset.keys():
                dataset[k] = dataset[k][good_indices]
            dataset['rewards'] -= 4
            Vmin -= 4/(1-discount)
            Vmax = 0
        else:
            Vmin = 0
            Vmax = 4/(1-discount)

    if 'antmaze' in env_name:
        # numerically better behaved?
        if shift_reward:
            dataset['rewards'] -= 1
            Vmin -= 1/(1-discount)
            Vmax = 0
        else:
            Vmin = 0
            Vmax = 1

    print("Vmin {} Vmax {}".format(Vmin, Vmax))
    load_d4rl_data_as_buffer(dataset, replay_buffer)

    # # Normalize the rewards to be in [-1, 1]
    # if normalize_reward:
    #     r_max = np.abs(np.max(dataset['rewards']))
    #     r_min = np.abs(np.min(dataset['rewards']))
    #     reward_scale = 1./(max(r_min, r_max) + 1e-6)
    # else:
    #     reward_scale = 1.0
    reward_scale = 1.0


    # Initialize the algorithm
    env_spec = env.spec

    policy = TanhGaussianMLPPolicy(
                env_spec=env_spec,
                hidden_sizes=policy_hidden_sizes,
                hidden_nonlinearity=eval('torch.nn.'+policy_activation),
                init_std=policy_init_std,
                min_std=min_std)

    qf1 = ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=value_hidden_sizes,
                hidden_nonlinearity=eval('torch.nn.'+value_activation),
                output_nonlinearity=None)

    qf2 = ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=value_hidden_sizes,
                hidden_nonlinearity=eval('torch.nn.'+value_activation),
                output_nonlinearity=None)

    # """ Overwrite the parameters for setting up the policy evaluation mode. """
    # if evaluation_mode:
    #     assert policy_path is not None
    #     policy_path, itr = policy_path.split(':')
    #     policy = load_algo(policy_path, itr=itr).policy
    #     policy_lr = 0
    #     n_warmstart_steps = 0
    #     n_epochs = int(n_warmstart_steps/max(1,n_grad_steps))

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
                fixed_alpha=fixed_alpha,
                reward_scale=reward_scale,
    )
    extra_algo_config = dict()
    if algo=='CQL':
        extra_algo_config = dict(
            lagrange_thresh=lagrange_thresh,
            min_q_weight=min_q_weight,
            n_bc_steps=n_warmstart_steps,
        )
    elif algo=='CAC':
        extra_algo_config = dict(
            beta=beta,
            norm_constraint=norm_constraint,
            use_two_qfs=use_two_qfs,
            n_warmstart_steps=n_warmstart_steps,
            optimizer=optimizer,
            q_eval_mode=q_eval_mode,
            q_eval_loss=q_eval_loss,
            init_q_eval_mode=init_q_eval_mode,
            max_n_warmstart_steps=max_n_warmstart_steps,
            bellman_surrogate=bellman_surrogate,
            lambd=lambd,
            Vmin=Vmin,
            Vmax=Vmax,
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
        log_dir = get_log_dir_name(train_kwargs, ['beta', 'discount', 'norm_constraint',
                                                  'policy_lr', 'value_lr',
                                                  'use_two_qfs',
                                                  'fixed_alpha',
                                                  'q_eval_mode',
                                                  'init_q_eval_mode',
                                                  'clip_qfs', 'shift_reward', 'use_prb',
                                                  'n_warmstart_steps', 'seed'])
    train_kwargs['return_mode'] = 'full'

    # Offline training
    log_dir_path = os.path.join(log_root,'exp_data','Offline'+train_kwargs['algo']+'_'+train_kwargs['env_name'], log_dir)
    full_score =  train_agent(train_func,
                    log_dir=log_dir_path,
                    train_kwargs=train_kwargs,
                    snapshot_frequency=snapshot_frequency,
                    x_axis='Epoch')

    # # Extra policy evaluation
    # if snapshot_frequency>0:
    #     eval_kwargs = train_kwargs.copy()
    #     eval_kwargs['evaluation_mode'] = True
    #     n_trainin_epochs = len(full_score)
    #     min_qf_losses = []
    #     policy_returns = []
    #     for n in range(0, n_trainin_epochs, snapshot_frequency):
    #         eval_kwargs['policy_path'] = log_dir_path+':'+str(n)
    #         log_dir_path_eval = os.path.join(log_dir_path, 'policy_'+str(n))
    #         full_score_eval =  train_agent(train_func,
    #                         log_dir=log_dir_path_eval,
    #                         train_kwargs=eval_kwargs,
    #                         snapshot_frequency=0,  # don't need extra logging
    #                         x_axis='Epoch')

    #         from garage.tools.utils import read_attr_from_csv
    #         min_qf1_loss = read_attr_from_csv(os.path.join(log_dir_path_eval,'progress.csv'), 'Algorithm/min_qf1_loss')
    #         min_qf2_loss = read_attr_from_csv(os.path.join(log_dir_path_eval,'progress.csv'), 'Algorithm/min_qf2_loss')
    #         min_qf_losses.append(min(min_qf1_loss[-1], min_qf2_loss[-1]))
    #         policy_returns.append(np.mean(full_score_eval))

    #     score = policy_returns[np.argmax(min_qf_losses)]
    #     best_score = max(policy_returns)
    #     print('Estimated best score', score, '\n', 'True best score', best_score)
    #     return {'score': score,
    #             'best_score': best_score}

    # else:
    window = 50
    score = np.median(full_score[-min(len(full_score),window):])
    print('Median of performance of last {} epochs'.format(window), score)
    return {'score': score,  # last 50 epochs
            'mean': np.mean(full_score)}

if __name__=='__main__':
    import argparse
    from garage.tools.utils import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='CAC')
    parser.add_argument('-e', '---env_name',  type=str, default='hopper-medium-v2')
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gpu_id', type=int, default=-1)  # use cpu by default
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--force_cpu_data_collection', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lagrange_thresh', type=float, default=5.0)
    parser.add_argument('--n_warmstart_steps', type=int, default=100000)
    parser.add_argument('--fixed_alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--norm_constraint', type=float, default=100)
    parser.add_argument('--policy_lr', type=float, default=5e-6)
    parser.add_argument('--value_lr', type=float, default=5e-4)
    parser.add_argument('--target_update_tau', type=float, default=5e-3)
    parser.add_argument('--use_deterministic_evaluation', type=str2bool, default=True)
    parser.add_argument('--use_two_qfs', type=str2bool, default=True)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--value_activation', type=str, default='ReLU')
    parser.add_argument('--q_eval_mode', type=str, default='0.5_0.5')
    parser.add_argument('--q_eval_loss', type=str, default='MSELoss')
    parser.add_argument('--lambd', type=float, default=0.0)
    parser.add_argument('--clip_qfs', type=str2bool, default=True)
    parser.add_argument('--shift_reward', type=str2bool, default=False)
    parser.add_argument('--use_prb', type=str2bool, default=False)

    train_kwargs = vars(parser.parse_args())
    run(**train_kwargs)