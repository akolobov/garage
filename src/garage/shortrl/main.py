import torch
import gym
import os
import numpy as np
import pickle

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.shortrl.trainer import Trainer
from garage.shortrl.algos import get_algo
from garage.shortrl import lambda_schedulers
from garage.shortrl.heuristics import load_policy_from_snapshot, load_heuristic_from_snapshot


from garage._dtypes import EpisodeBatch
import garage.shortrl.rl_utils as ru



def load_env(env_name, init_with_defaults=True):
    env_name_parts = env_name.split(':')

    if env_name_parts[0].lower() == 'procgen':
        env = gym.make("procgen:procgen-" +  env_name_parts[1] + "-v0",
                             num_levels=0, # this means we should use all levels higher than start_level
                             start_level=0,
                             paint_vel_info=False,
                             use_generated_assets=False,
                             debug_mode=0,
                             center_agent=True,
                             use_sequential_levels=False,  # When you reach the end of a level, the episode ends
                                                           # and a new level is selected. If use_sequential_levels
                                                           # is set to True, reaching the end of a level does not
                                                           # end the episode.
                             distribution_mode='hard'
                            )

        is_image = True
    elif env_name_parts[0].lower() == 'atari':
        env = gym.make(env_name_parts[1])
        env = Noop(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = ClipReward(env)
        env = StackFrames(env, 4, axis=0)
        is_image = True
    else:
        env = gym.make(env_name)
        is_image = False

    # HACK Create a sparse reward, long horizon reacher env
    assert env_name in ['Reacher-v2', 'Walker2d-v2']
    from gym.envs.mujoco.reacher import ReacherEnv
    class SparseReacherEnv(ReacherEnv):
        def step(self, a):
            thre = 0.01
            vec = self.get_body_com("fingertip") - self.get_body_com("target")
            reward_dist = -np.linalg.norm(vec)
            reward_ctrl = -np.square(a).sum()
            reward_sparse = -float(reward_dist<-thre)
            reward = reward_sparse # + reward_ctrl
            # reward = (1+reward_dist)*10 + reward_ctrl
            self.do_simulation(a, self.frame_skip)
            ob = self._get_obs()
            done = False
            # print(-reward_dist,reward_sparse)
            heuristic = reward_sparse + reward_dist/thre

            return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, heuristic=heuristic)

    from gym.envs.mujoco.walker2d import Walker2dEnv
    class SparseWalker2dEnv(Walker2dEnv):
        # TODO
        def step(self, a):
            posbefore = self.sim.data.qpos[0]
            self.do_simulation(a, self.frame_skip)
            posafter, height, ang = self.sim.data.qpos[0:3]
            alive_bonus = 1.0
            reward = (posafter - posbefore) / self.dt
            reward += alive_bonus
            reward -= 1e-3 * np.square(a).sum()
            done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
            ob = self._get_obs()
            return ob, reward, done, {}

    _env = gym.make(env_name)
    if env_name=='Reacher-v2':
        env = SparseReacherEnv()
    elif env_name == 'Walker2d-v2':
        env = SparseWalker2dEnv()

    env.spec = _env.env.spec
    env.spec.max_episode_steps = 500
    from gym.wrappers.time_limit import TimeLimit
    env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)

    env = GymEnv(env, is_image=is_image)
    return env


def offline_train(ctxt=None,
                  *,
                  algo_name,  # algorithm name
                  discount,  # original discount
                  episode_batch,  # EpisodeBatch
                  batch_size,  # number of samples collected per update
                  n_epochs,  # number of training epochs
                  init_policy_fun=None,
                  seed=1,  # random seed
                  save_mode='light',
                  ignore_shutdown=False,  # do not shutdown workers after training
                  return_mode='average', # 'full', 'average', 'last'
                  return_attr='Evaluation/AverageReturn',  # the log attribute
                  **kwargs  # other kwargs for get_algo
                  ):
    """ Train an agent in batch mode. """

    # Set the random seed
    set_seed(seed)

    # Make sure a large enough batch is used
    if 'opt_n_grad_steps' in kwargs and 'opt_minibatch_size' in kwargs:
        batch_size = kwargs['opt_n_grad_steps']*kwargs['opt_minibatch_size']

    # Initialize the algorithm
    init_policy = None if init_policy_fun is None else init_policy_fun()
    algo = get_algo(algo_name=algo_name,
                    discount=discount,
                    episode_batch=episode_batch,  # this overwrite the env
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    init_policy=init_policy,
                    **kwargs)

    # Initialize the trainer
    if algo_name=='BC':
        return_attr = 'BC/MeanLoss'

    trainer = Trainer(ctxt)
    trainer.setup(algo=algo,
                  env=None,
                  save_mode=save_mode,
                  return_mode=return_mode,
                  return_attr=return_attr)

    return trainer.train(n_epochs=n_epochs,
                         batch_size=batch_size,
                         ignore_shutdown=ignore_shutdown)


def online_train(ctxt=None,
                 *,
                 algo_name,  # algorithm name
                 discount,  # original discount
                 env_name,  # environment name
                 batch_size,  # number of samples collected per update
                 n_epochs,  # number of training epochs
                 init_policy_fun=None,
                 seed=1,  # random seed
                 save_mode='light',
                 ignore_shutdown=False,  # do not shutdown workers after training
                 return_mode='average', # 'full', 'average', 'last'
                 return_attr='Evaluation/AverageReturn',  # the log attribute
                 # short-horizon RL params
                 heuristic=None,  # a python function
                 lambd=1.0,  # extra discount
                 ls_rate=1, # n_epoch for lambd to converge to 1.0 (default: n_epoch)
                 ls_cls='TanhLS', # class of LambdaScheduler
                 **kwargs,  # other kwargs for get_algo
                 ):
    """ Train an agent online using short-horizon RL. """

    # Set the random seed
    set_seed(seed)

    # Wrap the gym env into our *gym* wrapper first and then into the standard garage wrapper.
    env = load_env(env_name, init_with_defaults=False)

    # Initialize the algorithm
    init_policy = None if init_policy_fun is None else init_policy_fun()
    lambd = getattr(lambda_schedulers, ls_cls)(init_lambd=lambd, n_epochs=ls_rate*n_epochs)
    algo = get_algo(algo_name=algo_name,
                    discount=discount,  #  algorithm sees a shorter horizon,
                    lambd=lambd,
                    heuristic=heuristic,
                    env=env,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    init_policy=init_policy,
                    **kwargs)

    # Initialize the trainer
    trainer = Trainer(ctxt)
    trainer.setup(algo=algo,
                  env=env,
                  save_mode=save_mode,
                  return_mode=return_mode,
                  return_attr=return_attr)

    return trainer.train(n_epochs=n_epochs,
                         batch_size=batch_size,
                         ignore_shutdown=ignore_shutdown)


def load_episode_batch(data_path, data_itr):
    filepath = os.path.join(data_path,'itr_'+str(data_itr)+'_batch.pkl')
    episode_batch = pickle.load(open(filepath, "rb"))
    return episode_batch

def get_snapshot_info(snapshot_frequency):
    snapshot_gap = snapshot_frequency if snapshot_frequency>0 else 1
    snapshot_mode = 'gap_and_last' if snapshot_frequency>0 else 'last'
    return snapshot_gap, snapshot_mode

def compute_episode_batch_returns(episode_batch):
    undiscounted_returns = []
    for eps in episode_batch.split():
        if 'orig_reward' in eps.env_infos:
            rewards = eps.env_infos['orig_reward']
        else:
            rewards = eps.rewards
        undiscounted_returns.append(sum(rewards))
    return np.mean(undiscounted_returns)

def parse_data_itr(data_itr):
    assert len(data_itr) in [2,3]
    if len(data_itr)==2:
        data_itr_st, data_itr_ed = data_itr
        data_itr_sp = 1
        data_itr_str = str(data_itr_st)+'_'+str(data_itr_ed)
    else:
        data_itr_st, data_itr_ed, data_itr_sp = data_itr
        data_itr_str = str(data_itr_st)+'_'+str(data_itr_ed)+'_'+str(data_itr_sp)


    return data_itr_st, data_itr_ed, data_itr_sp, data_itr_str


def train_heuristics(
                     data_path,
                     data_itr,
                     *,
                     algo_name,
                     discount,
                     n_epochs,
                     batch_size,
                     use_raw_snapshot=False,
                     snapshot_frequency=0,
                     save_mode='light',
                     vae_loss_percentile=0,
                     ignore_shutdown=True,
                     **kwargs
                     ):

    if use_raw_snapshot:
        heuristic = load_heuristic_from_snapshot(data_path, data_itr, vae_loss_percentile)
        return heuristic

    train_from_mixed_data = isinstance(data_itr, list) or isinstance(data_itr, tuple)
    episode_batch = None
    if train_from_mixed_data:
        # Train from a range of snapshots
        data_itr_st, data_itr_ed, data_itr_sp, data_itr_str = parse_data_itr(data_itr)
        log_dir = os.path.join(data_path,'itr_'+data_itr_str+'_heuristic_'+algo_name)
        snapshot_path = os.path.join(log_dir,'params.pkl')
        if not os.path.exists(snapshot_path):
            print("No saved heuristic snapshot found. Train one from batch data.")
            episode_batches = load_episode_batch(data_path, data_itr_str)
            episode_batches.sort(key=compute_episode_batch_returns)  # from low to high returns
            episode_batch = EpisodeBatch.concatenate(*episode_batches)
    else:
        # Train from a single of snapshots
        log_dir = os.path.join(data_path,'itr_'+str(data_itr)+'_heuristic_'+algo_name)
        snapshot_path = os.path.join(log_dir,'params.pkl')
        if not os.path.exists(snapshot_path):
            print("No saved heuristic snapshot found. Train one from batch data.")
            episode_batch = load_episode_batch(data_path, data_itr)

    if episode_batch is not None:
        snapshot_gap, snapshot_mode = get_snapshot_info(snapshot_frequency)
        exp = wrap_experiment(offline_train,
                            log_dir=log_dir,  # overwrite
                            snapshot_mode=snapshot_mode,
                            snapshot_gap=snapshot_gap,
                            archive_launch_repo=save_mode!='light',
                            use_existing_dir=True)  # overwrites existing directory
        exp(algo_name=algo_name,
            discount=discount,
            n_epochs=n_epochs,
            episode_batch=episode_batch,
            batch_size=batch_size,
            ignore_shutdown=True,
            randomize_episode_batch=True, #not train_from_mixed_data,
            **kwargs)


    print("Load heuristic snapshot.")
    heuristic = load_heuristic_from_snapshot(log_dir, 'last', vae_loss_percentile)
    assert heuristic is not None
    return heuristic


def pretrain_policy(data_path,
                    data_itr,
                    target_algo_name,
                    *,
                    algo_name,
                    discount,
                    n_epochs,
                    batch_size,
                    snapshot_frequency=0,
                    save_mode='light',
                    ignore_shutdown=True,
                    **kwargs
                    ):

    train_from_mixed_data = isinstance(data_itr, list) or isinstance(data_itr, tuple)
    episode_batch = None
    if train_from_mixed_data:
        # Train from a range of snapshots
        data_itr_st, data_itr_ed, data_itr_sp, data_itr_str = parse_data_itr(data_itr)
        log_dir = os.path.join(data_path,'itr_'+data_itr_str+'_init_policy_'+algo_name)
        snapshot_path = os.path.join(log_dir,'params.pkl')
        if not os.path.exists(snapshot_path):
            print("No saved init_policy snapshot found. Train one from batch data.")
            episode_batches = load_episode_batch(data_path, data_itr_str)
            episode_batches.sort(key=compute_episode_batch_returns)  # from low to high returns
            episode_batch = EpisodeBatch.concatenate(*episode_batches)
            # expert_policy = load_policy_from_snapshot(data_path, data_itr_ed) if algo_name=='BC' else None

    else:
        # Train from a single of snapshots
        log_dir = os.path.join(data_path,'itr_'+str(data_itr)+'_init_policy_'+target_algo_name)
        snapshot_path = os.path.join(log_dir, 'params.pkl')
        if not os.path.exists(snapshot_path):
            print("No saved init_policy snapshot found. Train one from batch data.")
            episode_batch = load_episode_batch(data_path, data_itr)
            # expert_policy = load_policy_from_snapshot(data_path, data_itr) if algo_name=='BC' else None


    if episode_batch is not None:
        snapshot_gap, snapshot_mode = get_snapshot_info(snapshot_frequency)
        exp = wrap_experiment(offline_train,
                                log_dir=log_dir,  # overwrite
                                snapshot_mode=snapshot_mode,
                                snapshot_gap=snapshot_gap,
                                archive_launch_repo=save_mode!='light',
                                use_existing_dir=True)  # overwrites existing directory
        def init_policy_fun():
            algo = get_algo(algo_name=target_algo_name,
                            discount=discount,  #  algorithm sees a shorter horizon,
                            episode_batch=episode_batch,
                            batch_size=1,
                            **kwargs)
            return algo.policy

        n_epochs = len(episode_batches) if train_from_mixed_data else n_epochs
        exp(algo_name=algo_name,
            discount=discount,
            n_epochs=n_epochs,
            episode_batch=episode_batch,
            batch_size=batch_size,
            # expert_policy=expert_policy,
            init_policy_fun=init_policy_fun,
            ignore_shutdown=True,
            randomize_episode_batch=True, #not train_from_mixed_data, # to avoid conflicts
            **kwargs)

    print("Load init_policy snapshot.")
    policy = load_policy_from_snapshot(log_dir, 'last')
    assert policy is not None
    return policy


def train_agent(*,
                algo_name,  # algorithm name
                discount,  # original discount
                n_epochs,  # number of learning epochs
                env_name,  # environment name
                batch_size,  # number of samples collected per update
                log_dir,  # path to the log
                heuristic=None,  # a python function
                snapshot_frequency=0,
                save_mode='light',
                **kwargs
                ):


    assert algo_name in ['SAC']  # only this is implemented now

    snapshot_gap, snapshot_mode = get_snapshot_info(snapshot_frequency)
    exp = wrap_experiment(online_train,
                          log_dir=log_dir,  # overwrite
                          snapshot_mode=snapshot_mode,
                          snapshot_gap=snapshot_gap,
                          archive_launch_repo=save_mode!='light',
                          use_existing_dir=True)  # overwrites existing directory

    score = exp(algo_name=algo_name,  # algorithm name
                discount=discount,  # original discount
                n_epochs=n_epochs,  # number of learning epochs
                env_name=env_name,  # environment name
                batch_size=batch_size,  # number of samples collected per update
                heuristic=heuristic,
                save_mode=save_mode,
                **kwargs)
    return score


def collect_batch_data(data_path,
                       data_itr,
                       *,
                       env,
                       episode_batch_size,
                       seed=1,
                       n_workers=4,
                       sampler_mode='ray',
                       log_path=None):
    # Collect episode_batch and save it
    train_from_mixed_data = isinstance(data_itr, list) or isinstance(data_itr, tuple)
    if train_from_mixed_data:
        data_itr_st, data_itr_ed, data_itr_sp, data_itr_str = parse_data_itr(data_itr)
        set_seed(seed)
        episode_batch = []
        for itr in range(data_itr_st, data_itr_ed, data_itr_sp):
            expert_policy = load_policy_from_snapshot(data_path, itr)
            eps = ru.collect_episode_batch(
                                policy=expert_policy,
                                env=env,
                                batch_size=episode_batch_size,
                                n_workers=n_workers,
                                sampler_mode=sampler_mode)
            episode_batch.append(eps)

    else:
        expert_policy = load_policy_from_snapshot(data_path, data_itr)
        set_seed(seed)
        episode_batch = ru.collect_episode_batch(
                            policy=expert_policy,
                            env=env,
                            batch_size=episode_batch_size,
                            n_workers=n_workers,
                            sampler_mode=sampler_mode)


    if train_from_mixed_data:
        data_itr_st, data_itr_ed, data_itr_sp, data_itr_str = parse_data_itr(data_itr)
    else:
        data_itr_str = str(data_itr)

    if log_path is None:
        filepath = os.path.join(data_path,'itr_'+data_itr_str+'_batch.pkl')
    else: # Log the batch data to a user-specified path
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        filepath = os.path.join(log_path,'itr_'+data_itr_str+'_batch.pkl')
    pickle.dump(episode_batch, open(filepath, "wb"))
    print('Saved batch data.')


# Run this.
def run_exp(*,
            algo_name,
            discount=None,
            n_epochs=None, # either n_epochs or total_n_samples needs to be provided
            total_n_samples=None,
            env_name,
            batch_size,
            seed=1,
            # offline batch data
            data_path=None,  # directory of the snapshot
            data_itr=None,
            episode_batch_size=50000,
            # pretrain policy
            warmstart_policy=False,
            w_algo_name='BC',
            w_n_epoch=30,
            # short-horizon RL params
            lambd=1.0,
            use_raw_snapshot=False,
            use_heuristic=False,
            h_algo_name='VPG',
            h_n_epoch=30,
            # logging
            load_pretrained_data=True,
            snapshot_frequency=0,  # 0 means only taking the last snapshot
            log_root=None,
            log_prefix='agents',
            save_mode='light',
            **kwargs  # kwargs for get_algo
            ):

    if use_heuristic or warmstart_policy:
        assert data_itr is not None and data_path is not None
        if not load_pretrained_data:
            import datetime
            batch_log_path = '_tmp_srl_data_'+datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        else:
            batch_log_path = data_path


    #env = GymEnv(ShortMDP(gym.make(env_name)))
    env = load_env(env_name, init_with_defaults=True)

    if discount is None:
        discount = 1 -1/env.spec.max_episode_length

    # Load or collect batch data
    successful_init = False
    heuristic = init_policy = None
    while not successful_init:
        # Load heuristic and init_policy
        try:
            if use_heuristic:
                # HACK
                if h_algo_name=='HACK':
                    heuristic = 'HACK'
                else:
                    heuristic = train_heuristics(
                                    batch_log_path,
                                    data_itr,
                                    algo_name=h_algo_name,
                                    discount=discount,
                                    n_epochs=h_n_epoch,
                                    batch_size=batch_size,
                                    seed=seed,
                                    use_raw_snapshot=use_raw_snapshot,
                                    **kwargs
                                    )
            if warmstart_policy:
                init_policy = pretrain_policy(
                                batch_log_path,
                                data_itr,
                                target_algo_name=algo_name,
                                algo_name=w_algo_name,
                                discount=discount,
                                n_epochs=w_n_epoch,
                                batch_size=batch_size,
                                seed=seed,
                                **kwargs
                                )
            successful_init = True

        except FileNotFoundError:
            print("No batch data found. Collect new data.")
            collect_batch_data(data_path,
                               data_itr,
                               env=env,
                               episode_batch_size=episode_batch_size,
                               seed=seed,
                               log_path=batch_log_path)


    # Define log_dir based on garage's logging convention
    exp_name = algo_name+'_'+ env_name[:min(len(env_name),5)]+\
                '_{}_{}_{}'.format(lambd, str(h_algo_name), str(warmstart_policy)[0])

    if 'reward_shaping_mode' in kwargs:
        exp_name += '_'+kwargs['reward_shaping_mode']
    if 'reward_scale' in kwargs:
        exp_name += '_{}'.format(kwargs['reward_scale'])

    prefix= os.path.join('shortrl',log_prefix,exp_name)
    name=str(seed)
    log_root = log_root or '.'
    log_dir = os.path.join(log_root,'data','local','experiment',prefix, name)

    # Train agent
    assert n_epochs is not None or total_n_samples is not None
    n_epochs = n_epochs or np.ceil(total_n_samples/batch_size)
    score = train_agent(
                algo_name=algo_name,  # algorithm name
                discount=discount,  # original discount
                n_epochs=n_epochs,  # number of learning epochs
                env_name=env_name,  # environment name
                batch_size=batch_size,  # number of samples collected per update
                log_dir=log_dir,  # path the log
                heuristic=heuristic,  # a python function
                init_policy_fun=lambda : init_policy,  #  initial policy
                snapshot_frequency=snapshot_frequency,
                save_mode=save_mode,
                lambd=lambd,
                seed=seed,
                **kwargs,
                )

    if use_heuristic or warmstart_policy:
        if not load_pretrained_data:
            import subprocess
            subprocess.run(['rm', '-r', batch_log_path])

    return score


if __name__ == '__main__':
    # Parse command line inputs.
    import argparse
    from garage.shortrl.utils import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo_name', type=str, default='PPO')
    parser.add_argument('-d', '--discount', type=float, default=None)
    parser.add_argument('-N', '--n_epochs', type=int, default=50)
    parser.add_argument('--total_n_samples', type=int, default=None)
    parser.add_argument('-e', '--env_name', type=str, default='InvertedDoublePendulum-v2')
    parser.add_argument('-b', '--batch_size', type=int, default=10000)
    parser.add_argument('-s', '--seed', type=int, default=1)
    # offline batch data
    parser.add_argument('--data_path', type=str, default='snapshots/VPG_HalfC_1.0_F_F/1/')
    parser.add_argument('--data_itr', type=int, default=0)
    # parser.add_argument('--data_path', type=str, default='snapshots/SAC_HalfC_1.0_F_F/210566759/')
    # parser.add_argument('--data_itr', type=int, default=(0,9))
    parser.add_argument('--episode_batch_size', type=int, default=10000) #50000)
    # pretrain policy
    parser.add_argument('-w', '--warmstart_policy', type=str2bool, default=False)
    parser.add_argument('--w_algo_name', type=str, default='BC')
    parser.add_argument('--w_n_epoch', type=int, default=8)
    # short-horizon RL params
    parser.add_argument('-l', '--lambd', type=float, default=1.0)
    parser.add_argument('-u', '--use_heuristic', type=str2bool, default=False)
    parser.add_argument('--use_raw_snapshot', type=str2bool, default=False)
    parser.add_argument('--h_algo_name', type=str, default='VPG')
    parser.add_argument('--h_n_epoch', type=int, default=30)
    parser.add_argument('--vae_loss_percentile', type=int, default=0)
    parser.add_argument('--ls_rate', type=float, default=1)
    parser.add_argument('--ls_cls', type=str, default='TanhLS')
    # logging
    parser.add_argument('--snapshot_frequency', type=int, default=0)
    parser.add_argument('--log_root', type=str, default=None)
    parser.add_argument('--log_prefix', type=str, default='agents')
    parser.add_argument('--save_mode', type=str, default='light')
    # kwargs for get_algo
    parser.add_argument('--value_lr', type=float, default=1e-3)
    parser.add_argument('--policy_lr', type=float, default=1e-3)
    parser.add_argument('-n', '--n_workers', type=int, default=4)
    parser.add_argument('--use_gpu', type=str2bool, default=False)
    parser.add_argument('--sampler_mode', type=str, default='ray')
    parser.add_argument('--reward_shaping_mode', type=str, default='hurl')
    parser.add_argument('--reward_scale', type=float, default=1.0)


    args = parser.parse_args()

    # Run experiment.
    args_dict = vars(args)

    run_exp(**args_dict)
