import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage.envs import GymEnv
from garage.sampler import RaySampler
from garage.torch import prefer_gpu
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch.policies import GaussianMLPPolicy, TanhGaussianMLPPolicy, DeterministicMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.optimizers import OptimizerWrapper, ConjugateGradientOptimizer

def get_algo(env, discount,
             *,
             algo_name,
             use_gpu=False,
             n_epochs=None,  # number of training epochs
             batch_size=None,  # batch size of the env sampler
             policy_network_hidden_sizes=(64, 64),
             policy_network_hidden_nonlinearity=torch.tanh,
             value_natwork_hidden_sizes=(64, 64),
             value_network_hidden_nonlinearity=torch.tanh,
             policy_lr=2e-4,  # optimization stepsize for policy update
             value_lr=5e-3,  # optimization stepsize for value regression
             opt_minibatch_size=128,  # optimization/replaybuffer minibatch size
             opt_n_grad_steps=1000,  # number of gradient updates
             kl_constraint=0.05,  # kl constraint between policy updates
             gae_lambda=0.98,  # lambda of gae estimator
             lr_clip_range=0.2, # the limit on the likelihood ratio between policies (PPO)
             steps_per_epoch=1,  # number of internal epochs steps per epoch
             n_workers=4,  # number of workers for data collection
             sampler_mode='ray'
             ):
    # return alg for env with discount

    assert isinstance(env, GymEnv)
    assert n_epochs is not None
    assert batch_size is not None

    def get_mlp_policy(stochastic, use_tanh):
        if stochastic and use_tanh:
            return TanhGaussianMLPPolicy(
                        env_spec=env.spec,
                        hidden_sizes=policy_network_hidden_sizes,
                        hidden_nonlinearity=policy_network_hidden_nonlinearity,
                        output_nonlinearity=None,
                        min_std=np.exp(-20.),
                        max_std=np.exp(2.))

        if stochastic and not use_tanh:
            return GaussianMLPPolicy(env.spec,
                        hidden_sizes=policy_network_hidden_sizes,
                        hidden_nonlinearity=policy_network_hidden_nonlinearity,
                        output_nonlinearity=None)

        if not stochastic:
            return DeterministicMLPPolicy(
                        env_spec=env.spec,
                        hidden_sizes=policy_network_hidden_sizes,
                        hidden_nonlinearity=policy_network_hidden_nonlinearity,
                        output_nonlinearity=torch.tanh if use_tanh else None)

    def get_mlp_value(form='Q'):
        if form=='Q':
            return ContinuousMLPQFunction(
                    env_spec=env.spec,
                    hidden_sizes=value_natwork_hidden_sizes,
                    hidden_nonlinearity=value_network_hidden_nonlinearity,
                    output_nonlinearity=None)
        if form=='V':
            return GaussianMLPValueFunction(
                    env_spec=env.spec,
                    hidden_sizes=policy_network_hidden_sizes,
                    hidden_nonlinearity=value_network_hidden_nonlinearity,
                    output_nonlinearity=None)

    def get_sampler(policy):
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


    def get_replay_buferr():
        return PathBuffer(capacity_in_transitions=int(1e6))

    def get_wrapped_optimizer(obj, lr, name='ADAM'):
        max_optimization_epochs = max(1, int(opt_n_grad_steps*opt_minibatch_size/batch_size))
        return OptimizerWrapper((torch.optim.Adam, dict(lr=lr)),
                                 obj,
                                 max_optimization_epochs=max_optimization_epochs,
                                 minibatch_size=opt_minibatch_size)

    if algo_name=='PPO':
        from garage.torch.algos import PPO
        policy = get_mlp_policy(stochastic=True, use_tanh=False)
        value_function = get_mlp_value('V')
        sampler = get_sampler(policy)
        algo = PPO(env_spec=env.spec,
                   policy=policy,
                   value_function=value_function,
                   sampler=sampler,
                   discount=discount,
                   center_adv=True,
                   positive_adv=False,
                   gae_lambda=gae_lambda,
                   lr_clip_range=lr_clip_range,  # The limit on the likelihood ratio between policies.
                   policy_optimizer=get_wrapped_optimizer(policy, policy_lr),
                   vf_optimizer=get_wrapped_optimizer(value_function, value_lr),
                   num_train_per_epoch=steps_per_epoch,
                )

    elif algo_name=='TRPO':
        from garage.torch.algos import TRPO
        policy = get_mlp_policy(stochastic=True, use_tanh=False)
        value_function = get_mlp_value('V')
        sampler = get_sampler(policy)
        policy_optimizer = OptimizerWrapper(
                            (ConjugateGradientOptimizer, dict(max_constraint_value=kl_constraint)),
                            policy)
        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    value_function=value_function,
                    sampler=sampler,
                    discount=discount,
                    center_adv=True,
                    positive_adv=False,
                    gae_lambda=gae_lambda,
                    policy_optimizer=policy_optimizer,
                    vf_optimizer=get_wrapped_optimizer(value_function),
                    num_train_per_epoch=steps_per_epoch,
                    )

    elif algo_name=='VPG':
        from garage.torch.algos import VPG
        policy = get_mlp_policy(stochastic=True, use_tanh=False)
        value_function = get_mlp_value('V')
        sampler = get_sampler(policy)
        algo = VPG(env_spec=env.spec,
                    policy=policy,
                    value_function=value_function,
                    sampler=sampler,
                    discount=discount,
                    center_adv=True,
                    positive_adv=False,
                    gae_lambda=gae_lambda,
                    policy_optimizer=OptimizerWrapper((torch.optim.Adam, dict(lr=policy_lr)),policy),
                    vf_optimizer=get_wrapped_optimizer(value_function, value_lr),
                    num_train_per_epoch=steps_per_epoch)

    elif algo_name=='SAC':
        from garage.torch.algos import SAC
        policy = get_mlp_policy(stochastic=True, use_tanh=True)
        qf1 = get_mlp_value('Q')
        qf2 = get_mlp_value('Q')
        replay_buffer = get_replay_buferr()
        sampler = get_sampler(policy)
        algo = SAC(env_spec=env.spec,
                   policy=policy,
                   qf1=qf1,
                   qf2=qf2,
                   sampler=sampler,
                   gradient_steps_per_itr=opt_n_grad_steps,
                   replay_buffer=replay_buffer,
                   min_buffer_size=1e4,
                   target_update_tau=5e-3,
                   discount=discount,
                   buffer_batch_size=opt_minibatch_size,
                   reward_scale=1.,
                   steps_per_epoch=steps_per_epoch,
                   policy_lr=policy_lr,
                   qf_lr=value_lr)

    elif algo_name=='TD3':
        from garage.np.exploration_policies import AddGaussianNoise
        from garage.np.policies import UniformRandomPolicy
        from garage.torch.algos import TD3
        num_timesteps = n_epochs * steps_per_epoch * batch_size
        policy = get_mlp_policy(stochastic=False, use_tanh=True)
        exploration_policy = AddGaussianNoise(env.spec,
                                              policy,
                                              total_timesteps=num_timesteps,
                                              max_sigma=0.1,
                                              min_sigma=0.1)
        uniform_random_policy = UniformRandomPolicy(env.spec)
        qf1 = get_mlp_value('Q')
        qf2 = get_mlp_value('Q')
        sampler = get_sampler(exploration_policy)
        replay_buffer = get_replay_buferr()
        algo = TD3(env_spec=env.spec,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  replay_buffer=replay_buffer,
                  sampler=sampler,
                  policy_optimizer=torch.optim.Adam,
                  qf_optimizer=torch.optim.Adam,
                  exploration_policy=exploration_policy,
                  uniform_random_policy=uniform_random_policy,
                  target_update_tau=0.005,
                  discount=discount,
                  policy_noise_clip=0.5,
                  policy_noise=0.2,
                  policy_lr=policy_lr,
                  qf_lr=value_lr,
                  steps_per_epoch=steps_per_epoch,
                  start_steps=1000,
                  grad_steps_per_env_step=opt_n_grad_steps,  # number of optimization steps
                  min_buffer_size=int(1e4),
                  buffer_batch_size=opt_minibatch_size)

    else:
        raise ValueError('Unknown algo_name')


    if use_gpu:
        prefer_gpu()
        if callable(getattr(self, 'to', None)):
            algo.to()

    return algo
