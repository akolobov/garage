import numpy as np
import torch
from functools import partial
from garage.envs import GymEnv
from garage.torch import prefer_gpu
from garage.torch.optimizers import OptimizerWrapper, ConjugateGradientOptimizer
from garage.shortrl import rl_utils as ru

from garage.shortrl.algos.sac import SAC
from garage.shortrl.algos.bc import BC
# from garage.shortrl.algos.dqn import DQN
# from garage.shortrl.algos.vpg import VPG
# from garage.shortrl.algos.ppo import PPO
# from garage.shortrl.algos.trpo import TRPO

from garage.torch.algos import VPG, PPO

__all__ = ['get_algo', 'log_performance', 'SAC', 'BC', 'DQN', 'VPG']

def get_algo(*,
             # required params
             algo_name,
             discount,
             env=None, # either env or episode_batch needs to be provided
             episode_batch=None,  # when provided, the algorithm is run in batch mode
             batch_size,  # batch size of the env sampler
             # heuristic guidance
             lambd=1.0,
             heuristic=None,
             # networks
             init_policy=None,  # learner policy
             policy_network_hidden_sizes=(256, 128),
             policy_network_hidden_nonlinearity=torch.tanh,
             value_natwork_hidden_sizes=(256, 128),
             value_network_hidden_nonlinearity=torch.tanh,
             # optimization
             policy_lr=1e-3,  # optimization stepsize for policy update
             value_lr=1e-3,  # optimization stepsize for value regression
             opt_minibatch_size=128,  # optimization/replaybuffer minibatch size
             opt_n_grad_steps=1000,  # number of gradient updates per epoch
             num_evaluation_episodes=10, # number of episodes to evaluate (only affect off-policy algorithms)
             steps_per_epoch=1,  # number of internal epochs steps per epoch
             n_epochs=None,  # number of training epochs
             randomize_episode_batch=True,
             reward_avg_rate=1e-3,
             # compute
             n_workers=4,  # number of workers for data collection
             use_gpu=False,  # try to use gpu, if implemented
             sampler_mode='ray',
             # algorithm specific hyperparmeters
             target_update_tau=5e-3, # for target network
             expert_policy=None,  # for BC
             kl_constraint=0.05,  # kl constraint between policy updates
             gae_lambda=0.98,  # lambda of gae estimator
             lr_clip_range=0.2, # the limit on the likelihood ratio between policies (PPO)
             vae_latent_dim=32,
             eps_greed_decay_ratio=1, # for DQN/DDQN
             **kwargs,
             ):
    # return alg for env with discount

    assert isinstance(env, GymEnv) or env is None
    assert not (env is None and episode_batch is None)
    assert batch_size is not None

    # Parse algo_name
    value_ensemble_mode='P'
    value_ensemble_size=1
    if '_' in algo_name:
        algo_name, ensemble_mode = algo_name.split('_')
        value_ensemble_size= int(ensemble_mode[:-1]) # ensemble size of value network
        value_ensemble_mode= ensemble_mode[-1] # ensemble mode of value network, P or O

    # For normalized behaviors
    opt_n_grad_steps = int(opt_n_grad_steps/steps_per_epoch)
    n_epochs = n_epochs or np.inf
    num_timesteps = n_epochs * steps_per_epoch * batch_size


    # Define some helper functions used by most algorithms
    if episode_batch is None:
        get_sampler = partial(ru.get_sampler,
                              env=env,
                              sampler_mode=sampler_mode,
                              n_workers=n_workers)
        env_spec = env.spec
    else:
        sampler = ru.BatchSampler(episode_batch=episode_batch, randomize=randomize_episode_batch)
        get_sampler = lambda p: sampler
        env_spec = episode_batch.env_spec
        num_evaluation_episodes=0

    if init_policy is None:
        get_mlp_policy = partial(ru.get_mlp_policy,
                                 env_spec=env_spec,
                                 hidden_sizes=policy_network_hidden_sizes,
                                 hidden_nonlinearity=policy_network_hidden_nonlinearity)
    else:
         get_mlp_policy = lambda *a, **kw : init_policy

    get_mlp_value = partial(ru.get_mlp_value,
                            env_spec=env_spec,
                            hidden_sizes=value_natwork_hidden_sizes,
                            hidden_nonlinearity=value_network_hidden_nonlinearity)

    get_replay_buferr = ru.get_replay_buferr
    max_optimization_epochs = max(1,int(opt_n_grad_steps*opt_minibatch_size/batch_size))
    get_wrapped_optimizer = partial(ru.get_optimizer,
                                    max_optimization_epochs=max_optimization_epochs,
                                    minibatch_size=opt_minibatch_size)

    # Create an algorithm instance
    if algo_name=='PPO':
        policy = get_mlp_policy(stochastic=True, clip_output=False)
        value_function = get_mlp_value('V',
                                       ensemble_mode=value_ensemble_mode,
                                       ensemble_size=value_ensemble_size)
        sampler = get_sampler(policy)
        algo = PPO(env_spec=env_spec,
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
                   num_train_per_epoch=steps_per_epoch)
                #    lambd=lambd,
                #    euristic=heuristic)


    # elif algo_name=='TRPO':
    #     policy = get_mlp_policy(stochastic=True, clip_output=False)
    #     value_function = get_mlp_value('V',
    #                                    ensemble_mode=value_ensemble_mode,
    #                                    ensemble_size=value_ensemble_size)
    #     sampler = get_sampler(policy)
    #     policy_optimizer = OptimizerWrapper(
    #                         (ConjugateGradientOptimizer, dict(max_constraint_value=kl_constraint)),
    #                         policy)
    #     algo = TRPO(env_spec=env_spec,
    #                 policy=policy,
    #                 value_function=value_function,
    #                 sampler=sampler,
    #                 discount=discount,
    #                 center_adv=True,
    #                 positive_adv=False,
    #                 gae_lambda=gae_lambda,
    #                 policy_optimizer=policy_optimizer,
    #                 vf_optimizer=get_wrapped_optimizer(value_function,lr=value_lr),
    #                 num_train_per_epoch=steps_per_epoch,
    #                 lambd=lambd,
    #                 heuristic=heuristic)

    elif algo_name in ['VPG', 'VAEVPG']:
        policy = get_mlp_policy(stochastic=True, clip_output=False)
        value_function = get_mlp_value('V',
                                       ensemble_mode=value_ensemble_mode,
                                       ensemble_size=value_ensemble_size)
        sampler = get_sampler(policy)

        # Use vae to induce pessimism.
        vae = vae_optimizer = None
        use_pessimism = False
        if algo_name=='VAEVPG':
            from garage.shortrl.vaes import StateVAE
            use_pessimism = True
            vae = StateVAE(env_spec=env_spec,
                 hidden_sizes=value_natwork_hidden_sizes,
                 hidden_nonlinearity=value_network_hidden_nonlinearity,
                 latent_dim=vae_latent_dim)
            vae_optimizer = get_wrapped_optimizer(vae, value_lr)

        algo = VPG(env_spec=env_spec,
                    policy=policy,
                    value_function=value_function,
                    sampler=sampler,
                    discount=discount,
                    center_adv=True,
                    positive_adv=False,
                    gae_lambda=gae_lambda,
                    policy_optimizer=OptimizerWrapper((torch.optim.Adam, dict(lr=policy_lr)),policy),
                    vf_optimizer=get_wrapped_optimizer(value_function, value_lr),
                    num_train_per_epoch=steps_per_epoch,
                    pessimistic_vae_filter=use_pessimism,
                    vae=vae,
                    vae_optimizer=vae_optimizer)


    elif algo_name=='SAC':
        # from garage.torch.algos import SAC
        policy = get_mlp_policy(stochastic=True, clip_output=True)
        qf1 = get_mlp_value('Q')
        qf2 = get_mlp_value('Q')
        replay_buffer = get_replay_buferr()
        sampler = get_sampler(policy)
        algo = SAC(env_spec=env_spec,
                   policy=policy,
                   qf1=qf1,
                   qf2=qf2,
                   sampler=sampler,
                   gradient_steps_per_itr=opt_n_grad_steps,
                   replay_buffer=replay_buffer,
                   min_buffer_size=int(0),
                   target_update_tau=target_update_tau,
                   discount=discount,
                   buffer_batch_size=opt_minibatch_size,
                   reward_scale=1.,
                   steps_per_epoch=steps_per_epoch,
                   num_evaluation_episodes=num_evaluation_episodes,
                   policy_lr=policy_lr,
                   qf_lr=value_lr,
                   lambd=lambd,
                   heuristic=heuristic,
                   reward_avg_rate=reward_avg_rate)

    # elif algo_name=='CQL':
    #     from garage.torch.algos import CQL
    #     policy = get_mlp_policy(stochastic=True, clip_output=True)
    #     qf1 = get_mlp_value('Q')
    #     qf2 = get_mlp_value('Q')
    #     replay_buffer = get_replay_buferr()
    #     sampler = get_sampler(policy)
    #     algo = CQL(env_spec=env_spec,
    #                policy=policy,
    #                qf1=qf1,
    #                qf2=qf2,
    #                sampler=sampler,
    #                gradient_steps_per_itr=opt_n_grad_steps,
    #                replay_buffer=replay_buffer,
    #                min_buffer_size=1e4,
    #                target_update_tau=target_update_tau,
    #                discount=discount,
    #                buffer_batch_size=opt_minibatch_size,
    #                reward_scale=1.,
    #                steps_per_epoch=steps_per_epoch,
    #                num_evaluation_episodes=num_evaluation_episodes,
    #                policy_lr=policy_lr,
    #                qf_lr=value_lr)

    # elif algo_name=='TD3':
    #     from garage.np.exploration_policies import AddGaussianNoise
    #     from garage.np.policies import UniformRandomPolicy
    #     from garage.torch.algos import TD3
    #     policy = get_mlp_policy(stochastic=False, clip_output=True)
    #     exploration_policy = AddGaussianNoise(env_spec,
    #                                           policy,
    #                                           total_timesteps=num_timesteps,
    #                                           max_sigma=0.1,
    #                                           min_sigma=0.1)
    #     uniform_random_policy = UniformRandomPolicy(env_spec)
    #     qf1 = get_mlp_value('Q')
    #     qf2 = get_mlp_value('Q')
    #     sampler = get_sampler(exploration_policy)
    #     replay_buffer = get_replay_buferr()
    #     algo = TD3(env_spec=env_spec,
    #               policy=policy,
    #               qf1=qf1,
    #               qf2=qf2,
    #               replay_buffer=replay_buffer,
    #               sampler=sampler,
    #               policy_optimizer=torch.optim.Adam,
    #               qf_optimizer=torch.optim.Adam,
    #               exploration_policy=exploration_policy,
    #               uniform_random_policy=uniform_random_policy,
    #               target_update_tau=target_update_tau,
    #               discount=discount,
    #               policy_noise_clip=0.5,
    #               policy_noise=0.2,
    #               policy_lr=policy_lr,
    #               qf_lr=value_lr,
    #               steps_per_epoch=steps_per_epoch,
    #               num_evaluation_episodes=num_evaluation_episodes,
    #               start_steps=1000,
    #               grad_steps_per_env_step=opt_n_grad_steps,  # number of optimization steps
    #               min_buffer_size=int(1e4),
    #               buffer_batch_size=opt_minibatch_size,
    #               )

    elif algo_name=='BC':
        sampler=get_sampler(expert_policy)
        assert init_policy is not None
        assert expert_policy is not None
        algo = BC(env_spec,
                  init_policy,
                  source=expert_policy,
                  sampler=sampler,
                  batch_size=batch_size,
                  gradient_steps_per_itr=opt_n_grad_steps,
                  minibatch_size=opt_minibatch_size,
                  policy_lr=policy_lr,
                  loss='mse', #'log_prob' if isinstance(policy,StochasticPolicy) else 'mse'
                  )

    # elif algo_name in ['DQN', 'DDQN']:
    #     from garage.torch.policies import DiscreteQFArgmaxPolicy
    #     from garage.torch.q_functions import DiscreteMLPQFunction
    #     from garage.np.exploration_policies import EpsilonGreedyPolicy

    #     double_q = algo_name is 'DDQN'

    #     replay_buffer = get_replay_buferr()
    #     qf = DiscreteMLPQFunction(env_spec=env_spec, hidden_sizes=value_natwork_hidden_sizes)
    #     policy = DiscreteQFArgmaxPolicy(env_spec=env_spec, qf=qf)
    #     exploration_policy = EpsilonGreedyPolicy(env_spec=env_spec,
    #                                              policy=policy,
    #                                              total_timesteps=num_timesteps,
    #                                              max_epsilon=1.0,
    #                                              min_epsilon=0.01,
    #                                              decay_ratio=eps_greed_decay_ratio)
    #     sampler = get_sampler(exploration_policy)
    #     algo = DQN(env_spec=env_spec,
    #                 policy=policy,
    #                 qf=qf,
    #                 exploration_policy=exploration_policy,
    #                 replay_buffer=replay_buffer,
    #                 sampler=sampler,
    #                 steps_per_epoch=steps_per_epoch,
    #                 qf_lr=value_lr,
    #                 discount=discount,
    #                 min_buffer_size=int(1e4),
    #                 n_train_steps=opt_n_grad_steps,
    #                 buffer_batch_size=opt_minibatch_size,
    #                 deterministic_eval=True,
    #                 num_eval_episodes=num_evaluation_episodes,
    #                 target_update_tau=target_update_tau,
    #                 double_q=double_q) # false



    else:
        raise ValueError('Unknown algo_name')

    if use_gpu:
        prefer_gpu()
        if callable(getattr(algo, 'to', None)):
            algo.to()

    return algo
