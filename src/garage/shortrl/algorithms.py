import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dowel import tabular

from garage.envs import GymEnv
from garage.torch import prefer_gpu
from garage.torch.optimizers import OptimizerWrapper, ConjugateGradientOptimizer
from garage.torch.algos import BC as garageBC
from garage.torch import as_torch
from garage import log_performance, obtain_evaluation_episodes

from garage.shortrl import rl_utils as ru
from functools import partial

def get_algo(*,
             env,
             algo_name,
             discount=0.99,
             init_policy=None,  # learner policy
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
             num_evaluation_episodes=10, # number of episodes to evaluate (only affect off-policy algorithms)
             steps_per_epoch=1,  # number of internal epochs steps per epoch
             n_workers=4,  # number of workers for data collection
             use_gpu=False,  # try to use gpu, if implemented
             sampler_mode='ray',
             #
             # algorithm specific hyperparmeters
             expert_policy=None,  # for BC
             kl_constraint=0.05,  # kl constraint between policy updates
             gae_lambda=0.98,  # lambda of gae estimator
             lr_clip_range=0.2, # the limit on the likelihood ratio between policies (PPO)
             **kwargs,
             ):
    # return alg for env with discount

    assert isinstance(env, GymEnv)
    assert n_epochs is not None
    assert batch_size is not None

    # Define some helper functions
    if init_policy is None:
        get_mlp_policy = partial(ru.get_mlp_policy,
                                env=env,
                                hidden_sizes=policy_network_hidden_sizes,
                                hidden_nonlinearity=policy_network_hidden_nonlinearity)
    else:
         get_mlp_policy = lambda *args, **kwargs : init_policy

    get_mlp_value = partial(ru.get_mlp_value,
                            env=env,
                            hidden_sizes=value_natwork_hidden_sizes,
                            hidden_nonlinearity=value_network_hidden_nonlinearity)

    get_sampler = partial(ru.get_sampler,
                          env=env,
                          sampler_mode=sampler_mode,
                          n_workers=n_workers)

    get_replay_buferr = ru.get_replay_buferr
    get_wrapped_optimizer = partial(ru.get_optimizer,
                                    max_optimization_epochs= max(1,int(opt_n_grad_steps*opt_minibatch_size/batch_size)),
                                    minibatch_size=opt_minibatch_size)

    # Create algo
    if algo_name=='PPO':
        from garage.torch.algos import PPO
        policy = get_mlp_policy(stochastic=True, clip_output=False)
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
        policy = get_mlp_policy(stochastic=True, clip_output=False)
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
                    vf_optimizer=get_wrapped_optimizer(value_function,lr=value_lr),
                    num_train_per_epoch=steps_per_epoch,
                    )

    elif algo_name=='VPG':
        from garage.torch.algos import VPG
        policy = get_mlp_policy(stochastic=True, clip_output=False)
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
        policy = get_mlp_policy(stochastic=True, clip_output=True)
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
                   num_evaluation_episodes=num_evaluation_episodes,
                   policy_lr=policy_lr,
                   qf_lr=value_lr)

    elif algo_name=='CQL':
        from garage.torch.algos import CQL
        policy = get_mlp_policy(stochastic=True, clip_output=True)
        qf1 = get_mlp_value('Q')
        qf2 = get_mlp_value('Q')
        replay_buffer = get_replay_buferr()
        sampler = get_sampler(policy)
        algo = CQL(env_spec=env.spec,
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
                   num_evaluation_episodes=num_evaluation_episodes,
                   policy_lr=policy_lr,
                   qf_lr=value_lr)

    elif algo_name=='TD3':
        from garage.np.exploration_policies import AddGaussianNoise
        from garage.np.policies import UniformRandomPolicy
        from garage.torch.algos import TD3
        num_timesteps = n_epochs * steps_per_epoch * batch_size
        policy = get_mlp_policy(stochastic=False, clip_output=True)
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
                  num_evaluation_episodes=num_evaluation_episodes,
                  start_steps=1000,
                  grad_steps_per_env_step=opt_n_grad_steps,  # number of optimization steps
                  min_buffer_size=int(1e4),
                  buffer_batch_size=opt_minibatch_size)

    elif algo_name=='BC':
        sampler=get_sampler(expert_policy)
        assert init_policy is not None
        assert expert_policy is not None
        algo = BC(env.spec,
                init_policy,
                source=expert_policy,
                sampler=sampler,
                batch_size=batch_size,
                gradient_steps_per_itr=opt_n_grad_steps,
                minibatch_size=opt_minibatch_size,
                policy_lr=policy_lr,
                loss='mse', #'log_prob' if isinstance(policy,StochasticPolicy) else 'mse'
                )

    else:
        raise ValueError('Unknown algo_name')

    if use_gpu:
        prefer_gpu()
        if callable(getattr(algo, 'to', None)):
            algo.to()

    return algo

class BC(garageBC):
    def __init__(self, *args,
                 gradient_steps_per_itr=1000,
                 minibatch_size=128,
                 **kwargs):
        self._gradient_steps_per_itr = gradient_steps_per_itr
        self._minibatch_size = minibatch_size
        kwargs['minibatches_per_epoch']=None
        super().__init__(*args, **kwargs)

    # Evaluate performance after training
    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, for services such as
                snapshotting and sampler control.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        for epoch in trainer.step_epochs():
            losses = self._train_once(trainer, epoch)
            if self._eval_env is not None:
                log_performance(epoch,
                                obtain_evaluation_episodes(
                                    self.learner, self._eval_env,
                                    num_eps=10),
                                discount=1.0)
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', np.mean(losses))
                tabular.record('StdLoss', np.std(losses))

    # Use a fixed number of updates and minibatch size instead.
    def _train_once(self, trainer, epoch):
        """Obtain samplers and train for one epoch.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            List[float]: Losses.

        """
        batch = self._obtain_samples(trainer, epoch)
        losses = []
        for _ in range(self._gradient_steps_per_itr):
            minibatch = np.random.randint(len(batch.observations), size=self._minibatch_size)
            observations = as_torch(batch.observations[minibatch])
            actions = as_torch(batch.actions[minibatch])
            self._optimizer.zero_grad()
            loss = self._compute_loss(observations, actions)
            loss.backward()
            losses.append(loss.item())
            self._optimizer.step()
        return losses
