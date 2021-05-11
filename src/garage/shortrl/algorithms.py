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
             # required params
             algo_name,
             discount,
             env=None, # either env or episode_batch needs to be provided
             episode_batch=None,  # when provided, the algorithm is run in batch mode
             batch_size,  # batch size of the env sampler
             # optimization params
             init_policy=None,  # learner policy
             policy_network_hidden_sizes=(256, 128),
             policy_network_hidden_nonlinearity=torch.tanh,
             value_natwork_hidden_sizes=(256, 128),
             value_network_hidden_nonlinearity=torch.tanh,
             policy_lr=1e-3,  # optimization stepsize for policy update
             value_lr=1e-3,  # optimization stepsize for value regression
             opt_minibatch_size=128,  # optimization/replaybuffer minibatch size
             opt_n_grad_steps=1000,  # number of gradient updates per epoch
             num_evaluation_episodes=10, # number of episodes to evaluate (only affect off-policy algorithms)
             steps_per_epoch=1,  # number of internal epochs steps per epoch
             n_epochs=None,  # number of training epochs
             #
             n_workers=4,  # number of workers for data collection
             use_gpu=False,  # try to use gpu, if implemented
             sampler_mode='ray',
             #
             # algorithm specific hyperparmeters
             target_update_tau=5e-3, # for target network
             expert_policy=None,  # for BC
             kl_constraint=0.05,  # kl constraint between policy updates
             gae_lambda=0.98,  # lambda of gae estimator
             lr_clip_range=0.2, # the limit on the likelihood ratio between policies (PPO)
             **kwargs,
             ):
    # return alg for env with discount

    assert isinstance(env, GymEnv) or env is None
    assert not (env is None and episode_batch is None)
    assert batch_size is not None

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
        sampler = ru.BatchSampler(episode_batch=episode_batch)
        get_sampler = lambda p: sampler
        env_spec = episode_batch.env_spec

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
        from garage.torch.algos import PPO
        policy = get_mlp_policy(stochastic=True, clip_output=False)
        value_function = get_mlp_value('V')
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
        algo = TRPO(env_spec=env_spec,
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
                    num_train_per_epoch=steps_per_epoch)

    elif algo_name=='SAC':
        from garage.torch.algos import SAC
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
                   min_buffer_size=1e4,
                   target_update_tau=target_update_tau,
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
        algo = CQL(env_spec=env_spec,
                   policy=policy,
                   qf1=qf1,
                   qf2=qf2,
                   sampler=sampler,
                   gradient_steps_per_itr=opt_n_grad_steps,
                   replay_buffer=replay_buffer,
                   min_buffer_size=1e4,
                   target_update_tau=target_update_tau,
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
        policy = get_mlp_policy(stochastic=False, clip_output=True)
        exploration_policy = AddGaussianNoise(env_spec,
                                              policy,
                                              total_timesteps=num_timesteps,
                                              max_sigma=0.1,
                                              min_sigma=0.1)
        uniform_random_policy = UniformRandomPolicy(env_spec)
        qf1 = get_mlp_value('Q')
        qf2 = get_mlp_value('Q')
        sampler = get_sampler(exploration_policy)
        replay_buffer = get_replay_buferr()
        algo = TD3(env_spec=env_spec,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  replay_buffer=replay_buffer,
                  sampler=sampler,
                  policy_optimizer=torch.optim.Adam,
                  qf_optimizer=torch.optim.Adam,
                  exploration_policy=exploration_policy,
                  uniform_random_policy=uniform_random_policy,
                  target_update_tau=target_update_tau,
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
                  buffer_batch_size=opt_minibatch_size,
                  )

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

    elif algo_name in ['DQN', 'DDQN']:
        from garage.torch.policies import DiscreteQFArgmaxPolicy
        from garage.torch.q_functions import DiscreteMLPQFunction
        from garage.np.exploration_policies import EpsilonGreedyPolicy

        double_q = algo_name is 'DDQN'

        replay_buffer = get_replay_buferr()
        qf = DiscreteMLPQFunction(env_spec=env_spec, hidden_sizes=value_natwork_hidden_sizes)
        policy = DiscreteQFArgmaxPolicy(env_spec=env_spec, qf=qf)
        exploration_policy = EpsilonGreedyPolicy(env_spec=env_spec,
                                                 policy=policy,
                                                 total_timesteps=num_timesteps,
                                                 max_epsilon=1.0,
                                                 min_epsilon=0.01,
                                                 decay_ratio=1.0)
        sampler = get_sampler(exploration_policy)
        algo = DQN(env_spec=env_spec,
                    policy=policy,
                    qf=qf,
                    exploration_policy=exploration_policy,
                    replay_buffer=replay_buffer,
                    sampler=sampler,
                    steps_per_epoch=steps_per_epoch,
                    qf_lr=value_lr,
                    discount=discount,
                    min_buffer_size=int(1e4),
                    n_train_steps=opt_n_grad_steps,
                    buffer_batch_size=opt_minibatch_size,
                    deterministic_eval=True,
                    num_eval_episodes=num_evaluation_episodes,
                    target_update_tau=target_update_tau,
                    double_q=double_q) # false



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

    # NOTE Evaluate performance after training
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

    # NOTE Use a fixed number of updates and minibatch size instead.
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




from garage.torch.algos import DQN as garageDQN

class DQN(garageDQN):
    """ Use Polyak average as the target. """

    def __init__(self, *args, target_update_tau=5e-3, **kwargs):
        self._tau = target_update_tau
        super().__init__(*args, **kwargs)

    # NOTE Use Polyak average to update the target instead
    def _train_once(self, itr, episodes):
        """Perform one iteration of training.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        """
        self.replay_buffer.add_episode_batch(episodes)

        epoch = itr / self._steps_per_epoch

        for _ in range(self._n_train_steps):
            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                timesteps = self.replay_buffer.sample_timesteps(
                    self._buffer_batch_size)
                qf_loss, y, q = tuple(v.cpu().numpy()
                                      for v in self._optimize_qf(timesteps))

                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y)
                self._epoch_qs.append(q)

                # Polyak update
                for t_param, param in zip(self._target_qf.parameters(), self._qf.parameters()):
                    t_param.data.copy_(t_param.data * (1.0 - self._tau) + param.data * self._tau)

        if itr % self._steps_per_epoch == 0:
            self._log_eval_results(epoch)

        # if itr % self._target_update_freq == 0:
            # self._target_qf = copy.deepcopy(self._qf)
