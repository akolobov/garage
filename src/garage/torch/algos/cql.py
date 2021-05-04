"""This modules creates a cql model in PyTorch."""
# yapf: disable
from collections import deque
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance, obtain_evaluation_episodes, StepType
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, global_device

# yapf: enable


class CQL(RLAlgorithm):
    """A CQL Model in Torch.

    Based on Conservative Q-Learning for Offline Reinforcement Learning:
        https://arxiv.org/abs/2006.04779

    Conservative Q-Learning (CQL) is an algorithm which builds on
    Soft Actor-Critic (SAC) by penalizing the learned Q-functions on 
    out-of-support state-action pairs. Such conservative Q-functions
    can allow more robust extrapolations in the offline setting where
    we cannot collect more data from environment interactions to correct
    over-optimistic extrapolations. The argument list inherits all of SAC's
    arguments as well as a few CQL-specific options.
    
    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by CQL.
        qf1 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        qf2 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        replay_buffer (ReplayBuffer): Stores transitions that are previously
            collected by the sampler.
        sampler (garage.sampler.Sampler): Sampler.
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `env_spec.max_episode_length`.
        gradient_steps_per_itr (int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): Discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): coefficient that controls the rate at which
            the target q_functions update over optimization iterations.
        policy_lr (float): learning rate for policy optimizers.
        qf_lr (float): learning rate for q_function optimizers.
        reward_scale (float): reward scale. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.
        eval_env (Environment): environment used for collecting evaluation
            episodes. If None, a copy of the train env is used.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.

        [https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py]
        temp (float): Temperature to use when exponentiating Q-values for 
            CQL's objective
        min_q_weight (float): Default is 1. Used for normalizing Q-values when 
            computing CQL objective.
        max_q_backup (bool): If True, we sample 10 actions from next state and
            take max Q as the approximation of max_a' Q(s',a')
        deterministic_backup (bool): True by default, If True this subtracts 
            the alpha*log pi factor that SAC also subtracts from critic 
            Q-values to construct regression targets
        logsumexp_continuous (bool): If set to True, then we approximate 
            log sum_a exp(Q(s,a)) by randomly sampling num_random actions. 
            Needed for continuous action tasks.
        num_random (int): How many actions to sample to approximate 
            log-sum-exp for continuous action tasks.
        with_lagrange (bool): Whether to tune CQL's alpha using 
            dual gradient descent (requires lagrange_thresh).
        lagrange_thresh (float): tau defined in Eq30 of Appendix F. If expected Q gap
            is larger than tau, CQL's alpha is automatically increased.
    """

    def __init__(
            self,
            env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            sampler,
            *,  # Everything after this is numbers.
            max_episode_length_eval=None,
            gradient_steps_per_itr,
            fixed_alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1.0,
            optimizer=torch.optim.Adam,
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            logsumexp_continuous=True,
            temp=1.0,
            min_q_weight=1.0,
            max_q_backup=False,
            deterministic_backup=False,
            num_random=10,
            with_lagrange=True,
            lagrange_thresh=10.0):

        self._qf1 = qf1
        self._qf2 = qf2
        self.replay_buffer = replay_buffer
        self._tau = target_update_tau
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_episodes = num_evaluation_episodes
        self._eval_env = eval_env

        self._min_buffer_size = min_buffer_size
        self._steps_per_epoch = steps_per_epoch
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        self.max_episode_length = env_spec.max_episode_length
        self._max_episode_length_eval = env_spec.max_episode_length

        if max_episode_length_eval is not None:
            self._max_episode_length_eval = max_episode_length_eval
        self._use_deterministic_evaluation = use_deterministic_evaluation

        self.policy = policy
        self.env_spec = env_spec
        self.replay_buffer = replay_buffer

        self._sampler = sampler

        self._reward_scale = reward_scale
        # use 2 target q networks
        self._target_qf1 = copy.deepcopy(self._qf1)
        self._target_qf2 = copy.deepcopy(self._qf2)
        self._policy_optimizer = self._optimizer(self.policy.parameters(),
                                                 lr=self._policy_lr)
        self._qf1_optimizer = self._optimizer(self._qf1.parameters(),
                                              lr=self._qf_lr)
        self._qf2_optimizer = self._optimizer(self._qf2.parameters(),
                                              lr=self._qf_lr)
        # automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self._log_alpha = torch.Tensor([self._initial_log_entropy
                                            ]).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha],
                                              lr=self._policy_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha]).log()
        self.episode_rewards = deque(maxlen=30)

        self._with_lagrange = with_lagrange
        if self._with_lagrange:
            self._target_action_gap = lagrange_thresh
            self._log_alpha_prime = torch.Tensor([self._initial_log_entropy
                                            ]).requires_grad_()
            self._alpha_prime_optimizer = self._optimizer([self._log_alpha_prime],
                lr=self._qf_lr,
            )
        
        self._temp = temp
        self._logsumexp_continuous = logsumexp_continuous
        self._min_q_weight = min_q_weight

        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._num_random = num_random
    
    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds
        
    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        
        new_obs_actions_dist = network(obs_temp)[0]
        new_obs_actions_pre_tanh, new_obs_actions = (
            new_obs_actions_dist.rsample_with_pre_tanh_value())
        new_obs_log_pi = new_obs_actions_dist.log_prob(
            value=new_obs_actions, pre_tanh_value=new_obs_actions_pre_tanh)
        
        return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
        
    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for _ in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                trainer.step_episode = trainer.obtain_samples(
                    trainer.step_itr, batch_size)
                path_returns = []
                for path in trainer.step_episode:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=np.array([
                                 step_type == StepType.TERMINAL
                                 for step_type in path['step_types']
                             ]).reshape(-1, 1)))
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) == len(trainer.step_episode)
                self.episode_rewards.append(np.mean(path_returns))
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()
            last_return = self._evaluate_policy(trainer.step_itr)
            self._log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            trainer.step_itr += 1

        return np.mean(last_return)

    def train_once(self, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
            samples = self.replay_buffer.sample_transitions(
                self._buffer_batch_size)
            samples = as_torch_dict(samples)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(samples)
            self._update_targets()

        return policy_loss, qf1_loss, qf2_loss

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        This function exists in case there are versions of sac that need
        access to a modified log_alpha, such as multi_task sac.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: log_alpha

        """
        del samples_data
        log_alpha = self._log_alpha
        return log_alpha

    def _temperature_objective(self, log_pi, samples_data):
        """Compute the temperature/alpha coefficient loss.

        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.

        """
        alpha_loss = 0
        if self._use_automatic_entropy_tuning:
            alpha_loss = (-(self._get_log_alpha(samples_data)) *
                          (log_pi.detach() + self._target_entropy)).mean()
        return alpha_loss

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        """Compute the Policy/Actor loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions (torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions (torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from the Policy/Actor.

        """
        obs = samples_data['observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
            
        min_q_new_actions = torch.min(self._qf1(obs, new_actions),
                                      self._qf2(obs, new_actions))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten()).mean()
        return policy_objective

    def _critic_objective(self, samples_data):
        """Compute the Q-function/critic loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten()
        terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        q1_pred = self._qf1(obs, actions)
        q2_pred = self._qf2(obs, actions)

        new_next_actions_dist = self.policy(next_obs)[0]
        new_next_actions_pre_tanh, new_next_actions = (
            new_next_actions_dist.rsample_with_pre_tanh_value())
        new_log_pi = new_next_actions_dist.log_prob(
            value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
        
        new_curr_actions_dist = self.policy(obs)[0]
        new_curr_actions_pre_tanh, new_curr_actions = (
            new_curr_actions_dist.rsample_with_pre_tanh_value())
        new_curr_log_pi = new_curr_actions_dist.log_prob(
            value=new_curr_actions, pre_tanh_value=new_curr_actions_pre_tanh)
        
        if not self._max_q_backup:
            target_q_values = torch.min(
                    self._target_qf1(next_obs, new_next_actions),
                    self._target_qf2(next_obs, new_next_actions)).flatten()
            if not self._deterministic_backup:
                target_q_values = target_q_values - (alpha * new_log_pi)
        else:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self._target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self._target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values).flatten()
        
        with torch.no_grad():
            q_target = rewards * self._reward_scale + (
                1. - terminals) * self._discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        #Add CQL
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self._num_random, actions.shape[-1]).uniform_(-1, 1).to(obs.device)
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self._num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self._num_random, network=self.policy)
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self._qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self._qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self._qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self._qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self._qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self._qf2)

        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)
        std_q2 = torch.std(cat_q2, dim=1)
        
        if self._logsumexp_continuous:
            # importance sampled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(), q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(), q2_curr_actions - curr_log_pis.detach()], 1
            )
        
        min_qf1_loss = torch.logsumexp(cat_q1 / self._temp, dim=1,).mean() * self._min_q_weight * self._temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self._temp, dim=1,).mean() * self._min_q_weight * self._temp
                    
        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self._min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self._min_q_weight
        
        if self._with_lagrange:
            alpha_prime = torch.clamp(self._log_alpha_prime.exp(), min=0.0, max=1000000.0).to(min_qf1_loss.device)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self._target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self._target_action_gap)

            self._alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
            alpha_prime_loss.backward(retain_graph=True)
            self._alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss
        
        return qf1_loss, qf2_loss

    def _update_targets(self):
        """Update parameters in the target q-functions."""
        target_qfs = [self._target_qf1, self._target_qf2]
        qfs = [self._qf1, self._qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                                   param.data * self._tau)

    def optimize_policy(self, samples_data):
        """Optimize the policy q_functions, and temperature coefficient.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self._qf1_optimizer.step()

        self._qf2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self._qf2_optimizer.step()

        action_dists = self.policy(obs)[0]
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)
        self._policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)

        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        eval_episodes = obtain_evaluation_episodes(
            self.policy,
            self._eval_env,
            self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            deterministic=self._use_deterministic_evaluation)
        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount)
        return last_return

    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        with torch.no_grad():
            tabular.record('AlphaTemperature/mean',
                           self._log_alpha.exp().mean().item())
        tabular.record('Policy/Loss', policy_loss.item())
        tabular.record('QF/{}'.format('Qf1Loss'), float(qf1_loss))
        tabular.record('QF/{}'.format('Qf2Loss'), float(qf2_loss))
        tabular.record('ReplayBuffer/buffer_size',
                       self.replay_buffer.n_transitions_stored)
        tabular.record('Average/TrainAverageReturn',
                       np.mean(self.episode_rewards))

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._qf1, self._qf2, self._target_qf1,
            self._target_qf2
        ]

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
        if not self._use_automatic_entropy_tuning:
            self._log_alpha = torch.Tensor([self._fixed_alpha
                                            ]).log().to(device)
        else:
            self._log_alpha = torch.Tensor([self._initial_log_entropy
                                            ]).to(device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._policy_lr)

        if self._with_lagrange:
            self._log_alpha_prime = torch.Tensor([self._initial_log_entropy
                                            ]).to(device).requires_grad_()
            self._alpha_prime_optimizer = self._optimizer([self._log_alpha_prime],
                lr=self._qf_lr)
