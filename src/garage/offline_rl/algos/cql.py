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
from garage.torch.algos import SAC
# yapf: enable

class CQL(SAC):
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
            target_update_tau=5e-3,  # 1e-2
            policy_lr=1e-4,  # 1e-3
            qf_lr=3e-4,  # 1e-3
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
            deterministic_backup=True,
            num_random=10,
            lagrange_thresh=-1,
            n_bc_steps=0):

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

        with_lagrange = lagrange_thresh > 0 #XXX
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
        self._n_updates_performed= 0
        self._n_bc_steps = n_bc_steps

        # XXX
        assert not max_q_backup
        assert deterministic_backup

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

        if not self._max_q_backup:
            target_q_values = torch.min(
                    self._target_qf1(next_obs, new_next_actions),
                    self._target_qf2(next_obs, new_next_actions)).flatten()
            if not self._deterministic_backup:
                target_q_values = target_q_values - (alpha * new_log_pi)
        else:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=num_random, network=self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self._target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self._target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values).flatten()

        with torch.no_grad():
            q_target = rewards * self._reward_scale + (
                1. - terminals) * self._discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        ## add CQL
        new_curr_actions_dist = self.policy(obs)[0]
        new_curr_actions_pre_tanh, new_curr_actions = (
            new_curr_actions_dist.rsample_with_pre_tanh_value())
        # new_curr_log_pi = new_curr_actions_dist.log_prob(
        #     value=new_curr_actions, pre_tanh_value=new_curr_actions_pre_tanh)

        with torch.no_grad():
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
                [q1_rand - random_density, q1_next_actions - new_log_pis, q1_curr_actions - curr_log_pis], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis, q2_curr_actions - curr_log_pis], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1 / self._temp, dim=1,).mean() * self._min_q_weight * self._temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self._temp, dim=1,).mean() * self._min_q_weight * self._temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self._min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self._min_q_weight

        if self._with_lagrange:
            alpha_prime = torch.clamp(self._log_alpha_prime.exp(), min=0.0, max=1000000.0)

            min_qf1_loss_ = alpha_prime * (min_qf1_loss - self._target_action_gap).detach()
            min_qf2_loss_ = alpha_prime * (min_qf2_loss - self._target_action_gap).detach()

            min_qf1_loss = alpha_prime.detach() * (min_qf1_loss - self._target_action_gap)
            min_qf2_loss = alpha_prime.detach() * (min_qf2_loss - self._target_action_gap)

            self._alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss_ - min_qf2_loss_)*0.5
            alpha_prime_loss.backward()
            self._alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        return qf1_loss, qf2_loss


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
        qf1_loss.backward()
        self._qf1_optimizer.step()

        self._qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self._qf2_optimizer.step()

        # Actior loss
        action_dists = self.policy(obs)[0]
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        log_pi_new_actions = action_dists.log_prob(
                value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()


        if self._n_updates_performed < self._n_bc_steps:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k
            gradient steps here, or not having it
            """
            policy_log_prob = action_dists.log_prob(samples_data['action'])
            with torch.no_grad():
                alpha = self._get_log_alpha(samples_data).exp()
            policy_loss = (alpha * log_pi_new_actions - policy_log_prob).mean()
        else:
            policy_loss = self._actor_objective(samples_data, new_actions,
                                                log_pi_new_actions)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # if self._use_automatic_entropy_tuning:  # it comes first
        #     alpha_loss = self._temperature_objective(log_pi_new_actions,
        #                                              samples_data)
        #     self._alpha_optimizer.zero_grad()
        #     alpha_loss.backward()
        #     self._alpha_optimizer.step()

        self._n_updates_performed += 1

        return policy_loss, qf1_loss, qf2_loss


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
