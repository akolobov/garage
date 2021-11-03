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

class CAC(RLAlgorithm):
    """A Conservative Actor Critic Model in Torch.


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
            target_update_tau=5e-3,# 1e-2
            policy_lr=5e-5,  # 1e-3
            qf_lr=5e-4,  # 1e-3
            alpha_lr=None,
            bc_policy_lr=None,
            reward_scale=1.0,
            optimizer=torch.optim.Adam,
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            beta=1.0,  # the regularization coefficient in front of the Bellman error
            n_bc_steps=20000,
            version=1,
            kl_constraint=0.05,
            policy_update_tau=None, # 1e-3,
            use_two_qfs=True,
            penalize_time_out=False,
            policy_lr_decay_rate=0,  # per epoch
            decorrelate_actions=False,
            terminal_value=0,
            backup_entropy=False,
            ):

        ## CAC parameters
        self._n_bc_steps = n_bc_steps
        self._n_updates_performed = 0  # counter for bc
        self._use_two_qfs = use_two_qfs
        self._penalize_time_out = penalize_time_out  # whether to penalize time out states' values
        self._policy_lr_decay_rate = policy_lr_decay_rate  # whether to sample different actions for the value and the actor loss
        self._scheduler = None  # scheduler of the policy lr
        self._decorrelate_actions = decorrelate_actions
        self._alpha_lr =  alpha_lr or qf_lr   # potentially a larger stepsize
        self._bc_policy_lr = bc_policy_lr or qf_lr  # potentially a larger stepsize
        self._policy_update_tau = policy_update_tau or target_update_tau
        self._terminal_value = terminal_value  # terminal value of of the absorbing state
        self._backup_entropy = backup_entropy  # whether to backup entropy in the q objective

        policy_lr = policy_lr or qf_lr  # use shared lr if not provided.

        # regularization constant on the Bellman error
        if beta>=0:  # use fixed reg coeff
            self._log_beta = torch.Tensor([np.log(beta)])
            self._beta_optimizer = self._bellman_target = None
        else:
            self._target_bellman_error = -beta
            self._log_beta = torch.Tensor([0]).requires_grad_()
            self._beta_optimizer = optimizer([self._log_beta],
                                              lr=self._alpha_lr)

        self._version = version
        self._target_policy = None
        if self._version==1:  # XXX Mirror Descent
            self._target_policy = copy.deepcopy(policy)
            if target_entropy is None:
                 target_entropy = -kl_constraint  # negative entropy

        # SAC parameters
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
                                                 lr=self._bc_policy_lr) #  lr=self._policy_lr)
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
                                              lr=self._alpha_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha]).log()
        self.episode_rewards = deque(maxlen=30)


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
        ## Critic Loss
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten()

        # Need to distinguish between timeout and true terminal states!
        terminals = (samples_data['terminal']==StepType.TERMINAL).float().flatten()
        timeouts =  (samples_data['terminal']==StepType.TIMEOUT ).float().flatten()

        # Bellman error
        q1_pred = self._qf1(obs, actions)
        if self._use_two_qfs:
            q2_pred = self._qf2(obs, actions)

        new_next_actions_dist = self.policy(next_obs)[0]
        new_next_actions_pre_tanh, new_next_actions = new_next_actions_dist.rsample_with_pre_tanh_value()
        with torch.no_grad():  # Compute target for regression
            target_q_values = self._target_qf1(next_obs, new_next_actions)
            if self._use_two_qfs:
                target_q_values = torch.min(target_q_values, self._target_qf2(next_obs, new_next_actions))  # no entropy term
            q_target = rewards * self._reward_scale + (1.-terminals) * self._discount * target_q_values.flatten() + terminals * self._terminal_value
            if self._backup_entropy:
                log_pi_new_next_actions = new_next_actions_dist.log_prob(
                     value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
                alpha = self._log_alpha.exp()
                q_target = q_target - alpha * log_pi_new_next_actions.flatten()

        bellman_qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        bellman_qf2_loss = F.mse_loss(q2_pred.flatten(), q_target) if self._use_two_qfs else 0.

        # Value difference
        new_actions_dist = self.policy(obs)[0]
        new_actions_pre_tanh, new_actions = new_actions_dist.rsample_with_pre_tanh_value()
        q1_new_actions = self._qf1(obs, new_actions.detach())
        min_qf1_loss = (q1_new_actions - q1_pred).mean()

        min_qf2_loss = 0.
        if self._use_two_qfs:
            q2_new_actions = self._qf2(obs, new_actions.detach())
            min_qf2_loss = (q2_new_actions - q2_pred).mean()

        if self._penalize_time_out and timeouts.sum()>0:  # Penalize timeout states
            # print('sampled {} timeout states'.format(timeouts.sum()))
            q1_new_next_actions = self._qf1(next_obs, new_next_actions.detach())
            min_qf1_loss += (q1_new_next_actions.flatten()*timeouts).mean()
            if self._use_two_qfs:
                q2_new_next_actions = self._qf2(next_obs, new_next_actions.detach())
                min_qf2_loss += (q2_new_next_actions.flatten()*timeouts).mean()

        beta = self._log_beta.exp().detach()
        qf1_loss = bellman_qf1_loss * beta + min_qf1_loss
        qf2_loss = bellman_qf2_loss * beta + min_qf2_loss

        # Autotune the regularization constant
        if self._beta_optimizer is not None:
            beta_loss = - self._log_beta * ((bellman_qf1_loss+bellman_qf2_loss).detach()/2 - self._target_bellman_error)
            self._beta_optimizer.zero_grad()
            beta_loss.backward()
            self._beta_optimizer.step()

        if self._version != 3:
            # update qfs first
            self._qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self._qf1_optimizer.step()

            if self._use_two_qfs:
                self._qf2_optimizer.zero_grad()
                qf2_loss.backward()
                self._qf2_optimizer.step()

        ## Actior Loss
        #  Perform BC for self._n_bc_steps iterations, and then CAC.
        if self._n_updates_performed==self._n_bc_steps:
            # Reset optimizers since the objective changes from BC to CAC.
            if self._use_automatic_entropy_tuning:
                self._log_alpha = torch.Tensor([self._initial_log_entropy]).requires_grad_().to(self._log_alpha.device)
                self._alpha_optimizer = self._optimizer([self._log_alpha], lr=self._alpha_lr)
            self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr)
            # Use decaying learning rate for no regret
            from torch.optim.lr_scheduler import LambdaLR
            self._scheduler = LambdaLR(self._policy_optimizer,
                                       lr_lambda=lambda i: 1.0/np.sqrt(1+i*self._policy_lr_decay_rate/self._gradient_steps))

        # Compuate entropy
        if self._decorrelate_actions:
            new_actions_dist = self.policy(obs)[0]
            new_actions_pre_tanh, new_actions = new_actions_dist.rsample_with_pre_tanh_value()
        log_pi_new_actions = new_actions_dist.log_prob(value=new_actions, pre_tanh_value=new_actions_pre_tanh)
        policy_entropy = -log_pi_new_actions.mean()

        if self._penalize_time_out and timeouts.sum()>0:
            if self._decorrelate_actions:
                new_next_actions_dist = self.policy(next_obs)[0]
                new_next_actions_pre_tanh, new_next_actions = new_next_actions_dist.rsample_with_pre_tanh_value()
            log_pi_new_next_actions = new_next_actions_dist.log_prob(
                     value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
            policy_entropy += -(log_pi_new_next_actions.flatten()*timeouts).mean()

        # Compute KL
        policy_kl = - policy_entropy  # to a uniform distribution up to a constant
        if self._version==1:  # XXX Mirror Descent
            with torch.no_grad():
                target_action_dists = self._target_policy(obs)[0]
            log_target_pi_new_actions = target_action_dists.log_prob(value=new_actions, pre_tanh_value=new_actions_pre_tanh)
            policy_kl += -log_target_pi_new_actions.mean()

            if self._penalize_time_out and timeouts.sum()>0:
                with torch.no_grad():
                    target_next_action_dists = self._target_policy(next_obs)[0]
                log_target_pi_new_next_actions = target_next_action_dists.log_prob(value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
                policy_kl += -(log_target_pi_new_next_actions.flatten()*timeouts).mean()


        alpha_loss = 0
        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
           # entropy - target = -kl - target
            alpha_loss = self._log_alpha * (-policy_kl.detach() - self._target_entropy)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        with torch.no_grad():
            alpha = self._log_alpha.exp()

        # Compute performance difference lower bound
        min_q_new_actions = self._qf1(obs, new_actions)
        if self._use_two_qfs:
            min_q_new_actions = torch.min(min_q_new_actions, self._qf2(obs, new_actions))
        lower_bound = min_q_new_actions.mean()

        if self._penalize_time_out and timeouts.sum()>0:
            min_q_new_next_actions = self._qf1(next_obs, new_next_actions)
            if self._use_two_qfs:
                min_q_new_next_actions = torch.min(min_q_new_next_actions, self._qf2(next_obs, new_next_actions))
            lower_bound += (min_q_new_next_actions.flatten()*timeouts).mean()

        if self._n_updates_performed < self._n_bc_steps: # BC warmstart
            policy_log_prob = new_actions_dist.log_prob(samples_data['action'])
            policy_loss = - policy_log_prob.mean() + alpha * policy_kl
        else:
            policy_loss = - lower_bound + alpha * policy_kl

        if self._version != 3:
            self._policy_optimizer.zero_grad()
            policy_loss.backward()
            self._policy_optimizer.step()
            self._scheduler.step() if self._scheduler is not None else None
        else:
            # update qf and policy together
            self._policy_optimizer.zero_grad()
            policy_loss.backward()
            self._policy_optimizer.step()

            self._qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self._qf1_optimizer.step()

            if self._use_two_qfs:
                self._qf2_optimizer.zero_grad()
                qf2_loss.backward()
                self._qf2_optimizer.step()

            self._scheduler.step() if self._scheduler is not None else None


        # For logging
        grad_norm = 0
        for param in self.policy.parameters():
            grad_norm += torch.sum(param.grad**2) if param.grad is not None else 0.
        policy_lr = self._scheduler.get_last_lr()[0] if self._scheduler is not None else self._bc_policy_lr

        q_new_actions_mean = (q1_new_actions+q2_new_actions).mean()/2
        q_pred_mean = (q1_pred+q2_pred).mean()/2

        self._n_updates_performed += 1

        log_info = dict(
                    policy_loss=policy_loss,
                    qf1_loss=qf1_loss,
                    qf2_loss=qf2_loss,
                    bellman_qf1_loss=bellman_qf1_loss,
                    min_qf1_loss=min_qf1_loss,
                    bellman_qf2_loss=bellman_qf2_loss,
                    min_qf2_loss=min_qf2_loss,
                    beta=beta,
                    beta_loss=beta_loss,
                    alpha_loss=alpha_loss,
                    policy_kl=policy_kl,
                    policy_entropy=policy_entropy,
                    alpha=alpha,
                    lower_bound=lower_bound,
                    reward_scale=self._reward_scale,
                    policy_lr=policy_lr,
                    grad_norm=grad_norm,
                    q_new_actions_mean=q_new_actions_mean,
                    q_pred_mean=q_pred_mean,
                    )

        return log_info

    # Update also the target policy if needed
    def _update_targets(self):
        """Update parameters in the target q-functions."""
        if self._use_two_qfs:
            target_qfs = [self._target_qf1, self._target_qf2]
            qfs = [self._qf1, self._qf2]
        else:
            target_qfs = [self._target_qf1]
            qfs = [self._qf1]

        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                                   param.data * self._tau)

        if self._version==1:  # XXX Mirror Descent
            for t_param, param in zip(self._target_policy.parameters(), self.policy.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._policy_update_tau) +
                                   param.data * self._policy_update_tau)

    # Set also the target policy if needed
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
                                                    lr=self._alpha_lr)
        if self._beta_optimizer is None:
            self._log_beta = torch.Tensor([self._log_beta]).to(device)
        else:
            self._log_beta = torch.Tensor([self._log_beta]).to(device).requires_grad_()
            self._beta_optimizer = optimizer([self._log_beta], lr=self._alpha_lr)


        if self._version==1:  # XXX Mirror Descent
            self._target_policy.to(device)

    # Return also the target policy if needed
    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        if self._use_two_qfs:
            networks = [
                self.policy, self._qf1, self._qf2, self._target_qf1,
                self._target_qf2
            ]
        else:
            networks = [
                self.policy, self._qf1, self._target_qf1
            ]

        if self._version==1:  # XXX Mirror Descent
            networks.append(self._target_policy)

        return networks

    # Evaluate both the deterministic and the stochastic policies
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
        if self._use_deterministic_evaluation:
            # XXX Also log the stochastic policy's performance
            eval_episodes_ = obtain_evaluation_episodes(
                                self.policy,
                                self._eval_env,
                                self._max_episode_length_eval,
                                num_eps=self._num_evaluation_episodes,
                                deterministic=False)
            log_performance(epoch,
                            eval_episodes_,
                            discount=self._discount,
                            prefix='Exploration')

        return last_return


    # Below is overwritten for general logging with log_info
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
                        # TODO need to update the terminal format
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
                    log_info = self.train_once()
            if self._num_evaluation_episodes>0:
                last_return = self._evaluate_policy(trainer.step_itr)
            self._log_statistics(log_info)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            trainer.step_itr += 1

        return np.mean(last_return) if last_return is not None else 0

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
            log_info = self.optimize_policy(samples)
            self._update_targets()

        return log_info

    def _log_statistics(self, log_info):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        for k, v in log_info.items():
            tabular.record('Algorithm/'+k, float(v))
        tabular.record('ReplayBuffer/buffer_size',
                       self.replay_buffer.n_transitions_stored)
