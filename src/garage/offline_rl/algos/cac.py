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

torch.set_flush_denormal(True)

def normalized_sum(loss, reg, w):
    # loss + w * reg
    if w>1:
        return loss/w + reg
    else:
        return loss + w*reg


def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint>0:
            w = module.weight
            norm = torch.norm(w)
            w.mul_(torch.clip(constraint/norm, max=1))
    return fn

def weight_l2(model):
    l2 = 0.
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2 += torch.norm(param)**2
    return l2

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
            buffer_batch_size=256,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=5e-5,
            qf_lr=5e-4,
            reward_scale=1.0,
            optimizer='Adam',
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            # CAC parameters
            n_warmstart_steps=20000,
            max_n_warmstart_steps=200000*3,
            beta=-1.0,  # the regularization coefficient in front of the Bellman error
            n_qf_steps=1,
            norm_constraint=100,
            terminal_value=None,
            use_two_qfs=True,
            stats_avg_rate=0.99, # for logging
            q_eval_mode='0.5_0.5', # 'max' 'w1_w2', 'adaptive'
            cons_inc_rate=0.0,  # XXX deprecated
            weigh_dist=False,  # XXX deprecated
            q_eval_loss='MSELoss', # 'MSELoss', 'SmoothL1Loss'
            beta_upper_bound=1e6,  # for numerical stability
            init_q_eval_mode='0.5_0.5',
            ):

        #############################################################################################

        # Parsing
        n_qf_steps = max(1, n_qf_steps)
        if 'Adam' in optimizer and '_' in optimizer:
            optimizer_name, beta1, beta2 = optimizer.split('_')
            from functools import partial
            optimizer = partial(torch.optim.Adam, betas=(float(beta1), float(beta2)))
        else:
            optimizer = eval('torch.optim.'+optimizer)

        ## CAC parameters
        self._n_warmstart_steps = n_warmstart_steps
        self._max_n_warmstart_steps = max_n_warmstart_steps
        self._n_qf_steps = n_qf_steps
        self._n_updates_performed = 0 # Counter of number of grad steps performed
        self._cac_learning=False
        self._norm_constraint = norm_constraint
        self._stats_avg_rate = stats_avg_rate
        self._weigh_dist = weigh_dist
        self._q_eval_loss = eval('torch.nn.'+q_eval_loss)(reduction='none')
        if q_eval_loss=='SmoothL1Loss':
            _q_eval_loss = self._q_eval_loss
            self._q_eval_loss = lambda *args, **kwargs : _q_eval_loss(*args, **kwargs)*2.0  # so it matches the unit in the MSE loss.
        self._q_eval_mode = [float(w) for w in q_eval_mode.split('_')] if '_' in q_eval_mode else  q_eval_mode
        self._q_eval_mode_desired =  self._q_eval_mode  #bkp
        self._init_q_eval_mode = [float(w) for w in q_eval_mode.split('_')] if '_' in init_q_eval_mode else  init_q_eval_mode
        # terminal value of of the absorbing state
        self._terminal_value = terminal_value if terminal_value is not None else lambda r, gamma: 0.

        # Stepsizes
        policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.
        self._alpha_lr = qf_lr # potentially a larger stepsize, for the most inner optimization.
        self._beta_lr = qf_lr
        self._bc_policy_lr = qf_lr  # potentially a larger stepsize
        self._use_two_qfs = use_two_qfs

        # Regularization constant on the Bellman error
        self._avg_bellman_error = 1.  # for logging; so this works with zero warm-start
        if beta>=0:  # use fixed reg coeff
            self._log_beta1 = self._log_beta2 = torch.Tensor([np.log(beta)])
            self._beta_optimizer = self._bellman_target = None
        else:
            self._bellman_constraint = -beta
            self._init_log_beta = np.log(1.0) # i.e. beta=1
            self._log_beta1 = torch.Tensor([self._init_log_beta]).requires_grad_()
            self._log_beta2 = torch.Tensor([self._init_log_beta]).requires_grad_()
            self._beta_optimizer = optimizer([self._log_beta1, self._log_beta2], lr=self._beta_lr)
            self._log_beta_upper_bound = np.log(beta_upper_bound)  # threshold to double the norm constraint
            self._cons_inc_rate = cons_inc_rate

        if self._q_eval_mode == 'adaptive':
            assert self._beta_optimizer is not None, "Adaptive q_eval_mode can only be used in the constrained version."  # beta needs to be tuned too
            self._log_Beta1 = torch.Tensor([self._init_log_beta]).requires_grad_()
            self._log_Beta2 = torch.Tensor([self._init_log_beta]).requires_grad_()
            self._Beta_optimizer = optimizer([self._log_Beta1, self._log_Beta2], lr=self._beta_lr)
            self._log_Beta_upper_bound = np.log(beta_upper_bound)  # threshold to double the norm cons
        else:
            self._log_Beta1 = self._log_Beta2 = torch.Tensor([np.log(0.)])  # not used
            self._Beta_optimizer = None

        #############################################################################################
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


    def optimize_policy(self,
                        samples_data,
                        warmstart=False,
                        qf_update_only=False):
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
        next_obs = samples_data['next_observation']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten() * self._reward_scale
        terminals =  samples_data['terminal'].flatten()
        timeouts =  samples_data['timeout'].flatten()
        timesteps =  samples_data['timestep'].flatten()

        ##### Update Critic #####

        # Bellman error
        def compute_target(q_pred_next):
            assert rewards.shape == q_pred_next.shape
            return rewards + (1.-terminals) * self._discount * q_pred_next + terminals * self._terminal_value(rewards, self._discount)

        def compute_mixed_bellman_loss(q_pred, q_pred_next, q_target):
            assert q_pred.shape == q_pred_next.shape == q_target.shape
            q_target_pred = compute_target(q_pred_next)
            target_error = self._q_eval_loss(q_pred, q_target)
            td_error = self._q_eval_loss(q_pred, q_target_pred)
            mean_target_error = target_error.mean()
            mean_td_error = td_error.mean()
            if self._q_eval_mode=='max':
                loss = torch.max(target_error, td_error).mean()
            elif self._q_eval_mode=='bmax':
                loss = torch.max(mean_target_error, mean_td_error)
            elif self._q_eval_mode=='adaptive':
                loss = None  # will compute it later
            else:
                w1, w2 = self._q_eval_mode
                loss = w1*mean_target_error+ w2*mean_td_error

            return loss, mean_target_error, mean_td_error

        with torch.no_grad():  # compute target for regression
            new_next_actions_dist = self.policy(next_obs)[0]
            new_next_actions_pre_tanh, new_next_actions = new_next_actions_dist.rsample_with_pre_tanh_value()
            target_q_values = self._target_qf1(next_obs, new_next_actions)
            if self._use_two_qfs:
                target_q_values = torch.min(target_q_values, self._target_qf2(next_obs, new_next_actions))
            q_target = compute_target(target_q_values.flatten())

        q1_pred = self._qf1(obs, actions).flatten()
        q1_pred_next = self._qf1(next_obs, new_next_actions).flatten()
        bellman_qf1_loss, q1_target_error, q1_td_error = compute_mixed_bellman_loss(q1_pred, q1_pred_next, q_target)

        bellman_qf2_loss = q2_target_error = q2_td_error = torch.Tensor([0.])
        if self._use_two_qfs:
            q2_pred = self._qf2(obs, actions).flatten()
            q2_pred_next = self._qf2(next_obs, new_next_actions).flatten()
            bellman_qf2_loss, q2_target_error, q2_td_error = compute_mixed_bellman_loss(q2_pred, q2_pred_next, q_target)

        if not qf_update_only or not warmstart:
            # These samples will be used for the actor update too, so they need to be traced.
            new_actions_dist = self.policy(obs)[0]
            new_actions_pre_tanh, new_actions = new_actions_dist.rsample_with_pre_tanh_value()

        beta_loss = gan_qf1_loss = gan_qf2_loss = 0
        if not warmstart:  # Compute beta_loss, gan_qf1_loss, gan_qf2_loss
            dist_weight = (1-self._discount**(timesteps+1)).reshape(-1,1) if self._weigh_dist else torch.ones_like(q1_pred)
            # Compute value difference
            q1_new_actions = self._qf1(obs, new_actions.detach())
            gan_qf1_loss = ((q1_new_actions - q1_pred)*dist_weight).mean()
            if self._use_two_qfs:
                q2_new_actions = self._qf2(obs, new_actions.detach())
                gan_qf2_loss = ((q2_new_actions - q2_pred)*dist_weight).mean()

            # Autotune the regularization constant to satisfy Bellman constraint
            if self._beta_optimizer is not None:
                beta_loss = - self._log_beta1 * (q1_td_error.detach() - self._bellman_constraint)
                if self._use_two_qfs:
                    beta_loss += - self._log_beta2 * (q2_td_error.detach() - self._bellman_constraint)
                self._beta_optimizer.zero_grad()
                beta_loss.backward()
                self._beta_optimizer.step()
                with torch.no_grad():
                    self._log_beta1.clamp_(max=self._log_beta_upper_bound)
                    self._log_beta2.clamp_(max=self._log_beta_upper_bound)

        with torch.no_grad():
            beta1 = self._log_beta1.exp()
            beta2 = self._log_beta2.exp()

        # # # Doubling the norm constraint if beta is too large (i.e. infeasible)
        # if self._beta_optimizer is not None and beta >= self._beta_upper_bound and self._cons_inc_rate>0:
        #     self._bellman_constraint *= 1+self._cons_inc_rate
        #     self._log_beta = torch.Tensor([self._init_log_beta]).requires_grad_()
        #     self._beta_optimizer = self._optimizer([self._log_beta], lr=self._beta_lr)

        # Prevent exploding gradient due to auto tuning
        # min_qf_loss + beta * bellman_qf_loss
        # qf1_loss = gan_qf1_loss + beta * bellman_qf1_loss
        # qf2_loss = gan_qf2_loss + beta * bellman_qf2_loss
        if self._q_eval_mode == 'adaptive':
            w_loss = - self._log_Beta1 * (q1_target_error.detach() - self._bellman_constraint)
            if self._use_two_qfs:
                w_loss += - self._log_Beta2 * (q2_target_error.detach() - self._bellman_constraint)
            self._Beta_optimizer.zero_grad()
            w_loss.backward()
            self._Beta_optimizer.step()
            with torch.no_grad():
                self._log_Beta1.clamp_(max=self._log_Beta_upper_bound)
                self._log_Beta2.clamp_(max=self._log_Beta_upper_bound)
            with torch.no_grad():
                Beta1 = self._log_Beta1.exp()
                Beta2 = self._log_Beta2.exp()
            bellman_qf1_loss = beta1 * q1_td_error + Beta1 * q1_target_error  # logging
            bellman_qf2_loss = beta2 * q2_td_error + Beta2 * q2_target_error
            qf1_loss = (gan_qf1_loss + bellman_qf1_loss) / torch.max(torch.Tensor([1, Beta1, beta1]))
            qf2_loss = gan_qf2_loss + bellman_qf2_loss / torch.max(torch.Tensor([1, Beta2, beta2]))
        else:
            qf1_loss = normalized_sum(gan_qf1_loss, bellman_qf1_loss, beta1)
            qf2_loss = normalized_sum(gan_qf2_loss, bellman_qf2_loss, beta2)

        # L2 regularization on weights not bias
        if self._norm_constraint<0:
            qf1_loss += -self._norm_constraint * weight_l2(self._qf1)
            qf2_loss += -self._norm_constraint * weight_l2(self._qf2)

        if beta1>0 or not warmstart:
            # no warmup for beta=0; otherwise, numerical singulartiy might happen due to weight decay.
            self._qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self._qf1_optimizer.step()
            self._qf1.apply(l2_projection(self._norm_constraint))

            if self._use_two_qfs:
                self._qf2_optimizer.zero_grad()
                qf2_loss.backward()
                self._qf2_optimizer.step()
                self._qf2.apply(l2_projection(self._norm_constraint))

        if qf_update_only:
            return

        ##### Update Actor #####

        # Compuate entropy
        log_pi_new_actions = new_actions_dist.log_prob(value=new_actions, pre_tanh_value=new_actions_pre_tanh)
        policy_entropy = -log_pi_new_actions.mean()
        policy_kl = -policy_entropy  # to a uniform distribution up to a constant

        alpha_loss = 0
        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
           # entropy - target = -kl - target
            alpha_loss = self._log_alpha * (-policy_kl.detach() - self._target_entropy)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        with torch.no_grad():
            alpha = self._log_alpha.exp()

        lower_bound = 0
        if warmstart: # BC warmstart
            policy_log_prob = new_actions_dist.log_prob(samples_data['action'])
            # policy_loss = - policy_log_prob.mean() + alpha * policy_kl
            policy_loss = normalized_sum(-policy_log_prob.mean(), policy_kl, alpha)
        else:
            # Compute performance difference lower bound
            min_q_new_actions = self._qf1(obs, new_actions)
            # if self._use_two_qfs:
            #     min_q_new_actions = torch.min(min_q_new_actions, self._qf2(obs, new_actions))
            lower_bound = min_q_new_actions.mean()
            # policy_loss = - lower_bound + alpha * policy_kl
            policy_loss = normalized_sum(-lower_bound, policy_kl, alpha)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # For logging
        with torch.no_grad():
            # bellman_qf_loss = torch.max(bellman_qf1_loss, bellman_qf2_loss)  # for logging
            bellman_qf_loss = torch.max(q1_td_error, q2_td_error)  # measure the TD error
            self._avg_bellman_error = self._avg_bellman_error*self._stats_avg_rate + bellman_qf_loss*(1-self._stats_avg_rate)
            policy_grad_norm = 0
            for param in self.policy.parameters():
                policy_grad_norm += torch.sum(param.grad**2) if param.grad is not None else 0.
            q1_pred_mean = q1_pred.mean()
            q2_pred_mean = q2_pred.mean() if self._use_two_qfs else 0.
            q1_new_actions_mean = q1_new_actions.mean() if not warmstart else 0.
            q2_new_actions_mean = q2_new_actions.mean() if not warmstart and self._use_two_qfs else 0.
            action_diff = torch.mean(torch.norm(samples_data['action'] - new_actions, dim=1))
            new_actions_norm = torch.norm(new_actions)

            # # Debug
            # qf1_norms = []
            # for param in self._qf1.parameters():
            #     qf1_norms.append(torch.norm(param).detach().numpy())

            # if self._use_two_qfs:
            #     qf2_norms = []
            #     for param in self._qf2.parameters():
            #         qf2_norms.append(torch.norm(param).detach().numpy())


        log_info = dict(
                    policy_loss=policy_loss,
                    qf1_loss=qf1_loss,
                    qf2_loss=qf2_loss,
                    bellman_qf1_loss=bellman_qf1_loss,
                    gan_qf1_loss=gan_qf1_loss,
                    bellman_qf2_loss=bellman_qf2_loss,
                    gan_qf2_loss=gan_qf2_loss,
                    beta1=beta1,
                    beta2=beta2,
                    beta_loss=beta_loss,
                    alpha_loss=alpha_loss,
                    policy_entropy=policy_entropy,
                    alpha=alpha,
                    lower_bound=lower_bound,
                    reward_scale=self._reward_scale,
                    policy_grad_norm=policy_grad_norm,
                    q1_new_actions_mean=q1_new_actions_mean,
                    q2_new_actions_mean=q2_new_actions_mean,
                    q1_pred_mean=q1_pred_mean,
                    q2_pred_mean=q2_pred_mean,
                    action_diff=action_diff,
                    new_actions_norm=new_actions_norm,
                    avg_bellman_error=self._avg_bellman_error,
                    q_target=q_target.mean(),
                    norm_constriant=self._norm_constraint,
                    q1_target_error=q1_target_error,
                    q1_td_error=q1_td_error,
                    q2_target_error=q2_target_error,
                    q2_td_error=q2_td_error,
                    bellman_constraint=self._bellman_constraint,
                    )
        if self._Beta_optimizer is not None:
            log_info['Beta1'] = Beta1
            log_info['Beta2'] = Beta2

        # # Debug
        # for n, norm in enumerate(qf1_norms):
        #     log_info['qf1_norm_'+str(n)] = norm
        # if self._use_two_qfs:
        #     for n, norm in enumerate(qf2_norms):
        #         log_info['qf2_norm_'+str(n)] = norm

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
            self._log_beta1 = torch.Tensor([self._log_beta2]).to(device)
            self._log_beta1 = torch.Tensor([self._log_beta2]).to(device)
        else:
            self._log_beta1 = torch.Tensor([self._log_beta1]).to(device).requires_grad_()
            self._log_beta2 = torch.Tensor([self._log_beta2]).to(device).requires_grad_()
            self._beta_optimizer = optimizer([self._log_beta1, self._log_beta2], lr=self._beta_lr)

        if self._Beta_optimizer is None:
            self._log_Beta1 = torch.Tensor([self._log_Beta2]).to(device)
            self._log_Beta1 = torch.Tensor([self._log_Beta2]).to(device)
        else:
            self._log_Beta1 = torch.Tensor([self._log_Beta1]).to(device).requires_grad_()
            self._log_Beta2 = torch.Tensor([self._log_Beta2]).to(device).requires_grad_()
            self._Beta_optimizer = optimizer([self._log_Beta1, self._log_Beta2], lr=self._beta_lr)


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

            warmstart = self._n_updates_performed<self._n_warmstart_steps
            if self._beta_optimizer is not None and not warmstart and not self._cac_learning: # a proposal to turn off warmstart is made
                # for constrained version, we need to check if we need to run the warmstart longer to satisfy the constraint.
                warmstart = self._avg_bellman_error >= self._bellman_constraint \
                             or self._n_updates_performed <= 10/(1-self._stats_avg_rate)  # time for self._avg_bellman_error to be meaningful
                if warmstart and self._n_updates_performed >=self._max_n_warmstart_steps:
                    warmstart = False  # reached the maximum updates
                    self._bellman_constraint=self._avg_bellman_error

            if warmstart:
                self._q_eval_mode = self._init_q_eval_mode
            else:
                self._q_eval_mode = self._q_eval_mode_desired

            if not warmstart and not self._cac_learning:  # self._n_updates_performed==self._n_warmstart_steps:
                self._cac_learning = True
                # Reset optimizers since the objective changes
                if self._use_automatic_entropy_tuning:
                    self._log_alpha = torch.Tensor([self._initial_log_entropy]).requires_grad_().to(self._log_alpha.device)
                    self._alpha_optimizer = self._optimizer([self._log_alpha], lr=self._alpha_lr)
                self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr)
                # self._qf1_optimizer = self._optimizer(self._qf1.parameters(),lr=self._qf_lr)
                # self._qf2_optimizer = self._optimizer(self._qf2.parameters(),lr=self._qf_lr)


            n_qf_steps = 1 if warmstart else self._n_qf_steps
            for i in range(n_qf_steps):
                samples = self.replay_buffer.sample_transitions(self._buffer_batch_size)
                samples = as_torch_dict(samples)
                log_info = self.optimize_policy(samples,
                                warmstart=warmstart,
                                qf_update_only=i<n_qf_steps-1)
                self._update_targets()


            self._n_updates_performed += 1
            log_info['n_updates_performed']=self._n_updates_performed
            log_info['warmstart'] = warmstart
            log_info['n_qf_steps'] = n_qf_steps
            # print(self._n_updates_performed )

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
