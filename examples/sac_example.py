
import gym, d4rl, torch, os
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.replay_buffer import PathBuffer
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.tools.rl_utils import train_agent, get_sampler, setup_gpu
from garage.tools.trainer import Trainer

def train_func(ctxt=None,
               *,
               # Environment parameters
               env_name,
               discount,  # original discount
               # Trainer parameters
               n_epochs=3000,  # number of training epochs
               batch_size=4000,  # number of samples collected per update
               # Network parameters
               policy_hidden_sizes=(256, 265),
               policy_hidden_nonlinearity=torch.nn.ReLU,
               policy_init_std=1.0,
               value_hidden_sizes=(256, 265),
               value_hidden_nonlinearity=torch.nn.ReLU,
               # Optimization parameters
               policy_lr=3e-4,  # optimization stepsize for policy update
               value_lr=3e-4,  # optimization stepsize for value regression
               target_update_tau=5e-3, # for target network
               minibatch_size=256,  # optimization/replaybuffer minibatch size
               n_grad_steps=1000,  # number of gradient updates per epoch
               steps_per_epoch=1,  # number of internal epochs steps per epoch
               # Evaluation
               use_deterministic_evaluation=True,
               num_evaluation_episodes=10, # number of episodes to evaluate (only affect off-policy algorithms)
               # Compute parameters
               seed=0,
               n_workers=1,  # number of workers for data collection
               gpu_id=-1,  # try to use gpu, if implemented
               force_cpu_data_collection=True,  # use cpu for data collection.
               sampler_mode='ray',
               # Logging parameters
               save_mode='full',
               ignore_shutdown=False,  # do not shutdown workers after training
               return_mode='average', # 'full', 'average', 'last'
               return_attr='Evaluation/AverageReturn',  # the log attribute
               ):

    """ Train an agent in batch mode. """

    # Set the random seed
    set_seed(seed)

    # Initialize gym env
    _env = gym.make(env_name)
    env = GymEnv(_env)

    # Initialize the algorithm
    env_spec = env.spec

    policy = TanhGaussianMLPPolicy(
                env_spec=env_spec,
                hidden_sizes=policy_hidden_sizes,
                hidden_nonlinearity=policy_hidden_nonlinearity,
                init_std=policy_init_std)

    qf1 = ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=value_hidden_sizes,
                hidden_nonlinearity=value_hidden_nonlinearity,
                output_nonlinearity=None)

    qf2 = ContinuousMLPQFunction(
                env_spec=env_spec,
                hidden_sizes=value_hidden_sizes,
                hidden_nonlinearity=value_hidden_nonlinearity,
                output_nonlinearity=None)

    replay_buffer = PathBuffer(capacity_in_transitions=int(2e6))

    sampler = get_sampler(policy, env, n_workers=n_workers)

    algo = SAC(env_spec=env_spec,
               policy=policy,
               qf1=qf1,
               qf2=qf2,
               sampler=sampler,
               gradient_steps_per_itr=n_grad_steps,
               replay_buffer=replay_buffer,
               min_buffer_size=int(0),
               target_update_tau=target_update_tau,
               discount=discount,
               buffer_batch_size=minibatch_size,
               steps_per_epoch=steps_per_epoch,
               num_evaluation_episodes=num_evaluation_episodes,
               policy_lr=policy_lr,
               qf_lr=value_lr,
               use_deterministic_evaluation=use_deterministic_evaluation)

    setup_gpu(algo, gpu_id=gpu_id)


    # Initialize the trainer
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


if __name__=='__main__':
    torch.set_num_threads(4)
    train_kwargs = dict(env_name='Hopper-v2',
                        discount=0.99,
                        batch_size=2000,
                        gpu_id=-1,
                        force_cpu_data_collection=True,
                        n_workers=1)
    train_agent(train_func,
                log_dir=os.path.join('./data','SAC'),
                train_kwargs=train_kwargs)
