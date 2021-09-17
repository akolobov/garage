# To set up a job, the user needs to specify `method` and `hp_dict` and then
# import `submit_xt_job`. Optionally, the user can provide additional arguments
# `submit_xt_job` takes. To run the job, go to the `rl_nexus` directory and then
# execute this python file.

from rl_nexus.hp_tuning_tools import submit_xt_job

log_root ='../results'
hp_tuning_mode='grid'
n_concurrent_runs_per_node=2
n_seeds_per_hp=4

method = 'rl_nexus.garage.examples.offline_rl_example.run'

hps_dict = dict(
            # discount=[0.99],
            policy_lr=[1e-3, 1e-4, 1e-5],
            value_lr=[3e-3, 3e-4],
            policy_update_tau=[5e-1, 5e-2, 5e-3],
            policy_update_version=[0, 1],
            kl_constraint=[0.1, 1.0],
            )

config = dict(
    log_root=log_root,
    algo='CAC',
    env_name='hopper-medium-v0',
    # Trainer parameters
    n_epochs=1000,  # number of training epochs
    # batch_size=0,  # number of samples collected per update
    # replay_buffer_size=int(2e6),
    # Network parameters
    policy_hidden_sizes=(256, 256, 256),
    # policy_hidden_nonlinearity=torch.nn.ReLU,
    policy_init_std=1.0,
    value_hidden_sizes=(256, 256, 256),
    # value_hidden_nonlinearity=torch.nn.ReLU,
    # Algorithm parameters
    discount=0.99,
    policy_lr=1e-4,  # optimization stepsize for policy update
    value_lr=3e-4,  # optimization stepsize for value regression
    target_update_tau=5e-3, # for target network
    minibatch_size=256,  # optimization/replaybuffer minibatch size
    # n_grad_steps=1000,  # number of gradient updates per epoch
    # steps_per_epoch=1,  # number of internal epochs steps per epoch
    n_bc_steps=10000,
    fixed_alpha=None,
    use_deterministic_evaluation=True,
    num_evaluation_episodes=10, # number of episodes to evaluate (only affect off-policy algorithms)
    # CQL parameters
    lagrange_thresh=5.0,
    min_q_weight=1.0,
    # CAC parameters
    policy_update_version=1,
    kl_constraint=0.1,
    policy_update_tau=5e-3, # for the policy.
    # Compute parameters
    seed='randint',
    n_workers=2,  # number of workers for data collection
    gpu_id=-1,  # try to use gpu, if implemented
    force_cpu_data_collection=True,  # use cpu for data collection.
)

xt_setup = {
  'activate':None,
  'other-cmds':[
      "sudo apt-get install -y patchelf",
      "export MUJOCO_PY_MJKEY_PATH=/opt/mjkey.txt",
      "export MUJOCO_PY_MUJOCO_PATH=/opt/mujoco200_linux",
      "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mujoco200_linux/bin",
      "cd /root",
      "ln -s /opt .mujoco",  # for d4rl # error: Header file '/root/.mujoco/mujoco200_linux/include/mjdata.h' does not exist./
      "cd -",  # dilbert directory
      "pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl",  # d4rl
      "cd rl_nexus",
      "git clone https://github.com/akolobov/garage.git",
      "cd garage",
      "git checkout conservative_ac",
      "pip install -e '.[all,dev]'",
      "cd ../../",   # dilbert directory
      ],
  'conda-packages': [],
  'pip-packages': [],
  'python-path': ["../"]
}

submit_xt_job(method,
              hps_dict,
              config=config,
              n_concurrent_runs_per_node=n_concurrent_runs_per_node,
              xt_setup=xt_setup,
              hp_tuning_mode=hp_tuning_mode,
              n_seeds_per_hp=n_seeds_per_hp,
              max_n_nodes=1
              )