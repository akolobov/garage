# To set up a job, the user needs to specify `method` and `hp_dict` and then
# import `submit_xt_job`. Optionally, the user can provide additional arguments
# `submit_xt_job` takes. To run the job, go to the `rl_nexus` directory and then
# execute this python file.

from rl_nexus.hp_tuning_tools import submit_xt_job


# hp compute
max_total_runs=2000
n_concurrent_runs_per_node=4
max_n_nodes=1000
compute_target='azb-cpu'
azure_service='rdlbatches' #'rdlbatches' # 'rdlbatches' # dilbertbatch'
gpu_id=-1

# algo config
torch_n_threads=2
log_root ='../results'

method = 'rl_nexus.garage.examples.offline_rl_example.run'

def run(n_epochs=3000,
        hp_tuning_mode='grid',
        n_seeds_per_hp=3,
        env_name='hopper-medium-v0',  # or a list
        env_yaml=None):

    # # Experiments on many gym envs
    # hps_dict = dict(
    #         policy_lr=[1e-5, 5e-6, 1e-6],
    #         value_lr=[5e-4],
    #         target_update_tau=[5e-3],
    #         min_q_weight = [1, 5, 10],
    #         policy_lr_decay_rate=[0],
    #         penalize_time_out=[False],
    #         discount=[0.99, 0.998],
    #         version=[0],
    # )

    # # Experiments on androit and kitchen
    # hps_dict = dict(
    #         policy_lr=[1e-5],
    #         value_lr=[5e-4, 5e-5], #, 5e-6],
    #         target_update_tau=[5e-3, 5e-4], #, 5e-5], #[5e-3],
    #         min_q_weight = [1, 5, 10, 50, 100, 500, 1000], #[0.001, 0.01, 0.05, 0.1, 0.5],#[1, 5, 10, 50, 100, 500, 1000],
    #         policy_lr_decay_rate=[0],
    #         penalize_time_out=[False],
    #         discount=[0.5, 0.9, 0.99],
    #         version=[0],
    # )


    # Experiments on androit and kitchen
    hps_dict = dict(
            policy_lr=[0e-5],
            value_lr=[5e-4, 5e-5], #, 5e-6],
            target_update_tau=[5e-3, 5e-4], #, 5e-5], #[5e-3],
            beta = [0.001, 0.01, 1, 10, 100],
            policy_lr_decay_rate=[0],
            penalize_time_out=[False],
            discount=[0.5, 0.9, 0.99],
            version=[0],
    )

    if env_yaml is not None:
        import yaml
        with open(env_yaml, 'r') as f:
            env_name = yaml.safe_load(f)
    if type(env_name) is str:
        env_name = [env_name]
    hps_dict['env_name'] = env_name

    config = dict(
        log_root=log_root,
        algo='CAC',
        env_name='hopper-medium-v0',
        # Trainer parameters
        n_epochs=n_epochs,  # number of training epochs
        # batch_size=0,  # number of samples collected per update
        # replay_buffer_size=int(2e6),
        # Network parameters
        policy_hidden_sizes=[256, 256, 256],
        # policy_hidden_nonlinearity=torch.nn.ReLU,
        policy_init_std=1.0,
        value_hidden_sizes=[256, 256, 256],
        # value_hidden_nonlinearity=torch.nn.ReLU,
        # Algorithm parameters
        discount=0.99,
        policy_lr=5e-5,  # optimization stepsize for policy update
        value_lr=5e-4,  # optimization stepsize for value regression
        target_update_tau=5e-3, # for target network
        minibatch_size=256,  # optimization/replaybuffer minibatch size
        # n_grad_steps=1000,  # number of gradient updates per epoch
        # steps_per_epoch=1,  # number of internal epochs steps per epoch
        n_bc_steps=20000,
        fixed_alpha=None,
        use_two_qfs=True,
        use_deterministic_evaluation=True,
        num_evaluation_episodes=5, # number of episodes to evaluate (only affect off-policy algorithms)
        # CQL parameters
        lagrange_thresh=5.0,
        min_q_weight=1.0,
        # CAC parameters
        beta=1.0,
        version=1,
        kl_constraint=0.05,
        policy_update_tau=None, # for the policy.
        # Compute parameters
        seed='randint',
        n_workers=1,  # number of workers for data collection
        gpu_id=gpu_id,  # try to use gpu, if implemented
        force_cpu_data_collection=True,  # use cpu for data collection.
        torch_n_threads=torch_n_threads
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
          "cp -r .mujoco/mujoco200_linux  .mujoco/mujoco210",   # error: Header file '/root/.mujoco/mujoco210/include/mjdata.h' does not exist.
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
                  max_total_runs=max_total_runs,
                  max_n_nodes=max_n_nodes,
                  azure_service=azure_service,
                  )

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    # parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--hp_tuning_mode', type=str, default='grid')
    parser.add_argument('--n_seeds_per_hp', type=int, default=3)
    parser.add_argument('--env_name', type=str, nargs='+', default='hopper-medium-v0')
    parser.add_argument('--env_yaml', type=str, default=None)

    run(**vars(parser.parse_args()))