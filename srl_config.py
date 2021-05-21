

def default_config(env_name,
                   algo_name='SAC',
                   h_algo_name='VPG',
                   w_algo_name='BC',
                   mode='train'):
    # without env_name and seed
    assert mode in ['train', 'test']

    # base config
    config = dict(
        algo_name=algo_name,
        discount = None,
        n_epochs = 50,
        # env_name = 'InvertedDoublePendulum-v2',
        batch_size = 10000,
        seed=1,
        # offline batch data
        data_path='',  # directory of the snapshot
        data_itr='',
        episode_batch_size = 50000,
        # pretrain policy
        warmstart_policy=w_algo_name is not None,
        w_algo_name=w_algo_name,
        w_n_epoch = 30,
        # short-horizon RL params
        lambd=1.0,
        ls_rate=1.0,
        ls_cls='TanhLS',
        use_raw_snapshot=False,
        use_heuristic=h_algo_name is not None,
        h_algo_name=h_algo_name,
        h_n_epoch=30,
        vae_loss_percentile=99,  # an interger from 0-99
        # logging
        snapshot_frequency=0,
        log_root=None,
        log_prefix='hp_tuning',
        save_mode='light',
        # optimization
        policy_lr=1e-3,       # Policy optimizer's learning rate
        value_lr=1e-3,          # Value function optimizer's learning rate
        opt_minibatch_size=128,  # optimization/replaybuffer minibatch size
        opt_n_grad_steps=1000,   # number of gradient updates
        num_evaluation_episodes=10,  # Number of evaluation episodes
        value_network_hidden_sizes=[256,256],
        policy_network_hidden_sizes=[64,64],
        n_workers=4,             # CAREFUL! Check the "conc_runs_per_node" property above. If conc_runs_per_node * n_workers > number of CPU cores on the target machine, the concurrent runs will likely interfere with each other.
        use_gpu=False,
        sampler_mode='ray',
        kl_constraint=0.05,      # kl constraint between policy updates
        gae_lambda=0.98,         # lambda of gae estimator
        lr_clip_range=0.2,
        eps_greed_decay_ratio=1.0,
        target_update_tau=5e-4,
        reward_avg_rate=1e-3,
    )

    # Provide data_path and data_itr below
    if env_name=='InvertedDoublePendulum-v2':
        # setup
        config['batch_size'] = 1000
        config['n_epochs'] = 20

        # optimization
        config['policy_lr'] = 0.00050
        config['value_lr'] = 0.00200
        config['discount'] = 0.99
        config['target_update_tau'] = 0.0400

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 30
        config['w_n_epoch'] = 30

        # srl
        if mode=='train':
            config['data_path'] = 'snapshots/SAC_Inver_1.0_F_F/120032374/'
            config['data_itr'] = [0,9]
        elif mode=='test':
            config['data_path'] = 'snapshots/SAC_Inver_1.0_F_F/640261488/'
            config['data_itr'] = [0,9]
        else:
            raise ValueError


    if env_name=='HalfCheetah-v2':
        # setup
        config['batch_size'] = 4000
        config['n_epochs'] = 50

        # optimization
        config['policy_lr'] = 0.00025 # 0.00200
        config['value_lr'] = 0.00050 # 0.00100
        config['discount'] = 0.99
        config['target_update_tau'] = 0.0400 # 0.0500

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 30
        config['w_n_epoch'] = 30

        # srl
        if mode=='train':
            # config['data_path'] = 'snapshots/SAC_HalfC_1.0_F_F/210566759/'
            config['data_path'] = 'snapshots/SAC_HalfC_1.0_None_F/967665318'
            config['data_itr'] = [0,30,2]
        elif mode=='test':
            config['data_path'] = 'snapshots/SAC_HalfC_1.0_None_F/940038543'
            config['data_itr'] = [0,30,2]
        else:
            raise ValueError

    if env_name=='Hopper-v2':
        # setup
        config['batch_size'] = 4000
        config['n_epochs'] = 100

        # optimization
        config['policy_lr'] = 0.00025
        config['value_lr'] = 0.00050
        config['discount'] = 0.999
        config['target_update_tau'] = 0.0200

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 30
        config['w_n_epoch'] = 30

        # srl
        if mode=='train':
            config['data_path'] = 'snapshots/SAC_Hoppe_1.0_None_F/646449184'
            config['data_itr'] = [0,99,4]
        elif mode=='test':
            config['data_path'] ='snapshots/SAC_Hoppe_1.0_None_F/177816424'
            config['data_itr'] = [0,99,4]
        else:
            raise ValueError


    # if env_name=='Humanoid-v2':
    #     config['data_path']= 'snapshots/SAC_Human_1.0_F_F/673933361/'
    #     config['data_itr'] = [0,200,4]
    #     config['episode_batch_size'] = config['batch_size']
    #     config['policy_network_hidden_sizes'] = [256,256]
    #     config['value_network_hidden_sizes'] = [256,256]
    #     config['n_epochs'] = 500
    #     config['h_n_epoch'] = 80
    #     config['w_n_epoch'] = 50
    #     config['value_lr'] = 1e-4
    #     config['policy_lr'] = 1e-3

    # if env_name=='Ant-v2':
    #     config['data_path']= 'snapshots/SAC_Ant-v_1.0_F_F/779696512'
    #     config['data_itr'] = [0,300,6]
    #     config['episode_batch_size'] = config['batch_size']
    #     config['policy_network_hidden_sizes'] = [256,256]
    #     config['value_network_hidden_sizes'] = [256,256]
    #     config['n_epochs'] = 500
    #     config['h_n_epoch'] = 80
    #     config['w_n_epoch'] = 50
    #     config['discount'] = 0.99  # somehow the horizon based one (0.999) doesn't work





    # if algo_name in ['DDQN', 'DQN']:
    #     # optimization
    #     config['policy_lr'] = 0.00050
    #     config['value_lr'] = 0.00200
    #     config['discount'] = 0.99
    #     config['policy_network_hidden_sizes'] = [64,64]
    #     config['value_network_hidden_sizes'] = [256,256]
    #     config['n_epochs'] = 20
    #     # batch training
    #     config['batch_size'] = 2000
    #     config['episode_batch_size'] = config['batch_size']
    #     config['h_n_epoch'] = 30
    #     config['w_n_epoch'] = 30
    #     # srl
    #     config['ls_rate'] = 1.0
    #     config['vae_loss_percentile'] = 98
    #     if mode=='train':
    #         config['data_path'] = 'snapshots/SAC_Inver_1.0_F_F/120032374/'
    #         config['data_itr'] = [0,9]
    #     elif mode=='test':
    #         config['data_path'] = 'snapshots/SAC_Inver_1.0_F_F/640261488/'
    #         config['data_itr'] = [0,9]
    #     else:
    #         raise ValueError
        # tuned hps
        # if config['h_algo_name']=='VPG':
        #     config['lambd'] = 0.93
        #     config['ls_rate'] = 1000000
        # elif config['h_algo_name']=='SAC':
        #     config['lambd'] = 0.98
        #     config['ls_rate'] = 1000000
        # elif config['h_algo_name'] is None:
        #     config['lambd'] = 0.99
        #     config['ls_rate'] = 0.30


    return config
