

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
        load_pretrained_data=False,
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
        reward_shaping_mode='hurl',
        reward_scale=1.0
    )

    # HACK Sparse Reacher Environment
    if env_name=='Reacher-v2':
        # setup
        config['batch_size'] = 10000
        config['n_epochs'] = 200


        # optimization run3001.172 (for thres 0.01)
        config['policy_lr'] = 0.00025
        config['value_lr'] =  0.00025
        config['discount'] = 0.9
        config['target_update_tau'] = 0.0200

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['warmstart_policy'] = False

        if config['h_algo_name']=='HACK':
            config['lambd'] = 0.5
            config['ls_rate'] =  100000.0

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
        config['n_epochs'] = 200

        # optimization run1823.49
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
            # config['data_path'] = 'snapshots/SAC_HalfC_1.0_None_F/967665318'  # default
            config['data_path'] = 'snapshots/SAC_HalfC_1.0_None_F_200/786495378/'
            config['data_itr'] = [0,199,4]
        elif mode=='test':
            config['data_path'] = 'snapshots/SAC_HalfC_1.0_None_F_200/786495378/'
            config['data_itr'] = [0,199,4]
        else:
            raise ValueError


        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.99 100000.0 78 4433.1 10 run2347.91
            config['lambd'] = 0.99
            config['ls_rate'] =  100000
        elif config['h_algo_name'] == 'VPG':
            # 0.99 100000.000000 48 4864.0 8 run2351.0
            config['lambd'] = 0.99
            config['ls_rate'] =  100000.0
        elif config['h_algo_name'] == 'VAEVPG':
            # 0.99 100000.0 50 25 4893.5 7 run2346.63
            config['lambd'] = 0.99
            config['ls_rate'] =  100000.00
            config['vae_loss_percentile'] = 50
        elif config['h_algo_name'] == 'VPG':
            # 0.99 0.000010 28 3503.7 9 run2354.113
            config['lambd'] = 0.99
            config['ls_rate'] =  0.000010


    if env_name=='Hopper-v2':
        # setup
        config['batch_size'] = 4000
        config['n_epochs'] = 200

        # optimization run1833.205
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
            # config['data_path'] = 'snapshots/SAC_Hoppe_1.0_None_F/646449184' default
            config['data_path'] = 'snapshots/SAC_Hoppe_1.0_None_F_200/581079651/'
            config['data_itr'] = [0,199,4]
        elif mode=='test':
            config['data_path'] = 'snapshots/SAC_Hoppe_1.0_None_F_200/581079651/'
            config['data_itr'] = [0,199,4]


        else:
            raise ValueError

        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.98 0.000010 21 1301.0 13 run2344.60
            config['lambd'] = 0.98
            config['ls_rate'] =  0.000010
        elif config['h_algo_name'] == 'VPG':
            # 0.95 100000.000000 13 1827.3 12 run2352.14
            config['lambd'] = 0.95
            config['ls_rate'] =  100000.000000
        elif config['h_algo_name'] == 'VAEVPG':
            # 0.95 100000.000000 99 11 1827.8 14 run2345.170
            config['lambd'] = 0.95
            config['ls_rate'] =  100000.000000
            config['vae_loss_percentile'] = 99
        elif config['h_algo_name'] == 'SAC':
            # 0.99 0.000010 8 1217.8 13 run2353.35
            config['lambd'] = 0.99
            config['ls_rate'] =  0.000010

    if env_name=='Humanoid-v2':
        # setup
        config['batch_size'] = 10000
        config['n_epochs'] = 500

        # optimization run1887.113
        config['policy_lr'] = 0.00200
        config['value_lr'] = 0.00025
        config['discount'] = 0.99
        config['target_update_tau'] = 0.0200

        # architecture
        config['policy_network_hidden_sizes'] = [256,256]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 80
        config['w_n_epoch'] = 50

        # srl
        if mode=='train':
            config['data_path'] = 'snapshots/SAC_Human_1.0_F_F/293494415/'
            config['data_itr'] = [0,200,4]
            # config['data_itr'] = 400
        elif mode=='test':
            config['data_path'] = 'snapshots/SAC_Human_1.0_F_F/293494415/'
            config['data_itr'] = [0,200,4]
        else:
            raise ValueError

        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.90 0.000010 23 1965.5 3 run2164.87
            # 0.95 0.000010 24 1907.1 8 run2377.94 (after batch bug fix)
            config['lambd'] = 0.95
            config['ls_rate'] =  0.000010
        elif config['h_algo_name'] == 'VPG':
            # 0.90 1.000000 54 2750.7 2 run2176.70
            # 0.90 1.000000 7 2640.7 11 run2379.119
            config['lambd'] = 0.9
            config['ls_rate'] =  1.0
        elif config['h_algo_name'] == 'VAEVPG':
            # 0.95 100000.000000 99 2 2465.4 3 run2163.34
            config['lambd'] = 0.95
            config['ls_rate'] =  100000.000000
            config['vae_loss_percentile'] = 99


    if env_name=='Swimmer-v2':
        # setup
        config['batch_size'] = 4000
        config['n_epochs'] = 200

        # optimization  run1888.258
        config['policy_lr'] = 0.00050
        config['value_lr'] = 0.00050
        config['discount'] = 0.999
        config['target_update_tau'] = 0.0100

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 30
        config['w_n_epoch'] = 30

        # srl
        if mode=='train':
            config['data_path'] = 'snapshots/SAC_Swimm_1.0_None_F/355552195'
            config['data_itr'] = [0,199,4]
        elif mode=='test':
            config['data_path'] = 'snapshots/SAC_Swimm_1.0_None_F/355552195'
            config['data_itr'] = [0,199,4]
        else:
            raise ValueError

        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.99 1.000000 24 164.8 8 run2343.41
            config['lambd'] = 0.99
            config['ls_rate'] =  1.000000
        elif config['h_algo_name'] == 'VPG':
            # 0.95 1.000000 6 205.1 10 run2350.74
            config['lambd'] = 0.95
            config['ls_rate'] =  1.000000
        elif config['h_algo_name'] == 'VAEVPG':
            # 0.95 100000.0 99 154 181.2 41 run2348.11
            config['lambd'] = 0.99
            config['ls_rate'] =  100000.0
            config['vae_loss_percentile'] = 99
        elif config['h_algo_name'] == 'SAC':
            # 0.98 100000.0 67 201.7 9 run2355.44
            config['lambd'] = 0.98
            config['ls_rate'] =  100000.0

    if env_name=='Ant-v2':
        # setup
        config['batch_size'] = 10000
        config['n_epochs'] = 1000

        # optimization run2101.213
        config['policy_lr'] = 0.00200
        config['value_lr'] = 0.00025
        config['discount'] = 0.99
        config['target_update_tau'] = 0.0050

        # architecture
        config['policy_network_hidden_sizes'] = [256,256]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 80
        config['w_n_epoch'] = 50

        # srl
        if mode=='train':
            config['data_path'] = 'snapshots/SAC_Ant-v_1.0_F_F/232869848'
            config['data_itr'] = [0,400,8]
        elif mode=='test':
            config['data_path'] = 'snapshots/SAC_Ant-v_1.0_F_F/232869848'
            config['data_itr'] = [0,400,8]
        else:
            raise ValueError

        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.93 0.500000 21 3907.8 3 run2187.7
            config['lambd'] = 0.93
            config['ls_rate'] =  0.500000
        elif config['h_algo_name'] == 'VPG':
            # 0.93 0.250000 11 3668.6 2 run2189.64
            config['lambd'] = 0.93
            config['ls_rate'] =  0.250000
        elif config['h_algo_name'] == 'VAEVPG':
            # 0.99 0.000010 25 1 3817.5 2 run2190.122
            config['lambd'] = 0.99
            config['ls_rate'] =  0.000010
            config['vae_loss_percentile'] = 25
        elif config['h_algo_name'] == 'SAC':
            # 0.93 0.250000 11 3668.6 2 run2189.64
            config['lambd'] = 0.93
            config['ls_rate'] =  0.250000

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
