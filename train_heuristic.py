from src.garage.shortrl.main import run_exp



# Config

algo_name='SAC'
discount = None
n_epochs = 50
env_name = 'HalfCheetah-v2'
batch_size = 10000
offline_value_ensemble_size = 1
h_algo_name='VAEVPG'
h_n_epoch = 50
vae_loss_percentile = 1  # an interger from 0-99


if env_name=='InvertedDoublePendulum-v2':
    data_path = 'snapshots/SAC_Inver_1.0_F_F/120032374/'
    data_itr = [0,9]
    policy_network_hidden_sizes=(64, 64)
    value_natwork_hidden_sizes=(256, 256)
    h_n_epoch = 30
    w_n_epoch = 30

if env_name=='HalfCheetah-v2':
    data_path= 'snapshots/SAC_HalfC_1.0_F_F/210566759/'
    data_itr = 20 # [0,20]
    policy_network_hidden_sizes=(64, 64)
    value_natwork_hidden_sizes=(256, 256)
    h_n_epoch = 30
    w_n_epoch = 30

if env_name=='Humanoid-v2':
    data_path= 'snapshots/SAC_Human_1.0_F_F/673933361/'
    data_itr = [0,200,4]
    policy_network_hidden_sizes=(256, 256)
    value_natwork_hidden_sizes=(256, 256)
    h_n_epoch = 80
    w_n_epoch = 50

if env_name=='Ant-v2':
    data_path= 'snapshots/SAC_Ant-v_1.0_F_F/779696512'
    data_itr = [0,300,6]
    policy_network_hidden_sizes=(256, 256)
    value_natwork_hidden_sizes=(256, 256)
    h_n_epoch = 80
    w_n_epoch = 50



w_n_epoch = data_itr[1]-data_itr[0] if len([data_itr])>1 else 30
episode_batch_size = batch_size

run_exp(algo_name=algo_name,
        discount=discount,
        n_epochs=n_epochs, # either n_epochs or total_n_samples needs to be provided
        env_name=env_name,
        batch_size=batch_size,
        seed=1,
        policy_network_hidden_sizes=policy_network_hidden_sizes,
        value_natwork_hidden_sizes=value_natwork_hidden_sizes,
        # offline batch data
        data_path=data_path,  # directory of the snapshot
        data_itr=data_itr,
        episode_batch_size=episode_batch_size,
        offline_value_ensemble_size=offline_value_ensemble_size,
        offline_value_ensemble_mode='P',
        # pretrain policy
        warmstart_policy=True,
        w_algo_name='BC',
        w_n_epoch=w_n_epoch,
        # short-horizon RL params
        lambd=0.99,
        use_raw_snapshot=False,
        use_heuristic=True,
        h_algo_name=h_algo_name,
        h_n_epoch=h_n_epoch,
        # vae parameters
        vae_loss_percentile=vae_loss_percentile,
        )