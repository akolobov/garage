from src.garage.shortrl.main import run_exp

from srl_config import default_config

# Config
algo_name='SAC'
env_name = 'Humanoid-v2'
h_algo_name='VAEVPG'
config = default_config(env_name=env_name,
                        algo_name=algo_name,
                        h_algo_name=h_algo_name,
                        w_algo_name='BC')
config['lambd'] = 0.99
run_exp(env_name=env_name,  **config)