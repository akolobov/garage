from ray import tune
from src.garage.shortrl.main import run_exp
from srl_config import default_config

n_workers = 4
log_prefix = 'tune'
algo_name='SAC'
env_name = 'Swimmer-v2'
h_algo_name= None

space = {
         "seed": tune.grid_search(list(range(4)))
         }

def trainable(new_config):
    config = default_config(env_name=env_name,
                            algo_name=algo_name,
                            h_algo_name=h_algo_name,
                            w_algo_name='BC',
                             mode='train')

    for key in new_config.keys():
        config[key] = new_config[key]
    config['ignore_shutdown']=True
    config['n_workers'] = n_workers
    config['warmstart_policy'] = False #True

    score = run_exp(env_name=env_name,  **config)
    tune.report(score=score)     # This sends the score to Tune.


analysis= tune.run(trainable,
        resources_per_trial=tune.PlacementGroupFactory([
        {"CPU": 1},
        {"CPU": n_workers},]),
        config=space,
        queue_trials=True,
        mode='max',
        )
