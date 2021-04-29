from ray import tune
from shortrl.train_agent import run_exp



n_workers = 4
log_prefix = 'tune'
algo_name='SAC'

space = {
         "policy_lr": tune.grid_search([1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]),
         "value_lr": tune.grid_search([1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]),,
         "seed": tune.grid_search(list(range(4)))
         }


def trainable(config):

    score = run_exp(exp_name=algo_name+'_'+str(config['policy_lr'])+'_'+str(config['value_lr']),
                    log_prefix='tune_tests',
                    algo_name=algo_name,
                    **config,
                    n_workers=n_workers,
                    ignore_shutdown=True,
                    n_epochs=50)
    tune.report(score=score)     # This sends the score to Tune.


analysis= tune.run(trainable,
        resources_per_trial=tune.PlacementGroupFactory([
        {"CPU": 1},
        {"CPU": n_workers},]),
        config=space,
        queue_trials=True,
        mode='max',)
