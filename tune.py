from ray import tune
from examples.main import run


hps_dict = dict(
        algo=['CAC'],
        env_name=['kitchen-partial-v0'], # 'kitchen-partial-v0'],
        policy_lr=[0],
        value_lr=[5e-4],
        beta = [0.1, 1, 10, 100, 1000],  # BC
        # target_update_tau=[5e-3, 5e-4],
        # beta = [-1.2],  # BC
        # discount=[0],
        n_qf_steps=[1],
        norm_constraint=[100],
        use_two_qfs=[False],
        # fixed_alpha=[0, None],
        # optimizer=['RMSprop'],
        # value_activation=['LeakyReLU']
        num_evaluation_episodes=[1],
        # n_warmstart_steps=[0]
)

for k, v in hps_dict.items():
    hps_dict[k] = tune.grid_search(v)

def trainable(config):
    config['ignore_shutdown']=True
    score = run(**config)
    tune.report(score=score)     # This sends the score to Tune.


analysis= tune.run(trainable,
        resources_per_trial=tune.PlacementGroupFactory([
        {"CPU": 1},
        {"CPU": 1},]),
        config=hps_dict,
        queue_trials=False,
        mode='max',
        )
