import os
import numpy as np
from d4rl import infos as d4rl_infos


from garage.tools.utils import read_attr_from_csv


ORDER = \
[
    'maze2d-unmaze',
    'maze2d-medium',
    'maze2d-large',
    'antmaze-umaze',
    'antmaze-umaze-diverse',
    'antmaze-medium-play',
    'antmaze-medium-diverse',
    'antmaze-large-play',
    'antmaze-large-diverse',
    'halfcheetah-random',
    'walker2d-random',
    'hopper-random',
    'halfcheetah-medium',
    'walker2d-medium',
    'hopper-medium',
    'halfcheetah-medium-replay',
    'walker2d-medium-replay',
    'hopper-medium-replay',
    'halfcheetah-medium-expert',
    'walker2d-medium-expert',
    'hopper-medium-expert',
    'halfcheetah-expert',
    'walker2d-expert',
    'hopper-expert',
    'pen-human',
    'hammer-human',
    'door-human',
    'relocate-human',
    'pen-cloned',
    'hammer-cloned',
    'door-cloned',
    'relocate-cloned',
    'pen-expert',
    'hammer-expert',
    'door-expert',
    'relocate-expert',
    'kitchen-complete',
    'kitchen-partial',
    'kitchen-mixed',
]



FILTER = {'init_q_eval_mode': '0.5_0.5'}

HPS = ['beta']
n_warmstart_epochs = 100
eval_freq = 100

def find_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]

def append_attr(info_dict, attr_name, csv_path):
    if attr_name not in info_dict:
        info_dict[attr_name] = []
    value = read_attr_from_csv(csv_path, attr_name)
    if value is not None:
        info_dict[attr_name].append(value)

def normalize_score(score, env_name):
    max_score = d4rl_infos.REF_MAX_SCORE[env_name]
    min_score = d4rl_infos.REF_MIN_SCORE[env_name]
    return (score - min_score)/(max_score - min_score)*100

def unnormalize_score(score, env_name):
    max_score = d4rl_infos.REF_MAX_SCORE[env_name]
    min_score = d4rl_infos.REF_MIN_SCORE[env_name]
    return (max_score - min_score)*score/100 + min_score

def analyze(log_dir,
            hp_selection_rule='best', # or fixed
            stop_rule='last', #'best' 'last', 'best',
            ):

    exp_names = find_dirs(log_dir)

    exp_scores = {}
    for exp_name in exp_names:
        # For each dataset
        algo_name, env_name = exp_name.split('_')
        runs = find_dirs(os.path.join(log_dir,exp_name))

        info = {}
        for run in runs:
            if any([ m+'_'+FILTER[m] not in run for m in FILTER]):
                continue

            csv_path = os.path.join(log_dir,exp_name,run,'progress.csv')
            hp_values = [ run.split(hp_name)[1].split('_')[1]  for hp_name in HPS ]
            key = '_'.join(hp_values)

            if key not in info:
                info[key] = {}

            append_attr(info[key], 'Evaluation/AverageReturn', csv_path)
            append_attr(info[key], 'Algorithm/bellman_qf1_loss', csv_path)
            append_attr(info[key], 'Algorithm/bellman_qf2_loss', csv_path)
            append_attr(info[key], 'Algorithm/lower_bound', csv_path)
            append_attr(info[key], 'Algorithm/avg_bellman_error', csv_path)

            append_attr(info[key], 'Evaluation/TerminationRate', csv_path)



        for key in info.keys():
            if 'antmaze' in env_name:
                scores = info[key]['Evaluation/TerminationRate']
            else:
                scores = info[key]['Evaluation/AverageReturn']

            for i, (score, error) in enumerate(zip(scores, info[key]['Algorithm/avg_bellman_error'])):
                score = score[n_warmstart_epochs:]  # exclude the warmup phase
                error = error[n_warmstart_epochs:]  # exclude the warmup phase
                scores[i] = score

            if stop_rule=='last':
                info[key]['score'] = np.mean([ x[-1] for x in scores ])
            if stop_rule=='best':
                info[key]['score'] = np.mean([ np.max(x[0:-1:eval_freq]) for x in scores])
            if stop_rule=='average':
                info[key]['score'] = np.mean([ np.mean(x) for x in scores])

            # print(exp_name, key, info[key]['score'])

        if hp_selection_rule=='best':
            scores = [info[key]['score'] for key in info]
            argmax = np.argmax(scores)
            exp_scores[env_name] = [normalize_score(scores[argmax], env_name), list(info.keys())[argmax]]

    # for key in sorted(exp_scores):
    #     print(key, ':', exp_scores[key])

    for data_name in ORDER:
        keys = [k for k in exp_scores.keys()  if data_name==k[:-3]]
        if len(keys)>0:
            key = keys[0]
            print(key, ':', exp_scores[key])

    print('\n')

    for data_name in ORDER:
        keys = [k for k in exp_scores.keys()  if data_name==k[:-3]]
        if len(keys)>0:
            key = keys[0]
            print(round(exp_scores[key][0], 1))

    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--log_dir',  type=str)
    analyze(**vars(parser.parse_args()))


# parser.add_argument('-e', '--env_name',  type=str, default='hopper-medium-v0')
# parser.add_argument('-s', '--score', type=float, default=100)
# env_name = args['env_name']
# max_score = infos.REF_MAX_SCORE[env_name]
# min_score = infos.REF_MIN_SCORE[env_name]
# original_score = (max_score - min_score)*args['score']/100 + min_score
