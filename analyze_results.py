import os
import numpy as np
from d4rl import infos as d4rl_infos


from garage.tools.utils import read_attr_from_csv


HPS = ['min_q_weight', 'discount']

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
            stop_rule='best', #'best' # of best
            ):

    exp_names = find_dirs(log_dir)

    exp_scores = {}
    for exp_name in exp_names:
        # For each dataset
        algo_name, env_name = exp_name.split('_')
        runs = find_dirs(os.path.join(log_dir,exp_name))

        info = {}
        for run in runs:
            csv_path = os.path.join(log_dir,exp_name,run,'progress.csv')
            hp_values = [ run.split(hp_name)[1].split('_')[1]  for hp_name in HPS ]
            key = '_'.join(hp_values)

            if key not in info:
                info[key] = {}

            append_attr(info[key], 'Evaluation/AverageReturn', csv_path)
            append_attr(info[key], 'Algorithm/bellman_qf1_loss', csv_path)
            append_attr(info[key], 'Algorithm/bellman_qf2_loss', csv_path)
            append_attr(info[key], 'Algorithm/lower_bound', csv_path)

        for key in info.keys():
            scores = info[key]['Evaluation/AverageReturn']
            if stop_rule=='last':
                info[key]['score'] = np.mean([ x[-1] for x in scores ])
            if stop_rule=='best':
                info[key]['score'] = np.mean([ np.max(x) for x in scores])


        if hp_selection_rule=='best':
            exp_scores[env_name] = normalize_score(np.max([info[key]['score'] for key in info]), env_name)
        # if hp_selection_rule=='fixed':


    for key in sorted(exp_scores):
        print(key, ':', exp_scores[key])

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
