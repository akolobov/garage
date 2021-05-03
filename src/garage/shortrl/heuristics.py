from garage.experiment import Snapshotter
import torch
from garage.shortrl.utils import torch_stop_grad

def vf_from_qfs(qfs, policy):
    policy = copy.deepcopy(policy)
    def vf(obs):
        acs = policy.get_actions(obs)
        pred_qs = [qf(obs, acs) for qf in qfs]
        return torch.min(pred_qs)
    return value_

def get_algo_vf(algo):
    # load heuristic
    if type(algo).__name__ in ['PPO','TRPO','VPG']:
        vf = algo._value_function
    elif type(algo).__name__ in ['SAC', 'TD3', 'CQL']:
        qfs = [algo._qf1, algo._qf2]
        vf = vf_from_qfs(policy, qfs)
    else:
        raise ValueError('Unsupported algorithm.')
    return vf

def get_algo_policy(algo):
    # load policy
    if type(algo).__name__ in ['PPO','TRPO','VPG','TD3','SAC', 'CQL']:
        policy = algo.policy
    else:
        raise ValueError('Unsupported algorithm.')
    return policy

def load_policy_and_heuristic_from_snapshot(path, itr='last'):
    snapshotter = Snapshotter()
    data = snapshotter.load(path, itr=itr)
    algo = data['algo']
    policy = get_algo_policy(algo)
    vf = get_algo_vf(algo)
    return policy, torch_stop_grad(vf)
