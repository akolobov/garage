from garage.experiment import Snapshotter
import torch
from garage.shortrl.utils import torch_stop_grad
import copy
from functools import partial


class _Vf:
    def __init__(self, policy, qfs):
        self.policy = policy
        self.qfs = qfs
    def __call__(self, obs):
        acs = self.policy.get_actions(obs)[0]
        acs = torch.Tensor(acs)
        pred_qs = [qf(obs, acs) for qf in self.qfs]
        return torch.minimum(*pred_qs)

def get_algo_vf(algo):
    # load heuristic
    if type(algo).__name__ in ['PPO','TRPO','VPG']:
        vf = algo._value_function
    elif type(algo).__name__ in ['SAC', 'TD3', 'CQL']:
        qfs = [algo._qf1, algo._qf2]
        vf = _Vf(algo.policy, qfs)
    else:
        raise ValueError('Unsupported algorithm.')
    return vf

def get_algo_policy(algo):
    # load policy
    if type(algo).__name__ in ['PPO','TRPO','VPG','TD3','SAC', 'CQL', 'BC']:
        policy = algo.policy
    else:
        raise ValueError('Unsupported algorithm.')
    return policy

def load_policy_from_snapshot(path, itr='last'):
    snapshotter = Snapshotter()
    data = snapshotter.load(path, itr=itr)
    algo = data['algo']
    policy = get_algo_policy(algo)
    return policy

def load_heuristic_from_snapshot(path, itr='last'):
    snapshotter = Snapshotter()
    data = snapshotter.load(path, itr=itr)
    algo = data['algo']
    vf = get_algo_vf(algo)
    return torch_stop_grad(vf)