from garage.experiment import Snapshotter
import torch
from garage.shortrl.utils import torch_method
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

class _Pessimistic_Vf:
    def __init__(self, vf, is_pessimistic, vae, vmin, beta):
        self._vf = vf
        self._vae = vae
        self._is_pessimistic = is_pessimistic
        self._vmin = vmin
        self._beta = beta

    def __call__(self, obs):
        if not self._is_pessimistic or self._beta is None or self._beta == 0:
            return self._vf(obs)
        else:
            score = -self._vae.compute_loss(obs)
            indicator = torch.sigmoid(100*(score - self._beta))

            values = indicator * self._vf(obs) + (1 - indicator) * self._vmin
            return values

def get_algo_vf(algo, pessimism_threshold):
    # load heuristic
    if type(algo).__name__ in ['PPO','TRPO', 'VPG']:
        vf = algo._value_function
    elif type(algo).__name__ in ['SAC', 'TD3', 'CQL']:
        qfs = [algo._qf1, algo._qf2]
        vf = _Vf(algo.policy, qfs)
    elif type(algo).__name__ in ['VAEVPG']:
        vf = _Pessimistic_Vf(algo._value_function, algo._is_pessimistic, algo.vae,
                algo._vmin, pessimism_threshold)
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

def load_heuristic_from_snapshot(path, itr='last', pessimism_threshold=0):
    snapshotter = Snapshotter()
    data = snapshotter.load(path, itr=itr)
    algo = data['algo']
    vf = get_algo_vf(algo, pessimism_threshold)
    return torch_method(vf)
