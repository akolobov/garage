import copy
from garage.torch.policies import TanhGaussianMLPPolicy, GaussianMLPPolicy, DeterministicMLPPolicy
from garage.torch.modules.gaussian_mlp_module import GaussianMLPBaseModule, GaussianMLPModule
from garage.torch.modules.mlp_module import MLPModule

def set_as_baseline(policy, baseline_policy):
    """ Try to set the init_policy as the baseline_policy. """

    # Support GaussianMLPPolicy, TanhGaussianMLPPolicy, DeterministicMLPPolicy
    if isinstance(policy._module, GaussianMLPBaseModule):
        if isinstance(baseline_policy._module, GaussianMLPBaseModule):
            # make sure the init_policy uses the same normal distribution
            init_policy = copy.deepcopy(baseline_policy)
            init_policy._module._norm_dist_class = policy._module._norm_dist_class
        elif isinstance(baseline_policy._module, MLPModule):
            # the baseline is deterministic, so we can only try to copy the mean
            if hasattr(policy, '_mean_module'):
                init_policy = policy
                init_policy._mean_module = copy.deepcopy(baseline_policy._module)
            else:
                raise TypeError('The policy does not have _mean_module but the baseline policy is deterministic.')
        else:
            raise NotImplementedError

    elif isinstance(policy, DeterministicMLPPolicy):
        # policy is deterministic
        if isinstance(baseline_policy._module,GaussianMLPBaseModule) \
            and hasattr(baseline_policy._module, '_mean_module'):
            init_policy = policy
            init_policy._module = copy.deepcopy(baseline_policy._module._mean_module)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return init_policy
