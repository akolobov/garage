import copy
from dowel import tabular

from garage.torch.policies import TanhGaussianMLPPolicy, GaussianMLPPolicy, DeterministicMLPPolicy
from garage.torch.modules.gaussian_mlp_module import GaussianMLPBaseModule, GaussianMLPModule
from garage.torch.modules.mlp_module import MLPModule
from garage.shortrl.trainer import Trainer
from garage.shortrl.algorithms import get_algo

def init_policy_from_baseline(policy, baseline_policy,
                              *,
                              use_bc=False,
                              # bc parameters
                              env=None,
                              n_epochs=1,
                              batch_size=None,
                              opt_n_grad_steps=1000,
                              opt_minibatch_size=128,
                              policy_lr=1e-3,
                              n_workers=4,
                              ctxt=None):
    if use_bc:
        assert batch_size is not None
        algo = get_algo(env=env,
                      discount=1.0,
                      algo_name='BC',
                      init_policy=policy,
                      expert_policy=baseline_policy,
                      n_epochs=n_epochs,
                      batch_size=batch_size,
                      opt_n_grad_steps=opt_n_grad_steps,
                      opt_minibatch_size=opt_minibatch_size,
                      policy_lr=policy_lr,
                      n_workers=n_workers)
        trainer = Trainer(ctxt)
        trainer.setup(algo, env, lambd=1.0, discount=1.0)
        with tabular.prefix('Pretraining' + '/'):
            trainer.train(n_epochs=n_epochs,
                          batch_size=batch_size,
                          ignore_shutdown=True)
        return algo.learner

    else:
        # Try to set the init_policy as the baseline_policy
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
