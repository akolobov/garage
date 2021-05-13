import torch
from torch import nn
from garage.torch.modules import GaussianMLPModule
from garage.torch.value_functions.gaussian_mlp_value_function import GaussianMLPValueFunction

class PessimisticGaussianMLPValueFunction(GaussianMLPValueFunction):

    def __init__(self,
                env_spec,
                hidden_sizes=(32, 32),
                hidden_nonlinearity=torch.tanh,
                hidden_w_init=nn.init.xavier_uniform_,
                hidden_b_init=nn.init.zeros_,
                output_nonlinearity=None,
                output_w_init=nn.init.xavier_uniform_,
                output_b_init=nn.init.zeros_,
                learn_std=True,
                init_std=1.0,
                layer_normalization=False,
                ensemble_size=1,
                name='GaussianMLPValueFunction'):
        super(GaussianMLPValueFunction, self).__init__(env_spec, name)

        input_dim = env_spec.observation_space.flat_dim
        output_dim = ensemble_size

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=None,
            max_std=None,
            std_parameterization='exp',
            layer_normalization=layer_normalization)


    def compute_loss(self, obs, returns):
        """Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """

        ind = returns[:,1]
        returns = returns[:,0]
        import pdb; pdb.set_trace()

        mask = torch.zeros_like(pred)>0.5


        pred = self.module(obs).mean
        returns = returns.unsqueeze(-1)
        error = mask*(pred-returns)
        error = (pred-returns)
        loss = 0.5*torch.mean(error**2)
        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        pred = self.module(obs).mean # .flatten(-2)
        pred, _ = torch.min(pred, dim=-1)
        return pred

    def uncertainty(self, obs):
        pred = self.module(obs).mean # .flatten(-2)
        pred_min, _ = torch.min(pred, dim=-1)
        pred_max, _ = torch.max(pred, dim=-1)
        return pred_max - pred_min
