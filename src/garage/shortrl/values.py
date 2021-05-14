import torch
from torch import nn
from garage.torch.modules import GaussianMLPModule
from garage.torch.value_functions.gaussian_mlp_value_function import GaussianMLPValueFunction

class EnsembleGaussianMLPValueFunction(GaussianMLPValueFunction):

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
                 ensemble_size=3,
                 share_embedding=False,
                 ensemble_mode='P',
                 name='GaussianMLPValueFunction'):
        super(GaussianMLPValueFunction, self).__init__(env_spec, name)

        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1

        assert ensemble_mode in ('P', 'Q')
        self.ensemble_size = ensemble_size
        self.share_embedding = share_embedding
        self.ensemble_mode = ensemble_mode

        if share_embedding:
            self._nets = GaussianMLPModule(
                input_dim=input_dim,
                output_dim=ensemble_size,
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
        else:
            self._nets = torch.nn.ModuleList([ GaussianMLPModule(
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
                        for _ in range(ensemble_size)])

    def compute_loss(self, obs, returns):
        mask = returns[:,1:]
        returns = returns[:,0]
        pred = self._predict(obs)
        returns = returns.unsqueeze(-1)
        error = mask*(pred-returns)
        loss = torch.mean(error**2)
        return loss

    def forward(self, obs):
        pred = self._predict(obs)
        if self.ensemble_mode == 'P':
            pred, _ = torch.min(pred, dim=-1)
        elif self.ensemble_mode == 'O':
            pred, _ = torch.max(pred, dim=-1)
        return pred

    def uncertainty(self, obs):
        pred = self._predict(obs)
        pred_min, _ = torch.min(pred, dim=-1)
        pred_max, _ = torch.max(pred, dim=-1)
        return pred_max - pred_min

    def _predict(self, obs):
        if self.share_embedding:
            pred = self._nets(obs).mean
        else:
            pred = torch.cat([ m(obs).mean for m in self._nets ], axis=-1)
        return pred