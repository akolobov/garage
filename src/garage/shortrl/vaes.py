import torch
from torch import nn
import torch.nn.functional as F
from garage.torch.modules import GaussianMLPModule
from garage.torch.value_functions.gaussian_mlp_value_function import GaussianMLPValueFunction

class StateVAE(GaussianMLPValueFunction):
    def __init__(self,
                 env_spec,
                 hidden_sizes=(64, 64),
                 hidden_nonlinearity=torch.ReLU,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 latent_dim=32,
                 layer_normalization=False,
                 name='GaussianVAE'):
        super(GaussianMLPValueFunction, self).__init__(env_spec, name)

        input_dim = env_spec.observation_space.flat_dim
        self._encoder = GaussianMLPTwoHeadedModule(
                        input_dim=input_dim,
                        output_dim=latent_dim,
                        hidden_sizes=hidden_sizes,
                        hidden_nonlinearity=hidden_nonlinearity,
                        hidden_w_init=hidden_w_init,
                        hidden_b_init=hidden_b_init,
                        output_nonlinearity=None,
                        learn_std=True,
                        init_std=1.0,
                        min_std=1e-4,
                        max_std=1e15,
                        std_parameterization='exp',
                        layer_normalization=layer_normalization)
                        
        self._decoder = GaussianMLPModule(
                        input_dim=latent_dim,
                        output_dim=input_dim,
                        hidden_sizes=hidden_sizes[::-1],
                        hidden_nonlinearity=hidden_nonlinearity,
                        hidden_w_init=hidden_w_init,
                        hidden_b_init=hidden_b_init,
                        output_nonlinearity=None,
                        learn_std=False,
                        layer_normalization=layer_normalization)

    def forward(self, obs):
        dist = self._encoder(obs)
        z = dist.rsample(obs.shape[0])
        print(obs, z)
        u = self._decoder(z)
        return u, dist.mean, dist.stddev

    def compute_loss(self, obs):
        pred, mean, std = self.forward(obs)
        reconstruction_loss = F.mse_loss(pred, obs)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        loss = recon_loss + 0.5 * KL_loss
        return loss
