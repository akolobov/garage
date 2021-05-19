from dowel import tabular

from garage.torch.algos import DQN as garageDQN

class DQN(garageDQN):
    def __init__(self, *args, target_update_tau=5e-3, **kwargs):
        self._tau = target_update_tau
        super().__init__(*args, **kwargs)

    # NOTE Use exponential moving average to update the target instead
    def _train_once(self, itr, episodes):
        """Perform one iteration of training.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        """
        self.replay_buffer.add_episode_batch(episodes)

        epoch = itr / self._steps_per_epoch

        for _ in range(self._n_train_steps):
            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                timesteps = self.replay_buffer.sample_timesteps(
                    self._buffer_batch_size)
                qf_loss, y, q = tuple(v.cpu().numpy()
                                      for v in self._optimize_qf(timesteps))

                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y)
                self._epoch_qs.append(q)

                # Exponential moving average
                for t_param, param in zip(self._target_qf.parameters(), self._qf.parameters()):
                    t_param.data.copy_(t_param.data * (1.0 - self._tau) + param.data * self._tau)

        if itr % self._steps_per_epoch == 0:
            self._log_eval_results(epoch)

        # if itr % self._target_update_freq == 0:
            # self._target_qf = copy.deepcopy(self._qf)
