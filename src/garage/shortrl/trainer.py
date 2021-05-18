import os
import time
from collections import namedtuple
from dowel import logger, tabular
import numpy as np

from garage.trainer import Trainer as garageTrainer
from garage.trainer import TrainArgs
from garage.experiment.experiment import dump_json
from garage.sampler.env_update import EnvUpdate
from garage.sampler import _apply_env_update
from garage.envs import GymEnv
from garage.shortrl.env_wrapper import ShortMDP
from garage.shortrl.lambda_schedulers import LambdaScheduler

from garage.shortrl.utils import read_attr_from_csv


def get_algodata_cls(algo):
    class Cls:
        def __init__(self, *, policy, vf, qf1, qf2, pessimistic, vae, vmin, vae_loss_percentile):
            self.policy = policy
            self._value_function = vf
            self._qf1 = qf1
            self._qf2 = qf2
            self._is_pessimistic = pessimistic
            self.vae = vae
            self._vmin = vmin
            self.vae_loss_percentile = vae_loss_percentile

    Cls.__name__ = type(algo).__name__
    return Cls



class Trainer(garageTrainer):
    """
        Add a light saving mode to minimze the stroage usage.
        Add a ignore_shutdown flag for running multiple experiments.
        Add a return_attr option
    """

    def setup(self, *args,
              save_mode='light',
              return_mode='average', # 'full', 'average', 'last'
              return_attr='Evaluation/AverageReturn',  # the log attribute
              **kwargs):
        output = super().setup(*args, **kwargs)
        self.save_mode = save_mode
        self.return_mode = return_mode
        self.return_attr = return_attr
        return output

    # Add a light saving mode (which saves only policy and value functions of an algorithm)
    def save(self, epoch):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the trainer is set up.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup trainer before saving.')

        logger.log('Saving snapshot...')

        if self.save_mode=='light':
            # HACK Save only the learned networks
            params = dict()
            # Save arguments
            params['seed'] = self._seed
            params['train_args'] = self._train_args
            params['stats'] = self._stats
            AlgoData = get_algodata_cls(self._algo)
            algodata = AlgoData(policy=self._algo.policy,
                                vf=getattr(self._algo, '_value_function', None),
                                qf1=getattr(self._algo, '_qf1', None),
                                qf2=getattr(self._algo, '_qf2', None),
                                pessimistic=getattr(self._algo, '_pessimistic_vae_filter', False),
                                vae=getattr(self._algo, '_vae', None),
                                vmin=getattr(self._algo, '_vmin', None),
                                vae_loss_percentile=getattr(self._algo, 'vae_loss_percentile', None))
            params['algo'] = algodata

        else:  # default behavior: save everything
            params = dict()
            # Save arguments
            params['seed'] = self._seed
            params['train_args'] = self._train_args
            params['stats'] = self._stats

            # Save states
            params['env'] = self._env
            params['algo'] = self._algo
            params['n_workers'] = self._n_workers
            params['worker_class'] = self._worker_class
            params['worker_args'] = self._worker_args

        self._snapshotter.save_itr_params(epoch, params)

        logger.log('Saved')

    # Include ignore_shutdown flag
    def train(self,
              n_epochs,
              batch_size=None,
              plot=False,
              store_episodes=False,
              pause_for_plot=False,
              ignore_shutdown=False):
        """Start training.

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int or None): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
            pause_for_plot (bool): Pause for plot.

        Raises:
            NotSetupError: If train() is called before setup().

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._has_setup:
            raise NotSetupError(
                'Use setup() to setup trainer before training.')

        # Save arguments for restore
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot
        self._start_worker()

        log_dir = self._snapshotter.snapshot_dir
        summary_file = os.path.join(log_dir, 'experiment.json')
        dump_json(summary_file, self)

        last_return = self._algo.train(self)

        ### HACK Ignore shutdown, if needed ###
        if not ignore_shutdown:
            self._shutdown_worker()

        ### HACK Return other statistics ###
        progress = self._read_progress(self.return_attr)
        progress = progress if progress is not None else 0
        if self.return_mode == 'average':
            score = np.mean(progress)
        elif self.return_mode == 'full':
            score = progress
        elif self.return_mode == 'last':
            score = last_return
        else:
            NotImplementedError
        return score

    def _read_progress(self, attr):
        csv_path = self._snapshotter._snapshot_dir
        csv_path = os.path.join(csv_path,'progress.csv')
        score = read_attr_from_csv(csv_path, attr)
        return score

class LambdaEnvUpdate(EnvUpdate):
    """ Update the lambda value of a ShortMDP environment. """
    def __init__(self, lambd, old_env_update):
        self._lambd = lambd
        self._old_env_update = old_env_update

    def __call__(self, old_env=None):
        assert isinstance(old_env, GymEnv)
        old_env, _ = _apply_env_update(old_env, self._old_env_update)
        if isinstance(old_env._env, ShortMDP):
            old_env._env._lambd = self._lambd
        return old_env


class SRLTrainer(Trainer):

    def setup(self, *args, discount=None, lambd=1.0, **kwargs):
        assert discount is not None
        output = super().setup(*args, **kwargs)
        self._lambd = lambd if isinstance(lambd, LambdaScheduler) else LambdaScheduler(lambd)
        self.discount = discount
        return output

    @property
    def lambd(self):
        return self._lambd()

    def update_lambd(self):
        if self._env is not None:  # to account for batch mode
            if isinstance(self._env._env, ShortMDP):
                self._lambd.update()
                self._env._env._lambd = self.lambd
                if hasattr(self._algo, '_eval_env') and isinstance(self._algo._eval_env._env, ShortMDP):
                    self._algo._eval_env._env._lambd = self.lambd


    def obtain_episodes(self,
                        itr,
                        batch_size=None,
                        agent_update=None,
                        env_update=None):
        # Update the discount factor of env and algo
        try:
            self._algo.discount = self.discount*self.lambd
        except AttributeError:
            self._algo._discount = self.discount*self.lambd
        # Update the lambda value stored in envs
        env_update = LambdaEnvUpdate(self.lambd, env_update)
        return super().obtain_episodes(itr,
                                       batch_size=batch_size,
                                       agent_update=agent_update,
                                       env_update=env_update)

    # add lambda update
    def step_epochs(self):
        """Step through each epoch.

        This function returns a magic generator. When iterated through, this
        generator automatically performs services such as snapshotting and log
        management. It is used inside train() in each algorithm.

        The generator initializes two variables: `self.step_itr` and
        `self.step_episode`. To use the generator, these two have to be
        updated manually in each epoch, as the example shows below.

        Yields:
            int: The next training epoch.

        Examples:
            for epoch in trainer.step_epochs():
                trainer.step_episode = trainer.obtain_samples(...)
                self.train_once(...)
                trainer.step_itr += 1

        """
        self._start_time = time.time()
        self.step_itr = self._stats.total_itr
        self.step_episode = None

        # Used by integration tests to ensure examples can run one epoch.
        n_epochs = int(
            os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                           self._train_args.n_epochs))

        logger.log('Obtaining samples...')

        for epoch in range(self._train_args.start_epoch, n_epochs):
            self._itr_start_time = time.time()
            with logger.prefix('epoch #%d | ' % epoch):
                yield epoch

                self.update_lambd()  ### HACK Update lambda scheduler ###

                save_episode = (self.step_episode
                                if self._train_args.store_episodes else None)

                self._stats.last_episode = save_episode
                self._stats.total_epoch = epoch
                self._stats.total_itr = self.step_itr

                self.save(epoch)

                if self.enable_logging:
                    self.log_diagnostics(self._train_args.pause_for_plot)
                    logger.dump_all(self.step_itr)
                    tabular.clear()
