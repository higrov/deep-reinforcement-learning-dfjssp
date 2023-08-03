from asyncio.log import logger
import datetime
import logging
import time
import warnings
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb

from rl.seac import utils

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None


#logger = logging.getLogger(__name__)


class TrainingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, total_timesteps, log_interval, save_interval, num_envs, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        # args
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.save_interval = save_interval
        # to use
        self.start = 0
        self.i = 0
        self.all_infos = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.start = time.time()
        self.logger.info(f"Training Start")
        logger.info(f"Training Start")

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        if self.i > 0:
            self.logger.info(f"Updating model trajectories...Done")
            logger.info(f"Updating model trajectories...Done")

            self.logger.info(f"UPDATE # {self.i}, TOTAL STEPS: {self.num_timesteps}/{self.total_timesteps}, FPS: {int(self.num_timesteps/(time.time() - self.start))}")
            logger.info(f"UPDATE # {self.i}, TOTAL STEPS: {self.num_timesteps}/{self.total_timesteps}, FPS: {int(self.num_timesteps/(time.time() - self.start))}")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        for info in self.locals['infos']:
            if info:
                self.all_infos.append(info)

        if self.log_interval is not None and self.num_timesteps % self.log_interval == 0 and len(self.all_infos) > 1:
            squashed = utils._squash_info(self.all_infos)
            self.logger.info(f"Last 100 episodes mean reward: {np.mean(squashed['episode_reward']) if 'episode_reward' in squashed else 0:.3f}")
            self.logger.info(f"Time elapsed: {str(datetime.timedelta(seconds=(time.time() - self.start)))}s")
            
            for k, v in squashed.items():
                wandb.log({f"env_metrics/{k}": v})
            self.all_infos.clear()

        if self.save_interval is not None and self.num_timesteps % self.save_interval == 0:
            #self.logger.info(f"Saving model after {self.num_timesteps} steps.")
            #logger.info(f"Saving model after {self.num_timesteps} steps.")
            # generate gif of episode
            anim_file = self.training_env.env_method('generate_animation', *(100, f"_{self.num_timesteps}"))
            wandb.log({"animation": wandb.Video(anim_file[0], fps=4, format="gif")})

        if self.num_timesteps % 100 == 0:
            self.logger.dump()
            
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.logger.info(f"Updating model trajectories...")
        logger.info(f"Updating model trajectories...")
        self.i+=1

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.logger.info("Training End")



class ProgressBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )
        self.pbar = None

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
            
        return True

    def _on_training_end(self) -> None:
        # Close progress bar
        self.pbar.close()
