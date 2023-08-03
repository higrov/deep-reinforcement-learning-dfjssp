import logging
import os
from typing import Callable
import gymnasium

import stable_baselines3 as sbln3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, EventCallback
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

from envs.overcooked_environment import OvercookedEnvironment
from rl.seac import utils
from rl.seac.envs import make_vec_envs_sbln
from rl.training_callback import ProgressBarCallback, TrainingCallback

from rl.seac.wrappers import (
    DictifyActions,
    FlattenObservation,
    GlobalizeReward,
    RecordEpisodeStatistics,
    SquashDones,
)


logger = logging.getLogger(__name__)


def learn_ppo(
    env_id,
    arglist,
    run_id,
    num_total_timesteps,
    num_steps_per_update,
    num_processes,
    device,
    image_obs=True,
    wrappers=(DictifyActions, FlattenObservation, GlobalizeReward, RecordEpisodeStatistics, SquashDones),
    save_dir="./models",
    load_dir="./models",
    log_dir="./logs",
    log_interval=5000,
    save_interval=100_000,
    lr=0.001,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    entropy_coef=0.1,
    value_loss_coef=0.2,
    max_grad_norm=0.5,
    restore=True,
    notes="",
    tags=[],
    sweep=False,
):
    
    if not sweep:
        logger.info(f"Starting Training loop for run-id: {arglist.run_id}")
        run = wandb.init(project="Paper-Results", name=run_id, notes=notes, tags=[], sync_tensorboard=True, mode="disabled")
    else:
        logger.info(f"Starting Sweep-Training for run-id: {wandb.config.run_id}")
        wandb.init(sync_tensorboard=True)

    config = {
    "policy_type": "CnnPolicy" if image_obs else "MlpPolicy",
    "total_timesteps": num_total_timesteps,
    "env_name": "overcookedEnv-v0",
    }
    wandb.config.update(config)

    folder = utils.get_folder_name(arglist, suffix="")

    if save_interval:
        save_dir = f"{save_dir}/{folder}"
        save_dir = os.path.expanduser(save_dir)
        utils.cleanup_log_dir(save_dir)

    if log_interval:
        log_dir = f"{log_dir}/{folder}"
        log_dir = os.path.expanduser(log_dir)
        utils.cleanup_log_dir(log_dir)

    if image_obs:
        wrappers = (DictifyActions, GlobalizeReward, RecordEpisodeStatistics, SquashDones,)
    
    logger.info(f"{'NOT ' if num_processes == 1 else ''}USING PARALLEL-PROCESSING: Preparing {num_processes} environment(s)")
    envs = make_vec_envs_sbln(env_id, arglist, run_id, image_obs, num_processes, wrappers)
    logger.info(f"Preparing {arglist.num_agents} PPO agent(s) per environment")
    model = sbln3.PPO(config["policy_type"], envs, tensorboard_log=f'{log_dir}',learning_rate=linear_schedule(lr), n_steps=num_steps_per_update, batch_size=batch_size, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=entropy_coef, vf_coef=value_loss_coef, max_grad_norm=max_grad_norm)
    
    # add restore code here
    model_paths = [_ for _ in [arglist.model1_path, arglist.model2_path, arglist.model3_path, arglist.model4_path][:arglist.num_agents] if _ is not None and "ppo" in _]
    if restore and any(model_paths):
        logger.info(f"Loading previous agent checkpoints")
        for mp in model_paths:
            if not mp or not os.path.exists(f'{load_dir}/{mp}'):
                # raise ValueError(f"Model {mp} does not exist at location {load_dir}. Please fix args or config.")
                pass
            else:
                envs.reset()
                model = model.load(f'{load_dir}/{mp}', envs, verbose=1, device=device, tensorboard_log=f'{log_dir}', seed=arglist.seed)


    wandb_callback = WandbCallback(
        gradient_save_freq=log_interval,
        log="all",
        verbose=2,
    )
    progress_bar_callback = ProgressBarCallback()
    log_stats_callback = TrainingCallback(num_total_timesteps, log_interval, save_interval, num_processes, verbose=2)
    save_callback = CheckpointCallback(save_freq=save_interval, save_path=f'{save_dir}/{run_id}/', name_prefix=f'{run_id}')
     
    training_callbacks = CallbackList([wandb_callback, progress_bar_callback, log_stats_callback, save_callback])

    model.learn(
        total_timesteps=num_total_timesteps,
        callback=training_callbacks,
        log_interval=log_interval, tb_log_name=f'{run_id}')

    envs.close()
    if not sweep:
        run.finish()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return (progress_remaining * initial_value)

    return func