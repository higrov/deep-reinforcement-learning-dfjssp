import logging
import os
import gymnasium as gym
import numpy as np
import torch

import wandb

from rl.seac import utils
from rl.mappo.mappo_runner import MAPPORunner as Runner
from rl.mappo.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from rl.mappo.envs.overcooked_environment_ma_wrapper import OverCookedMAEnv

logger = logging.getLogger(__name__)


def make_train_envs(env_name, env_arglist, run_id, parallel):
    def get_env_fn(rank):
        def init_env():
            env: OverCookedMAEnv = OverCookedMAEnv(env_name, env_arglist, f'{run_id}_{rank}', True) 
            env.seed(env_arglist.seed + rank * 1000)
            return env
        return init_env
    
    if parallel == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(parallel)])


def learn_mappo(
    env_id,
    arglist,
    run_id,
    num_total_timesteps,
    num_processes,
    device,
    image_obs=False,
    save_dir="./models",
    load_dir="./models",
    log_dir="./logs",
    log_interval=5_000,
    save_interval=200_000,
    use_linear_lr_decay=False,
    # NETWORK PARAMS
    share_policy=True,
    use_centralized_v=True,
    stacked_frames=1,
    use_stacked_frames=False,
    hidden_size=64,
    layer_N=1,
    use_ReLU=True,
    use_popart=False,
    use_valuenorm=False,
    use_feature_normalization=True,
    use_orthogonal=True,
    gain=0.01,
    # RECURRENT PARAMS
    use_naive_recurrent_policy=False,
    use_recurrent_policy=True,
    recurrent_N=1,
    data_chunk_length=10,
    # OTPIMIZER PARAMS
    lr=3e-4,
    critic_lr=3e-4,
    adam_eps=1e-5,
    weight_decay=0,
    # PPO PARAMS
    ppo_epoch=10,
    use_clipped_value_loss=True,
    clip_param=0.2,
    num_mini_batch=1,
    entropy_coef=0.01,
    value_loss_coef=1.0,
    use_max_grad_norm=True,
    max_grad_norm=0.5,
    use_gae=True,
    gamma=0.99,
    gae_lambda=0.95,
    use_proper_time_limits=False,
    use_huber_loss=True,
    use_value_active_masks=True,
    use_policy_active_masks=True,
    huber_delta=10.0,
    restore=True,
    notes="",
    tags=[],
    sweep=False,
):
    
    if not sweep:
        logger.info(f"Starting MAPPO Training loop for run-id: {arglist.run_id}")
        os.environ["WANDB_START_METHOD"] = 'thread'
        run = wandb.init(project="Paper-Results", id=run_id, name=run_id, notes=notes, tags=tags, sync_tensorboard=False, resume="allow")#mode="disabled")
    else:
        logger.info(f"Starting MAPPO Sweep for run-id: {wandb.config.run_id}")
        wandb.init(sync_tensorboard=False)

    folder = utils.get_folder_name(arglist, suffix="")

    if arglist.device == 'cuda' and torch.cuda.is_available():
        logger.info("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(32)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(8)
        
    if save_interval:
        save_dir = f"{save_dir}/{folder}"
        save_dir = os.path.expanduser(save_dir)
        utils.cleanup_log_dir(save_dir)

    if log_interval:
        log_dir = f"{log_dir}/{folder}"
        log_dir = os.path.expanduser(log_dir)
        utils.cleanup_log_dir(log_dir)

    # add restore code here
    model_paths = [_ for _ in [arglist.model1_path, arglist.model2_path, arglist.model3_path, arglist.model4_path][:arglist.num_agents] if _ is not None and ("MAPPO" in _ or "mappo" in _)]
    if restore and any(model_paths):
        logger.info(f"Loading previous agent checkpoints")
        for mp in model_paths:
            if not os.path.exists(f'{load_dir}/{mp}'):
                raise ValueError(f"Model '{mp}' does not exist at location {load_dir}. Please fix args or config.")
            else:
                load_dir = f'{load_dir}/{mp}'
    else: restore = False


    # seed
    torch.manual_seed(arglist.seed)
    torch.cuda.manual_seed_all(arglist.seed)
    np.random.seed(arglist.seed)

    logger.info(f"{'NOT ' if num_processes == 1 else ''}USING PARALLEL-PROCESSING: Preparing {num_processes} environment(s)")
    envs = make_train_envs(env_id, arglist, run_id, num_processes)
    logger.info(f"Preparing {arglist.num_agents} MAPPO agent(s) per environment")
    
    cfg = {
        # RUN PARAMS
        "all_args": arglist,
        "run_id": run_id,
        "run_dir": save_dir + "/" + run_id,
        "log_dir": log_dir, 
        "model_dir": load_dir,
        "restore": restore,
        "continue_run": arglist.continue_run,
        "save_interval": save_interval,
        "log_interval": log_interval,
        "seed": arglist.seed,
        "device": device,
        "record": arglist.record,
        "num_agents": arglist.num_agents,
        "max_num_agents": 5,
        "n_training_threads": 16, # PYTORCH THREADS
        "n_rollout_threads": num_processes,
        "num_env_steps": num_total_timesteps,
        "use_linear_lr_decay": use_linear_lr_decay,
        # REPLAY BUFFER PARAMS
        "episode_length": arglist.max_num_timesteps,
        # NETWORK PARAMS
        "share_policy": share_policy,
        "use_centralized_v": use_centralized_v,
        "stacked_frames": stacked_frames,
        "use_stacked_frames": use_stacked_frames,
        "hidden_size": hidden_size,
        "layer_N": layer_N,
        "use_ReLU": use_ReLU,
        "use_popart": use_popart,
        "use_valuenorm": use_valuenorm if not use_popart else use_valuenorm,
        "use_feature_normalization": use_feature_normalization,
        "use_orthogonal": use_orthogonal,
        "gain": gain,
        # RECURRENT PARAMS
        "use_naive_recurrent_policy": use_naive_recurrent_policy,
        "use_recurrent_policy": use_recurrent_policy,
        "recurrent_N": recurrent_N,
        "data_chunk_length": data_chunk_length,
        # OTPIMIZER PARAMS
        "lr": lr,
        "critic_lr": critic_lr,
        "adam_eps": adam_eps,
        "weight_decay": weight_decay,
        # PPO PARAMS
        "ppo_epoch": ppo_epoch,
        "use_clipped_value_loss": use_clipped_value_loss,
        "clip_param": clip_param,
        "num_mini_batch": num_mini_batch,
        "entropy_coef": entropy_coef,
        "value_loss_coef": value_loss_coef,
        "use_max_grad_norm": use_max_grad_norm,
        "max_grad_norm": max_grad_norm,
        "use_gae": use_gae,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "use_proper_time_limits": use_proper_time_limits,
        "use_huber_loss": use_huber_loss,
        "use_value_active_masks": use_value_active_masks,
        "use_policy_active_masks": use_policy_active_masks,
        "huber_delta": huber_delta,
        # ENVS
        "envs": envs,
    }

    
    wandb.config.update(cfg, allow_val_change=True)
    import sys
    if  hasattr(sys, 'gettrace') and sys.gettrace() is not None:
        runner = Runner(cfg)
        runner.run()
        envs.close()
        if not sweep:
            logger.info('finishing')
            run.finish()
    else:
        try:
            runner = Runner(cfg)
            runner.run()
            envs.close()
            if not sweep:
                logger.info('finishing')
                run.finish()
        except Exception as e:
            logging.exception(e)
            exit()
        