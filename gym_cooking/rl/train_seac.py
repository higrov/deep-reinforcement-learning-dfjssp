from asyncio.log import logger
import datetime
import logging
import shutil
import time
import os
from collections import deque

import numpy as np
import torch

import wandb

from rl.seac import a2c, utils

from rl.seac.envs import make_vec_envs
from rl.seac.wrappers import (
    DictifyActions,
    FlattenMultiObservation,
    GlobalizeReward,
    RecordEpisodeStatistics,
    SquashDones,
)

logger = logging.getLogger(__name__)


def learn_seac(
    env_id,
    arglist,
    run_id,
    num_episodes,
    num_steps_per_episode,
    num_processes,
    device,
    image_obs=False,
    wrappers=(GlobalizeReward, DictifyActions, FlattenMultiObservation, RecordEpisodeStatistics, SquashDones),
    save_dir=f"./models",
    load_dir=f"./models",
    log_dir=f"./logs",
    log_interval=500,
    save_interval=5000,
    lr=0.0003,
    adam_eps=0.005,
    use_gae=True,
    gamma=0.95,
    gae_lambda=0.95,
    value_loss_coef=0.5,
    entropy_coef=0.05,
    seac_coef=1.00,
    max_grad_norm=0.5,
    recurrent_policy=False,
    restore=True,
    notes="",
    tags=[],
    sweep=False,
):
    if not sweep:
        logger.info(f"Starting Training loop for run-id: {arglist.run_id}")
        run = wandb.init(project="Paper-Results", name=run_id, notes=notes, tags=[], mode="disabled")
        wandb.config.update(dict(zip(["lr", "adam_eps", "use_gae", "gamma", "gae_lambda", "value_loss_coef", "entropy_coef", "seac_coef", "max_grad_norm"], [lr, adam_eps, use_gae, gamma, gae_lambda, value_loss_coef, entropy_coef, seac_coef, max_grad_norm])))
        
    folder = utils.get_folder_name(arglist, suffix=f"/{run_id}/")
    
    if save_interval:
        save_dir = f"{save_dir}/{folder}"
        save_dir = os.path.expanduser(save_dir)
        utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(16) # ! This Number is only for the cluster
    
    logger.info(f"{'NOT ' if num_processes == 1 else ''}USING PARALLEL-PROCESSING: Preparing {num_processes} environment(s)")
    envs = make_vec_envs(
        env_id,
        arglist,
        run_id,
        image_obs,
        num_processes,
        wrappers,
        device,
    )

    logger.info(f"Preparing {len(envs.action_space)} SEAC agents per environment")
    agents = [
        a2c.A2C(
            i + 1, # Agent ID
            osp,
            asp,
            num_steps_per_episode,
            num_processes,
            device,
            lr=lr,
            adam_eps=adam_eps,
            recurrent_policy=recurrent_policy,
        )
        for i, (osp, asp) in enumerate(zip(envs.observation_space, envs.action_space))
    ][:arglist.num_agents]
    
    model_paths = [_ for _ in [arglist.model1_path, arglist.model2_path, arglist.model3_path, arglist.model4_path][:arglist.num_agents] if _ is not None and "seac" in _]
    if restore and any(model_paths):
        logger.info(f"Loading previous agent checkpoints")
        for agent, model in zip(agents, model_paths):
            if not model or not os.path.exists(f'{load_dir}/{model}'):
                raise ValueError(f"Model '{model}' does not exist at location {load_dir}. Please fix args or config.")
            else:
                shutil.unpack_archive(f"{load_dir}/{model}", f"{save_dir}", 'xztar')
                agent.restore(f"{save_dir}/{model.replace('.tar.xz', '')}")
                shutil.rmtree(f"{save_dir}/{model.replace('.tar.xz', '')}")
                
        
    obs = envs.reset()
    #envs.render()
    for i in range(len(agents)):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(device)
        
    start = time.time()
    
    all_infos = deque(maxlen=log_interval)
    env_metrics = []
    model_metrics = []
    total_timesteps = 0
    rewards = deque([], maxlen = num_steps_per_episode)

    logger.info(f"Training Start")
    for j in range(1, num_episodes + 1):
        logger.info(f"EPISODE: {j}/{num_episodes}, TOTAL STEPS: {total_timesteps} FPS: {int(total_timesteps/(time.time() - start))}")
        wandb.log({'episode_reward': sum(rewards) / (num_processes * rewards.maxlen) }, step=total_timesteps)
        rewards.clear()
        for step in range(num_steps_per_episode):
            # Sample actions
            with torch.no_grad():
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(
                    *[
                        agent.model.act(
                            agent.storage.obs[step],
                            agent.storage.recurrent_hidden_states[step],
                            agent.storage.masks[step],
                        )
                        for agent in agents
                    ]
                )
            # Observe reward and next obs
            obs, reward, done, infos = envs.step(n_action)
            #logger.info(infos)
            rewards.append(reward.sum())
            #envs.render()
            
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            bad_masks = torch.FloatTensor(
                [
                    [0.0] if info.get("TimeLimit.truncated", False) else [1.0]
                    for info in infos
                ]
            )
            
            for i in range(len(agents)):
                agents[i].storage.insert(
                    obs[i],
                    n_recurrent_hidden_states[i],
                    n_action[i],
                    n_action_log_prob[i],
                    n_value[i],
                    reward.unsqueeze(1),
                    masks,
                    bad_masks,
                )

            for info in infos:
                if info:
                    all_infos.append(info)
                    env_metrics.append(info)
                    
        total_timesteps += (num_steps_per_episode * num_processes)
        
        logger.info(f"Updating model trajectories...")
        agent_data = []
        for agent in agents:
            agent.compute_returns(use_gae=use_gae, gamma=gamma, gae_lambda=gae_lambda)

        for agent in agents:
            agent_stats = agent.update([a.storage for a in agents], value_loss_coef, entropy_coef, seac_coef, max_grad_norm)
            agent_data.append(agent_stats)
            
        for agent in agents:
            agent.storage.after_update()

        
        for k, v in utils._squash_info(agent_data).items():
            wandb.log({f"model_metrics/{k}": v}, step=total_timesteps)
                
        model_metrics.append(agent_data)
        logger.info(f"Updating model trajectories...Done")

        if ((total_timesteps % log_interval) == 0) and len(all_infos) > 1:
            squashed = utils._squash_info(all_infos)
            logger.info(f"Last {j} episodes mean reward: {np.mean(squashed['episode_reward']) if 'episode_reward' in squashed else 0:.3f}")
            logger.info(f"Time elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}s\n")
            for k, v in squashed.items():
                wandb.log({f"env_metrics/{k}": v}, step=total_timesteps)
            all_infos.clear()

        if save_interval is not None and ((total_timesteps % save_interval) == 0 or (total_timesteps % 10_000) == 0 or (j == num_episodes)):
            #logger.info(f"Saving models after {total_timesteps} steps.")
            # generate gif of episode
            anim_file = envs.env_method('generate_animation', *(100, f"_{total_timesteps}"))
            wandb.log({"animation": wandb.Video(anim_file[0], fps=4, format="gif")}, step=total_timesteps)
            cur_save_dir = os.path.join(save_dir, f'{run_id}_seac_{total_timesteps}')
            for agent in agents:
                os.makedirs(cur_save_dir, exist_ok=True)
                agent.save(cur_save_dir)
            archive_name = shutil.make_archive(cur_save_dir, "xztar", save_dir, f'{run_id}_seac_{total_timesteps}')
            shutil.rmtree(cur_save_dir)
            #logger.info(f"Model saved at {archive_name}")
            art = wandb.Artifact(f'{total_timesteps}', type="model")
            art.add_file(archive_name)
            wandb.log_artifact(art)
            
    envs.close()
    if not sweep:
        run.finish()



def evaluate_seac(
    agents,
    env_id=None,
    env_arglist=None,
    run_id=f'{int(time.time())}',
    image_obs=False,
    num_processes=4,
    wrappers=(GlobalizeReward, DictifyActions, FlattenMultiObservation, RecordEpisodeStatistics, SquashDones),
    device="cuda",
):

    eval_envs = make_vec_envs(
        env_id,
        env_arglist,
        run_id,
        image_obs,
        num_processes,
        wrappers,
        device,
    )

    n_obs = eval_envs.reset()
    #eval_envs.render()
    n_recurrent_hidden_states = [
        torch.zeros(
            num_processes, agent.model.recurrent_hidden_state_size, device=device
        )
        for agent in agents
    ]
    masks = torch.zeros(num_processes, 1, device=device)

    all_infos = []

    while len(all_infos) < num_processes:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(
                *[
                    agent.model.act(
                        n_obs[agent.agent_id], recurrent_hidden_states, masks
                    )
                    for agent, recurrent_hidden_states in zip(
                        agents, n_recurrent_hidden_states
                    )
                ]
            )

        # Observe reward and next obs
        n_obs, _, done, infos = eval_envs.step(n_action)
        #eval_envs.render()
        n_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
        all_infos.extend([i for i in infos if i])

    eval_envs.close()
    info = utils._squash_info(all_infos)
    reward = info["episode_reward"] if "episode_reward" in info else info["rewards"]
    logger.info(
        f"Evaluation using {len(all_infos)} episodes, Mean Reward: {reward:.5f}\n"
    )

