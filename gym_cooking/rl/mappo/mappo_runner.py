from collections import defaultdict
from asyncio.log import logger
import logging
import shutil
import time
import warnings
from rl.seac import utils
from rl.mappo.base_runner import Runner

import numpy as np
import torch
import wandb


logger = logging.getLogger(__name__)

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None

def _t2n(x):
    return x.detach().cpu().numpy()

class MAPPORunner(Runner):
    def __init__(self, config):
        super(MAPPORunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        if tqdm is None:
            raise ImportError(
                "You must install tqdm and rich in order to use the progress bar callback. "
                "It is included if you install stable-baselines with the extra packages: "
                "`pip install stable-baselines3[extra]`"
            )
        self.pbar = None
        

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = int(self.previous_stats.get('total_steps', 0)) if self.continue_run else 0
        offset = total_num_steps // self.episode_length // self.n_rollout_threads     
        all_infos = []

        self.pbar = tqdm(total=self.num_env_steps)
        self.pbar.update(total_num_steps)
        
        for episode in range(offset, episodes):
            
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # insert data into buffer
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)
                # update progress bar
                self.pbar.update(self.n_rollout_threads)
                # unfortunately episode gif has to be logged here, the env resets as soon as the last step is reached.
                # reset deletes older screenshots, and we want to avoid constantly generating gifs at each episode during training for performance reasons
                # the final gifs will of course be only n-1 timesteps instead of the full episode
                if self.record and episode % 5 == 0 and step == self.episode_length - 2:
                    self.log_animation(total_num_steps)
                # log env infos
                for info in infos:
                    if info:
                        all_infos.append(info)
                
            # compute return and update network
            self.compute()
            train_infos = self.train()
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                #logger.info(f"Saving model after {total_num_steps} steps.")
                path = self.save(total_num_steps)
                #logger.info(f"Model saved at {path}")

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                #logger.info(f"\nRun {self.run_id} updates {episode + 1}/{episodes} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}, FPS {int(total_num_steps) / (end-start)}.\n")
                train_infos["episode_reward"] = np.mean(self.buffer.rewards) * self.episode_length
                #logger.info(f"average episode rewards is {train_infos['episode_reward']}")
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.log_squashed(all_infos, total_num_steps)
                all_infos.clear()
                self.env_infos = defaultdict(list)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # insert obs to buffer
        self.buffer.share_obs[0] = obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step])
        )

        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        dones_env = np.all(dones, axis=1)

        # reset rnn and mask args for done envs
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.max_num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.max_num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.max_num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.max_num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.max_num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.max_num_agents, 1), dtype=np.float32)

        self.buffer.insert(
            share_obs=obs,
            obs=obs,
            rnn_states_actor=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks,
            active_masks=active_masks,
        )

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                wandb.log({k: np.mean(v)}, step=total_num_steps)
                #self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    def log_squashed(self, all_infos, total_num_steps):
        squashed = utils._squash_info(all_infos)

        for k, v in squashed.items():
            wandb.log({f"env_metrics/{k}": v}, step=total_num_steps)
            # self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def log_animation(self, total_num_steps):
        anim_file = self.envs.env_method('generate_animation', *(100, f"_{total_num_steps}"))
        wandb.log({"animation": wandb.Video(anim_file[0], fps=4, format="gif")})