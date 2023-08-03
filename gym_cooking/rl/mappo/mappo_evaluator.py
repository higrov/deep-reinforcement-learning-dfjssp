from collections import defaultdict
import os
import shutil

import numpy as np
import torch

from rl.mappo.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()

class MAPPOEvaluator():
    def __init__(self, env, idx, device, config):

        # network parameters
        self.hidden_size = config.hidden_size
        self.layer_N = config.num_mlp_hidden_layers
        self.recurrent_N = config.num_rnn_hidden_layers
        self.max_num_agents = env.max_num_agents
        self.use_centralized_v = config.use_centralized_v
        self.max_num_timesteps = config.max_num_timesteps
        self.agent_idx = idx
        self.cfg = vars(config)
        self.cfg.update({
        # RUN PARAMS
        "all_args": config,
        "max_num_agents": self.max_num_agents,
        "n_rollout_threads": 1,
        "num_env_steps": config.max_num_timesteps,
        "stacked_frames": 1,
        "use_stacked_frames": False,
        "hidden_size": self.hidden_size,
        "layer_N": self.layer_N,
        "use_ReLU": True,
        "use_popart": config.use_popart,
        "use_valuenorm": False and not config.use_popart,
        "use_feature_normalization": config.use_featurenorm,
        "use_orthogonal": True,
        "gain": 0.01,
        # RECURRENT PARAMS
        "use_naive_recurrent_policy": True,
        "use_recurrent_policy": True,
        "recurrent_N": self.recurrent_N,
        "data_chunk_length": 10,
        # OTPIMIZER PARAMS
        "lr": 1e-3,#
        "critic_lr": 1e-3,#
        "adam_eps": 5e-5,#
        "weight_decay": 1.0,#
        # PPO PARAMS
        "ppo_epoch": 10,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "num_mini_batch": 1,
        "entropy_coef": 0.01,
        "value_loss_coef": 1.0,
        "use_max_grad_norm": True,
        "max_grad_norm": 0.5,
        "use_gae": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "use_proper_time_limits": False,
        "use_huber_loss": True,
        "use_value_active_masks": True,#
        "use_policy_active_masks": True, #
        "huber_delta": 10,
        "episode_length": self.max_num_timesteps,
    })
        
        
        from gym_cooking.rl.mappo.algo.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from gym_cooking.rl.mappo.algo.r_mappo.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = env.share_observation_space[0] if self.use_centralized_v else env.observation_space[idx]
        
        self.policy = Policy(env.observation_space[self.agent_idx], share_observation_space, env.action_space[self.agent_idx], device=device, args=self.cfg)

        self.trainer = TrainAlgo(self.policy, device=device, args=self.cfg)

    def restore(self, path):
        """Restore policy's networks from a saved model."""
        model_path = os.path.expanduser(path)
        shutil.unpack_archive(f"{model_path}", f"./models", 'xztar')
        archive_name = path.split('/')[-1].replace('.tar.xz', '')
        unpacked = f"./models/{archive_name}"

        policy_actor_state_dict = torch.load(f'{unpacked}/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        
        policy_critic_state_dict = torch.load(f'{unpacked}/critic.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)
        
        actor_optimizer_state_dict = torch.load(f'{unpacked}/actor_optimizer.pt')
        self.policy.actor_optimizer.load_state_dict(actor_optimizer_state_dict)

        critic_optimizer_state_dict = torch.load(f'{unpacked}/critic_optimizer.pt')
        self.policy.critic_optimizer.load_state_dict(critic_optimizer_state_dict)
        
        if self.trainer._use_valuenorm:
            policy_vnorm_state_dict = torch.load(f'{unpacked}/vnorm.pt')
            self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

        shutil.rmtree(unpacked)


    def predict(self, obs, t, deterministic=True):
        if t is None or t == 0:
            self.eval_rnn_states = np.zeros((1, self.max_num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            self.eval_masks = np.ones((1, self.max_num_agents, 1), dtype=np.float32)
            
            # get actions
            self.trainer.prep_rollout()

        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        self.eval_actions, self.eval_rnn_states = self.trainer.policy.act(
            np.array(obs),
            np.concatenate(self.eval_rnn_states),
            np.concatenate(self.eval_masks),
            deterministic=deterministic
        )

        self.eval_actions = np.array(np.split(_t2n(self.eval_actions), 1))
        self.eval_rnn_states = np.array(np.split(_t2n(self.eval_rnn_states), 1))

        eval_actions = self.eval_actions[0, :, 0]

        return eval_actions