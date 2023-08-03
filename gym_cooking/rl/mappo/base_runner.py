import shutil
import wandb
import os
import numpy as np
import torch

import logging
from asyncio.log import logger

from tensorboardX import SummaryWriter

from rl.mappo.utils.shared_buffer import SharedReplayBuffer

logger = logging.getLogger(__name__)

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.run_id = config["run_id"]
        self.envs = config['envs']
        self.device = config['device']
        self.record = config['record']
        self.num_agents = config['num_agents']
        self.max_num_agents = config['max_num_agents']  
        self.continue_run = config['continue_run']    
        # training parameters
        self.use_centralized_v = config['use_centralized_v'] 
        self.num_env_steps = config['num_env_steps']
        self.episode_length = config['episode_length']
        self.n_rollout_threads = config['n_rollout_threads']
        self.use_linear_lr_decay = config['use_linear_lr_decay']

        # network parameters
        self.hidden_size = config['hidden_size']
        self.recurrent_N = config['recurrent_N']

        # interval
        self.save_interval = config['save_interval']
        self.log_interval = config['log_interval']

        # dir
        self.model_dir = config['model_dir']
        
        self.run_dir = config["run_dir"]
        self.log_dir = config['log_dir']
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)
        #self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.previous_stats = {}
        
        from gym_cooking.rl.mappo.algo.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from gym_cooking.rl.mappo.algo.r_mappo.rMAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_v else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device = self.device, args=config)

        # algorithm
        self.trainer = TrainAlgo(self.policy, device=self.device, args=config)

        if config['restore'] and self.model_dir is not None and self.model_dir != './models':
            self.restore()
        
            
        # buffer
        self.buffer = SharedReplayBuffer(self.max_num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0],
                                        config,
                                        )


    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, num_total_steps):
        """Save policy's actor and critic networks."""
        cur_save_dir = os.path.join(self.save_dir, f'mappo_{num_total_steps}')
        running_save_dir = os.path.join('./models')
        os.makedirs(cur_save_dir, exist_ok=True)
        
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), f"{cur_save_dir}/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), f"{cur_save_dir}/critic.pt")

        actor_optimizer = self.trainer.policy.actor_optimizer
        torch.save(actor_optimizer.state_dict(), f"{cur_save_dir}/actor_optimizer.pt")
        critic_optimizer = self.trainer.policy.critic_optimizer
        torch.save(critic_optimizer.state_dict(), f"{cur_save_dir}/critic_optimizer.pt")

        stats_dict = {
            'total_steps': num_total_steps,
            'use_popart': self.trainer._use_popart,
            'use_valuenorm': self.trainer._use_valuenorm,
            'use_featurenorm': self.trainer._use_featurenorm,
        }
        torch.save(stats_dict, f"{cur_save_dir}/stats.pt")

        
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), f"{cur_save_dir}/vnorm.pt")

        archive_name = shutil.make_archive(cur_save_dir, "xztar", self.save_dir, f'mappo_{num_total_steps}')
        shutil.rmtree(cur_save_dir)
        art = wandb.Artifact(f'{num_total_steps}', type="model")
        art.add_file(archive_name)
        wandb.log_artifact(art)

        return archive_name
    
    def restore(self):
        """Restore policy's networks from a saved model."""
        shutil.unpack_archive(f"{self.model_dir}", f"{self.save_dir}", 'xztar')
        archive_name = self.model_dir.split('/')[-1].replace('.tar.xz', '')
        unpacked = f"{self.save_dir}/{archive_name}"

        policy_actor_state_dict = torch.load(f'{unpacked}/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        
        policy_critic_state_dict = torch.load(f'{unpacked}/critic.pt')
        self.policy.critic.load_state_dict(policy_critic_state_dict)

        actor_optimizer_state_dict = torch.load(f'{unpacked}/actor_optimizer.pt')
        self.policy.actor_optimizer.load_state_dict(actor_optimizer_state_dict)

        critic_optimizer_state_dict = torch.load(f'{unpacked}/critic_optimizer.pt')
        self.policy.critic_optimizer.load_state_dict(critic_optimizer_state_dict)
        
        self.previous_stats = torch.load(f'{unpacked}/stats.pt')
        
        if self.previous_stats.get('use_valuenorm', False):
            policy_vnorm_state_dict = torch.load(f'{unpacked}/vnorm.pt')
            self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)
            
        shutil.rmtree(unpacked)
        
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            wandb.log({f'model_metrics/{k}': v}, step=total_num_steps)
            #self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                wandb.log({k: np.mean(v)}, step=total_num_steps)
                #self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
