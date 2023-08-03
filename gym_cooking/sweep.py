def learn_with_sweep():
    os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = 'true'
    run = wandb.init(project="Paper-Results")
    model_types = [m for m in [arglist.model1, arglist.model2, arglist.model3, arglist.model4] if m is not None]

    if all(x == "seac" for x in model_types):
        train_seac.learn_seac(wandb.config.env_id,
        arglist,
        wandb.config.run_id,
        wandb.config.num_episodes,
        wandb.config.num_steps_per_episode,
        wandb.config.num_processes,
        wandb.config.device,
        lr=wandb.config.lr,
        adam_eps=wandb.config.adam_eps,
        use_gae=wandb.config.use_gae,
        gamma=wandb.config.gamma,
        gae_lambda=wandb.config.gae_lambda,
        value_loss_coef=wandb.config.value_loss_coef,
        entropy_coef=wandb.config.entropy_coef,
        seac_coef=wandb.config.seac_coef,
        max_grad_norm=wandb.config.max_grad_norm,
        sweep=True)
    elif all(x == "ppo" for x in model_types):
        train_ppo.learn_ppo(wandb.config.env_id, 
        arglist, 
        wandb.config.run_id, 
        wandb.config.num_total_timesteps,
        wandb.config.num_steps_per_update,
        wandb.config.num_processes,
        wandb.config.device,
        lr=wandb.config.lr,
        batch_size=wandb.config.batch_size,
        gamma=wandb.config.gamma,
        gae_lambda=wandb.config.gae_lambda,
        clip_range=wandb.config.clip_range,
        entropy_coef=wandb.config.entropy_coef,
        value_loss_coef=wandb.config.value_loss_coef,
        max_grad_norm=wandb.config.max_grad_norm,
        sweep=True)
    elif all(x == "mappo" for x in model_types):
        train_mappo.learn_mappo(wandb.config.env_id, arglist, 
        wandb.config.run_id, 
        wandb.config.num_total_timesteps,
        wandb.config.num_processes, 
        wandb.config.device, 
        lr=wandb.config.lr,
        critic_lr=wandb.config.critic_lr, 
        ppo_epoch=wandb.config.ppo_epoch, 
        num_mini_batch=wandb.config.num_mini_batch,
        sweep=True)
    
    run.finish()



def train_loop_with_sweep_seac(arglist):
    """The train loop for tuning SEAC Hyperparameters."""
    arglist.run_id = ((arglist.run_id + '-') if arglist.run_id else 'SEAC-SWEEP-') + str(int(time.time()))
    logger.info(f"Initializing sweep controller for and agents for training SEAC Agents. Sweep ID: {arglist.run_id}")
    sweep_config = {
    'method': 'bayes',
    'name': arglist.run_id,
    'metric': {
        'goal': 'maximize', 
        'name': 'episode_reward'
		},
    'parameters': {
        'env_id': { 'value': "gym_cooking:overcookedEnv-v0"},
        'arglist': { 'value': arglist },
        'run_id': { 'value': arglist.run_id},
        'num_episodes': {'value': arglist.num_episodes},
        'num_processes': {'value': arglist.num_processes},
        'num_steps_per_episode': {'value': 100},
        'device': {'value': arglist.device},
        'randomize': {'value': arglist.randomize},
        'seac_coef': {'value': 1.0},
        # HPARAMS
        # fixed
        'value_loss_coef': {'value': 0.5},
        'max_grad_norm': {'value': 0.5},
        'use_gae': {'value': False},
        'gae_lambda': {'value': 0.9},
        'adam_eps': {'value': 0.001 },
        # Optimizing
        'lr': {'min': 0.0001, 'max': 0.001},
        'gamma': {'min': 0.9, 'max': 0.99},
        'entropy_coef': {'min': 0.01, 'max': 0.05}, 
        }
    }
    
    sweep_id = wandb.sweep(sweep=sweep_config, project='Paper-Results', )
    wandb.agent(sweep_id, function=learn_with_sweep, count=50)    



def train_loop_with_sweep_ppo(arglist):
    """The train loop for tuning PPO Hyperparameters."""
    arglist.run_id = ((arglist.run_id + '-') if arglist.run_id else 'PPO-SWEEP-') + str(int(time.time()))
    logger.info(f"Initializing sweep controller for and agents for training PPO Agents. Sweep ID: {arglist.run_id}")
    sweep_config = {
    'method': 'bayes',
    'name': arglist.run_id,
    'metric': {
        'goal': 'maximize', 
        'name': 'env_metrics/reward'
		},
    'parameters': {
        'env_id': { 'value': "gym_cooking:overcookedEnv-v0"},
        'arglist': { 'value': arglist },
        'run_id': { 'value': arglist.run_id},
        'num_total_timesteps': {'value': arglist.num_total_timesteps},
        'num_processes': {'value': arglist.num_processes},
        'device': {'value': arglist.device},
        'randomize': {'value': arglist.randomize},
        # HPARAMS
        # fixed
        'max_grad_norm': {'value': 0.5},
        # optimizing
        'lr': {'min': 3e-6, 'max': 0.001},
        'batch_size': {'values': [32, 64, 512]},
        'num_steps_per_update': {'values': [16, 32, 512]},
        'gamma': {'min': 0.9, 'max': 0.99},
        'gae_lambda': {'min': 0.9, 'max': 0.99},
        'clip_range': {'min': 0.01, 'max': 0.2},
        'entropy_coef': {'values': [0.01, 0.05, 0.1, 0.25, 0.5]},
        'value_loss_coef': {'values': [0.01, 0.05, 0.1, 0.25, 0.5]},
        }
    }
    
    sweep_id = wandb.sweep(sweep=sweep_config, project='Paper-Results', )
    wandb.agent(sweep_id, function=learn_with_sweep, count=50)    


def train_loop_with_sweep_mappo(arglist):
    """The train loop for tuning MAPPO Hyperparameters."""
    arglist.run_id = ((arglist.run_id + '-') if arglist.run_id else 'MAPPO-SWEEP-') + str(int(time.time()))
    logger.info(f"Initializing sweep controller for and agents for training MAPPO Agents. Sweep ID: {arglist.run_id}")
    sweep_config = {
    'method': 'bayes',
    'name': arglist.run_id,
    'metric': {
        'goal': 'maximize', 
        'name': 'env_metrics/reward'
		},
    'parameters': {
        'env_id': { 'value': "gym_cooking:overcookedEnv-v0"},
        'arglist': { 'value': arglist },
        'run_id': { 'value': arglist.run_id},
        'num_total_timesteps': {'value': 2_000_000},
        'num_processes': {'value': arglist.num_processes},
        'device': {'value': arglist.device},
        # HPARAMS
        "lr": {'values': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]},
        "critic_lr": {'values': [0.00009, 0.0001, 0.0002, 0.0003]},
        "ppo_epoch": {'values': [5, 10, 15]},
        "num_mini_batch": {'values': [2, 4, 8, 16]},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project='Paper-Results', )
    wandb.agent(sweep_id, function=learn_with_sweep, count=50)  

