import argparse
import yaml
import time
import torch
from dataclasses import dataclass

def parse_tags(tags):
    return {tag.strip() for tag in tags.upper().split(',')} if tags is not None and tags.strip() != "" else []


def parse_config(config_file):
    with open(
        f"configs/{str(config_file).replace('.yml', '').replace('.yaml', '')}.yaml", "r"
    ) as f:
        config = yaml.safe_load(f)
    
    return parse_arguments(config)

def parse_run_arguments(parser, fromConfig):
    runParser = parser.add_argument_group("RunInfo")
    runParser.add_argument(
        "--run-id",
        type=str,
        default=fromConfig("run-id", str(int(time.time()))),
        help="The Identifier for this run",
    )
    runParser.add_argument(
        "--notes",
        type=str,
        default=fromConfig("notes", ""),
        help="What makes this run special",
    )
    runParser.add_argument(
        "--tags",
        type=str,
        default=fromConfig("tags", ""),
        help="Comma separated list of tags for this run",
    )
    runParser.add_argument(
        "--device",
        type=str,
        default=fromConfig("device", "cuda" if torch.cuda.is_available() else "cpu"),
        help="Select CPU or CUDA processing",
    )
    runParser.add_argument(
        "--continue-run",
        type=bool,
        default=fromConfig("continue-run", False),
        help="Continue a run from a checkpoint (specifies the wandb run name)",
    )
    
    return runParser

def parse_env_arguments(parser, fromConfig):

    envParser = parser.add_argument_group("Environment")

    envParser.add_argument(
        "--level",
        type=str,
        default=fromConfig("level", ""),
        help="Comma separated list of levels for this run, defaults to 1st when in --train mode",
    )
    envParser.add_argument(
        "--schedule_filename",
        type=str,
        default=fromConfig("schedule_filename", ""),
        help="Name of file of the schedule with Maximum Reward",
    )
    envParser.add_argument(
        "--model_filename",
        type=str,
        default=fromConfig("model_filename", "ddqn_model"),
        help="Name of the file with weights of the DDQN Model",
    )
    envParser.add_argument(
        "--num-agents",
        type=int,
        default=fromConfig("num-agents", 2),
        help="number of agents in the level",
    )
    envParser.add_argument(
        "--num-orders",
        type=int,
        default=fromConfig("num-orders", 3),
        help="Max number of orders per level",
    )
    envParser.add_argument(
        "--max-num-timesteps",
        type=int,
        default=fromConfig("max-num-timesteps", 200),
        help="Max number of timesteps before run ends",
    )
    envParser.add_argument(
        "--max-num-subtasks",
        type=int,
        default=fromConfig("max-num-subtasks", 14),
        help="Max number of subtasks for recipe",
    )
    envParser.add_argument(
        "--seed", type=int, default=fromConfig("seed", 1), help="Fix pseudorandom seed"
    )
    envParser.add_argument(
        "--num-processes",
        type=int,
        default=fromConfig("num-processes", 4),
        help="Parallel envs to train",
    )
    # Record run or not
    envParser.add_argument(
        "--record",
        type=bool,
        default=fromConfig("record", False),
        help="Save screenshot at each time step as an image in recordings folder",
    )
    # Randomize starting layout or not
    envParser.add_argument(
        "--randomize",
        type=bool,
        default=fromConfig("randomize", False),
        help="Randomizes starting layout (Agents, Stations and Ingredients)",
    )
    return envParser

def parse_model_arguments(parser, fromConfig):
    modelParser = parser.add_argument_group("Agent Models")
    # Valid options: `rl` = Reinforcement Learning; `bd` = Bayes Delegation; `up` = Uniform Priors `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    modelParser.add_argument(
        "--model1",
        type=str,
        default=fromConfig("model1", None),
        help="Model type for agent 1 (mappo, ppo, seac, bd, up, dc, fb, or greedy)",
    )
    modelParser.add_argument(
        "--model2",
        type=str,
        default=fromConfig("model2", None),
        help="Model type for agent 2 (mappo, ppo, seac, bd, up, dc, fb, or greedy)",
    )
    modelParser.add_argument(
        "--model3",
        type=str,
        default=fromConfig("model3", None),
        help="Model type for agent 3 (mappo, ppo, seac, bd, up, dc, fb, or greedy)",
    )
    modelParser.add_argument(
        "--model4",
        type=str,
        default=fromConfig("model4", None),
        help="Model type for agent 4 (mappo, ppo, seac, bd, up, dc, fb, or greedy)",
    )
    modelParser.add_argument(
        "--model5",
        type=str,
        default=fromConfig("model5", None),
        help="Model type for agent 5 (mappo, ppo, seac, bd, up, dc, fb, or greedy)",
    )
    # Model files to load pretrained agents
    modelParser.add_argument(
        "--model1-path",
        type=str,
        default=fromConfig("model1-path", None),
        help="Saved Model file for agent 1",
    )
    modelParser.add_argument(
        "--model2-path",
        type=str,
        default=fromConfig("model2-path", None),
        help="Saved Model file for agent 2",
    )
    modelParser.add_argument(
        "--model3-path",
        type=str,
        default=fromConfig("model3-path", None),
        help="Saved Model file for agent 3",
    )
    modelParser.add_argument(
        "--model4-path",
        type=str,
        default=fromConfig("model4-path", None),
        help="Saved Model file for agent 4",
    )
    modelParser.add_argument(
        "--model5-path",
        type=str,
        default=fromConfig("model5-path", None),
        help="Saved Model file for agent 5",
    )
    return modelParser

def parse_bd_arguments(parser, fromConfig):
    bdParser = parser.add_argument_group("Bayesian Delegation Parameters")
    # Delegation Planner
    bdParser.add_argument(
        "--beta",
        type=float,
        default=fromConfig("beta", 1.3),
        help="Beta for softmax in Bayesian delegation updates",
    )
    # Navigation Planner
    bdParser.add_argument(
        "--alpha", 
        type=float,
        default=fromConfig("alpha", 0.01),
        help="Alpha for BRTDP"
    )
    bdParser.add_argument(
        "--tau", 
        type=int, 
        default=fromConfig("tau", 2), 
        help="Normalize v diff"
    )
    bdParser.add_argument(
        "--cap",
        type=int,
        default=fromConfig("cap", 75),
        help="Max number of steps in each main loop of BRTDP",
    )
    bdParser.add_argument(
        "--main-cap",
        type=int,
        default=fromConfig("main-cap", 100),
        help="Max number of main loops in each run of BRTDP",
    )
    return bdParser

def parse_seac_arguments(parser, fromConfig):

    seacParser = parser.add_argument_group("SEAC Training HyperParameters")

    seacParser.add_argument(
        "--num-episodes",
        type=int,
        default=fromConfig("num-episodes", 100),
        help="Max number of episodes to run",
    )
    seacParser.add_argument(
        "--num-timesteps-per-episode",
        type=int,
        default=fromConfig("num-timesteps-per-episode", 200),
        help="Max number of steps per episodes before updating model",
    )
    seacParser.add_argument(
        "--adam-eps",
        type=float,
        default=fromConfig("adam-eps", 0.001),
        help="SEAC - HyperParam: Adam Optimizer Epsilon",
    )
    seacParser.add_argument(
        "--seac-coef",
        type=float,
        default=fromConfig("seac-coef", 1.00),
        help="SEAC - HyperParam: SEAC coefficient",
    )
    seacParser.add_argument(
        "--use-gae",
        type=bool,
        default=fromConfig("use-gae", False),
        help="SEAC - HyperParam: Use Generalized Advantage Estimation",
    )
    seacParser.add_argument(
        "--recurrent-policy",
        type=float,
        default=fromConfig("recurrent-policy", False),
        help="SEAC - HyperParam: recurrent policy",
    )
    
    return seacParser

def parse_ppo_arguments(parser, fromConfig):
    ppoParser = parser.add_argument_group("PPO Training HyperParameters")
    ppoParser.add_argument(
        "--num-total-timesteps",
        type=int,
        default=fromConfig("num-total-timesteps", 100_000),
        help="Max number of timesteps to run",
    )
    ppoParser.add_argument(
        "--num-steps-per-update",
        type=int,
        default=fromConfig("num-steps-per-update", 100),
        help="Max number of steps per episodes before updating model",
    )
    ppoParser.add_argument(
        "--batch-size",
        type=int,
        default=fromConfig("batch-size", 32),
        help="Size of a minibatch for PPO",
    )
    ppoParser.add_argument(
        "--lr",
        type=float,
        default=fromConfig("lr", 1e-4),
        help="RL - HyperParam: learning rate",
    )
    ppoParser.add_argument(
        "--gamma",
        type=float,
        default=fromConfig("gamma", 0.95),
        help="RL - HyperParam: discount factor",
    )
    ppoParser.add_argument(
        "--gae-lambda",
        type=float,
        default=fromConfig("gae-lambda", 0.95),
        help="RL - HyperParam: GAE lambda",
    )
    ppoParser.add_argument(
        "--clip-range",
        type=float,
        default=fromConfig("clip-range", 0.02),
        help="PPO - HyperParam: Clipping Parameter",
    )
    ppoParser.add_argument(
        "--entropy-coef",
        type=float,
        default=fromConfig("entropy-coef", 0.05),
        help="RL - HyperParam: entropy coefficient",
    )
    ppoParser.add_argument(
        "--value-loss-coef",
        type=float,
        default=fromConfig("value-loss-coef", 0.5),
        help="RL - HyperParam: value loss coefficient",
    )
    ppoParser.add_argument(
        "--max-grad-norm",
        type=float,
        default=fromConfig("max-grad-norm", 0.5),
        help="RL - HyperParam: max gradient norm",
    )
    return ppoParser

def parse_mappo_arguments(parser, fromConfig):
    mappoParser = parser.add_argument_group("MAPPO Training HyperParameters")
    mappoParser.add_argument(
        "--share-policy",
        type=bool,
        default=fromConfig("--share-policy", True),
        help="RL - HyperParam: To Share Policy between agents otherwise separate Replay Buffers will be used",
    )
    # similar for use-centralized-v
    mappoParser.add_argument("--use-centralized-v", 
        type=bool, 
        default=fromConfig("use-centralized-v", True), 
        help="RL - HyperParam: To share Value function between agents")

    mappoParser.add_argument("--hidden-size",
        type=int,
        default=fromConfig("hidden-size", 64),
        help="RL - HyperParam: Hidden layer size of the policy network")

    mappoParser.add_argument("--num-mlp-hidden-layers",
        type=int,
        default=fromConfig("num-mlp-hidden-layers", 1),
        help="RL - HyperParam: Number of hidden layers in the policy network")

    mappoParser.add_argument("--use-popart", 
        type=bool, 
        default=fromConfig("use-popart", True), 
        help="RL - HyperParam: To normalize rewards using popart")

    mappoParser.add_argument("--use-valuenorm", 
        type=bool, 
        default=fromConfig("use-valuenorm", False), 
        help="RL - HyperParam: To normalize value function. Only works if popart is disabled")
    
    mappoParser.add_argument("--use-featurenorm", 
        type=bool, 
        default=fromConfig("use-featurenorm", True), 
        help="RL - HyperParam: To normalize the feature space")

    mappoParser.add_argument("--use-naive-recurrent-policy", 
        type=bool, 
        default=fromConfig("use-naive-recurrent-policy", False), 
        help="RL - HyperParam: To use a naive policy for the RNN and train on whole trajectories")

    mappoParser.add_argument("--use-recurrent-policy",
        type=bool,
        default=fromConfig("use-recurrent-policy", True),
        help="RL - HyperParam: To use an RNN policy and train on Chunks of data")

    mappoParser.add_argument("--num-rnn-hidden-layers",
        type=int,
        default=fromConfig("num-rnn-hidden-layers", 1),
        help="RL - HyperParam: Number of recurrent layers in the policy network")

    mappoParser.add_argument("--rnn-data-length",
        type=int,
        default=fromConfig("rnn-data-length", 10),
        help="RL - HyperParam: Time length of chunks used to train a recurrent_policy")

    mappoParser.add_argument("--critic-lr", 
        type=float,
        default=fromConfig("critic-lr", 5e-4),
        help="RL - HyperParam: Critic learning rate")

    mappoParser.add_argument("--num-epoch",
        type=int,
        default=fromConfig("num-epoch", 10),
        help="RL - HyperParam: Number of epochs when optimizing the policy")
    
    return mappoParser

def parse_mode_arguments(parser, fromConfig):
    modeParser = parser.add_argument_group("Mode")

    modeParser.add_argument(
        "--play",
        type=bool,
        default=fromConfig("play", False),
        help="Play interactive game with keys",
    )
    modeParser.add_argument(
        "--train",
        type=bool,
        default=fromConfig("train", False),
        help="Train reinforcement learning agents",
    )
    modeParser.add_argument(
        "--sweep",
        type=bool,
        default=fromConfig("sweep", False),
        help="To Train models or sweep for hyperparameter optimization",
    )
    modeParser.add_argument(
        "--evaluate",
        type=bool,
        default=fromConfig("evaluate", False),
        help="Evaluate trained agents",
    )

    modeParser.add_argument(
        "--test",
        type=bool,
        default=fromConfig("test", False),
        help="Test trained scheduling agent",
    )
    
    return modeParser


def parse_arguments(config=None):
    fromConfig = lambda x, y: y if config is None or x not in config else config[x]

    parser = argparse.ArgumentParser("McFAT-RL argument parser")

    # Overall config file
    configParser = parser.add_argument_group("Config")
    configParser.add_argument(
        "--config", type=str, default="", help="Path to the config file"
    )
    # Run Info (name, notes, device, tags)
    runParser = parse_run_arguments(parser, fromConfig)

    # Environment switches (level, num-agents, max-num-timesteps, seed, num-processes, record)
    envParser = parse_env_arguments(parser, fromConfig)
    
    # Agent Models (model1, model2, model3, model4)
    modelParser = parse_model_arguments(parser, fromConfig)

    # Bayesian Delegation HParams (beta, alpha, tau, cap)
    bdParser = parse_bd_arguments(parser, fromConfig)

    # SEAC Training Hparams
    seacParser = parse_seac_arguments(parser, fromConfig)

    # PPO Training Hparams
    ppoParser = parse_ppo_arguments(parser, fromConfig)

    # MAPPO Training Hparams
    mappoParser = parse_mappo_arguments(parser, fromConfig)
    
    # Mode (train, evaluate, etc.)
    modeParser = parse_mode_arguments(parser, fromConfig)
    
    return parser.parse_args()



@dataclass
class ArgList:
    run_id: str
    notes: str
    tags: str
    device: str
    continue_run: bool
    level: str
    num_agents: int
    num_orders: int
    max_num_timesteps: int
    max_num_subtasks: int
    seed: int
    num_processes: int
    record: bool
    randomize: bool
    model1: str
    model2: str
    model3: str
    model4: str
    model5: str
    model1_path: str
    model2_path: str
    model3_path: str
    model4_path: str
    model5_path: str
    beta: float
    alpha: float
    tau: int
    cap: int
    main_cap: int
    num_episodes: int
    num_timesteps_per_episode: int
    adam_eps: float
    seac_coef: float
    use_gae: bool
    recurrent_policy: bool
    num_total_timesteps: int
    num_steps_per_update: int
    batch_size: int
    lr: float
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coef: float
    value_loss_coef: float
    max_grad_norm: float
    share_policy: bool
    use_centralized_v: bool
    hidden_size: int
    num_mlp_hidden_layers: int
    use_popart: bool
    use_valuenorm: bool
    use_featurenorm: bool
    use_naive_recurrent_policy: bool
    use_recurrent_policy: bool
    num_rnn_hidden_layers: int
    rnn_data_length: int
    critic_lr: float
    num_epoch: int
    play: bool
    train: bool
    sweep: bool
    evaluate: bool
    test : bool
    schedule_filename : str
    model_filename : str
    config: str