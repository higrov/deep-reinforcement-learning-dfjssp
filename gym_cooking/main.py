
from asyncio.log import logger
import logging

import wandb
from envs.overcooked_environment import OvercookedEnvironment
from rl import train_mappo
from rl.mappo.envs.overcooked_environment_ma_wrapper import OverCookedMAEnv

from recipe_planner.recipe import *
from utils.agent import RealAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from utils.world import World
from rl import train_ppo, train_seac

import simpy
from ddqnscheduler.scheduler import Agent as SchedulingAgent
from schedulingrules import *

import utils.utils as utils
import parsers as parsers
import sweep as sweep

import gymnasium as gym
from gymnasium.envs.registration import register

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: (%(process)d) [%(levelname).1s] - %(name)s: %(message)s",
    datefmt="%m/%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

def define_arglist():
    global global_arglist
    global_arglist = None

def change_arglist(val):
    global global_arglist
    global_arglist = val


def initialize_agents(arglist, orders: tuple[Order], jobshop_env, model_types, model_paths,) -> list[RealAgent]:
    real_agents = []

    with open(f"utils/levels/{arglist.level}.txt", "r") as f:
        phase = 1
        recipes = []

        RL = ["mappo", "ppo", "seac"]
        
        for line in f:
            line = line.strip("\n")

            if (
                line == ""
            ):  # empty line changes phase from level layout -> recipes -> agent locations
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(" ")
                    real_agent = RealAgent(
                        arglist=arglist,
                        name="agent-" + str(len(real_agents) + 1),
                        id_color=COLORS[len(real_agents)],
                        jobshop_env=jobshop_env,
                        capacity= 1
                        )
                    real_agents.append(real_agent)

    return real_agents


def initialize_jobShop(num_machines, capacity_per_machine, jobshop_env):
    machines= []
    for i in range(num_machines):
        newMachine = RealAgent(arglist=arglist,
                               jobshop_env=jobshop_env,
                               name= 'agent-'+str(len(i)+1),
                               capacity= capacity_per_machine,
                               id_color=COLORS[len(machines)]
                             )
        machines.append(newMachine)
    return machines


def eval_loop(arglist):
    """The main evaluation loop for running trials and experiments."""
    logger.info("Initializing environment and agents.")
    all_levels = arglist.level.split(',')
    run_id = arglist.run_id
    eval_group = run_id
    
    model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4, arglist.model5][:arglist.num_agents]
    model_paths = [arglist.model1_path, arglist.model2_path, arglist.model3_path, arglist.model4_path, arglist.model5_path][:arglist.num_agents]
    RL = ["mappo", "ppo", "seac"]
    BD = ["bd", "up", "fb", "dc", "greedy"]

    eval_columns = ['run_id', 'layout', 'model1', 'model2', 'model3', 'model4', 'model5', 'successful', 'failed', 'episode_length', 'episode_duration', 'deliveries', 'handovers', 'collisions', 'shuffles', 'invalid_actions', 'location_repeaters', 'holding_repeaters', 'order_1_delivery', 'order_2_delivery', 'order_3_delivery', 'order_4_delivery', 'order_5_delivery']
    
    eval_table = wandb.Table(columns=eval_columns)
    for level in all_levels:
        arglist.level = level.strip()
        arglist.run_id = f"{run_id}-{arglist.level}-{int(time.time())}"

        eval_dict = {k: 0 for k in eval_columns}
        eval_dict['run_id'] = arglist.run_id
        eval_dict['layout'] = arglist.level
        for i in range(5):
            eval_dict[f'model{i+1}'] = '' if i >= arglist.num_agents else model_types[i]
            eval_dict[f'order_{i+1}_delivery'] = 0


        trial = wandb.init(project="Paper-Results", id=arglist.run_id, name=arglist.run_id, group=eval_group, notes=arglist.notes, tags=parsers.parse_tags(arglist.tags), sync_tensorboard=False, resume="allow")
        trial_table = wandb.Table(columns=['layout', 'seed', *eval_columns[7:]])

        NUM_TRIALS = arglist.num_processes
        logger.info(f'Starting trials for level {arglist.level}')
        for i in range(1, NUM_TRIALS + 1):

            logger.info(f"Trial {i} of {NUM_TRIALS}")
            logger.info(f"Preparing env")
        
            if not any([model_type in RL for model_type in model_types]):
                env = gym.envs.make("overcookedEnv-v0", arglist=arglist)
            else:
                env: OvercookedEnvironment = gym.envs.make( # type: ignore
                    "overcookedEnv-v0", arglist=arglist, early_termination=False
                )
                # # wrapper env for multi-agent
                env: OverCookedMAEnv = OverCookedMAEnv.fromEnv(env)

            utils.fix_seed(i)
            env.seed(i)
            obs = env.reset()
                
            real_agents = initialize_agents(
                    arglist=arglist,
                    orders=env.orders,
                    model_types=model_types,
                    model_paths=model_paths,
                    env=env,
            )

            logger.info(f"Preparing agents")
            env.render()
            while not all(env.done()):
                #print(env.t)
                action_dict = {}
                for agent in real_agents:
                    t = env.t
                    agent_idx = int(agent.name[-1]) - 1
                    sim_agent = env.sim_agents[agent_idx]
                    # take action according to agent's model
                    if agent.model_type == "mappo":
                        osp = obs # MAPPO agents require our shaped observation space
                        action_dict[agent.name] = agent.select_action(t, osp, sim_agent)
                    else:
                        osp = env # BD agents require whole env object as obs
                        action_dict[agent.name] = agent.select_action(t, osp, sim_agent)

                obs, _, done, info = env.step(action_dict)
                # if all(done):
                #     print(env.termination_info)
                env.render()
                # Agents
                for agent in real_agents:
                    if agent.model_type not in RL:
                        agent.refresh_subtasks(remaining_orders=env.get_remaining_orders(), world=env.world)


            trial.log(env.termination_stats, step=i)
            trial_table.add_data(*[arglist.level, i, *list(env.termination_stats.values())])
            if arglist.record:
                anim_file = env.get_animation_path()
                trial.log({"animation": wandb.Video(anim_file, fps=4, format="gif")}, step=i)
            # update eval_dict with running sum for average later
            for k, v in env.termination_stats.items():
                eval_dict[k] += v


        env.close()
        trial.log({"run_stats": trial_table})
        
        # average termination stats of all trials
        for k, v in eval_dict.items():
            if k in ['run_id', 'layout', 'model1', 'model2', 'model3', 'model4', 'model5']:
                continue

            eval_dict[k] = v / NUM_TRIALS if k not in ['successful', 'failed'] else v

        eval_table.add_data(*list(eval_dict.values()))
        trial.finish()

    eval_run_summary = f'{run_id}_summary-{int(time.time())}'
    eval_run = wandb.init(project="Paper-Results", id=eval_run_summary, name=eval_run_summary, group=eval_group, notes=arglist.notes, tags=parsers.parse_tags(arglist.tags), sync_tensorboard=False, resume="allow")
    eval_run.log({"eval_stats": eval_table})
    eval_run.finish()

def train_loop(arglist):
    """The train loop for training RL Agents."""
    logger.info("Initializing environment and agents for training RL Agents.")
    arglist.run_id = arglist.run_id if arglist.continue_run else f"{arglist.run_id}{'-' if arglist.run_id else ''}{int(time.time())}"
    model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]

    overcooked_env: OvercookedEnvironment = gym.envs.make(
            "overcookedEnv-v0", arglist=arglist
        )
    
    overcooked_obs = overcooked_env.reset()
    jobshop = simpy.Environment()

    real_jobshop_machines = initialize_jobShop(num_machines=4, capacity_per_machine=1, jobshop_env= jobshop)

    ddqn = SchedulingAgent(nb_total_operations= all_operations, nb_input_params=4, nb_actions=4)

    state = [0,0,0,0]

    for i in range(arglist.num_episodes):

        while not overcooked_env.done():
            action_dict = {}

            scheduling_rule = ddqn.choose_action(state)

            selected_job, selected_machine = scheduling_rules[scheduling_rule](orders, real_jobshop_machines) 

            action_dict[selected_machine.name] = selected_job.get_next_action()

            obs, reward, done, info = overcooked_env.step(action_dict=action_dict)

            if(info['action_successful']):
                machine = next([machine for machine in real_jobshop_machines if machine.name == selected_machine.name])
                selected_action= selected_job.get_next_action()
                with machine.queue.request() as request:
                    yield request
                    print(f"{jobshop.now:.2f}: Job {selected_job.name}, operation {str(selected_action)} started on {machine.name}")
                    yield jobshop.process(machine.process_job(selected_job.name, str(selected_action), machine.get_processing_time(selected_action)))

    
    # if any(x == "ppo" for x in model_types):
    #     train_ppo.learn_ppo(
    #         env_id="overcookedEnv-v0",
    #         arglist=arglist,
    #         run_id=arglist.run_id,
    #         num_total_timesteps=arglist.num_total_timesteps,
    #         num_steps_per_update=arglist.num_steps_per_update,
    #         num_processes=arglist.num_processes,
    #         device=arglist.device,
    #         lr=arglist.lr,
    #         batch_size=arglist.batch_size,
    #         gamma=arglist.gamma,
    #         gae_lambda=arglist.gae_lambda,
    #         clip_range=arglist.clip_range,
    #         entropy_coef=arglist.entropy_coef,
    #         value_loss_coef=arglist.value_loss_coef,
    #         max_grad_norm=arglist.max_grad_norm,
    #         restore=True,
    #         notes=arglist.notes,
    #         tags=parsers.parse_tags(arglist.tags),
    #     )
    # if any(x == "seac" for x in model_types):
    #     train_seac.learn_seac(
    #         env_id="overcookedEnv-v0",
    #         arglist=arglist,
    #         run_id=arglist.run_id,
    #         num_episodes=arglist.num_episodes,
    #         num_steps_per_episode=arglist.num_timesteps_per_episode,
    #         num_processes=arglist.num_processes,
    #         device=arglist.device,
    #         lr=arglist.lr,
    #         adam_eps=arglist.adam_eps,
    #         use_gae=arglist.use_gae,
    #         gamma=arglist.gamma,
    #         value_loss_coef=arglist.value_loss_coef,
    #         entropy_coef=arglist.entropy_coef,
    #         seac_coef=arglist.seac_coef,
    #         max_grad_norm=arglist.max_grad_norm,
    #         restore=True,
    #         notes=arglist.notes,
    #         tags=parsers.parse_tags(arglist.tags),
    #     )
    # if any(x == "mappo" for x in model_types):
    #     train_mappo.learn_mappo(
    #         env_id="overcookedEnv-v0",
    #         arglist=arglist,
    #         run_id=arglist.run_id,
    #         num_total_timesteps=arglist.num_total_timesteps,
    #         num_processes=arglist.num_processes,
    #         device=arglist.device,
    #         share_policy=arglist.share_policy,
    #         use_centralized_v=arglist.use_centralized_v,
    #         hidden_size=arglist.hidden_size,
    #         layer_N=arglist.num_mlp_hidden_layers,
    #         use_popart=arglist.use_popart,
    #         use_valuenorm=arglist.use_valuenorm,
    #         use_feature_normalization=arglist.use_featurenorm,
    #         use_naive_recurrent_policy=arglist.use_naive_recurrent_policy,
    #         use_recurrent_policy=arglist.use_recurrent_policy,
    #         recurrent_N=arglist.num_rnn_hidden_layers,
    #         data_chunk_length=arglist.rnn_data_length,
    #         lr=arglist.lr,
    #         critic_lr=arglist.critic_lr,
    #         adam_eps=arglist.adam_eps,
    #         ppo_epoch=arglist.num_epoch,
    #         clip_param=arglist.clip_range,
    #         num_mini_batch=arglist.batch_size,
    #         entropy_coef=arglist.entropy_coef,
    #         value_loss_coef=arglist.value_loss_coef,
    #         max_grad_norm=arglist.max_grad_norm,
    #         use_gae=arglist.use_gae,
    #         gamma=arglist.gamma,
    #         gae_lambda=arglist.gae_lambda,
    #         restore=True,
    #         notes=arglist.notes,
    #         tags=parsers.parse_tags(arglist.tags),
    #     )

if __name__ == "__main__":
    # initializes command line arguments, all missing arguments have default values
    define_arglist()
    change_arglist(parsers.parse_arguments())
    # if path to config file is provided then command line / default arguments are overridden
    if global_arglist.config:
        config = parsers.parse_config(global_arglist.config)
        change_arglist(config)
        
    arglist = parsers.ArgList(**vars(global_arglist))
    # validating agent types
    model_types = [m for m in [arglist.model1, arglist.model2, arglist.model3, arglist.model4, arglist.model5] if m is not None]

    utils.fix_seed(seed=arglist.seed)
    register(
        id="overcookedEnv-v0",
        entry_point="envs:OvercookedEnvironment",
        )
    env: OvercookedEnvironment = gym.envs.make(
            "overcookedEnv-v0", arglist=arglist
        )
    if arglist.play:
        env: OvercookedEnvironment = gym.envs.make(
            "overcookedEnv-v0", arglist=arglist
        )
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_execute()
    elif arglist.train or arglist.sweep or arglist.evaluate:
        if arglist.train or arglist.sweep:    
            assert any(_ in ["mappo", "ppo", "seac"] for _ in model_types), "at least one agent must be trained with an RL algorithm for training mode. Please recheck your model types."
        else:
            assert len(model_types) == arglist.num_agents, "num_agents should match the number of models specified. Please recheck your config or arguments."

    if arglist.train:
        train_loop(arglist=arglist)

    elif arglist.sweep:
        if all(x == 'ppo' for x in model_types):
            sweep.train_loop_with_sweep_ppo(arglist=arglist)
        elif all(x == 'seac' for x in model_types):
            sweep.train_loop_with_sweep_seac(arglist=arglist)
        elif all(x == 'mappo' for x in model_types):
            sweep.train_loop_with_sweep_mappo(arglist=arglist)

    elif arglist.evaluate:
        eval_loop(arglist=arglist)
        
    else:
        raise ValueError("Please specify either --play, --train, --sweep, or --evaluate mode.")