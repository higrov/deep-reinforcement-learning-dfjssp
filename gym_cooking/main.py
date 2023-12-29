from asyncio.log import logger
import logging

import wandb
from envs.overcooked_environment import OvercookedEnvironment

from recipe_planner.recipe import *
from utils.core import *
from misc.game.gameplay import GamePlay
from utils.world import World

from ddqnscheduler.scheduler import SchedulingAgent as Scheduler
from ddqnscheduler.parameter import *
from schedulingrules import *

import utils.utils as utils
import parsers as parsers
import sweep as sweep

import gymnasium as gym
from gymnasium.envs.registration import register

from envs.jobshop_env import JobShop
from schedule_generator import ScheduleGenerator
import random
from sklearn.model_selection import train_test_split
import pandas as pd

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

def getSchedule(train = True):
    scheduleGenerator =  ScheduleGenerator()
    listofglobalschedule = scheduleGenerator.generateSchedule()
    if train:
        train_schedule, _= train_test_split(listofglobalschedule, train_size=0.7)
        return train_schedule
    else:
        
        _, test_schedule= train_test_split(listofglobalschedule, train_size=0.7)
        return test_schedule


def train_loop():
    scheduler = Scheduler(nb_total_operations=10000, nb_input_params=4, nb_actions=4)
    rewards = []
    schedules= []
    log = pd.DataFrame(
            columns=['Episode', 'Duration', 'Score', 'Epsilon', 'min_loss'])

    listofglobalschedule = getSchedule(train = True)
    j = 0
    for i in range(MAX_EPISODE): # Training episodes
        # Start the job generation process
        globalSchedule = listofglobalschedule[j]
        globalSchedule= sorted(globalSchedule, key= lambda x: x.queued_at)
        job_shop = JobShop(scheduler= scheduler, num_machines=4, globalSchedule=globalSchedule)
        job_shop.run(1200)
        scheduler.replay()

        if i % UPDATE == 0:
            print("Target models update")
            scheduler.update_target_model()

        scheduler.policy.reset()
        
        if np.sum(job_shop.rewards) != 0:
            rewards.append((i, np.sum(job_shop.rewards)))
        max_reward = max(rewards, key= lambda x: x[1])

        schedules.append((job_shop.schedule, np.sum(job_shop.rewards)))

        j += 1

        if j>=len(listofglobalschedule):
            random.shuffle(listofglobalschedule)
            j = 0
    

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

    if arglist.train: 
        train_loop()
    