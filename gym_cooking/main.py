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
    schedule_dataframe = pd.read_csv(f"{arglist.schedule_filename}.csv", index_col=0)
    group_schedule= schedule_dataframe.groupby('Time')

    eval_columns = ['run_id', 'layout', 'model1', 'model2', 'model3', 'model4', 'model5', 'successful', 'failed', 'episode_length', 'episode_duration', 'deliveries', 'handovers', 'collisions', 'shuffles', 'invalid_actions', 'location_repeaters', 'holding_repeaters', 'order_1_delivery', 'order_2_delivery', 'order_3_delivery', 'order_4_delivery', 'order_5_delivery']
    
    #eval_table = wandb.Table(columns=eval_columns)
    
    arglist.run_id = f"{run_id}-{arglist.level}-{int(time.time())}"

    eval_dict = {k: 0 for k in eval_columns}
    eval_dict['run_id'] = arglist.run_id
    eval_dict['layout'] = arglist.level
    for i in range(5):
        eval_dict[f'model{i+1}'] = '' if i >= arglist.num_agents else model_types[i]
        eval_dict[f'order_{i+1}_delivery'] = 0


    trial = wandb.init(project="Paper-Results", id=arglist.run_id, name=arglist.run_id, group=eval_group, notes=arglist.notes, tags=parsers.parse_tags(arglist.tags), sync_tensorboard=False, resume="allow")
    # trial_table = wandb.Table(columns=['layout', 'seed', *eval_columns[7:]])

    NUM_TRIALS = arglist.num_processes
    logger.info(f'Starting trials for level {arglist.level}')
    for i in range(1, NUM_TRIALS + 1):

        logger.info(f"Trial {i} of {NUM_TRIALS}")
        logger.info(f"Preparing env")
        
        env : OvercookedEnvironment = gym.envs.make("overcookedEnv-v0", arglist=arglist)

        utils.fix_seed(i)
        env.seed(i)
        obs = env.reset()
                
        logger.info(f"Preparing agents")
        env.render()
                #print(env.t)

        for group in list(group_schedule.groups):
            action_dict = {}
            agent_names = schedule_dataframe['Machine'].unique()
            for i in agent_names:
                action_dict[i] = None
            
            data = group_schedule.get_group(group)

            for index, row in data.iterrows():
                action_dict[row['Machine']] = (group, row['Task'],row['Points'])

            

            obs, _, done, info = env.step(action_dict)
            env.render()
                


    #trial.log(env.termination_stats, step=i)
    #trial_table.add_data(*[arglist.level, i, *list(env.termination_stats.values())])
    if arglist.record:
        anim_file = env.get_animation_path()
        trial.log({"animation": wandb.Video(anim_file, fps=4, format="gif")}, step=i)
            # update eval_dict with running sum for average later
    for k, v in env.termination_stats.items():
        eval_dict[k] += v


    env.close()
    #trial.log({"run_stats": trial_table})
        
        # average termination stats of all trials
    for k, v in eval_dict.items():
        if k in ['run_id', 'layout', 'model1', 'model2', 'model3', 'model4', 'model5']:
            continue

        eval_dict[k] = v / NUM_TRIALS if k not in ['successful', 'failed'] else v

    #eval_table.add_data(*list(eval_dict.values()))
    #trial.finish()

    eval_run_summary = f'{run_id}_summary-{int(time.time())}'
    eval_run = wandb.init(project="Paper-Results", id=eval_run_summary, name=eval_run_summary, group=eval_group, notes=arglist.notes, tags=parsers.parse_tags(arglist.tags), sync_tensorboard=False, resume="allow")
    #eval_run.log({"eval_stats": eval_table})
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

def test_loop(arglist):
    scheduler = Scheduler(nb_total_operations=10000, nb_input_params=4, nb_actions=4,train=False,network_model_file=arglist.model_filename)
    rewards = []
    schedules= []
    test_log = pd.DataFrame(
            columns=['Episode', 'Score', 'Epsilon', 'min_loss'])

    listofglobalschedule = getSchedule(train = False)
    j = 0
    for i in range(MAX_EPISODE): # Training episodes
        # Start the job generation process
        globalSchedule = listofglobalschedule[j]
        globalSchedule= sorted(globalSchedule, key= lambda x: x.queued_at)
        job_shop = JobShop(scheduler= scheduler, num_machines=4, globalSchedule=globalSchedule)
        job_shop.run(1200)
        min_loss= scheduler.replay()

        scheduler.policy.reset()
        
        if np.sum(job_shop.rewards) != 0:
            rewards.append((i, np.sum(job_shop.rewards)))

        max_reward = max(rewards, key= lambda x: x[1])

        schedules.append((job_shop.schedule, np.sum(job_shop.rewards)))

        j += 1

        if j>=len(listofglobalschedule):
            random.shuffle(listofglobalschedule)
            j = 0
        
        test_log = pd.concat([test_log,  pd.DataFrame([[i,np.sum(job_shop.rewards),scheduler.policy.epsilon, scheduler.min_loss]], columns = test_log.columns)], axis=0, ignore_index=True)

    max_reward_schedule = max(schedules, key= lambda x: x[1])

    max_reward_schedule[0].to_csv('./schedules/max_reward_schedule_test.csv')
    test_log.to_csv("./logs/test_log/" + "log-"+ "[" + str(MAX_EPISODE) + "]" + str(round(max_reward, 2)) + ".csv")

def train_loop():
    scheduler = Scheduler(nb_total_operations=10000, nb_input_params=4, nb_actions=4,train=True)
    rewards = []
    schedules= []
    log = pd.DataFrame(
            columns=['Episode', 'Score', 'Epsilon', 'min_loss'])

    listofglobalschedule = getSchedule(train = True)
    j = 0
    for i in range(MAX_EPISODE): # Training episodes
        # Start the job generation process
        globalSchedule = listofglobalschedule[j]
        globalSchedule= sorted(globalSchedule, key= lambda x: x.queued_at)
        job_shop = JobShop(scheduler= scheduler, num_machines=4, globalSchedule=globalSchedule)
        job_shop.run(1200)
        min_loss= scheduler.replay()

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
        
        log = pd.concat([log,  pd.DataFrame([[i,np.sum(job_shop.rewards),scheduler.policy.epsilon, scheduler.min_loss]], columns = log.columns)], axis=0, ignore_index=True)


    scheduler.model.save_model("./models/pretrained/DDQN/" + "DDQN-" + "[" + str(MAX_EPISODE) + "]" + str(round(max_reward, 2)) + ".h5")
    max_reward_schedule = max(schedules, key= lambda x: x[1])

    max_reward_schedule[0].to_csv('./schedules/max_reward_schedule.csv')
    log.to_csv("./logs/train_log/" + "log-"+ "[" + str(MAX_EPISODE) + "]" + str(round(max_reward, 2)) + ".csv")

    

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
    
    elif arglist.evaluate:
        eval_loop(arglist)
    elif arglist.test:
        test_loop(arglist)
    