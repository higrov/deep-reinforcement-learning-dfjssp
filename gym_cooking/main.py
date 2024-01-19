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
    run_id = arglist.run_id

    schedule_dataframe = pd.read_csv(f"./schedules/{arglist.schedule_filename}.csv", index_col=0)
    group_schedule= schedule_dataframe.groupby('Time')
    
    arglist.run_id = f"{run_id}-{arglist.level}-{int(time.time())}"

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
        i = 0
        for group in list(group_schedule.groups):
            action_dict = {}
            agent_names = schedule_dataframe['Machine'].unique()
            for i in agent_names:
                action_dict[i] = None
            
            data = group_schedule.get_group(group)

            for index, row in data.iterrows():
                action_dict[row['Machine']] = (group, row['Task'],row['Points'])
            i = group
            obs, _, done, info = env.step(action_dict)
            env.render()
                

    env.close()


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
    scheduler = Scheduler(nb_total_operations=10000, nb_input_params=4, nb_actions=4,train=False,
                          network_model_file="./models/pretrained/DDQN/trained/" + arglist.model_filename)

    schedules= []
    test_log = pd.DataFrame(
            columns=['Episode', 'Score','Num Operations'])

    listofglobalschedule = getSchedule(train = False)
    j = 0
    for i in range(len(listofglobalschedule)): # Training episodes
        # Start the job generation process
        globalSchedule = listofglobalschedule[j]
        globalSchedule= sorted(globalSchedule, key= lambda x: x.queued_at)
        job_shop = JobShop(scheduler= scheduler, num_machines=4, globalSchedule=globalSchedule)
        job_shop.run(1200000)
        
        # if np.sum(job_shop.rewards) != 0:
        #     rewards.append((i, np.sum(job_shop.rewards)))

        if(max_reward < np.sum(job_shop.rewards)):
            max_reward = np.sum(job_shop.rewards)
            schedules.append((job_shop.schedule, np.sum(job_shop.rewards)))


        j += 1

        if j>=len(listofglobalschedule):
            random.shuffle(listofglobalschedule)
            j = 0
        
        test_log = pd.concat([test_log,  pd.DataFrame([[i,np.sum(job_shop.rewards),job_shop.num_op_exceuted]], columns = test_log.columns)], axis=0, ignore_index=True)

    #max_reward_schedule = max(schedules, key= lambda x: x[1])

    # max_reward_schedule[0].to_csv('./schedules/max_reward_schedule_test.csv')
    # test_log.to_csv("./logs/test_log/" + "log-"+ "[" + str(len(listofglobalschedule)) + "]" + str(round(max_reward[1], 2)) + ".csv")

    return test_log

def train_loop(arglist):
    schedules= []
    max_reward = 0
    log = pd.DataFrame(
            columns=['Episode', 'Score','Num Operations', 'Epsilon', 'min_loss'])
    test_log = pd.DataFrame(
            columns=['Episode', 'Score','Num Operations'])

    listofglobalschedule = getSchedule(train = True)
    n_operations = 0
    for schedule in listofglobalschedule:
        for order in schedule:
            n_operations += len(order.recipe.actions)

    scheduler = Scheduler(nb_total_operations=n_operations, nb_input_params=4, nb_actions=4,train=True)
    j = 0
    for i in range(MAX_EPISODE): # Training episodes
        # Start the job generation process
        globalSchedule = listofglobalschedule[j]
        globalSchedule= sorted(globalSchedule, key= lambda x: x.queued_at)
        job_shop = JobShop(scheduler= scheduler, num_machines=4, globalSchedule=globalSchedule)
        job_shop.run(1200000)
        min_loss= scheduler.replay()

        if i % UPDATE == 0:
            print("Target models update")
            scheduler.update_target_model()

        scheduler.policy.reset()

        if(i % 10000 == 0):
            scheduler.model.save_model("./models/pretrained/DDQN/trained/" + arglist.model_filename)
            test_log_temp= test_loop(arglist)
            test_log = pd.concat([test_log, test_log_temp], axis=0, ignore_index=True)
            test_log.to_csv("./logs/test_log/" + "test_log-"+ "[10000]" + ".csv")
            
        
        # if np.sum(job_shop.rewards) != 0:
        #     rewards.append((i, np.sum(job_shop.rewards)))

        if(max_reward < np.sum(job_shop.rewards)):
            max_reward = np.sum(job_shop.rewards)
            schedules.append((job_shop.schedule, np.sum(job_shop.rewards)))

        j += 1

        if j>=len(listofglobalschedule):
            #random.shuffle(listofglobalschedule)
            j = 0
        
        log = pd.concat([log,  pd.DataFrame([[i,np.sum(job_shop.rewards),job_shop.num_op_exceuted,scheduler.policy.epsilon, scheduler.min_loss]], columns = log.columns)], axis=0, ignore_index=True)
        print(log)


    scheduler.model.save_model("./models/pretrained/DDQN/" + "DDQN-" + "[" + str(MAX_EPISODE) + "]" + str(round(max_reward[1], 2)) + ".h5")
    max_reward_schedule = max(schedules, key= lambda x: x[1])

    max_reward_schedule[0].to_csv('./schedules/max_reward_schedule.csv')
    log.to_csv("./logs/train_log/" + "log-"+ "[" + str(MAX_EPISODE) + "]" + str(round(max_reward[1], 2)) + ".csv")
    #test_log.to_csv("./logs/test_log/" + "test_log-"+ "[" + str(10000) + "]" + ".csv")

    

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
        train_loop(arglist)
    
    elif arglist.evaluate:
        eval_loop(arglist)
    elif arglist.test:
        test_loop(arglist)
    