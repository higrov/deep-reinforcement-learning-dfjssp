from asyncio.log import logger
import logging

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

import gymnasium as gym
from gymnasium.envs.registration import register

from envs.jobshop_env import JobShop
from envs.jobshop_gym_env import JobShopEnv
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
            columns=['Episode', 'Score','Num Operations','num_jobs_completed'])

    listofglobalschedule = getSchedule(train = False)
    max_reward = 0
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
        
        test_log = pd.concat([test_log,  pd.DataFrame([[i,np.sum(job_shop.rewards),job_shop.num_op_exceuted,job_shop.jobs_completed]], columns = test_log.columns)], axis=0, ignore_index=True)

    max_reward_schedule = max(schedules, key= lambda x: x[1])

    max_reward_schedule[0].to_csv('./schedules/max_reward_schedule_test.csv')
    
    test_log.to_csv("./logs/test_log/" + "log-"+ "[" + str(len(listofglobalschedule)) + "]" + str(int(max_reward)) + ".csv")

    return test_log

def train_loop(arglist):
    schedules= []
    max_reward = 0
    log = pd.DataFrame(
            columns=['Episode', 'Score','Num Operations','num_jobs_completed', 'Epsilon', 'min_loss'])
    test_log = pd.DataFrame(
            columns=['Episode', 'Score','Num Operations','num_jobs_completed'])
    
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
            test_log.to_csv("./logs/test_log/" + "test_log-"+ "[10000]_" + str(i) + ".csv")
            
        
        # if np.sum(job_shop.rewards) != 0:
        #     rewards.append((i, np.sum(job_shop.rewards)))

        if(max_reward < np.sum(job_shop.rewards)):
            max_reward = np.sum(job_shop.rewards)
            schedules.append((job_shop.schedule, np.sum(job_shop.rewards)))

        j += 1

        if j>=len(listofglobalschedule):
            #random.shuffle(listofglobalschedule)
            j = 0
        
        log = pd.concat([log,  pd.DataFrame([[i,np.sum(job_shop.rewards),job_shop.num_op_exceuted,job_shop.jobs_completed,scheduler.policy.epsilon, scheduler.min_loss]], columns = log.columns)], axis=0, ignore_index=True)
        print(log)

        log.to_csv("./logs/train_log/" + "log-"+ "[" + str(MAX_EPISODE) + "]" + ".csv")

    scheduler.model.save_model("./models/pretrained/DDQN/" + "DDQN-" + "[" + str(MAX_EPISODE) + "]" + str(int(max_reward)) + ".h5")
    max_reward_schedule = max(schedules, key= lambda x: x[1])

    max_reward_schedule[0].to_csv('./schedules/max_reward_schedule.csv')
    log.to_csv("./logs/train_log/" + "log-"+ "[" + str(MAX_EPISODE) + "]" + str(int(max_reward)) + ".csv")
    #test_log.to_csv("./logs/test_log/" + "test_log-"+ "[" + str(10000) + "]" + ".csv")

def train_loop_test(arglist):
    schedules = []
    max_reward = -np.inf

    # set up log DataFrames
    log = pd.DataFrame(columns=[
        'Episode', 'Score', 'Num Operations', 'Num Jobs Completed', 'Epsilon', 'Min Loss'
    ])
    test_log = pd.DataFrame(columns=[
        'Episode', 'Score', 'Num Operations', 'Num Jobs Completed'
    ])

    # load training schedules
    listofglobalschedule = getSchedule(train=True)

    # count total operations once (for scheduler buffer sizing)
    total_ops = sum(
        len(order.recipe.actions)
        for sched in listofglobalschedule
        for order in sched
    )

    # initialize your DDQN agent
    scheduler = Scheduler(
        nb_total_operations=total_ops,
        nb_input_params=4,       # Gym env will infer obs-dim dynamically
        nb_actions=4,
        train=True
    )

    j = 0
    for episode in range(MAX_EPISODE):
        # pick next schedule and sort by arrival
        globalSchedule = sorted(listofglobalschedule[j], key=lambda o: o.queued_at)
        j = (j + 1) % len(listofglobalschedule)

        # create & reset Gym env
        env = JobShopEnv(global_schedule=globalSchedule, num_machines=4)
        state = env.reset()
        done = False

        total_reward = 0.0
        num_ops = 0
        num_jobs_done = 0
        while not done:
            action = scheduler.choose_action(state)
            next_state, reward, done, info = env.step(action)
            scheduler.observation(state, action, reward, next_state, done)
            total_reward += reward

            # update counters if you added these to your env
            num_ops += info.get('num_ops', 0)
            num_jobs_done = info.get('jobs_completed', num_jobs_done)

            state = next_state

        # after episode ends: train and log
        min_loss = scheduler.replay()
        if episode % UPDATE == 0:
            scheduler.update_target_model()
        scheduler.policy.reset()

        # periodic evaluation
        if episode and episode % 10000 == 0:
            scheduler.model.save_model(f"./models/pretrained/DDQN/trained/{arglist.model_filename}")
            tlog = test_loop(arglist)
            test_log = pd.concat([test_log, tlog], ignore_index=True)
            test_log.to_csv(f"./logs/test_log/test_log-[10000]_{episode}.csv")

        # track best schedule
        if total_reward > max_reward:
            max_reward = total_reward
            # if you’ve captured the schedule inside env, you can save it:
            # schedules.append((env.schedule_df, total_reward))

        # append to train log
        log = pd.concat([
            log,
            pd.DataFrame([[
                episode, total_reward, num_ops, num_jobs_done,
                scheduler.policy.epsilon, min_loss
            ]], columns=log.columns)
        ], ignore_index=True)

        if episode % 100 == 0: 
            print(log.tail(10))

        log.to_csv(f"./logs/train_log/log-[{MAX_EPISODE}].csv", index=False)

    # final save
    scheduler.model.save_model(f"./models/pretrained/DDQN/DDQN-[{MAX_EPISODE}]-{int(max_reward)}.h5")
    # and if you captured a best schedule DataFrame:
    # schedules[0][0].to_csv('./schedules/max_reward_schedule.csv')
    log.to_csv(f"./logs/train_log/log-[{MAX_EPISODE}]-{int(max_reward)}.csv", index=False)

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

    elif arglist.train: 
        train_loop_test(arglist)

    elif arglist.test:
        test_loop(arglist)
    