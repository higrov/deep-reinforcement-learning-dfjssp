import numpy as np

from utils.agent import RealAgent, COLORS
from schedulingrules import scheduling_rules
from utils.core import *
from recipe_planner.recipe import *

schedule= [Order(Salad(), (0,7), 0, (30,40)), 
              Order(SimpleTomato(), (1,7), 2, (35,45)),
              Order(Salad(), (1,7), 10, (35,45)),
              Order(SimpleLettuce(), (1,7), 10, (35,45)),
              Order(Salad(), (1,7), 12, (35,45)),
              Order(SimpleTomato(), (1,7), 15, (35,45)),
              Order(SimpleTomato(), (0,7), 20, (30,40)), 
              Order(Salad(), (1,7), 27, (35,45))]

class MockJobshop: 
    def __init__(self, num_machines, jobshop_env):
        self.env = jobshop_env
        self.machines = self.initialize_jobShop(num_machines, capacity_per_machine = 1, jobshop_env=self.env)
        self.penalty = 0
        self.timestamp = 0
        self.schedule = copy.deepcopy(schedule)




    def initialize_jobShop(self,num_machines, capacity_per_machine, jobshop_env):
        machines= []
        for i in range(num_machines):
            newMachine = RealAgent(arglist=None,
                                   jobshop_env=jobshop_env,
                                   name= 'agent-'+str(len(machines)+1),
                                   capacity= capacity_per_machine,
                                   id_color=COLORS[len(machines)]
                                 )
            machines.append(newMachine)
        return machines

    def step(self,action):

        uncompleted_jobs = [job for job in self.schedule if not job.get_completed()]

        test = schedule
        
        scheduling_rule = scheduling_rules[action]

        selected_job , selected_machine = scheduling_rule(uncompleted_jobs, self.machines)

        selected_operation = selected_job.get_next_operation()

        time_to_execute = selected_machine.possible_operations[selected_operation.__class__]

        self.timestamp += time_to_execute
        
        selected_machine.set_last_operation_executed(selected_operation)
        selected_machine.set_last_operation_performed_at(self.timestamp)
        selected_job.add_completed_tasks(selected_operation,selected_machine.name,self.timestamp) 


        # selected_machine.perform_operation(selected_operation)


        #State Features
        average_utilization_rate = calculate_average_utilization_rate(schedule= uncompleted_jobs, list_machines=self.machines)

        estimated_earliness_tardiness_rate= calculate_estimated_earliness_tardiness_rate(schedule= uncompleted_jobs, list_machines=self.machines)
        
        actual_earliness_tardiness_rate = calculate_actual_earliness_tardiness_rate(schedule= uncompleted_jobs, list_machines=self.machines)
        
        penalty= actual_penalty_cost(schedule= uncompleted_jobs, list_machines=self.machines)

        breakpt= True
        reward = self.penalty - penalty

        self.penalty = penalty

        state = [average_utilization_rate, estimated_earliness_tardiness_rate,actual_earliness_tardiness_rate, penalty]

        done = self.calculate_done()

        return state, reward, done
    

    def calculate_done(self):
        if self.timestamp == 1200:
            return True
        if all(job.get_completed() for job in self.schedule):
            return True
        return False
    
    def reset(self):
        self.schedule = copy.deepcopy(schedule)
        self.timestamp = 0
        self.penalty = 0

# State Features

def calculate_average_utilization_rate(schedule, list_machines):
    machine_util = 0
    num_machines = len(list_machines)

    for machine in list_machines:
        job_util = 0
        for job in schedule:
            task_util = 0 
            completed_operations_machine =[oper for oper in job.completed_task_machine if oper[1] == machine.name]

            for (operation, machine_name) in completed_operations_machine:
                processing_time = machine.get_possible_operations()[operation.__class__]
                task_util += processing_time
        
            job_util+= task_util
        
        machine_util += (job_util/machine.last_operation_executed_at)

    average_utilization_rate = machine_util / num_machines  # Calculate average over tasks and jobs

    return average_utilization_rate

def calculate_estimated_earliness_tardiness_rate(schedule, list_machines):
    Tcur = sum([machine.last_operation_executed_at if machine.last_operation_executed_at<0 else 0 for machine in list_machines])/len(list_machines)

    NJtard = 0
    NJearly = 0
    
    for job in schedule: 
        if len(job.completed_tasks)< len(job.tasks):
            Tleft = 0
            
            left_tasks= [task for task in job.tasks if task not in job.completed_tasks]

            for left_task in left_tasks: 
                tij = np.sum([machine.get_possible_operations()[left_task.__class__] for machine in list_machines])/len(list_machines)
                Tleft += tij
                if Tcur + Tleft > job.delivery_window[1]:
                    NJtard += 1
                    break
            
            if Tleft + Tcur < job.delivery_window[0]:
                NJearly+= 1
    

    Ete = (NJearly+NJtard) / len(schedule)

    print("Number of estimated early Jobs: ", NJearly)
    print("Number of estimated Tardy Jobs: ", NJtard)

    return Ete

def calculate_actual_earliness_tardiness_rate(schedule, list_machines):
    NJa_tard = 0
    NJa_early = 0

    for job in schedule:
        if len(job.completed_tasks)< len(job.tasks):
            Tleft = 0
            last_completed_task_timestamp = job.get_last_task_completion_timestamp()
            if last_completed_task_timestamp > job.delivery_window[1]:
                NJa_tard += 1
            
            else:
                left_tasks= [task for task in job.tasks if task not in job.completed_tasks]

                for left_task in left_tasks:
                    tij = np.sum([machine.get_possible_operations()[left_task.__class__] for machine in list_machines])/len(list_machines)
                    Tleft+= tij
                    if last_completed_task_timestamp+Tleft> job.delivery_window[1]:
                        NJa_tard +=1
                        break
                
                if last_completed_task_timestamp +Tleft < job.delivery_window[0]:
                    NJa_early += 1
    
    ETa = (NJa_early+NJa_tard)/len(schedule)
    print("Number of actual early Jobs: ", NJa_early)
    print("Number of actual Tardy Jobs: ", NJa_tard)

    return ETa

def actual_penalty_cost(schedule, list_machines): 

    p_num_list = [0]
    p_den_list = [1]

    for job in schedule:
        if len(job.get_completed_tasks()) < len(job.tasks):
            Tleft = 0
            
            left_tasks= [task for task in job.tasks if task not in job.completed_tasks]
            
            for left_task in left_tasks:
                tij = np.sum([machine.get_possible_operations()[left_task.__class__] for machine in list_machines])/len(list_machines)
                Tleft+= tij
            
            last_completed_task_timestamp = job.get_last_task_completion_timestamp()
            
            if(last_completed_task_timestamp> job.delivery_window[1]):
                penalty =  job.earliness_tardiness_weights[1] * (last_completed_task_timestamp+ Tleft - job.delivery_window[1])
                p_num_list.append(penalty)
                p_den_list.append(penalty + 10)
            
            if(last_completed_task_timestamp +Tleft < job.delivery_window[0]):
                penalty = job.earliness_tardiness_weights[0] * (job.delivery_window[0] - last_completed_task_timestamp - Tleft)
                p_num_list.append(penalty)
                p_den_list.append(penalty + 10)
    
    p_total = sum(p_num_list) / sum(p_den_list)

    return p_total
