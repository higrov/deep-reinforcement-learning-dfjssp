import numpy as np

from recipe_planner.utils import Get, Chop, Merge, Deliver

class StateCalculator:
    reward_points_operation = {Get: 3, Chop: 10, Merge: 20, Deliver: 50}
    def __init__(self):
        self.penalty = 0
       
    def calculate_state_features(self, uncompleted_jobs, list_machines):
        average_utilization_rate = self.calculate_average_utilization_rate(schedule= uncompleted_jobs, list_machines= list_machines)
        estimated_earliness_tardiness_rate= self.calculate_estimated_earliness_tardiness_rate(schedule= uncompleted_jobs, list_machines= list_machines)

        actual_earliness_tardiness_rate = self.calculate_actual_earliness_tardiness_rate(schedule= uncompleted_jobs, list_machines= list_machines)

        penalty= self.actual_penalty_cost(schedule= uncompleted_jobs, list_machines= list_machines)
        
        self.penalty = penalty
        state = [average_utilization_rate, estimated_earliness_tardiness_rate,actual_earliness_tardiness_rate, penalty]
        return state

    def calculate_average_utilization_rate(self, schedule, list_machines):
        machine_util = 0
        num_machines = len(list_machines)

        for machine in list_machines:
            job_util = 0
            for job in schedule:
                task_util = 0 
                completed_operations_machine =[oper for oper in job.completed_task_machine if oper[1] == machine.name]

                for (operation, machine_name) in completed_operations_machine:
                    processing_time = machine.get_processing_time(operation)
                    task_util += processing_time

                job_util+= task_util

            machine_util += ((job_util/machine.last_operation_executed_at) if machine.last_operation_executed_at > 0 else 0)

        average_utilization_rate = machine_util / num_machines  # Calculate average over tasks and jobs

        return average_utilization_rate

    def calculate_estimated_earliness_tardiness_rate(self, schedule, list_machines):
        Tcur = sum([machine.last_operation_executed_at if machine.last_operation_executed_at > 0 else 0 for machine in list_machines])/len(list_machines)

        NJtard = 0
        NJearly = 0

        for job in schedule: 
            if len(job.completed_tasks)< len(job.tasks):
                Tleft = 0

                left_tasks= [task for task in job.tasks if task not in job.completed_tasks]

                for left_task in left_tasks: 
                    tij = np.sum([machine.get_processing_time(left_task) for machine in list_machines])/len(list_machines)
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

    def calculate_actual_earliness_tardiness_rate(self, schedule, list_machines):
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
                        tij = np.sum([machine.get_processing_time(left_task) for machine in list_machines])/len(list_machines)
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

    def actual_penalty_cost(self, schedule, list_machines): 

        p_num_list = [0]
        p_den_list = [1]

        for job in schedule:
            if len(job.get_completed_tasks()) < len(job.tasks):
                Tleft = 0

                left_tasks= [task for task in job.tasks if task not in job.completed_tasks]

                for left_task in left_tasks:
                    tij = np.sum([machine.get_processing_time(left_task) for machine in list_machines])/len(list_machines)
                    Tleft+= tij

                last_completed_task_timestamp = job.get_last_task_completion_timestamp()

                if(last_completed_task_timestamp > job.delivery_window[1]):
                    penalty =  job.earliness_tardiness_weights[1] * (last_completed_task_timestamp+ Tleft - job.delivery_window[1])
                    p_num_list.append(penalty)
                    p_den_list.append(penalty + 100000000)

                if(last_completed_task_timestamp +Tleft < job.delivery_window[0]):
                    penalty = job.earliness_tardiness_weights[0] * (job.delivery_window[0] - last_completed_task_timestamp - Tleft)
                    p_num_list.append(penalty)
                    p_den_list.append(penalty + 100000000)

        p_total = sum(p_num_list) / sum(p_den_list)

        return p_total

    def calculate_reward(self,schedule):
        reward_total = 0
        for job in schedule:
            # max_reward =  sum([self.reward_points_operation[operation.__class__] for operation in job.tasks])
            # reward = sum([self.reward_points_operation[operation.__class__] for operation in job.completed_tasks])

            reward_total += job.reward

        return reward_total