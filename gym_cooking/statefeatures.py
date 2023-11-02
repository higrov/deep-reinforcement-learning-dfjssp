import numpy as np

# State Features

def calculate_average_utilization_rate(schedule, list_machines):
    machine_util = 0
    num_machines = len(list_machines)

    for machine in list_machines:
        job_util = 0
        for job in schedule:
            task_util = 0 
            completed_operations_machine =[oper for oper in job.get_completed_tasks() if oper[1] == machine]

            for (operation, machine, _) in completed_operations_machine:
                processing_time = machine.get_possible_tasks()[operation]
                task_util += processing_time
        
            job_util+= task_util
        
        machine_util += (job_util/machine.timestamp_last_operation_executed)

    average_utilization_rate = machine_util / num_machines  # Calculate average over tasks and jobs

    return average_utilization_rate

def calculate_estimated_earliness_tardiness_rate(schedule, list_machines):
    Tcur = sum([machine.timestamp_last_operation_executed for machine in list_machines])/len(list_machines)

    NJtard = 0
    NJearly = 0
    
    for job in schedule: 
        if len(job.completed_tasks)< len(job.tasks):
            Tleft = 0
            
            left_tasks= [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]

            for left_task in left_tasks: 
                tij = np.sum([machine.get_possible_tasks()[left_task] for machine in list_machines])/len(list_machines)
                Tleft += tij
                if Tcur + Tleft > job.due_date[1]:
                    NJtard += 1
                    break
            
            if Tleft + Tcur < job.due_date[0]:
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
            last_completed_task = job.get_completed_tasks()[-1]
            if last_completed_task[2] > job.due_date:
                NJa_tard += 1
            
            else:
                left_tasks= [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]

                for left_task in left_tasks:
                    tij = np.sum([machine.get_possible_tasks()[left_task] for machine in list_machines])/len(list_machines)
                    Tleft+= tij
                    if last_completed_task[2]+Tleft> job.due_date[1]:
                        NJa_tard +=1
                        break
                
                if last_completed_task[2] +Tleft < job.due_date[0]:
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
            
            left_tasks= [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]
            
            for left_task in left_tasks:
                tij = np.sum([machine.get_possible_tasks()[left_task] for machine in list_machines])/len(list_machines)
                Tleft+= tij
            
            last_completed_task = job.get_completed_tasks()[-1]
            
            if(last_completed_task[2]> job.due_date[1]):
                penalty =  job.earliness_tardiness_weights[1] * (last_completed_task[2]+ Tleft - job.due_date)
                p_num_list.append(penalty)
                p_den_list.append(penalty + 10)
            
            if(last_completed_task[2] +Tleft < job.due_date[0]):
                penalty = job.earliness_tardiness_weights[0] * (job.due_date - last_completed_task[2] - Tleft)
                p_num_list.append(penalty)
                p_den_list.append(penalty + 10)
    
    p_total = sum(p_num_list) / sum(p_den_list)

    return p_total