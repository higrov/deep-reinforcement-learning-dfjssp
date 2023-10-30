import numpy as np

def action_dispatching_rule1(schedule, list_machines):
    average_machine_completion_time= np.sum([machine.timestamp_last_operation_executed for machine in list_machines]) / len(list_machines)

    urgency_list= [(job,job.due_date[0] - average_machine_completion_time) for job in schedule]

    select_func = lambda x: x[1]

    selected_job = min(urgency_list, key=select_func)

    next_uncompleted_task = [task for task in selected_job[0].tasks if task not in [t[0] for t in selected_job[0].completed_tasks]][0]

    last_completed_task_selected_job = selected_job[0].completed_tasks[-1]

    machine_set = [machine for machine in list_machines if next_uncompleted_task in machine.possible_tasks.keys() ]

    machine_appro = []

    for machine in machine_set: 
        temp = max(machine.timestamp_last_operation_executed, last_completed_task_selected_job[2],selected_job[0].arrival_time)
        temp2 = temp + machine.possible_tasks[next_uncompleted_task]
        machine_appro.append((machine, temp2))
    
    selected_machine= min(machine_appro, key= select_func)

    return(selected_machine,selected_job[0])

def action_dispatching_rule2(uncompleted_jobs, list_machines):
    for job in uncompleted_jobs:
        execution_times : list = []
        next_uncompleted_tasks = [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]
        job_execution_time= 0
        for task in next_uncompleted_tasks:
            machine_set = [machine for machine in list_machines if task in machine.possible_tasks.keys()]
            time = np.sum([machine.possible_tasks[task] for machine in machine_set])/len(machine_set)
            job_execution_time += time

        execution_times.append((job, job_execution_time))
        
    select_func = lambda x: x[1]
    selected_job = max(execution_times,key= select_func)


    next_uncompleted_task = [task for task in selected_job[0].tasks if task not in [t[0] for t in selected_job[0].completed_tasks]][0]

    last_completed_task_selected_job = selected_job[0].completed_tasks[-1]

    machine_set = [machine for machine in list_machines if next_uncompleted_task in machine.possible_tasks.keys() ]

    machine_appro = []

    for machine in machine_set: 
        temp = max(machine.timestamp_last_operation_executed, last_completed_task_selected_job[2],selected_job[0].arrival_time) + machine.possible_tasks[next_uncompleted_task]
        machine_appro.append((machine, temp))
    
    selected_machine= min(machine_appro, key= select_func)

    return (selected_job[0], selected_machine[0])

def action_dispatching_rule3(schedule, list_machines):
    weight_calc =[]
    uncompleted_jobs = [job for job in schedule if not job.isCompleted]
    for job in uncompleted_jobs:
        calc = (0.2* job.earliness_tardiness_weights[0]) + (0.8*job.earliness_tardiness_weights[1])
        weight_calc.append((job,calc))
    
    select_func = lambda x: x[1]

    selected_job = max(weight_calc, key= select_func)

    next_uncompleted_task = [task for task in selected_job[0].tasks if task not in [t[0] for t in selected_job[0].completed_tasks]][0]

    suitable_machines = [machine for machine in list_machines if next_uncompleted_task in machine.possible_tasks.keys()]

    machine_load = []
    for machine in suitable_machines: 
        tasks_performed = []
        for job in uncompleted_jobs: 
            tasks_performed.extend([completed_task for completed_task in job.completed_tasks if machine== completed_task[1]])

        machine_load.append((machine,sum(integer for _,_ , integer in tasks_performed)))

    selected_machine = min(machine_load, key= select_func)

    return(selected_job[0], selected_machine[0])

def action_dispatching_rule4(uncompleted_jobs, list_machines): 
    mean_job_execution_times =[]
    for job in uncompleted_jobs:
        next_uncompleted_tasks = [task for task in job.tasks if task not in [t[0] for t in job.completed_tasks]]
        job_execution_time = 0
        for task in next_uncompleted_tasks:
            suitable_machines = [machine for machine in list_machines if task in machine.possible_tasks.keys()]

            average_execution_time = sum([machine.possible_tasks[task] for machine in suitable_machines])/len(suitable_machines)
            job_execution_time += average_execution_time
        
        mean_job_execution_times.append((job,job_execution_time))
    
    selected_job = min(mean_job_execution_times,key= lambda x: x[1])

    next_uncompleted_task = [task for task in selected_job[0].tasks if task not in [t[0] for t in selected_job[0].completed_tasks]][0]

    last_completed_task_selected_job = selected_job[0].completed_tasks[-1]

    machine_set = [machine for machine in list_machines if next_uncompleted_task in machine.possible_tasks.keys() ]

    machine_appro = []

    for machine in machine_set: 
        temp = max(machine.timestamp_last_operation_executed, last_completed_task_selected_job[2],selected_job[0].arrival_time)
        temp2 = temp + machine.possible_tasks[next_uncompleted_task]
        machine_appro.append((machine, temp2))
    
    selected_machine= min(machine_appro, key= lambda x: x[1])

    return(selected_job[0], selected_machine[0])


scheduling_rules = {0: action_dispatching_rule1,
                    1: action_dispatching_rule2,
                    2: action_dispatching_rule3,
                    3: action_dispatching_rule4}