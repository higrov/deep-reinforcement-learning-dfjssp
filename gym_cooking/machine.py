import simpy
from recipe_planner.utils import *
from utils.core import Order

# Define the machines in the job shop
class Machine:
    possible_operations = {Get: 5, Merge: 2, Chop: 1, Deliver: 5}
    def __init__(self, env, name, capacity):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.queue = simpy.Resource(env, capacity=capacity)
        self.last_operation_executed = None
        self.last_operation_executed_at = -1

    def process_job(self, job: Order):
        operation = job.get_next_operation()
        start_time = self.env.now
        print(f"{start_time:.2f}: Job {job.full_name}, operation {str(operation)} started on {self.name}")
        processing_time = self.get_processing_time(operation)
        completed_time = start_time + processing_time
        yield self.env.timeout(processing_time)  # Simulate processing time
        print(f"{completed_time:.2f}: Job {job}, operation {str(operation)} completed on {self.name}")
        self.last_operation_executed = operation
        self.last_operation_executed_at = completed_time
        job.add_completed_tasks(operation,self.name,completed_time)

    def get_possible_operations(self): 
        return self.possible_operations
    
    def set_last_operation_executed(self,val): self.last_operation_executed = val

    def set_last_operation_performed_at(self, val): self.last_operation_executed_at = val

    def get_processing_time(self, action): 
        processing_time = self.possible_operations[action.__class__]
        return processing_time