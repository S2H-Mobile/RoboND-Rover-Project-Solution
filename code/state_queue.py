class StateQueue(object):
    def __init__(self, initial_task):
        self.queue = [initial_task]
        
    def get_current(self):
        return self.queue[0] if len(self.queue) else None
    
    def run_current(self, rover):
        ct = self.get_current()
        return ct.run(rover) if ct is not None else rover
        
    def get_next(self, rover):
        next_task = self.get_current().next(rover)
        if next_task is None:
            # Current task has finished.
            self.queue.pop(0)
        elif next_task is not self.get_current():
            # Current task has spawned a subtask.
            self.queue.insert(0, next_task)
            
        # Return self if tasks left or none when all tasks have finished.  
        return self if self.get_current() is not None else None