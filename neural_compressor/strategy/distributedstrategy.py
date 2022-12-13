# distributed_strategy.py

from .strategy import TuneStrategy
from .utils.remote_task import Scheduler

class NewTuneStrategy(TuneStrategy):
    def next_tune_cfg(self):
        return super().next_tune_cfg()
    
    def traverse(self):
        self.trails_sheduler = Scheduler()
        tune_cfg_lst = []
        for tune_cfg in self.next_tune_cfg():
            tune_cfg_lst.append(tune_cfg)
        self.trails_scheduler.dispatch_trial(tune_cfg_lst)
        
        tune_cfg_lst = []
        for tune_cfg in self.next_tune_cfg():
            tune_cfg_lst.append(tune_cfg)
        self.trails_scheduler.dispatch_trial(tune_cfg_lst)

        tune_cfg_lst = []
        for tune_cfg in self.next_tune_cfg():
            tune_cfg_lst.append(tune_cfg)
        self.trails_scheduler.dispatch_trial(tune_cfg_lst)