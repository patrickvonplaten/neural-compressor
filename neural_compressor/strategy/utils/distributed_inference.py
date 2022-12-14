import ray
import math
import time
import random
from copy import deepcopy

# @ray.remote
# class ResultMonitor:
#     def __init__(self) -> None:
#         print(f"[ResultMonitor] Initializing a new ResultMonitor.")
#         pass
    
#     def get_result(self, trial_id, result):
#         print(f"")
#         return trial_id, result


@ray.remote
class Trial:
    def __init__(self, tune_cfg) -> None:
        print(f"[Trial] Initializing a new Trial {id(self)} with tune cfg: {id(tune_cfg)}.")
        self.tune_cfg = tune_cfg
        # self.adaptor = adaptor
        # self.eval = eval
        self.eval_result = None
        self.finished_trial = False
        pass

    def compute(self, tune_cfg, adaptor, model, calib_dataloader, q_func, evaluate):
        print(f"[Trial] Start to compute precess for one trial.")
        q_modle = adaptor.quantize(tune_cfg, model, calib_dataloader, q_func)
        eval_result = evaluate(q_modle)
        print(f"[Trial] Finished the compute with eval result {eval_result}.")
        self.eval_result = eval_result
        self.finished_trial = True
        return eval_result
    
    def report_state(self):
        print(f"[Trial] Report state for {id(self)}.")
        return self.finished_trial
    
    def report_eval_result(self):
        print(f"[Trial] Report eval result for {id(self)}.")
        return self, self.eval_result

class Scheduler:
    def __init__(self) -> None:
        ray.init(address="local")
        print(f"[Scheduler] Initializing a new scheduler.")
        self.trials_lst = []
        pass

    def dispatch_trails(self, tune_cfg_lst, adaptor, model, calib_dataloader, q_func, evaluate):
        for tune_cfg in tune_cfg_lst:
            print(f"[Scheduler] Create a new trial for the {id(tune_cfg)}.")
            trial = Trial.remote(tune_cfg)
            self.trials_lst.append(trial)
            trial.compute.remote(tune_cfg, adaptor, model, calib_dataloader, q_func, evaluate)
            
