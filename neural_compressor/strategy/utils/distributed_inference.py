import ray
import math
import time
import random
from copy import deepcopy
from typing import List, Dict

import ray

"""
Tasks:
1. optype-wise stage
2. calculate op sensitivity 

TODO:
------ Runner
1. How to start ray in program? ( start by command and connect the ray cluster.)
2. ray config
------ Strategy
1. Split traverse process into multi independence parts.
2. Update the result and q_model according to the results from runner.
"""
@ray.remote
class ResultMonitor:
    def __init__(self) -> None:
        """Class for monitor all trials state and report the evaluation result.

        The completed trial: the trial has been completed but not reported to strategy.
        The reported trial: the trial has been completed and reported to strategy.
        """
        self.trials_lst = []
        self.completed_trials = []
        self.reported_trials = set()

    def all_trials_reported(self):
        """Check if all trials are completed and reported.

        Returns:
            result(bool): the check result. True for all completed and reported trials, False for none.
        """
        return len(self.trials_lst) == len(self.reported_trials)

    def report_results(self):
        """Collect all completed but not reported trails at current state.

        Returns:
            result_record (Dict): the result collection. key: trial id, val : evaluation result.
        """
        result_record = []
        for trial in self.completed_trials():
            if trial not in self.reported_trials:
                result_record.append({trial.id: trial.result()})
                self.reported_trials.add(trial.id)
        return result_record

    def add_finished_trial(self):
        """Trial can report its result by this interface.

        Returns:
            None
        """
        pass

@ray.remote
class Trial:
    def __init__(self, tune_cfg, adaptor, model, calib_dataloader, q_func, evaluate, result_monitor: ResultMonitor) -> None:
        """Class to run one independent task including calibration, quantization and evaluation for specific tuning config. 
        And report the quantized model and evaluation result to result monitor.
        Args:

        """
        print(f"[Trial] Initializing a new Trial {id(self)} with tune cfg: {id(tune_cfg)}.")
        self.result_monitor = result_monitor
        self.tune_cfg = tune_cfg
        # self.adaptor = adaptor
        # self.eval = eval
        self.eval_result = None
        self.finished_trial = False
        
    def compute(self):
        """Execute one trial including calibration, quantization and evaluation. 

        Args:
            tune_cfg: _description_

        Returns:
            result(Dict):
                q_model: the quantized model.
                eval_result: the evaluation result.
        """
        result = {}
        print(f"[Trial] Start to compute precess for one trial.")
        q_modle = self.adaptor.quantize(self.tune_cfg, self.model, self.calib_dataloader, self.q_func)
        eval_result = self.evaluate(q_modle)
        print(f"[Trial] Finished the compute with eval result {eval_result}.")
        self.eval_result = eval_result
        self.finished_trial = True
        # report to result monitor 
        self.result_monitor.add_finished_trial.remote(result)
        return eval_result

    def report_state(self):
        print(f"[Trial] Report state for {id(self)}.")
        return self.finished_trial
    
    def report_result(self):
        print(f"[Trial] Report eval result for {id(self)}.")
        return self, self.eval_result
    

@ray.remote
class Scheduler:
    def __init__(self, tune_cfg_lst, result_monitor: ResultMonitor) -> None:
        """Class for manage the hardware resource and dispatch trials. 
        Specific the CPUs and GPUs for each trial.
        """
        print(f"[Scheduler] Initializing a new scheduler.")
        self.result_monitor = result_monitor
        self.trials_lst = []

    def trials_lst(self) -> List[Trial]:
        return self.trials_lst

    def dispatch_trails(self):
        """Dispatch the trial to remote actor.
        """
        for tune_cfg in self.tune_cfg_lst:
            # TODO check the hardware resource before schedule new trial
            print(f"[Scheduler] Create a new trial for the {id(tune_cfg)}.")
            trial = Trial.remote(tune_cfg, self.result_monitor)
            self.trials_lst.append(trial)
            trial.compute.remote(tune_cfg, adaptor, model, calib_dataloader, q_func, evaluate)

class DistributedRunner:
    def __init__(self, tune_cfg_lst, agrs) -> None:
        """Initialize the components for distributed tuning.

        Args:
            tune_cfg_lst: _description_
            agrs: _description_
            
        Example:
            runner = DistributedRunner(tune_cfg_lst=[1, 2, 3], args = None)
            for result in runner.next_result:
                yield result
        """
        ray.init(address="local")
        self.result_monitor = ResultMonitor()
        self.scheduler = Scheduler(result_monitor=self.result_monitor, *agrs)
        self.scheduler.dispatch_trails()

    def next_result(self):
        """
        Collect all completed trials and report it to strategy.
        
        """
        while not self.result_monitor.all_trials_reported():
            for result in self.result_monitor.report_results:
                yield result  # {tune_cfg, quantized_model, evaluation result}
            # TODO may need to find a better way.
            time.sleep(5) # pause 5s to do next query. 
