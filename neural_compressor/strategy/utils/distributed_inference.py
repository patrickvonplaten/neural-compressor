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
            result_record (List): the result collection. 
                item: 
                {"id":trial id, "tune_cfg": tune config, "eval_result": evaluation result}
        """
        result_record = []
        for trial in self.completed_trials:
            if trial["id"] not in self.reported_trials:
                result_record.append(trial)
                self.reported_trials.add(trial["id"])
        return result_record

    def add_finished_trial(self, result: Dict):
        """Trial can report its result by this interface.
        Arg:
        result (Dict):
            {"id":trial id, "tune_cfg": tune config, "eval_result": evaluation result} 
        Returns:
            None
        """
        self.completed_trials.append(result)
        print(f"[Monitor{id(self)}] finished trail", result["id"], "len(completed_trials):", len(self.completed_trials),
              "len(self.trials_lst):", len(self.trials_lst))
        pass

    def add_trial(self, trail):
        self.trials_lst.append(trail)

    def get_attribute(self):
        return self.trials_lst, self.completed_trials, self.reported_trials

    def add_finished_qmodel(self, q_model):
        """Test report qmodel
        """
        print("#" * 50)
        print(type(q_model))

@ray.remote
class Trial:
    def __init__(self, tune_cfg, adaptor, model, calib_dataloader, q_func, evaluate, result_monitor: ResultMonitor) -> None:
        """Class to run one independent task including calibration, quantization and evaluation for specific tuning config. 
        And report the quantized model and evaluation result to result monitor.
        Args:

        """
        print(f"[Trial] Initializing a new Trial {id(self)} with tune cfg: {id(tune_cfg)}.")
        self.tune_cfg = tune_cfg
        self.adaptor = adaptor
        self.model = model
        self.calib_dataloader = calib_dataloader
        self.q_func = q_func
        self.evaluate = evaluate
        self.result_monitor = result_monitor
        self.eval_result = None
        self.finished_trial = False
        self.id = id(self)
        
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
        print(f"[Trial] Start to compute process for one trial.")
        q_model = self.adaptor.quantize(self.tune_cfg, self.model, self.calib_dataloader, self.q_func)
        eval_result = self.evaluate(q_model)
        print(f"[Trial] Finished the compute with eval result {eval_result}.")
        self.eval_result = eval_result
        self.finished_trial = True
        # report to result monitor 
        # self.result_monitor.add_finished_trial.remote({"id": "test"})
        self.result_monitor.add_finished_trial.remote({"id": self.id, "tune_cfg": self.tune_cfg, "eval_result": eval_result}) #, "q_model": q_model
        # self.result_monitor.add_finished_qmodel.remote(q_model)
        time.sleep(20)
        return eval_result

    def report_state(self):
        
        print(f"[Trial] Report state for {self.id}.")
        return self.finished_trial
    
    def report_result(self):
        print(f"[Trial] Report eval result for {id(self)}.")
        return self, self.eval_result
    
    def get_id(self):
        return self.id

@ray.remote
class Scheduler:
    def __init__(self, tune_cfg_lst, adaptor, model, calib_dataloader, q_func, evaluate,result_monitor: ResultMonitor) -> None:
        """Class for manage the hardware resource and dispatch trials. 
        Specific the CPUs and GPUs for each trial.
        """
        print(f"[Scheduler] Initializing a new scheduler.")
        print("[Scheduler] Cluster Resources:", ray.cluster_resources())
        self.trials_lst = []
        self.tune_cfg_lst = tune_cfg_lst
        self.adaptor = adaptor
        self.model = model
        self.calib_dataloader = calib_dataloader
        self.q_func = q_func
        self.evaluate = evaluate
        self.result_monitor = result_monitor
        self.num_cpus =  ray.cluster_resources()["CPU"]
        self.num_gpus = None if "GPU" not in ray.cluster_resources().keys() else ray.cluster_resources()["GPU"]

    def trials_lst(self) -> List[Trial]:
        return self.trials_lst

    def dispatch_trails(self):
        """Dispatch the trial to remote actor.
        """
        for tune_cfg in self.tune_cfg_lst:
            # TODO check the hardware resource before schedule new trial
            print(f"[Scheduler] Avaliable Resources: {ray.available_resources()}.")
            print(f"[Scheduler] Create a new trial for the {id(tune_cfg)}.")
            trial = Trial.options(num_cpus=None, num_gpus=None).remote(tune_cfg, self.adaptor, self.model, self.calib_dataloader, self.q_func, self.evaluate, self.result_monitor)
            self.trials_lst.append(trial)
            self.result_monitor.add_trial.remote(ray.get(trial.get_id.remote()))
            trial.compute.remote()

class DistributedRunner:
    def __init__(self, tune_cfg_lst, adaptor, model, calib_dataloader, q_func, evaluate) -> None:
        """Initialize the components for distributed tuning.

        Args:
            tune_cfg_lst: _description_
            agrs: _description_
            
        Example:
            runner = DistributedRunner(tune_cfg_lst=[1, 2, 3], args = None)
            for result in runner.next_result:
                yield result
        """
        ray.init(address="10.112.228.232:6378", num_cpus=None, num_gpus=None)
        assert ray.is_initialized()
        self.result_monitor = ResultMonitor.remote()
        self.scheduler = Scheduler.remote(tune_cfg_lst, adaptor, model, calib_dataloader, q_func, evaluate, self.result_monitor)
        self.result = self.scheduler.dispatch_trails.remote()
        # for trail in self.result_monitor.trials_lst:
        #     print(f"[Runner[Monitor{id(self.result_monitor)}]]", id(trail))
        # for trail in self.scheduler.trials_lst:
        #     print("[Runner]", id(trail), "Trail.id", ray.get(trail.get_id.remote()))

    def next_result(self):
        """
        Collect all completed trials and report it to strategy.
        
        """
        # print("*" * 50, ray.get(self.result_monitor.all_trials_reported.remote()))
        _ = ray.get(self.result)
        # trials_lst, compeleted_trials, report_trials = ray.get(self.result_monitor.get_attribute.remote())
        # print("*" * 50, len(trials_lst), len(compeleted_trials), len(report_trials))
        while not ray.get(self.result_monitor.all_trials_reported.remote()):
            for result in ray.get(self.result_monitor.report_results.remote()):
                yield result  # {tune_cfg, quantized_model, evaluation result}
            # TODO may need to find a better way.
            time.sleep(5) # pause 5s to do next query. 
