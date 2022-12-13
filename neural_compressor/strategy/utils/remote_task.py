# remote_task.py

import ray
from ..strategy import TuneStrategy

class ResultMonitor:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "This is result monitor."
@ray.remote()
class Trial(TuneStrategy):
    def __init__(self, monitor: ResultMonitor) -> None:
        self.monitor = monitor

    def compute(self, tune_cfg):
        # 1. calibration
        # 2. quantization
        # 3. evaluation
        q_model = self.adaptor.quantize(tune_cfg)
        eval_result = self._evaluate(q_model)
        return eval_result

    def report_state(self):
        self.monitor.add_result(self.id, self.result)
        pass
    
    def __repr__(self) -> str:
        return "This is Trial."


class Scheduler:
    def __init__(self) -> None:
        pass
    
    def dispatch_trails(self, tune_cfg_lst):
        self.trail_lst = []
        for tune_cfg in tune_cfg_lst:
            trial = Trial(tune_cfg)
            self.trail_lst.append(trial)
