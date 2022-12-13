import ray

@ray.remote()
class Trial:
    def __init__(self) -> None:
        pass

    def compute(self):
        # 1. calibration
        # 2. quantization
        # 3. evaluation
        pass

    def report_state(self):
        pass
    
    def __repr__(self) -> str:
        return "This is Trial."


class ResultMonitor:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "This is result monitor."
