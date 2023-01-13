from typing import List

class Adaptor:
    def __init__(self) -> None:
        pass
    
    def evaluate(self, model):
        pass
    
    def quantize(self, tune_cfg):
        pass
    
    
class TuneStrategy:
    def __init__(self) -> None:
        pass
    
    def traverse(self):
        if self.cfgs.num_worker > 1:
            return self.distributed_traverse()
        else:
            for tune_cfg in self.next_tune_cfg():
                pass
            
    def distributed_traverse(self):
        from mpi4py import MPI
        comm = MPI.Comm()
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank == 0:
            self.master_worker_setup(comm)
        else:
            self.slave_worker_setup(comm)

    def master_worker_setup(self, comm):
        """
        assign task 
        query the finished task
        update best q_model
        send new task to last free node
        send end tag
        :return: 
        """
        ...

    def slave_worker_setup(self, comm):
        """
        receive task
        send eval result
        :return:
        """
        ...
        
@strategy_registry
class BasicTuneStrategy(TuneStrategy):
    """The basic tuning strategy."""

    def next_tune_cfg(self):
        """Generate and yield the next tuning config with below order.
        
            1. OP Type Wise Tuning
            2. Fallback OP One by One
            3. Fallback Multiple OPs Accumulated
        Yields:
            tune_config (dict): A dict containing the tuning configuration for quantization.
        """
        if self.cfgs.num_worker > 1:
            return self.distributed_next_tune_cfg()
        optype_wise_tuning_sampler = OpTypeWiseTuningSampler(tuning_space,...)
        for tune_cfg in optype_wise_tuning_sampler:
            q_model = self.adaptor.quantize(tune_cfg)
            eval_res = self.adaptor.evaluate(q_model)
            self.update_best_q_model(eval_res)
        ...
    
    
    def distributed_next_tune_cfg(self):
        distributed_optype_wise_tuning_sampler = DistributedOpTypeWiseTuningSampler(tuning_space,...)
        for tune_cfg in distributed_optype_wise_tuning_sampler:
            q_model = self.adaptor.quantize(tune_cfg)
            eval_res = self.adaptor.evaluate(q_model)
            self.update_best_q_model(eval_res)
            
        
class OpTypeWiseTuningSampler:
    def __init__(self) -> None:
        pass

    def __iter__(self):
        for tune_cfg in tune_cfg_lst:
            yield tune_cfg


class DistributedOpTypeWiseTuningSampler(OpTypeWiseTuningSampler):
    def __init__(self, ..., rank):
         super().__init__(...)
         self.rank = rank
        
    def __iter__(self, rank, comm_msg_tag):
        """Generate the next tune_cfg
        Args:
            rank: node rank
            comm_msg_tag: The tag of message from master node.
        Yields:
            tune_cfg
        """
        
        tune_task_id_lst = self.mapping_rank_to_id(self.rank, comm_msg_tag)
        for tune_task_id in tune_task_id_lst:
            tune_cfg  = tune_cfg_list[tune_task_id]
            yield tune_cfg