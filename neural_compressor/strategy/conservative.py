#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os

from collections import deque
from copy import deepcopy
import numpy as np

from .strategy import strategy_registry, TuneStrategy
from ..utils import logger
from ..utils.utility import Statistics

@strategy_registry
class ConservativeTuneStrategy(TuneStrategy):
    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):
        super(
            ConservativeTuneStrategy,
            self).__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts,
            q_hooks)
        self.acc_meet_flag = False

    def next_tune_cfg(self):
        """
        Conservative tuning: accuracy first, performance second
        
        1. Query all quantifiable ops and save as a list: quantifiable_ops = [(op_name, op_type), ...]
        2. Classify the op by its op type
        3. Add op to quant_queue according to the op type priority
        4. Go through the quant_queue and replace it with the fp32 config in tune_cfg if
           accuracy meets the requirements else continue
           1) if accuracy meets the requirements, update the model with the best performance
           
        
        Stop condition:
        For bf16 and fp16, do the same thing as int8
        Note:
        1) other tunable items will using the first option as the default value.
        2) If auto: for op support both dynamic and static, use dynamic.

        Yields:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """
        
        tuning_space = self.tuning_space
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        calib_sampling_size = calib_sampling_size_lst[0]
        tune_cfg = self._initialize_tune_cfg()
        tune_cfg['calib_sampling_size'] = calib_sampling_size
        op_type_priority = self._get_op_type_priority()
        quant_queue = self._quant_queue(op_type_priority)
        # Try to add quantized ops.
        logger.info(f"*** Quantized op list: {[(pair[0].name, pair[1]) for pair in quant_queue]}")
        while quant_queue:
            op_item, target_dtype = quant_queue.popleft()
            op_info = op_item.name
            op_config = tuning_space.set_deafult_config(op_info, target_dtype)
            tmp_tune_cfg = deepcopy(tune_cfg)
            tmp_tune_cfg[op_info] = op_config
            yield tmp_tune_cfg
            if self.acc_meet_flag:
                logger.info(f"*** Convert op {op_info[0]} to {target_dtype} and accuracy still meet the requiments")
                tune_cfg[op_info] = op_config
            else:
                logger.info(f"*** Skip quantize op {op_info[0]}.")
        logger.info(f"*** Ending tuning process due to no quantifiable op left.")

    def traverse(self):
        if not (self.cfg.evaluation and self.cfg.evaluation.accuracy and \
            (self.cfg.evaluation.accuracy.metric or self.cfg.evaluation.accuracy.multi_metrics)) \
            and self.eval_func is None:
            logger.info("Neither evaluation function nor metric is defined." \
                        " Generate a quantized model with default quantization configuration.")
            self.cfg.tuning.exit_policy.performance_only = True
            logger.info("Force setting 'tuning.exit_policy.performance_only = True'.")
            logger.info("Generate a fake evaluation function.")
            self.eval_func = self._fake_eval_func

        # Get fp32 model baseline
        if self.baseline is None:
            logger.info("Get FP32 model baseline.")
            self._fp32_model = self.model
            self.baseline = self._evaluate(self.model)       
            self.objectives.baseline = self.baseline
            self.best_tune_result = self.baseline
            # Initialize the best qmodel as fp32 model
            self.best_qmodel = self._fp32_model
            # Record the FP32 baseline
            self._add_tuning_history()
        self.show_baseline_info()

        # Start tuning
        trials_count = 0
        for op_tuning_cfg in self.next_tune_cfg():
            tune_cfg = self._tune_cfg_converter(op_tuning_cfg)
            trials_count += 1
            tuning_history = self._find_tuning_history(tune_cfg)
            if tuning_history and trials_count < self.cfg.tuning.exit_policy.max_trials:
                self.last_tune_result = tuning_history['last_tune_result']
                self.best_tune_result = tuning_history['best_tune_result']
                logger.warn("Find evaluated tuning config, skip.")
                continue
            logger.debug("Dump current tuning configuration:")
            logger.debug(tune_cfg)
            self.tuning_times += 1
            self.q_model = self.adaptor.quantize(
                copy.deepcopy(tune_cfg), self.model, self.calib_dataloader, self.q_func)
            self.algo.calib_iter = tune_cfg['calib_iteration']
            self.algo.q_model = self.q_model
            # TODO align the api to let strategy has access to pre_optimized model
            assert self.adaptor.pre_optimized_model
            self.algo.origin_model = self.adaptor.pre_optimized_model
            if self.cfg.quantization.recipes.fast_bias_correction:
                self.algo.algorithms[0].quantization_cfg = tune_cfg
            self.last_qmodel = self.algo()
            assert self.last_qmodel
            self.last_tune_result = self._evaluate(self.last_qmodel)
            self.acc_meet_flag = self.objectives.accuracy_meets()
            if self.acc_meet_flag:
                # Update current tuning config and model with best performance
                get_better_performance = self.compare_performace(self.last_tune_result, self.best_tune_result)
                if get_better_performance:
                    logger.info(f"*** Update the model with better performance.")
                    self.best_qmodel = self.last_qmodel
                    self.best_tune_result = self.last_tune_result
                else:
                    logger.info(f"*** The model meets the accuracy requirements but not achive the better performance.")
            # Dump the current state to log
            self.dump_tuning_state(trials_count, self.last_tune_result, self.best_tune_result, self.baseline)
            # Judge stop or continue tuning
            need_stop = self.stop(trials_count)
            # Record the tuning history
            saved_tune_cfg = copy.deepcopy(tune_cfg)
            saved_last_tune_result = copy.deepcopy(self.last_tune_result)
            self._add_tuning_history(saved_tune_cfg,
                                    saved_last_tune_result,
                                    q_config=self.q_model.q_config)
            self.tune_result_record.append(copy.deepcopy(self.last_tune_result))
            self.tune_cfg = tune_cfg
            self._dump_tuning_process_statistics()
            if need_stop:
                if self.cfg.tuning.diagnosis and self.cfg.tuning.diagnosis.diagnosis_after_tuning:
                    logger.debug(f'*** Start to do diagnosis (inspect tensor).')
                    self._diagnosis()
                if self.use_multi_objective and len(self.tune_result_record) > 1 and \
                    self.best_tune_result is not None:
                    best_trail, best_result = self.objectives.best_result(self.tune_result_record,
                                                                          copy.deepcopy(self.baseline))
                    if best_result != self.best_tune_result:
                        from neural_compressor.utils.utility import recover
                        self.best_qmodel = recover(self.model.model, 
                            os.path.join(self.cfg.tuning.workspace.path, 'history.snapshot'),
                            best_trail)
                        self.best_tune_result = best_result
                    self._dump_tuning_process_statistics()
                break

    def stop(self, trials_count):
        need_stop = False
        if trials_count >= self.cfg.tuning.exit_policy.max_trials:
            need_stop = True
        return need_stop
            
    def compare_performace(self, last_tune_result, best_tune_result):
        _, last_perf = last_tune_result
        _, best_perf = best_tune_result
        return last_perf[0] < best_perf[0]
    
    def dump_tuning_state(self, trials_count, last_tune_result, best_tune_result, baseline):
        if last_tune_result:
            last_tune = last_tune_result[0] if \
                isinstance(last_tune_result[0], list) else [last_tune_result[0]]
            for name, data in zip(self.metric_name, last_tune):
                if len(self.tune_data[name]) == 1:
                    self.tune_data[name].append(data)
                else:
                    self.tune_data[name][1] = data

            if self.metric_weight and len(last_tune) > 1:
                weighted_acc = np.mean(np.array(last_tune) * self.metric_weight)                    
                if len(self.tune_data['Weighted accuracy']) == 1:
                    self.tune_data['Weighted accuracy'].append(weighted_acc)
                else:
                    self.tune_data['Weighted accuracy'][1] = weighted_acc
                last_tune = [weighted_acc]

            last_tune_msg = '[Accuracy (int8|fp32):' + \
                ''.join([' {:.4f}|{:.4f}'.format(last, base) for last, base in \
                zip(last_tune, self.tune_data['baseline'])]) + \
                ''.join([', {} (int8|fp32): {:.4f}|{:.4f}'.format( \
                x, y, z) for x, y, z in zip( \
                self.objectives.representation, last_tune_result[1], baseline[1]) \
                if x != 'Accuracy']) + ']'
        else: # pragma: no cover
            last_tune_msg = 'n/a'
            for name in self.tune_data.keys() - {'baseline'}:
                if len(self.tune_data[name]) == 1:
                    self.tune_data[name].append('n/a')
                else:
                    self.tune_data[name][1] = 'n/a'

        if best_tune_result:
            best_tune = best_tune_result[0] if isinstance(best_tune_result[0], list) \
                        else [best_tune_result[0]]
 
            for name, data in zip(self.metric_name, best_tune):
                if len(self.tune_data[name]) == 2:
                     self.tune_data[name].append(data)
                else:
                    self.tune_data[name][2] = data

            if self.metric_weight and len(best_tune) > 1:
                weighted_acc = np.mean(np.array(best_tune) * self.metric_weight)

                if len(self.tune_data['Weighted accuracy']) == 2:
                    self.tune_data['Weighted accuracy'].append(weighted_acc)
                else: # pragma: no cover
                    self.tune_data['Weighted accuracy'][2] = weighted_acc

                best_tune = [weighted_acc]

            best_tune_msg = '[Accuracy:' + ''.join([' {:.4f}'.format(best) \
                for best in best_tune]) + ''.join([', {}: {:.4f}'.format(x,y) \
                for x,y in zip(self.objectives.representation, \
                best_tune_result[1]) if x != 'Accuracy']) + ']'

        else:
            best_tune_msg = 'n/a'
            for name in self.tune_data.keys() - {'baseline'}:
                if len(self.tune_data[name]) == 2:
                    self.tune_data[name].append('n/a')
                else:
                    self.tune_data[name][2] = 'n/a'

        logger.info("Tune {} result is: {}, Best tune result is: {}".format(trials_count,
                                                                            last_tune_msg,
                                                                            best_tune_msg))
        output_data = [[info_type, 
            '{:.4f} '.format(self.tune_data[info_type][0]) if \
            not isinstance(self.tune_data[info_type][0], str) else self.tune_data[info_type][0], 
            '{:.4f} '.format(self.tune_data[info_type][1]) if \
            not isinstance(self.tune_data[info_type][1], str) else self.tune_data[info_type][1],
            '{:.4f} '.format(self.tune_data[info_type][2]) if \
            not isinstance(self.tune_data[info_type][2], str) else self.tune_data[info_type][2]] \
            for info_type in self.tune_data.keys() if info_type != 'baseline']

        output_data.extend([[obj, 
            '{:.4f} '.format(baseline[1][i]) if baseline else 'n/a',
            '{:.4f} '.format(last_tune_result[1][i]) if last_tune_result else 'n/a',
            '{:.4f} '.format(best_tune_result[1][i]) if best_tune_result else 'n/a'] \
            for i, obj in enumerate(self.objectives.representation)])

        Statistics(output_data,
                   header='Tune Result Statistics',
                   field_names=['Info Type', 'Baseline', 'Tune {} result'.format(trials_count), \
                                                                'Best tune result']).print_stat()

    def _get_op_type_priority(self):
        optypewise_cap = self.capability['optypewise']
        op_type_priority = list(optypewise_cap.keys())
        return op_type_priority
            
    def _sorted_item_by_op_type(self, items_lst, op_type_priority):
        """ Socring the tuning items according to its op type.
        Args:
            items_lst: The tuning item list.
            op_type_priority: The op type list with the order.

        Returns:
            The tuning items list that sorted according to its op type.
        """
        priority_val_lst = range(len(op_type_priority), 0, -1)
        # For items whose op type does not exist in the priority list, give it the lowest priority.
        min_priority_val = 0
        priority_map = dict(zip(op_type_priority, priority_val_lst))
        def _map_priority(item):
            item_op_type = item[0].name[0]
            priority_val =  priority_map[item_op_type] if item_op_type in op_type_priority else min_priority_val
            return (item, priority_val)
        item_priority_lst = list(map(_map_priority, items_lst))
        # Quantized the element with higher priority value first.
        sorted_item_priority = sorted(item_priority_lst, key=lambda x: x[1], reverse=True)  
        sorted_items_lst = [item[0] for item in sorted_item_priority]
        return sorted_items_lst
            
    def _initialize_tune_cfg(self):
        """Initialize the tuning config with fp32 AMAP.

        Returns:
            The intialized tuning config.
        """
        tuning_space = self.tuning_space
        quant_mode_wise_items = tuning_space.quant_mode_wise_items
        # Initialize the tuning config
        initial_tuning_cfg = {}
        all_ops = set()
        fp32_ops = []
        for quant_mode, items_lst in quant_mode_wise_items.items():
            items_name_lst = [item.name for item in items_lst]
            all_ops = all_ops.union(set(items_name_lst))
            if quant_mode == "fp32":
                fp32_ops += [item.name for item in items_lst]
        non_fp32_ops_dtype = {}
        fp32_ops_set = set(fp32_ops)
        for quant_mode, items_lst in quant_mode_wise_items.items():
            items_name_set = set([item.name for item in items_lst])
            tmp_non_fp32_ops = items_name_set.difference(fp32_ops_set)
            if tmp_non_fp32_ops:
                for op_info in tmp_non_fp32_ops:
                    non_fp32_ops_dtype[op_info] = quant_mode
        for op_info in fp32_ops:
            initial_tuning_cfg[op_info] = tuning_space.set_deafult_config(op_info, "fp32")
        for op_info, quant_mode in non_fp32_ops_dtype.items():
            initial_tuning_cfg[op_info] = tuning_space.set_deafult_config(op_info, quant_mode)
        return initial_tuning_cfg
            
    def _quant_queue(self, op_type_priority):
        """Create the op queue to be quantized.
        --------------------------------------------------------------------------
        | Level 1 |         bf16       |         fp16       |  static/dynamic    |
        | Level 2 | conv2d, linear, ...| conv2d, linear, ...| conv2d, linear, ...|
        
        Returns:
            The op queue to be quantized.
        """
        quant_mode_wise_items = self.tuning_space.quant_mode_wise_items
        # Add all quantized pair into queue
        quant_ops_queue = deque([])
        for quant_mode in  ['bf16', 'fp16']:
            if quant_mode in quant_mode_wise_items:
                op_item_pairs = [(op_item, quant_mode) for op_item in quant_mode_wise_items[quant_mode]]
                op_item_pairs = self._sorted_item_by_op_type(op_item_pairs, op_type_priority)
                quant_ops_queue.extend(op_item_pairs)
        op_item_pairs = []
        quant_ops_name_set = set()
        for quant_mode, items_lst in quant_mode_wise_items.items():
            if "static" in quant_mode or 'dynamic' in quant_mode:
                op_item_pairs += [(item, quant_mode) for item in items_lst if item.name not in quant_ops_name_set]
                quant_ops_name_set = quant_ops_name_set.union([item.name for item in items_lst])
                op_item_pairs = self._sorted_item_by_op_type(op_item_pairs, op_type_priority)
                quant_ops_queue.extend(op_item_pairs)
        return quant_ops_queue
        
        
            
            
            
            
            
                
            
        
         
         
        

            
            