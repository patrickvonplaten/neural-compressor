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
from ...utils.utility import LazyImport

torch = LazyImport("torch")

import copy
import numpy as np
from collections import OrderedDict
import torch.nn
from torch.quantization.quantize_fx import fuse_fx
import torch.nn.intrinsic.quantized as nniq
from torch.fx import symbolic_trace, graph_module
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Any, Union, Callable, Set

import numpy as np
import random
seed=9527
np.random.seed(seed)
random.seed(seed)
import torch
torch.manual_seed(seed) #cpu
torch.cuda.manual_seed_all(seed)  #并行gpu
torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速

# Define Collector based on hook, which is used to record the intermediate result
class Node_collector:
    def __init__(self, m):
        self.handle = m.register_forward_hook(self.hook_fn_act)

    def hook_fn_act(self, m, inp, outp):
        self.out_features = outp.clone()
        self.in_features = inp
        self.m = m

    def remove(self):
        self.handle.remove()


class ParameterHandler:
    def __init__(self, parameters, device="cpu"):
        self._device = device
        self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    def get_gradients(self):
        gradients = []
        for parameter in self.parameters:
            gradients.append(0. if parameter.grad is None else parameter.grad + 0.)
        return gradients

    def sample_rademacher_like_params(self):
        def sample(parameter):
            r = torch.randint_like(parameter, high=2, device=self._device)
            return r.masked_fill_(r == 0, -1)

        return [sample(p) for p in self.parameters]

    def sample_normal_like_params(self):
        return [torch.randn(p.size(), device=self._device) for p in self.parameters]

##shameless copy from https://github.com/openvinotoolkit/nncf
class GradientsCalculator:

    def __init__(self, model: nn.Module,
                 data_loader, num_data_iter: int,
                 paramerter_handler: ParameterHandler, criterion):
        self._model = model
        self._data_loader = data_loader
        self._num_data_iter = num_data_iter
        self._parameter_handler = paramerter_handler
        self.num_iter = 0
        if criterion == None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def __iter__(self):
        self.data_loader_iter = iter(self._data_loader)
        self.num_iter = 0
        return self

    def __next__(self):
        if self.num_iter >= self._num_data_iter:
            raise StopIteration
        self.num_iter += 1
        inupt, target = next(self.data_loader_iter)

        self._model.zero_grad()

        outputs = self._model(inupt)
        loss = self.criterion(outputs, target)

        loss.backward(create_graph=True)
        grads = self._parameter_handler.get_gradients()
        self._model.zero_grad()
        return grads
class GradientsCalculator2:
    
    def __init__(self, model: nn.Module,
                 data_loader, num_data_iter: int,
                 criterion):
        self._model = model
        self._data_loader = data_loader
        self._num_data_iter = num_data_iter
        self.num_iter = 0
        self.parameters=[p for p in self._model.parameters() if p.requires_grad]
        if criterion == None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def __iter__(self):
        self.data_loader_iter = iter(self._data_loader)
        self.num_iter = 0
        return self
    def get_gradients(self):
        gradients = []
        for parameter in self.parameters:
            gradients.append(0. if parameter.grad is None else parameter.grad + 0.)
        return gradients
    def __next__(self):
        if self.num_iter >= self._num_data_iter:
            raise StopIteration
        self.num_iter += 1
        inupt, target = next(self.data_loader_iter)

        self._model.zero_grad()

        outputs = self._model(inupt)
        loss = self.criterion(outputs, target)

        loss.backward(create_graph=True)
        
        grads = self.get_gradients()
        self._model.zero_grad()
        return grads
##shameless copy from https://github.com/openvinotoolkit/nncf
class HessianTraceEstimator:
    """
    Performs estimation of Hessian Trace based on Hutchinson algorithm.
    """

    def __init__(self, model_tmp: nn.Module, data_loader, criterion=None,
                 device="cpu",
                 num_data_points=0):
        self._model = fuse_fx(model_tmp)
        self.criterion=criterion
        parameters = [p for p in self._model.parameters() if p.requires_grad]
        self._parameter_handler = ParameterHandler(parameters, device)
        self._batch_size = data_loader.batch_size
        logger.info("self._batch_size:"+str(self._batch_size))
        self._num_data_iter = num_data_points // self._batch_size if num_data_points >= self._batch_size else 1
        self._gradients_calculator = GradientsCalculator(self._model, data_loader,
                                                         self._num_data_iter,
                                                         self._parameter_handler,
                                                         criterion=self.criterion)
        self._diff_eps = 1e-6

    def get_average_traces(self, max_iter=500, tolerance=1e-5):
        """
        Estimates average hessian trace for each parameter
        :param max_iter: maximum number of iterations for Hutchinson algorithm
        :param tolerance: - minimum relative tolerance for stopping the algorithm.
        It's calculated  between mean average trace from previous iteration and current one.
        :return: Tensor with average hessian trace per parameter
        """
        avg_total_trace = 0.
        avg_traces_per_iter = []
        mean_avg_traces_per_param = None
        import tqdm

        for i in tqdm.tqdm(range(max_iter)):
            avg_traces_per_iter.append(self._calc_avg_traces_per_param())

            mean_avg_traces_per_param = self._get_mean(avg_traces_per_iter)
            mean_avg_total_trace = torch.sum(mean_avg_traces_per_param)

            diff_avg = abs(mean_avg_total_trace - avg_total_trace) / (avg_total_trace + self._diff_eps)
            if diff_avg < tolerance:
                # return mean_avg_traces_per_param
                print("End of hessian cacluation")
                break
            avg_total_trace = mean_avg_total_trace

        results = {}
        index = 0
        for n, p in self._model.named_parameters():
            results[n] = float(mean_avg_traces_per_param[index])
            index += 1
        return results

    def _calc_avg_traces_per_param(self):
        v = self._parameter_handler.sample_rademacher_like_params()
        vhp = self._parameter_handler.sample_normal_like_params()
        num_all_data = self._num_data_iter * self._batch_size
        for gradients in self._gradients_calculator:
            vhp_curr = torch.autograd.grad(gradients,
                                           self._parameter_handler.parameters,
                                           grad_outputs=v,
                                           only_inputs=True,
                                           retain_graph=False)
            vhp = [a + b * float(self._batch_size) + 0. for a, b in zip(vhp, vhp_curr)]
        vhp = [a / float(num_all_data) for a in vhp]
        avg_traces_per_param = torch.stack([torch.sum(a * b) / a.size().numel() for (a, b) in zip(vhp, v)])
        return avg_traces_per_param

    @staticmethod
    def _get_mean(data):
        return torch.mean(torch.stack(data), dim=0)


class HessianTrace:
    """
    please refer to
    Yao, Zhewei, et al. "Pyhessian: Neural networks through the lens of the hessian."
    2020 IEEE international conference on big data (Big data). IEEE, 2020.
    Dong, Zhen, et al. "Hawq-v2: Hessian aware trace-weighted quantization of neural networks."
    Advances in neural information processing systems 33 (2020): 18518-18529.
    https://github.com/openvinotoolkit/nncf/blob/develop/nncf/torch/quantization/hessian_trace.py
    """

    def __init__(self, model, dataloader, q_model, criterion=None):
        self.unfused_model = model.model
        self.q_model = q_model
        tmp_model = model.model
        if 'graph' in (str(dir(tmp_model))):  # check the attribute and it's length
            logger.info("This is aready fused model")
            self.model = model.model
        else:
            logger.info("fusing model")
            self.model = fuse_fx(model.model)  ##TODO need to check whether model has been already fused
        self.dataloader = dataloader
        self.max_iter = 500
        self.tolerance = 1e-5
        self.eps = 1e-6
        self.index = 0
        self.device = self.get_device(self.model)
        self.criterion = criterion
        self._batch_size=dataloader.batch_size
        self.params= [p for p in self.model.parameters() if p.requires_grad]
        num_data_points=0
        self._num_data_iter = num_data_points // self._batch_size if num_data_points >= self._batch_size else 1
        if self.criterion == None:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)  ##TODO need to set in config
        self.criterion = self.criterion.to(self.device)
        self.weight_to_op, self.op_list = self.get_fused_mapping()
        self.get_params()
        self._gradients_calculator=GradientsCalculator2(self.model,data_loader=self.dataloader,num_data_iter=
                                                       self._num_data_iter,criterion=self.criterion)

    def is_fused_module(self, module):
        """This is a helper function for `_propagate_qconfig_helper` to detecte
           if this module is fused.
        Args:
            module (object): input module
        Returns:
            (bool): is fused or not
        """
        op_type = str(type(module))
        if 'fused' in op_type:
            return True
        else:
            return False

    def mapping_module_to_op(self, name):
        # length = len("_model.")
        # if len(name) < length:
        #     return name
        # else:
        return name

    def mse_metric_gap(self, fp32_tensor, dequantize_tensor):
        """Calculate the euclidean distance between fp32 tensor and int8 dequantize tensor
        Args:
            fp32_tensor (tensor): The FP32 tensor.
            dequantize_tensor (tensor): The INT8 dequantize tensor.
        """
        fp32_max = np.max(fp32_tensor)
        fp32_min = np.min(fp32_tensor)
        dequantize_max = np.max(dequantize_tensor)
        dequantize_min = np.min(dequantize_tensor)
        fp32_tensor = (fp32_tensor - fp32_min) / (fp32_max - fp32_min)
        dequantize_tensor = (dequantize_tensor - dequantize_min) / \
                            (dequantize_max - dequantize_min)
        diff_tensor = fp32_tensor - dequantize_tensor
        euclidean_dist = np.sum(diff_tensor ** 2)
        return euclidean_dist / fp32_tensor.size

    def get_fused_mapping(self):
        model = self.model
        weights_info = dict(model.named_parameters())
        weight_to_op = {}
        for op_name, child in model.named_modules():
            if self.is_fused_module(child):
                for name, _ in child.named_children():
                    if op_name + "." + name + ".weight" in weights_info:  ##TODO check if this is right

                        weight_to_op[op_name + "." + name + ".weight"] = self.mapping_module_to_op(op_name)
                        break
            else:
                name = op_name + ".weight"
                if name in weights_info and name not in weight_to_op.keys():
                    weight_to_op[op_name + ".weight"] = op_name
        op_list = []
        for key in weight_to_op.keys():
            op_list.append(weight_to_op[key])
        return weight_to_op, op_list

    def get_device(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            return p.data.device

    def _get_act_grad_hook(self, name):
        def act_grad_hook(model, grad_input, grad_output):
            ##print(name, grad_input[0].shape, grad_output[0].shape)
            if type(model) == torch.nn.Linear:  ##TODO very tricky
                self.layer_acts_grads[name] = grad_input[1]
            else:
                self.layer_acts_grads[name] = grad_input[0]

        return act_grad_hook

    def _get_enable_act_grad_hook(self, name):
        def enable_act_grad_hook(model, inputs, outputs):
            input = inputs[0]
            if input.requires_grad is False:
                input.requires_grad = True
            self.layer_acts[name] = input

        return enable_act_grad_hook

    # def _get_disable_input_grad_hook(self, name):
    #     def disable_input_grad_hook(model, inputs, outputs):
    #         try:
    #             input = inputs[0]  ##TODO check whether this is right
    #         except:
    #             input = inputs
    #         if input.is_leaf == False:## you can only change requires_grad flags of leaf variables
    #             if input.requires_grad is True:
    #                 input.requires_grad = False
    #
    #
    #     return disable_input_grad_hook

    def _unregister_hook(self):
        for handel in self.hook_handles:
            handel.remove()

    def register_act_grad_hooks(self, model):
        for name, module in model.named_modules():
            if self.mapping_module_to_op(name) in self.op_list:
                hook_handle = module.register_forward_hook(self._get_enable_act_grad_hook(name))
                self.hook_handles.append(hook_handle)
                hook_handle = module.register_backward_hook(self._get_act_grad_hook(name))
                self.hook_handles.append(hook_handle)

    def reset_act_gradient_and_hooks(self):
        # tmp_input = torch.zeros(self._input_shape, device=self.device)
        # for name, module in self.model.named_modules():
        #     if name in self.op_list:
        #         hook_handle = module.register_forward_hook(self._get_disable_input_grad_hook(name))
        #         self.hook_handles.append(hook_handle)
        # self.model(tmp_input)
        self._unregister_hook()

    def get_params(self):
        weight_names = [n for n, p in self.model.named_parameters() if
                        p.requires_grad ]  ##remove bias and "bias" not in n
        params = [p for n, p in self.model.named_parameters() if p.requires_grad ]  ##remove bias and "bias" not in n
        self.weight_names = weight_names
        self.params = params

    def forward_backward(self, model, data, create_graph=False, return_w_grad=True):
        model.zero_grad()
        input = data[0].to(self.device)
        ##self._input_shape = input.shape  ## for resetting input activation
        target = data[1].to(self.device)
        input.requires_grad = True
        output = model(input)
        loss = self.criterion(output, target)
        torch.autograd.backward(loss, create_graph=create_graph)
        ##loss.backward(create_graph=create_graph)
        if return_w_grad:
            gradients = []
            for n, p in self.model.named_parameters():
                if p.grad != None and n in self.weight_names:
                    gradient = p.grad
                    gradients.append(gradient + 0.0)  ## add 0 to create a copy
            model.zero_grad()
            return gradients
        else:
            model.zero_grad()

    # def get_params(self, model):
    #     parameters = [p for p in model.parameters() if p.requires_grad]
    #     return parameters

    def sample_rademacher(self, params):
        samples = []
        for param in params:
            r = torch.randint_like(param, high=2, device=self.device)
            r.masked_fill_(r == 0, -1)
            samples.append(r)
        return samples
    
    def sample_rademacher_like_params(self):
        def sample(parameter):
            r = torch.randint_like(parameter, high=2, device=self.device)
            return r.masked_fill_(r == 0, -1)
        return [sample(p) for p in self.params]
    
    def sample_normal_like_params(self):
        return [torch.randn(p.size(), device=self.device) for p in self.params]

    def get_vtHv_weight(self, params, num_samples):
        v = self.sample_rademacher(params)
        H_v = [0] * len(v)
        cnt = 0
        for step, data in enumerate(self.dataloader):
            batch_size = data[0].shape[0]
            logger.info("batch_size:"+str(batch_size)+"|"+str(self.dataloader.batch_size))
            cnt += batch_size
            gradients = self.forward_backward(self.model, data, create_graph=True)
            H_v_one = torch.autograd.grad(gradients, params, v, only_inputs=True, retain_graph=False)
            H_v = [pre + cur * float(batch_size) for cur, pre in zip(H_v_one, H_v)]
            if cnt >= num_samples:
                break
        if cnt > 0:
            H_v = [item / cnt for item in H_v]
        v_t_H_v = torch.stack([torch.mean(h_v * v_t) for (h_v, v_t) in zip(H_v, v)])  ##maybe sum is better
        return v_t_H_v

    # def get_vtHv_act(self, params, num_samples):
    #     v = self.sample_rademacher(params)
    #     H_v = [0] * len(v)
    #     cnt = 0
    #     for step, data in enumerate(self.dataloader):
    #         if cnt >= num_samples:
    #             break
    #         for i in range(self.dataloader.batchsize):  ##force to batchsize to be 1
    #             input = data[0][i:i + 1]
    #             target = data[1][i:i + 1]

    #             self.get_gradients(self.model, (input, target), self.criterion, create_graph=True)
    #             layer_acts = [self.layer_acts[key] for key in self.layer_acts.keys()]
    #             layer_act_gradients = [self.layer_acts_grads[key] for key in self.layer_acts.keys()]
    #             hv_one = torch.autograd.grad(layer_act_gradients, layer_acts, v,
    #                                          only_inputs=True, retain_graph=False)
    #             cnt += 1
    #             if cnt >= num_samples:
    #                 break

    def get_weight_traces(self, num_samples):
        import tqdm
        layer_traces_per_iter = []
        prev_avg_model_trace = 0
        for iter in tqdm.tqdm(range(self.max_iter)):
            layer_traces = self.get_vtHv_weight(self.params, num_samples)
            layer_traces_per_iter.append(layer_traces)
            layer_traces_estimate = torch.mean(torch.stack(layer_traces_per_iter), dim=0)
            model_trace = torch.sum(layer_traces_estimate)
            diff_ratio = abs(model_trace - prev_avg_model_trace) / (prev_avg_model_trace + self.eps)
            if diff_ratio < self.tolerance and iter > 10:  ##TODO magic number
                break
            # if iter == 20:  ##TODO for debugging
            #     break
            prev_avg_model_trace = model_trace
        weight_name_to_traces = {}
        layer_traces = layer_traces_estimate
        for weight_name, trace in zip(self.weight_names, layer_traces):
            weight_name_to_traces[weight_name] = float(trace)  # tensor->float
        op_name_to_trace = {}
        for weight_name in self.weight_names:
            if "weight" in weight_name:
                op_name = self.weight_to_op[weight_name]
                op_name_to_trace[op_name] = weight_name_to_traces[weight_name]
        return op_name_to_trace

    def get_act_traces(self, num_samples):
        unfused_training = self.unfused_model.training
        self.unfused_model.eval()
        self.hook_handles = []
        self.layer_acts = {}
        self.layer_acts_grads = {}
        self.register_act_grad_hooks(self.unfused_model)
        cnt = 0
        act_traces_per_sample = []
        for step, data in enumerate(self.dataloader):
            if cnt >= num_samples:
                break
            bs = data[0].shape[0]
            act_traces_sum = 0
            act_traces_per_iter = []
            prev_avg_model_trace = 0
            act_traces_sums = None
            for i in range(bs):  ##force the bs to be one
                input = data[0][i:i + 1]
                target = data[1][i:i + 1]
                self.forward_backward(self.unfused_model, (input, target), create_graph=True, return_w_grad=False)
                acts = [self.layer_acts[key] for key in self.layer_acts.keys()]
                if act_traces_sums == None:
                    act_traces_sums = [0] * len(acts)
                acts_grad = [self.layer_acts_grads[key] for key in self.layer_acts.keys()]  ##same order with acts
                vt_H_v_sum_per_act = [0] * len(acts)

                prev_model_act_trace = 0
                for iter in range(self.max_iter):
                    v = self.sample_rademacher(acts)
                    H_v = torch.autograd.grad(acts_grad, acts, v, only_inputs=True, retain_graph=True)
                    vt_H_v = [torch.mean(h_v * v_t) for (h_v, v_t) in zip(H_v, v)]

                    vt_H_v_sum_per_act = [vt_H_v_sum_per_act[index] + vt_H_v[index] for index, item in
                                          enumerate(vt_H_v_sum_per_act)]
                    vt_H_v_mean_per_act = [item / (iter + 1) for item in vt_H_v_sum_per_act]
                    current_model_act_trace = torch.mean(torch.stack(vt_H_v_mean_per_act))

                    diff_ratio = abs(current_model_act_trace - prev_model_act_trace) / (
                            prev_model_act_trace + self.eps)
                    if diff_ratio < self.tolerance and iter > 10:  ##TODO magic number
                        break
                    # if iter == 50:  ##TODO for debug
                    #     break

                    prev_model_act_trace = current_model_act_trace
                act_traces_per_sample.append(vt_H_v_mean_per_act)
                cnt += 1
                if cnt >= num_samples:
                    break

        if unfused_training:
            self.unfused_model.train()
        self.reset_act_gradient_and_hooks()  ##TODO have issues to reset the input grad to False
        act_traces_stack = torch.stack([torch.stack(item) for item in act_traces_per_sample])
        act_traces = torch.mean(act_traces_stack, dim=0)
        res_dict = {}
        for index, key in enumerate(self.layer_acts.keys()):
            res_dict[key] = act_traces[index]

        self.layer_acts = []
        self.layer_acts_grads = []
        return res_dict

    def insert_hook(self, model, target_module_list):
        intern_outputs = []
        for layer, module in model.named_modules():
            for target_module in target_module_list:
                # print("layer:",layer)
                # print("target_model:",target_module)
                if layer == target_module:
                    logging.debug("Collect: %s" % (module))
                    # print("Collect: %s" % (module))
                    intern_outputs.append(Node_collector(module))

        logging.info("Total %d hook inserted" % (len(intern_outputs)))
        # print("Total %d hook inserted" % (len(intern_outputs)))
        return model, intern_outputs

    def insert_hook_quantize(self, model, target_module_list):
        intern_outputs = []
        for layer, module in model.named_modules():
            for target_module in target_module_list:
                # print("layer:",layer)
                length = len("_model.")
                new_key = layer[length:]
                # print("target_model:",target_module)
                if new_key == target_module:
                    logging.debug("Collect: %s" % (module))
                    # print("Collect: %s" % (module))
                    intern_outputs.append(Node_collector(module))
        logging.info("Total %d hook inserted" % (len(intern_outputs)))
        # print("Total %d hook inserted" % (len(intern_outputs)))
        return model, intern_outputs

    def get_act_gap(self, fp32_model, q_model):
        """
        Estimates each activation gap between quantized model and float model
        """
        self.handle_acts = []
        fp32_model.eval()
        # temp_model = fuse_fx(fp32_model.model)
        temp_model = fp32_model
        # target_module_list = [nn.ReLU] # Insert hook for FP32 model
        target_module_list = self.op_list
        temp_model, intern_outputs = self.insert_hook(temp_model, target_module_list)
        # intern_outputs={}
        for input, target in self.dataloader:
            temp_model(input)
            break

        fp32_act_out = {}
        for i, intern_output in enumerate(intern_outputs):
            stat_features = intern_output.out_features.view(-1)
            fp32_act_out[target_module_list[i]] = stat_features.cpu().data.numpy()
            # break
        for i in intern_outputs:
            # print(i)
            i.remove()
        target_module_list = self.op_list
        q_model, intern_outputs = self.insert_hook_quantize(q_model, target_module_list)
        for input, target in self.dataloader:  # only one sample
            q_model(input)
            break
        qnt_act_out = {}
        intern_outputs = {}
        for i, intern_output in enumerate(intern_outputs):
            stat_features = intern_output.out_features.view(-1)
            qnt_act_out[target_module_list[i]] = stat_features.dequantize().cpu().data.numpy()
            # break
        for i in intern_outputs:
            # print(i)
            i.remove()
        act_gap = {}
        mse_gap = {}
        for fp_i, int_i in zip(fp32_act_out, qnt_act_out):
            activation_qnt_error = fp32_act_out[fp_i] - qnt_act_out[int_i]
            mse_gap[fp_i] = self.mse_metric_gap(fp32_act_out[fp_i], qnt_act_out[int_i])
            act_gap[fp_i] = np.sum(activation_qnt_error) / activation_qnt_error.size
        return act_gap, mse_gap

    def get_avg_traces(self, enable_act=True, num_samples=32):
        """
        Estimates average hessian trace for each parameter
        """
        assert num_samples > 0
        traces = {}
        weight_traces = self.get_weight_traces(num_samples)
        traces['weight'] = weight_traces
        act_trace = {}
        if enable_act:
            act_gap, mse_gap = self.get_act_gap(self.model, self.q_model)
            act_traces = self.get_act_traces(num_samples)
            for i, j in zip(act_traces, mse_gap):
                # currently use mse to analysis
                act_trace[i] = float(act_traces[i]) + float(mse_gap[j])  # Tensor->float
            traces['activation'] = act_traces
        return traces
    def cal_avg_traces(self, enable_act=False,num_samples=32):
        """calculate the average traces for given model

        Args:
            enable_act (_type_): _description_
            num_samples (int, optional): _description_. Defaults to 32.

        Returns:
            _type_: _description_
        """
        assert num_samples>0
        import tqdm
        avg_total_trace=0.
        avg_traces_per_iter=[]
        mean_avg_traces_per_param=None
        for i in tqdm.tqdm(range(self.max_iter)):
            avg_traces_per_iter.append(self._calc_avg_traces_per_param())
            mean_avg_traces_per_param=self._get_mean(avg_traces_per_iter)
            mean_avg_total_trace=torch.sum(mean_avg_traces_per_param)
            diff_avg=abs(mean_avg_total_trace-avg_total_trace)/(avg_total_trace+self.eps)
            # logger.info("self.tolerance:"+str(self.tolerance))
            if diff_avg<self.tolerance:
                logger.info("End of hessian trace calculation")
                break
            avg_total_trace=mean_avg_total_trace
        
        results={}
        indx=0
        for n,p in self.model.named_parameters():
            results[n]=float(mean_avg_traces_per_param[indx])
            indx+=1
        return results
            
        
    def _calc_avg_traces_per_param(self):
        # v = self._parameter_handler.sample_rademacher_like_params()
        v=self.sample_rademacher_like_params()
        # vhp = self._parameter_handler.sample_normal_like_params()
        vhp=self.sample_normal_like_params()
        num_all_data = self._num_data_iter * self._batch_size
        idx=0
        for gradients in self._gradients_calculator:
            idx+=1
            vhp_curr = torch.autograd.grad(gradients,
                                           self.params,
                                           grad_outputs=v,
                                           only_inputs=True,
                                           retain_graph=False)
            vhp = [a + b * float(self._batch_size) + 0. for a, b in zip(vhp, vhp_curr)]
        vhp = [a / float(num_all_data) for a in vhp]
        avg_traces_per_param = torch.stack([torch.sum(a * b) / a.size().numel() for (a, b) in zip(vhp, v)])
        logger.info("idx:",idx)
        return avg_traces_per_param
    @staticmethod
    def _get_mean(data):
        return torch.mean(torch.stack(data), dim=0)


# copy from torch.quantization._numeric_suite
def _find_match(
        str_list: Union[Dict[str, Any], List[str]], key_str: str,
        postfix: str,
) -> Optional[str]:
    split_str = key_str.split(".")
    if split_str[-1] == postfix:
        match_string = "".join(key_str.split(".")[0:-1])
        for s2 in str_list:
            pattern1 = "".join(s2.split(".")[0:-1])
            pattern2 = "".join(s2.split(".")[0:-2])
            if match_string == pattern1:
                return s2
            if match_string == pattern2:
                return s2

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        if postfix == "_packed_params":
            match_string = "".join(key_str.split(".")[0:-2])
            if len(match_string) == 0:
                return None
            for s2 in str_list:
                pattern1 = "".join(s2.split(".")[0:-1])
                pattern2 = "".join(s2.split(".")[0:-2])
                if match_string == pattern1:
                    return s2
                if match_string == pattern2:
                    return s2
        return None
    else:
        return None


##copy form torch.quantization._numeric_suite
def compare_weights(
        float_dict: Dict[str, Any], quantized_dict: Dict[str, Any]
) -> Dict[str, Dict[str, torch.Tensor]]:
    r"""Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage::

        wt_compare_dict = compare_weights(
            float_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(
                key,
                compute_error(
                    wt_compare_dict[key]['float'],
                    wt_compare_dict[key]['quantized'].dequantize()
                )
            )

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """

    weight_dict: Dict[str, Dict] = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, "weight")
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]["float"] = float_dict[match_key]
            weight_dict[key]["quantized"] = quantized_dict[key]
            continue

        # For matching "fc.weight" and "fc._packed_params._packed_params"
        match_key = _find_match(float_dict, key, "_packed_params")
        if match_key is not None:
            weight_dict[match_key] = {}
            weight_dict[match_key]["float"] = float_dict[match_key]
            weight_dict[match_key]["quantized"] = quantized_dict[key][0]
            ##TODO:should consider more models in further work

        # For LSTM
        split_str = key.split(".")
        if split_str[-1] == "param" and split_str[-3] == "_all_weight_values":
            layer = split_str[-2]
            module_name = ".".join(split_str[:-3])
            float_weight_ih_key = module_name + ".weight_ih_l" + layer
            float_weight_hh_key = module_name + ".weight_hh_l" + layer
            if float_weight_ih_key in float_dict and float_weight_hh_key in float_dict:
                weight_dict[key] = {}
                weight_dict[key]["float"] = float_dict[float_weight_ih_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][0].__getstate__()[0][0]
                )
                weight_dict[key]["float"] = float_dict[float_weight_hh_key]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][1].__getstate__()[0][0]
                )

    return weight_dict


def hawq_top(fp32_model, q_model, dataloader, criterion, enable_act):
    orig_eval = True
    if fp32_model.training:
        orig_eval = False
    fp32_model.eval()
    use_nccf = 2
    if use_nccf==0:
        fp32_model=fuse_fx(fp32_model.model)
        ht = HessianTraceEstimator(fp32_model, data_loader=dataloader,criterion=criterion)
        traces = ht.get_average_traces()
        logger.info("use nncf")
    elif use_nccf==1:
        ht = HessianTrace(fp32_model, dataloader,q_model)
        traces=ht.cal_avg_traces(enable_act=False,num_samples=32)
        logger.info("use hawq")
    else:
        ht = HessianTrace(fp32_model, dataloader,q_model)
        traces = ht.get_avg_traces(False,100)['weight']
        return traces
    op_to_traces={}
    for i in traces:
        # print(i,traces[i])
        length=len(".weight") #weights->op
        op_name=i[0:-length]
        if 'weight' in i : # remove bias
            op_to_traces[op_name]=traces[i]
    for i in op_to_traces:#
        # length=len(".weight") #weights->op
        # op_name=i[0:-length]
        logger.info(str(i)+"  "+str(op_to_traces[i]))
    return op_to_traces
    # assert False
    ######################Woring: please don't remove below code , it will be avaliable in next debug############################
    assert False
    q_model_state_dict = {}
    for key in q_model.state_dict().keys():
        length = len("_model.")
        new_key = key[length:]
        q_model_state_dict[new_key] = q_model.state_dict()[key]
    weight_quant_loss = compare_weights(ht.model.state_dict(), q_model_state_dict)
    pertur_lst = {}
    for key in weight_quant_loss:
        op_float_tensor = weight_quant_loss[key]['float']
        op_qnt_tensor = weight_quant_loss[key]['quantized'].dequantize()
        diff_l2 = (torch.norm(op_float_tensor - op_qnt_tensor, p=2) ** 2)
        pertur_lst[key] = diff_l2

    if enable_act:
        act_to_traces = traces['activation']
        for trace_i, pertur_i, act_i in zip(op_to_traces.keys(), pertur_lst.keys(), act_to_traces.keys()):
            # Formula:Omig=Trace*L2+act_trace
            op_to_traces[trace_i] = pertur_lst[pertur_i] * op_to_traces[trace_i] + act_to_traces[act_i]
    else:
        for trace_i, pertur_i in zip(op_to_traces.keys(), pertur_lst.keys()):
            op_to_traces[trace_i] = pertur_lst[pertur_i] * op_to_traces[trace_i]  # Formula:Omig=Trace*L2
    if orig_eval == False:
        fp32_model.train()
    return op_to_traces