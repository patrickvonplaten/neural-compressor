import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import tqdm
from datasets import Dataset
from transformers import EvalPrediction
from transformers.trainer_pt_utils import nested_concat
from transformers.trainer_utils import EvalLoopOutput
from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)

class ORTModel:
    def __init__(
        self,
        model,
        execution_provider: Optional[str] = "CPUExecutionProvider",
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        label_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model:
                onnx.onnx_ml_pb2.ModelProto.
            execution_provider (:obj:`str`, `optional`):
                ONNX Runtime execution provider to use.
            compute_metrics (`Callable[[EvalPrediction], Dict]`, `optional`):
                The function that will be used to compute metrics at evaluation. Must take an `EvalPrediction` and
                return a dictionary string to metric values.
            label_names (`List[str]`, `optional`):
                The list of keys in your dictionary of inputs that correspond to the labels.
        """
        self.compute_metrics = compute_metrics
        self.label_names = ["labels"] if label_names is None else label_names
        self.session = InferenceSession(model.SerializeToString(), providers=[execution_provider])
        self.onnx_input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}

    def evaluation_loop(self, dataset: Dataset):
        """
        Run evaluation and returns metrics and predictions.
        Args:
            dataset (`datasets.Dataset`):
                Dataset to use for the evaluation step.
        """
        logger.info(f"***** Running evaluation *****")
        all_preds = None
        all_labels = None
        for step, inputs in tqdm.tqdm(enumerate(dataset), desc='eval'):
            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            if has_labels:
                labels = tuple(np.array([inputs.get(name)]) for name in self.label_names)
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None
            onnx_inputs = {key: np.array([inputs[key]]) for key in self.onnx_input_names if key in inputs}
            preds = self.session.run(None, onnx_inputs)
            if len(preds) == 1:
                preds = preds[0]
            all_preds = preds if all_preds is None else nested_concat(all_preds, preds, padding_index=-100)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=len(dataset))