import os
import sys
import numpy as np
import re
import abc
import subprocess
import json
import argparse
import time
from PIL import Image

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
import os
import logging
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
import torch
import transformers

logger = logging.getLogger()


class ONNXRTBertDatasetForINC(CalibrationDataReader):
    def __init__(self, data_dir='/home/yuwenzho/datasets/MRPC/', model_name_or_path='bert-base-cased-finetuned-mrpc', max_seq_length=128,\
                do_lower_case=True, task='mrpc', model_type='bert', dynamic_length=False,\
                evaluate=True, transform=None, filter=None, augmented_model_path='augmented_model.onnx'):
        task = task.lower()
        model_type = model_type.lower()
        self.dynamic_length = dynamic_length
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,
            do_lower_case=do_lower_case)
        dataset = load_and_cache_examples(data_dir, model_name_or_path, \
            max_seq_length, task, model_type, tokenizer, evaluate)
        self.augmented_model_path = augmented_model_path
        self.dataset = []
        i = 20
        for data in dataset:
            i -= 1
            batch = tuple(t.detach().cpu().numpy() if not isinstance(t, np.ndarray) else t for t in data[0:3])
            batch = [batch[0].reshape(1,128), batch[1].reshape(1,128), batch[2].reshape(1,128)]
            self.dataset.append(batch)
            if i == 0:
                break
        self.datasize = len(self.dataset)
        session = onnxruntime.InferenceSession(self.augmented_model_path, None)
        self.input_name = [i.name for i in session.get_inputs()]
        self.enum_data_dicts = iter([dict(zip(self.input_name, data)) for data in self.dataset])

    def get_next(self):
        return next(self.enum_data_dicts, None)

def load_and_cache_examples(data_dir, model_name_or_path, max_seq_length, task, \
    model_type, tokenizer, evaluate):
    from torch.utils.data import TensorDataset

    processor = transformers.glue_processors[task]()
    output_mode = transformers.glue_output_modes[task]
    # Load data features from cache or dataset file
    if not os.path.exists("./dataset_cached"):
        os.makedirs("./dataset_cached")
    cached_features_file = os.path.join("./dataset_cached", 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, model_name_or_path.split('/'))).pop(),
        str(max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Load features from cached file {}.".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        logger.info("Create features from dataset file at {}.".format(data_dir))
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(data_dir) if evaluate else \
            processor.get_train_examples(data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                task=task,
                                                label_list=label_list,
                                                max_length=max_seq_length,
                                                output_mode=output_mode,
        )
        logger.info("Save features into cached file {}.".format(cached_features_file))
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, \
        all_seq_lengths, all_labels)
    return dataset

def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=128,
    task=None,
    label_list=None,
    output_mode="classification",
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    processor = transformers.glue_processors[task]()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Use label list {} for task {}.".format(label_list, task))
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        seq_length = len(input_ids)
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + \
            ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, \
            "Error with input_ids length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, \
            "Error with attention_mask length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, \
            "Error with token_type_ids length {} vs {}".format(
            len(token_type_ids), max_length
        )
        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        feats = InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label=label,
            seq_length=seq_length,
        )
        features.append(feats)
    return features

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED,
            ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        seq_length: (Optional) The length of input sequence before padding.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    seq_length: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def benchmark(model_path, dr):
    session = onnxruntime.InferenceSession(model_path)

    total = 0.0
    runs = 10
    # Warming up
    run = 10
    _ = session.run([], dr.get_next())
    for i in range(runs):
        data = dr.get_next()
        if not data:
            break
 
        start = time.perf_counter()
        _ = session.run([], data)
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def parse_dummy_input(model, benchmark_nums, max_seq_length):
    session = onnxruntime.InferenceSession(model.SerializeToString(), None)
    shapes = []
    lows = []
    highs = []
    for i in range(len(session.get_inputs())):
        input_name = session.get_inputs()[i].name
        input_shapes = session.get_inputs()[i].shape
        shape = [benchmark_nums]
        if input_name == "input_ids":
            low = 0.0
            high = 1000.0
            shape.append(max_seq_length)
        elif 'unique_ids' in input_name:
            low = 0.0
            high = 1000.0
        else:
            low = 0.0
            high = 2.0
            shape.append(max_seq_length)
        shapes.append(tuple(shape))
        lows.append(low)
        highs.append(high)
    return shapes, lows, highs

