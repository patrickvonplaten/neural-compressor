#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import argparse
from tensorflow.keras.applications import ResNet50


def get_resnet50_model(saved_path):
    assert saved_path is not None, "save path should not be None"
    model = ResNet50(weights='imagenet')
    model.save(saved_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export pretained keras model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_model',
                        type=str,
                        help='path to exported model file')

    args = parser.parse_args()
    get_resnet50_model(args.output_model)
