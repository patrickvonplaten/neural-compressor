Step-by-Step
============

This document is used to enable Tensorflow Keras models using Intel® Neural Compressor.
This example can run on Intel CPUs and GPUs.


## Prerequisite

### 1. Installation
```shell
# Install Intel® Neural Compressor
pip install neural-compressor
```
### 2. Install Intel Tensorflow
```shell
pip install intel-tensorflow
```
> Note: Supported Tensorflow [Version](../../../../../../../README.md).

### 3. Quantizing the model on Intel CPU
Install the Intel Extension for Tensorflow for Intel CPUs. It's mandatory for quantizing the input Keras model and saving Keras model out on Intel CPUs.

```shell
pip install --upgrade intel-extension-for-tensorflow[cpu]
```

### 4. Prepare Pretrained model

The pretrained model is provided by [Keras Applications](https://keras.io/api/applications/). prepare the model, Run as follow: 
 ```
python prepare_model.py   --output_model=/path/to/model
 ```
`--output_model ` the model should be saved as SavedModel format or H5 format.

## Run Command
  ```shell
  bash run_tuning.sh --input_model=./path/to/model --output_model=./result --dataset_location=/path/to/evaluation/
  bash run_benchmark.sh --input_model=./path/to/model --mode=performance --dataset_location=/path/to/evaluation/dataset
  ```

