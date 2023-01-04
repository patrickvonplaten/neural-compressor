Step-by-Step
============

This document describes the step-by-step instructions for reproducing the tuning results of PyTorch ResNet50/ResNet18/ResNet101/InceptionV3/Mobilenet_v2/Efficientnet_b0/Efficientnet_b3/Efficientnet_b7 with Intel® Neural Compressor.

# Prerequisite

### 1. Installation

PyTorch 1.8 or higher version is needed with pytorch_fx backend.

```shell
cd examples/pytorch/image_recognition/torchvision_models/quantization/ptq/cpu/fx
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download [ImageNet](http://www.image-net.org/) Raw image to dir: /path/to/imagenet.  The dir include below folder:

```bash
ls /path/to/imagenet
train  val
```

# Run

### 1. ResNet50

```shell
python main.py -t -a resnet50 --pretrained /path/to/imagenet
```

### 2. ResNet18

```shell
python main.py -t -a resnet18 --pretrained /path/to/imagenet
```

### 3. ResNeXt101_32x8d

```shell
python main.py -t -a resnext101_32x8d --pretrained /path/to/imagenet
```

### 4. InceptionV3

```shell
python main.py -t -a inception_v3 --pretrained /path/to/imagenet
```

### 5. Mobilenet_v2

```shell
python main.py -t -a mobilenet_v2 --pretrained /path/to/imagenet
```

### 6. Efficientnet_b0

```shell
python main.py -t -a efficientnet_b0 --pretrained /path/to/imagenet
```
> **Note**
>
> To reduce tuning time and get the result faster, it is recommended to use the
> [`MSE_V2`](/docs/source/tuning_strategies.md#MSE_v2) strategy by specifying the `strategy` field of the `TuningCriterion` with `mse_v2` in `main.py`.

### 7. Efficientnet_b3

```shell
python main.py -t -a efficientnet_b3 --pretrained /path/to/imagenet
```
> **Note**
>
> To reduce tuning time and get the result faster, it is recommended to use the
> [`MSE_V2`](/docs/source/tuning_strategies.md#MSE_v2) strategy by specifying the `strategy` field of the `TuningCriterion` with `mse_v2` in `main.py`.
### 8. Efficientnet_b7

```shell
python main.py -t -a efficientnet_b7 --pretrained /path/to/imagenet
```
> **Note**
>
> To reduce tuning time and get the result faster, it is recommended to use the
> [`MSE_V2`](/docs/source/tuning_strategies.md#MSE_v2) strategy by specifying the `strategy` field of the `TuningCriterion` with `mse_v2` in `main.py`.

