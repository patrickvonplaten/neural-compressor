#!/bin/bash

# get parameters
PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"
do
    case $i in
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --tune_acc=*)
            tune_acc=`echo $i | sed "s/${PATTERN}//"`;;
        --build_id=*)
            build_id=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

FRAMEWORK="onnxrt"
FRAMEWORK_VERSION="1.11.0"


# ======== set up config for onnxrt models ========
if [ "${model}" == "resnet50-v1-12" ]; then
    model_src_dir="image_recognition/onnx_model_zoo/resnet50/quantization/ptq"
    dataset_location="/tf_dataset2/datasets/imagenet/ImagenetRaw/ImagenetRaw_small_5000/ILSVRC2012_img_val"
    input_model="/tf_dataset2/models/onnx/resnet50-v1-12/resnet50-v1-12.onnx"
    yaml="resnet50_v1_5.yaml"
    strategy="basic"
    batch_size=1
    new_benchmark=true
    tuning_cmd="bash run_tuning.sh --input_model=${input_model} --config=${yaml}"
    benchmark_cmd="bash run_benchmark.sh --config=${yaml}"
elif [ "${model}" == "bert_base_MRPC_static" ]; then
    model_src_dir="language_translation/bert/quantization/ptq"
    dataset_location="/tf_dataset/pytorch/glue_data/MRPC"
    input_model="/tf_dataset2/models/onnx/bert_base_MRPC/bert.onnx"
    yaml="bert_static.yaml"
    strategy="basic"
    batch_size=8
    new_benchmark=true
    tuning_cmd="bash run_tuning.sh --input_model=${input_model} --config=${yaml}"
    benchmark_cmd="bash run_benchmark.sh --config=${yaml}"
elif [ "${model}" == "bert_base_MRPC_dynamic" ]; then
    model_src_dir="language_translation/bert/quantization/ptq"
    dataset_location="/tf_dataset/pytorch/glue_data/MRPC"
    input_model="/tf_dataset2/models/onnx/bert_base_MRPC/bert.onnx"
    yaml="bert_dynamic.yaml"
    strategy="basic"
    batch_size=8
    new_benchmark=true
    tuning_cmd="bash run_tuning.sh --input_model=${input_model} --config=${yaml}"
    benchmark_cmd="bash run_benchmark.sh --config=${yaml}"
fi


/bin/bash run_model_trigger_common.sh --yaml=${yaml} --framework=${FRAMEWORK} --fwk_ver=${FRAMEWORK_VERSION} \
--model=${model} --model_src_dir=${model_src_dir} --dataset_location=${dataset_location} \
--input_model=${input_model} --batch_size=${batch_size} --strategy=${strategy} --new_benchmark=${new_benchmark} \
--tuning_cmd="${tuning_cmd}" --benchmark_cmd="${benchmark_cmd} --mode=performance" --tune_acc=${tune_acc} --build_id=${build_id}