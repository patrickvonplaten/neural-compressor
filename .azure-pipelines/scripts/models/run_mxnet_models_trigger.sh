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

FRAMEWORK="mxnet"
FRAMEWORK_VERSION="1.7.0"


# ======== set up config for mxnet models ========
if [ "${model}" == "resnet50v1" ]; then
    model_src_dir="image_recognition/cnn_models/quantization/ptq"
    dataset_location="/tf_dataset/mxnet/val_256_q90.rec"
    input_model="/tf_dataset/mxnet/resnet50_v1"
    yaml="cnn.yaml"
    strategy="mse"
    batch_size=32
    new_benchmark=false
    tuning_cmd="bash run_tuning.sh --topology=resnet50_v1 --dataset_location=${dataset_location} --input_model=${input_model}"
    benchmark_cmd="bash run_benchmark.sh --topology=resnet50_v1 --dataset_location=${dataset_location} --batch_size=1 --iters=500 --mode=benchmark"
fi


/bin/bash run_model_trigger_common.sh --yaml=${yaml} --framework=${FRAMEWORK} --fwk_ver=${FRAMEWORK_VERSION} \
--model=${model} --model_src_dir=${model_src_dir} --dataset_location=${dataset_location} \
--input_model=${input_model} --batch_size=${batch_size} --strategy=${strategy} --new_benchmark=${new_benchmark} \
--tuning_cmd="${tuning_cmd}" --benchmark_cmd="${benchmark_cmd}" --tune_acc=${tune_acc} --build_id=${build_id}