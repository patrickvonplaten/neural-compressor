#!/bin/bash
set -x

echo "specify fwk version..."
export tensorflow_version='2.9.1'
export pytorch_version='1.12.0+cpu'
export torchvision_version='0.13.0+cpu'
export onnx_version='1.11.0'
export onnxruntime_version='1.11.0'
export mxnet_version='1.7.0'

echo "set up UT env..."
bash /neural-compressor/.azure-pipelines/scripts/ut/env_setup.sh
lpot_path=$(python -c 'import neural_compressor; import os; print(os.path.dirname(neural_compressor.__file__))')
cd /neural-compressor/test || exit 1
find ./adaptor -name "test*.py" | sed 's,\.\/,coverage run --source='"${lpot_path}"' --append ,g' | sed 's/$/ --verbose/'> run.sh

LOG_DIR=/neural-compressor/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut_tf_${tensorflow_version}_pt_${pytorch_version}.log

echo "cat run.sh..."
cat run.sh | tee ${ut_log_name}
echo "-------------"
bash run.sh 2>&1 | tee -a ${ut_log_name}
cp .coverage ${LOG_DIR}/.coverage.adaptor
echo "list all in ${LOG_DIR}"
ls -a ${LOG_DIR}
if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ] || [ $(grep -c "OK" ${ut_log_name}) == 0 ];then
    exit 1
fi