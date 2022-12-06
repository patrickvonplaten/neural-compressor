#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --data_path=*)
          data_path=$(echo $var |cut -f2 -d=)
      ;;
      --model_name_or_path=*)
          model_name_or_path=$(echo $var |cut -f2 -d=)
      ;;
      --task=*)
          task=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_benchmark
function run_benchmark {

    python main_onnx.py \
            --model_name_or_path ${model_name_or_path} \
            --model_path ${input_model} \
            --config ${config} \
            --data_path ${data_path} \
            --task ${task} \
            --mode=${mode} \
            --benchmark
            
}

main "$@"

