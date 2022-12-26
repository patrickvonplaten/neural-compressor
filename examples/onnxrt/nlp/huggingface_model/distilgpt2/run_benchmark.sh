#!/bin/bash
set -x

function main {

  init_params "$@"
  define_mode
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
    esac
  done

}

function define_mode {
    if [[ ${mode} == "accuracy" ]]; then
      mode_cmd=" --benchmark --mode=accuracy"
    elif [[ ${mode} == "performance" ]]; then
      mode_cmd=" --iter ${iters} --benchmark"
    else
      echo "Error: No such mode: ${mode}"
      exit 1
    fi
}

# run_benchmark
function run_benchmark {
    model_type='gpt2'
    model_name_or_path='Intel/distilgpt2-wikitext2'
    batch_size=1
    test_data='wiki.test.raw'
    data_path='/tf_dataset2/datasets/wikitext/wikitext-2-raw/'
    python gpt2.py --model_path ${input_model} \
                        --data_path ${data_path}${test_data} \
                        --model_type ${model_type} \
                        --model_name_or_path ${model_name_or_path} \
                        --config ${config} \
                        --per_gpu_eval_batch_size ${batch_size} \
                        ${mode_cmd}
}

main "$@"

