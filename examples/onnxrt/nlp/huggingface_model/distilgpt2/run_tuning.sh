#!/bin/bash
set -x

function main {
  init_params "$@"
  run_tuning
}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --data_path=*)
          data_path=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;; 
    esac
  done

}

# run_tuning
function run_tuning {
    model_type='gpt2'
    model_name_or_path='Intel/distilgpt2-wikitext2'
    test_data='wiki.test.raw'
    
    python main.py --model_path ${input_model} \
                  --data_path ${data_path}${test_data} \
                  --model_type ${model_type} \
                  --model_name_or_path ${model_name_or_path} \
                  --tune \
                  --config ${config} \
                  --output_model ${output_model}
}

main "$@"



