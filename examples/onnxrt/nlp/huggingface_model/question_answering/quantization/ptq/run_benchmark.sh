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
    esac
  done

}

# run_benchmark
function run_benchmark {

    if [[ "${input_model}" =~ "spanbert" ]]; then
        model_name_or_path="mrm8488/spanbert-finetuned-squadv1"
        extra_cmd='--version_2_with_negative=False'
    elif [[ "${input_model}" =~ "bert-base-multilingual" ]]; then
        model_name_or_path="salti/bert-base-multilingual-cased-finetuned-squad"
        extra_cmd='--version_2_with_negative=False'
    elif [[ "${input_model}" =~ "distilbert-base-uncased" ]]; then
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        extra_cmd='--version_2_with_negative=False'
    elif [[ "${input_model}" =~ "xlm-roberta-large" ]]; then
        model_name_or_path="deepset/xlm-roberta-large-squad2"
        extra_cmd='--version_2_with_negative=True'
    elif [[ "${input_model}" =~ "bert-large-uncased-whole-word-masking" ]]; then
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        extra_cmd='--version_2_with_negative=False'
    elif [[ "${input_model}" =~ "roberta-large" ]]; then
        model_name_or_path="deepset/roberta-large-squad2"
        extra_cmd='--version_2_with_negative=True'
    fi

    python main.py \
            --model_path ${input_model} \
            --config ${config} \
            --mode=${mode} \
            --model_name_or_path=${model_name_or_path} \
            --output_dir './output' \
            --benchmark \
            ${extra_cmd}
            
}

main "$@"