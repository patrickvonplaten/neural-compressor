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
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;    
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          output_model=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_tuning {
  
    if [[ "${input_model}" =~ "spanbert" ]]; then
        model_name_or_path="mrm8488/spanbert-finetuned-squadv1"
        num_heads=12
        hidden_size=768
        extra_cmd='--version_2_with_negative=False --dataset_name=squad'
    elif [[ "${input_model}" =~ "bert-base-multilingual" ]]; then
        model_name_or_path="salti/bert-base-multilingual-cased-finetuned-squad"
        num_heads=12
        hidden_size=768
        extra_cmd='--version_2_with_negative=False --dataset_name=squad'
    elif [[ "${input_model}" =~ "distilbert-base-uncased" ]]; then
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        num_heads=12
        hidden_size=768
        extra_cmd='--version_2_with_negative=False --dataset_name=squad'
    elif [[ "${input_model}" =~ "xlm-roberta-large" ]]; then
        model_name_or_path="deepset/xlm-roberta-large-squad2"
        num_heads=16
        hidden_size=1024
        extra_cmd='--version_2_with_negative=True --dataset_name=squad'
    elif [[ "${input_model}" =~ "bert-large-uncased" ]]; then
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        num_heads=16
        hidden_size=1024
        extra_cmd='--version_2_with_negative=False --dataset_name=squad'
    elif [[ "${input_model}" =~ "roberta-large" ]]; then
        model_name_or_path="deepset/roberta-large-squad2"
        num_heads=16
        hidden_size=1024
        extra_cmd='--version_2_with_negative=True --dataset_name=squad'
    fi

    python main.py \
            --model_path ${input_model} \
            --save_path ${output_model} \
            --config ${config} \
            --output_dir './output' \
            --model_name_or_path=${model_name_or_path} \
            --num_heads ${num_heads} \
            --hidden_size ${hidden_size} \
            --tune \
            ${extra_cmd}
}

main "$@"
