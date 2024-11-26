#!/bin/bash

# Define the lists of languages and models
ft_languages=("pl" "de" "ru" "uk" "fr" "es" "it" "en")

config_dir="configs" # path to NOT TRAINED MODELS (.yaml configurations
config_names=("w2v_aasist.yaml" "whisper_aasist.yaml" "rawgat_st.yaml" "lfcc_mesonet.yaml")

model_out_dir="all_models/mlaad"


for cf in "${config_names[@]}"; do # iterate over each model
  for ft_lang in "${ft_languages[@]}"; do
    cmd="python finetune.py --config ${config_dir}/${cf} --model_out_dir ${model_out_dir} --ft_languages ${ft_lang} --lr 0.0001 --epochs 25 --batch_size 32"
    echo "Running: $cmd"
    $cmd
  done
done
