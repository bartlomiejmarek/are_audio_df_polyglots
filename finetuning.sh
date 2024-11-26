#!/bin/bash

# Define the lists of languages and models
ft_languages=("pl" "de" "ru" "uk" "fr" "es" "it" "en")

config_dir="all_models" # path to pre-trained models
model_out_dir="all_models/fine_tuned"

for folder in "$config_dir"/*/; do  # iterate over each model
  for ft_lang in "${ft_languages[@]}"; do
    cmd="python finetune.py --config ${folder}configuration.yaml --model_out_dir ${model_out_dir} --ft_languages ${ft_lang}"
    echo "Running: $cmd"
    $cmd
  done
done
