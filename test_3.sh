#!/bin/bash

# Define the lists of languages and models
eval_languages=("pl" "de" "ru" "uk" "fr" "es" "it" "en" )

config_dir="all_models/all_models_3/fine_tuned" # path to folders with models to evaluate

output_file="all_models/all_models_3/evaluation_results.csv"

for folder in "$config_dir"/*/; do  # iterate over each model
  for lang in "${eval_languages[@]}"; do
    cmd="python test.py --config ${folder}configuration.yaml --output_file ${output_file} --eval_languages ${lang}"
    echo "Running: $cmd"
    $cmd
  done
done

#### to comment out if flat structure is used
base_line_dir="all_models/all_models_3"
output_file="all_models/all_models_3/baseline_evaluation_results.csv"


for folder in "$base_line_dir"/*/; do  # iterate over each model
  for lang in "${eval_languages[@]}"; do
    cmd="python test.py --config ${folder}configuration.yaml --output_file ${output_file} --eval_languages ${lang}"
    echo "Running: $cmd"
    $cmd
  done
done
