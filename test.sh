#!/bin/bash

# Define the lists of languages and models
eval_languages=("pl" "de" "ru" "uk" "fr" "es" "it" "en" )

config_dir="/net/people/plgrid/plgbartlomiejmarek/are_audio_df_polyglots/models/baselines/" # path to folders with models to evaluate

output_file="${config_dir}/evaluation_results.csv"

for folder in "$config_dir"/*/; do  # iterate over each model
  for lang in "${eval_languages[@]}"; do
    cmd="python test.py --config ${folder}configuration.yaml --output_file ${output_file} --eval_languages ${lang}"
    echo "Running: $cmd"
    $cmd
  done
done

