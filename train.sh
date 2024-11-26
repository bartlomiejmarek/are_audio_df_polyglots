#!/bin/bash

# Install the required packages for w2v
#fairseq

# git clone https://github.com/pytorch/fairseq
# cd fairseq
# pip install --editable ./
# cd ..
# Pre-trained wav2vec 2.0 XLSR (300M) model
# wget -P src/models/assets/ https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt

# Define the base directories
out_dir="all_models"
config_dir="configs"
config_names=("w2v_aasist.yaml" "whisper_aasist.yaml" "rawgat_st.yaml" "lfcc_mesonet.yaml" "lfcc_aasist.yaml")

# Iterate over each model
for cf in "${config_names[@]}"; do
      cmd="python train.py --config ${config_dir}/${cf} --model_out_dir ${out_dir}"
      echo "Running: $cmd"
      $cmd
done
