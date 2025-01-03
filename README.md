# Are audio DeepFake detection models polyglots?

## Overview

This repository contains the official implementation of the paper **"Are audio DeepFake detection models polyglots?"**, authored by Bartłomiej Marek, Piotr Kawa, and Piotr Syga. 


In this study, we introduce a benchmark for advancing audio DF detection in multilingual settings and empirically explore three essential questions in this area.  Specifically, we aim to check the extent to which detection efficacy varies by language, whether English benchmark-trained models are sufficient for effective cross-linguistic detection, and which targeted strategies best support DF detection in specific languages, precisely intra- or cross-lingual adaptations, \textbf{even assuming access to very limited non-English resources}.

The paper is available [here](https://arxiv.org/abs/2412.17924)

The codebase supports training, fine-tuning, and evaluating multilingual audio DeepFake detection models:

- W2V+AASIST
- Whisper+AASIST
- LFCC+AASIST
- LFCC+MesoNet
- RawGAT-ST

Available languages include:

```
- Germanic: en (English), de (German),  
- Romance: fr (French), es (Spanish), it (Italian),  
- Slavic: pl (Polish), ru (Russian), uk (Ukrainian) 
```

---
## Datasets
---
For the experiments we use: ASVspoof2019LA and MLAAD. 

MLAAD dataset has the following structure
```
MLAAD/
README.md (36-42)
├── bonafide
│   ├── de_DE
│   │   └── by_book
│   │       ├── female
│   │       │   └── angela_merkel
│   │       │   │    └── merkel_alone
│   │       |   │        └── wavs
│   │       |   ├── eva_k
│   │       │   │    ├── grune_haus
│   │       |   │    |    └── wavs
...
|   ├── en_UK
│   │   └── by_book
│   │       └── female
│   │           └── elizabeth_klett
│   │               ├── jane_eyre
│   │               │   └── wavs
│   │               └── wives_and_daughters
│   │                   └── wavs
....

├── fake
│   ├── ar
│   │   └── tts_models_multilingual_multi-dataset_xtts_v1.1
│   ├── bg
│   │   └── tts_models_bg_cv_vits
│   ├── cs
│   │   └── tts_models_cs_cv_vits
```

### Repository Structure

```
├── configs/                # Example configuration files for training
├── scripts/                # Shell scripts for running experiments
├── src/                    # Core Python scripts for training, fine-tuning, and evaluation
├── test.py                 # Main evaluation script
├── train.py                # Main training script
├── finetune.py             # Main fine-tuning script
├── requirements.txt        # Required dependencies
├── .env                    # Define constant values 

---

## Usage

### Training
Train models from scratch using `train.sh` or the Python script directly.

#### Using `train.sh`
```bash
bash scripts/train.sh
```

#### Using `train.py`

```bash
python train.py \
  --config ${config_dir}/${cf} \
  --model_out_dir ${out_dir}
```

### Fine-Tuning

Use `fine-tuning.sh` or the Python script directly for fine-tuning pre-trained models on specific languages.

#### Using `fine-tuning.sh`

```bash
bash scripts/fine-tuning.sh
```

#### Using `finetune.py`

```bash
python finetune.py \
  --config ${folder}/configuration.yaml \
  --model_out_dir ${model_out_dir} \
  --ft_languages ${ft_lang} \
  --lr ${lr}\
  --weight_decay ${weight_decay}
```

---

### Evaluation

Run the evaluation pipeline using `test.sh` or the Python script directly.

#### Using `test.sh`

Edit `test.sh` to specify the languages for evaluation and execute:

```bash
bash scripts/test.sh
```

#### Using `test.py`

Run the Python script directly with specified parameters:

```bash
python test.py \
  --config ${folder}/configuration.yaml \
  --output_file ${output_file} \
  --eval_languages ${lang}
```

## Results

Results from the evaluation will be saved to the specified `output_file`. Use these outputs to analyze model performance across different languages.
The example of processing of results is shown in `notebooks\post_processing.ipynb`. 


## Citation

If you use this code in your research, please cite:

```
@misc{marek2024audiodeepfakedetectionmodels,
      title={Are audio DeepFake detection models polyglots?}, 
      author={Bartłomiej Marek and Piotr Kawa and Piotr Syga},
      year={2024},
      eprint={2412.17924},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2412.17924}, 
}
```

---

## Contact

For questions or issues, please open an issue in the repository.
