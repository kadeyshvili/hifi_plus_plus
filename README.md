# Generative Adversarial Networks for Audio Super-Resolution with Varying Sample Rate

<p align="center">
  <a href="#about">About</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#license">License</a>
</p>


## About

This repository contains the implementation for my diploma research on Generative Adversarial Networks for Audio Super-Resolution with Varying Sample Rate. The current branch implements our approach of training blocks separately. This methodology involves alternating between different batches during training - first processing a 4$\rightarrow$8 kHz batch, then an 8$\rightarrow$16 kHz batch. Each upsampling task is optimized with its own dedicated loss functions, and performance metrics are calculated independently for each resolution stage.


## Features

- Hybrid Architecture: Combines strengths of NU-Wave2 and HiFi++ models
- Multi-Resolution Support: Handles multiple upsampling scenarios (4$\rightarrow$8 kHz, 8$\rightarrow$16 kHz, and 4$\rightarrow$16 kHz)
- Modular Design: Components can be trained independently or jointly
- Optimized Training: Custom training techniques for optimizing model performance
- Efficient Processing: Reduced computational overhead compared to traditional approaches

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## Usage


### Training a Model
The basic command for training:

```bash
python3 train.py -cn=hifigan HYDRA_CONFIG_ARGUMENTS
```

### Configuration

The model uses Hydra for configuration management. Key configuration parameters include:

- *datasets.train.dataset_split_file*: Path to training dataset split
- *datasets.val.dataset_split_file*: Path to validation dataset split
- *datasets.train.wavs_dir_4khz*: Directory containing 4kHz audios
- *datasets.val.wavs_dir_4khz*: Directory containing 4kHz audios
- *datasets.train.wavs_dir_8khz*: Directory containing 8kHz audios
- *datasets.val.wavs_dir_8khz*: Directory containing 8kHz audios
- *datasets.train.wavs_dir_16khz*: Directory containing 16kHz audios
- *datasets.val.wavs_dir_16khz*: Directory containing 16kHz audios

## Experiments

### How to resample initial audio 
To resample your high-resolution audio files to a lower sample rate (e.g., from 48 kHz to 4 kHz), use the following command:
```bash
python3 resample_data.py --data_dir <path for 48 kHz data> --out_dir <path for 4 kHz data> --target_sr 4000
```

### Reproducing Best Results

To train the model with our optimal configuration:

```bash
python3 train.py  model.generator_config.upsample_block_rates=[2] model.generator_config.upsample_block_kernel_sizes=[4] model.generator_config.use_spectralmasknet=False datasets.train.split=True datasets.val.split=True datasets.train.wavs_dir_4khz=<path_to_4kHz_dir> datasets.train.wavs_dir_8khz=<path_to_8kHz_dir> datasets.train.wavs_dir_16khz=<path_to_16kHz_dir> datasets.val.wavs_dir_4khz=<path_to_4kHz_dir> datasets.val.wavs_dir_8khz=<path_to_8kHz_dir> datasets.val.wavs_dir_16khz=<path_to_16kHz_dir> "trainer.monitor=min val_LSD" dataloader.train.batch_size=4 dataloader.val.batch_size=4 trainer.log_step=500 trainer.n_epochs=400 trainer.epoch_len=500 datasets.train.dataset_split_file=<path_to_split_file/training.txt> datasets.val.dataset_split_file=/<path_to_split_file/val.txt> +writer.api_key=<your_api_key>
```

### How to run inference
To evaluate a trained model:
```bash
python3 inference.py -cn="inference_config" inferencer.from_pretrained="path_to_pretrained_model" model.generator_config.upsample_block_rates=[2] model.generator_config.upsample_block_kernel_sizes=[4] model.generator_config.use_spectralmasknet=False datasets.test.split=True datasets.test.wavs_dir_4khz=<path_to_4kHz_dir> datasets.test.wavs_dir_8khz=<path_to_8kHz_dir> datasets.test.wavs_dir_16khz=<path_to_16kHz_dir> dataloader.test.batch_size=4 datasets.test.dataset_split_file=<path_to_split_file/test.txt> 
```



## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
