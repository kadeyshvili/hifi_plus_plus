# Generative Adversarial Networks for Audio Super-Resolution with Varying Sample Rate

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#license">License</a>
</p>


## About

This repository contains the implementation for my diploma research on Generative Adversarial Networks for Audio Super-Resolution with Varying Sample Rate. The current branch implements classic HiFi++ pipeline as a baseline of our research.



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
- *datasets.train.vctk_wavs_dir*: Directory containing high-resolution training audio
- *datasets.val.vctk_wavs_dir*: Directory containing high-resolution validation audio
- *model.sampling_rate*: Source sampling rate (e.g., 16000)
- *model.input_freq*: Downgraded sampling rate (e.g., 8000 or 4000)

## Experiments

### Reproducing Best Results

To train the model with our optimal configuration:

```bash
python3 train.py  "trainer.monitor=min val_LSD" datasets.train.vctk_wavs_dir=<path_to_dir_with_wavs> datasets.val.vctk_wavs_dir=<path_to_dir_with_wavs> dataloader.train.batch_size=8 dataloader.val.batch_size=8 trainer.log_step=500 trainer.n_epochs=400 trainer.epoch_len=500 datasets.train.dataset_split_file=<path_to_split_file_for_training> datasets.val.dataset_split_file=<path_to_split_file_for_validation> datasets.train.split=True datasets.val.split=True datasets.train.sampling_rate=16000 datasets.val.sampling_rate=16000 datasets.train.input_freq=4000 datasets.val.input_freq=4000 
```

### How to run inference
To evaluate a trained model:
```bash
python3 inference.py -cn="inference_config" inferencer.from_pretrained="path_to_pretrained_model" datasets.test.split=True datasets.test.vctk_wavs_dir=<path_to_dir_with_wavs> dataloader.test.batch_size=4 datasets.test.dataset_split_file=<path_to_split_file/test.txt> 
```



## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)