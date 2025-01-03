# Implementation of neural vocoder HiFi-GAN

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>


## About

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw3_nv).

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

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=hifigan HYDRA_CONFIG_ARGUMENTS
```


### How to reproduce the results of the best model (train)

- to train your model use config *hifi_dataset* in configs specify path to audio files in *datasets.train.data_path=""  datasets.val.data_path=""*

run the following command (fill in the paths to a data set):

```bash
python3 train.py -cn="hifigan" trainer.n_epochs=110 trainer.epoch_len=500 HYDRA_CONFIG_ARGUMENTS
```

## Link to pretrained HiFi-GAN model 
[link](https://drive.google.com/file/d/17C3iA42W5fkoCyxZqO-Tu_-Z92IyBaMO/view?usp=sharing)

## How to run synthesize
- First you need to download pretrained model directly by following the link above or by running script that automatically downloads pretrained model. The pretrained model will be saved to scripts directory. To run script use the following commands

```bash
python3 scripts/download_weights.py
```


- To synthesize audios from inital wavs provide specify path to directory containing files with audios .wav  by using *datasets.test.data_path=""*  you need also provide path to pretrained model by specifying *inferencer.from_pretrained='path_to_your_model'*

```bash
python3 synthesize.py -cn="synthesize_from_wav"  HYDRA_CONFIG_ARGUMENTS
```

- If you want to synthesize audio from text from file .txt you need to specify path to directory containing files with text .txt *datasets.test.data_path=""*  you need also provide path to pretrained model by specifying *inferencer.from_pretrained='path_to_your_model'* Then you need to run the following command

```bash
python3 synthesize.py -cn="synthesize_from_text"  HYDRA_CONFIG_ARGUMENTS
```
- If you want to synthesize audio by typing in terminal you need to use *inferencer.text_from_console="your text"* you need also provide path to pretrained model by specifying *inferencer.from_pretrained='path_to_your_model'* Then you need to run the following command

```bash
python3 synthesize.py -cn="synthesize_from_text"  inferencer.text_from_console="your text" HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
