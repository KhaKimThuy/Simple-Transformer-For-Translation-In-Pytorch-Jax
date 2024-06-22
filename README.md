# Performance report of Transformer model for translate task based on Pytorch and Jax framwork 

## Dataset

Dataset from https://huggingface.co/datasets/Helsinki-NLP/opus_books/viewer/en-hu , there are a lot of other translation language datasets, you can get another you wish there.

## Requirements

The code requires several libraries, just install what it needs 
`pip install torch datasets jax`

## Config

Change configure file in folder `config` to your wish

## Training

`python train.py --config_path ../config/config.json`
