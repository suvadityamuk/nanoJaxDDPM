# nanoJaxDDPM

## Introduction
This is a minimal (and a little faithless) implementation of the [DDPM paper by Ho et. al.](https://arxiv.org/abs/2006.11239) in JAX and Equinox. This includes a feature-complete DDPM pipeline, a very small UNet, Sinusoidal Positional Embeddings, and logging to Weights & Biases.

Inspiration for this has been taken from nanoDiT, nanoGPT, nanoVLM, and the other nano-class of model implementations coming around lately.

## Getting Started

### Script use
- Clone the repository:
```bash
git clone https://github.com/suvadityamuk/nanoJaxDDPM.git
cd nanoJaxDDPM
```
- Set the following Environment variable (to enable logging to WandB):
```bash
export WANDB_API_KEY=<YOUR-KEY>
```
- Place/upload a `kaggle.json` to your folder (get your credentials through [Kaggle](https://www.kaggle.com) > Settings > Create New Token)
- Run the following set of commands
```bash
python -m venv .venv
source .venv/bin/activate # Assuming Linux (use .venv/Scripts/Activate for Windows)
pip install equinox kagglehub wandb # Install required libraries
mkdir -p /root/.config/kaggle # Place kaggle.json in the right directory
mv kaggle.json /root/.config/kaggle/kaggle.json
chmod 600 /root/.config/kaggle/kaggle.json # Adjust permissions on the file
kaggle competitions download -c cifar-10 # Download dataset
unzip cifar-10.zip # Unzip dataset splits
mkdir train_images # Create dataset to unzip split files into
7z e train.7z -o/content/train_images -y # Unzip files of the train split only
```
- Run training script
```bash
python train.py
```

## Method (What's same, and what's not)

### Benefits
- Implemented in JAX (and using Equinox for neural network utilities), supports TPUs out of the box
- Minimal, can run on free-tier Google Colaboratory as well

### Convergences
- Training objective: We use the original `l_simple` idea introduced in the paper, which just performs a MSE over the original Gaussian noise added v/s the noise predicted by the model for subsequent denoising.
- Beta sampling: The beta sampling schedule is the exact same used as the paper.

### Divergences
- UNet size: The model is much smaller ("nano") compared to the original implementation.
- Exponential Moving Average: Training with an EMA model helps stabilize training as seen in future works downstream of DDPMs.

## File Structure
- `model.py`: This file houses the raw implementation of the model.
- `train.py`: This file houses the training code along with logging utilities. This file will allow the user to train on the CIFAR-10 dataset.

## WandB Dashboard

Find the WandB dashboard [here](https://wandb.ai/suvadityamuk/nano-ddpm)

## Citation
```bibtex
@misc{Mukherjee_2025, 
    title={nanoJaxDDPM: A minimal JAX implementation of DDPM}, 
    url={https://github.com/suvadityamuk/nanoJaxDDPM}, 
    website={GitHub}, 
    author={Mukherjee, Suvaditya}, 
    year={2025}, 
    month={Jun}
}
```

## Acknowledgements
A huge thank-you to the Google ML Developer Programs team for supporting this with compute resources (Colab Compute Units + GCP Credits)!