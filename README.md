# Diffusion Experiments

An educational implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239),
with corresponding [blog post](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html).

Includes: 
- A toy U-Net Model, which can be fit to MNIST - `notebooks/diffusion.ipynb`
- Notebook used for blog visualisations - `notebooks/visualize_diffusion.ipynb`

This repo is a WIP. I hope to add some more substantial experiments soon.

### Requirements

Required:
- Python >= 3.7
- PyTorch >= 1.7

Recommended:
- Linux and CUDA

### Install

```shell
pip install -r requirements.txt
```

Uses the [PopGen](#https://github.com/Popgun-Labs/PopGen) framework to manage experiments.