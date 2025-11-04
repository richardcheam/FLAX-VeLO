# Single-Objective Optimizer Benchmarks 

This repo contains the scripts used to compare Google's VeLO (Versatile Learned Optimizer) against classic optimizers used in deep learning like Adam, SGDM, and more, on image classification task across four datasets: MNIST, Kuzushiji-MNIST (KMNIST), FashionMNIST. For CIFAR10 find here.

## Hyperparameters Tuning: Is VeLO inherently better, or just better than poorly tuned baselines?

The code for hyperparamters tuning of baseline optimizers to see if they surpass the performance of VeLO is in /hparams_search.

## How to run?

Make sure to have virtual environment or ```pip3 install virtualenv```

```
git clone git@github.com:google/learned_optimization.git
cd FLAX-VeLO
python3 -m venv env
source env/bin/activate
pip install -e .
```
