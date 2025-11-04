# Single-Objective Optimizer Benchmarks 

This repo contains the scripts used to compare Google's VeLO (Versatile Learned Optimizer) against classic optimizers used in deep learning like Adam, SGDM, and more, on image classification task across four datasets: MNIST, Kuzushiji-MNIST (KMNIST), FashionMNIST. For CIFAR10 find here.

## Hyperparameters Tuning: Is VeLO inherently better, or just better than poorly tuned baselines?

The code for hyperparamters tuning of baseline optimizers to see if they surpass the performance of VeLO is in /hparams_search.

## How to run locally?

Make sure to have virtual environment. If not, in terminal do ```pip3 install virtualenv```. Then,
```bash
git clone https://github.com/richardcheam/FLAX-VeLO.git
cd FLAX-VeLO
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

To run benchmark script:
```bash
python benchmarks/benchmark.py \
  --model resnet18 \
  --dataset mnist \
  --train_batch_size 32 \
  --test_batch_size 32 \
  --epochs 100
```
Note that for each optimizer, this script runs multiple times depending on the number of seeds specified. Optimizer and its hyperparameters can be modified at variable `{OPT}` and `HPARAMS`.
