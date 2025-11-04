# Single-Objective Optimizer Benchmarks 

This repo contains the scripts used to compare Google's VeLO (Versatile Learned Optimizer) against classic optimizers used in deep learning like Adam, SGDM, and more, on image classification task across four datasets: MNIST, Kuzushiji-MNIST (KMNIST), FashionMNIST. For CIFAR10 find here.

## How to run locally?

Make sure to have virtual environment. If not, in terminal do ```pip3 install virtualenv```. Then,
```bash
git clone https://github.com/richardcheam/FLAX-VeLO.git
cd FLAX-VeLO
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

To run benchmark:
```bash
python benchmarks/benchmark.py \
  --model resnet1 \
  --dataset mnist \
  --train_batch_size 32 \
  --test_batch_size 32 \
  --epochs 100
```
Note that for each optimizer, this script runs multiple times depending on the number of seeds specified. Optimizer and its hyperparameters can be modified at variable `OPT` and `HPARAMS`. The results will be available in /results/. /results/metrics contains the results needed for plot and so on, while /results/checkpoints basically stores the training checkpoints for inference.

To get plot:
```bash
python benchmarks/plot.py \
  --model resnet1 \
  --dataset mnist \
```

To do inference for a checkpoint (a dataset and an optimizer): 
```bash
python benchmarks/inference.py \
  --model resnet18
  --dataset mnist
  --ckpt results/checkpoints/mnist
``` 

To do aggregated inference for multiple checkpoints (a dataset, an optimizer, many seeds): 
```bash
python benchmarks/agg_infer.py
  --model resnet1
  --dataset mnist
  --opt velo
```

# Hyperparameters Tuning: Is VeLO inherently better, or just better than poorly tuned baselines?

Goal: test whether tuned baselines can surpass VeLO. Scripts to run are in /hparams_search.

Example: tune Adam on MNIST with ResNet1 for 20 trials
```
python hparams_search\optuna_search.py \
  --model resnet1 \
  --dataset mnist \
  --batch_size 32 \
  --alt_opt adam \
  --epochs 5 \
  --n_trials 20
```

# Repo structure
FLAX-VeLO/
├─ benchmarks/
│  ├─ benchmark.py
│  ├─ plot.py
│  ├─ inference.py
│  └─ agg_infer.py
├─ hparams_search/
│  ├─ tune.py                # your Optuna script (example shown in README)
│  ├─ checkpoints/
│  └─ study/
├─ config/
│  ├─ hparams.py             # suggest_hparams(trial)
│  └─ optimizer.py           # build_optimizer(...)
├─ trainer/
│  └─ __init__.py            # trainer()
├─ utils/
│  └─ ...                    # split_train_val, save_checkpoint, etc.
├─ results/
│  ├─ metrics/
│  ├─ checkpoints/
│  └─ plots/
└─ requirements.txt
