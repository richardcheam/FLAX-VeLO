import sys 
sys.path.append('../')

from evaluate import evaluate
from utils import *
import argparse
import optuna
from dataloader import TFDSDataLoader

def parse_args():
    p = argparse.ArgumentParser("")
    p.add_argument("--model", default="resnet1", help="resnet1 | resnet18 | resnet34")
    p.add_argument("--dataset", default="mnist", help="mnist | cifar100 | kmnist | fashion-mnist")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--best_ckpt", type=str, help="best trial name")
    return p.parse_args()

args = parse_args()
DATASET = args.dataset
MODEL = args.model
BATCH_SIZE = args.batch_size
BEST_CKPT = args.best_ckpt

#db_url = f"sqlite:///study/{MODEL}_{DATASET}.db"
#study  = optuna.load_study(study_name=f"{MODEL}_{DATASET}", storage=db_url)
#l2reg = study.best_params['l2reg']

test_loader = TFDSDataLoader(dataset=DATASET, split="test", is_training=False, batch_size=BATCH_SIZE)
ds_info = test_loader.get_info()
num_classes = ds_info.features["label"].num_classes
iter_per_epoch_test = ds_info.splits["test"].num_examples // BATCH_SIZE


net = build_net(MODEL, num_classes=num_classes)

print("\nFinal model")
state = restore_checkpoint(f"checkpoints/{BEST_CKPT}", net)
_, _ = evaluate(state, iter(test_loader))

#print("EMA")
#state = restore_checkpoint(f"checkpoints/ema_{BEST_CKPT}", net)
#_, _ = evaluate(state, iter(test_loader), iter_per_epoch_test)

#print("SWA")
#state = restore_checkpoint(f"checkpoints/swa_{BEST_CKPT}", net)
#_, _ = evaluate(state, iter(test_loader), iter_per_epoch_test)
