from collections import defaultdict
from . import dataset
from . import model, optimizer, params
import argparse
import torch
from torch.utils.data import DataLoader
from utils.tensors import onehot_to_ratings, onehot_to_ranking
import torchmetrics.functional as tm
from utils.tensors import split, leave_one_out, ohwmv
from .dataset import Dataset
import pandas as pd

hyperParams = params.HyperParams(
    batch_size=8, lr=1e-3, early_stopping=True, max_epochs=200, patience=10
)

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--ratings-path", type=str)
parser.add_argument("--result-path", type=str)
parser.add_argument("--user-threshold", type=int, default=0)

args = parser.parse_args()

device = (
    torch.device("cuda")
    if args.cuda and torch.cuda.is_available()
    else torch.device("cpu")
)

dataset = Dataset(args.ratings_path, ut=args.user_threshold)
metrics = defaultdict(list)
for train, valid, test in dataset.userKFold(5, kind="3-way"):
    trainset = train
    validset = valid
    testset = test
    # lootestset = dataset.LeaveOneOutSet(f"data/folds/{i}/test.csv", device)

    rbm = model.Model((dataset.n_items, 5), (100,), device="cuda")
    rbm.train()

    opt = optimizer.Optimizer(
        hyperParams,
        rbm,
        trainset,
        validset,
        True,
    )
    opt.t = 1
    opt.fit()
    # load best model
    rbm.load("rbm/rbm.pt")

    loader = DataLoader(testset, batch_size=1, shuffle=False)
    opt.leave_one_out_evaluation(metrics, 0, loader)
    opt.ranking_evaluation(metrics, 0, loader)
    opt.reconstruction_validation(metrics, 0, loader)

    pd.DataFrame.from_dict(metrics).to_csv(args.result_path)
