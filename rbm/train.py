from . import dataset
from . import model, optimizer, params
import argparse
import torch
from torch.utils.data import DataLoader
from utils.tensors import onehot_to_ratings, onehot_to_ranking
import torchmetrics.functional as tm
from tabulate import tabulate
import numpy as np

hyperParams = params.HyperParams(
    batch_size=10, lr=1e-3, early_stopping=True, max_epochs=200, patience=20
)

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store_true")

args = parser.parse_args()

device = (
    torch.device("cuda")
    if args.cuda and torch.cuda.is_available()
    else torch.device("cpu")
)

n100_list = []
r50_list = []
r20_list = []
p10_list = []
p5_list = []
hr1_list = []
hr5_list = []
hr10_list = []
arhr20_list = []

# for all folds
for i in range(1, 6):

    trainset = dataset.Trainset(f"data/folds/{i}/train.csv", device)
    validset = dataset.Testset(f"data/folds/{i}/valid.csv", device)
    testset = dataset.Testset(f"data/folds/{i}/test.csv", device)
    lootestset = dataset.LeaveOneOutSet(f"data/folds/{i}/test.csv", device)

    rbm = model.Model((3416, 5), (100,), device="cuda")
    rbm.train()
    opt = optimizer.Optimizer(hyperParams, rbm, trainset, validset, True)
    opt.t = 1
    opt.fit()

    n100 = 0
    r50 = 0
    r20 = 0
    p10 = 0
    p5 = 0
    hr1 = 0
    hr5 = 0
    hr10 = 0
    arhr = 0

    rbm.eval()
    loader = DataLoader(testset, batch_size=1, shuffle=True)
    # Top-N validation
    for fi, ho in loader:
        fi = fi.to(device)
        ho = ho.to(device)

        rec = rbm(fi)
        rec = onehot_to_ranking(rec)
        nz = fi[0].nonzero()[:, 0]
        rec[0][nz] = torch.zeros_like(rec[0][0])
        ho = onehot_to_ratings(ho)

        n100 += tm.retrieval_normalized_dcg(rec, ho, 10)
        r50 += tm.retrieval_recall(rec, ho > 1, 50)
        r20 += tm.retrieval_recall(rec, ho > 1, 20)
        p10 += tm.retrieval_precision(rec, ho > 1, 10)
        p5 += tm.retrieval_precision(rec, ho > 1, 5)

    loader = DataLoader(lootestset, batch_size=1, shuffle=True)
    for fi, ho in loader:
        fi = fi.to(device)
        ho = ho.to(device)
        rec = rbm(fi)
        rec = onehot_to_ranking(rec)
        nz = fi[0].nonzero()[:, 0]
        rec[0][nz] = torch.zeros_like(rec[0][0])
        ho = onehot_to_ratings(ho)

        hr1 += tm.retrieval_hit_rate(rec, ho > 3.5, 1)
        hr5 += tm.retrieval_hit_rate(rec, ho > 3.5, 5)
        hr10 += tm.retrieval_hit_rate(rec, ho > 3.5, 10)
        arhr += tm.retrieval_reciprocal_rank(rec, ho > 3.5)

    n100 /= len(loader)
    r50 /= len(loader)
    r20 /= len(loader)
    p10 /= len(loader)
    p5 /= len(loader)
    hr1 /= len(loader)
    hr5 /= len(loader)
    hr10 /= len(loader)
    arhr /= len(loader)

    n100_list += [n100.cpu().numpy()]
    r50_list += [r50.cpu().numpy()]
    r20_list += [r20.cpu().numpy()]
    p10_list += [p10.cpu().numpy()]
    p5_list += [p5.cpu().numpy()]
    hr1_list += [hr1.cpu().numpy()]
    hr5_list += [hr5.cpu().numpy()]
    hr10_list += [hr10.cpu().numpy()]
    arhr20_list += [arhr.cpu().numpy()]

print(
    tabulate(
        [
            [
                "mean",
                np.mean(n100_list),
                np.mean(r50_list),
                np.mean(r20_list),
                np.mean(p10_list),
                np.mean(p5_list),
                np.mean(arhr20_list),
                np.mean(hr10_list),
                np.mean(hr5_list),
                np.mean(hr1_list),
            ],
            [
                "std",
                np.std(n100_list),
                np.std(r50_list),
                np.std(r20_list),
                np.std(p10_list),
                np.std(p5_list),
                np.std(arhr20_list),
                np.std(hr10_list),
                np.std(hr5_list),
                np.std(hr1_list),
            ],
        ],
        headers=[
            "",
            "ndcg@100",
            "recall@50",
            "recall@20",
            "precision@10",
            "precision5",
            "arhr@20",
            "hitrate@10",
            "hitrate@5",
            "hitrate@1",
        ],
    )
)
