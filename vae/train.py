import argparse
from collections import defaultdict
from statistics import mean

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import evaluation
import metrics.tensor as tm
from utils.tensors import leave_one_out, split

from .dataset import Dataset
from .model import Model as VAE
from .optimizer import elbo

PATIENCE = 10


def merge_dictionaries(dict1, dict2):
    merged_dict = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        if key in dict1 and key in dict2:
            merged_dict[key] = np.concatenate((dict1[key], dict2[key]))
        elif key in dict1:
            merged_dict[key] = dict1[key]
        else:
            merged_dict[key] = dict2[key]

    return merged_dict


class Trainer:
    def __init__(
        self,
        model: VAE,
        trainset: Dataset,
        validset: Dataset,
        testset: Dataset,
        epochs: int = 100,
        batch_size: int = 100,
        anneal_steps: int = 2000,
        anneal_cap: float = 0.2,
        lr: float = 1e-4,
        device: str = "cpu",
        save_path: str = "vae/vae.pt",
    ):
        self.model = model
        self.trainset = trainset
        self.validset = validset
        self.testset = testset
        self.epochs = epochs
        self.anneal_steps = anneal_steps
        self.anneal_cap = anneal_cap
        self.batch_size = batch_size
        self.lr = lr

        self.device = (
            torch.device("cuda")
            if device == "cuda" and torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.n = 0

        self.train_loader = DataLoader(self.trainset, batch_size=batch_size)
        self.valid_loader = DataLoader(self.validset, batch_size=1)
        self.test_loader = DataLoader(self.testset, batch_size=1)

        self.writter = SummaryWriter()
        self.save_path = save_path
        self.patience = 0

    def fit(self):
        self.opt = Adam(self.model.parameters(), lr=self.lr)
        best_ndcg = 0
        self.n = 0
        for epoch in range(0, self.epochs):

            train_loss = self.train()
            self.writter.add_scalar("train/loss", train_loss, epoch)
            current_ndcg = self.ranking_eval(epoch, self.validset)
            self.leave_one_out_eval(epoch, self.validset)
            if current_ndcg > best_ndcg:
                self.model.save(self.save_path)
                best_ndcg = current_ndcg
                self.patience = 0
            else:
                self.patience += 1

            if self.patience > PATIENCE:
                return

    def train(self):
        self.model.train()
        n = 0
        loss = 0
        for minibatch in self.train_loader:
            minibatch = minibatch.to(self.device)

            if self.anneal_steps > 0:
                self.anneal = min(
                    self.anneal_cap, self.anneal_cap * (self.n / self.anneal_steps)
                )

            x, mu, logvar = self.model(minibatch)
            self.opt.zero_grad()
            batch_loss = elbo(x, minibatch, mu, logvar, self.anneal)
            batch_loss.backward()
            loss += batch_loss / self.batch_size
            self.opt.step()

            n += 1
            self.n += 1

        return loss / n

    def ranking_eval(self, epoch, testset: Dataset, benchmarking: bool = False):
        loader = DataLoader(testset, batch_size=1)

        ks = [10, 20, 50, 100]
        loss = 0

        metrics = defaultdict(list)

        with torch.no_grad():
            # topN metrics
            for case in loader:
                case = case.to(device)
                fi, ho = split(case[0])

                self.model.train()

                x, mu, logvar = self.model(fi.unsqueeze(0))
                batch_loss = elbo(x, fi, mu, logvar, self.anneal)
                loss += batch_loss / loader.batch_size

                self.model.eval()
                rec = self.model(fi.unsqueeze(0))

                rec = rec[0]

                rec[fi > 0] = 0.0
                metrics[f"num_ratings1"] += [fi.count_nonzero().item()]
                metrics[f"ndcg@100"] += [tm.ndcg(rec, ho, 100).item()]
                metrics[f"ndcg@10"] += [tm.ndcg(rec, ho, 10).item()]
                for k in ks:
                    metrics[f"recall@{k}"] += [tm.recall(rec, ho, k).item()]
                    metrics[f"precision@{k}"] += [tm.precision(rec, ho, k).item()]

        if benchmarking:
            return metrics

        self.writter.add_scalar("valid/loss", loss, epoch)
        for key in metrics:
            self.writter.add_scalar(f"valid/{key}", mean(metrics[key]), epoch)

        return np.mean(metrics["ndcg@100"])

    def leave_one_out_eval(self, epoch, testset: Dataset, benchmarking: bool = False):
        loader = DataLoader(testset, batch_size=1)
        metrics = defaultdict(list)

        ks = [1, 5, 10, 20]

        # leave one out metrics
        with torch.no_grad():
            for case in loader:
                case = case.to(device)
                fi, ho = leave_one_out(t=case[0], threshold=3.5)
                if fi == None:
                    continue
                self.model.eval()
                rec = self.model(fi.unsqueeze(0))

                rec = rec[0]
                rec[fi > 0] = 0.0
                metrics[f"num_ratings2"] += [fi.count_nonzero().item()]
                metrics[f"arhr"] += [tm.mrr(rec, ho).item()]
                for k in ks:
                    metrics[f"hr@{k}"] += [tm.hr(rec, ho, k).item()]

        if benchmarking:
            return metrics

        for key in metrics:
            self.writter.add_scalar(f"valid/{key}", mean(metrics[key]), epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cuda", action="store_true", help="Should we use cuda")
    parser.add_argument("--lt", action="store_true")
    parser.add_argument("--user-threshold", type=int, default=0)
    parser.add_argument("--result-path", type=str)
    parser.add_argument("--ratings-path", type=str)
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    lt = True

    dataset = Dataset(args.ratings_path, ut=args.user_threshold)
    i = 0
    metrics = defaultdict(list)
    for train, valid, test in dataset.userKFold(5, kind="3-way"):

        model = VAE(dataset.n_items, 200, [600], [600], device)
        trainer = Trainer(
            model,
            train,
            valid,
            test,
            epochs=200,
            device="cuda",
            save_path=f"vae/vae{i}.pt",
        )

        trainer.fit()
        trainer.model.load(f"vae/vae{i}.pt")
        m1 = trainer.ranking_eval(0, test, True)
        m2 = trainer.leave_one_out_eval(0, test, True)
        metrics = merge_dictionaries(metrics, m1)
        metrics = merge_dictionaries(metrics, m2)
        pd.DataFrame.from_dict(metrics).to_csv(args.result_path)
        i += 1
