import argparse
from tabulate import tabulate
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import evaluation
import metrics.tensor as tm

from . import dataset
from .model import Model as VAE
from .optimizer import elbo

PATIENCE = 10


class Trainer:
    def __init__(
        self,
        model: VAE,
        trainset: dataset.Trainset,
        validset: dataset.Testset,
        testset: dataset.Testset,
        loovalidset: dataset.LeaveOneOutSet,
        lootestset: dataset.LeaveOneOutSet,
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
        self.loovalidset = loovalidset
        self.lootestset = lootestset
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
            current_ndcg = self.test_top_n(epoch, self.validset)
            self.test_loo(epoch, self.loovalidset)
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

    def test_top_n(self, epoch, testset: dataset.Testset, benchmarking: bool = False):
        loader = DataLoader(testset, batch_size=1)

        loss = 0
        r50 = 0
        r20 = 0
        r10 = 0
        p50 = 0
        p20 = 0
        p10 = 0
        hr5 = 0
        hr10 = 0
        ndcg = 0
        n = 0
        # topN metrics
        with torch.no_grad():
            # topN metrics
            for fi, ho in loader:
                fi = fi.to(self.device)
                ho = ho.to(self.device)

                self.model.train()

                x, mu, logvar = self.model(fi)
                batch_loss = elbo(x, fi, mu, logvar, self.anneal)
                loss += batch_loss / loader.batch_size

                self.model.eval()
                rec = self.model(fi)

                rec[fi > 0] = -1.0

                rec = rec[0]
                ho = ho[0]

                ndcg += tm.ndcg(rec, ho, 100)
                r50 += tm.recall(rec, ho, 50)
                r20 += tm.recall(rec, ho, 20)
                r10 += tm.recall(rec, ho, 10)
                p50 += tm.precision(rec, ho, 50)
                p20 += tm.precision(rec, ho, 20)
                p10 += tm.precision(rec, ho, 10)

                n += 1

        loss /= n
        r50 /= n
        r20 /= n
        r10 /= n
        p50 /= n
        p20 /= n
        p10 /= n
        ndcg /= n

        if benchmarking:
            return ndcg, r50, r20, r10, p50, p20, p10

        self.writter.add_scalar("valid/loss", loss, epoch)
        self.writter.add_scalar(f"valid/ndcg{100}", ndcg, epoch)
        self.writter.add_scalar(f"valid/r{50}", r50, epoch)
        self.writter.add_scalar(f"valid/r{20}", r20, epoch)
        self.writter.add_scalar(f"valid/r{10}", r10, epoch)
        self.writter.add_scalar(f"valid/p{50}", p50, epoch)
        self.writter.add_scalar(f"valid/p{20}", p20, epoch)
        self.writter.add_scalar(f"valid/p{10}", p10, epoch)

        return ndcg

    def test_loo(
        self, epoch, testset: dataset.LeaveOneOutSet, benchmarking: bool = False
    ):
        loader = DataLoader(testset, batch_size=1)

        hr1 = 0
        hr5 = 0
        hr10 = 0
        arhr20 = 0
        n = 0
        # leave one out metrics
        with torch.no_grad():
            # topN metrics
            for fi, ho in loader:
                fi = fi.to(self.device)
                ho = ho.to(self.device)

                self.model.eval()
                rec = self.model(fi)

                rec[fi > 0] = -1.0

                rec = rec[0]
                ho = ho[0]

                hr1 += tm.hr(rec, ho, 1)
                hr5 += tm.hr(rec, ho, 5)
                hr10 += tm.hr(rec, ho, 10)
                arhr20 += tm.mrr(rec, ho)
                n += 1

        hr1 /= n
        hr5 /= n
        hr10 /= n
        arhr20 /= n

        if benchmarking:
            return hr1, hr5, hr10, arhr20
        self.writter.add_scalar(f"valid/hr1", hr1, epoch)
        self.writter.add_scalar(f"valid/hr5", hr5, epoch)
        self.writter.add_scalar(f"valid/hr10", hr10, epoch)
        self.writter.add_scalar(f"valid/arhr", arhr20, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Should we use cuda")
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )

    r50_list = []
    r20_list = []
    r10_list = []
    p50_list = []
    p20_list = []
    p10_list = []
    n100_list = []
    hr1_list = []
    hr5_list = []
    hr10_list = []
    arhr_list = []

    for i in range(1, 6):
        trainset = dataset.Trainset(f"data/folds/{i}/train.csv", device)
        validset = dataset.Testset(f"data/folds/{i}/valid.csv", device)
        testset = dataset.Testset(f"data/folds/{i}/test.csv", device)
        loovalidset = dataset.LeaveOneOutSet(f"data/folds/{i}/valid.csv", device)
        lootestset = dataset.LeaveOneOutSet(f"data/folds/{i}/test.csv", device)

        model = VAE(3416, 200, [600], [600], device)
        trainer = Trainer(
            model,
            trainset,
            validset,
            testset,
            loovalidset,
            lootestset,
            epochs=200,
            device="cuda",
            save_path=f"vae/vae{i}.pt",
        )

        trainer.fit()
        trainer.model.load(f"vae/vae{i}.pt")
        n100, r50, r20, r10, p50, p20, p10 = trainer.test_top_n(0, testset, True)
        hr1, hr5, hr10, arhr = trainer.test_loo(0, lootestset, True)

        n100_list += [n100.cpu().numpy()]
        r50_list += [r50.cpu().numpy()]
        r20_list += [r20.cpu().numpy()]
        r10_list += [r10.cpu().numpy()]
        p50_list += [p50.cpu().numpy()]
        p20_list += [p20.cpu().numpy()]
        p10_list += [p10.cpu().numpy()]
        hr1_list += [hr1.cpu().numpy()]
        hr5_list += [hr5.cpu().numpy()]
        hr10_list += [hr10.cpu().numpy()]
        arhr_list += [arhr.cpu().numpy()]

        print(
            tabulate(
                [
                    [
                        "mean",
                        np.mean(n100_list),
                        np.mean(r50_list),
                        np.mean(r20_list),
                        np.mean(r10_list),
                        np.mean(p50_list),
                        np.mean(p20_list),
                        np.mean(p10_list),
                        np.mean(hr10_list),
                        np.mean(hr5_list),
                        np.mean(hr1_list),
                        np.mean(arhr_list),
                    ],
                    [
                        "std",
                        np.std(n100_list),
                        np.std(r50_list),
                        np.std(r20_list),
                        np.std(r10_list),
                        np.std(p50_list),
                        np.std(p20_list),
                        np.std(p10_list),
                        np.std(hr10_list),
                        np.std(hr5_list),
                        np.std(hr1_list),
                        np.std(arhr_list),
                    ],
                ],
                headers=[
                    "",
                    "ndcg@100",
                    "r@50",
                    "r@20",
                    "r@10",
                    "p@50",
                    "p@20",
                    "p@10",
                    "hr@10",
                    "hr@5",
                    "hr@1",
                    "arhr",
                ],
            )
        )
