import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import evaluation
import metrics.tensor as tm

from . import dataset
from .model import Model as VAE
from .optimizer import elbo


class Trainer:
    def __init__(
        self,
        model: VAE,
        trainset: dataset.Trainset,
        validset: dataset.Testset,
        testset: dataset.Testset,
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

    def fit(self):
        self.opt = Adam(self.model.parameters(), lr=self.lr)
        best_ndcg = 0
        self.n = 0
        for epoch in range(0, self.epochs):

            train_loss = self.train()
            self.writter.add_scalar("train/loss", train_loss, epoch)
            current_ndcg = self.test(epoch, self.validset)

            if current_ndcg > best_ndcg:
                torch.save(self.model, self.save_path)

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

    def test(self, epoch, testset: dataset.Testset, benchmarking: bool = False):
        loader = DataLoader(testset, batch_size=1)

        loss = 0
        recall50 = 0
        recall20 = 0
        precision5 = 0
        precision10 = 0
        ndcg = 0
        n = 0
        with torch.no_grad():
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
                recall50 += tm.recall(rec, ho, 50)
                recall20 += tm.recall(rec, ho, 20)
                precision5 += tm.precision(rec, ho, 5)
                precision10 += tm.precision(rec, ho, 10)
                n += 1

        loss /= n
        recall50 /= n
        recall20 /= n
        precision5 /= n
        precision10 /= n
        ndcg /= n

        self.writter.add_scalar("valid/loss", loss, epoch)
        self.writter.add_scalar(f"valid/recall@{20}", recall20, epoch)
        self.writter.add_scalar(f"valid/recall@{50}", recall50, epoch)
        self.writter.add_scalar(f"valid/ndcg@{100}", ndcg, epoch)
        self.writter.add_scalar(f"valid/precision@{5}", precision5, epoch)
        self.writter.add_scalar(f"valid/precision@{10}", precision10, epoch)

        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Should we use cuda")
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    for i in range(1, 6):
        trainset = dataset.Trainset(f"data/folds/{i}/train.csv", device)
        validset = dataset.Testset(f"data/folds/{i}/valid.csv", device)
        testset = dataset.Testset(f"data/folds/{i}/test.csv", device)

        model = VAE(3416, 200, [600], [600], device)
        trainer = Trainer(
            model,
            trainset,
            validset,
            testset,
            device="cuda",
            save_path=f"vae/vae{i}.pt",
        )

        trainer.fit()
