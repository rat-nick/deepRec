from typing import List, Tuple
import numpy as np
import torch
from surprise import PredictionImpossible

from ..model import VAE

from DataLoader import DataLoader
from ..RecommenderBase import RecommenderBase
from data.dataset import MyDataset


class VAEAlgorithm(RecommenderBase):
    def __init__(
        self,
        epochs=50,
        batchSize=1,
        latentDim=100,
        dropout=0.0,
        learningRate=0.001,
        optimizer=torch.optim.Adam,
    ):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.batchSize = batchSize
        self.latentDim = latentDim
        self.learningRate = learningRate
        self.optimizer = optimizer

    def fit(self, data: MyDataset):
        # AlgoBase.fit(self, trainset)

        self.dataset = data
        self.model = VAE(self.dataset.nItems, self.latentDim, self.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)

        for epoch in range(1, self.epochs + 1):
            losses = 0
            print(f"Epoch{epoch} :", end="")
            for minibatch in self.dataset.batches(
                self.dataset.trainData, self.batchSize
            ):
                loss = self.trainIter(self.optimizer, minibatch, epoch)

            losses += loss.item()
            print(losses / self.train.shape[0])

        self.model.training = False

        self.predictions = np.ndarray(self.train.shape)

        for u in self.trainset.all_users():
            rec, _, _, _ = self.model(self.train[u])
            self.predictions[u] = rec.detach().numpy() * 5

    def trainIter(self, optimizer, batch, epoch):

        # TODO: implement annealing

        rec, _, mu, logvar = self.model(batch)
        loss = elbo(rec, batch, mu, logvar)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        return self.predictions[u][i]

    def getRecommendations(self, ratings: List[Tuple[int, int]]) -> List[int]:
        # TODO: convert ratings list to tensor
        t = torch.zeros(self.dataset.nItems)
        for id, rating in ratings:
            t[id] = [1 if rating > 7 else 0]
        # TODO: feed input tensor to model and get output
        out, _, _, _ = self.model.forward(t)
        # TODO: convert output tensor to list and sort
        out = list(enumerate(out))
        out.sort(key=lambda x: x[1], reverse=True)
        print(out)
        return out


def elbo(x_hat, x, mu, logvar, anneal=1.0):
    bce = torch.nn.CrossEntropyLoss()(x_hat, x)
    # mse = torch.nn.MSELoss()(x_hat, x)
    kld = -5e-1 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    return torch.sum(bce + anneal * kld)


if __name__ == "__main__":
    vae = VAEAlgorithm(10, 128, dropout=0.2, latentDim=100, learningRate=0.05)
    dataset = MyDataset(
        data_dir="ml-10m",
        items_path="movies.dat",
        ratings_path="ratings.dat",
        ratings_sep="::",
        items_sep="::",
    )
    vae.fit(dataset)
