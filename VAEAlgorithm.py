from decimal import Subnormal
from surprise import AlgoBase, PredictionImpossible, Trainset, Dataset
import torch
import numpy as np
from DataLoader import DataLoader
from VAE import VAE


class VAEAlgorithm(AlgoBase):
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
        self.dropout = dropout
        self.learningRate = learningRate
        self.optimizer = optimizer

    def fit(self, trainset: Trainset):
        AlgoBase.fit(self, trainset)
        self.trainset = trainset
        self.train = DataLoader.implicitRatingsToTensor(trainset)
        self.ratings = self.train
        self.model = VAE(self.train.shape[1], self.latentDim, self.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)

        for epoch in range(1, self.epochs + 1):
            self.train = self.train[torch.randperm(self.train.shape[0])]
            count = 0
            sumLoss = 0
            losses = 0
            print(f"Epoch{epoch} :", end="")
            for i in range(0, self.train.shape[0], self.batchSize):
                batch = self.train[i : i + self.batchSize]
                loss = self.trainIter(self.optimizer, batch)
                sumLoss += loss.item()
                count += 1
                losses += loss.item()
            print(losses / self.train.shape[0])

        self.model.training = False

        self.predictions = np.ndarray(self.train.shape)

        for u in self.trainset.all_users():
            rec, _, _, _ = self.model(self.train[u])
            self.predictions[u] = rec.detach().numpy()

    def trainIter(self, optimizer, batch):
        rec, _, mu, logvar = self.model(batch)
        loss = elbo(rec, batch, mu, logvar)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
        optimizer.step()
        # optimizer.zero_grad()
        return loss

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        return self.predictions[u][i]


def elbo(x_hat, x, mu, logvar, anneal=1.0):
    bce = -torch.mean(torch.sum(torch.log_softmax(x_hat, 1) * x), -1)
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return bce + anneal * kld


if __name__ == "__main__":
    vae = VAEAlgorithm(10, 128, dropout=0.2, latentDim=100, learningRate=0.05)
    trainset = Dataset.load_builtin("ml-100k").build_full_trainset()
    vae.fit(trainset)
