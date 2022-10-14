from decimal import Subnormal
from surprise import AlgoBase, PredictionImpossible, Trainset, Dataset
import torch
import numpy as np
from DataLoader import DataLoader
from AE import AE


class AEAlgorithm(AlgoBase):
    def __init__(
        self,
        epochs=50,
        batchSize=1,
        splitRatio=0.8,
        latentDim=100,
        dropout=0.0,
        learningRate=0.001,
        optimizer=torch.optim.Adam,
    ):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.batchSize = batchSize
        self.splitRatio = splitRatio
        self.latentDim = latentDim
        self.dropout = dropout
        self.learningRate = learningRate
        self.optimizer = optimizer

    def lossFunction(self, recon_x, x):
        # recon_x = torch.tensor(recon_x, dtype=torch.float32)
        # x = torch.tensor(x, dtype=torch.float32)
        # ignore movies that the user hasn't rated
        loss = torch.mean(-torch.sum(recon_x * x, 1).cpu())

        # y = torch.ones_like(x)
        # y[x == 0] = x[x == 0]
        # r = recon_x * y
        # x *= 5
        # r *= 5
        # criterion = torch.nn.MSELoss()

        # loss = criterion(x, r)

        return loss

    def fit(self, trainset: Trainset):
        AlgoBase.fit(self, trainset)
        self.trainset = trainset
        self.train = DataLoader.normalizedRatingsToTensor(trainset)
        self.ratings = self.train * 5
        self.model = AE(self.train.shape[1], self.latentDim, self.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)

        for epoch in range(1, self.epochs + 1):
            self.train = self.train[torch.randperm(self.train.shape[0])]
            count = 0
            losses = 0
            for i in range(0, self.train.shape[0], self.batchSize):
                batch = self.train[i : i + self.batchSize]
                loss = self.trainIter(self.optimizer, batch)
                losses += loss.item()
            print(losses / self.train.shape[0])

        self.model.training = False

        self.predictions = np.ndarray(self.train.shape)

        for u in self.trainset.all_users():
            rec = self.model(self.train[u])
            rec *= 5
            self.predictions[u] = rec.detach().numpy()

    def trainIter(self, optimizer, batch):
        x = self.model(batch)
        loss = self.lossFunction(x, batch)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        return self.predictions[u][i]


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    vae = AEAlgorithm(10, 32, dropout=0.5, latentDim=100, learningRate=0.01)
    trainset = Dataset.load_builtin("ml-100k").build_full_trainset()
    vae.fit(trainset)
