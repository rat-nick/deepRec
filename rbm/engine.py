import os

import torch

from data.dataset import MyDataset as Dataset
from utils.tensors import onehot_to_ranking

from .model import Model


class Engine:
    def __init__(self, model: Model, dataset: Dataset):
        self.dataset = dataset
        self.model = model

    def recommend(self, ratings):

        t = torch.zeros((1, self.dataset.nItems, 5))

        for movie, rating in ratings:
            t[0][movie][rating - 1] = 1

        rec = self.model(t)

        rec = onehot_to_ranking(rec)

        rec = rec[0]
        for movie, _ in ratings:
            rec[movie] = 0

        rec = list(rec.detach().numpy())
        rec = [(i, x.item()) for i, x in enumerate(rec)]

        rec.sort(key=lambda x: x[1], reverse=True)
        return rec


if __name__ == "__main__":
    dataset = Dataset()
    print(os.getcwd())
    rbm = Model((dataset.nItems, 5), (100,), path="rbm/rbm.pt")
    recommender_engine = Engine(rbm, dataset)
    print(recommender_engine.recommend([(3, 1), (15, 5), (28, 2)])[:10])
