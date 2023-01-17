import os

import torch

from data.dataset import MyDataset as Dataset
from utils.tensors import onehot_to_ranking, softmax_to_rating

from .model import Model


class Engine:
    def __init__(self, model: Model, dataset: Dataset):
        self.dataset = dataset
        self.model = model

    def recommend(self, ratings, evaluating=False):

        t = torch.zeros((1, self.dataset.nItems, 5))

        for movie, rating in ratings:
            t[0][movie][rating - 1] = 1

        rec = self.model(t)

        rec = softmax_to_rating(rec)

        if not evaluating:
            for movie, _ in ratings:
                rec[movie] = 0

        rec = list(rec.cpu().detach().numpy())
        rec = [(i, x.item()) for i, x in enumerate(rec)]

        rec.sort(key=lambda x: x[1], reverse=True)
        return rec

    def recommendForUser(self, user):
        df = self.dataset.innerRatingsDF

        ratings = df[df["user"] == user]
        ratings = [
            (i, r) for u, i, r in list(ratings.itertuples(index=False, name=None))
        ]
        return self.recommend(ratings, True)


if __name__ == "__main__":
    dataset = Dataset()
    print(os.getcwd())
    rbm = Model((dataset.nItems, 5), (100,), path="rbm/rbm.pt")
    recommender_engine = Engine(rbm, dataset)
    print(recommender_engine.recommend([(3, 1), (15, 5), (28, 2)])[:10])
