import torch

from data.dataAccess import MyDataset as Dataset

from .model import Model


class Engine:
    def __init__(self, model: Model, dataset: Dataset):
        self.dataset = dataset
        self.model = model

    def recommend(self, ratings, evaluating=False):

        t = torch.zeros(self.dataset.nItems)

        for movie, rating in ratings:
            t[movie] = 1.0 if rating > 3.5 else 0.0

        rec = self.model(t)

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
