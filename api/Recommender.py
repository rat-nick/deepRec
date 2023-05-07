import pandas as pd
import torch

from vae.model import Model


class Recommender:
    def __init__(
        self,
    ):
        self.df = pd.read_csv("../data/ml-1m/ratings.csv")
        self.n_items = len(self.df.item.unique())
        self.model = Model(
            self.n_items,
            latent_size=200,
            path="../models/vae.pt",
            device=torch.device("cpu"),
        )
        self.model.eval()

    def recommend(self, ids, n=100):
        t = torch.zeros(self.n_items)
        t[ids] = 1.0

        res = self.model(t) - t * 100
        res = list(res.cpu().detach().numpy())
        res = [(i, x.item()) for i, x in enumerate(res)]

        return sorted(res, key=lambda x: x[1], reverse=True)[:n]
