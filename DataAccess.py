from random import shuffle

import numpy as np
import pandas as pd
import torch
from surprise import Dataset, Reader
from surprise.dataset import DatasetAutoFolds

from recommender.utils.tensors import ratings_softmax

DATA_DIR = "./data/ml-20m"
RATINGS_FILE = "ratings.csv"
ITEMS_FILE = "movies.csv"
LINKS_FILE = "links.csv"


class DataAccess:
    def __init__(self):
        self.ratingsDF = pd.read_csv(f"{DATA_DIR}/{RATINGS_FILE}", sep=",", header=0)
        self.itemsDF = pd.read_csv(f"{DATA_DIR}/{ITEMS_FILE}", sep=",", header=0)
        linksDF = pd.read_csv(f"{DATA_DIR}/{LINKS_FILE}", sep=",", header=0)

        linksDF = linksDF.fillna(0)
        linksDF["tmdbId"] = linksDF["tmdbId"].astype("int")

        self.itemsDF = pd.concat([self.itemsDF, linksDF], axis=1, join="inner")
        self.itemsDF = self.itemsDF.loc[:, ~self.itemsDF.T.duplicated(keep="first")]

        self.ratingsDF["rating"] *= 2
        self.ratingsDF = self.ratingsDF.rename(
            columns={"userId": "user", "movieId": "item"}
        )

        self.ratingsDF.drop("timestamp", inplace=True, axis=1)
        self.ratingsDS = Dataset.load_from_df(
            df=self.ratingsDF,
            reader=Reader(
                # line_format="user item rating timestamp",
                rating_scale=(1, 10),
            ),
        )

    def batches(self, batch_size):

        trainset = self.ratingsDS.build_full_trainset()
        outer2inner = trainset.to_inner_iid
        inner2outer = trainset.to_raw_uid
        users = [u for u in trainset.all_users()]
        shuffle(users)
        t = torch.zeros((batch_size, trainset.n_items, 10))

        current = 0
        for u in users:
            userRatings = self.ratingsDF[self.ratingsDF["user"] == inner2outer(u)]
            innerItemIDs = [outer2inner(i) for i in userRatings["item"].to_numpy()]
            ratings = userRatings["rating"].values
            f = np.vectorize(lambda x: int(x) - 1)
            ratings = f(ratings)
            t[current, innerItemIDs, ratings] = 1.0
            current += 1
            if current == batch_size:
                yield t
                current = 0
        yield t
        return


class Batcher:
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        size: tuple,
    ):
        self.df = df
        self.bs = batch_size
        self.size = size

    def next(self):
        users = self.df["user"].unique()
        t = torch.zeros(self.bs + self.size)
        current = 0
        for u in users:
            userRatings = self.df[self.df["user"] == u]
            t[
                current,
                userRatings["item"].to_numpy(),
                userRatings["rating"].to_numpy() - 1,
            ] = 1.0
            current += 1
            if current == self.bs:
                yield t
                current = 0
        yield t
        return


if __name__ == "__main__":
    dataAccess = DataAccess()
    for b in dataAccess.batches(10):
        print("Batch")
