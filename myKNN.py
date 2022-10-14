from typing import List

import pandas as pd
from surprise import KNNBasic
from surprise.dataset import DatasetAutoFolds

from RecommenderBase import RecommenderBase


class myKNN(RecommenderBase):
    def __init__(self, dataset: DatasetAutoFolds):
        self.algo = KNNBasic(k=9, min_k=5, verbose=True)
        self.dataset = dataset
        self.df = pd.DataFrame(
            self.dataset.__dict__["raw_ratings"],
            columns=["uid", "iid", "rui"],
        )

    def estimate(self, u, i):
        return self.algo.estimate(u, i)

    def getRecommendations(self, ratings: List[(int, int)]) -> List[int]:
        trainset = self.dataset.build_full_trainset()
        numberOfUsers = trainset.n_users
        newRatings = [(numberOfUsers, i, r) for i, r in ratings]
        newRatings = pd.DataFrame(
            newRatings,
            columns=["uid", "iid", "rui"],
        )
        df = pd.concat(self.df, newRatings)
        data = DatasetAutoFolds(df=df)
        self.algo.fit(data.build_full_trainset())
        ratedItems = map(ratings, lambda x: x[0])

        # generate candidates
        candidates = [
            (i, self.algo.estimate(numberOfUsers, i))
            for i in trainset.all_items()
            if i not in ratedItems
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return map(candidates, lambda x: x[0])
