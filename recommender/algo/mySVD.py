from typing import List, Tuple

import pandas as pd
from surprise import SVD
from surprise.dataset import DatasetAutoFolds

from ..RecommenderBase import RecommenderBase


class mySVD(RecommenderBase):
    def __init__(self, dataset: DatasetAutoFolds):
        self.algo = SVD()
        self.dataset = dataset
        self.df = dataset.df

    def estimate(self, u, i):
        return self.algo.estimate(u, i)

    def getRecommendations(self, ratings: List[Tuple[int, int]]) -> List[int]:
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
            (i, self.estimate(numberOfUsers, i))
            for i in trainset.all_items()
            if i not in ratedItems
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return map(candidates, lambda x: x[0])
