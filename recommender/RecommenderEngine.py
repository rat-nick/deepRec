from typing import List, Tuple

from surprise.dataset import Trainset

from .RecommenderBase import RecommenderBase


class RecommenderEngine:
    def __init__(self, algo: RecommenderBase, trainset: Trainset):
        self.algo = algo
        self.trainset = trainset
        # self.algo.fit(trainset)

    def getRecommendations(self, ratings: List[Tuple[int, int]], n=50) -> List[int]:
        # convert from rawID to innerID
        ratings = [(self.trainset.to_inner_iid(id), rating) for id, rating in ratings]
        recs = self.algo.getRecommendations(ratings)
        print(recs[:n])
        recs.sort(reverse=True, key=lambda x: x[1])
        recs = recs[:n]
        # convert from innerID to rawID
        res = [(self.trainset.to_raw_iid(id), rating) for id, rating in recs]

        print(res)
        return res
