from typing import List

from surprise.dataset import Trainset

from RecommenderBase import RecommenderBase


class RecommenderEngine:
    def __init__(self, algo: RecommenderBase, trainset: Trainset):
        self.algo = algo
        self.trainset = trainset
        self.algo.fit(trainset)

    def getRecommendations(self, ratings: List[int, int]) -> List[int]:
        # convert from rawID to innerID
        ratings = [(self.trainset.to_inner_iid(id), rating) for id, rating in ratings]
        recs = self.algo.getRecommendations(ratings)
        # convert from innerID to rawID
        return [self.trainset.to_raw_iid(id) for id in recs]
