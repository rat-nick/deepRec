from typing import List

from surprise import AlgoBase


class RecommenderBase(AlgoBase):
    def getRecommendations(self, ratings: List[(int, int)]) -> List[int]:
        pass
