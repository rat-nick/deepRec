from typing import List, Tuple

from surprise import AlgoBase


class RecommenderBase(AlgoBase):
    def __init__(self):
        super().__init__()

    def getRecommendations(self, ratings: List[Tuple[int, int]]) -> List[int]:
        pass
