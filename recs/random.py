import random

from surprise import AlgoBase


class RandomPredictior(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):

        return random.uniform(1, 5)
