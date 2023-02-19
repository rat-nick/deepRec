from surprise import AlgoBase
import random


class RandomPredictior(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):

        return random.randint(1, 5)
