from surprise import Trainset

from DataLoader import DataLoader
from metrics import *


class Evaluator:
    def __init__(self, rs, data: Trainset):
        self.rs = rs
        self.data = data
        self.metrics = RecommenderMetrics()

    def evaluate(self, k=10):
        recall = 0
        precision = 0
        users = self.data.all_users()
        n = len(users)
        for u in users:
            recs = self.rs.recommendations(u)
            ratings = DataLoader.getUserRatings(u, self.data)
            recall += self.metrics.RecallAtK(k, recs, ratings)
            precision += self.metrics.PrecissionAtK(k, recs, ratings)

        return {"recall": recall / n, "precision": precision / n}
