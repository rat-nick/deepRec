from data.dataset import MyDataset
from data.EvaluationData import EvaluationData
from metrics import *


class Evaluator:
    def __init__(self, engine, dataset: MyDataset):
        self.engine = engine
        self.dataset = dataset
        self.evalData = EvaluationData(dataset)
        self.metrics = RecommenderMetrics()

    def evaluate(self, k=50):
        recall = 0
        precision = 0
        f1 = 0
        ndcg = 0
        users = self.dataset.testUsers
        n = len(users)
        for u in users:
            # u = self.dataset.raw2InnerUser(u)
            recs = self.engine.recommendForUser(u)
            ratings = self.dataset.getInnerUserRatings(u)
            ratings = list(ratings.itertuples(index=False, name=None))
            ratings = [(x, y) for _, x, y in ratings]
            recall += self.metrics.RecallAtK(k, recs, ratings)
            precision += self.metrics.PrecissionAtK(k, recs, ratings)
            f1 += self.metrics.F1AtK(k, recs, ratings)
            ndcg += self.metrics.NDCGAtK(k, recs, ratings)
        return {
            "recall": recall / n,
            "precision": precision / n,
            "f1": f1 / n,
            "ndcg": ndcg / n,
        }

    def evaluate_on_held_out_ratings(self, k=50):
        recall = 0
        precision = 0
        f1 = 0
        ndcg = 0
        users = self.dataset.testUsers
        n = len(users)
        for u in users:
            u = self.dataset.raw2InnerUser(u)
            ratings, held_out = self.evalData.splitUsersRatings(u)
            recs = self.engine.recommend(ratings)
            # ratings = list(ratings.itertuples(index=False, name=None))
            ratings = [(x, y) for x, y in ratings]
            recall += self.metrics.RecallAtK(k, recs, held_out)
            precision += self.metrics.PrecissionAtK(k, recs, held_out)
            f1 += self.metrics.F1AtK(k, recs, held_out)
            ndcg += self.metrics.NDCGAtK(k, recs, held_out)

        return {
            "recall": recall / n,
            "precision": precision / n,
            "f1": f1 / n,
            "ndcg": ndcg / n,
        }
