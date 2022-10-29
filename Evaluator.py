from recommender.RecommenderEngine import RecommenderEngine
from data.dataset import MyDataset
from metrics import *
from data.EvaluationData import EvaluationData


class Evaluator:
    def __init__(self, rs: RecommenderEngine, dataset: MyDataset):
        self.rs = rs
        self.dataset = dataset
        self.evalData = EvaluationData(dataset)
        self.metrics = RecommenderMetrics()

    def evaluate(self, k=50):
        recall = 0
        precision = 0
        users = self.dataset.testUsers
        n = len(users)
        for u in users:
            # u = self.dataset.raw2InnerUser(u)
            recs = self.rs.recommendationsForUser(u)
            ratings = self.dataset.getInnerUserRatings(u)
            ratings = list(ratings.itertuples(index=False, name=None))
            ratings = [(x, y) for _, x, y in ratings]
            recall += self.metrics.RecallAtK(k, recs, ratings)
            precision += self.metrics.PrecissionAtK(k, recs, ratings)

        return {"recall": recall / n, "precision": precision / n}
