import torch
from ..model.RBM import RBM
from ..RecommenderBase import RecommenderBase
from surprise import PredictionImpossible
from surprise.dataset import Dataset
from surprise.model_selection import ShuffleSplit

from ..utils.tensors import onehot_to_ratings, softmax_to_rating
from data.dataset import MyDataset


class RBMAlgorithm(RecommenderBase):
    def __init__(
        self,
        dataset: MyDataset,
        model: RBM,
        model_path="",
    ):
        self.dataset = dataset

        if model_path != "":
            try:
                self.model = RBM()
                self.model.load_model_from_file(model_path)
            except:
                print("could not load model from file!")
        else:
            self.model = model

        RecommenderBase.__init__(self)

    def fit(self, data: MyDataset):
        self.model.fit(data)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        ratings = self.dataset.getInnerUserRatings(u)
        rec = self.model.reconstruct(ratings)
        if self.use_softmax:
            return softmax_to_rating(rec[i])
        else:
            return onehot_to_ratings(rec)[i].float().item() + 1

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        return super().predict(uid, iid, r_ui, clip, verbose)

    def recommendationsForUser(self, user):
        # TODO: implement properly
        user = int(user)

        ratings = self.dataset.getInnerUserRatings(user)
        t = torch.zeros((self.dataset.nItems, 10))
        t[ratings["item"].to_numpy(), ratings["rating"].to_numpy() - 1] = 1.0
        y = self.model.reconstruct(t.unsqueeze(0))
        y = softmax_to_rating(y)
        y = y.detach().numpy()
        y = list(enumerate(y))

        y.sort(key=lambda x: x[1], reverse=True)
        return y

    def getRecommendationsFromRatings(self, ratings):
        # TODO: implement properly
        t = torch.zeros_like(self.trainset)
        for movie, rating in ratings:
            t[movie][rating - 1] = 1
        rec = self.model.reconstruct(t)
        rec = onehot_to_ratings(rec)
        for movie, rating in ratings:
            rec[movie] = 0
        rec = rec.detach().numpy()
        rec = [(i, x) for i, x in enumerate(rec)]
        rec.sort(key=lambda x: x[1], reverse=True)
        return rec


if __name__ == "__main__":
    pass
