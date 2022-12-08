import torch
from ..model.RBM import RBM
from ..RecommenderBase import RecommenderBase
from surprise import PredictionImpossible
from surprise.dataset import Dataset
from surprise.model_selection import ShuffleSplit

from ..utils.tensors import *
from ...data.dataset import MyDataset


class RBMAlgorithm(RecommenderBase):
    def __init__(
        self,
        dataset: MyDataset,
        model: RBM,
        model_path="",
    ):
        self.dataset = dataset

        # self.model = RBM()
        # self.model.load_model_from_file(model_path)
        if model_path != "":
            self.model = RBM(0)
            self.model.load_model_from_file(model_path)

            # print("could not load model from file!")
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
        # FIXME: implement properly
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

    def getRecommendations(self, ratings):

        t = torch.zeros((1, self.model.n_visible, 10))

        for movie, rating in ratings:
            t[0][movie][rating - 1] = 1

        rec = self.model.reconstruct(t)

        rec = onehot_to_ratings(rec)

        print(rec.shape)
        # print(rec)
        # print(ratings)
        rec = rec[0]
        for movie, _ in ratings:
            rec[movie] = 0

        # print(rec)
        rec = list(rec.detach().numpy())
        rec = [(i, x.item()) for i, x in enumerate(rec)]
        # print(rec)
        # rec.sort(key=lambda x: x[1], reverse=True)
        # print("BBBB")
        # print("RECOMMENDING:")
        # print(rec[:50])
        return rec


if __name__ == "__main__":
    pass
