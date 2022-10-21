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
        n_hidden: int = 100,
        learning_rate: float = 0.001,
        l1=0.0,
        l2=0.0,
        momentum=0.0,
        batch_size=1,
        early_stopping=False,
        patience=5,
        max_epoch=20,
        verbose=False,
        split_ratio: float = 0.9,
        use_softmax=True,
        model_from_file=False,
        model_fpath="",
        data_access=MyDataset,
    ):
        self.split_ratio = split_ratio
        self.use_softmax = use_softmax
        self.data_access = data_access()
        self.model = RBM(
            n_visible=0,
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            l1=l1,
            l2=l2,
            momentum=momentum,
            batch_size=batch_size,
            early_stopping=early_stopping,
            patience=patience,
            max_epoch=max_epoch,
            verbose=verbose,
        )
        RecommenderBase.__init__(self)

    def fit(self, data: MyDataset):
        # RecommenderBase.fit(self, trainset)
        # self.trainset = trainset
        # self.ratings = ratingsToTensor(trainset)

        self.model.n_visible = data.nItems * 10
        self.model.fit(data)

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        rec = self.model.reconstruct(self.ratings[u])
        if self.use_softmax:
            return softmax_to_rating(rec[i])
        else:
            return onehot_to_ratings(rec)[i].float().item() + 1

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        return super().predict(uid, iid, r_ui, clip, verbose)

    def recommendations(self, uid):
        uid = int(uid)
        x = self.ratings[int(uid)]
        recs = []
        for item in self.trainset.all_items():
            recs += [(self.trainset.to_raw_iid(item), self.estimate(uid, item))]

        recs.sort(key=lambda x: x[1], reverse=True)
        return recs

    def getRecommendations(self, ratings):
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
    Udata = Dataset.load_builtin("ml-100k")
    # print(Udata.raw_ratings)
    Ualgo = RBMAlgorithm(
        verbose=True,
        max_epoch=200,
        patience=10,
        n_hidden=100,
        learning_rate=0.001,
        l1=0.001,
        l2=0.001,
        batch_size=10,
        momentum=0.5,
        early_stopping=True,
    )
    cv = ShuffleSplit(n_splits=1, test_size=0.05)
    for train, test in cv.split(Udata):
        Ualgo.fit(train)

        # seval = Evaluator(Ualgo, test)

        print(eval.evaluate(k=10))
        print(eval.evaluate(k=20))
        print(eval.evaluate(k=50))
