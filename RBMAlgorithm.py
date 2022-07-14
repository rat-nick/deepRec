from surprise import AlgoBase, PredictionImpossible, Reader, Trainset
from surprise.dataset import Dataset

from sklearn.model_selection import train_test_split
from RBM import RBM
from utils.data import ratingsToTensor
from utils.tensors import onehot_to_ratings, softmax_to_rating


class RBMAlgorithm(AlgoBase):
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
    ):

        self.split_ratio = split_ratio
        self.use_softmax = use_softmax
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
        AlgoBase.__init__(self)

    def fit(self, trainset: Trainset):
        AlgoBase.fit(self, trainset)
        self.trainset = trainset
        self.ratings = ratingsToTensor(trainset)

        train, validation = train_test_split(
            self.ratings,
            train_size=self.split_ratio,
            random_state=42,
        )

        self.model.n_visible = self.ratings.shape[1] * self.ratings.shape[2]
        self.model.fit(train, validation)

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


if __name__ == "__main__":
    Udata = Dataset.load_builtin("ml-1m")
    fpath = "fullSet-5.txt"
    # Udata = Dataset.load_from_file(fpath, reader=Reader(line_format="user item rating"))
    Idata = Dataset.load_from_file(fpath, reader=Reader(line_format="item user rating"))
    # items = data.build_full_trainset().n_items

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
    Ualgo.fit(Idata.build_full_trainset())

    # Ialgo = RBMAlgorithm(
    #     verbose=True,
    #     max_epoch=40,
    #     n_hidden=100,
    #     learning_rate=0.001,
    #     l1=0.001,
    #     l2=0.001,
    #     batch_size=5,
    #     momentum=0.3,
    # )
    # Ialgo.fit(Idata.build_full_trainset())
    print(Ualgo.model.rmse)
    while True:
        u, i = map(int, input().split())
        Upred = Ualgo.predict(str(u), str(i))
        # Ipred = Ialgo.predict(str(i), str(u))
        print(Upred, sep="\n")
