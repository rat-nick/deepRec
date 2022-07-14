from surprise import SVD
from surprise import BaselineOnly
from surprise import Dataset
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise import Dataset, Reader
from RBMAlgorithm import RBMAlgorithm


# from models import RBM
reader = Reader(line_format="user item rating", sep="\t")
dataset = Dataset(reader=reader)
yd = dataset.load_from_file("fullSet-5.txt", reader=reader)

# data = Dataset.load_builtin("ml-1m")
data = yd


bl = BaselineOnly()
np = NormalPredictor()
rbm = RBMAlgorithm(
    early_stopping=True,
    patience=4,
    n_hidden=20,
    learning_rate=0.1,
    batch_size=200,
    momentum=0.5,
    verbose=True,
    max_epoch=100,
    l1=0.001,
    l2=0.001,
    split_ratio=0.9,
)


folds = 2
cross_validate(bl, data, measures=["RMSE", "MAE"], cv=folds, verbose=True, n_jobs=folds)
cross_validate(np, data, measures=["RMSE", "MAE"], cv=folds, verbose=True, n_jobs=folds)
cross_validate(
    rbm, data, measures=["RMSE", "MAE"], cv=folds, verbose=True, n_jobs=folds
)
