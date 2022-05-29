from surprise import SVD
from surprise import BaselineOnly
from surprise import Dataset
from surprise import NormalPredictor
from surprise.model_selection import cross_validate

from RBMAlgorithm import RBMAlgorithm


# from models import RBM

data = Dataset.load_builtin("ml-100k")
svd = SVD()
baseline = BaselineOnly()
np = NormalPredictor()
rbm = RBMAlgorithm(
    early_stopping=True,
    patience=5,
    n_hidden=100,
    learning_rate=0.1,
    # batch_size=1,
    # momentum=0.5,
    # verbose=True,
    max_epoch=20,
    l1=0.001,
    l2=0.01,
)

cross_validate(rbm, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
# cross_validate(rbm2, data, measures=["RMSE", "MAE"], cv=3, verbose=True)
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
cross_validate(baseline, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
cross_validate(np, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

# rbm = RBM()
