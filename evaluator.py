from surprise import SVD
from surprise import BaselineOnly
from surprise import Dataset
from surprise import NormalPredictor
from surprise.model_selection import cross_validate
from surprise import Dataset, Reader
from DataLoader import DataLoader
from RBMAlgorithm import RBMAlgorithm
from VAEAlgorithm import VAEAlgorithm

# from AEAlgorithm import AEAlgorithm

# from models import RBM
reader = Reader(line_format="user item rating", sep="\t")
dataset = Dataset(reader=reader)
yd = dataset.load_from_file("fullSet-5.txt", reader=reader)

data = Dataset.load_builtin("ml-100k")
# data = yd


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


vae = VAEAlgorithm(
    batchSize=128, latentDim=300, dropout=0.5, epochs=10, learningRate=1e-3
)
# ae = AEAlgorithm(batchSize=32, latentDim=200, dropout=0, epochs=20, learningRate=1e-4)


folds = 2
cross_validate(
    vae, data, measures=["RMSE", "MAE"], cv=folds, verbose=True, n_jobs=folds
)
cross_validate(bl, data, measures=["RMSE", "MAE"], cv=folds, verbose=True, n_jobs=folds)

cross_validate(np, data, measures=["RMSE", "MAE"], cv=folds, verbose=True, n_jobs=folds)


# cross_validate(ae, data, measures=["RMSE", "MAE"], cv=folds, verbose=True, n_jobs=folds)

# cross_validate(
#     rbm, data, measures=["RMSE", "MAE"], cv=folds, verbose=True, n_jobs=folds
# )
