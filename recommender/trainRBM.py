from algo.RBMAlgorithm import RBMAlgorithm
from surprise import Dataset, Reader
from surprise.dataset import DatasetAutoFolds, Trainset

from RBM import RBM

dataset = DatasetAutoFolds(
    "../data/ml-20m/ratings.csv",
    Reader(line_format="user item rating timestamp", skip_lines=1, sep=","),
)
print("Succesfully read data!")

rbm = RBMAlgorithm(verbose=True)
rbm.fit(dataset.build_full_trainset())
