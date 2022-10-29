from data.EvaluationData import EvaluationData
from data.dataset import MyDataset

dataset = MyDataset(
    data_dir="ml-1m",
    ratings_path="ratings.dat",
    ratings_sep="::",
    items_path="movies.dat",
    items_sep="::",
)

from recommender.algo.RBMAlgorithm import RBMAlgorithm
from recommender.model.RBM import RBM
from Evaluator import Evaluator

model = RBM(0)
model.load_model_from_file("rbm.pt")
rbmAlgo = RBMAlgorithm(dataset=dataset, model=model)

dataset.trainTestValidationSplit()

print(rbmAlgo.recommendationsForUser(10))
