from recommender.algo.RBMAlgorithm import RBMAlgorithm
from data.dataset import MyDataset

dataset = MyDataset()
dataset.trainTestValidationSplit()
print("Succesfully read data!")

rbm = RBMAlgorithm(verbose=True, device="cuda")

rbm.fit(dataset)
