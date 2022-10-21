from recommender.algo.RBMAlgorithm import RBMAlgorithm
from data.dataset import MyDataset

dataset = MyDataset()
print("Succesfully read data!")

rbm = RBMAlgorithm(verbose=True)
rbm.fit(dataset)
