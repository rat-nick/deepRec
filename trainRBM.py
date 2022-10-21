from recommender.algo.RBMAlgorithm import RBMAlgorithm
from data.dataset import MyDataset

dataset = MyDataset()
dataset.trainTestValidationSplit()
print("Succesfully read data!")
bs = int(input())
rbm = RBMAlgorithm(verbose=True, device="cuda", batch_size=bs)

rbm.fit(dataset)
