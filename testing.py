import re
from RBMAlgorithm import RBMAlgorithm
from surprise import Reader, Dataset

rbm = RBMAlgorithm(100, verbose=True)
reader = Reader(line_format="user item rating")

train = Dataset.load_from_file(
    "train-ratings-5.txt", reader=reader
).build_full_trainset()
test = Dataset.load_from_file("test-ratings-5.txt", reader=reader).build_full_trainset()

rbm.fit(train)
