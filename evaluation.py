from RBMAlgorithm import RBMAlgorithm
from surprise import Reader, Dataset
from surprise import BaselineOnly
import pandas as pd
from Evaluator import Evaluator

rbm = RBMAlgorithm(10, verbose=True, early_stopping=True)
reader = Reader(line_format="user item rating")

train = Dataset.load_from_file(
    "train-ratings-5.txt", reader=reader
).build_full_trainset()
test = Dataset.load_from_file("test-ratings-5.txt", reader=reader).build_full_trainset()

fullset = (
    Dataset.load_from_file("fullSet-5.txt", reader=reader)
    .build_full_trainset()
    .all_ratings()
)

print(fullset)
rbm.fit(trainset=train)

eval = Evaluator(RBMAlgorithm, test)
print(eval.evaluate(k=10))
print(eval.evaluate(k=20))
print(eval.evaluate(k=50))


baseline = BaselineOnly()
baseline.fit(trainset=train)
