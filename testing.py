from recommender.algo.RBMAlgorithm import RBMAlgorithm
from data.dataset import MyDataset

dataset = MyDataset()

testCase = [
    {"id": 84, "rating": 5},
    {"id": 85, "rating": 5},
    {"id": 87, "rating": 5},
    {"id": 86, "rating": 1},
    {"id": 82, "rating": 5},
    {"id": 89, "rating": 1},
]

algo = RBMAlgorithm(dataset=dataset, model=None, model_path="dummyRBM.pt")
algo.getRecommendations(testCase)
