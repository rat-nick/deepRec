from surprise import Dataset, Trainset
from surprise.dataset import DatasetAutoFolds
import torch
import pandas as pd


class DataLoader:
    def __init__(self, dataset: DatasetAutoFolds):
        self.dataset = dataset
        self.trainset = dataset.build_full_trainset()

    @classmethod
    def ratingsToSparseTensor(self, trainset: Trainset) -> torch.Tensor:
        """
        Returns a binary tensor representation of the user ratings

        shape : (n_users, n_items, ratingScale)
        """

        ratingScale = trainset.rating_scale[1] - trainset.rating_scale[0] + 1

        t = torch.zeros(trainset.n_users, trainset.n_items, ratingScale)

        if torch.cuda.is_available():
            t = t.to(device="cuda")
        for u, i, r in trainset.all_ratings():
            t[int(u)][int(i)][int(r) - 1] = 1.0

        return t

    @classmethod
    def normalizedRatingsToTensor(self, trainset: Trainset) -> torch.Tensor:
        """
        Returns a normalized tensor representation of the user ratings

        shape : (n_users, n_items)
        """
        ratingScale = trainset.rating_scale[1] - trainset.rating_scale[0] + 1

        t = torch.zeros(trainset.n_users, trainset.n_items)
        if torch.cuda.is_available():
            t = t.to(device="cuda")
        for u, i, r in trainset.all_ratings():
            t[int(u)][int(i)] = int(r) / ratingScale

        return t

    @classmethod
    def implicitRatingsToTensor(self, trainset: Trainset, cutoff=4) -> torch.Tensor:
        t = torch.zeros(trainset.n_users, trainset.n_items)
        if torch.cuda.is_available():
            t = t.to(device="cuda")
        for u, i, r in trainset.all_ratings():
            if r < cutoff:
                continue
            t[int(u)][int(i)] = 1.0
        return t


if __name__ == "__main__":
    dataset = Dataset.load_builtin("ml-100k")
    trainset = dataset.build_full_trainset()
    dl = DataLoader(dataset)
    print(DataLoader.normalizedRatingsToTensor(trainset).shape)
    print(DataLoader.ratingsToSparseTensor(trainset).shape)
