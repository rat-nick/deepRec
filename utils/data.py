from audioop import mul
import torch
from surprise.dataset import DatasetAutoFolds


def ratingsToTensor(dataset) -> torch.Tensor:
    trainset = dataset
    scale = trainset.rating_scale[1] - trainset.rating_scale[0] + 1
    t = torch.zeros(trainset.n_users, trainset.n_items, scale)

    for u, i, r in trainset.all_ratings():
        t[int(u)][int(i)][int(r) - 1] = 1.0

    return t
