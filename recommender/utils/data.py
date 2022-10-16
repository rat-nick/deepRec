from surprise import Trainset

import torch


def ratingsToTensor(trainset: Trainset) -> torch.Tensor:
    scale = trainset.rating_scale[1] - trainset.rating_scale[0] + 1
    t = torch.zeros(trainset.n_users, trainset.n_items, scale)

    for u, i, r in trainset.all_ratings():
        t[int(u)][int(i)][int(r) - 1] = 1.0
    if torch.cuda.is_available():
        t = t.to("cuda")
        print("Using CUDA!")
    return t
