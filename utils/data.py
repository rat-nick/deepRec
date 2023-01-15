from surprise import Trainset

import torch


def ratingsToTensor(trainset: Trainset) -> torch.Tensor:
    scale = trainset.rating_scale[1] - trainset.rating_scale[0] + 1

    if torch.cuda.is_available():
        print("CUDA is available!")
        t = torch.zeros(trainset.n_users, trainset.n_items, scale, device="cuda")
    else:
        print("CUDA is NOT available!")
        t = torch.zeros(trainset.n_users, trainset.n_items, scale)

    for u, i, r in trainset.all_ratings():
        t[int(u)][int(i)][int(r) - 1] = 1.0

    return t
