from math import sqrt
from typing import Tuple
import random
import torch
from sklearn.model_selection import train_test_split

mae = torch.nn.L1Loss()
mse = torch.nn.MSELoss()


def softmax_to_onehot(v):
    imax = torch.argmax(v, 1, keepdim=True)
    return torch.zeros_like(v).scatter(1, imax, 1)


def softmax_to_rating(v):
    ratings = torch.arange(1, v.shape[2] + 1).float()
    return torch.matmul(v[0], ratings)


def sm2r(v):
    ratings = torch.arange(1, v.shape[2] + 1).float()
    ratings = ratings.expand_as(v)
    ret = torch.mul(v, ratings).sum(dim=1)
    return ret


def onehot_to_ratings(v):
    # print(torch.max(v, dim=2))
    mask = v.sum(dim=2)
    return (torch.argmax(v, dim=2) + 1) * mask


def onehot_to_ranking(v):
    # print(torch.max(v, dim=2))
    return torch.argmax(v, dim=2) + 1 + torch.max(v, dim=2).values


def absolute_error(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]

    r = sm2r(r)  # .float()
    o = onehot_to_ratings(o).float()

    ret = torch.sum(torch.abs(o - r)).item()
    return ret


def squared_error(o, r):
    bkpr = r
    r = r[o.sum(dim=2) > 0]
    o = o[o.sum(dim=2) > 0]
    bkpo = o
    r = sm2r(r)  # .float()
    o = onehot_to_ratings(o).float()
    ret = torch.sum(torch.square(o - r)).item()
    return ret


def reconstruction_rmse(o, r):
    r = r[o.sum(dim=2) > 0]
    o = o[o.sum(dim=2) > 0]

    r = onehot_to_ratings(r).float()
    o = onehot_to_ratings(o).float()

    return sqrt(mse(o, r).item())


def reconstruction_mae(o, r):
    r = r[o.sum(dim=2) > 0]
    o = o[o.sum(dim=2) > 0]

    r = onehot_to_ratings(r).float()
    o = onehot_to_ratings(o).float()

    return mae(o, r).item()


def ratings_softmax(v, num_ratings=10):
    # v = v.reshape(v.shape[0], v.shape[1] // num_ratings, num_ratings)
    v = torch.softmax(v, dim=2)
    return v


def split(
    t: torch.Tensor, ratio: float = 0.2, nonzero_only: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = t.nonzero()
    train_idx, test_idx = train_test_split(
        idx, test_size=ratio, random_state=42, shuffle=True
    )
    train = torch.zeros_like(t)
    train[train_idx] = t[train_idx]
    test = torch.zeros_like(t)
    test[test_idx] = t[test_idx]

    return train, test


def leave_one_out(
    t: torch.Tensor, threshold: float = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    random.seed(a=42)
    idx = torch.where(t >= threshold)
    if len(idx[0]) == 0:
        return None, None
    lo_idx = random.choice(idx[0])
    train = t.clone()
    test = torch.zeros_like(t)
    train[lo_idx] = 0.0
    test[lo_idx] = t[lo_idx]
    return train, test


if __name__ == "__main__":
    torch.manual_seed(42)
    v1 = torch.randint(
        high=5,
        size=(20,),
    ).float()

    tr, ts = split(v1)
    assert (v1 - tr - ts).sum().item() == 0.0
    tr, ts = leave_one_out(v1)
    assert (v1 - tr - ts).sum().item() == 0.0
    print(tr, ts, sep="\n")
