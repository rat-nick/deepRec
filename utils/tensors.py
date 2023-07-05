import random
from math import sqrt
from typing import Tuple

import torch
import torch.nn.functional as F
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


def ohwmv(x: torch.Tensor) -> torch.Tensor:
    """One-Hot encoding with missing ratings

    Args:
        x (torch.Tensor): the input tensor

    Returns:
        torch.Tensor: returns One-Hot encoded tensor where the missing values are zero vectors
    """
    return F.one_hot(x.to(torch.int64), num_classes=6).float()[..., 1:]


def split(
    t: torch.Tensor, ratio: float = 0.2, nonzero_only: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:

    all_but_first = lambda x: tuple([t for t in range(1, len(x.shape))])
    reduce_to_first = lambda x: x.sum(dim=all_but_first(x))

    # if len(t.shape) == 1:
    #     idx = t.nonzero()
    # else:
    #     idx = reduce_to_first(t.nonzero())
    idx = t.nonzero()
    train_idx, test_idx = train_test_split(
        idx, test_size=ratio, random_state=42, shuffle=True
    )
    train = torch.zeros_like(t)
    idx = train_idx
    train[idx[:, 0], idx[:, 1]] = t[idx[:, 0], idx[:, 1]]

    test = torch.zeros_like(t)
    idx = test_idx
    test[idx[:, 0], idx[:, 1]] = t[idx[:, 0], idx[:, 1]]
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
    x = torch.randint(0, 6, size=(5, 20))
    tr, ts = split(x)
    tr, ts = ohwmv(tr), ohwmv(ts)
    print(x)
    assert (x - tr - ts).sum().item() == 0.0
    tr, ts = leave_one_out(x)
    assert (x - tr - ts).sum().item() == 0.0
    print(tr, ts, sep="\n")
