import imp
from math import sqrt
import torch
from torch.nn import LogSoftmax
import matplotlib.pyplot as plt
import math

mae = torch.nn.L1Loss()
mse = torch.nn.MSELoss()


def softmax_to_onehot(v):
    imax = torch.argmax(v, 1, keepdim=True)
    return torch.zeros_like(v).scatter(1, imax, 1)


def softmax_to_rating(v):
    ratings = torch.arange(1, v.shape[0] + 1).float()
    return torch.dot(v, ratings).item()


def sm2r(v):
    ratings = torch.arange(start=1, end=6).float()
    ratings = ratings.expand_as(v)
    ret = torch.mul(v, ratings).sum(dim=1)
    return ret


def onehot_to_ratings(v):
    return torch.argmax(v, dim=1) + 1


def absolute_error(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]

    r = sm2r(r)  # .float()
    o = onehot_to_ratings(o).float()

    ret = torch.sum(torch.abs(o - r)).item()
    return ret


def squared_error(o, r):
    bkpr = r
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]
    bkpo = o
    r = sm2r(r)  # .float()
    o = onehot_to_ratings(o).float()
    ret = torch.sum(torch.square(o - r)).item()
    return ret


def reconstruction_rmse(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]

    r = onehot_to_ratings(r).float()
    o = onehot_to_ratings(o).float()

    return sqrt(mse(o, r).item())


def reconstruction_mae(o, r):
    r = r[o.sum(dim=1) > 0]
    o = o[o.sum(dim=1) > 0]

    r = onehot_to_ratings(r).float()
    o = onehot_to_ratings(o).float()

    return mae(o, r).item()


def ratings_softmax(v, num_ratings=5):
    v = v.reshape(v.shape[0] // num_ratings, num_ratings)
    v = torch.softmax(v, dim=1)
    return v


if __name__ == "__main__":
    v1 = torch.randint(high=10, size=(5,)).float()
    v2 = torch.randint(high=10, size=(5,)).float()
    v3 = sqrt(mse(v1, v2).item())
    v4 = mae(v1, v2)
    print(v1, v2, v3, v4)
