import torch
import torch.nn as nn


def elbo(x_hat, x, mu, logvar, anneal=1.0):
    bce = nn.CrossEntropyLoss()(x_hat, x)
    kld = -5e-1 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    return torch.sum(bce + anneal * kld)
