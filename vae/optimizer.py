import torch
import torch.nn.functional as F


def elbo(x_hat, x, mu, logvar, anneal=1.0):
    bce = -torch.mean(torch.sum(F.log_softmax(x_hat, 1) * x, -1))
    kld = -5e-1 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

    return bce + anneal * kld
