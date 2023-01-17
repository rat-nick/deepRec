import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from . import dataset
from .model import Model as VAE
from .optimizer import elbo

EPOCHS = 50

parser = argparse.ArgumentParser(
    description="Training script for Variational Autoencoder"
)

parser.add_argument(
    "--dataset", type=str, default="ml-1m", help="Which dataset should the model fit"
)


parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--wd", type=float, default=0.00, help="weight decay coefficient")
parser.add_argument("--batch_size", type=int, default=200, help="batch size")
parser.add_argument("--epochs", type=int, default=200, help="upper epoch limit")
parser.add_argument(
    "--total_anneal_steps",
    type=int,
    default=10000,
    help="the total number of gradient updates for annealing",
)
parser.add_argument(
    "--anneal_cap", type=float, default=0.2, help="largest annealing parameter"
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--log-interval", type=int, default=100, metavar="N", help="report interval"
)
parser.add_argument(
    "--save", type=str, default="model.pt", help="path to save the final model"
)
args = parser.parse_args()

ds = dataset.Dataset(args.dataset)

print(ds.n_items)

train_size = int(0.8 * len(ds))
valid_size = int((len(ds) - train_size) / 2)
test_size = len(ds) - train_size - valid_size

trainset, validset, testset = random_split(
    ds, lengths=[train_size, valid_size, test_size]
)

train_loader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    generator=torch.Generator(device="cuda"),
)
valid_loader = DataLoader(
    validset,
    batch_size=args.batch_size,
    shuffle=True,
    generator=torch.Generator(device="cuda"),
)
test_loader = DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=True,
    generator=torch.Generator(device="cuda"),
)

vae = VAE(ds.n_items, 100, [400, 300], [300, 400], "cuda")
vae.save("vae/untrainedVAE.pt")
opt = Adam(
    vae.parameters(),
    lr=1e-3,
)

decay = lambda init, rate, time: init * ((1 - rate) ** time)

n_updates = 0
anneal_steps = args.total_anneal_steps

for epoch in range(1, EPOCHS + 1):
    n = 0
    train_loss = 0
    print(f"Epoch {epoch} \t Train: ", end="", flush=True)

    # train
    vae.train()
    for minibatch in train_loader:

        if anneal_steps > 0:
            anneal = min(args.anneal_cap, 1 * n_updates / anneal_steps)

        x, mu, logvar = vae(minibatch)
        batch_loss = elbo(x, minibatch, mu, logvar, anneal)
        train_loss += batch_loss
        batch_loss.backward()
        opt.step()
        opt.zero_grad()
        n += 1
        n_updates += 1

    print(f"{train_loss/n}\tValid: ", end="")

    # validation
    valid_loss = 0
    n = 0
    vae.eval()
    with torch.no_grad():
        for minibatch in valid_loader:
            x, mu, logvar = vae(minibatch)
            batch_loss = elbo(x, minibatch, mu, logvar, anneal)
            valid_loss += batch_loss
            n += 1

    print(f"{valid_loss/n}\t")
