import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import evaluation
import metrics.tensor as tm

from . import dataset
from .model import Model as VAE
from .optimizer import elbo

gen = torch.manual_seed(42)

parser = argparse.ArgumentParser(
    description="Training script for Variational Autoencoder"
)
parser.add_argument(
    "--dataset", type=str, default="ml-1m", help="Which dataset should the model fit"
)
parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--wd", type=float, default=0.00, help="weight decay coefficient")
parser.add_argument("--batch-size", type=int, default=100, help="batch size")
parser.add_argument("--epochs", type=int, default=200, help="upper epoch limit")
parser.add_argument(
    "--total-anneal-steps",
    type=int,
    default=2000,
    help="the total number of gradient updates for annealing",
)
parser.add_argument(
    "--anneal-cap", type=float, default=0.2, help="largest annealing parameter"
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


anneal_steps = args.total_anneal_steps


def validate():

    valid_loss = 0
    n = 0
    recall50 = 0
    recall20 = 0
    ndcg = 0
    with torch.no_grad():
        for minibatch in valid_loader:
            minibatch.to(device)
            vae.train()
            x, mu, logvar = vae(minibatch)
            batch_loss = elbo(x, minibatch, mu, logvar, anneal)
            valid_loss += batch_loss / valid_loader.batch_size

            t, v = e.splitRatings(minibatch[0])
            vae.eval()
            rec = vae(t)
            rec[t.nonzero()] = 0
            ndcg += tm.ndcg(rec, v, 100)
            recall50 += tm.recall(rec, v, 50)
            recall20 += tm.recall(rec, v, 20)
            n += 1

    writter.add_scalar("valid/loss", valid_loss / n, epoch)
    writter.add_scalar(f"valid/recall@{20}", recall20 / n, epoch)
    writter.add_scalar(f"valid/recall@{50}", recall50 / n, epoch)
    writter.add_scalar(f"valid/ndcg@{100}", ndcg / n, epoch)

    return ndcg


def train():
    global n_updates
    global n
    train_loss = 0
    vae.train()
    for minibatch in train_loader:
        minibatch.to(device)

        if anneal_steps > 0:
            anneal = min(args.anneal_cap, args.anneal_cap * (n_updates / anneal_steps))
        # print(f"Anneal:{anneal}")

        x, mu, logvar = vae(minibatch)
        opt.zero_grad()
        batch_loss = elbo(x, minibatch, mu, logvar, anneal)
        batch_loss.backward()
        train_loss += batch_loss
        opt.step()
        n_updates += 1
        n += 1

    writter.add_scalar("train/loss", (train_loss / args.batch_size) / n, epoch)


device = (
    torch.device("cuda")
    if torch.cuda.is_available() and args.cuda
    else torch.device("cpu")
)


ds = dataset.Dataset(device=device)

train_size = int(0.8 * len(ds))
valid_size = int((len(ds) - train_size) / 2)
test_size = len(ds) - train_size - valid_size

trainset, validset, testset = random_split(
    ds, lengths=[train_size, valid_size, test_size], generator=gen
)


train_loader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    # shuffle=True,
    # generator=torch.Generator(device=device),
)
valid_loader = DataLoader(
    validset,
    batch_size=1,
    # shuffle=True,
    # generator=torch.Generator(device=device),
)
test_loader = DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=True,
    # generator=torch.Generator(device=device),
)


vae = VAE(ds.n_items, 200, [600], [600], device=device)
print(vae.trainable_parameters)
vae.save("vae/untrainedVAE.pt")
opt = Adam(
    vae.parameters(),
    lr=args.lr,
)

decay = lambda init, rate, time: init * ((1 - rate) ** time)

n_updates = 0

writter = SummaryWriter()
e = evaluation.Evaluator(vae)


anneal = 0

best_ndcg = 0


for epoch in range(0, args.epochs):
    n = 0
    train_loss = 0
    print(f"Epoch {epoch}\t", flush=True)

    train()
    current_ndcg = validate()

    if current_ndcg > best_ndcg:
        torch.save(vae, "vae/vae.pt")
