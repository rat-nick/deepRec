from .model import Model as VAE
from .optimizer import elbo
import torch
from . import dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam


ds = dataset.Trainset("ml-1m")

EPOCHS = 1000

loader = DataLoader(
    ds, batch_size=64, shuffle=True, generator=torch.Generator(device="cuda")
)


vae = VAE(ds.n_items, 200, [400, 300, 300], [300, 300, 400], "cuda")

opt = Adam(vae.parameters(), lr=1e-3)
decay = lambda init, rate, time: init * ((1 - rate) ** time)
minibatch = next(iter(loader))
for epoch in range(1, EPOCHS + 1):
    vae.train()
    print(f"Epoch {epoch} \t Loss is ", end="", flush=True)
    x, mu, logvar = vae(minibatch)
    batch_loss = elbo(x, minibatch, mu, logvar, anneal=decay(1, 0.01, epoch))
    batch_loss.backward()
    opt.step()
    opt.zero_grad()

    vae.eval()
    batch_loss = elbo(x, minibatch, mu, logvar, anneal=decay(1, 0.01, epoch))

    print(batch_loss.item())
