from .model import Model as VAE
from .optimizer import elbo
import torch
from . import dataset
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

EPOCHS = 50

ds = dataset.Dataset("ml-100k")

print(ds.n_items)

train_size = int(0.8 * len(ds))
valid_size = int((len(ds) - train_size) / 2)
test_size = len(ds) - train_size - valid_size

trainset, validset, testset = random_split(
    ds, lengths=[train_size, valid_size, test_size]
)

train_loader = DataLoader(
    trainset, batch_size=32, shuffle=True, generator=torch.Generator(device="cuda")
)
valid_loader = DataLoader(
    validset, batch_size=32, shuffle=True, generator=torch.Generator(device="cuda")
)
test_loader = DataLoader(
    testset, batch_size=32, shuffle=True, generator=torch.Generator(device="cuda")
)

vae = VAE(ds.n_items, 200, [400, 300], [300, 400], "cuda")

opt = Adam(vae.parameters(), lr=1e-3)

for epoch in range(1, EPOCHS + 1):
    n = 0
    epoch_loss = 0
    print(f"Epoch {epoch} \t Loss is ", end="", flush=True)
    for minibatch in train_loader:
        x, mu, logvar = vae(minibatch)
        batch_loss = elbo(x, minibatch, mu, logvar)
        epoch_loss += batch_loss
        batch_loss.backward()
        opt.step()
        opt.zero_grad()
        n += 1

    print(f"{epoch_loss/n}")
