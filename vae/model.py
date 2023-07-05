import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as tmf
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from utils.tensors import split
from vae.dataset import Dataset

DEFAULT_PATH = "vae/vae.pt"


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias, std=0.01)


def elbo(x_hat, x, mu, logvar, anneal=1.0):
    bce = -torch.mean(torch.sum(F.log_softmax(x_hat, 1) * x, -1))
    kld = -5e-1 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return bce + anneal * kld


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, layers=[1024, 512, 512, 256]):
        super(Encoder, self).__init__()
        self.deepNN = nn.Sequential()
        i = 0
        for layer in layers:
            self.deepNN.add_module(f"ff{i}", nn.Linear(input_size, layer))
            self.deepNN.add_module(f"activation{i}", nn.Tanh())
            input_size = layer
            i += 1

        self.mu = nn.Linear(layer, output_size)
        self.sigma = nn.Linear(layer, output_size)

    def forward(self, inputs):
        tensor = self.deepNN(inputs)
        return self.mu(tensor), self.sigma(tensor)

    @property
    def trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, layers=[1024, 512, 512, 256]):
        super(Decoder, self).__init__()
        self.deepNN = nn.Sequential()

        i = 0
        for layer in layers:
            self.deepNN.add_module(f"ff{i}", nn.Linear(input_size, layer))
            self.deepNN.add_module(f"activation{i}", nn.Tanh())
            input_size = layer
            i += 1

        self.deepNN.add_module("output", nn.Linear(layer, output_size))
        # self.deepNN.add_module("final_activation", nn.Sigmoid())

    def forward(self, inputs):
        tensor = self.deepNN(inputs)
        return tensor

    @property
    def trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


class Model(pl.LightningModule):
    def __init__(
        self,
        input_size,
        latent_size,
        encoder_layers=[600],
        decoder_layers=[600],
    ):
        super(Model, self).__init__()

        self.drop = nn.Dropout()
        self.encoder = Encoder(input_size, latent_size, encoder_layers)
        self.decoder = Decoder(latent_size, input_size, decoder_layers)

        self.apply(init_weights)

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps.mul(sigma).add_(mu)

    def forward(self, x):
        # FIXME: x should be a binary vector
        mu, _ = self.encoder(x)
        tensor = self.decoder(mu)
        return tensor

    def training_step(self, batch, batch_idx):
        batch = self.drop(batch)
        mu, logvar = self.encoder(batch)
        tensor = self.reparametrize(mu, logvar)
        tensor = self.decoder(tensor)
        # TODO: implement annealing
        loss = elbo(tensor, batch, mu, logvar)
        return loss

    def validation_step(self, batch, batch_idx):
        input, hold = split(batch)
        tensor = self.forward(input)
        tensor[input > 0] = 0.0
        # TODO: implement following validation metrics:
        # * recall and precision
        # * arhr and hr
        metrics = {}
        metrics["ndcg@100"] = tmf.retrieval_normalized_dcg(tensor, hold, 100)
        self.log(
            "ndcg@100", metrics["ndcg@100"], on_step=False, on_epoch=True, prog_bar=True
        )
        return metrics

    def test_step(self, batch, batch_idx):
        input, hold = split(batch)
        tensor = self.forward(input)
        # TODO: implement following validation metrics:
        # * recall and precision
        # * arhr and hr
        metrics = {}
        metrics["ndcg@100"] = tmf.retrieval_normalized_dcg(tensor, hold, 100)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    @property
    def trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def save(self, path=DEFAULT_PATH):
        torch.save(self.state_dict(), path)

    def load(self, path=DEFAULT_PATH):
        self.load_state_dict(torch.load(path, map_location=self.device))


# data-loaders
dataset = Dataset("data/ml-1m/ratings.csv", ut=20, rs=5)
train, valid, test = next(dataset.userKFold(5, kind="3-way"))
train = DataLoader(train, batch_size=100, num_workers=12)
valid = DataLoader(valid, batch_size=100, num_workers=12)
test = DataLoader(test, batch_size=100, num_workers=12)

# model definition
model = Model(dataset.n_items, 200)

# traning
trainer = pl.Trainer(
    max_epochs=200,
    log_every_n_steps=10,
    callbacks=[EarlyStopping(monitor="ndcg@100", mode="max", patience=20)],
)
trainer.fit(model, train, valid)
trainer.test(model, test)
