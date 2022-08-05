import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=10,
        dropout=0.0,
    ):
        super(VAE, self).__init__()
        # encoder
        self.dense1 = nn.Linear(input_dim, 1000)
        self.dense2 = nn.Linear(1000, 500)
        self.dense3 = nn.Linear(500, 200)

        # variational part used to sample latent representation
        self.mu = nn.Linear(200, latent_dim)
        self.sigma = nn.Linear(200, latent_dim)

        # decoder
        self.dense4 = nn.Linear(latent_dim, 200)
        self.dense5 = nn.Linear(200, 500)
        self.dense6 = nn.Linear(500, 1000)
        self.dense7 = nn.Linear(1000, input_dim)

        self.training = True
        self.dropout = dropout

    def encoder(self, x):
        x = torch.tanh(self.dense1(x))
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        return self.mu(x), self.sigma(x)

    def decoder(self, z):
        z = torch.tanh(self.dense4(z))
        z = torch.tanh(self.dense5(z))
        z = torch.tanh(self.dense6(z))
        z = self.dense7(z)
        # TODO: might need to change output
        return torch.sigmoid(z)

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps.mul(sigma).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x = self.decoder(z)
        return x, z, mu, logvar


if __name__ == "__main__":
    vae = VAE(10)
    x = torch.zeros(10)
    output = vae.forward(x)[0]
    print(output)
    for i in vae.parameters():
        print(i)
