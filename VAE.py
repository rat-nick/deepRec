from pickle import TRUE
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
        self.dense4 = nn.Linear(latent_dim, 500)
        self.dense5 = nn.Linear(500, 1000)
        self.dense6 = nn.Linear(1000, input_dim)

        self.training = True

    def encoder(self, x):
        x = F.dropout(x, 0.5, self.training)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.mu(x), self.sigma(x)

    def decoder(self, z):
        z = F.tanh(self.fc4(z))
        z = F.tanh(self.fc5(z))
        z = self.fc6(z)
        return F.log_softmax(z)

    def reparametrize(self, mu, sigma):
        eps = torch.randn_like(mu)
        return eps.mul(sigma).add_(mu)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparametrize(mu, sigma)
        x = self.decoder(z)
        return x, mu, sigma


if __name__ == "__main__":
    vae = VAE(10)
    x = torch.zeros(10)
    print(vae.forward(x)[0])
