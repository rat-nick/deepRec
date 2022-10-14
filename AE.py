import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim=10,
        dropout=0.0,
    ):
        super(AE, self).__init__()
        # encoder
        self.dense1 = nn.Linear(input_dim, 1000)
        self.dense2 = nn.Linear(1000, 500)
        self.dense3 = nn.Linear(500, 200)

        # variational part used to sample latent representation
        self.latent1 = nn.Linear(200, latent_dim)
        self.latent2 = nn.Linear(latent_dim, 200)

        # decoder
        self.dense4 = nn.Linear(200, 500)
        self.dense5 = nn.Linear(500, 1000)
        self.dense6 = nn.Linear(1000, input_dim)

        self.training = True
        self.dropout = dropout

    def encoder(self, x):
        x = torch.sigmoid(self.dense1(x))
        x = torch.dropout(x, self.dropout, self.training)
        x = torch.sigmoid(self.dense2(x))
        x = torch.dropout(x, self.dropout, self.training)
        x = torch.sigmoid(self.dense3(x))
        x = torch.dropout(x, self.dropout, self.training)
        return self.latent1(x)

    def decoder(self, z):
        z = torch.sigmoid(self.latent2(z))
        z = torch.dropout(z, self.dropout, self.training)
        z = torch.sigmoid(self.dense4(z))
        z = torch.dropout(z, self.dropout, self.training)
        z = torch.sigmoid(self.dense5(z))
        z = torch.dropout(z, self.dropout, self.training)
        z = self.dense6(z)
        # TODO: might need to change output
        return torch.sigmoid(z)

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps.mul(sigma).add_(mu)

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x


if __name__ == "__main__":
    vae = AE(10)
    x = torch.zeros(10)
    output = vae.forward(x)[0]
    print(output)
