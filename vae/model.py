import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PATH = "vae/vae.pt"


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        encoder_layers=[600],
        decoder_layers=[600],
        device=torch.device("cpu"),
        path="",
    ):
        super(Model, self).__init__()

        self.drop = nn.Dropout()
        self.encoder = Encoder(input_size, latent_size, encoder_layers)
        self.decoder = Decoder(latent_size, input_size, decoder_layers)

        if device == torch.device("cuda"):
            self.encoder.to("cuda")
            self.decoder.to("cuda")
        self.device = device
        self.apply(init_weights)

        if path != "":
            self.load(path)
            self.eval()

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        if self.training:
            return eps.mul(sigma).add_(mu)
        else:
            return mu

    def forward(self, inputs):
        inputs = inputs / 5.0
        if self.training:
            inputs = self.drop(inputs)
        mu, logvar = self.encoder(inputs)
        tensor = self.reparametrize(mu, logvar)
        tensor = self.decoder(tensor)
        if self.training:
            return tensor, mu, logvar
        else:
            return tensor

    @property
    def trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def save(self, path=DEFAULT_PATH):
        torch.save(self.state_dict(), path)

    def load(self, path=DEFAULT_PATH):
        self.load_state_dict(torch.load(path, map_location=self.device))

    @property
    def latestLoss(self):
        pass

    @property
    def bestLoss(self):
        pass


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


if __name__ == "__main__":
    vae = Model(3000, 500)

    print(vae.trainable_parameters)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias, std=0.01)
