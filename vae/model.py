import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        encoder_layers=[1024, 512, 256, 128],
        decoder_layers=[128, 256, 512, 1024],
        device="cpu",
    ):
        super(Model, self).__init__()
        self.drop = nn.Dropout()
        if device == "cuda" and torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            print("Using cuda!")
        self.encoder = Encoder(input_size, latent_size, encoder_layers)
        self.decoder = Decoder(latent_size, input_size, decoder_layers)

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps.mul(sigma).add_(mu)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        mu, logvar = self.encoder(inputs)
        tensor = self.reparametrize(mu, logvar)
        tensor = self.decoder(tensor)
        return tensor, mu, logvar

    @property
    def trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, layers=[1024, 512, 512, 256]):
        super(Encoder, self).__init__()
        self.deepNN = nn.Sequential()
        i = 0
        for layer in layers:
            self.deepNN.add_module(f"ff{i}", nn.Linear(input_size, layer))
            self.deepNN.add_module(f"activation{i}", nn.Sigmoid())
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
            self.deepNN.add_module(f"activation{i}", nn.Sigmoid())
            input_size = layer
            i += 1

        self.deepNN.add_module("output", nn.Linear(layer, output_size))
        self.deepNN.add_module("final_activation", nn.Sigmoid())

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
