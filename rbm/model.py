import sys
from typing import Tuple

import torch
import torch.nn as nn

from utils import tensors

from .params import *

DEFAULT_PATH = "rbm/rbm.pt"


class Model(nn.Module):
    """
    Class representing an RBM model
    """

    def __init__(
        self,
        visible_shape: Tuple = None,
        hidden_shape: Tuple = None,
        device: str = "cpu",
        path: str = "",
    ) -> None:
        """Creates an RBM model with the given visible and hidden shapes

        Parameters
        ----------
        visible_shape : Tuple, required
            The shape of the visible layer, by default None
        hidden_shape : Tuple, required
            The shape of the hidden layer, by default None
        dev : str, optional
            The devices where the tensors will be processed, by default "cpu"
        """
        super().__init__()

        # initialize model parameters
        self.vB = nn.Parameter(
            torch.zeros(visible_shape, device=device), requires_grad=False
        )
        self.hB = nn.Parameter(
            torch.zeros(hidden_shape, device=device), requires_grad=False
        )
        self.w = nn.Parameter(
            torch.randn(visible_shape + hidden_shape, device=device) * 1e-2,
            requires_grad=False,
        )

        # inintalize buffers for metrics
        self.register_buffer("train_rmse", torch.ones(1000) * 10)
        self.register_buffer("valid_rmse", torch.ones(1000) * 10)
        self.register_buffer("train_mae", torch.ones(1000) * 10)
        self.register_buffer("valid_mae", torch.ones(1000) * 10)
        self.register_buffer("epoch", torch.zeros(1, dtype=torch.int32))

        # set the device for the tensors
        self.device = device

        if path != "":
            self.load(path)
            self.to(self.device)
            self.eval()

    @property
    def latestRMSE(self):
        return self.get_buffer("valid_rmse")[self.current_epoch].item()

    @property
    def latestMAE(self):
        return self.get_buffer("valid_mae")[self.current_epoch].item()

    @property
    def bestRMSE(self):
        return self.get_buffer("valid_rmse").min().item()

    @property
    def bestMAE(self):
        return self.get_buffer("valid_mae").min().item()

    def save(self):
        """
        Persist the model to disk on default path
        """
        torch.save(self.state_dict(), DEFAULT_PATH)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def sample_h_given_v(
        self,
        v: torch.Tensor,
        activation=torch.sigmoid,
        sampler=torch.bernoulli,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass given the input tensor `v`

        Parameters
        ----------
        `v` : torch.Tensor
            tensor representing the input values
        activation : function, optional
            the activation function to be used, by default torch.sigmoid
        sampler : function, optional
            the sampling function to be used, by default torch.bernoulli

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            the probability tensor and the sampled probability tensor of the hidden layer `h`
        """
        w = self.w
        h = self.hB

        a = torch.mm(v.flatten(-2), w.flatten(end_dim=1))
        a = h + a
        ph = activation(a)
        return ph, sampler(ph)

    def sample_v_given_h(
        self,
        h: torch.Tensor,
        activation=tensors.ratings_softmax,
    ) -> torch.Tensor:
        """
        Performs a backward pass given the hidden activations h

        Parameters
        ----------
        h : torch.Tensor
            The values of the hidden layer
        activation : function, optional
            The activation function for the visible units, by default tensors.ratings_softmax

        Returns
        -------
        torch.Tensor
            The value of the visible layer
        """
        w = self.w
        v = self.vB

        a = torch.matmul(w, h.t())

        pv = v.unsqueeze(2) + a
        pv = activation(pv.permute(2, 0, 1))
        return pv

    def gibbs_sample(
        self, input: torch.Tensor, t: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform Gibb's sampling for t steps

        Parameters
        ----------
        input : torch.Tensor
            value of the visible state tensor `v`
        t : int, optional
            number of forward-backward passes, by default 1

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The tensors needed for generating the gradient
        """
        ph0, h0 = self.sample_h_given_v(input)
        hk = phk = h0

        # do Gibbs sampling for t steps
        for _ in range(t):
            vk = self.sample_v_given_h(hk)
            # vk[input.sum(dim=2) == 0] = input[input.sum(dim=2) == 0]
            phk, hk = self.sample_h_given_v(vk)

        vk[input.sum(dim=2) == 0] = input[input.sum(dim=2) == 0]

        return input, ph0, vk, phk

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Performs reconstruction of the input tensor `v`

        Parameters
        ----------
        v : torch.Tensor
            Sparse one-hot encoded input tensor

        Returns
        -------
        torch.Tensor
            The reconstrutction of v
        """
        ph, h = self.sample_h_given_v(v)
        if self.training:
            ret = self.sample_v_given_h(h)
        else:
            ret = self.sample_v_given_h(ph)

        return ret

    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the input tensor v
        Differs from `forward` method where this method is always called in evaluation mode

        Parameters
        ----------
        v : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            Reconstruction of v
        """
        ph, _ = self.sample_h_given_v(v)
        return self.sample_v_given_h(ph)

    def next_epoch(self):
        self.get_buffer("epoch")[0] += 1

    @property
    def current_epoch(self):
        return self.get_buffer("epoch")[0].item()

    def summarize(self):
        for p in self.state_dict():
            print(p, "\t", self.state_dict()[p].size())

        total_params = sum(p.numel() for p in self.parameters())
        print("Trainable parameters: ", total_params)
