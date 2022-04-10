from random import sample
import torch


class RBM:
    def __init__(
        self, visible_units: int, hidden_units: int, learning_rate: float = 0.01
    ) -> None:
        """
        Construct the RBM model with given number of visible and hidden units

        :arg visible_units: number of visible units
        :arg hidden_units: number of hidden units
        """
        self.learning_rate = learning_rate
        self.w = torch.zeros(hidden_units, visible_units, 5)

        self.v_bias = torch.zeros(visible_units, 5)
        self.h_bias = torch.zeros(hidden_units)

    def sample_h(self, v):
        """
        Sample hidden units given that v is the visible layer
        :param v: visible layer
        """
        # print(self.w.shape)
        # print(v.shape)

        a = torch.sum(torch.matmul(self.w, v.t()), dim=[1, 2])

        activation = self.h_bias + a

        p_h_given_v = torch.sigmoid(activation)
        # print(torch.bernoulli(p_h_given_v))
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        """
        Sample visible units given that h is the hidden layer
        :param h: hidden layer
        """

        hw = torch.matmul(self.w.permute(1, 2, 0), h.t())
        # print(hw.shape)
        top = torch.exp(hw + self.v_bias.expand_as(hw))
        bottom = torch.sum(top, dim=1)
        # print(bottom.shape)
        p_v_given_h = torch.div(top.t(), bottom).t()
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, goodSample, badSample):
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """

        good_h = self.sample_h(goodSample)[1]
        bad_h = self.sample_h(badSample)[1]

        hb_delta = torch.mul(
            good_h - bad_h,
            self.learning_rate,
        )
        vb_delta = torch.mul(
            goodSample - badSample,
            self.learning_rate,
        )

        w_delta = (self.w.permute(1, 2, 0) * hb_delta).permute(2, 0, 1)

        self.v_bias += vb_delta
        self.h_bias += hb_delta
        self.w += w_delta
