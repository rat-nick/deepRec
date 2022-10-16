import math
from typing import Tuple

import matplotlib.pyplot as plt
import torch

from utils.tensors import *


class RBM:
    """
    Class representing a Restricted Bolzmann Machine

    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        device: str = "cpu",
        learning_rate: float = 0.001,
        l1=0.0,
        l2=0.0,
        momentum=0.0,
        batch_size=1,
        early_stopping=False,
        patience=5,
        delta=0.005,
        max_epoch=20,
        verbose=False,
    ) -> None:
        """
        Instantiates a Restricted Bolzmann Machine

        Parameters
        ----------
        n_visible : int
            number of visible units
        n_hidden : int
            number of hidden units
        device : str, optional
            device to be used when instantiating tensors, by default "cpu"
        learning_rate : float, optional
            learining rate to be used when performing fitting, by default 0.001
        l1 : float, optional
            l1 coefficient for regularization, by default 0.0
        l2 : float, optional
            l2 coefficient for regularization, by default 0.0
        momentum : float, optional
            momentum coefficient, by default 0.0
        batch_size : int, optional
            number of training cases to be processed in one batch, by default 1
        early_stopping : bool, optional
            should the algorithm use early stopping while fitting, by default False
        patience : int, optional
            number of epochs to tolerate if the model doesn't perform better, by default 5
        delta : float, optional
            the minimal value that is considered an improvement, by default 0.005
        max_epoch : int, optional
            the maximum number of epochs the fitting will run, by default 20
        verbose : bool, optional
            should the algorithm log additional data, by default False
        """

        # hyperparameters
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.alpha = learning_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.l1 = l1
        self.l2 = l2
        self.max_epoch = max_epoch
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta = delta

        self.device = device

        self.verbose = verbose

    def __save_checkpoint(self, epoch):
        """
        Saves current parameters as best
        """
        self.best_w = self.w
        self.best_v = self.v
        self.best_h = self.h
        self._best_epoch = epoch - 1

    def __load_checkpoint(self):
        """
        Loads best parameters as current
        """
        self.w = self.best_w
        self.v = self.best_v
        self.h = self.best_h

    def forward_pass(
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
        # flatten the input tensor
        if len(v.shape) > 1:
            v = v.flatten()

        a = torch.matmul(v, self.w)

        a = self.h + a

        ph = activation(a)

        return ph, sampler(ph)

    def backward_pass(
        self,
        h: torch.Tensor,
        activation=ratings_softmax,
    ) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        h : torch.Tensor
            _description_
        activation : _type_, optional
            _description_, by default ratings_softmax

        Returns
        -------
        torch.Tensor
            _description_
        """

        a = torch.matmul(h, self.w.t())

        pv = self.v + a
        pv = activation(pv.flatten())
        return pv

    def apply_gradient(
        self, minibatch: torch.Tensor, t: int = 1, decay=lambda x: x
    ) -> None:
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """
        vb_delta = torch.zeros(self.n_visible, device=self.device)
        hb_delta = torch.zeros(self.n_hidden, device=self.device)
        w_delta = torch.zeros(self.n_visible, self.n_hidden, device=self.device)
        activations = torch.zeros(self.n_hidden, device=self.device)
        # perform multiple forward passes for each case
        # and save the deltas
        for case in minibatch:
            v0 = case

            v0, ph0, vt, pht = self.gibbs_sample(v0, t)
            activations += ph0
            # caluclate the deltas
            hb_delta += ph0 - pht
            vb_delta += v0 - vt

        w_delta = torch.outer(vb_delta, hb_delta)

        # divide learning rate by the size of the minibatch
        hb_delta /= len(minibatch)
        vb_delta /= len(minibatch)
        w_delta /= len(minibatch)

        # apply learning rate decay
        self.alpha = decay(self.learning_rate)

        # update the parameters of the model
        self.v += vb_delta * self.alpha
        self.h += hb_delta * self.alpha
        self.w += w_delta * self.alpha

        # apply momentum if applicable
        if self.momentum > 0.0:
            self.v += self.prev_vb_delta * self.momentum
            self.h += self.prev_hb_delta * self.momentum
            self.w += self.prev_w_delta * self.momentum

        # remember the deltas for next training step when using momentum
        self.prev_w_delta = w_delta
        self.prev_hb_delta = hb_delta
        self.prev_vb_delta = vb_delta

        # calculate the regularization terms
        reg_w = self.w * self.l2
        # reg_h = hb_delta * self.l1

        # apply regularization
        self.w -= reg_w * len(minibatch)
        reg_h = (activations / len(minibatch)) * self.l1
        self.h -= reg_h
        self.w -= torch.ones_like(self.w) * reg_h

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
        ph0, h0 = self.forward_pass(input)
        hk = phk = h0

        # do Gibbs sampling for t steps
        for i in range(t):
            vk = self.backward_pass(hk)
            vk[input.sum(dim=1) == 0] = input[input.sum(dim=1) == 0]
            phk, hk = self.forward_pass(vk)

        # vk = softmax_to_onehot(vk)

        input = input.flatten()
        vk = vk.flatten()
        return input, ph0, vk, phk

    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        """
        For a given v input tensor, let the RBM reconstruct it
        by performing a forward and backward pass
        :arg v: the input tensor
        """
        ph, h = self.forward_pass(v)
        ret = self.backward_pass(ph)

        return ret

    def __early_stopping(self):
        """
        Checks whether the condition for early stopping is satisfied

        Returns
        -------
        bool

        """
        if len(self._metrics["rmse"]) < self.patience:
            return False

        if self._metrics["rmse"][-1] <= self._metrics["rmse"][self._best_epoch]:
            self._current_patience = 0
            return False
        else:
            self._current_patience += 1

        return self._current_patience >= self.patience

    @property
    def hyperparameters(self):
        ret = {}
        ret["n_visible"] = self.n_visible
        ret["n_hidden"] = self.n_hidden
        ret["learning_rate"] = self.learning_rate
        ret["momentum"] = self.momentum
        ret["l1"] = self.l1
        ret["l2"] = self.l2

        return ret

    def fit(self, train, test, t=1, decay=lambda x: x):
        self._metrics = {"rmse": [], "mae": []}
        self._best_epoch = 0
        self._current_patience = 0

        self.setup_weights_and_biases()

        self.prev_w_delta = torch.zeros(
            self.n_visible, self.n_hidden, device=self.device
        )
        self.prev_vb_delta = torch.zeros(self.n_visible, device=self.device)
        self.prev_hb_delta = torch.zeros(self.n_hidden, device=self.device)

        self.best_w = self.w
        self.best_v = self.v
        self.best_h = self.h

        p = torch.sum(train, dim=0) / train.shape[0]
        p = p / (1 - p)

        p[torch.isinf(p)] = 1
        p = torch.log(p)
        p = torch.nan_to_num(p)
        self.v = p.flatten()
        self.w *= 0.01
        self.h *= torch.abs(self.h + 10) * -1
        if self.verbose:
            print("----------test------------|--------train----")
            print("Epoch\tRMSE\tMAE\tRMSE\tMAE")
        for epoch in range(1, self.max_epoch + 1):
            if self.verbose:
                print(epoch, end="\t")

            trainset = train[torch.randperm(train.shape[0])]

            for user in range(0, len(trainset), self.batch_size):

                minibatch = trainset[user : user + self.batch_size]
                self.apply_gradient(
                    minibatch=minibatch,
                    t=t,
                    decay=decay,
                )

            rmse, mae = self.__calculate_errors(test)
            self._metrics["rmse"] += [rmse]
            self._metrics["mae"] += [mae]

            if self.verbose:
                print(format(rmse, ".4f"), end="\t")
                print(format(mae, ".4f"), end="\t")
                rmse, mae = self.__calculate_errors(train)

                print(format(rmse, ".4f"), end="\t")
                print(format(mae, ".4f"), end="\t")
                print()

            if (
                len(self._metrics["rmse"]) == 1
                or self._metrics["rmse"][-1] < self._metrics["rmse"][self._best_epoch]
            ):
                self.__save_checkpoint(epoch)

            if self.early_stopping and self.__early_stopping():
                self.__load_checkpoint()
                return

        self.__load_checkpoint()

    def setup_weights_and_biases(self):
        self.w = torch.zeros(self.n_visible, self.n_hidden, device=self.device)
        self.v = torch.zeros(self.n_visible, device=self.device)
        self.h = torch.zeros(self.n_hidden, device=self.device)

    def __calculate_errors(self, dataset):
        se = 0
        ae = 0
        n = 0
        for v in dataset:
            rec = self.reconstruct(v)
            n += len(v[v.sum(dim=1) > 0])
            se += squared_error(v, rec)
            ae += absolute_error(v, rec)

        return math.sqrt(se / n), ae / n

    @property
    def rmse(self):
        return self._metrics["rmse"][self._best_epoch]


if __name__ == "__main__":
    model = RBM(100, 20)

    v = torch.randn(20, 5)
    _, h = model.forward_pass(v)
    v = model.backward_pass(h)

    batch = torch.randn(20, 20, 5)
    model.apply_gradient(batch)
