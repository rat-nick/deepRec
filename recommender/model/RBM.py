import math
from typing import Tuple

import torch

from ..utils.tensors import *
from data.dataset import MyDataset
import time
from .RBMParams import *


class RBM:
    """
    Class representing a Restricted Bolzmann Machine

    """

    def __init__(
        self,
        n_visible: int,
        ratings: int = 1,
        n_hidden: int = 10,
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
        sparsity=0.01,
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

        self.device = device
        self.verbose = verbose

        self.metrics = RBMMetrics()
        self.params = RBMParams()

        self.hyperParams = RBMHyperParams(
            batch_size=batch_size,
            early_stopping=early_stopping,
            hidden_shape=(n_hidden,),
            visible_shape=(
                n_visible,
                ratings,
            ),
            l1=l1,
            l2=l2,
            lr=learning_rate,
            max_epochs=max_epoch,
            momentum=momentum,
            patience=patience,
        )

        self.init_params()

        self.trainingParams = RBMTrainingParams(
            current_patience=0,
            prev_wd=torch.zeros_like(self.params.w),
            prev_hd=torch.zeros_like(self.params.h),
            prev_vd=torch.zeros_like(self.params.v),
            epoch=0,
        )

        self.bestParams = RBMParams(w=self.params.w, v=self.params.v, h=self.params.h)
        self.init_training_params()

    def __save_checkpoint(self, epoch):
        """
        Saves current parameters as best
        """
        self.bestParams = self.params

    def __load_checkpoint(self):
        """
        Loads best parameters as current
        """
        self.params = self.bestParams

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

        w = self.params.w
        h = self.params.h

        a = torch.mm(v.flatten(-2), w.flatten(end_dim=1))
        a = h + a
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
        w = self.params.w
        v = self.params.v

        a = torch.matmul(w, h.t())

        pv = v.unsqueeze(2) + a
        pv = activation(pv.permute(2, 0, 1))
        return pv

    def apply_gradient(
        self, minibatch: torch.Tensor, t: int = 1, decay=lambda x: x
    ) -> None:
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """

        activations = torch.zeros_like(self.params.h, device=self.device)
        v0 = minibatch

        v0, ph0, vt, pht = self.gibbs_sample(v0, t)
        activations = ph0.sum(dim=0) / len(minibatch)

        hb_delta = (ph0 - pht).sum(dim=0) / len(minibatch)
        vb_delta = (v0 - vt).sum(dim=0) / len(minibatch)

        w_delta = torch.matmul(vb_delta.unsqueeze(2), hb_delta.unsqueeze(0))

        # apply learning rate decay
        self.alpha = decay(self.hyperParams.lr)

        # update the parameters of the model
        self.params.v += vb_delta * self.alpha
        self.params.h += hb_delta * self.alpha
        self.params.w += w_delta * self.alpha

        # apply momentum if applicable
        if self.hyperParams.momentum > 0.0 and hasattr(self, "prev_w_delta"):
            self.params.v += self.trainingParams.prev_vd * self.hyperParams.momentum
            self.params.h += self.trainingParams.prev_hd * self.hyperParams.momentum
            self.params.w += self.trainingParams.prev_wd * self.hyperParams.momentum

        # remember the deltas for next training step when using momentum
        # self.prev_w_delta = w_delta
        # self.prev_hb_delta = hb_delta
        # self.prev_vb_delta = vb_delta

        # # calculate the regularization terms
        # reg_w = self.w * self.l2
        # # reg_h = hb_delta * self.l1

        # # apply regularization
        # self.w -= reg_w  # * len(minibatch)
        # if self.l1 > 0:
        #     q_new = (activations / len(minibatch)) * (
        #         1 - self.l1
        #     ) + self.l1 * self.q_prev
        #     sparsity_penalty = (
        #         -self.sparsity * torch.log(q_new)
        #         - (1 - self.sparsity)
        #         - torch.log(1 - q_new)
        #     )
        #     self.h -= sparsity_penalty
        #     self.w += torch.ones_like(self.w) * sparsity_penalty
        #     self.q_prev = q_new

        rmse, mae = self.batch_error(minibatch)

        return rmse, mae

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
        for _ in range(t):
            vk = self.backward_pass(hk)
            # vk[input.sum(dim=2) == 0] = input[input.sum(dim=2) == 0]
            phk, hk = self.forward_pass(vk)

        vk[input.sum(dim=2) == 0] = input[input.sum(dim=2) == 0]
        # vk = softmax_to_onehot(vk)

        # input = input.flatten()
        # vk = vk.flatten()
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
        if len(self.metrics.validRMSE) < self.hyperParams.patience:
            return False

        if self.metrics.validRMSE[-1] <= self.metrics.bestRMSE["value"]:
            self.trainingParams.current_patience = 0
            return False
        else:
            self.trainingParams.current_patience += 1

        return self.trainingParams.current_patience >= self.hyperParams.patience

    def load_model_from_file(self, fpath, device="cpu"):
        model_state = torch.load(fpath, map_location=torch.device(device))
        self.params = model_state["params"]
        self.trainingParams = model_state["trainingParams"]
        self.metrics = model_state["metrics"]
        self.hyperParams = model_state["hyperParams"]

    def fit(self, data: MyDataset, t=1, decay=lambda x: x):
        self.data = data
        self.__load_checkpoint()
        self.init_training_params()
        self.trainingParams.epoch = len(self.metrics.trainMAE)
        loading = "-" * 20
        if self.verbose:
            print(f"#####\t{loading}\tTRAIN\t\t\t\tVALIDATION")
            print(f"Epoch\t{loading}\tRMSE\t\tMAE\t\tRMSE\t\tMAE")

        numBatches = len(data.trainUsers) / self.hyperParams.batch_size
        _5pct = numBatches / 20

        for epoch in range(self.trainingParams.epoch, self.hyperParams.max_epochs + 1):

            if self.verbose:
                print(epoch, end="\t", flush=True)

            current = 0
            rmse = mae = n = 0

            for minibatch in data.batches(data.trainData, self.hyperParams.batch_size):
                _rmse, _mae = self.apply_gradient(
                    minibatch=minibatch,
                    t=t,
                    decay=decay,
                )
                rmse += _rmse
                mae += _mae
                n += 1

                if self.verbose:
                    current += 1
                    if current >= _5pct:
                        print("#", end="", flush=True)
                        current = 0

            rmse /= n
            mae /= n

            if self.verbose:
                print("\t", end="", flush=True)
                print(format(rmse, ".6f"), end="\t")
                print(format(mae, ".6f"), end="\t")

            self.metrics.trainRMSE += [rmse]
            self.metrics.trainMAE += [mae]

            rmse, mae = self.calculate_errors("validation")
            self.metrics.validRMSE += [rmse]
            self.metrics.validMAE += [mae]

            if self.verbose:
                print(format(rmse, ".6f"), end="\t")
                print(format(mae, ".6f"), end="\t")
                print()

            if (
                len(self.metrics.validRMSE) == 1
                or self.metrics.validRMSE[-1] < self.metrics.bestRMSE["value"]
            ):
                self.__save_checkpoint(epoch)
                self.save_model_to_file(f"rbm{time.time()}.pt")

            if self.hyperParams.early_stopping and self.__early_stopping():
                self.__load_checkpoint()
                self.save_model_to_file(f"rbm{time.time()}.pt")
                return

        self.__load_checkpoint()
        self.save_model_to_file(f"rbm{time.time()}.pt")

    def init_training_params(self):

        self.trainingParams.current_patience = 0

        self.trainingParams.prev_wd = torch.zeros_like(
            self.params.w, device=self.device
        )
        self.trainingParams.prev_vd = torch.zeros_like(
            self.params.v, device=self.device
        )
        self.trainingParams.prev_hd = torch.zeros_like(
            self.params.h, device=self.device
        )

    def save_model_to_file(self, fpath):
        model_state = {
            "params": self.bestParams,  # better to always save the best parameters
            "trainingParams": self.trainingParams,
            "metrics": self.metrics,
            "hyperParams": self.hyperParams,
        }
        torch.save(model_state, fpath)

    def init_params(self):
        hp = self.hyperParams
        w_shape = hp.visible_shape + (hp.hidden_shape)

        self.params.w = torch.randn(w_shape, device=self.device) * 1e-2
        self.params.v = torch.zeros(hp.visible_shape, device=self.device)
        self.params.h = torch.zeros(hp.hidden_shape, device=self.device)

    def calculate_errors(self, s):
        rmse = 0
        mae = 0
        n = 0

        if s == "validation":
            data = self.data.validationData
        elif s == "test":
            data = self.data.testData
        else:
            data = self.data.trainData
        n = 0
        for v in self.data.batches(data, self.hyperParams.batch_size):
            _rmse, _mae = self.batch_error(v)
            rmse += _rmse
            mae += _mae
            n += 1
        return rmse / n, mae / n

    def batch_error(self, v):
        rec = self.reconstruct(v)
        n = v.sum().item()

        vRating = onehot_to_ratings(v)
        recRating = onehot_to_ratings(rec)

        # set the same value for missing values so they don't affect error calculation
        recRating[v.sum(dim=2) == 0] = vRating[v.sum(dim=2) == 0]

        se = ((recRating - vRating) ** 2).sum().item()
        ae = (recRating - vRating).abs().sum().item()

        return sqrt(se / n), ae / n


if __name__ == "__main__":
    model = RBM(100, 20)

    v = torch.randn(20, 5)
    _, h = model.forward_pass(v)
    v = model.backward_pass(h)

    batch = torch.randn(20, 20, 5)
    model.apply_gradient(batch)
