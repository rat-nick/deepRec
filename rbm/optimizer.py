import math
from typing import Tuple

import torch

from data import dataset
from utils.tensors import onehot_to_ratings

from .model import Model
from .params import HyperParams


class Optimizer:
    def __init__(
        self,
        params: HyperParams,
        model: Model,
        dataset: dataset.MyDataset,
        verbose: bool = False,
        lr=1e-3,
        t=1,
    ):
        self.params = params
        self.verbose = verbose
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.t = t
        self.decay = lambda x: x

        self.patience = 0

    def fit(self):
        self.patience = 0
        loading = "-" * 20
        if self.verbose:
            print(f"#####\t{loading}\tTRAIN\t\t\t\tVALIDATION")
            print(f"Epoch\t{loading}\tRMSE\t\tMAE\t\tRMSE\t\tMAE")

        numBatches = len(self.dataset.trainUsers) / self.params.batch_size
        _5pct = numBatches / 20

        for epoch in range(self.model.current_epoch + 1, self.params.max_epochs + 1):

            if self.verbose:
                print(epoch, end="\t", flush=True)

            rmse = mae = 0
            current = 0
            for minibatch in self.dataset.batches(
                self.dataset.trainData, self.params.batch_size
            ):
                _rmse, _mae = self.apply_gradient(minibatch, self.t, self.decay)
                rmse += _rmse
                mae += _mae

                if self.verbose:
                    current += 1
                    if current >= _5pct:
                        print("#", end="", flush=True)
                        current = 0

            if self.verbose:
                print("\t", end="", flush=True)
                rmse = rmse / numBatches
                mae = mae / numBatches
                print(format(rmse, ".6f"), end="\t")
                print(format(mae, ".6f"), end="\t")

            self.model.get_buffer("train_rmse")[self.model.current_epoch] = rmse
            self.model.get_buffer("train_mae")[self.model.current_epoch] = mae

            rmse, mae = self.calculate_errors("validation")
            self.model.get_buffer("valid_rmse")[self.model.current_epoch] = rmse
            self.model.get_buffer("valid_mae")[self.model.current_epoch] = mae

            if self.verbose:
                print(format(rmse, ".6f"), end="\t")
                print(format(mae, ".6f"), end="\t")
                print()

            if (
                self.model.current_epoch < 1
                or self.model.latestRMSE == self.model.bestRMSE
            ):
                self.model.save()

            if self.params.early_stopping and self.should_stop():
                self.model.next_epoch()
                return

            self.model.next_epoch()

    def calculate_errors(self, s):
        rmse = 0
        mae = 0
        n = 0

        if s == "validation":
            data = self.dataset.validationData
        elif s == "test":
            data = self.dataset.testData
        else:
            data = self.dataset.trainData

        for v in self.dataset.batches(data, self.params.batch_size):
            _rmse, _mae = self.batch_error(v)
            mae += _mae
            rmse += _rmse
            n += 1

        return math.sqrt(rmse / n), mae / n

    def should_stop(self):
        """
        Checks whether the condition for early stopping is satisfied

        Returns
        -------
        bool

        """
        if self.model.latestRMSE > self.model.bestRMSE:
            self.patience += 1
        else:
            self.patience = 0

        return self.patience >= self.params.patience

    def batch_error(self, minibatch: torch.Tensor) -> Tuple[float, float]:
        """
        Calculates the RMSE and MAE for the given minibatch and returns them

        Parameters
        ----------
        minibatch : torch.Tensor
            The minibatch to calculate the errors on

        Returns
        -------
        Tuple[float, float]
            The RMSE and MAE on the given minibatch
        """
        rec = self.model.reconstruct(minibatch)
        n = minibatch.sum().item()

        vRating = onehot_to_ratings(minibatch)
        recRating = onehot_to_ratings(rec)

        # set the same value for missing values so they don't affect error calculation
        recRating[minibatch.sum(dim=2) == 0] = vRating[minibatch.sum(dim=2) == 0]

        se = ((recRating - vRating) * (recRating - vRating)).sum().item()
        ae = torch.abs(recRating - vRating).sum().item()

        return math.sqrt(se / n), ae / n

    def apply_gradient(
        self,
        minibatch: torch.Tensor,
        t: int = 1,
        decay=lambda x: x,
    ) -> None:
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """

        activations = torch.zeros_like(self.model.hB, device=self.model.device)
        v0 = minibatch

        v0, ph0, vt, pht = self.model.gibbs_sample(v0, t)
        activations = ph0.sum(dim=0) / len(minibatch)

        hb_delta = (ph0 - pht).sum(dim=0) / len(minibatch)
        vb_delta = (v0 - vt).sum(dim=0) / len(minibatch)

        w_delta = torch.matmul(vb_delta.unsqueeze(2), hb_delta.unsqueeze(0))

        # apply learning rate decay
        self.alpha = decay(self.lr)

        # update the parameters of the model
        self.model.vB += vb_delta * self.alpha
        self.model.hB += hb_delta * self.alpha
        self.model.w += w_delta * self.alpha

        # apply momentum if applicable
        # if self.momentum > 0.0 and hasattr(self, "prev_w_delta"):
        #     self.params.v += self.trainingParams.prev_vd * self.hyperParams.momentum
        #     self.params.h += self.trainingParams.prev_hd * self.hyperParams.momentum
        #     self.params.w += self.trainingParams.prev_wd * self.hyperParams.momentum

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
