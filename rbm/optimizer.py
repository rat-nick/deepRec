import math
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from . import dataset
from utils.tensors import onehot_to_ratings
from torch.utils.data import DataLoader
from .model import Model
from .params import HyperParams
from utils.tensors import *
import torchmetrics.functional as tm


class Optimizer:
    def __init__(
        self,
        params: HyperParams,
        model: Model,
        trainset: dataset.Trainset,
        validset: dataset.Testset,
        verbose: bool = False,
        lr=1e-3,
        t=1,
    ):
        self.params = params
        self.verbose = verbose
        self.model = model
        self.trainset = trainset
        self.validset = validset
        self.lr = lr
        self.t = t
        self.epoch = 0
        self.decay = lambda x: x
        self.train_loader = DataLoader(
            trainset, batch_size=params.batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(validset, batch_size=1)
        self.patience = 0
        self.writter = SummaryWriter()

    def fit(self):
        self.patience = 0
        best = 0
        for epoch in range(self.epoch, self.params.max_epochs):

            rmse = mae = 0
            n = 0

            # training loop
            for minibatch in self.train_loader:
                minibatch = minibatch.to(torch.device("cuda"))
                _rmse, _mae = self.apply_gradient(minibatch, self.t, self.decay)
                rmse += _rmse
                mae += _mae
                n += 1

            self.writter.add_scalar("train/rmse", rmse / n, epoch)
            self.writter.add_scalar("train/mae", mae / n, epoch)

            # self.valid_loader.batch_size = len(self.valid_loader)
            for fi, ho in self.valid_loader:
                fi += ho
                fi = fi.to(torch.device("cuda"))
                _rmse, _mae = self.batch_error(fi)
                rmse += _rmse
                mae += _mae
                n += 1

            rmse /= n
            mae /= n

            self.writter.add_scalar("Validation/RMSE", rmse, epoch)
            self.writter.add_scalar("Validation/MAE", mae, epoch)

            # self.writter.add_histogram("Params/W", self.model.w, epoch)
            # self.writter.add_histogram("Params/vB", self.model.vB, epoch)
            # self.writter.add_histogram("Params/hB", self.model.hB, epoch)

            self.writter.flush()

            # perform validation

            n100 = 0
            r50 = 0
            r20 = 0
            p10 = 0
            p5 = 0
            h10 = 0
            h5 = 0
            h1 = 0
            # self.valid_loader.batch_size = 1
            for fi, ho in self.valid_loader:
                fi = fi.to(torch.device("cuda"))
                ho = ho.to(torch.device("cuda"))

                idx = fi[0].any(1).nonzero()[:, 0]

                rec = self.model.reconstruct(fi)
                rec[0][idx] = torch.zeros(5, device="cuda")
                rec = onehot_to_ranking(rec)
                ho = onehot_to_ratings(ho)

                n100 += tm.retrieval_normalized_dcg(rec, ho, k=100)
                r50 += tm.retrieval_recall(rec, ho > 3.5, k=50)
                r20 += tm.retrieval_recall(rec, ho > 3.5, k=20)
                p10 += tm.retrieval_precision(rec, ho > 3.5, k=10)
                p5 += tm.retrieval_precision(rec, ho > 3.5, k=5)
                # h10 += tm.retrieval_hit_rate(rec, ho > 3.5, k=10)
                # h5 += tm.retrieval_hit_rate(rec, ho > 3.5, k=5)
                # h1 += tm.retrieval_hit_rate(rec, ho > 3.5, k=1)

            n100 /= len(self.valid_loader)
            r50 /= len(self.valid_loader)
            r20 /= len(self.valid_loader)
            p10 /= len(self.valid_loader)
            p5 /= len(self.valid_loader)
            # h10 /= len(self.valid_loader)
            # h5 /= len(self.valid_loader)
            # h1 /= len(self.valid_loader)

            n100 = n100.item()
            r50 = r50.item()
            r20 = r20.item()
            p10 = p10.item()
            p5 = p5.item()
            # h10 = h10.item()
            # h5 = h5.item()
            # h1 = h1.item()

            print(n100, r50, r20, p10, p5)
            self.writter.add_scalar("valid/ndcg100", n100, epoch)
            self.writter.add_scalar("valid/r50", r50, epoch)
            self.writter.add_scalar("valid/r20", r20, epoch)
            self.writter.add_scalar("valid/p10", p10, epoch)
            self.writter.add_scalar("valid/p5", p5, epoch)
            # self.writter.add_scalar("valid/h10", h10, epoch)
            # self.writter.add_scalar("valid/h5", h5, epoch)
            # self.writter.add_scalar("valid/h1", h1, epoch)

            if n100 > best:
                best = n100
                self.model.save()
                self.patience = 0
            else:
                self.patience += 1

            if self.patience > self.params.patience:
                print(f"Early stopping after {epoch} epochs")
                self.epoch = epoch
                return

        self.epoch = epoch

    def calculate_errors(self, s):
        rmse = 0
        mae = 0
        n = 0

        if s == "validation":
            data = self.trainset.validationData
        elif s == "test":
            data = self.trainset.testData
        else:
            data = self.trainset.trainData

        for v in self.trainset.batches(data, self.params.batch_size):
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
