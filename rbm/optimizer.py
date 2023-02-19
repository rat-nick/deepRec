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
        t=1,
        momentum=0,
        l1=0,
        l2=0,
        sparsity=0,
    ):
        self.params = params
        self.verbose = verbose
        self.model = model
        self.trainset = trainset
        self.validset = validset
        self.lr = params.lr
        self.t = t
        self.epoch = 0
        self.decay = lambda x: x
        self.train_loader = DataLoader(
            trainset, batch_size=params.batch_size, shuffle=True
        )
        self.valid_loader = DataLoader(validset, batch_size=1)
        self.patience = 0
        self.writter = SummaryWriter()

        self.momentum = momentum
        self.l1 = l1
        self.l2 = l2
        self.sparsity = sparsity

        self.prev_wd = torch.zeros_like(self.model.w)
        self.prev_hd = torch.zeros_like(self.model.hB)
        self.prev_vd = torch.zeros_like(self.model.vB)
        self.q_prev = torch.zeros_like(self.model.hB)

    def fit(self):
        self.patience = 0
        best = 0
        for epoch in range(self.epoch, self.params.max_epochs):

            rmse = mae = 0
            n = 0
            self.model.train()
            # training loop
            for minibatch in self.train_loader:
                minibatch = minibatch.to(torch.device("cuda"))
                _rmse, _mae = self.apply_gradient(minibatch, self.t, self.decay)
                rmse += _rmse
                mae += _mae
                n += 1

            self.writter.add_scalar("train/rmse", rmse / n, epoch)
            self.writter.add_scalar("train/mae", mae / n, epoch)

            self.model.eval()
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
            r10 = 0
            p50 = 0
            p20 = 0
            p10 = 0
            h10 = 0
            h5 = 0
            h1 = 0
            # self.valid_loader.batch_size = 1
            for fi, ho in self.valid_loader:
                fi = fi.to(torch.device("cuda"))
                ho = ho.to(torch.device("cuda"))

                idx = fi[0].any(1).nonzero()[:, 0]

                rec = self.model.reconstruct(fi)
                rec = onehot_to_ranking(rec)[0]
                rec[idx] = 0
                ho = onehot_to_ratings(ho)[0]

                n100 += tm.retrieval_normalized_dcg(rec, ho, k=100)
                r50 += tm.retrieval_recall(rec, ho > 3.5, k=50)
                r20 += tm.retrieval_recall(rec, ho > 3.5, k=20)
                r10 += tm.retrieval_recall(rec, ho > 3.5, k=10)
                p50 += tm.retrieval_precision(rec, ho > 3.5, k=50)
                p20 += tm.retrieval_precision(rec, ho > 3.5, k=20)
                p10 += tm.retrieval_precision(rec, ho > 3.5, k=10)
                # h10 += tm.retrieval_hit_rate(rec, ho > 3.5, k=10)
                # h5 += tm.retrieval_hit_rate(rec, ho > 3.5, k=5)
                # h1 += tm.retrieval_hit_rate(rec, ho > 3.5, k=1)

            n100 /= len(self.valid_loader)
            r50 /= len(self.valid_loader)
            r20 /= len(self.valid_loader)
            r10 /= len(self.valid_loader)
            p50 /= len(self.valid_loader)
            p20 /= len(self.valid_loader)
            p10 /= len(self.valid_loader)
            # h10 /= len(self.valid_loader)
            # h5 /= len(self.valid_loader)
            # h1 /= len(self.valid_loader)

            n100 = n100.item()
            r50 = r50.item()
            r20 = r20.item()
            r10 = r10.item()
            p50 = p50.item()
            p20 = p20.item()
            p10 = p10.item()
            # h10 = h10.item()
            # h5 = h5.item()
            # h1 = h1.item()

            print(n100, r50, r20, p50, p20)
            self.writter.add_scalar("valid/ndcg100", n100, epoch)
            self.writter.add_scalar("valid/r50", r50, epoch)
            self.writter.add_scalar("valid/r20", r20, epoch)
            self.writter.add_scalar("valid/r10", r10, epoch)
            self.writter.add_scalar("valid/p50", p50, epoch)
            self.writter.add_scalar("valid/p20", p20, epoch)
            self.writter.add_scalar("valid/p10", p10, epoch)
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
        vpos = minibatch

        vpos, phpos, vneg, phneg = self.model.gibbs_sample(vpos, t)
        activations = phpos.sum(dim=0) / len(minibatch)

        hb_delta = (phpos - phneg).sum(dim=0) / len(minibatch)
        vb_delta = (vpos - vneg).sum(dim=0) / len(minibatch)

        w_delta = torch.matmul(vb_delta.unsqueeze(2), hb_delta.unsqueeze(0))

        # apply learning rate decay
        self.alpha = decay(self.lr)

        # update the parameters of the model
        self.model.vB += vb_delta * self.alpha
        self.model.hB += hb_delta * self.alpha
        self.model.w += w_delta * self.alpha

        # apply momentum if applicable
        if self.momentum > 0.0:
            self.model.vB += self.prev_vd * self.momentum
            self.model.hB += self.prev_hd * self.momentum
            self.model.w += self.prev_wd * self.momentum

        # remember the deltas for next training step when using momentum
        self.prev_wd = w_delta
        self.prev_hd = hb_delta
        self.prev_vd = vb_delta

        # # calculate the regularization terms
        reg_w = self.model.w * self.l2

        # # apply regularization
        self.model.w -= reg_w  # * len(minibatch)
        if self.l1 > 0:
            q_new = (activations / len(minibatch)) * (
                1 - self.l1
            ) + self.l1 * self.q_prev
            sparsity_penalty = (
                -self.sparsity * torch.log(q_new)
                - (1 - self.sparsity)
                - torch.log(1 - q_new)
            )
            self.model.hB -= sparsity_penalty
            self.model.w += torch.ones_like(self.model.w) * sparsity_penalty
            self.q_prev = q_new

        rmse, mae = self.batch_error(minibatch)

        return rmse, mae
