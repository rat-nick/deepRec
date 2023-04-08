from collections import defaultdict
import math
from statistics import mean
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
        trainset: dataset.Dataset,
        validset: dataset.Dataset,
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
            metrics = defaultdict(list)
            self.train_loop(epoch)
            self.reconstruction_validation(metrics, epoch, self.valid_loader, True)
            ndcg = self.ranking_evaluation(metrics, epoch, self.valid_loader, True)
            self.leave_one_out_evaluation(metrics, epoch, self.valid_loader, True)

            if ndcg > best:
                best = ndcg
                self.model.save()
                self.patience = 0
            else:
                self.patience += 1

            if self.patience > self.params.patience:
                print(f"Early stopping after {epoch} epochs")
                self.epoch = epoch
                return

        self.epoch = epoch

    def leave_one_out_evaluation(self, metrics, epoch, loader, log=False):
        self.model.eval()
        for case in loader:
            case = case[0].to(torch.device("cuda"))
            fi, ho = leave_one_out(case, threshold=3.5)
            if fi == None:
                continue
            fi = ohwmv(fi).unsqueeze(0)

            rec = self.model(fi)
            rec = onehot_to_ranking(rec)[0]
            fi = onehot_to_ratings(fi)[0]
            rec[fi > 0] = 0
            ks = [1, 5, 10, 20]
            metrics["n_ratings2"] += [fi.count_nonzero().item()]
            metrics["arhr"] += [tm.retrieval_reciprocal_rank(rec, ho > 3.5).item()]
            for k in ks:
                metrics[f"hr@{k}"] += [tm.retrieval_hit_rate(rec, ho > 3.5, k).item()]
        if log:
            for key in metrics:
                self.writter.add_scalar(f"valid/{key}", mean(metrics[key]), epoch)

    def ranking_evaluation(self, metrics, epoch, loader, log=False):
        self.model.eval()
        l = len(metrics["ndcg@100"])
        for case in loader:
            case = case[0].to(torch.device("cuda"))
            fi, ho = split(case)
            metrics["n_ratings1"] += [fi.count_nonzero().item()]
            fi = ohwmv(fi).unsqueeze(0)

            rec = self.model(fi)
            rec = onehot_to_ranking(rec)[0]
            fi = onehot_to_ratings(fi)
            rec[fi[0] > 0] = 0
            ks = [10, 20, 50, 100]

            metrics["ndcg@10"] += [tm.retrieval_normalized_dcg(rec, ho, 10).item()]
            metrics["ndcg@100"] += [tm.retrieval_normalized_dcg(rec, ho, 100).item()]

            for k in ks:
                metrics[f"recall@{k}"] += [tm.retrieval_recall(rec, ho > 0, k).item()]
                metrics[f"precision@{k}"] += [
                    tm.retrieval_precision(rec, ho > 0, k).item()
                ]
        if log:
            for key in metrics:
                self.writter.add_scalar(f"valid/{key}", mean(metrics[key]), epoch)

        return mean(metrics["ndcg@100"][l:])

    def reconstruction_validation(self, metrics, epoch, loader, log=False):
        self.model.eval()

        # RMSE/MAE validation loop
        for case in loader:
            case = case.to(torch.device("cuda"))
            case = ohwmv(case)
            _rmse, _mae = self.batch_error(case)
            metrics["rmse"] += [_rmse]
            metrics["mae"] += [_mae]
            metrics["n_ratings3"] += [case.count_nonzero().item()]
        if log:
            self.writter.add_scalar("valid/RMSE", mean(metrics["rmse"]), epoch)
            self.writter.add_scalar("valid/MAE", mean(metrics["mae"]), epoch)

    def train_loop(self, epoch):
        rmse = mae = 0
        n = 0

        self.model.train()
        # training loop
        for minibatch in self.train_loader:
            minibatch = minibatch.to(torch.device("cuda"))
            minibatch = ohwmv(minibatch)
            _rmse, _mae = self.apply_gradient(minibatch, self.t, self.decay)
            rmse += _rmse
            mae += _mae
            n += 1

        self.writter.add_scalar("train/rmse", rmse / n, epoch)
        self.writter.add_scalar("train/mae", mae / n, epoch)

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
