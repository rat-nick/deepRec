import math
from typing import Tuple

import torch
from ..utils.tensors import *
from data.dataset import MyDataset


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
        self.ratings = ratings
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

        self.setup_weights_and_biases()

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
        # if len(v.shape) > 1:
        #    v = v.flatten()

        # XXX: get rid of flattening
        # v = v.flatten(start_dim=1)

        a = torch.mm(v.flatten(-2), self.w.flatten(end_dim=1))

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

        a = torch.matmul(self.w, h.t())

        pv = self.v.unsqueeze(2) + a
        pv = activation(pv.permute(2, 0, 1))
        return pv

    def apply_gradient(
        self, minibatch: torch.Tensor, t: int = 1, decay=lambda x: x
    ) -> None:
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """

        se, ae, n = self.batch_error(minibatch)
        rmse = math.sqrt(se / n)
        mae = ae / n

        # vb_delta = torch.zeros(self.n_visible, device=self.device)
        # hb_delta = torch.zeros(self.n_hidden, device=self.device)
        # w_delta = torch.zeros(self.n_visible, self.n_hidden, device=self.device)
        activations = torch.zeros(self.n_hidden, device=self.device)
        v0 = minibatch

        v0, ph0, vt, pht = self.gibbs_sample(v0, t)
        activations = ph0.sum(dim=0) / len(minibatch)

        hb_delta = (ph0 - pht).sum(dim=0) / len(minibatch)
        vb_delta = (v0 - vt).sum(dim=0) / len(minibatch)

        w_delta = torch.matmul(vb_delta.unsqueeze(2), hb_delta.unsqueeze(0))

        # divide learning rate by the size of the minibatch
        # w_delta /= len(minibatch)

        # apply learning rate decay
        self.alpha = decay(self.learning_rate)

        # update the parameters of the model
        self.v += vb_delta * self.alpha
        self.h += hb_delta * self.alpha
        self.w += w_delta * self.alpha

        # apply momentum if applicable
        if self.momentum > 0.0 and hasattr(self, "prev_w_delta"):
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
        for i in range(t):
            vk = self.backward_pass(hk)
            vk[input.sum(dim=2) == 0] = input[input.sum(dim=2) == 0]
            phk, hk = self.forward_pass(vk)

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
        if len(self._metrics["rmse"]) < self.patience:
            return False

        if self._metrics["rmse"][-1] <= self._metrics["rmse"][self._best_epoch]:
            self._current_patience = 0
            return False
        else:
            self._current_patience += 1

        return self._current_patience >= self.patience

    def load_model_from_file(self, fpath):
        params = torch.load(fpath)
        self.w = params["w"]
        self.v = params["v"]
        self.h = params["h"]

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

    def fit(self, data: MyDataset, t=1, decay=lambda x: x):
        self.data = data
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

        # p = torch.sum(train, dim=0) / train.shape[0]
        # p = p / (1 - p)

        # p[torch.isinf(p)] = 1
        # p = torch.log(p)
        # p = torch.nan_to_num(p)
        # self.v = p.flatten()

        # self.v = torch.zeros(data.nItems * 10)
        self.w *= 0.01
        self.h *= torch.abs(self.h + 10) * -1
        loading = "-" * 20
        if self.verbose:
            print(f"#####\t{loading}\tTRAIN\t\t\t\tVALIDATION")
            print(f"Epoch\t{loading}\tRMSE\t\tMAE\t\tRMSE\t\tMAE")

        # nCases = math.ceil(self.data.trainData / self.batch_size)
        numBatches = len(data.trainUsers) / self.batch_size
        _5pct = numBatches / 20

        for epoch in range(1, self.max_epoch + 1):
            if self.verbose:
                print(epoch, end="\t", flush=True)
            current = 0
            rmse = mae = 0

            for minibatch in data.batches(data.trainData, self.batch_size):
                _rmse, _mae = self.apply_gradient(
                    minibatch=minibatch,
                    t=t,
                    decay=decay,
                )
                rmse += _rmse
                mae += _mae

                current += 1
                if current >= _5pct:
                    print("#", end="", flush=True)
                    current = 0
            print("\t", end="", flush=True)

            print(format(rmse / numBatches, ".6f"), end="\t")
            print(format(mae / numBatches, ".6f"), end="\t")

            rmse, mae = self.calculate_errors("validation")
            self._metrics["rmse"] += [rmse]
            self._metrics["mae"] += [mae]

            if self.verbose:
                print(format(rmse, ".6f"), end="\t")
                print(format(mae, ".6f"), end="\t")

            if (
                len(self._metrics["rmse"]) == 1
                or self._metrics["rmse"][-1] < self._metrics["rmse"][self._best_epoch]
            ):
                self.__save_checkpoint(epoch)

            if self.early_stopping and self.__early_stopping():
                self.__load_checkpoint()
                self.save_model_to_file("rbm.pt")
                return
            print()

        self.__load_checkpoint()

    def save_model_to_file(self, fpath):
        params = {"w": self.w, "v": self.v, "h": self.h}
        torch.save(params, fpath)

    def setup_weights_and_biases(self):
        self.w = torch.zeros(
            self.n_visible, self.ratings, self.n_hidden, device=self.device
        )
        self.v = torch.zeros(self.n_visible, self.ratings, device=self.device)
        self.h = torch.zeros(self.n_hidden, device=self.device)

    # FIXME: fix error calculation

    def calculate_errors(self, s):
        se = 0
        ae = 0
        n = 0

        if s == "validation":
            data = self.data.validationData
        elif s == "test":
            data = self.data.testData
        else:
            data = self.data.trainData

        for v in self.data.batches(data, self.batch_size):
            _se, _ae, _n = self.batch_error(v)
            ae += _ae
            se += _se
            n += _n
        return math.sqrt(se / n), ae / n

    def batch_error(self, v):
        se = 0
        ae = 0
        n = 0

        rec = self.reconstruct(v)
        n += len(v[v.sum(dim=2) > 0])
        vRating = onehot_to_ratings(v)
        recRating = onehot_to_ratings(rec)

        # set the same value for missing values so they don't affect error calculation
        recRating[v.sum(dim=2) == 0] = vRating[v.sum(dim=2) == 0]

        se += ((recRating - vRating) * (recRating - vRating)).sum().item()
        ae += torch.abs(recRating - vRating).sum().item()

        return se, ae, n

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
