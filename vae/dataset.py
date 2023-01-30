import torch
from surprise import Dataset as sDataset, Reader
from torch import Tensor
from torch.utils.data import Dataset as tDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset(tDataset):
    def __init__(self, device=torch.device("cpu")):
        super(tDataset, self).__init__()
        self.data = torch.load("vae/sparse.pt", map_location=device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index) -> Tensor:
        return self.data[index]

    @property
    def n_items(self):
        return self.data.shape[1]

    @property
    def n_users(self):
        return self.data.shape[0]

    @property
    def ratings_scale(self):
        return 5


class Trainset(tDataset):
    def __init__(self, path: str, device=torch.device("cpu")):
        torch.manual_seed(42)

        super(tDataset, self).__init__()

        df = pd.read_csv(path, header=None, index_col=None)

        self.data = torch.zeros(6040, 3416)
        for row in df.itertuples():
            self.data[row[1]][row[2]] = float(row[3])

        # remove all empty rows
        self.data = self.data[self.data.sum(dim=1) != 0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index) -> Tensor:
        return self.data[index]

    @property
    def n_items(self):
        return self.data.shape[1]

    @property
    def n_users(self):
        return self.data.shape[0]

    @property
    def ratings_scale(self):
        return 5


class Testset(Trainset):
    def __init__(self, path: str, device=torch.device("cpu"), ratio: float = 0.2):
        super().__init__(path, device)

        self.foldin = torch.zeros_like(self.data)
        self.holdout = torch.zeros_like(self.data)

        # hold out data
        for i in range(len(self.data)):
            # find indicies of all ratings
            idx = self.data[i].nonzero()

            fi_idx, ho_idx = train_test_split(idx, test_size=ratio, random_state=42)
            self.foldin[i, fi_idx] = self.data[i, fi_idx]
            self.holdout[i, ho_idx] = self.data[i, ho_idx]
        assert self.foldin.sum() + self.holdout.sum() == self.data.sum()

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return self.foldin[index], self.holdout[index]


class LeaveOneOutSet(Trainset):
    def __init__(
        self,
        path: str,
        device=torch.device("cpu"),
        rt: float = 3.5,
    ):
        super().__init__(path, device)

        self.foldin = torch.clone(self.data)
        self.holdout = torch.zeros_like(self.data)

        # hold out data
        for i in range(len(self.data)):
            # find indicies of all relevant ratings
            idx = self.data[i] > rt
            idx = idx.nonzero().flatten()
            if len(idx) == 0:
                continue
            # select one from the relevant to holdout
            ho_idx = np.random.choice(idx, size=1)
            self.foldin[i, ho_idx] = 0  # null the value in the foldin set
            self.holdout[i, ho_idx] = self.data[
                i, ho_idx
            ]  # perserve the value in the holdout

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        return self.foldin[index], self.holdout[index]


def hold_out_ratings(ratings: pd.DataFrame, ratio: float = 0.2):
    held_out = ratings.groupby("user").sample(frac=ratio, random_state=42)
    ratings = ratings[~ratings.index.isin(held_out.index)]
    return ratings, held_out
