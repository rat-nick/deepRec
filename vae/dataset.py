import logging
import sys

sys.path.append("/home/nikola/projects/deepRec")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split as tts
from surprise.dataset import Dataset
from torch import Tensor
from torch.utils.data import Dataset as tDataset

from data.dataset import Dataset as DS

logger = logging.getLogger("vae.dataset")
logging.basicConfig(level=logging.INFO)


class UserRatingsDataset:
    def __init__(self, path, threshold, rating_function=lambda x: x):
        self.ds = DS(path, user_threshold=threshold)
        self.n_users = self.ds.n_users
        self.n_items = self.ds.n_items
        self.user_ratings = self.ds.trainset.ur
        self.rating_function = rating_function
        del self.ds  # free the memory

    def tvt_split(self):
        train, test = tts(self.user_ratings, test_size=0.2)
        validation, test = tts(test, test_size=0.5)

        return train, validation, test

    def tvt_datasets(self):
        train, valid, test = self.tvt_split()
        train = Dataset(train, self.rating_function, self.n_items)
        valid = Dataset(valid, self.rating_function, self.n_items)
        test = Dataset(test, self.rating_function, self.n_items)
        return train, valid, test


class Dataset(tDataset):
    def __init__(
        self,
        ratings_path: str = None,
        user_ratings: list = None,
        n_items: int = None,
        ut: int = 0,
        ratings_scale: int = 5,
    ):
        super(tDataset, self).__init__()
        if ratings_path != None:
            self.ds = DS(ratings_path, user_threshold=ut)
            self.n_users = self.ds.n_users
            self.n_items = self.ds.n_items

        if n_items != None:
            self.n_items = self.ds.n_items

        self.ratings_scale = ratings_scale
        # we only need user ratings
        self.user_ratings = self.ds.trainset.ur
        # then we delete the surprise object from memory
        del self.ds
        logger.info("Finished creating base dataset object")

    @classmethod
    def build(cls, user_ratings, n_items):
        dataset = Dataset(user_ratings=user_ratings)
        dataset.n_items = n_items
        return dataset

    def __len__(self):
        return len(self.user_ratings)

    def __getitem__(self, index) -> Tensor:
        indicies = list(map(lambda x: x[0], self.user_ratings[index]))
        values = list(map(lambda x: 1 if x[1] >= 3.5 else 0, self.user_ratings[index]))

        tensor = torch.sparse_coo_tensor(
            torch.tensor(indicies).unsqueeze(0),
            torch.tensor(values),
            torch.Size([self.n_items]),
        ).to_dense()

        return tensor

    def train_test_split(self):
        train, test = tts(self.user_ratings, test_size=0.2)
        return (
            Dataset.build(user_ratings=train, n_items=self.n_items),
            Dataset.build(user_ratings=test, n_items=self.n_items),
        )

    def train_test_validation_split(self):
        train, test = tts(self.user_ratings, test_size=0.2)
        validation, test = tts(test, test_size=0.5)
        return (
            Dataset.build(user_ratings=train, n_items=self.n_items),
            Dataset.build(user_ratings=validation, n_items=self.n_items),
            Dataset.build(user_ratings=test, n_items=self.n_items),
        )

    def userKFold(self, n_splits=5, kind: str = "2-way"):
        for train, test in self.ds.userKFold(n_splits):

            size = len(train)
            train_data = torch.zeros((size, self.ds.n_items))
            user = 0
            for _, value in train.items():
                for item, rating in value:
                    train_data[user][item] = 1.0 if float(rating) >= 3.5 else 0
                user += 1

            size = len(test)
            test_data = torch.zeros((size, self.ds.n_items))
            user = 0
            for _, value in test.items():
                for item, rating in value:
                    test_data[user][item] = 1.0 if float(rating) >= 3.5 else 0
                user += 1

            if kind == "2-way":
                yield (
                    Dataset(data=train_data),
                    Dataset(data=test_data),
                )

            elif kind == "3-way":
                valid_idx, test_idx = tts(
                    list(range(len(test_data))),
                    test_size=0.5,
                    shuffle=True,
                    random_state=42,
                )
                valid_data = test_data[valid_idx]

                yield (
                    Dataset(data=train_data, sparse=True),
                    Dataset(data=valid_data, sparse=True),
                    Dataset(data=test_data[test_idx], sparse=True),
                )


class Dataset(tDataset):
    def __init__(
        self, user_ratings=None, rating_function=lambda x: x, n_items: int = None
    ):
        self.data = user_ratings
        self.rating_function = rating_function
        self.n_items = n_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tensor:
        indicies = list(map(lambda x: x[0], self.data[index]))
        values = list(map(lambda x: self.rating_function(x[1]), self.data[index]))

        tensor = torch.sparse_coo_tensor(
            torch.tensor(indicies).unsqueeze(0),
            torch.tensor(values, dtype=torch.float16),
            torch.Size([self.n_items]),
        ).to_dense()

        return tensor


class Trainset(tDataset):
    def __init__(self, n_users, n_items, path: str, device=torch.device("cpu")):
        torch.manual_seed(42)

        super(tDataset, self).__init__()

        df = pd.read_csv(path, header=None, index_col=None)

        self.data = torch.zeros(n_users, n_items)
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
    def __init__(
        self,
        n_users,
        n_items,
        path: str,
        device=torch.device("cpu"),
        ratio: float = 0.2,
    ):
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


class DatasetBuilder:
    def __init__(self):
        self.dataset = Dataset()

    @property
    def product(self):
        pass

    def set_user_ratings(self, user_ratings):
        self.instance.user_ratings = user_ratings
        return self
