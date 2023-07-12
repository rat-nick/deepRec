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
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.NOTSET)


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
            torch.tensor(values, dtype=torch.float32),
            torch.Size([self.n_items]),
        ).to_dense()

        return tensor
