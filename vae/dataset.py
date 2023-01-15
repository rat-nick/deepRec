import torch
from torch import Tensor
from torch.utils.data import Dataset as tDataset
from surprise import Dataset as sDataset


class Dataset(tDataset):
    def __init__(self, name="ml-1m"):
        super(tDataset, self).__init__()
        self.data = sDataset.load_builtin(name)
        self.data = self.data.build_full_trainset()

    def __len__(self):
        return self.data.n_items

    def __getitem__(self, index) -> Tensor:
        t = torch.zeros(self.n_items)

        for u, i, r in self.data.all_ratings():
            if u == index:
                t[i] = r / 5.0

        if torch.cuda.is_available():
            t = t.to(torch.device("cuda"))

        return t

    @property
    def n_items(self):
        return self.data.n_items

    @property
    def n_users(self):
        return self.data.n_users

    @property
    def ratings_scale(self):
        return 5
