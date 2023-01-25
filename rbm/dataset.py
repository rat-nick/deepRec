import torch
from surprise import Dataset as sDataset
from torch import Tensor
from torch.utils.data import Dataset as tDataset


class Dataset(tDataset):
    def __init__(self, device=torch.device("cpu")):
        super(tDataset, self).__init__()
        self.data = torch.load("rbm/sparse.pt", map_location=device)

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
        return self.data.shape[2]
