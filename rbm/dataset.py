from numpy import int32, int8
import pandas as pd
from surprise.dataset import Dataset as sDataset
import torch


class Dataset:
    def __init__(self) -> None:
        self.load()

    def load(self):
        ds = sDataset.load_builtin(name="ml-100k", prompt=True)
        self.data = ds

    def getDatasetAsTensor(self):
        """
        Convert ratings into a tensor of shape `(u, m, 5)` where u is the number of users, and m is the number of items.
        """
        trainset = self.data.build_full_trainset()
        self.df = pd.DataFrame(trainset.all_ratings())
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(trainset.build_anti_testset(fill=0)),
            ]
        )
        self.df.columns = ["user", "item", "rating"]
        self.df["rating"] = self.df["rating"].astype("int")
        self.df = pd.concat([self.df, pd.get_dummies(self.df.rating)], axis=1)
        self.df = self.df.drop(["rating", 0], axis=1)
        # print(self.df)

        u = len(trainset.all_users())
        i = len(trainset.all_items())
        k = 5

        data = self.df.to_numpy(dtype=int8)
        return torch.FloatTensor(data[:, 2:].reshape(u, i, 5))
        # self.t = torch.zeros(, len(trainset.all_items(), 5))


if __name__ == "__main__":
    data = Dataset()
    t = data.reshapeForTraining()
