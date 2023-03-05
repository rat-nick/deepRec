import surprise
from sklearn.model_selection import KFold
from collections import defaultdict
import pandas as pd

diff = lambda x, y: pd.concat([x, y, y]).drop_duplicates(keep=False)


class Dataset:
    def __init__(self, path: str = None, sep: str = ",", user_threshold: int = 20):
        df = pd.read_csv(path, sep=sep, engine="python", encoding="latin-1")
        df = df.groupby("user").filter(lambda x: len(x) >= user_threshold)
        df = df.iloc[:, :3]
        self.dataset = surprise.Dataset.load_from_df(
            df, surprise.Reader(line_format="user item rating")
        )

        self.trainset = self.dataset.build_full_trainset()

    def all_users(self):
        return self.trainset.all_users()

    def all_ratings(self):
        return self.trainset.all_ratings()

    def all_items(self):
        return self.trainset.all_items()

    @property
    def n_items(self):
        return self.trainset.n_items

    @property
    def n_users(self):
        return self.trainset.n_users

    @property
    def n_ratings(self):
        return self.trainset.n_ratings

    def userKFold(self, n_splits: int = 5):
        kf = KFold(n_splits, shuffle=True, random_state=42)
        users = [u for u in self.all_users()]

        for trainset, testset in kf.split(users):
            train_data = list()
            test_data = list()

            for key in trainset:
                for i, r in self.trainset.ur[key]:
                    train_data += [(key, i, r)]

            for key in testset:
                for i, r in self.trainset.ur[key]:
                    test_data += [(key, i, r)]

            yield train_data, test_data

    def fihoUserKFold(self, n_splits: int = 5, ho_ratio: float = 0.2):

        for train_data, test_data in self.userKFold(n_splits):
            train = pd.DataFrame(train_data, columns=["user", "item", "rating"])
            test = pd.DataFrame(test_data, columns=["user", "item", "rating"])

            ho = test.groupby("user").sample(frac=ho_ratio, random_state=42)
            fi = diff(test, ho)
            train = list(pd.concat([train, fi]).itertuples(index=False))
            test = list(ho.itertuples(index=False))
            ds = self.dataset
            ts = self.trainset
            train = [(ts.to_raw_uid(u), ts.to_raw_iid(i), r, 0) for u, i, r in train]
            test = [(ts.to_raw_uid(u), ts.to_raw_iid(i), r) for u, i, r in test]
            yield ds.construct_trainset(train), test

    def looUserKFold(self, n_splits: int = 5):

        for train_data, test_data in self.userKFold(n_splits):
            train = pd.DataFrame(train_data, columns=["user", "item", "rating"])
            test = pd.DataFrame(test_data, columns=["user", "item", "rating"])

            left_out = test.groupby("user").sample(n=1, random_state=42)
            fi = diff(test, left_out)

            train = list(pd.concat([train, fi]).itertuples(index=False))
            test = list(left_out.itertuples(index=False))

            ds = self.dataset
            ts = self.trainset
            train = [(ts.to_raw_uid(u), ts.to_raw_iid(i), r, 0) for u, i, r in train]
            test = [(ts.to_raw_uid(u), ts.to_raw_iid(i), r) for u, i, r in test]

            yield ds.construct_trainset(train), test
