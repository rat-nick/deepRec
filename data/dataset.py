import logging
from typing import List, Tuple

import pandas as pd
import surprise
from sklearn.model_selection import KFold
from surprise import Trainset

diff = lambda x, y: pd.concat([x, y, y]).drop_duplicates(keep=False)

logger = logging.getLogger("data.dataset")
logging.basicConfig(level=logging.DEBUG)


class Dataset:
    def __init__(
        self,
        path: str = None,
        sep: str = ",",
        user_threshold: int = 20,
        less_than: bool = False,
        # filter_strategies: List[function] = [],
        sparse: bool = False,
    ):
        df = pd.read_csv(
            path, sep=sep, engine="python", encoding="latin-1", low_memory=True
        )
        logger.debug("Loaded pandas dataframe into memory")

        df = df.groupby("user").filter(
            lambda x: len(x) <= user_threshold
            if less_than
            else len(x) >= user_threshold
        )

        logger.debug("Applying filter strategies to dataset")
        # FIXME: this
        # we want to apply all filter strategies
        # for fs in filter_strategies:
        #     df = fs(df)

        logger.debug("Finished applying filter strategies to dataset")

        # remove all but the first 3 columns
        df = df.iloc[:, :3]
        logger.debug("Finished cleaning")

        self.dataset = surprise.Dataset.load_from_df(
            df, surprise.Reader(line_format="user item rating")
        )
        logger.info("Created surprise dataset")
        self.trainset = self.dataset.build_full_trainset()
        logger.info("Finished building surprise trainset")

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
            train = {key: self.trainset.ur[key] for key in trainset}
            test = {key: self.trainset.ur[key] for key in testset}

            yield train, test

    def fihoUserKFold(
        self, n_splits: int = 5, ho_ratio: float = 0.2
    ) -> Tuple[Trainset, List[Tuple]]:

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
