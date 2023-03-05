import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def hold_out_ratings(ratings: pd.DataFrame, ratio: float = 0.2):
    held_out = ratings.groupby("user").sample(frac=ratio, random_state=RANDOM_STATE)
    ratings = ratings[~ratings.index.isin(held_out.index)]
    return ratings, held_out


def split_folds(
    folds: int = 5,
    data_dir: str = "data/folds",
    df: pd.DataFrame = None,
):
    users = df["user"].unique()
    items = df["item"].unique()

    with open(f"{data_dir}/dims", "x") as f:
        f.write("%d, %d " % (len(users), len(items)))

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kf.split(users)):
        train_users = df[df["user"].isin(train_idx)]

        dir = f"{data_dir}/{i + 1}"
        if not os.path.exists(dir):
            os.makedirs(dir)
        train_users.to_csv(f"{dir}/train.csv", sep=",", header=None, index=False)
        valid_idx, test_idx = train_test_split(
            test_idx, test_size=0.5, shuffle=True, random_state=42
        )
        test_users = df[df["user"].isin(test_idx)]
        test_users.to_csv(f"{dir}/test.csv", sep=",", header=None, index=False)
        valid_users = df[df["user"].isin(valid_idx)]
        valid_users.to_csv(f"{dir}/valid.csv", sep=",", header=None, index=False)
