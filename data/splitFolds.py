import argparse
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds to split the data for cross validation",
    )
    parser.add_argument(
        "--held-out-ratio",
        type=float,
        default=0.2,
        help="Ratio of users ratings to be held out for testing",
    )
    parser.add_argument(
        "--dir", type=str, default="data/folds", help="directory to save folds in"
    )

    df = pd.read_csv(
        "data/clean/ratings.csv", names=["user", "item", "rating", "timestamp"]
    ).drop(columns=["timestamp"])
    users = df["user"].unique()

    args = parser.parse_args()

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    for i, (train_idx, test_idx) in enumerate(kf.split(users)):
        train_users = df[df["user"].isin(train_idx)]

        dir = f"{args.dir}/{i + 1}"
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
