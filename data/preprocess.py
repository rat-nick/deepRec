import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser(description="Parameters for data preprocessing")

parser.add_argument(
    "--userThreshold",
    type=int,
    default=20,
    help="The minimum number of ratings a user has to give to stay in the dataset",
)

parser.add_argument(
    "--itemThreshold",
    type=int,
    default=5,
    help="The minimum number of times an item has to be rated to stay in the dataset",
)

parser.add_argument(
    "--path",
    type=str,
    default="./data/clean/ratings.csv",
    help="The default location to save the preprocessed data",
)

args = parser.parse_args()

df = pd.read_csv("data/ml-1m/ratings.dat", sep="::", engine="python", header=None)
unique_users = lambda: len(df[0].unique())
unique_items = lambda: len(df[1].unique())
print(f"There are {unique_users()} unique users in the dataset")
print(f"There are {unique_items()} unique items in the dataset")


print("Performing dataset cleaning...")
df = df.groupby(0).filter(lambda x: len(x) >= args.userThreshold)
df = df.groupby(1).filter(lambda x: len(x) >= args.itemThreshold)
print("Completed!")
print(f"There are {unique_users()} unique users in the dataset")
print(f"There are {unique_items()} unique items in the dataset")
print(df[0].value_counts().min())
print(df[1].value_counts().min())

df.to_csv(args.path, sep=",", header=False, index=False)


def remove_users(df: pd.DataFrame, threshold: int = 20) -> pd.DataFrame:
    return df.groupby(0).filter(lambda x: len(x) >= threshold)
