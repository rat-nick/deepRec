from .translator import internalize_ids
from .split import split_folds
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("data/clean/ratings.csv", sep=",", header=0)
    df = df.drop(columns=["timestamp"])
    df = internalize_ids(df)
    split_folds(df=df)
