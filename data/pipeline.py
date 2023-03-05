from .translator import internalize_ids
from .split import split_folds
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ratings-path", type=str)
parser.add_argument("--out-dir", type=str)

args = parser.parse_args()
if __name__ == "__main__":
    df = pd.read_csv(args.ratings_path, sep=",", header=0)
    df = df.drop(columns=["timestamp"])
    df = internalize_ids(df)
    split_folds(df=df, data_dir=args.out_dir)
