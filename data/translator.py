import pandas as pd
from surprise import Dataset, Reader


def internalize_ids(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert user and item ids from raw to inner (0-based)

    Args:
        df (pd.DataFrame, optional): ratings dataframe of raw ids. Defaults to None.

    Returns:
        pd.DataFrame: transformed datatframe with inner ids
    """
    reader = Reader(line_format="user item rating")
    ds = Dataset.load_from_df(df, reader=reader).build_full_trainset()

    df["user"] = df["user"].apply(ds.to_inner_uid)
    df["item"] = df["item"].apply(ds.to_inner_iid)

    return df
