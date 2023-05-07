import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd


def create_database():
    ml1m = pd.read_csv(
        "data/ml-1m/movies.csv",
        encoding="latin-1",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
    )
    ml1m["year"] = ml1m["title"].str.extract(r"\((\d{4})\)")
    ml1m["title"] = ml1m["title"].str.extract(r"^(.+)\s\(\d{4}\)$")
    ml1m.set_index("movieId", inplace=True)
    links = pd.read_csv("data/ml-20m/links.csv", header=0).set_index("movieId")
    joined = ml1m.join(links, how="left")
    joined["tmdbId"].fillna("0", inplace=True)
    joined["imdbId"].fillna("0", inplace=True)
    joined["tmdbId"] = joined["tmdbId"].astype("int")
    joined["imdbId"] = joined["imdbId"].astype("int")
    joined.to_csv("data/warehouse/movies.csv")

    ml20m = pd.read_csv("data/ml-20m/movies.csv", header=0, encoding="latin-1", sep=",")

    ml20m["englishTitle"] = ml20m["title"].str.extract(r"^([^(]+)")
    ml20m["altTitle"] = ml20m["title"].str.extract(r"\(([^)]+)\)")
    ml20m["year"] = ml20m["title"].str.extract(r"\((\d{4})\)$")
    ml20m.drop(["title"], axis=1, inplace=True)
    ml20m.rename(columns={"englishTitle": "title"}, inplace=True)
    ml20m["altTitle"] = ml20m["altTitle"].apply(
        lambda x: None if str(x).isnumeric() and len(str(x)) == 4 else x
    )
    ml20m["title"] = ml20m["title"].str.replace("a.k.a", "").str.strip()
    ml20m["altTitle"] = ml20m["altTitle"].str.replace("a.k.a", "").str.strip()
    joined = ml20m.join(links, how="left")
    ml1m.set_index(["title", "year"], inplace=True)
    ml20m.set_index(["title", "year"], inplace=True)

    joined = ml1m.join(ml20m, lsuffix="1m", rsuffix="20m", how="left")


if __name__ == "__main__":
    create_database()
