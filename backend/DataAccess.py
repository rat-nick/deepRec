import pandas as pd
from surprise import Dataset, Reader


def flatten_dicts(res):
    return [{"id": k, **v} for k, v in res.items()]


class DataAccess:
    def __init__(
        self,
    ):
        self.movies = pd.read_csv("../data/warehouse/movies.csv", header=0).set_index(
            "movieId"
        )
        self.translator = Dataset.load_from_file(
            "../data/ml-1m/ratings.csv",
            Reader(line_format="user item rating timestamp", sep=",", skip_lines=1),
        ).build_full_trainset()
        ids = self.inner2raw([i for i in self.translator.all_items()])
        ids = list(map(int, ids))
        valid_ids = self.movies.index.intersection(ids)
        self.movies = self.movies.loc[valid_ids]
        pass

    def inner2raw(self, movies):
        return list(map(self.translator.to_raw_iid, movies))

    def raw2inner(self, movies):
        return list(map(self.translator.to_inner_iid, movies))

    def attachInfo(self, ids):
        ids = list(map(int, ids))
        res = self.movies.loc[ids].to_dict(orient="index")
        res = flatten_dicts(res)
        return res

    def all_items(self):
        res = self.movies.to_dict(orient="index")
        res = flatten_dicts(res)
        return res

    def sample_items(self, n=20):
        res = self.movies.sample(n).to_dict(orient="index")
        res = flatten_dicts(res)
        return res

    def search(self, term):
        df = self.movies
        df = df[
            df.apply(
                lambda row: row.astype(str).str.contains(term, case=False).any(), axis=1
            )
        ]
        res = df.to_dict(orient="index")
        res = flatten_dicts(res)
        return res
