import pandas as pd

DATA_DIR = "../data/"


class DAL:
    def __init__(self):
        with open(DATA_DIR + "/ml-20m/movies.csv", "r") as f:
            movies_df = pd.read_csv(f)
            # movies_df = movies_df.set_index("movieId")
            # print(movies_df.head(20))

        with open(DATA_DIR + "/ml-20m/links.csv", "r") as f:
            links_df = pd.read_csv(f)
            # links_df = links_df.set_index("movieId")
            links_df = links_df.fillna(0)
            links_df["tmdbId"] = links_df["tmdbId"].astype("int")

        self.movies = pd.concat([movies_df, links_df], axis=1, join="inner")
        self.movies = self.movies.loc[:, ~self.movies.T.duplicated(keep="first")]

        with open(DATA_DIR + "/ml-20m/ratings.csv", "r") as f:
            self.ratings = pd.read_csv(f)
            self.ratings.drop("timestamp", axis=1, inplace=True)
            self.ratings["rating"] *= 2

    def getTopNPopularMovies(self, n):
        r = self.ratings[["movieId", "rating"]]
        avg = r["rating"].mean()
        r["relative_rating"] = r["rating"] - avg
        print(r.groupby(["movieId"]).sum(["relative_rating"]))
        print(r.head())

    def searchItems(self, term):
        if term == "":
            return self.movies.head(100)
        return self.movies[self.movies["title"].str.contains(term, case=False, na=True)]


if __name__ == "__main__":
    dal = DAL()
    print(dal.movies.head())
