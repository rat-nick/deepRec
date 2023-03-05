from typing import List, Tuple

from .dataAccess import MyDataset
import numpy as np


class EvaluationData:
    def __init__(self, dataset: MyDataset):
        self.dataset = dataset

    def getUserRatings(self, user) -> List[Tuple[int, float]]:

        ratings = self.dataset.innerRatingsDF.loc[
            self.dataset.innerRatingsDF["user"] == user
        ]
        ratings = ratings[["item", "rating"]]
        ratings = [(i, r) for i, r in list(ratings.itertuples(index=False))]

        return ratings

    def splitUsersRatings(self, user, ratio: float = 0.2) -> Tuple[list, list]:
        ratings = self.getUserRatings(user)
        np.random.shuffle(ratings)
        ratings, held_out_ratings = np.split(ratings, [int(ratio * len(ratings))])
        return ratings, held_out_ratings


if __name__ == "__main__":
    dataset = MyDataset(
        data_dir="ml-1m",
        ratings_path="ratings.dat",
        ratings_sep="::",
        items_path="movies.dat",
        items_sep="::",
    )
    evalData = EvaluationData(dataset)
    print(evalData.dataset.innerRatingsDF.head())
    print(evalData.getUserRatings(5))
