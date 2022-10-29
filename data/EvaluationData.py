from .dataset import MyDataset


class EvaluationData:
    def __init__(self, dataset: MyDataset):
        self.dataset = dataset

    def getUserRatings(self, user):
        ratings = self.dataset.innerRatingsDF.loc[
            self.dataset.innerRatingsDF["user"] == user
        ]
        ratings = ratings[["item", "rating"]]
        return ratings


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
