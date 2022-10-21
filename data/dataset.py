from random import shuffle
from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader
from surprise.dataset import Trainset

DATA_DIR = Path(__file__).parent / "ml-20m"
RATINGS_FILE = "ratings.csv"
ITEMS_FILE = "movies.csv"
LINKS_FILE = "links.csv"


class MyDataset:
    def __init__(self):
        self.rawRatingsDF = pd.read_csv(f"{DATA_DIR}/{RATINGS_FILE}", sep=",", header=0)
        self.itemsDF = pd.read_csv(f"{DATA_DIR}/{ITEMS_FILE}", sep=",", header=0)
        linksDF = pd.read_csv(f"{DATA_DIR}/{LINKS_FILE}", sep=",", header=0)

        linksDF = linksDF.fillna(0)
        linksDF["tmdbId"] = linksDF["tmdbId"].astype("int")

        self.itemsDF = pd.concat([self.itemsDF, linksDF], axis=1, join="inner")
        self.itemsDF = self.itemsDF.loc[:, ~self.itemsDF.T.duplicated(keep="first")]

        self.rawRatingsDF["rating"] *= 2
        self.rawRatingsDF = self.rawRatingsDF.rename(
            columns={"userId": "user", "movieId": "item"}
        )

        self.rawRatingsDF.drop("timestamp", inplace=True, axis=1)

        self.ratingsDS = Dataset.load_from_df(
            df=self.rawRatingsDF,
            reader=Reader(
                # line_format="user item rating timestamp",
                rating_scale=(1, 10),
            ),
        )

        self.trainset = self.ratingsDS.build_full_trainset()
        self.nItems = self.trainset.n_items
        self.nUsers = self.trainset.n_users
        self.allUsers = [u for u in self.trainset.all_users()]
        self.allItems = [i for i in self.trainset.all_items()]
        self.convertRatings2InnerIDs()

    def trainTestValidationSplit(self):
        self.trainUsers, self.testUsers = train_test_split(self.allUsers, test_size=0.2)
        self.validationUsers, self.testUsers = train_test_split(
            self.testUsers, test_size=0.5
        )

        toRawUser = self.trainset.to_raw_uid

        self.trainData = self.innerRatingsDF[
            self.innerRatingsDF["user"].isin(self.trainUsers)
        ]
        self.validationData = self.innerRatingsDF[
            self.innerRatingsDF["user"].isin(self.validationUsers)
        ]
        self.testData = self.innerRatingsDF[
            self.innerRatingsDF["user"].isin(self.testUsers)
        ]

    def convertRatings2InnerIDs(self):
        self.innerRatingsDF = pd.concat(
            [
                self.rawRatingsDF["user"].apply(
                    lambda x: self.trainset.to_inner_uid(x)
                ),
                self.rawRatingsDF["item"].apply(
                    lambda x: self.trainset.to_inner_iid(x)
                ),
                self.rawRatingsDF["rating"],
            ],
            axis=1,
        )
        self.innerRatingsDF = pd.DataFrame(
            self.innerRatingsDF, columns=["user", "item", "rating"]
        )

    def batches(self, data: pd.DataFrame, batch_size: int):
        batcher = Batcher(data, batch_size, (self.nItems, 10))
        return batcher.next()


class Batcher:
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        size: tuple,
    ):
        self.df = df
        self.bs = batch_size
        self.size = size

    def next(self):
        users = self.df["user"].unique()
        shuffle(users)
        if torch.cuda.is_available():
            print("CUDA available!")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        t = torch.zeros((self.bs,) + self.size)

        current = 0
        for u in users:
            userRatings = self.df[self.df["user"] == u]
            t[
                current,
                userRatings["item"].to_numpy(),
                userRatings["rating"].to_numpy() - 1,
            ] = 1.0
            current += 1
            if current == self.bs:
                yield t
                current = 0
        yield t
        return


if __name__ == "__main__":
    dataAccess = MyDataset()
    dataAccess.trainTestValidationSplit()
    trainset = dataAccess.trainset
    for b in dataAccess.batches(dataAccess.trainData, 10):
        print(b)
