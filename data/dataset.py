import time
from dataclasses import dataclass
from pathlib import Path
from random import shuffle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader

DATA_DIR = Path(__file__).parent / "ml-20m"
RATINGS_FILE = "ratings.csv"
ITEMS_FILE = "movies.csv"
LINKS_FILE = "links.csv"


@dataclass
class DataLoadingParams:
    ratings_path: str
    ratings_sep: str
    items_path: str
    items_sep: str


DATASETS_DICT = {
    "ml-100k": DataLoadingParams(
        "u.data",
        "\t",
        "u.item",
        "|",
    ),
    "ml-1m": DataLoadingParams(
        "ratings.dat",
        "::",
        "movies.dat",
        "::",
    ),
}


class MyDataset:
    def __init__(
        self,
        dataset="ml-100k",
    ):

        loading_params = DATASETS_DICT[dataset]
        data_dir = Path(__file__).parent / dataset
        if torch.cuda.is_available():
            print("CUDA available! Setting default tensor type to cuda.FloatTensor")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        try:
            self.rawRatingsDF = pd.read_csv(
                f"{data_dir}/{loading_params.ratings_path}",
                sep=loading_params.ratings_sep,
                header=0,
                engine="python",
                encoding="latin-1",
            )
        except FileNotFoundError:
            print("The ratings file doesn't exist!")

        try:
            self.itemsDF = pd.read_csv(
                f"{data_dir}/{loading_params.items_path}",
                sep=loading_params.items_sep,
                header=0,
                engine="python",
                encoding="latin-1",
            )
        except FileNotFoundError:
            print("The items file doesn't exist!")

        try:
            linksDF = pd.read_csv(
                f"{data_dir}/{links_path}",
                sep=links_sep,
                header=0,
                engine="python",
                encoding="latin-1",
            )

            linksDF = linksDF.fillna(0)
            linksDF["tmdbId"] = linksDF["tmdbId"].astype("int")

            self.itemsDF = pd.concat([self.itemsDF, linksDF], axis=1, join="inner")
            self.itemsDF = self.itemsDF.loc[:, ~self.itemsDF.T.duplicated(keep="first")]

        except FileNotFoundError:
            print("The links file doesn't exist!")
        except NameError:
            print("The links parameter aren't initialized!")

        self.rawRatingsDF.columns = ["user", "item", "rating", "timestamp"]
        self.rawRatingsDF["rating"] *= 1
        self.rawRatingsDF = self.rawRatingsDF.rename(
            columns={"userId": "user", "movieId": "item"}
        )
        try:
            self.rawRatingsDF.drop("timestamp", inplace=True, axis=1)
        except:
            print("No timestamp...")
        self.ratingsDS = Dataset.load_from_df(
            df=self.rawRatingsDF,
            reader=Reader(
                # line_format="user item rating timestamp",
                rating_scale=(1, 10),
            ),
        )

        self.trainset = self.ratingsDS.build_full_trainset()

        self.inner2RawUser = self.trainset.to_raw_uid
        self.inner2RawItem = self.trainset.to_raw_iid
        self.raw2InnerUser = self.trainset.to_inner_uid
        self.raw2InnerItem = self.trainset.to_inner_iid

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
        batcher = Batcher(data, batch_size, (self.nItems, 5))
        return batcher.next()

    def getRawUserRatings(self, uid):
        return self.rawRatingsDF.loc[self.rawRatingsDF["user"] == uid]

    def getInnerUserRatings(self, uid):
        return self.innerRatingsDF.loc[self.innerRatingsDF["user"] == uid]


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

        # initialize the tensor to be returned
        t = torch.zeros((self.bs,) + self.size)

        current = 0
        for u in users:
            start = time.time()
            ratings = self.df[self.df["user"] == u]
            t[
                current,
                ratings["item"].to_numpy(),
                ratings["rating"].to_numpy() - 1,
            ] = 1.0
            current += 1
            if current >= self.bs:
                yield t
                # print(f"Batching lasted {time.time() - start}")
                current = 0

        yield t
        return


if __name__ == "__main__":
    dataAccess = MyDataset("ml-1m")
    dataAccess.trainTestValidationSplit()
    trainset = dataAccess.trainset
    for b in dataAccess.batches(dataAccess.trainData, 1024):
        print(b.shape)
