import pandas as pd
from surprise import Dataset


class EvaluationData:
    def __init__(self, trainset=None, testset=None):
        self.testset = testset
        self.trainset = trainset
        self.testset_df = self.datasetToDataFrame(testset)
        self.trainset_df = self.datasetToDataFrame(trainset)

        self.fullset_df = self.trainset_df.appe

    def datasetToDataFrame(self, dataset: Dataset):
        df = pd.DataFrame(
            dataset.__dict__["raw_ratings"],
            columns=["user_id", "item_id", "rating", "timestamp"],
        )

        return df

    def getUserRatings(self, user):
        pass
