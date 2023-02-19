import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from surprise import SVD, BaselineOnly, Dataset, KNNBasic, NormalPredictor, Reader

from metrics.tensor import ndcg, precision, recall

from .random import RandomPredictior

sim_options = {"name": "cosine", "user_based": True}

set_diff = lambda x, y: pd.concat([x, y, y]).drop_duplicates(keep=False)
fT = lambda x: torch.tensor([x]).float()  # convert to float tensor
relevant = lambda x: np.array(x) > 3.5


class Engine:
    def __init__(self, trainset: pd.DataFrame, algo):
        self.algo = algo
        self.reader = Reader(line_format="user item rating timestamp", sep=",")
        self.trainset = trainset

    def fit(self, data: pd.DataFrame) -> None:
        dataset = Dataset(self.reader)
        data = dataset.load_from_df(data, self.reader).build_full_trainset()
        self.algo.fit(data)

    def recommend(self, ratings: pd.DataFrame) -> np.ndarray:
        # Add the new_ratings to the current trainset
        new_users = ratings[0].unique()
        s = np.random.choice(new_users, size=50)
        df = pd.concat([self.trainset, ratings])
        df = df.drop(columns=["timestamp"], errors="ignore")
        data = Dataset.load_from_df(df, reader=self.reader).build_full_trainset()

        # Fit the algo to the new data
        self.algo.fit(data)

        n_items = data.n_items
        n_users = len(ratings[0].unique())
        res = np.zeros((n_users, n_items))

        num = 0
        for u in data.all_users():
            if data.to_raw_uid(u) in new_users:
                try:
                    for i in data.all_items():
                        if i not in [item for (item, _) in data.ur[u]]:
                            est = self.algo.estimate(u, i)
                            if type(est) == Tuple:
                                res[num][i] = est[0]
                            else:
                                res[num][i] = est

                except Exception as e:
                    print(str(e))
                    pass
                num += 1
                # if num % 10 == 0:
                #     print(f"{num}/{n_users}")

        return res


svd = SVD(verbose=True, n_factors=200)
knn = KNNBasic(sim_options=sim_options, k=40, min_k=1, verbose=True)
baseline = BaselineOnly(verbose=True)
nrm = NormalPredictor()
rnd = RandomPredictior()

algos = {
    "random": rnd,
    "svd": svd,
    "knn": knn,
    "baseline": baseline,
    "normal": nrm,
}


def evaluate(algo, res_path):
    ndcg_lst = []
    r50_lst = []
    r20_lst = []
    r10_lst = []
    p50_lst = []
    p20_lst = []
    p10_lst = []
    hr10_lst = []
    hr5_lst = []
    hr1_lst = []
    arhr_lst = []

    for i in range(1, 6):
        # load the train and testsets from folds directory
        trainset = pd.read_csv(f"data/folds/{i}/train.csv", header=None, index_col=None)
        testset = pd.read_csv(f"data/folds/{i}/test.csv", header=None, index_col=None)

        train_users = trainset[0].unique()
        test_users = testset[0].unique()
        np.random.seed = 42
        sampled_users = np.random.choice(test_users, 10)
        sampled_users = test_users

        # initialize the engine
        engine = Engine(trainset, algo)

        # define foldin and holdout ratings for each user in the testset
        foldin = pd.DataFrame(testset[testset[0].isin(sampled_users)])
        heldout = foldin.groupby(0).sample(frac=0.2, random_state=42)
        foldin = set_diff(foldin, heldout)

        n100 = 0
        r50 = 0
        r20 = 0
        r10 = 0
        p50 = 0
        p20 = 0
        p10 = 0
        n = 0

        # convert heldout ratings to interaction matrix
        n_users = len(test_users)
        n_items = len(trainset[1].unique())

        y_true = np.zeros((len(sampled_users), n_items))
        x = Dataset.load_from_df(
            heldout, reader=Reader(line_format="user item rating")
        ).build_full_trainset()

        for u, i, r in x.all_ratings():
            y_true[u][x.to_raw_iid(i)] = float(r)

        y_pred = engine.recommend(foldin[foldin[0].isin(sampled_users)])

        # handle impossible predictions
        # y_true = y_true[~np.all(y_pred == 0, axis=1)]
        # y_pred = y_pred[~np.all(y_pred == 0, axis=1)]
        # y_true[y_pred.sum(axis=1) == 0] = y_pred[y_pred.sum(axis=1) == 0]

        for i in range(len(y_pred)):
            true = fT(y_true[i])
            pred = fT(y_pred[i])
            n100 += ndcg(pred, true, k=100)
            r50 += recall(pred, true, k=50)
            r20 += recall(pred, true, k=20)
            r10 += recall(pred, true, k=10)
            p50 += precision(pred, true, k=50)
            p20 += precision(pred, true, k=20)
            p10 += precision(pred, true, k=10)
            # hr20 += tm.retrieval_hit_rate(pred, actual, k=20)

        ndcg_lst += [n100 / len(y_pred)]
        r50_lst += [r50 / len(y_pred)]
        r20_lst += [r20 / len(y_pred)]
        r10_lst += [r10 / len(y_pred)]
        p50_lst += [p50 / len(y_pred)]
        p20_lst += [p20 / len(y_pred)]
        p10_lst += [p10 / len(y_pred)]
        # hr20_lst += [hr20 / len(y_pred)]

    with open(f"results/{res_path}", "x") as f:
        f.write("n100:\t%.4f ± %.4f\n" % (np.mean(ndcg_lst), np.std(ndcg_lst)))
        f.write("r50:\t%.4f ± %.4f\n" % (np.mean(r50_lst), np.std(r50_lst)))
        f.write("r20:\t%.4f ± %.4f\n" % (np.mean(r20_lst), np.std(r20_lst)))
        f.write("r10:\t%.4f ± %.4f\n" % (np.mean(r10_lst), np.std(r10_lst)))
        f.write("p50:\t%.4f ± %.4f\n" % (np.mean(p50_lst), np.std(p50_lst)))
        f.write("p20:\t%.4f ± %.4f\n" % (np.mean(p20_lst), np.std(p20_lst)))
        f.write("p10:\t%.4f ± %.4f\n" % (np.mean(p10_lst), np.std(p10_lst)))


if __name__ == "__main__":
    for a in algos:
        evaluate(algos[a], f"{a}.result")
    # print("hr20:\t%.4f ± %.4f" % (np.mean(hr20_lst), np.std(hr20_lst)))
