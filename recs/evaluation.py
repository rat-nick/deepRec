from surprise import SVD, BaselineOnly, NormalPredictor, KNNWithZScore
from surprise import Trainset, Dataset as sDataset, Reader
from data.dataset import Dataset
import pandas as pd
from .random import RandomPredictior
from surprise import accuracy
from collections import defaultdict
from itertools import groupby
import numpy as np
import argparse
import time
from . import metrics as mt

parser = argparse.ArgumentParser()
parser.add_argument("--user-threshold", type=int, default=0)
parser.add_argument("--ratings-path", type=str)
parser.add_argument("--result-path", type=str)
parser.add_argument("--ranking", action="store_true")
parser.add_argument("--leave-one-out", action="store_true")
args = parser.parse_args()


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


def contains_item(set, i):
    return any(item[0] == i for item in set)


def antitestset_for_users(test, train: Trainset):
    ats = list()

    for u in test:
        uid = train.to_inner_uid(u)
        for i in train.all_items():
            if not contains_item(train.ur[uid], i):
                ats += [(u, train.to_raw_iid(i), 0)]

    return ats


ds = Dataset(args.ratings_path, user_threshold=args.user_threshold)

rnd = RandomPredictior()
bl = BaselineOnly()
algo = KNNWithZScore()
algos = {
    "random": RandomPredictior(),
    "baseline": BaselineOnly(),
    "svd": SVD(n_factors=20),
    "knn": KNNWithZScore(k=40),
    "normal": NormalPredictor(),
}


def ranking_eval(res_path, ds, algos):
    metrics = defaultdict(list)
    for train, test in ds.fihoUserKFold(5):
        test_users = {x[0] for x in test}
        test2 = test + antitestset_for_users(test_users, train)

        for key in algos:
            # fit the algo and time it
            start = time.time()
            algos[key].fit(train)
            end = time.time()

            metrics[f"{key}_fittime"] += [end - start]

            # test for rmse and mae
            pred = algos[key].test(test)

            rmse = accuracy.rmse(pred, False)
            mae = accuracy.mae(pred, False)

            metrics[f"{key}_rmse"] += [rmse]
            metrics[f"{key}_mae"] += [mae]

            # test for ranking metrics
            start = time.time()
            pred = algos[key].test(test2)
            end = time.time()

            metrics[f"{key}_predtime"] += [end - start]

            for k in [10, 20, 50, 100]:
                p, r = precision_recall_at_k(pred, k)
                p = np.mean(list(p.values()))
                r = np.mean(list(r.values()))
                metrics[f"{key}_p{k}"] += [p]
                metrics[f"{key}_r{k}"] += [r]

        pd.DataFrame.from_dict(metrics).to_csv(res_path)


def loo_eval(res_path, ds, algos):
    metrics = defaultdict(list)

    for train, test in ds.looUserKFold(5):
        test_users = {x[0] for x in test}
        test2 = test + antitestset_for_users(test_users, train)

        for key in algos:
            algos[key].fit(train)
            pred = algos[key].test(test2)
            pred_dict = {
                k: [(t[1], t[3]) for t in list(g)]
                for k, g in groupby(pred, key=lambda x: x[0])
            }

            arhr = mt.AverageReciprocalHitRank(pred_dict, test)
            metrics[f"{key}_arhr"] += [arhr]
            for k in [1, 5, 10, 20]:
                hr = mt.CumulativeHitRate(k, pred_dict, test)
                metrics[f"{key}_hr@{k}"] += [hr]

        pd.DataFrame.from_dict(metrics).to_csv(res_path)


if args.leave_one_out:
    loo_eval(args.result_path, ds, algos)
if args.ranking:
    ranking_eval(args.result_path, ds, algos)
