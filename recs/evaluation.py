from surprise import SVD, BaselineOnly, NormalPredictor, KNNWithZScore
from surprise import Trainset
from data.dataset import Dataset
import pandas as pd
from .random import RandomPredictior
from collections import defaultdict
from itertools import groupby
import argparse
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
from . import metrics as mt

parser = argparse.ArgumentParser()
parser.add_argument("--user-threshold", type=int, default=0)
parser.add_argument("--ratings-path", type=str)
parser.add_argument("--result-path", type=str)
parser.add_argument("--ranking", action="store_true")
parser.add_argument("--leave-one-out", action="store_true")
parser.add_argument("--ndcg", action="store_true")

args = parser.parse_args()

project = lambda lst, i: [t[i] for t in lst]


def pred2dict(pred):
    pred_dict = {
        k: [(t[1], t[3]) for t in list(g)] for k, g in groupby(pred, key=lambda x: x[0])
    }

    return pred_dict


def pred2dict2(pred):
    pred = sorted(pred, key=lambda x: x[2], reverse=True)
    pred_dict = defaultdict(list)
    for u, i, true, pred, _ in pred:
        if any(item[0] == i for item in pred_dict[u]):
            continue
        pred_dict[u].append((i, true, pred))

    return pred_dict


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

    # for u, i, r in train.all_ratings():
    #     if not any(
    #         train.to_inner_uid(tr[0]) == u and train.to_inner_iid(tr[1]) == i
    #         for tr in test
    #     ):
    #         ats += [(train.to_raw_uid(u), train.to_raw_iid(i), 0)]

    # return ats

    for u in test:
        uid = train.to_inner_uid(u)
        for i in train.all_items():
            if not contains_item(train.ur[uid], i):
                ats += [(u, train.to_raw_iid(i), 0)]

    return ats


ds = Dataset(args.ratings_path, user_threshold=args.user_threshold)
print(ds.n_items)
algos = {
    "random": RandomPredictior(),
    "baseline": BaselineOnly(),
    "svd": SVD(n_factors=20),
    "knn": KNNWithZScore(k=40),
    "normal": NormalPredictor(),
}


def ranking_eval(ds, algos, metrics):

    for train, test in ds.fihoUserKFold(5):
        test_users = {x[0] for x in test}
        metrics["num_ratings1"] += [
            len(train.ur[train.to_inner_uid(u)]) for u in test_users
        ]
        full_testset = test + antitestset_for_users(test_users, train)

        for key in algos:
            # fit the algo and time it
            start = time.time()
            algos[key].fit(train)
            end = time.time()

            metrics[f"{key}_fittime"] += [end - start]

            # test for rmse and mae
            pred = algos[key].test(test)
            pd = pred2dict2(pred)
            for user in pd:
                metrics[f"{key}_n_ratings1"] += [
                    len(train.ur[train.to_inner_uid(user)])
                ]
                y_true = project(pd[user], 1)
                y_pred = project(pd[user], 2)
                metrics[f"{key}_rmse"] += [
                    mean_squared_error(y_true, y_pred, squared=False)
                ]
                metrics[f"{key}_mae"] += [mean_absolute_error(y_true, y_pred)]

            # test for ranking metrics
            start = time.time()
            pred = algos[key].test(full_testset)
            end = time.time()

            metrics[f"{key}_predtime"] += [end - start]

            for k in [10, 20, 50, 100]:
                p, r = precision_recall_at_k(pred, k)
                metrics[f"{key}_p@{k}"] += list(p.values())
                metrics[f"{key}_r@{k}"] += list(r.values())

            pd = pred2dict2(pred)

            for user in pd:
                y_true = project(pd[user], 1)
                y_pred = project(pd[user], 2)

                for k in [10, 100]:
                    metrics[f"{key}_ndcg@{k}"] += [ndcg_score([y_true], [y_pred], k=k)]


def ndcg_eval(ds, algos, metrics):
    for train, test in ds.fihoUserKFold(5):
        test_users = {x[0] for x in test}
        ats = antitestset_for_users(test_users, train)

        fulltest = test + ats

        for key in algos:

            algos[key].fit(train)
            pred = algos[key].test(fulltest)
            pd = pred2dict2(pred)

            for user in pd:
                y_true = project(pd[user], 1)
                y_pred = project(pd[user], 2)

                for k in [10, 100]:
                    metrics[f"{key}_ndcg@{k}"] += [ndcg_score([y_true], [y_pred], k=k)]


def loo_eval(ds, algos, metrics):
    for train, test in ds.looUserKFold(5):
        test_users = {x[0] for x in test}
        metrics["num_ratings2"] += [
            len(train.ur[train.to_inner_uid(u)]) for u in test_users
        ]
        test2 = test + antitestset_for_users(test_users, train)

        for key in algos:
            algos[key].fit(train)
            pred = algos[key].test(test2)
            pred_dict = pred2dict(pred)

            arhr = mt.AverageReciprocalHitRank(pred_dict, test)
            metrics[f"{key}_arhr"] += arhr
            for k in [1, 5, 10, 20]:
                hr = mt.CumulativeHitRate(k, pred_dict, test)
                metrics[f"{key}_hr@{k}"] += hr


metrics = defaultdict(list)
if args.ranking:
    ranking_eval(ds, algos, metrics)
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in metrics.items()])).to_csv(
        args.result_path
    )
if args.leave_one_out:
    loo_eval(ds, algos, metrics)
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in metrics.items()])).to_csv(
        args.result_path
    )
if args.ndcg:
    ndcg_eval(ds, algos, metrics)
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in metrics.items()])).to_csv(
        args.result_path
    )
