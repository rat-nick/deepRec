import itertools
import math
from collections import defaultdict

from surprise import accuracy
import sklearn.metrics as slm
import numpy as np
import torch
import torchmetrics as tm


def MAE(predictions):
    return accuracy.mae(predictions, verbose=False)


def RMSE(predictions):
    return accuracy.rmse(predictions, verbose=False)


def topN(k, userRatings):
    actual = list(enumerate(userRatings))
    actual.sort(key=lambda x: x[1], reverse=True)
    actual = actual[:k]
    actual = list(map(lambda x: (x[0], x[1].item()), actual))
    return actual


def HitRate(topNPredicted, leftOutPredictions):
    hits = 0
    total = 0
    # For each left-out rating
    for leftOut in leftOutPredictions:
        userID = leftOut[0]
        leftOutMovieID = leftOut[1]
        # Is it in the predicted top 10 for this user?
        hit = False
        for movieID, predictedRating in topNPredicted[int(userID)]:
            if int(leftOutMovieID) == int(movieID):
                hit = True
                break
        if hit:
            hits += 1
        total += 1
    # Compute overall precision
    return hits / total


def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
    hits = 0
    total = 0
    # For each left-out rating
    for (
        userID,
        leftOutMovieID,
        actualRating,
        estimatedRating,
        _,
    ) in leftOutPredictions:
        # Only look at ability to recommend things the users actually liked...
        if actualRating >= ratingCutoff:
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if int(leftOutMovieID) == movieID:
                    hit = True
                    break
            if hit:
                hits += 1
            total += 1
    # Compute overall precision
    return hits / total


def RatingHitRate(topNPredicted, leftOutPredictions):
    hits = defaultdict(float)
    total = defaultdict(float)
    # For each left-out rating
    for (
        userID,
        leftOutMovieID,
        actualRating,
        estimatedRating,
        _,
    ) in leftOutPredictions:
        # Is it in the predicted top N for this user?
        hit = False
        for movieID, predictedRating in topNPredicted[int(userID)]:
            if int(leftOutMovieID) == movieID:
                hit = True
                break
        if hit:
            hits[actualRating] += 1
        total[actualRating] += 1
    # Compute overall precision
    for rating in sorted(hits.keys()):
        print(rating, hits[rating] / total[rating])


def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
    summation = 0
    total = 0
    # For each left-out rating
    for (
        userID,
        leftOutMovieID,
        actualRating,
        estimatedRating,
        _,
    ) in leftOutPredictions:
        # Is it in the predicted top N for this user?
        hitRank = 0
        rank = 0
        for movieID, predictedRating in topNPredicted[int(userID)]:
            rank = rank + 1
            if int(leftOutMovieID) == movieID:
                hitRank = rank
                break
        if hitRank > 0:
            summation += 1.0 / hitRank
        total += 1
    return summation / total


# What percentage of users have at least one "good" recommendation
def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
    hits = 0
    for userID in topNPredicted.keys():
        hit = False
        for movieID, predictedRating in topNPredicted[userID]:
            if predictedRating >= ratingThreshold:
                hit = True
                break
        if hit:
            hits += 1
    return hits / numUsers


def Diversity(topNPredicted, simsAlgo):
    n = 0
    total = 0
    simsMatrix = simsAlgo.compute_similarities()
    for userID in topNPredicted.keys():
        pairs = itertools.combinations(topNPredicted[userID], 2)
        for pair in pairs:
            movie1 = pair[0][0]
            movie2 = pair[1][0]
            innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
            innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
            similarity = simsMatrix[innerID1][innerID2]
            total += similarity
            n += 1

        S = total / n
        return 1 - S


def Novelty(topNPredicted, rankings):
    n = 0
    total = 0
    for userID in topNPredicted.keys():
        for rating in topNPredicted[userID]:
            movieID = rating[0]
            rank = rankings[movieID]
            total += rank
            n += 1
    return total / n


def precission(k, y_pred: torch.Tensor, y_true: torch.Tensor):

    relevant = len([i for i in y_true if i > 0.7])
    y_pred = topN(k, y_pred)
    y_true = topN(k, y_true)
    y_true = [x[0] for x in y_true]
    total = 0
    for item in y_pred:
        if item[0] in y_true:
            total += 1

    return total / relevant


def prep(k, recs, actual):
    recs = topN(k, recs)
    idx = [item[0] for item in recs]
    # arrange the rankings in the order of users best items
    actual = actual[idx]
    recs = [1 if rank > 0.8 else 0 for rank in recs]
    actual = list(map(lambda x: 1 if x[1] > 3.5 else 0, actual))
    return recs, actual


def recall(k, y_pred, y_true):
    y_pred = topN(k, y_pred)
    y_true = topN(k, y_true)

    y_true = [x[0] for x in y_true]
    total = 0
    for item in y_pred:
        if item[0] in y_true:
            total += 1

    return total / len(y_true)


def f1(k, recommendations, userRatings):
    r = recall(k, recommendations, userRatings)
    p = precission(k, recommendations, userRatings)
    if (r + p) == 0:
        return 0
    return 2 * (r * p) / (r + p)


def dcg(k, recommendations, userRatings):
    relevant = userRatings
    relevant = list(map(lambda x: x[0], relevant))[:k]
    recommendations = recommendations[:k]
    dcg = 0
    i = 1
    for r in recommendations:
        if r[0] in relevant:
            dcg += 1 / math.log2(i + 1)
        i += 1
    return dcg


def idcg(k, userRatings):
    ideal = sorted(userRatings, key=lambda x: x[1], reverse=True)
    ideal = ideal[:k]
    idcg = 0
    i = 1
    for r in ideal:
        idcg += r[1] / math.log2(i + 1)
    return idcg


def ndcg(k, recs, ratings):
    recs, actual = prep(k, recs, ratings)
    recs = np.asarray([recs])
    actual = np.asarray([actual])
    return slm.ndcg_score(actual, recs)
