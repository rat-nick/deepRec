import itertools
import math
from collections import defaultdict

from surprise import accuracy


class RecommenderMetrics:
    def MAE(self, predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(self, predictions):
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(self, predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)

        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if estimatedRating >= minimumRating:
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def HitRate(self, topNPredicted, leftOutPredictions):
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

    def CumulativeHitRate(self, topNPredicted, leftOutPredictions, ratingCutoff=0):
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

    def RatingHitRate(self, topNPredicted, leftOutPredictions):
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

    def AverageReciprocalHitRank(self, topNPredicted, leftOutPredictions):
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
    def UserCoverage(self, topNPredicted, numUsers, ratingThreshold=0):
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

    def Diversity(self, topNPredicted, simsAlgo):
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

    def Novelty(self, topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n

    def PrecissionAtK(self, k, recommendations, userRatings, relevancyThreshold=3.5):
        relevant = [(i, r) for i, r in userRatings if r > relevancyThreshold]
        # TODO: see if this makes sense
        relevant.sort(key=lambda x: x[1], reverse=True)
        relevant = relevant[:k]
        relevant = list(map(lambda x: x[0], relevant))
        total = 0
        recommendations = recommendations[:k]
        for p in recommendations:
            if p[0] in relevant:
                total += 1

        return total / k

    def RecallAtK(self, k, recommendations, userRatings, relevancyThreshold=3.5):
        relevant = [(i, r) for i, r in userRatings if r > relevancyThreshold]
        # TODO: see if this makes sense
        relevant.sort(key=lambda x: x[1], reverse=True)
        relevant = relevant[:k]
        relevant = list(map(lambda x: x[0], relevant))
        total = 0
        recommendations = recommendations[:k]
        for p in recommendations:
            if p[0] in relevant:
                total += 1
        if len(relevant) == 0:
            return 0
        return total / len(relevant)

    def F1AtK(self, k, recommendations, userRatings, relevancyThreshold=3.5):
        r = self.RecallAtK(k, recommendations, userRatings, relevancyThreshold)
        p = self.PrecissionAtK(k, recommendations, userRatings, relevancyThreshold)
        if (r + p) == 0:
            return 0
        return 2 * (r * p) / (r + p)

    def DCGAtK(self, k, recommendations, userRatings, relevancyThreshold=3.5):
        relevant = sorted(userRatings, key=lambda x: x[1], reverse=True)
        relevant = list(map(lambda x: x[0], relevant))[:k]
        recommendations = recommendations[:k]
        dcg = 0
        i = 1
        for r in recommendations:
            if r[0] in relevant:
                dcg += 1 / math.log2(i + 1)
            i += 1

        return dcg

    def iDCGAtK(self, k, userRatings, relevancyThreshold=3.5):
        ideal = sorted(userRatings, key=lambda x: x[1], reverse=True)
        ideal = ideal[:k]
        idcg = 0
        i = 1
        for r in ideal:
            v = 1 if r[1] >= relevancyThreshold else 0
            idcg += v / math.log2(i + 1)
        return idcg

    def NDCGAtK(self, k, recommendations, userRatings, relevancyThreshold=3.5):
        dcg = self.DCGAtK(k, recommendations, userRatings, relevancyThreshold)
        idcg = self.iDCGAtK(k, userRatings, relevancyThreshold)

        if idcg == 0:
            return 0
        return dcg / idcg
