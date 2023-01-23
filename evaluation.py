import metrics
import torch
import sklearn.model_selection as ms


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, validation_data: torch.Tensor, k=50):
        recall = 0
        precision = 0
        f1 = 0
        ndcg = 0

        self.model.eval()
        n = len(validation_data)

        for u in range(n):
            # split users ratings for train and validation
            train, valid = self.splitRatings(validation_data[u])
            recs = self.model(train)
            recs[train.nonzero()] = 0
            recall += metrics.recall(k, recs, valid)
            precision += metrics.precission(k, recs, valid)
            f1 += metrics.f1(k, recs, valid)
            # ndcg += metrics.ndcg(k, recs, ratings)

            return (
                recall / n,
                precision / n,
                (2 * (recall * precision) / (recall + precision)) / n,
            )
            # "f1": f1 / n,
            # "ndcg": ndcg / n,

    def splitRatings(self, tensor, ratio=0.8):
        idx = tensor.nonzero().cpu()
        # split indicies in 80:20 ratio
        t_idx, v_idx = ms.train_test_split(idx, train_size=ratio)

        t_mask = torch.zeros_like(tensor)
        t_mask[t_idx] = 1
        v_mask = torch.zeros_like(tensor)
        v_mask[v_idx] = 1
        train = tensor * t_mask
        valid = tensor * v_mask
        return train, valid
