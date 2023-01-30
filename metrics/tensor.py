import torch
import torchmetrics.functional as tmf

# TODO: should not asume that the functions should process input values


def recall(preds: torch.Tensor, target: torch.Tensor, k: int = 10):
    target = target > 3.5
    preds -= preds.min()
    preds /= preds.max()
    return tmf.retrieval_recall(preds, target, k=k)


def precision(preds: torch.Tensor, target: torch.Tensor, k: int = 10):
    target = target > 3.5
    preds -= preds.min()
    preds /= preds.max()
    return tmf.retrieval_precision(preds, target, k=k)


def ndcg(preds: torch.Tensor, target: torch.Tensor, k: int = 10):
    target = target > 3.5
    # preds = preds / 5
    preds -= preds.min()
    preds /= preds.max()
    return tmf.retrieval_normalized_dcg(preds, target, k=k)


def hr(preds: torch.Tensor, target: torch.Tensor, k: int = 10):
    target = target > 3.5
    preds -= preds.min()
    preds /= preds.max()
    return tmf.retrieval_hit_rate(preds, target, k)


def mrr(preds: torch.Tensor, target: torch.Tensor, k: int = 10):
    target = target > 3.5
    preds -= preds.min()
    preds /= preds.max()
    return tmf.retrieval_reciprocal_rank(preds, target)


if __name__ == "__main__":
    t = torch.tensor([1, 1, 4, 4, 5, 4, 3, 5]).float()
    p = torch.tensor([5, 5, 4, 2, 1, 4, 1, 3]).float()
    k = 10
    print(recall(p, t, 3))
    print(precision(p, t, 3))
    print(ndcg(p, t, 10))
    print(hr(p, t, 3))
