"""Evaluation metrics for recommendation systems."""


def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    """Precision@k: fraction of top-k recommendations in the relevant set."""
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return len(set(rec_k) & relevant) / len(rec_k)
