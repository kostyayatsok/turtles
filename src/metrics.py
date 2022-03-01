import numpy as np

def apk(actual, predicted, k=5):
    """Computes the average precision at k.

    Args:
        actual: The turtle ID to be predicted.
        predicted : A list of predicted turtle IDs (order does matter).
        k : The maximum number of predicted elements.

    Returns:
        The average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score


def mapk(actual, predicted, k=5):
    """ Computes the mean average precision at k.

    The turtle ID at actual[i] will be used to score predicted[i][:k] so order
    matters throughout!

    actual: A list of the true turtle IDs to score against.
    predicted: A list of lists of predicted turtle IDs.
    k: The size of the window to score within.

    Returns:
        The mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])