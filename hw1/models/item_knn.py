import numpy as np


class ItemKNN:
    """Item-based kNN с косинусным сходством и сглаживанием редких совпадений."""

    def __init__(self, neighbors=50, shrinkage=10):
        self.neighbors, self.shrinkage = neighbors, shrinkage

    def fit(self, interactions):
        self.history = interactions.tocsr()
        gram = (self.history.T @ self.history).toarray()
        norm = np.sqrt(np.diag(gram))
        similarity = gram / np.maximum(norm[:, None] * norm[None, :], 1e-12)
        similarity *= gram / (gram + self.shrinkage)
        np.fill_diagonal(similarity, 0)
        k = min(self.neighbors, len(gram) - 1)
        keep = np.argpartition(similarity, -k, axis=0)[-k:]
        weights = np.zeros_like(similarity)
        cols = np.arange(len(gram))[None, :]
        weights[keep, cols] = similarity[keep, cols]
        self.weights = weights
        return self

    def score(self, users):
        return np.asarray(self.history[users] @ self.weights)
