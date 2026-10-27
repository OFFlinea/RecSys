import numpy as np
from scipy.linalg import solve


class EASE:
    def __init__(self, regularization=100):
        self.regularization = regularization

    def fit(self, interactions):
        self.history = interactions.tocsr()
        gram = (interactions.T @ interactions).toarray().astype(np.float64)
        gram.flat[::len(gram) + 1] += self.regularization
        inverse = solve(gram, np.eye(len(gram)), assume_a='pos')
        self.weights = (-inverse / np.diag(inverse)[None, :]).astype(np.float32)
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return np.asarray(self.history[users] @ self.weights)
