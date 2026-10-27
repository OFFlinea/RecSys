import numpy as np


class ImplicitALS:
    """ALS для неявной обратной связи с L2-регуляризацией факторов."""

    def __init__(self, factors=16, regularization=10.0, alpha=20.0,
                 iterations=6, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha, self.iterations, self.seed = alpha, iterations, seed

    def _update(self, targets, fixed):
        out = np.zeros((targets.shape[0], self.factors), dtype=np.float32)
        base = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        degrees = np.diff(targets.indptr)
        for degree in np.unique(degrees):
            if degree == 0:
                continue
            group = np.flatnonzero(degrees == degree)
            if degree == 1:
                neighbours = np.unique(targets.indices[targets.indptr[group]])
                projected = np.linalg.solve(base, fixed[neighbours].T).T
                for start in range(0, len(group), 512):
                    rows = group[start:start + 512]
                    offsets = targets.indptr[rows]
                    indices = targets.indices[offsets]
                    solved = projected[np.searchsorted(neighbours, indices)]
                    denominator = 1 + self.alpha * np.einsum('bi,bi->b', fixed[indices], solved)
                    out[rows] = solved * ((1 + self.alpha) * targets.data[offsets] / denominator)[:, None]
                continue
            for start in range(0, len(group), 512):
                rows = group[start:start + 512]
                offsets = targets.indptr[rows, None] + np.arange(degree)
                vectors = fixed[targets.indices[offsets]]
                preferences = targets.data[offsets]
                lhs = base[None] + self.alpha * np.einsum('bdi,bdj->bij', vectors, vectors)
                rhs = (1 + self.alpha) * np.einsum('bd,bdi->bi', preferences, vectors)
                out[rows] = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        return out

    def fit(self, interactions):
        targets = interactions.copy().tocsr().astype(np.float32)
        targets.data[:] = 1
        targets.sort_indices()
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, .01, (targets.shape[1], self.factors)).astype('float32')
        for _ in range(self.iterations):
            self.users = self._update(targets, self.items)
            self.items = self._update(targets.T.tocsr(), self.users)
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
