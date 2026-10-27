import numpy as np
import torch
from torch import nn


class FactorizationMachine(nn.Module):
    """FM второго порядка с ID, агрегатами истории и метаданными товаров."""

    def __init__(self, factors=16, epochs=5, learning_rate=.01,
                 regularization=1e-5, use_metadata=True, seed=42):
        super().__init__()
        self.factors, self.epochs = factors, epochs
        self.learning_rate, self.regularization = learning_rate, regularization
        self.use_metadata, self.seed = use_metadata, seed

    def forward(self, users, items):
        embeddings = [self.user_embedding(users), self.item_embedding(items)]
        linear = self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1) + self.bias
        if self.use_metadata:
            numeric = torch.cat([self.user_features[users], self.item_features[items]], dim=1)
            embeddings.append(numeric @ self.feature_embedding)
            sum_squared = (numeric.square() @ self.feature_embedding.square()).sum(dim=1)
            linear = linear + numeric @ self.feature_linear
        else:
            sum_squared = 0
        total = sum(embeddings)
        squared = embeddings[0].square().sum(1) + embeddings[1].square().sum(1) + sum_squared
        return linear + .5 * (total.square().sum(1) - squared)

    def fit(self, interactions, user_features, item_features, observed_targets=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_users, n_items = interactions.shape
        self.user_embedding = nn.Embedding(n_users, self.factors)
        self.item_embedding = nn.Embedding(n_items, self.factors)
        self.user_bias, self.item_bias = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer('user_features', torch.tensor(user_features, dtype=torch.float32))
        self.register_buffer('item_features', torch.tensor(item_features, dtype=torch.float32))
        n_features = user_features.shape[1] + item_features.shape[1]
        self.feature_embedding = nn.Parameter(torch.empty(n_features, self.factors))
        self.feature_linear = nn.Parameter(torch.zeros(n_features))
        for p in [self.user_embedding.weight, self.item_embedding.weight, self.feature_embedding]:
            nn.init.normal_(p, std=.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.regularization)
        positive_users, positive_items = interactions.nonzero()
        if observed_targets is None:
            observed_labels = np.ones(len(positive_users), dtype=np.float32)
        else:
            if observed_targets.shape != interactions.shape:
                raise ValueError('Размер матрицы обучающих меток не совпадает с взаимодействиями')
            observed_labels = np.asarray(observed_targets[positive_users, positive_items]).ravel().astype(np.float32)
            if not np.isfinite(observed_labels).all() or not np.all((observed_labels >= 0) & (observed_labels <= 1)):
                raise ValueError('Метки FM должны быть конечными числами от 0 до 1')
        negative_items = rng.integers(n_items, size=len(positive_users))
        invalid = np.asarray(interactions[positive_users, negative_items]).ravel() > 0
        while invalid.any():
            negative_items[invalid] = rng.integers(n_items, size=invalid.sum())
            invalid = np.asarray(interactions[positive_users, negative_items]).ravel() > 0
        users = torch.tensor(np.tile(positive_users, 2), dtype=torch.long)
        items = torch.tensor(np.concatenate([positive_items, negative_items]), dtype=torch.long)
        labels = torch.cat([torch.tensor(observed_labels), torch.zeros(len(positive_users))])
        self.loss_history = []
        batch_generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            order = torch.randperm(len(users), generator=batch_generator)
            total_loss = 0
            for batch in order.split(4096):
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(self(users[batch], items[batch]), labels[batch])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.loss_history.append(total_loss / len(users))
        return self

    @torch.no_grad()
    def score(self, users):
        u = torch.as_tensor(users, dtype=torch.long)
        item_ids = torch.arange(self.item_embedding.num_embeddings)
        uv, iv = self.user_embedding(u), self.item_embedding(item_ids)
        ub, ib = self.user_bias(u).squeeze(1), self.item_bias(item_ids).squeeze(1)
        if self.use_metadata:
            split = self.user_features.shape[1]
            uf, itf = self.user_features[u], self.item_features
            ue = uf @ self.feature_embedding[:split]
            ie = itf @ self.feature_embedding[split:]
            ub = ub + uf @ self.feature_linear[:split] + (uv * ue).sum(1)
            ib = ib + itf @ self.feature_linear[split:] + (iv * ie).sum(1)
            ub = ub + .5 * (ue.square().sum(1) - (uf.square() @ self.feature_embedding[:split].square()).sum(1))
            ib = ib + .5 * (ie.square().sum(1) - (itf.square() @ self.feature_embedding[split:].square()).sum(1))
            uv, iv = uv + ue, iv + ie
        return (uv @ iv.T + ub[:, None] + ib[None, :] + self.bias).numpy()
