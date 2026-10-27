"""TimeSVD++: временные смещения, дрейф факторов и неявная история.

Реализация по Koren (2009) с дневными смещениями и векторами пользователя.
Для будущих дней необученные дневные параметры принимаются равными нулю.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn


class TimeSVDPlusPlus(nn.Module):
    def __init__(self, factors=16, epochs=8, regularization=1e-4,
                 learning_rate=.005, bin_days=90, beta=.4, temporal=True, seed=42):
        super().__init__()
        self.factors, self.epochs, self.regularization = factors, epochs, regularization
        self.learning_rate, self.bin_days, self.beta = learning_rate, bin_days, beta
        self.temporal, self.seed = temporal, seed

    def _days(self, timestamps):
        return np.asarray((timestamps - self.origin).total_seconds() / 86400, dtype=np.float32)

    def _implicit(self, users):
        history = self.history[users]
        return self.implicit_factors(history).sum(1) / self.history_norm[users, None]

    def _deviation(self, users, days):
        delta = (days - self.mean_day[users]) / 365.25
        return delta.sign() * delta.abs().pow(self.beta)

    def forward(self, users, items, days, day_keys=None):
        vectors = self.user_factors(users) + self._implicit(users)
        bias = self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1)
        if self.temporal:
            dev = self._deviation(users, days)
            vectors = vectors + self.user_drift(users) * dev[:, None]
            bias = bias + self.bias_drift(users).squeeze(1) * dev
            bins = (days / self.bin_days).long().clamp(0, self.n_bins - 1)
            bias = bias + self.item_bin_bias(items * self.n_bins + bins).squeeze(1)
            if day_keys is not None:
                bias = bias + self.user_day_bias(day_keys).squeeze(1)
                vectors = vectors + self.user_day_factors(day_keys)
        return self.global_mean + bias + (vectors * self.item_factors(items)).sum(1)

    def fit(self, train, interactions, cutoff):
        if train.timestamp.isna().any() or (train.timestamp >= cutoff).any():
            raise ValueError('В train timeSVD++ попали события не раньше границы обучения')
        torch.manual_seed(self.seed)
        n_users, self.n_items = interactions.shape
        self.origin = train.timestamp.min().normalize()
        self.score_time = cutoff
        days = self._days(pd.DatetimeIndex(train.timestamp))
        self.n_bins = int(days.max() // self.bin_days) + 1
        users = torch.tensor(train.u.to_numpy(), dtype=torch.long)
        items = torch.tensor(train.i.to_numpy(), dtype=torch.long)
        time_tensor = torch.tensor(days)
        ratings = torch.tensor(train.rating.to_numpy(), dtype=torch.float32)
        self.register_buffer('global_mean', ratings.mean())
        means = np.bincount(train.u, weights=days, minlength=n_users) / np.bincount(train.u, minlength=n_users)
        self.register_buffer('mean_day', torch.tensor(means, dtype=torch.float32))
        history = interactions.tocsr()
        degrees = np.diff(history.indptr)
        padded = np.full((n_users, degrees.max()), self.n_items, dtype=np.int64)
        for user in range(n_users):
            padded[user, :degrees[user]] = history[user].indices
        self.register_buffer('history', torch.tensor(padded))
        self.register_buffer('history_norm', torch.tensor(np.sqrt(degrees), dtype=torch.float32))
        # Дневные остатки обучаются только для наблюдавшихся в train дней.
        day_pairs = pd.MultiIndex.from_arrays([train.u.to_numpy(), np.floor(days).astype(int)])
        keys, unique_days = pd.factorize(day_pairs, sort=True)
        day_keys = torch.tensor(keys + 1, dtype=torch.long)
        self.user_factors = nn.Embedding(n_users, self.factors)
        self.item_factors = nn.Embedding(self.n_items, self.factors)
        self.implicit_factors = nn.Embedding(self.n_items + 1, self.factors, padding_idx=self.n_items)
        self.user_bias, self.item_bias = nn.Embedding(n_users, 1), nn.Embedding(self.n_items, 1)
        self.user_drift, self.bias_drift = nn.Embedding(n_users, self.factors), nn.Embedding(n_users, 1)
        self.item_bin_bias = nn.Embedding(self.n_items * self.n_bins, 1)
        self.user_day_bias = nn.Embedding(len(unique_days) + 1, 1, padding_idx=0)
        for layer in [self.user_factors, self.item_factors, self.implicit_factors]:
            nn.init.normal_(layer.weight, std=.01)
        with torch.no_grad():
            self.implicit_factors.weight[self.n_items].zero_()
        for layer in [self.user_bias, self.item_bias, self.user_drift,
                      self.bias_drift, self.item_bin_bias, self.user_day_bias]:
            nn.init.zeros_(layer.weight)
        # Нулевой ключ соответствует дню без обученной истории.
        # Сохраняем зерно для одинакового порядка батчей у временной и статической моделей.
        with torch.random.fork_rng(devices=[]):
            self.user_day_factors = nn.Embedding(len(unique_days) + 1, self.factors, padding_idx=0)
            nn.init.zeros_(self.user_day_factors.weight)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.regularization)
        self.loss_history = []
        for _ in range(self.epochs):
            total = 0
            for batch in torch.randperm(len(train)).split(2048):
                optimizer.zero_grad()
                prediction = self(users[batch], items[batch], time_tensor[batch], day_keys[batch])
                loss = nn.functional.mse_loss(prediction, ratings[batch])
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            self.loss_history.append(total / len(train))
        return self

    @torch.no_grad()
    def score(self, users):
        if self.score_time <= self.origin:
            raise ValueError('Время рекомендации предшествует обучающей истории')
        u = torch.as_tensor(users, dtype=torch.long)
        day = float((self.score_time - self.origin).total_seconds() / 86400)
        vectors = self.user_factors(u) + self._implicit(u)
        ub = self.user_bias(u).squeeze(1)
        item_ids = torch.arange(self.n_items)
        ib = self.item_bias(item_ids).squeeze(1)
        if self.temporal:
            dev = self._deviation(u, torch.full((len(u),), day))
            vectors = vectors + self.user_drift(u) * dev[:, None]
            ub = ub + self.bias_drift(u).squeeze(1) * dev
            bin_id = min(max(int(day // self.bin_days), 0), self.n_bins - 1)
            ib = ib + self.item_bin_bias(item_ids * self.n_bins + bin_id).squeeze(1)
        return (self.global_mean + ub[:, None] + ib[None, :] +
                vectors @ self.item_factors.weight.T).numpy()
