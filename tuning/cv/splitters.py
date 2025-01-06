import dataclasses
import math
import typing as tp

import numpy as np
import pandas as pd


@dataclasses.dataclass
class DataSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series


class Splitter:
    def split(self, X: pd.DataFrame, y: pd.Series) -> tp.Generator[DataSplit, None, None]:
        """Generates a DataSplit object for each split used in cross-validation."""
        raise NotImplementedError()    # abstract method


class LeaveOneOutSplitter(Splitter):
    def split(self, X: pd.DataFrame, y: pd.Series) -> tp.Generator[DataSplit, None, None]:
        assert len(X) == len(y), "X and y should be the same length"
        for i in range(len(X)):
            yield DataSplit(X_train=X.drop(i), y_train=y.drop(i), X_val=X.iloc[i:i + 1], y_val=y.iloc[i:i + 1])


class KFoldSplitter(Splitter):
    def __init__(self, k: int, random_state: int | None = None):
        super().__init__()
        self.k = k
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y: pd.Series) -> tp.Generator[DataSplit, None, None]:
        assert len(X) == len(y), "X and y should be the same length"

        fold_id = np.arange(len(X)) % self.k
        np.random.default_rng(self.random_state).shuffle(fold_id)

        for val_fold_id in range(self.k):
            train_idx = fold_id != val_fold_id
            val_idx = fold_id == val_fold_id
            yield DataSplit(X_train=X[train_idx], y_train=y[train_idx], X_val=X[val_idx], y_val=y[val_idx])


class RegressionStratifiedKFoldSplitter(Splitter):
    def __init__(self, k: int, random_state: int | None = None):
        super().__init__()
        self.k = k
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y: pd.Series) -> tp.Generator[DataSplit, None, None]:
        assert len(X) == len(y), "X and y should be the same length"

        sorted_idx = np.argsort(y)[::-1]
        X_sorted = X.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]

        n_blocks = (len(X) - 1) // self.k + 1    # ceil(X / k)
        fold_id = np.arange(n_blocks * self.k) % self.k    # shape: [ceil(X / k) * k]
        fold_id = fold_id.reshape((-1, self.k))            # shape: [ceil(X / k), k]
        fold_id = np.random.default_rng(self.random_state).permuted(fold_id, axis=1)    # shape: [ceil(X / k), k]
        fold_id = fold_id.flatten()    # shape: [ceil(X / k) * k]
        fold_id = fold_id[:len(X)]     # shape: [len(X)]

        for val_fold_id in range(self.k):
            train_idx = fold_id != val_fold_id
            val_idx = fold_id == val_fold_id
            yield DataSplit(
                X_train=X_sorted[train_idx],
                y_train=y_sorted[train_idx],
                X_val=X_sorted[val_idx],
                y_val=y_sorted[val_idx]
            )


class MonteCarloSplitter(Splitter):
    def __init__(self, n_splits: int, val_size: float, random_state: int | None = None):
        super().__init__()
        self.n_splits = n_splits
        self.val_size = val_size
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y: pd.Series) -> tp.Generator[DataSplit, None, None]:
        assert len(X) == len(y), "X and y should be the same length"

        rng = np.random.default_rng(self.random_state)
        val_idx = np.zeros(len(X), dtype=bool)
        val_idx[:math.ceil(len(X) * self.val_size)] = True

        for _ in range(self.n_splits):
            val_idx = rng.permuted(val_idx)
            yield DataSplit(X_train=X[~val_idx], y_train=y[~val_idx], X_val=X[val_idx], y_val=y[val_idx])
