import dataclasses
import math
import typing as tp

import numpy as np
import pandas as pd
import sklearn.base


class DataSplit:
    def __init__(self, X: pd.DataFrame, y: pd.Series, val_idx: np.ndarray):
        self.X = X
        self.y = y
        self.val_idx = val_idx

    @property
    def X_train(self):
        return self.X[~self.val_idx]

    @property
    def y_train(self):
        return self.y[~self.val_idx]

    @property
    def X_val(self):
        return self.X[self.val_idx]

    @property
    def y_val(self):
        return self.y[self.val_idx]

    @property
    def train_idx(self):
        return ~self.val_idx


@dataclasses.dataclass
class CrossValidationResult:
    mean_score: float
    split_scores: list[float]


class Splitter:
    def split(self, X: pd.DataFrame, y: pd.Series) -> tp.Generator[DataSplit, None, None]:
        """Generates a DataSplit object for each split used in cross-validation."""
        raise NotImplementedError()    # abstract method

    def cross_validate(
        self, estimator: sklearn.base.BaseEstimator, X: pd.DataFrame, y: pd.Series, scorer: tp.Callable | None = None
    ) -> CrossValidationResult:
        scores = []
        for split in self.split(X, y):
            estimator = sklearn.base.clone(estimator).fit(split.X_train, split.y_train)
            if scorer is None:
                scores.append(estimator.score(split.X_val, split.y_val))
            else:
                scores.append(scorer(split.y_val, estimator.predict(split.X_val)))
        return CrossValidationResult(mean_score=np.mean(scores), split_scores=scores)


class LeaveOneOutSplitter(Splitter):
    def split(self, X: pd.DataFrame, y: pd.Series) -> tp.Generator[DataSplit, None, None]:
        assert len(X) == len(y), "X and y should be the same length"
        arange = np.arange(len(X))
        for i in range(len(X)):
            yield DataSplit(X, y, arange == i)


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
            yield DataSplit(X, y, fold_id == val_fold_id)


class RegressionStratifiedKFoldSplitter(Splitter):
    def __init__(self, k: int, random_state: int | None = None):
        super().__init__()
        self.k = k
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y: pd.Series) -> tp.Generator[DataSplit, None, None]:
        assert len(X) == len(y), "X and y should be the same length"

        sorted_idx = np.argsort(y)[::-1]

        n_blocks = (len(X) - 1) // self.k + 1    # ceil(X / k)
        fold_id = np.arange(n_blocks * self.k) % self.k    # shape: [ceil(X / k) * k]
        fold_id = fold_id.reshape((-1, self.k))            # shape: [ceil(X / k), k]
        fold_id = np.random.default_rng(self.random_state).permuted(fold_id, axis=1)    # shape: [ceil(X / k), k]
        fold_id = fold_id.flatten()    # shape: [ceil(X / k) * k]
        fold_id = fold_id[:len(X)]     # shape: [len(X)]

        fold_id_unsorted = np.empty_like(fold_id)
        fold_id_unsorted[sorted_idx] = fold_id

        for val_fold_id in range(self.k):
            yield DataSplit(X, y, fold_id_unsorted == val_fold_id)


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
            yield DataSplit(X, y, rng.permuted(val_idx))
