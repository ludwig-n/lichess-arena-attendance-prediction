import pickle
import random

import gentun.algorithms
import gentun.genes
import gentun.models.base
import gentun.populations
import numpy as np
import pandas as pd
import sklearn.ensemble

import cv.splitters
import preprocessing


def hash_dict(dct):
    return hash(tuple(dct.items())) % (2 ** 32)


class GradientBoostingHandler(gentun.models.base.Handler):
    def __init__(self, n_folds=5, **kwargs):
        super().__init__(**kwargs)
        random_state = hash_dict(kwargs)
        self.model = preprocessing.make_pipeline(
            sklearn.ensemble.HistGradientBoostingRegressor(
                categorical_features=preprocessing.CATEGORICAL_FEATURES, random_state=random_state, **kwargs
            ),
            ohe=False
        )
        self.splitter = cv.splitters.RegressionStratifiedKFoldSplitter(k=n_folds, random_state=random_state)

    def create_train_evaluate(self, x_train, y_train, x_test=None, y_test=None):
        return self.splitter.cross_validate(self.model, x_train, y_train).mean_score


class RandomForestHandler(gentun.models.base.Handler):
    def __init__(self, n_folds=5, **kwargs):
        super().__init__(**kwargs)
        random_state = hash_dict(kwargs)
        self.model = preprocessing.make_pipeline(
            sklearn.ensemble.RandomForestRegressor(random_state=random_state, **kwargs)
        )
        self.splitter = cv.splitters.RegressionStratifiedKFoldSplitter(k=n_folds, random_state=random_state)

    def create_train_evaluate(self, x_train, y_train, x_test=None, y_test=None):
        return self.splitter.cross_validate(self.model, x_train, y_train).mean_score


def save_population(population, pickle_path=None, csv_path=None):
    if pickle_path is not None:
        with open(pickle_path, "wb") as fout:
            pickle.dump(population, fout)

    if csv_path is not None:
        rows = [ind.hyperparameters | {"fitness": ind.fitness} for ind in population]
        pd.DataFrame(rows).sort_values("fitness", ascending=False).to_csv(csv_path, index=None)


def run_genetic_search(
    handler, genes, X_train, y_train, n_individuals, n_generations,
    common_params=None, pickle_path=None, csv_path=None, random_seed=27
):
    random.seed(random_seed)
    if common_params is None:
        common_params = {}
    population = gentun.populations.Population(
        genes=genes,
        handler=handler,
        individuals=n_individuals,
        x_train=X_train,
        y_train=y_train,
        **common_params
    )
    algorithm = gentun.algorithms.Tournament(population)
    algorithm.run(generations=n_generations)
    save_population(algorithm.population, pickle_path=pickle_path, csv_path=csv_path)


def run_grid_search(
    handler, genes, X_train, y_train, gene_samples,
    common_params=None, pickle_path=None, csv_path=None
):
    if common_params is None:
        common_params = {}
    population = gentun.populations.Grid(
        genes=genes,
        handler=handler,
        gene_samples=gene_samples,
        x_train=X_train,
        y_train=y_train,
        **common_params
    )
    population.get_fittest()
    save_population(population, pickle_path=pickle_path, csv_path=csv_path)


if __name__ == "__main__":
    X_train, y_train = preprocessing.read_tsv_with_all_features("data/tournament_dataset/train.tsv")
    model_type = "random_forest"

    if model_type == "gradient_boosting":
        genes = [
            gentun.genes.RandomLogUniform("learning_rate", 0.01, 1),
            gentun.genes.RandomChoice("max_leaf_nodes", list(2 ** np.arange(2, 9)) + [None]),
            gentun.genes.RandomChoice("max_depth", list(range(3, 10)) + [None]),
            gentun.genes.RandomChoice("min_samples_leaf", range(1, 51)),
            gentun.genes.RandomUniform("max_features", 0.25, 1)
        ]
        run_genetic_search(
            GradientBoostingHandler, genes, X_train, y_train,
            n_individuals=50, n_generations=20, pickle_path="tuning/results/genetic_population.p"
        )
        run_grid_search(
            GradientBoostingHandler, genes, X_train, y_train,
            gene_samples=(4, 4, 4, 4, 4), pickle_path="tuning/results/grid_population.p"
        )

    elif model_type == "random_forest":
        genes_genetic = [
            gentun.genes.RandomUniform("max_features", 0.05, 1),
            gentun.genes.RandomChoice("min_samples_leaf", range(1, 51)),
            gentun.genes.RandomChoice("max_depth", [5, 10, 20, 30, 40, None])
        ]
        run_genetic_search(
            RandomForestHandler, genes_genetic, X_train, y_train,
            n_individuals=25, n_generations=10, pickle_path="tuning/results/rf_genetic_population.p"
        )

        genes_grid = [
            gentun.genes.RandomChoice("max_features", [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
            gentun.genes.RandomChoice("max_depth", [20, 30, 40, None])
        ]
        common_params_grid = {"n_estimators": 50}
        run_grid_search(
            RandomForestHandler, genes_grid, X_train, y_train,
            gene_samples=(7, 4),
            common_params=common_params_grid,
            pickle_path="tuning/results/rf_grid_population.p",
            csv_path="tuning/results/rf_grid_table.csv"
        )

        # Looking for a good number of trees without losing too much quality

        genes_grid_n_estimators = [
            gentun.genes.RandomChoice("n_estimators", range(10, 110, 10)),
            gentun.genes.RandomChoice("max_depth", [20, None])
        ]
        run_grid_search(
            RandomForestHandler, genes_grid_n_estimators, X_train, y_train,
            gene_samples=(10, 2),
            pickle_path="tuning/results/rf_grid_n_estimators_population.p",
            csv_path="tuning/results/rf_grid_n_estimators_table.csv"
        )
