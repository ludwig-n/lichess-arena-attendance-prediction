import pickle
import random

import gentun.algorithms
import gentun.genes
import gentun.models.base
import gentun.populations
import numpy as np
import sklearn.ensemble

import cv.splitters
import preprocessing


GENES = [
    gentun.genes.RandomLogUniform("learning_rate", 0.01, 1),
    gentun.genes.RandomChoice("max_leaf_nodes", list(2 ** np.arange(2, 9)) + [None]),
    gentun.genes.RandomChoice("max_depth", list(range(3, 10)) + [None]),
    gentun.genes.RandomChoice("min_samples_leaf", range(1, 51)),
    gentun.genes.RandomUniform("max_features", 0.25, 1)
]


class GradientBoostingHandler(gentun.models.base.Handler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        random_state = hash(tuple(kwargs.items())) % (2 ** 32)
        self.model = preprocessing.make_pipeline(
            sklearn.ensemble.HistGradientBoostingRegressor(
                categorical_features=preprocessing.CATEGORICAL_FEATURES, random_state=random_state, **kwargs
            ),
            ohe=False
        )
        self.splitter = cv.splitters.RegressionStratifiedKFoldSplitter(k=5, random_state=random_state)

    def create_train_evaluate(self, x_train, y_train, x_test=None, y_test=None):
        return self.splitter.cross_validate(self.model, x_train, y_train).mean_score


def run_genetic_search(X_train, y_train, save_path, n_individuals=50, n_generations=20, random_seed=27):
    random.seed(random_seed)
    population = gentun.populations.Population(
        genes=GENES,
        handler=GradientBoostingHandler,
        individuals=n_individuals,
        x_train=X_train,
        y_train=y_train
    )
    algorithm = gentun.algorithms.Tournament(population)
    algorithm.run(generations=n_generations)
    with open(save_path, "wb") as fout:
        pickle.dump(population, fout)


def run_grid_search(X_train, y_train, save_path, gene_samples=(4, 4, 4, 4, 4)):
    population = gentun.populations.Grid(
        genes=GENES,
        handler=GradientBoostingHandler,
        gene_samples=gene_samples,
        x_train=X_train,
        y_train=y_train
    )
    population.get_fittest()
    with open(save_path, "wb") as fout:
        pickle.dump(population, fout)


if __name__ == "__main__":
    X_train, y_train = preprocessing.read_tsv_with_all_features("data/tournament_dataset/train.tsv")
    run_genetic_search(X_train, y_train, "tuning/results/genetic_population.p")
    run_grid_search(X_train, y_train, "tuning/results/grid_population.p")
