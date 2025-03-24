import pickle

import sklearn.ensemble

import preprocessing


X_train, y_train = preprocessing.read_tsv_with_all_features("data/tournament_dataset/train.tsv")
X_test, y_test = preprocessing.read_tsv_with_all_features("data/tournament_dataset/test.tsv")

regressor = preprocessing.make_pipeline(
    sklearn.ensemble.HistGradientBoostingRegressor(
        # Best hyperparameters found by the genetic algorithm
        learning_rate=0.362595,
        max_leaf_nodes=None,
        max_depth=9,
        min_samples_leaf=1,
        max_features=0.9091,
        categorical_features=preprocessing.CATEGORICAL_FEATURES,
        random_state=27
    ),
    ohe=False
)

regressor.fit(X_train, y_train)
print(f"score = {regressor.score(X_test, y_test)}")

path = "models/hist_gradient_boosting_tuned.p"
with open(path, "wb") as fout:
    pickle.dump(regressor, fout, protocol=5)
print(f"saved to {path}")
