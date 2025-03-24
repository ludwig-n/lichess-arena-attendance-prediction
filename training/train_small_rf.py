import pickle

import sklearn.ensemble

import preprocessing


X_train, y_train = preprocessing.read_tsv_with_all_features("data/tournament_dataset/train.tsv")
X_test, y_test = preprocessing.read_tsv_with_all_features("data/tournament_dataset/test.tsv")

regressor = preprocessing.make_pipeline(
    sklearn.ensemble.RandomForestRegressor(
        # Hyperparameters chosen to produce a smaller model without a big drop in quality
        n_estimators=20,
        max_depth=20,
        random_state=28
    )
)

regressor.fit(X_train, y_train)
print(f"score = {regressor.score(X_test, y_test)}")

path = "models/random_forest_small.p"
with open(path, "wb") as fout:
    pickle.dump(regressor, fout, protocol=5)
print(f"saved to {path}")
