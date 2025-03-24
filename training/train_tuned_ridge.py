import pickle

import sklearn.linear_model

import preprocessing


X_train, y_train = preprocessing.read_tsv_with_all_features("data/tournament_dataset/train.tsv")
X_test, y_test = preprocessing.read_tsv_with_all_features("data/tournament_dataset/test.tsv")

regressor = preprocessing.make_pipeline(
    sklearn.linear_model.Ridge(
        # Alpha chosen with grid search
        alpha=31.6,
        random_state=29
    )
)

regressor.fit(X_train, y_train)
print(f"score = {regressor.score(X_test, y_test)}")

path = "models/ridge_tuned.p"
with open(path, "wb") as fout:
    pickle.dump(regressor, fout, protocol=5)
print(f"saved to {path}")
