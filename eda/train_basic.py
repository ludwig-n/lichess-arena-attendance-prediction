import pickle

import sklearn.ensemble
import sklearn.linear_model

import preprocessing


X_train, y_train = preprocessing.read_tsv_with_all_features("data/tournament_dataset/train.tsv")
X_test, y_test = preprocessing.read_tsv_with_all_features("data/tournament_dataset/test.tsv")

regressors = [
    ("ridge_basic", sklearn.linear_model.Ridge(random_state=27)),
    ("random_forest_basic", sklearn.ensemble.RandomForestRegressor(random_state=27, verbose=2))
]

model_dir = "models"
for name, regressor in regressors:
    print(f"fitting {name}")
    pipeline = preprocessing.make_pipeline(regressor).fit(X_train, y_train)
    print(f"{name} score = {pipeline.score(X_test, y_test)}")

    path = f"{model_dir}/{name}.p"
    with open(path, "wb") as fout:
        pickle.dump(pipeline[-1].regressor_, fout, protocol=5)
    print(f"saved to {path}\n")
