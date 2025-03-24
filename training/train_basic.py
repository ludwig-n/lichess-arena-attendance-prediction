import pickle

import sklearn.ensemble
import sklearn.linear_model

import preprocessing


X_train, y_train = preprocessing.read_tsv_with_all_features("data/tournament_dataset/train.tsv")
X_test, y_test = preprocessing.read_tsv_with_all_features("data/tournament_dataset/test.tsv")

regressors = [
    (
        "random_forest_basic",
        preprocessing.make_pipeline(sklearn.ensemble.RandomForestRegressor(random_state=27))
    ),
    (
        "hist_gradient_boosting_basic",
        preprocessing.make_pipeline(
            sklearn.ensemble.HistGradientBoostingRegressor(
                categorical_features=preprocessing.CATEGORICAL_FEATURES, random_state=27
            ),
            ohe=False
        )
    )
]

model_dir = "models"
for name, regressor in regressors:
    print(f"fitting {name}")
    regressor.fit(X_train, y_train)
    print(f"{name} score = {regressor.score(X_test, y_test)}")

    path = f"{model_dir}/{name}.p"
    with open(path, "wb") as fout:
        pickle.dump(regressor, fout, protocol=5)
    print(f"saved to {path}\n")
