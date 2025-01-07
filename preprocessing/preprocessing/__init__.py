import os
import typing as tp

import lime.lime_tabular
import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
import sklearn.base
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing


class RawDatasetSchema(pa.DataFrameModel):
    id: str
    n_players: tp.Optional[int]
    name: str

    starts_at: str
    duration_mins: int

    variant: str
    perf: str
    freq: str

    speed: str
    clock_limit_secs: int
    clock_increment_secs: int

    rated: bool
    berserkable: bool
    only_titled: bool
    is_team: bool

    # Semantically these are ints but regular ints aren't nullable in Pandas so we consider them floats
    max_rating: float = pa.Field(nullable=True, coerce=True)
    min_rating: float = pa.Field(nullable=True, coerce=True)
    min_rated_games: float = pa.Field(nullable=True, coerce=True)

    position_fen: str = pa.Field(nullable=True, coerce=True)
    position_opening_name: str = pa.Field(nullable=True, coerce=True)
    position_opening_eco: str = pa.Field(nullable=True, coerce=True)

    headline: str = pa.Field(nullable=True, coerce=True)
    description: str = pa.Field(nullable=True, coerce=True)

    class Config:
        strict = True


# All features used for training (including derived features).
CATEGORICAL_FEATURES = [
    "variant", "perf", "freq", "speed", "starts_at_month", "starts_at_weekday", "starts_at_hour", "ends_at_hour"
]
BINARY_FEATURES = [
    "rated", "berserkable", "only_titled", "is_team",
    "has_clock_increment", "has_max_rating", "has_min_rating", "has_min_rated_games",
    "has_custom_position", "has_description", "has_prizes"
]
NUMERIC_FEATURES = [
    # min_rating is not useful with the current dataset
    "duration_mins", "clock_limit_secs", "clock_increment_secs", "min_rated_games", "max_rating"
]


@pa.check_types
def api_objects_to_dataframe(objects: list[dict[str, tp.Any]]) -> pat.DataFrame[RawDatasetSchema]:
    rows = []
    for obj in objects:
        rows.append({
            "id": obj["id"],
            "n_players": obj["nbPlayers"],
            "name": obj["fullName"],

            "starts_at": obj["startsAt"],
            "duration_mins": obj["minutes"],

            "variant": obj["variant"],
            "perf": obj["perf"]["key"],
            "freq": obj["schedule"]["freq"],

            "speed": obj["schedule"]["speed"],
            "clock_limit_secs": obj["clock"]["limit"],
            "clock_increment_secs": obj["clock"]["increment"],

            "rated": obj["rated"],
            "berserkable": obj.get("berserkable", False),
            "only_titled": obj.get("onlyTitled", False),
            "is_team": "teamBattle" in obj,

            "max_rating": obj.get("maxRating", {}).get("rating"),
            "min_rating": obj.get("minRating", {}).get("rating"),
            "min_rated_games": obj.get("minRatedGames", {}).get("nb"),

            "position_fen": obj.get("position", {}).get("fen"),
            "position_opening_name": obj.get("position", {}).get("name"),
            "position_opening_eco": obj.get("position", {}).get("eco"),

            "headline": obj.get("spotlight", {}).get("headline"),
            "description": obj["description"].replace("\n", " ").replace("\r", "") if "description" in obj else None
        })
    return pd.DataFrame(rows)


@pa.check_types
def add_derived_features(df: pat.DataFrame[RawDatasetSchema]) -> pd.DataFrame:
    df = df.copy()
    start_time = pd.to_datetime(df.starts_at, format="ISO8601").dt.round("30min")
    df["starts_at_month"] = start_time.dt.month
    df["starts_at_weekday"] = start_time.dt.weekday
    df["starts_at_hour"] = start_time.dt.hour

    duration = pd.to_timedelta(df.duration_mins, "minutes")
    end_time = (start_time + duration).dt.round("30min")
    df["ends_at_hour"] = end_time.dt.hour

    df["has_clock_increment"] = df.clock_increment_secs > 0
    df["has_max_rating"] = df.max_rating.notna()
    df["has_min_rating"] = df.min_rating.notna()
    df["has_min_rated_games"] = df.min_rated_games.notna()
    df["has_custom_position"] = df.position_fen.notna()
    df["has_description"] = df.description.notna()
    df["has_prizes"] = df.description.str.contains("Prizes:", na=False)

    return df


def read_tsv_with_all_features(path: str | os.PathLike | tp.IO) -> tuple[pd.DataFrame, pd.Series | None]:
    df = RawDatasetSchema.validate(pd.read_csv(path, sep="\t"))
    if "n_players" in df.columns:
        X, y = df.drop(columns="n_players"), df.n_players
    else:
        X, y = df, None
    return add_derived_features(X), y


def ratings_to_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sigmoid heuristic to transform Lichess ratings to approximate playerbase quantiles.
    If ratings are already passed as quantiles, leaves them unchanged.
    Fills NAs in min_rating and max_rating as appropriate.
    """
    for col in ["min_rating", "max_rating"]:
        if col in df.columns:
            if df[col].max() > 1:
                norm = (df[col] - 1500) / 300
                df = df.assign(**{col: 1 / (1 + np.exp(-norm))})
            df = df.fillna({col: 1 if col == "max_rating" else 0})
    return df


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna({"min_rated_games": 0})


def make_pipeline(
    regressor: sklearn.base.RegressorMixin | None,
    ohe: bool = True,
    scale_numeric: bool = True
) -> sklearn.pipeline.Pipeline:
    one_hot_encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
    max_abs_scaler = sklearn.preprocessing.MaxAbsScaler()
    rating_scaler = sklearn.preprocessing.FunctionTransformer(ratings_to_quantiles, feature_names_out="one-to-one")
    na_filler = sklearn.preprocessing.FunctionTransformer(fill_na, feature_names_out="one-to-one")

    steps = [
        sklearn.compose.make_column_transformer(
            (
                one_hot_encoder if ohe else "passthrough",
                CATEGORICAL_FEATURES
            ),
            (
                max_abs_scaler if scale_numeric else "passthrough",
                [col for col in NUMERIC_FEATURES if col != "max_rating"]
            ),
            (
                rating_scaler,
                ["max_rating"]
            ),
            (
                "passthrough",
                BINARY_FEATURES
            ),
            verbose_feature_names_out=False
        ),
        na_filler
    ]
    if regressor is not None:
        steps.append(sklearn.compose.TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp))

    pipeline = sklearn.pipeline.make_pipeline(*steps)
    pipeline.set_output(transform="pandas")
    return pipeline


class PartialOrdinalEncoder(sklearn.base.OneToOneFeatureMixin, sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """
    Applies OrdinalEncoder to only *some* columns in a dataset. This is used in LimeWrapper.
    For some reason sklearn does not natively support this if you need the inverse transformation too.
    """
    def __init__(self, columns: list[str]):
        self.all_columns = []
        self.encoders = {col: sklearn.preprocessing.OrdinalEncoder() for col in columns}
        self.set_output(transform="pandas")

    def fit(self, X: pd.DataFrame, y=None) -> tp.Self:
        self.all_columns = X.columns
        for col, encoder in self.encoders.items():
            encoder.fit(X[[col]])
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_encoded = X.copy()
        for col, encoder in self.encoders.items():
            X_encoded[col] = encoder.transform(X_encoded[[col]]).flatten()
        return X_encoded

    def inverse_transform(self, X: np.ndarray | pd.DataFrame, y=None) -> pd.DataFrame:
        X_decoded = pd.DataFrame(X, columns=self.all_columns, copy=True)
        for col, encoder in self.encoders.items():
            X_decoded[col] = encoder.inverse_transform(X_decoded[[col]]).flatten()
        return X_decoded


class LimeWrapper:
    def __init__(self, X_train: pd.DataFrame):
        # Just scale ratings, fill NA and drop unused columns.
        # All of these transformations are idempotent (see ratings_to_quantiles)
        # so it's ok to later do them again when predicting with the standard pipeline in predict_fn.
        self.input_fixer = make_pipeline(regressor=None, ohe=False, scale_numeric=False)
        X_train_fixed = self.input_fixer.fit_transform(X_train)

        # Encode categorical & binary features as integers. This transformation is reversed in predict_fn.
        self.encoder = PartialOrdinalEncoder(CATEGORICAL_FEATURES + BINARY_FEATURES)
        X_train_encoded = self.encoder.fit_transform(X_train_fixed)

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_encoded.to_numpy(),
            mode="regression",
            feature_names=X_train_encoded.columns,
            categorical_features=[
                X_train_encoded.columns.get_loc(feat)
                for feat in CATEGORICAL_FEATURES + BINARY_FEATURES
            ],
            categorical_names={
                X_train_encoded.columns.get_loc(feat): self.encoder.encoders[feat].categories_[0]
                for feat in CATEGORICAL_FEATURES + BINARY_FEATURES
            },
            discretize_continuous=False,
            random_state=27
        )

    def explain_instance(
        self, pipeline: sklearn.pipeline.Pipeline, one_row_df: pd.DataFrame, **kwargs
    ) -> lime.lime_tabular.explanation.Explanation:
        return self.explainer.explain_instance(
            data_row=self.encoder.transform(self.input_fixer.transform(one_row_df)).to_numpy().flatten(),
            predict_fn=lambda X: pipeline.predict(self.encoder.inverse_transform(X)),
            **kwargs
        )
