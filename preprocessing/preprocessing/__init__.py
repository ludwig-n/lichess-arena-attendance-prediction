import json

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.compose
import sklearn.pipeline
import sklearn.preprocessing


def json_list_to_dataframe(jsons: list[str]) -> pd.DataFrame:
    rows = []
    for json_ in jsons:
        dct = json.loads(json_)
        rows.append({
            "id": dct["id"],
            "n_players": dct["nbPlayers"],
            "name": dct["fullName"],

            "starts_at": dct["startsAt"],
            "duration_mins": dct["minutes"],

            "variant": dct["variant"],
            "perf": dct["perf"]["key"],
            "freq": dct["schedule"]["freq"],

            "speed": dct["schedule"]["speed"],
            "clock_limit_secs": dct["clock"]["limit"],
            "clock_increment_secs": dct["clock"]["increment"],

            "rated": dct["rated"],
            "berserkable": dct.get("berserkable", False),
            "only_titled": dct.get("onlyTitled", False),
            "is_team": "teamBattle" in dct,

            "max_rating": dct.get("maxRating", {}).get("rating"),
            "min_rating": dct.get("minRating", {}).get("rating"),
            "min_rated_games": dct.get("minRatedGames", {}).get("nb"),

            "position_fen": dct.get("position", {}).get("fen"),
            "position_opening_name": dct.get("position", {}).get("name"),
            "position_opening_eco": dct.get("position", {}).get("eco"),

            "headline": dct.get("spotlight", {}).get("headline"),
            "description": dct["description"].replace("\n", " ").replace("\r", "") if "description" in dct else None
        })
    return pd.DataFrame(rows)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start_time = pd.to_datetime(df.starts_at, format="ISO8601").dt.round("30min")
    df["starts_at_month"] = start_time.dt.month
    df["starts_at_weekday"] = start_time.dt.weekday
    df["starts_at_hour"] = start_time.dt.hour

    duration = pd.to_timedelta(df.duration_mins, "minutes")
    end_time = (start_time + duration).dt.round("30min")
    same_day = start_time.dt.day == end_time.dt.day

    start_half_hour = start_time.dt.hour * 2 + start_time.dt.minute // 30
    end_half_hour = end_time.dt.hour * 2 + end_time.dt.minute // 30

    for hour in range(24):
        for minute in [0, 30]:
            col_name = f"open_at_{str(hour).zfill(2)}h{str(minute).zfill(2)}m"
            half_hour = hour * 2 + minute // 30
            after_start = start_half_hour <= half_hour
            before_end = half_hour <= end_half_hour
            df[col_name] = (same_day & after_start & before_end) | (~same_day & (after_start | before_end))

    df["has_clock_increment"] = df.clock_increment_secs > 0
    df["has_max_rating"] = df.max_rating.notna()
    df["has_min_rating"] = df.min_rating.notna()
    df["has_min_rated_games"] = df.min_rated_games.notna()
    df["has_custom_position"] = df.position_fen.notna()
    df["has_headline"] = df.headline.notna()
    df["has_description"] = df.description.notna()
    df["has_prizes"] = df.description.str.contains("Prizes:", na=False)

    return df


def read_tsv_with_all_features(path: str) -> tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(path, sep="\t")
    if "n_players" in df.columns:
        X, y = df.drop(columns="n_players"), df.n_players
    else:
        X, y = df, None
    return add_derived_features(X), y


def custom_scale_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    min_rating_norm = (df.min_rating.astype(float) - 1500) / 300
    max_rating_norm = (df.max_rating.astype(float) - 1500) / 300
    return (df
        .assign(
            min_rating=1 / (1 + np.exp(-min_rating_norm)),
            max_rating=1 / (1 + np.exp(-max_rating_norm))
        )
        .fillna({
            "min_rating": 0,
            "max_rating": 1,
            "min_rated_games": 0
        })
    )


def make_pipeline(
    regressor: sklearn.base.RegressorMixin | None,
    ohe: bool = True,
    scale: bool = True
) -> sklearn.pipeline.Pipeline:
    steps = []
    if scale:
        steps.append(sklearn.preprocessing.FunctionTransformer(custom_scale_and_fill, feature_names_out="one-to-one"))
    steps.append(
        sklearn.compose.make_column_transformer(
            (
                sklearn.preprocessing.OneHotEncoder(sparse_output=False) if ohe else "passthrough",
                ["variant", "perf", "freq", "speed", "starts_at_month", "starts_at_weekday", "starts_at_hour"]
            ),
            (
                sklearn.preprocessing.MaxAbsScaler() if scale else "passthrough",
                ["duration_mins", "clock_limit_secs", "clock_increment_secs", "min_rated_games"]
            ),
            (
                "passthrough",
                sklearn.compose.make_column_selector(dtype_include=bool)
            ),
            (
                "passthrough",
                ["min_rating", "max_rating"]    # already scaled to [0, 1] in custom_scale_and_fill if scale=True
            ),
            verbose_feature_names_out=False
        )
    )
    if regressor is not None:
        steps.append(sklearn.compose.TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp))
    pipeline = sklearn.pipeline.make_pipeline(*steps)
    pipeline.set_output(transform="pandas")
    return pipeline
