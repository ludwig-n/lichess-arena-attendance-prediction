import pathlib
import pickle
import re

import fastapi
import pydantic
import requests
import sklearn.metrics
import sklearn.pipeline
import uvicorn

import preprocessing


MODELS_DIR = pathlib.Path("models")

TOURNAMENT_LINK_OR_ID_PATTERN = re.compile(r"(?:https://lichess.org/tournament/)?(\w+)", flags=re.ASCII)
API_BASE_URL = "https://lichess.org/api/tournament"
REQUEST_TIMEOUT = 5

app = fastapi.FastAPI()


class RegressionMetrics(pydantic.BaseModel):
    r2: float
    mape: float
    mae: float
    mse: float


class PredictTsvResponse(pydantic.BaseModel):
    n_players_pred: list[float]
    n_players_true: list[int]
    metrics: RegressionMetrics | None


class PredictLinkResponse(pydantic.BaseModel):
    n_players_pred: int
    n_players_true: int | None
    name: str
    starts_at: str


def get_model(model_name: str) -> sklearn.pipeline.Pipeline:
    model_path = MODELS_DIR / f"{model_name}.p"
    if not model_path.exists():
        raise fastapi.HTTPException(fastapi.status.HTTP_404_NOT_FOUND, f"Model {model_name} not found")
    with open(model_path, "rb") as fin:
        return pickle.load(fin)


@app.get("/list_models")
def list_models() -> list[str]:
    return [path.stem for path in MODELS_DIR.glob("*.p")]


@app.post("/predict_tsv/{model_name}")
def predict_tsv(model_name: str, tsv_file: fastapi.UploadFile) -> PredictTsvResponse:
    X_test, y_true = preprocessing.read_tsv_with_all_features(tsv_file.file)
    y_pred = get_model(model_name).predict(X_test)

    if y_true is not None:
        metrics = RegressionMetrics(
            r2=sklearn.metrics.r2_score(y_true, y_pred),
            mape=sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred),
            mae=sklearn.metrics.mean_absolute_error(y_true, y_pred),
            mse=sklearn.metrics.mean_squared_error(y_true, y_pred)
        )
    else:
        metrics = None

    return PredictTsvResponse(n_players_pred=y_pred, n_players_true=y_true, metrics=metrics)


@app.post("/predict_link/{model_name}")
def predict_link(model_name: str, tournament_link_or_id: str) -> PredictLinkResponse:
    match = TOURNAMENT_LINK_OR_ID_PATTERN.fullmatch(tournament_link_or_id)
    if match is None:
        raise fastapi.HTTPException(fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid tournament link or id")

    try:
        response = requests.get(f"{API_BASE_URL}/{match.group(1)}", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        if isinstance(e, requests.HTTPError) and e.response.status_code == 404:
            raise fastapi.HTTPException(fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid tournament link or id")
        else:
            raise fastapi.HTTPException(fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR, "Couldn't connect to Lichess")

    df_raw = preprocessing.json_list_to_dataframe([response.text])
    df = preprocessing.add_derived_features(df_raw.drop(columns=["n_players"], errors="ignore"))

    return PredictLinkResponse(
        n_players_pred=round(get_model(model_name).predict(df)[0]),
        n_players_true=df_raw.n_players[0] if response.json().get("isFinished") else None,
        name=df_raw.name[0],
        starts_at=df_raw.starts_at[0]
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8500)
