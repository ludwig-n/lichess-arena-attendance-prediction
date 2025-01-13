## Web app

This web app can be used to try out the models in a friendly interface. It consists of a server, written in FastAPI, and a client, written in Streamlit.

### Running the app

After setting up and activating the project virtual environment (see [`README.md`](../README.md)), run the following commands from the project root folder:
- `python app/server/main.py` to start the server,
- `streamlit run app/client/main.py` to start the client.

### Client UI

At the top of the page, you can select a model to infer out of the models available on the server. Then you can choose one of two prediction modes:

- Predict attendance for a tournament from a Lichess link or tournament ID (for example, https://lichess.org/tournament/RpFFP5T8).
- Predict attendance for a dataset of tournaments in TSV format. The file must have the same format as the files in [`data/tournament_dataset`](../data/tournament_dataset).
    - The script [`make_dataset.py`](../data_collection/make_dataset.py) can create datasets in this format from .ndjson files containing raw JSONs obtained from [this Lichess API endpoint](https://lichess.org/api#tag/Arena-tournaments/operation/tournament).
    - This format is also described as a Pandera schema in the [`preprocessing`](../preprocessing/preprocessing/__init__.py#L15) module.

After entering a link or uploading a file, click **Predict** to make a prediction.

If you entered a Lichess tournament link or ID, the app will display the predicted number of players for that tournament, as well as the actual number of players if the tournament is already finished. It will also show an **Explain this prediction** button, which generates an explanation chart based on [LIME](https://github.com/marcotcr/lime) (for non-linear models) or the model coefficients themselves (for linear models). The chart shows the most important features for that particular prediction, with a numerical weight for each feature that can be either positive or negative, indicating how that feature impacted the predicted number of players.

If you uploaded a dataset, the app will make predictions for each tournament in it and show a **Download predictions (TSV)** button, allowing you to download the predictions as a text file (each prediction on a separate line, in the same order as in the input file). If the dataset had an `n_players` column containing the true attendance figures, it will also calculate and show the model's regression metrics: R², MAPE, MAE and MSE.

### Server API

The server uses an API to communicate with the client. Here are the available methods:

`GET /list_models` — lists the models available on the server. The client calls this once on startup to display the available models in its interface. The models are taken from the `models` directory in the folder the server is run from.

All other methods require the caller to specify a model as the `{model_name}` path parameter. It must be one of the models returned by `list_models`.

`POST /predict_tsv/{model_name}` — makes predictions from an uploaded dataset file in TSV format. This returns a JSON object containing the predicted values as a list, as well as regression metrics if the uploaded dataset had an `n_players` column.

`POST /predict_link/{model_name}` — makes predictions from a Lichess tournament link or ID, passed as the query parameter `tournament_link_or_id`. Internally, the server calls the Lichess API to obtain information about the tournament, converts it to the format required for prediction, makes a prediction and returns it along with some other info about the tournament (actual attendance if the tournament is finished, name, starting date and time).

`POST /explain/{model_name}` — generates an explanation for a prediction made from a Lichess tournament link or ID, passed as the query parameter `tournament_link_or_id`. For linear models, this uses model coefficients multiplied by the value of the corresponding feature for this tournament. For non-linear models, it uses LIME. Returns an object containing the key `source`, equal to either `lime` or `coefs`, and the key `items`, containing the most important features and their corresponding weights ordered in decreasing order of importance. No more than 20 features are returned. Features with weight 0 are not returned.