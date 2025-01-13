## Lichess Arena Attendance Prediction ♟️

A project that uses classic ML regression models to predict the number of players who will join a tournament on [Lichess.org](https://lichess.org).

### Installation

The project virtual environment can be set up with the following commands (assuming you have python and git installed):
```bash
git clone https://github.com/ludwig-n/lichess-arena-attendance-prediction.git
cd lichess-arena-attendance-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
After these steps, you should be able to run all the scripts and notebooks in the project. Note that scripts (.py files) should be run from the project root folder.

### Project structure

- [`data_collection`](data_collection): scripts that download the data for the project and compile it into TSV dataset files.
    - [`get_tournament_lists.py`](data_collection/get_tournament_lists.py): downloads tournament lists from the [Lichess tournament history page](https://lichess.org/tournament/history) for a given time period and saves the available information in TSV format (note that this is not the final dataset format).
    - [`get_tournament_info.py`](data_collection/get_tournament_info.py): downloads detailed info on these tournaments from the Lichess API and saves it in NDJSON format.
    - [`make_dataset.py`](data_collection/make_dataset.py): processes the NDJSON files into a TSV dataset.

- [`data/tournament_dataset`](data/tournament_dataset): the dataset used for training the models, comprising all Lichess tournaments listed in the tournament history page from 2023-01-01 to 2024-12-15.

- [`preprocessing`](preprocessing): code for data preprocessing. This folder is installed as an editable module in the project environment.

- [`eda`](eda): exploratory data analysis.
    - [`eda.ipynb`](eda/eda.ipynb): the main EDA notebook.
    - [`train_basic.py`](eda/train_basic.py): a script that trains 3 basic models (with mostly default parameters): ridge regression, random forest and gradient boosting.
    - [`importances.ipynb`](eda/importances.ipynb): a notebook analyzing feature importances for the basic random forest model.

- [`models`](models): the trained models in pickle format. The random forest model is not included due to its large file size.

- [`tuning`](tuning): hyperparameter tuning.
    - [`cv`](tuning/cv): cross-validation methods.
        - [`splitters.py`](tuning/cv/splitters.py): code for 4 hand-implemented cross-validation methods.
        - [`visualize_methods.ipynb`](tuning/cv/visualize_methods.ipynb): descriptions and visualizations for these methods.
        - [`compare_methods.ipynb`](tuning/cv/compare_methods.ipynb): a quality and performance comparison of these methods.
    - [`search.py`](tuning/search.py): a script that uses grid search and a genetic algorithm from the [gentun](https://github.com/gmontamat/gentun) library to perform hyperparameter search for the gradient boosting model. Saves the results (gentun `Population` objects) to pickle files for later analysis.

- [`app`](app): a web app that provides a friendly interface to the models.
    - [`README.md`](app/README.md): a more detailed description of the app components.
    - [`server/main.py`](app/server/main.py): code for the server, written in FastAPI.
    - [`client/main.py`](app/client/main.py): code for the client, written in Streamlit.
