import json
import pathlib

import pandas as pd
import requests
import tqdm

import utils


BASE_URL = "https://lichess.org/api/tournament"


def get_info(in_path, out_path, use_cache, overwrite=True):
    cache = {}
    with open(out_path, "w" if overwrite else "a") as fout, requests.Session() as session:
        df = pd.read_csv(in_path, sep="\t")
        for row in tqdm.tqdm(df.itertuples(), desc=str(in_path), total=len(df)):
            if use_cache and (row.name, row.details) in cache:
                dct = cache[row.name, row.details] | {
                    "id": row.id,
                    "startsAt": row.date,
                    "nbPlayers": row.n_players
                }
                dct.pop("podium", None)
                dct.pop("stats", None)
            else:
                dct = utils.try_get(session, f"{BASE_URL}/{row.id}").json()
                dct.pop("standing", None)
                if use_cache and "Thematic" not in row.details:
                    cache[row.name, row.details] = dct
            print(json.dumps(dct, separators=(",", ":")), file=fout)    # dump compact JSON


in_dir = "data/tournament_lists"
out_dir = "data/tournament_info"
for file in pathlib.Path(in_dir).glob("*.tsv"):
    get_info(file, f"{out_dir}/{file.stem}.ndjson", file.stem == "hourly")
