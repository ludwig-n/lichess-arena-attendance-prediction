import json
import pathlib

import pandas as pd
import requests
import tqdm

import utils


BASE_URL = "https://lichess.org/api/tournament"


def get_info(in_path, out_path, overwrite=True):
    with open(out_path, "w" if overwrite else "a") as fout, requests.Session() as session:
        for id in tqdm.tqdm(pd.read_csv(in_path, sep="\t").id, desc=str(in_path)):
            dct = utils.try_get(session, f"{BASE_URL}/{id}").json()
            dct.pop("standing", None)
            print(json.dumps(dct, separators=(",", ":")), file=fout)    # dump compact JSON


in_dir = "data/tournament_lists"
out_dir = "data/tournament_info"
for file in pathlib.Path(in_dir).glob("*.tsv"):
    if file.stem == "hourly":
        continue
    get_info(file, f"{out_dir}/{file.stem}.ndjson")
