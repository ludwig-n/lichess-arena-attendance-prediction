import json

import pandas as pd


def json_list_to_dataframe(jsons):
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
