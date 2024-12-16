import datetime
import time

import bs4
import requests

import utils


BASE_URL = "https://lichess.org/tournament/history"
CATEGORIES = [
    "unique",
    "marathon",
    "shield",
    "yearly",
    "monthly",
    "weekend",
    "weekly",
    "daily",
    "eastern",
    "hourly"
]


def get_tournaments(category, start_date, end_date, save_dir, overwrite=True, request_interval=1):
    if overwrite:
        fout = open(f"{save_dir}/{category}.tsv", "w")
        print("id\tdate\tn_players\tname\tdetails", file=fout)
    else:
        fout = open(f"{save_dir}/{category}.tsv", "a")
    session = requests.Session()
    page = 0
    n_loaded = 0
    last_date_iso = None
    done = False
    while not done:
        page += 1
        time.sleep(request_interval)
        response = utils.try_get(session, f"{BASE_URL}/{category}", params={"page": page}).text
        rows = bs4.BeautifulSoup(response, "lxml").find_all("tr", {"class": "paginated"})
        if not rows:    # out of tournaments
            break
        for row in rows:
            date_iso = row.find("time")["datetime"]
            date = datetime.datetime.fromisoformat(date_iso)
            if date >= end_date:     # tournament is too new
                continue
            if date < start_date:    # tournament is too old
                done = True
                break
            spans = row.find_all("span")
            cols = [
                row.find("a")["href"].removeprefix("/tournament/"),    # id
                date_iso,         # date
                ''.join(c for c in spans[-1].text if c.isdigit()),     # n_players
                spans[0].text,    # name
                spans[1].text     # details
            ]
            print("\t".join(cols), file=fout)
            n_loaded += 1
            last_date_iso = date_iso
        print(f"{category}: loaded page {page} - {n_loaded} tournaments, last at {last_date_iso}")
    session.close()
    fout.close()


start_date = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
end_date = datetime.datetime(2024, 12, 16, tzinfo=datetime.timezone.utc)
save_dir = "data/tournament_lists"
for category in CATEGORIES:
    get_tournaments(category, start_date, end_date, save_dir)
