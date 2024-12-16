import datetime
import time

import bs4
import pandas as pd
import requests


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


def get_tournaments(category, start_date, end_date, save_dir, request_interval=1):
    session = requests.Session()
    page = 0
    results = []
    done = False
    while not done:
        page += 1
        time.sleep(request_interval)
        response = session.get(f"{BASE_URL}/{category}", params={"page": page}).text
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
            results.append({
                "id": row.find("a")["href"].removeprefix("/tournament/"),
                "date": date_iso,
                "n_players": int(''.join(c for c in spans[-1].text if c.isdigit())),
                "name": spans[0].text,
                "details": spans[1].text
            })
        print(f"{category}: loaded page {page} - {len(results)} tournaments, last at {results[-1]['date'] if results else None}")
    session.close()
    pd.DataFrame(results).to_csv(f"{save_dir}/{category}.tsv", sep="\t", index=False)


start_date = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
end_date = datetime.datetime(2024, 12, 16, tzinfo=datetime.timezone.utc)
save_dir = "data/tournament-lists"

for category in CATEGORIES:
    get_tournaments(category, start_date, end_date, save_dir)
