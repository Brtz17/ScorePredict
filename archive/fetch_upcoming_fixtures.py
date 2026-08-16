import csv
import os
import sys
import time
from pathlib import Path

import requests

from league_mapping import FOOTBALL_DATA_ORG_LEAGUES

API_KEY = os.environ.get("FOOTBALL_DATA_ORG_KEY")
BASE_URL = "https://api.football-data.org/v4"
OUTPUT_DIR = Path("output")

MIN_SECONDS_BETWEEN_REQUESTS = 6.5

_last_request_time = 0.0


def get_leagues_to_track() -> list[str]:
    return list(FOOTBALL_DATA_ORG_LEAGUES.keys())


def _headers():
    if not API_KEY:
        sys.exit("Hianyzik a FOOTBALL_DATA_ORG_KEY environment variable")
    return {"X-Auth-Token": API_KEY}


def _throttle():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        print(f"  Varakozas {wait:.1f}s (rate limit)...")
        time.sleep(wait)


def api_get(endpoint: str, params: dict | None = None) -> dict:
    global _last_request_time
    url = f"{BASE_URL}/{endpoint}"

    for _ in range(5):
        _throttle()
        response = requests.get(url, headers=_headers(), params=params or {}, timeout=30)
        _last_request_time = time.time()

        if response.status_code == 429:
            wait_for = int(response.headers.get("X-RequestCounter-Reset", 60))
            print(f"  429-es hiba, varakozas {wait_for}s...")
            time.sleep(wait_for)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Sikertelen lekerdezes: {endpoint}")


def fetch_upcoming_fixtures(competition_code: str) -> list[dict]:
    print(f"Jovobeli meccsek lekerese: {competition_code}...")
    payload = api_get(f"competitions/{competition_code}/matches", {"status": "SCHEDULED"})

    rows = []
    for match in payload.get("matches", []):
        rows.append({
            "source": "football-data.org",
            "competition_code": competition_code,
            "competition_name": FOOTBALL_DATA_ORG_LEAGUES.get(competition_code),
            "match_id": match["id"],
            "utc_date": match["utcDate"],
            "matchday": match.get("matchday"),
            "home_team_id": match["homeTeam"]["id"],
            "home_team_name": match["homeTeam"]["name"],
            "away_team_id": match["awayTeam"]["id"],
            "away_team_name": match["awayTeam"]["name"],
        })

    print(f"  {len(rows)} jovobeli meccs talalva.")
    return rows


def write_csv(filename: str, rows: list[dict], fieldnames: list[str]):
    if not rows:
        print(f"  {filename}: nincs adat, kihagyva")
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Mentve: {path} ({len(rows)} sor)")


def main():
    all_rows = []
    for code in get_leagues_to_track():
        try:
            all_rows.extend(fetch_upcoming_fixtures(code))
        except Exception as e:
            print(f"  Hiba ({code}): {e}")

    write_csv(
        "upcoming_fixtures.csv",
        all_rows,
        ["source", "competition_code", "competition_name", "match_id", "utc_date",
         "matchday", "home_team_id", "home_team_name", "away_team_id", "away_team_name"],
    )

    print(f"\nOsszesen {len(all_rows)} jovobeli meccs "
          f"{len(get_leagues_to_track())} bajnoksagbol.")


if __name__ == "__main__":
    main()
