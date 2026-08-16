import csv
import difflib
import unicodedata
from pathlib import Path

from league_mapping import LEAGUE_ID_MAP

TEAMS_CSV = "output/teams.csv"
UPCOMING_FIXTURES_CSV = "output/upcoming_fixtures.csv"
OUTPUT_PATH = Path("output/team_mapping.csv")

MANUAL_REVIEW_THRESHOLD = 0.80


def _normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    for junk in (" fc", " cf", " sc", " afc", ".", ","):
        name = name.replace(junk, "")
    return name.strip()


def load_api_football_teams(path: str) -> dict[int, list[dict]]:
    latest_by_key: dict[tuple[int, int], dict] = {}

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            league_id = int(row["league_id"])
            team_id = int(row["team_id"])
            season = int(row["season"])
            key = (league_id, team_id)

            if key not in latest_by_key or season > latest_by_key[key]["season"]:
                latest_by_key[key] = {"season": season, "team_id": team_id, "name": row["name"]}

    by_league: dict[int, list[dict]] = {}
    for (league_id, _team_id), entry in latest_by_key.items():
        by_league.setdefault(league_id, []).append(entry)
    return by_league


def load_upcoming_teams(path: str) -> dict[str, list[dict]]:
    seen: dict[tuple[str, int], str] = {}

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for side in ("home", "away"):
                team_id = int(row[f"{side}_team_id"])
                name = row[f"{side}_team_name"]
                seen[(row["competition_code"], team_id)] = name

    by_competition: dict[str, list[dict]] = {}
    for (code, team_id), name in seen.items():
        by_competition.setdefault(code, []).append({"team_id": team_id, "name": name})
    return by_competition


def best_match(name: str, candidates: list[dict]) -> tuple[dict | None, float]:
    normalized = _normalize(name)
    best_entry, best_score = None, 0.0

    for candidate in candidates:
        score = difflib.SequenceMatcher(None, normalized, _normalize(candidate["name"])).ratio()
        if score > best_score:
            best_entry, best_score = candidate, score

    return best_entry, round(best_score, 3)


def build_mapping() -> list[dict]:
    api_football_teams = load_api_football_teams(TEAMS_CSV)
    upcoming_teams = load_upcoming_teams(UPCOMING_FIXTURES_CSV)

    rows = []
    for competition_code, (league_id, league_name) in LEAGUE_ID_MAP.items():
        candidates = api_football_teams.get(league_id, [])
        source_teams = upcoming_teams.get(competition_code, [])

        if not source_teams:
            print(f"  [skip] {competition_code} ({league_name}): nincs adat az upcoming_fixtures.csv-ben")
            continue

        for team in source_teams:
            match, score = best_match(team["name"], candidates)
            rows.append({
                "league_id": league_id,
                "league_name": league_name,
                "competition_code": competition_code,
                "source_team_id": team["team_id"],
                "source_team_name": team["name"],
                "api_football_team_id": match["team_id"] if match else None,
                "api_football_team_name": match["name"] if match else None,
                "match_score": score,
                "needs_review": score < MANUAL_REVIEW_THRESHOLD,
            })

    return rows


def main():
    rows = build_mapping()
    if not rows:
        print("Nincs mit menteni - eloszor futtasd le sikeresen a fetch_upcoming_fixtures.py-t.")
        return

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "league_id", "league_name", "competition_code",
            "source_team_id", "source_team_name",
            "api_football_team_id", "api_football_team_name",
            "match_score", "needs_review",
        ])
        writer.writeheader()
        writer.writerows(rows)

    review_count = sum(1 for r in rows if r["needs_review"])
    print(f"Mentve: {OUTPUT_PATH} ({len(rows)} csapat, ebbol {review_count} kezi ellenorzest igenyel)")


if __name__ == "__main__":
    main()
