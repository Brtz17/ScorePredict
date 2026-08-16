import csv
import json
import os
from datetime import datetime
from pathlib import Path

import joblib

from feature_engine import get_running_attack_defense_carryover
from train_bivariate import FEATURE_COLUMNS, predict_match

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEAM_FORM_STATE_PATH = os.path.join(BASE_DIR, "output", "team_form_state.json")
TEAM_MAPPING_PATH = os.path.join(BASE_DIR, "output", "team_mapping.csv")
UPCOMING_FIXTURES_PATH = os.path.join(BASE_DIR, "output", "upcoming_fixtures.csv")
MODEL_PATH = os.path.join(BASE_DIR, "output", "models", "bivariate_goals_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "output", "models", "feature_columns.pkl")
OUTPUT_PATH = Path(BASE_DIR) / "output" / "live_predictions.csv"

NEUTRAL_FORM = 1.0          
NEUTRAL_DAYS_SINCE = 365    


def infer_season(match_date: datetime) -> int:
    return match_date.year if match_date.month >= 7 else match_date.year - 1


def load_team_form_state(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)

    team_goal_stats = {}
    for key, entry in raw["team_goal_stats"].items():
        team_id, league_id, season = key.split(":")
        team_goal_stats[(int(team_id), int(league_id), int(season))] = entry

    ewma_attack = {int(k): v for k, v in raw["ewma_attack"].items()}
    ewma_defense = {int(k): v for k, v in raw["ewma_defense"].items()}
    last_match_date = {
        int(k): datetime.fromisoformat(v) for k, v in raw["last_match_date"].items()
    }

    return {
        "team_goal_stats": team_goal_stats,
        "ewma_attack": ewma_attack,
        "ewma_defense": ewma_defense,
        "last_match_date": last_match_date,
    }


def load_team_mapping(path: str) -> dict:
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["competition_code"], int(row["source_team_id"]))
            mapping[key] = row
    return mapping


def load_next_round_fixtures(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    min_matchday = {}
    for row in rows:
        code = row["competition_code"]
        if not row["matchday"]:
            continue
        matchday = int(row["matchday"])
        if code not in min_matchday or matchday < min_matchday[code]:
            min_matchday[code] = matchday

    return [
        row for row in rows
        if row["matchday"] and int(row["matchday"]) == min_matchday.get(row["competition_code"])
    ]


def team_features(team_form_state: dict, team_id: int, league_id: int, season: int,
                   match_date: datetime) -> dict:
    attack_defense = get_running_attack_defense_carryover(team_form_state["team_goal_stats"], team_id, league_id, season)
    goals_lastN = team_form_state["ewma_attack"].get(team_id, NEUTRAL_FORM)
    conceded_lastN = team_form_state["ewma_defense"].get(team_id, NEUTRAL_FORM)

    last_date = team_form_state["last_match_date"].get(team_id)
    days_since = (match_date - last_date).total_seconds() / 86400 if last_date else NEUTRAL_DAYS_SINCE

    return {
        "attack": attack_defense["attack"],
        "defense": attack_defense["defense"],
        "goals_lastN": goals_lastN,
        "conceded_lastN": conceded_lastN,
        "days_since_last_match": days_since,
    }


def build_model_features(home: dict, away: dict) -> tuple:
    home_features = {
        "team_attack": home["attack"],
        "team_defense": home["defense"],
        "opp_attack": away["attack"],
        "opp_defense": away["defense"],
        "team_goals_lastN": home["goals_lastN"],
        "team_conceded_lastN": home["conceded_lastN"],
        "opp_goals_lastN": away["goals_lastN"],
        "opp_conceded_lastN": away["conceded_lastN"],
        "days_since_last_match": home["days_since_last_match"],
    }
    away_features = {
        "team_attack": away["attack"],
        "team_defense": away["defense"],
        "opp_attack": home["attack"],
        "opp_defense": home["defense"],
        "team_goals_lastN": away["goals_lastN"],
        "team_conceded_lastN": away["conceded_lastN"],
        "opp_goals_lastN": home["goals_lastN"],
        "opp_conceded_lastN": home["conceded_lastN"],
        "days_since_last_match": away["days_since_last_match"],
    }
    return home_features, away_features


def main():
    saved_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    if saved_feature_columns != FEATURE_COLUMNS:
        raise RuntimeError(
        )

    team_form_state = load_team_form_state(TEAM_FORM_STATE_PATH)
    team_mapping = load_team_mapping(TEAM_MAPPING_PATH)
    fixtures = load_next_round_fixtures(UPCOMING_FIXTURES_PATH)
    model = joblib.load(MODEL_PATH)

    print(f"{len(fixtures)} matches in the next round")

    results = []
    for fx in fixtures:
        code = fx["competition_code"]
        home_map = team_mapping.get((code, int(fx["home_team_id"])))
        away_map = team_mapping.get((code, int(fx["away_team_id"])))

        if not home_map or not home_map["api_football_team_id"]:
            print(f"  [skipped] {fx['home_team_name']} not mapped ({code})")
            continue
        if not away_map or not away_map["api_football_team_id"]:
            print(f"  [skipped] {fx['away_team_name']} not mapped ({code})")
            continue
        if home_map["needs_review"] == "True" or away_map["needs_review"] == "True":
            print(f"{fx['home_team_name']} vs {fx['away_team_name']} - check team_mapping.csv")

        league_id = int(home_map["league_id"])
        match_date = datetime.fromisoformat(fx["utc_date"].replace("Z", "+00:00"))
        season = infer_season(match_date)

        home_id = int(home_map["api_football_team_id"])
        away_id = int(away_map["api_football_team_id"])

        home_feats = team_features(team_form_state, home_id, league_id, season, match_date)
        away_feats = team_features(team_form_state, away_id, league_id, season, match_date)
        home_model_feats, away_model_feats = build_model_features(home_feats, away_feats)

        probs = predict_match(model, home_model_feats, away_model_feats)
        outcome_probs = {"H": probs["home_win_prob"], "D": probs["draw_prob"], "A": probs["away_win_prob"]}
        decision = max(outcome_probs, key=outcome_probs.get)

        results.append({
            "competition_code": code,
            "league_id": league_id,
            "match_id": fx["match_id"],
            "utc_date": fx["utc_date"],
            "matchday": fx["matchday"],
            "home_team_name": fx["home_team_name"],
            "away_team_name": fx["away_team_name"],
            "exp_home_goals": probs["expected_home_goals"],
            "exp_away_goals": probs["expected_away_goals"],
            "home_win_prob": probs["home_win_prob"],
            "draw_prob": probs["draw_prob"],
            "away_win_prob": probs["away_win_prob"],
            "decision": decision,
        })

    if not results:
        print("No prediction")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {OUTPUT_PATH} ({len(results)} prediction)")


if __name__ == "__main__":
    main()