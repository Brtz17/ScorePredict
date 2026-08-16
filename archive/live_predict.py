"""
ScorePredict - live predikcio meg le NEM jatszott (upcoming) meccsekre.

FONTOS KULONBSEG a predict_dataset.py-hoz kepest:
- Nincs home_goals/away_goals a bemenetben, tehat nincs actual_outcome/
  correct/tevesztesi matrix - csak a probs (varhato golok, H/D/V
  valoszinusegek, argmax dontes) kerul kimenetre.
- A feature-oket NEM a build_dataset() ujraszamolja vegig a teljes
  fixtures.csv-n, hanem a mar elmentett output/dataset.csv-bol OLVASSUK
  ki csapatonkent a legutobbi lejatszott meccs UTANI allapotot (a
  feature_engine.build_dataset() altal mentett *_after oszlopokbol) -
  update nelkul. A days_since_last_match-hez az output/fixtures.csv-t
  hasznaljuk, mert az tartalmaz valos datumot (a dataset.csv nem).
- A FEATURE_COLUMNS-ban (train_bivariate.py) nincs motivacio-feature,
  ezert itt nincs is szukseg standings.csv-re / knockout-logikara.

Bemenet, amit ez a modul FELTETELEZ (ha nalad mas a formatum, csak az
alabbi load_* fuggvenyeket kell igazitani):

1) dataset.csv - a feature_engine.main() altal mentett fajl, a
   home_team_id/away_team_id es a *_after oszlopokkal (attack/defense/
   EWMA-form a sor meccse UTAN). A sorok mar kronologikus sorrendben
   vannak, mert a build_dataset() datum szerint rendezve dolgozza fel a
   fixtures-t.

1b) fixtures.csv - kizarolag a days_since_last_match kiszamitasahoz,
   mert csak ez tartalmaz valos meccsdatumot.

2) team_mapping.csv - a ket API (tortenelmi API-Football vs. elo/jovobeli
   fixtures API) csapat ID-inak parositasa. Valos oszlopok:
     league_id, league_name, competition_code,
     source_team_id, source_team_name,
     api_football_team_id, api_football_team_name,
     match_score, needs_review
   A league_id KOZOS a ket API kozott (nincs kulon liga-ID forditas),
   csak a csapat-ID-k ternek el. A needs_review=True sorok hasznalatakor
   a script figyelmezteto uzenetet ir ki (de nem hagyja ki a meccset).

3) upcoming_fixtures - DataFrame/CSV a meg le nem jatszott meccsekrol.
   Valos oszlopok (a te CSV-d fejleceb): source, competition_code,
   competition_name, match_id, utc_date, matchday, home_team_id,
   home_team_name, away_team_id, away_team_name.
   Nincs benne "season" - ezert a CURRENT_SEASON_BY_LEAGUE_ID konfigbol
   olvassuk ki league_id alapjan (ezt neked kell karbantartanod, amikor
   fordul a szezon). A competition_code (pl. "PL", "PD", "BL1") a
   COMPETITION_CODE_TO_LEAGUE_ID tablan keresztul forditodik at a
   team_mapping.csv altal hasznalt numerikus (api-football) league_id-re.

   MEGJEGYZES: a jovobeli meccsek lekero kodja meg nincs megirva -
   ez a script onmagaban csak a mar lekert upcoming_fixtures adatbol
   szamol predikciot. A tenyleges API-hivast (2. lepes a tervedbol)
   kulon kell hozzaadni, ami ezt a DataFrame-et eloallitja.
"""

from __future__ import annotations

from datetime import datetime, timezone

import joblib
import pandas as pd

from feature_engine import NEUTRAL_GOALS_AVG
from train_bivariate import FEATURE_COLUMNS, NON_HOME_COLS, predict_match

MODEL_DIR = "Trained"
DATASET_PATH = "output/dataset.csv"
FIXTURES_PATH = "output/fixtures.csv"
TEAM_MAPPING_PATH = "output/team_mapping.csv"
OUTPUT_PATH = "output/live_predictions.csv"

COMPETITION_CODE_TO_LEAGUE_ID = {
    "PL":  39,   # Premier League
    "PD":  140,  # La Liga
    "BL1": 78,   # Bundesliga
    "SA":  135,  # Serie A
    "FL1": 61,   # Ligue 1
    "DED": 88,   # Eredivisie
    "PPL": 94,   # Primeira Liga
    "ELC": 40,   # Championship
    "CL":  2,    # UEFA Champions League
}

CURRENT_SEASON_BY_LEAGUE_ID = {
    39: 2025,   # Premier League
    140: 2025,  # La Liga
    78: 2025,   # Bundesliga
    135: 2025,  # Serie A
    61: 2025,   # Ligue 1
    88: 2025,   # Eredivisie
    94: 2025,   # Primeira Liga
    40: 2025,   # Championship
    2: 2025,    # UEFA Champions League
}

NEUTRAL_FORM = 1.0
NEUTRAL_DAYS_SINCE = 365




def load_team_state_from_dataset(path: str = DATASET_PATH) -> dict:
    dataset = pd.read_csv(path)

    attack_defense_state: dict = {}
    form_state: dict = {}

    for _, row in dataset.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        league_id = row["league_id"]
        season = row["season"]

        attack_defense_state[(home_id, league_id, season)] = {
            "attack": row["home_attack_after"],
            "defense": row["home_defense_after"],
        }
        attack_defense_state[(away_id, league_id, season)] = {
            "attack": row["away_attack_after"],
            "defense": row["away_defense_after"],
        }

        form_state[home_id] = {
            "goals_scored_lastN": row["home_goals_scored_lastN_after"],
            "goals_conceded_lastN": row["home_goals_conceded_lastN_after"],
        }
        form_state[away_id] = {
            "goals_scored_lastN": row["away_goals_scored_lastN_after"],
            "goals_conceded_lastN": row["away_goals_conceded_lastN_after"],
        }

    return {"attack_defense": attack_defense_state, "form": form_state}


def load_fixtures(path: str = FIXTURES_PATH) -> pd.DataFrame:
    fixtures = pd.read_csv(path)
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    return fixtures




def load_team_mapping(path: str = TEAM_MAPPING_PATH) -> dict:
    mapping_df = pd.read_csv(path)
    mapping = {}
    for _, row in mapping_df.iterrows():
        key = (row["source_team_id"], row["league_id"])
        mapping[key] = {
            "api_football_team_id": row["api_football_team_id"],
            "needs_review": bool(row.get("needs_review", False)),
            "match_score": row.get("match_score"),
        }
    return mapping




def get_attack_defense_with_season_fallback(attack_defense_state: dict, team_id, league_id, season) -> dict:
    entry = attack_defense_state.get((team_id, league_id, season))
    if entry is not None:
        return entry

    prior_seasons = [
        s for (t, l, s) in attack_defense_state
        if t == team_id and l == league_id and s != season
    ]
    if not prior_seasons:
        return {"attack": NEUTRAL_GOALS_AVG, "defense": NEUTRAL_GOALS_AVG}

    fallback_season = max(prior_seasons)
    return attack_defense_state[(team_id, league_id, fallback_season)]


def get_days_since_last_match(fixtures: pd.DataFrame, team_id, before_date: pd.Timestamp) -> float:
    played = fixtures.dropna(subset=["home_goals", "away_goals"])

    team_matches = played[
        ((played["home_team_id"] == team_id) | (played["away_team_id"] == team_id))
        & (played["date"] < before_date)
    ]

    if team_matches.empty:
        return float(NEUTRAL_DAYS_SINCE)

    last_date = team_matches["date"].max()
    return (before_date - last_date).total_seconds() / 86400


def compute_live_features(dataset_state: dict, fixtures: pd.DataFrame,
                           home_id, away_id, league_id, season,
                           match_date: pd.Timestamp) -> tuple[dict, dict]:
    attack_defense_state = dataset_state["attack_defense"]
    form_state = dataset_state["form"]

    home_ad = get_attack_defense_with_season_fallback(attack_defense_state, home_id, league_id, season)
    away_ad = get_attack_defense_with_season_fallback(attack_defense_state, away_id, league_id, season)

    home_form = form_state.get(home_id, {"goals_scored_lastN": NEUTRAL_FORM, "goals_conceded_lastN": NEUTRAL_FORM})
    away_form = form_state.get(away_id, {"goals_scored_lastN": NEUTRAL_FORM, "goals_conceded_lastN": NEUTRAL_FORM})

    home_days_since = get_days_since_last_match(fixtures, home_id, match_date)
    away_days_since = get_days_since_last_match(fixtures, away_id, match_date)

    home_features = {
        "team_attack": home_ad["attack"],
        "team_defense": home_ad["defense"],
        "opp_attack": away_ad["attack"],
        "opp_defense": away_ad["defense"],
        "team_goals_lastN": home_form["goals_scored_lastN"],
        "team_conceded_lastN": home_form["goals_conceded_lastN"],
        "opp_goals_lastN": away_form["goals_scored_lastN"],
        "opp_conceded_lastN": away_form["goals_conceded_lastN"],
        "days_since_last_match": home_days_since,
    }
    away_features = {
        "team_attack": away_ad["attack"],
        "team_defense": away_ad["defense"],
        "opp_attack": home_ad["attack"],
        "opp_defense": home_ad["defense"],
        "team_goals_lastN": away_form["goals_scored_lastN"],
        "team_conceded_lastN": away_form["goals_conceded_lastN"],
        "opp_goals_lastN": home_form["goals_scored_lastN"],
        "opp_conceded_lastN": home_form["goals_conceded_lastN"],
        "days_since_last_match": away_days_since,
    }

    assert set(home_features) == set(NON_HOME_COLS)
    return home_features, away_features




def predict_upcoming(model, dataset_state: dict, fixtures: pd.DataFrame, team_mapping: dict,
                      upcoming_fixtures: pd.DataFrame) -> pd.DataFrame:
    records = []
    skipped = []
    needs_review_teams = set()

    for _, fx in upcoming_fixtures.iterrows():
        competition_code = fx["competition_code"]
        league_id = COMPETITION_CODE_TO_LEAGUE_ID.get(competition_code)

        if league_id is None:
            skipped.append({
                "match_id": fx["match_id"],
                "reason": f"ismeretlen competition_code: {competition_code!r} "
                          f"(add hozza a COMPETITION_CODE_TO_LEAGUE_ID-hez)",
                "home_team_id": fx["home_team_id"],
                "home_team_name": fx.get("home_team_name"),
                "away_team_id": fx["away_team_id"],
                "away_team_name": fx.get("away_team_name"),
            })
            continue

        season = CURRENT_SEASON_BY_LEAGUE_ID.get(league_id)
        if season is None:
            skipped.append({
                "match_id": fx["match_id"],
                "reason": f"nincs season megadva a CURRENT_SEASON_BY_LEAGUE_ID-ben "
                          f"league_id={league_id}-hez",
                "home_team_id": fx["home_team_id"],
                "home_team_name": fx.get("home_team_name"),
                "away_team_id": fx["away_team_id"],
                "away_team_name": fx.get("away_team_name"),
            })
            continue

        home_key = (fx["home_team_id"], league_id)
        away_key = (fx["away_team_id"], league_id)

        home_map = team_mapping.get(home_key)
        away_map = team_mapping.get(away_key)

        if home_map is None or away_map is None:
            missing_side = "home" if home_map is None else "away"
            skipped.append({
                "match_id": fx["match_id"],
                "reason": f"missing team_mapping ({missing_side})",
                "home_team_id": fx["home_team_id"],
                "home_team_name": fx.get("home_team_name"),
                "away_team_id": fx["away_team_id"],
                "away_team_name": fx.get("away_team_name"),
            })
            continue

        if home_map["needs_review"]:
            needs_review_teams.add((fx["home_team_id"], fx.get("home_team_name")))
        if away_map["needs_review"]:
            needs_review_teams.add((fx["away_team_id"], fx.get("away_team_name")))

        home_id = home_map["api_football_team_id"]
        away_id = away_map["api_football_team_id"]
        match_date = pd.to_datetime(fx["utc_date"])

        home_features, away_features = compute_live_features(
            dataset_state, fixtures, home_id, away_id, league_id, season, match_date
        )

        probs = predict_match(model, home_features, away_features)
        outcome_probs = {"H": probs["home_win_prob"], "D": probs["draw_prob"], "A": probs["away_win_prob"]}
        decision = max(outcome_probs, key=outcome_probs.get)

        records.append({
            "match_id": fx["match_id"],
            "competition_code": competition_code,
            "league_id": league_id,
            "season": season,
            "matchday": fx.get("matchday"),
            "date": match_date,
            "home_team_id": fx["home_team_id"],
            "away_team_id": fx["away_team_id"],
            "api_football_home_id": home_id,
            "api_football_away_id": away_id,
            "exp_home_goals": probs["expected_home_goals"],
            "exp_away_goals": probs["expected_away_goals"],
            "home_win_prob": probs["home_win_prob"],
            "draw_prob": probs["draw_prob"],
            "away_win_prob": probs["away_win_prob"],
            "decision": decision,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    if needs_review_teams:
        print(f"\nFIGYELEM: {len(needs_review_teams)} csapat nem ellenorzott (needs_review) "
              f"mappinget hasznal - erdemes kezzel atnezni a team_mapping.csv-ben:")
        for team_id, team_name in sorted(needs_review_teams, key=lambda t: str(t[1])):
            print(f"  {team_name} (source_team_id={team_id})")

    if skipped:
        print(f"\nKIHAGYVA {len(skipped)} meccs mappalatlan csapat/liga/szezon miatt:")
        for s in skipped:
            print(f"  match {s['match_id']}: {s['reason']} "
                  f"(home={s['home_team_name']} [{s['home_team_id']}], "
                  f"away={s['away_team_name']} [{s['away_team_id']}])")

    result_df = pd.DataFrame(records)
    return result_df




def main():
    model = joblib.load(f"{MODEL_DIR}/bivariate_goals_model.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
    assert feature_columns == FEATURE_COLUMNS, "A betoltott modell mas feature-listaval keszult!"

    dataset_state = load_team_state_from_dataset(DATASET_PATH)
    fixtures = load_fixtures(FIXTURES_PATH)
    team_mapping = load_team_mapping(TEAM_MAPPING_PATH)

    upcoming_fixtures = pd.read_csv("output/upcoming_fixtures.csv")  # nalad mar letezik - ha mashol van, igazitsd az utvonalat

    result_df = predict_upcoming(model, dataset_state, fixtures, team_mapping, upcoming_fixtures)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nElo predikciok elmentve: {OUTPUT_PATH} ({len(result_df)} meccs)")
    print(result_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()