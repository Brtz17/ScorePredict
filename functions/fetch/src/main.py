import csv
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

import requests
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from appwrite.exception import AppwriteException


DATABASE_ID = os.environ.get("APPWRITE_DATABASE_ID", "prediction_db")

LEAGUES_CONFIG_COL = "leagues_config"
UPCOMING_FIXTURES_COL = "upcoming_fixtures"

FIXTURES_COL = "fixtures"
TEAM_CURRENT_STATE_COL = "team_current_state"
TEAM_FORM_GLOBAL_COL = "team_form_global"
LEAGUE_RUNNING_STATS_COL = "league_running_stats"
DATASET_HISTORY_COL = "dataset_history"

LEAGUE_ID_MAP = {
    "PL":  (39,  "Premier League"),
    "PD":  (140, "La Liga"),
    "BL1": (78,  "Bundesliga"),
    "SA":  (135, "Serie A"),
    "FL1": (61,  "Ligue 1"),
    "DED": (88,  "Eredivisie"),
    "PPL": (94,  "Primeira Liga"),
    "ELC": (40,  "Championship"),
    "CL":  (2,   "UEFA Champions League"),
}

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
MIN_SECONDS_BETWEEN_REQUESTS = 6.5

RESULTS_LOOKBACK_DAYS = int(os.environ.get("RESULTS_LOOKBACK_DAYS", "10"))

TIME_BUDGET_S = int(os.environ.get("TIME_BUDGET_S", "240"))
STATE_BATCH_SIZE = int(os.environ.get("FIXTURES_BATCH_SIZE", "30"))

NEUTRAL_GOALS_AVG = 1.3
NEUTRAL_FORM = 1.0
EWMA_ALPHA = 0.35

TEAM_MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_mapping.csv")



class ApiCreds:
    __slots__ = ("endpoint", "project_id", "api_key")

    def __init__(self, endpoint: str, project_id: str, api_key: str):
        self.endpoint = endpoint.rstrip("/")
        self.project_id = project_id
        self.api_key = api_key

    def headers(self) -> dict:
        return {
            "X-Appwrite-Project": self.project_id,
            "X-Appwrite-Key": self.api_key,
            "content-type": "application/json",
            "accept": "application/json",
            "cache-control": "no-cache, no-store",
            "pragma": "no-cache",
        }


def _raw_request(creds: ApiCreds, method: str, path: str, params: dict | None = None) -> dict:
    url = f"{creds.endpoint}{path}"
    attempts = 2 if method == "get" else 1
    last_response = None
    
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, headers=creds.headers(), params=params, timeout=30)
            if response.status_code >= 400:
                raise AppwriteException(response.text, response.status_code)
            try:
                return response.json()
            except ValueError:
                print(f"NON-JSON ANSWER ({attempt + 1}/{attempts}. attempt): status={response.status_code} "
                      f"url={response.url} body[:300]={response.text[:300]!r}")
                last_response = response
                continue
        except requests.RequestException as e:
            print(f"Request failed ({attempt + 1}/{attempts}. attempt): {e}")
            last_response = None
            continue
    
    raise AppwriteException(f"Non-JSON answer after {attempts} attempts too", 
                           last_response.status_code if last_response else 500)


def raw_get_document(creds: ApiCreds, database_id: str, collection_id: str, document_id: str) -> dict:
    path = f"/databases/{database_id}/collections/{collection_id}/documents/{document_id}"
    return _raw_request(creds, "get", path)


def raw_list_documents(creds: ApiCreds, database_id: str, collection_id: str, queries: list) -> dict:
    path = f"/databases/{database_id}/collections/{collection_id}/documents"
    params = {}
    for i, q in enumerate(queries or []):
        params[f"queries[{i}]"] = q
    return _raw_request(creds, "get", path, params=params)


def get_leagues_to_track(creds: ApiCreds) -> list[dict]:
    try:
        result = raw_list_documents(
            creds, DATABASE_ID, LEAGUES_CONFIG_COL,
            queries=[Query.equal("active", True), Query.limit(200)],
        )
        return [
            {"competition_code": doc["competition_code"], "competition_name": doc.get("competition_name")}
            for doc in result.get("documents", [])
        ]
    except AppwriteException as e:
        print(f"Error fetching leagues: {e}")
        return []



_last_request_time = 0.0


def _football_data_headers() -> dict:
    api_key = os.environ.get("FOOTBALL_DATA_ORG_KEY")
    if not api_key:
        raise RuntimeError("FOOTBALL_DATA_ORG_KEY environment variable is missing")
    return {"X-Auth-Token": api_key}


def _sleep_within_budget(seconds: float, deadline: float | None, context, reason: str) -> None:
    if deadline is not None and time.monotonic() + seconds > deadline:
        raise RuntimeError(
            f"{reason}: stopping function because running out of time"
        )
    time.sleep(seconds)


def _throttle(context, deadline: float | None = None) -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed
    if wait > 0:
        try:
            context.log(f"  Waiting for {wait:.1f}s (rate limit)...")
        except:
            print(f"  Waiting for {wait:.1f}s (rate limit)...")
        _sleep_within_budget(wait, deadline, context, "throttle")


def football_data_api_get(context, endpoint: str, params: dict | None = None,
                           deadline: float | None = None) -> dict:
    global _last_request_time
    url = f"{FOOTBALL_DATA_BASE_URL}/{endpoint}"

    MAX_RETRY_WAIT_S = 45

    for attempt in range(3):
        _throttle(context, deadline)
        try:
            response = requests.get(url, headers=_football_data_headers(), params=params or {}, timeout=30)
            _last_request_time = time.time()

            if response.status_code == 429:
                wait_for = int(response.headers.get("X-RequestCounter-Reset", 60))
                if wait_for > MAX_RETRY_WAIT_S:
                    raise RuntimeError(
                        f"429 too long wait ({wait_for}s > {MAX_RETRY_WAIT_S}s cap) at {endpoint} - "
                        f"probably daily quota exhausted, giving up on this league for now"
                    )
                try:
                    context.log(f"Error 429, waiting for {wait_for}s... (attempt {attempt + 1}/3)")
                except:
                    print(f"Error 429, waiting for {wait_for}s... (attempt {attempt + 1}/3)")
                _sleep_within_budget(wait_for, deadline, context, f"429 retry at {endpoint}")
                continue

            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise RuntimeError(f"Querying failed after 3 attempts: {endpoint} - {e}")
            try:
                context.log(f"Request failed (attempt {attempt + 1}/3): {e}")
            except:
                print(f"Request failed (attempt {attempt + 1}/3): {e}")
            time.sleep(2)

    raise RuntimeError(f"Querying failed: {endpoint}")


def fetch_upcoming_fixtures(context, competition_code: str, competition_name: str,
                             deadline: float | None = None) -> list[dict]:
    try:
        context.log(f"Querying future matches: {competition_code}...")
    except:
        print(f"Querying future matches: {competition_code}...")
    
    payload = football_data_api_get(context, f"competitions/{competition_code}/matches",
                                     {"status": "SCHEDULED"}, deadline=deadline)

    rows = []
    for match in payload.get("matches", []):
        rows.append({
            "source": "football-data.org",
            "competition_code": competition_code,
            "competition_name": competition_name,
            "match_id": match["id"],
            "utc_date": match["utcDate"],
            "matchday": match.get("matchday"),
            "home_team_id": match["homeTeam"]["id"],
            "home_team_name": match["homeTeam"]["name"],
            "away_team_id": match["awayTeam"]["id"],
            "away_team_name": match["awayTeam"]["name"],
        })

    try:
        context.log(f"  {len(rows)} future matches found")
    except:
        print(f"  {len(rows)} future matches found")
    return rows


def fetch_recent_results(context, competition_code: str, competition_name: str,
                          deadline: float | None = None) -> list[dict]:
    date_to = date.today()
    date_from = date_to - timedelta(days=RESULTS_LOOKBACK_DAYS)
    try:
        context.log(f"Querying recent results: {competition_code} ({date_from} - {date_to})...")
    except:
        print(f"Querying recent results: {competition_code} ({date_from} - {date_to})...")

    payload = football_data_api_get(context, f"competitions/{competition_code}/matches", {
        "status": "FINISHED",
        "dateFrom": date_from.isoformat(),
        "dateTo": date_to.isoformat(),
    }, deadline=deadline)

    rows = []
    for match in payload.get("matches", []):
        full_time = match.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        if home_goals is None or away_goals is None:
            continue

        rows.append({
            "source": "football-data.org",
            "competition_code": competition_code,
            "competition_name": competition_name,
            "match_id": match["id"],
            "utc_date": match["utcDate"],
            "matchday": match.get("matchday"),
            "home_team_id": match["homeTeam"]["id"],
            "home_team_name": match["homeTeam"]["name"],
            "away_team_id": match["awayTeam"]["id"],
            "away_team_name": match["awayTeam"]["name"],
            "home_goals": home_goals,
            "away_goals": away_goals,
        })

    try:
        context.log(f"  {len(rows)} finished matches found")
    except:
        print(f"  {len(rows)} finished matches found")
    return rows


def upsert_document(db: Databases, collection_id: str, doc_id: str, row: dict) -> None:
    try:
        db.update_document(DATABASE_ID, collection_id, doc_id, row)
    except AppwriteException as e:
        if e.code == 404:  # Document not found
            db.create_document(DATABASE_ID, collection_id, doc_id, row)
        else:
            raise




def load_team_mapping(path: str) -> dict:
    mapping = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["competition_code"], int(row["source_team_id"]))
                mapping[key] = row
    except FileNotFoundError:
        print(f"team_mapping.csv not found here: {path}")
        raise
    return mapping


def infer_season(match_date: datetime) -> int:
    return match_date.year if match_date.month >= 7 else match_date.year - 1


def result_rows_to_fixture_docs(context, result_rows: list[dict], team_mapping: dict) -> list[dict]:
    docs = []
    for row in result_rows:
        code = row["competition_code"]
        home_map = team_mapping.get((code, int(row["home_team_id"])))
        away_map = team_mapping.get((code, int(row["away_team_id"])))

        if not home_map or not home_map["api_football_team_id"] or \
           not away_map or not away_map["api_football_team_id"]:
            try:
                context.log(f"  [skipped] {row['home_team_name']} vs {row['away_team_name']} "
                            f"({code}) - no team-mapping")
            except:
                print(f"  [skipped] {row['home_team_name']} vs {row['away_team_name']} "
                      f"({code}) - no team-mapping")
            continue

        if str(home_map.get("needs_review", "")).strip().lower() == "true" or \
           str(away_map.get("needs_review", "")).strip().lower() == "true":
            try:
                context.log(f"  [skipped] {row['home_team_name']} vs {row['away_team_name']} "
                            f"({code}) - needs_review=True, check team_mapping.csv")
            except:
                print(f"  [skipped] {row['home_team_name']} vs {row['away_team_name']} "
                      f"({code}) - needs_review=True, check team_mapping.csv")
            continue

        match_date = datetime.fromisoformat(row["utc_date"].replace("Z", "+00:00"))
        docs.append({
            "doc_id": str(row["match_id"]),
            "data": {
                "league_id": int(home_map["league_id"]),
                "season": infer_season(match_date),
                "round": int(row["matchday"]) if row["matchday"] else None,
                "date": row["utc_date"],
                "home_team_id": int(home_map["api_football_team_id"]),
                "away_team_id": int(away_map["api_football_team_id"]),
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
                "processed": False,
            },
        })
    return docs


def insert_fixture_if_new(db: Databases, doc_id: str, data: dict) -> bool:
    try:
        db.create_document(DATABASE_ID, FIXTURES_COL, doc_id, data)
        return True
    except AppwriteException as exc:
        if exc.code == 409:
            return False
        raise



def get_or_default_team_state(creds: ApiCreds, team_id, league_id, season) -> dict:
    doc_id = f"{team_id}_{league_id}_{season}"
    try:
        doc = raw_get_document(creds, DATABASE_ID, TEAM_CURRENT_STATE_COL, doc_id)
        return {
            "for_sum": doc.get("for_sum", 0.0), 
            "for_count": doc.get("for_count", 0),
            "against_sum": doc.get("against_sum", 0.0), 
            "against_count": doc.get("against_count", 0),
        }
    except AppwriteException:
        return {"for_sum": 0.0, "for_count": 0, "against_sum": 0.0, "against_count": 0}


def attack_defense_from_state(state: dict) -> dict:
    attack = state["for_sum"] / state["for_count"] if state["for_count"] > 0 else NEUTRAL_GOALS_AVG
    defense = state["against_sum"] / state["against_count"] if state["against_count"] > 0 else NEUTRAL_GOALS_AVG
    return {"attack": attack, "defense": defense}


def upsert_team_state(db: Databases, team_id, league_id, season, state: dict, fixture_id: str) -> None:
    doc_id = f"{team_id}_{league_id}_{season}"
    ad = attack_defense_from_state(state)
    data = {
        "team_id": int(team_id), 
        "league_id": int(league_id), 
        "season": int(season),
        "for_sum": state["for_sum"], 
        "for_count": state["for_count"],
        "against_sum": state["against_sum"], 
        "against_count": state["against_count"],
        "attack": ad["attack"], 
        "defense": ad["defense"],
        "last_processed_fixture_id": fixture_id,
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        db.update_document(DATABASE_ID, TEAM_CURRENT_STATE_COL, doc_id, data)
    except AppwriteException:
        db.create_document(DATABASE_ID, TEAM_CURRENT_STATE_COL, doc_id, data)


def get_or_default_form(creds: ApiCreds, team_id) -> dict:
    doc_id = str(team_id)
    try:
        doc = raw_get_document(creds, DATABASE_ID, TEAM_FORM_GLOBAL_COL, doc_id)
        return {
            "goals_scored_lastN": doc.get("goals_scored_lastN", NEUTRAL_FORM), 
            "goals_conceded_lastN": doc.get("goals_conceded_lastN", NEUTRAL_FORM)
        }
    except AppwriteException:
        return {"goals_scored_lastN": NEUTRAL_FORM, "goals_conceded_lastN": NEUTRAL_FORM}


def upsert_form(db: Databases, team_id, goals_scored_lastN: float, goals_conceded_lastN: float, fixture_id: str) -> None:
    doc_id = str(team_id)
    data = {
        "team_id": int(team_id),
        "goals_scored_lastN": goals_scored_lastN,
        "goals_conceded_lastN": goals_conceded_lastN,
        "last_processed_fixture_id": fixture_id,
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        db.update_document(DATABASE_ID, TEAM_FORM_GLOBAL_COL, doc_id, data)
    except AppwriteException:
        db.create_document(DATABASE_ID, TEAM_FORM_GLOBAL_COL, doc_id, data)


def get_or_default_league_avg(creds: ApiCreds, league_id, season) -> dict:
    doc_id = f"{league_id}_{season}"
    try:
        doc = raw_get_document(creds, DATABASE_ID, LEAGUE_RUNNING_STATS_COL, doc_id)
        return {"goals_sum": doc.get("goals_sum", 0.0), "match_count": doc.get("match_count", 0)}
    except AppwriteException:
        return {"goals_sum": 0.0, "match_count": 0}


def league_avg_value(stats: dict) -> float:
    if stats["match_count"] == 0:
        return NEUTRAL_GOALS_AVG
    return stats["goals_sum"] / (2 * stats["match_count"])


def upsert_league_avg(db: Databases, league_id, season, stats: dict) -> None:
    doc_id = f"{league_id}_{season}"
    data = {
        "league_id": int(league_id), 
        "season": int(season),
        "goals_sum": stats["goals_sum"], 
        "match_count": stats["match_count"],
    }
    try:
        db.update_document(DATABASE_ID, LEAGUE_RUNNING_STATS_COL, doc_id, data)
    except AppwriteException:
        db.create_document(DATABASE_ID, LEAGUE_RUNNING_STATS_COL, doc_id, data)


def expected_goals(team_attack: float, opp_defense: float, league_avg: float) -> float | None:
    if league_avg is None or league_avg <= 0:
        return None
    return max((team_attack * opp_defense) / league_avg, 0.05)


def update_ewma(previous, new_ratio: float, alpha: float = EWMA_ALPHA) -> float:
    if previous is None:
        return new_ratio
    return alpha * new_ratio + (1 - alpha) * previous


def archive_to_dataset_history(db: Databases, fixture: dict, home_pre: dict, away_pre: dict,
                                home_form_pre: dict, away_form_pre: dict,
                                home_ad_post: dict, away_ad_post: dict,
                                home_form_post: dict, away_form_post: dict) -> None:
    doc_id = str(fixture["$id"])
    data = {
        "fixture_id": doc_id,
        "league_id": fixture["league_id"], 
        "season": fixture["season"], 
        "round": fixture["round"],
        "home_team_id": fixture["home_team_id"], 
        "away_team_id": fixture["away_team_id"],
        "home_attack": home_pre["attack"], 
        "home_defense": home_pre["defense"],
        "away_attack": away_pre["attack"], 
        "away_defense": away_pre["defense"],
        "home_goals_scored_lastN": home_form_pre["goals_scored_lastN"],
        "home_goals_conceded_lastN": home_form_pre["goals_conceded_lastN"],
        "away_goals_scored_lastN": away_form_pre["goals_scored_lastN"],
        "away_goals_conceded_lastN": away_form_pre["goals_conceded_lastN"],
        "home_attack_after": home_ad_post["attack"], 
        "home_defense_after": home_ad_post["defense"],
        "away_attack_after": away_ad_post["attack"], 
        "away_defense_after": away_ad_post["defense"],
        "home_goals_scored_lastN_after": home_form_post["goals_scored_lastN"],
        "home_goals_conceded_lastN_after": home_form_post["goals_conceded_lastN"],
        "away_goals_scored_lastN_after": away_form_post["goals_scored_lastN"],
        "away_goals_conceded_lastN_after": away_form_post["goals_conceded_lastN"],
        "home_goals": fixture["home_goals"], 
        "away_goals": fixture["away_goals"],
    }
    try:
        db.create_document(DATABASE_ID, DATASET_HISTORY_COL, doc_id, data)
    except AppwriteException:
        db.update_document(DATABASE_ID, DATASET_HISTORY_COL, doc_id, data)


def process_one_fixture(creds: ApiCreds, db: Databases, fixture: dict) -> None:
    league_id = fixture["league_id"]
    season = fixture["season"]
    home_id = fixture["home_team_id"]
    away_id = fixture["away_team_id"]
    home_goals = fixture["home_goals"]
    away_goals = fixture["away_goals"]
    fixture_id = str(fixture["$id"])

    home_state = get_or_default_team_state(creds, home_id, league_id, season)
    away_state = get_or_default_team_state(creds, away_id, league_id, season)
    home_ad_pre = attack_defense_from_state(home_state)
    away_ad_pre = attack_defense_from_state(away_state)

    league_stats = get_or_default_league_avg(creds, league_id, season)
    league_avg = league_avg_value(league_stats)

    home_form_pre = get_or_default_form(creds, home_id)
    away_form_pre = get_or_default_form(creds, away_id)

    exp_home_goals = expected_goals(home_ad_pre["attack"], away_ad_pre["defense"], league_avg)
    exp_away_goals = expected_goals(away_ad_pre["attack"], home_ad_pre["defense"], league_avg)

    home_scored_lastN_after = home_form_pre["goals_scored_lastN"]
    away_conceded_lastN_after = away_form_pre["goals_conceded_lastN"]
    if exp_home_goals is not None:
        ratio = home_goals / exp_home_goals
        home_scored_lastN_after = update_ewma(home_form_pre["goals_scored_lastN"], ratio)
        away_conceded_lastN_after = update_ewma(away_form_pre["goals_conceded_lastN"], ratio)

    away_scored_lastN_after = away_form_pre["goals_scored_lastN"]
    home_conceded_lastN_after = home_form_pre["goals_conceded_lastN"]
    if exp_away_goals is not None:
        ratio = away_goals / exp_away_goals
        away_scored_lastN_after = update_ewma(away_form_pre["goals_scored_lastN"], ratio)
        home_conceded_lastN_after = update_ewma(home_form_pre["goals_conceded_lastN"], ratio)

    home_state["for_sum"] += home_goals
    home_state["for_count"] += 1
    home_state["against_sum"] += away_goals
    home_state["against_count"] += 1

    away_state["for_sum"] += away_goals
    away_state["for_count"] += 1
    away_state["against_sum"] += home_goals
    away_state["against_count"] += 1

    league_stats["goals_sum"] += home_goals + away_goals
    league_stats["match_count"] += 1

    upsert_team_state(db, home_id, league_id, season, home_state, fixture_id)
    upsert_team_state(db, away_id, league_id, season, away_state, fixture_id)
    upsert_form(db, home_id, home_scored_lastN_after, home_conceded_lastN_after, fixture_id)
    upsert_form(db, away_id, away_scored_lastN_after, away_conceded_lastN_after, fixture_id)
    upsert_league_avg(db, league_id, season, league_stats)

    home_ad_post = attack_defense_from_state(home_state)
    away_ad_post = attack_defense_from_state(away_state)
    archive_to_dataset_history(
        db, fixture, home_ad_pre, away_ad_pre, home_form_pre, away_form_pre,
        home_ad_post, away_ad_post,
        {"goals_scored_lastN": home_scored_lastN_after, "goals_conceded_lastN": home_conceded_lastN_after},
        {"goals_scored_lastN": away_scored_lastN_after, "goals_conceded_lastN": away_conceded_lastN_after},
    )

    db.update_document(DATABASE_ID, FIXTURES_COL, fixture_id, {"processed": True})



def main(context):
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    sys.stdout.flush()
    sys.stderr.flush()

    try:
        context.log("FUTAS INDUL")
    except:
        print("FUTAS INDUL")
    sys.stdout.flush()
    
    try:
        required_env = [
            "APPWRITE_FUNCTION_API_ENDPOINT",
            "APPWRITE_FUNCTION_PROJECT_ID",
            "FOOTBALL_DATA_ORG_KEY"
        ]
        
        missing_vars = [var for var in required_env if var not in os.environ]
        if missing_vars:
            error_msg = f"Missing env: {', '.join(missing_vars)}"
            try:
                context.error(error_msg)
            except:
                print(f"ERROR: {error_msg}")
            sys.stdout.flush()
            return {"error": error_msg}
        
        client = Client()
        
        api_endpoint = os.environ["APPWRITE_FUNCTION_API_ENDPOINT"]
        project_id = os.environ["APPWRITE_FUNCTION_PROJECT_ID"]
        
        client.set_endpoint(api_endpoint)
        client.set_project(project_id)
        
        api_key = None
        
        try:
            if hasattr(context, 'bearer_token') and context.bearer_token:
                api_key = context.bearer_token
                context.log("Using bearer token")
        except:
            pass
        
        if not api_key:
            try:
                if hasattr(context, 'headers'):
                    api_key = context.headers.get("x-appwrite-key", "")
                    if api_key:
                        context.log("API key from header")
            except:
                pass
        
        if not api_key:
            try:
                if hasattr(context, 'req') and hasattr(context.req, 'headers'):
                    api_key = context.req.headers.get("x-appwrite-key", "")
                    if api_key:
                        context.log("API key from req headers")
            except:
                pass
        
        if not api_key:
            api_key = os.environ.get("APPWRITE_FUNCTION_API_KEY", "")
            if api_key:
                context.log("API key from env")
        
        if not api_key:
            try:
                if hasattr(context, 'jwt') and context.jwt:
                    api_key = context.jwt
                    context.log("Using JWT token")
            except:
                pass
        
        if not api_key:
            api_key = os.environ.get("APPWRITE_FUNCTION_API_KEY", "")
            if api_key:
                context.log("Function API key használata")
        
        if not api_key:
            error_msg = "No API keys found"
            try:
                context.error(error_msg)
            except:
                print(f"ERROR: {error_msg}")
            sys.stdout.flush()
            return {"error": error_msg}
        
        client.set_key(api_key)
        
        try:
            context.log(f"Authentication completed (key length: {len(api_key)})")
        except:
            print(f"Authentication completed (key length: {len(api_key)})")
        sys.stdout.flush()

        db = Databases(client)
        
        creds = ApiCreds(api_endpoint, project_id, api_key)

        if os.environ.get("SEED_LEAGUES_CONFIG", "false").lower() == "true":
            seeded = 0
            for code, (api_football_league_id, name) in LEAGUE_ID_MAP.items():
                data = {
                    "competition_code": code,
                    "competition_name": name,
                    "league_id": api_football_league_id,
                    "active": True,
                }
                try:
                    db.update_document(DATABASE_ID, LEAGUES_CONFIG_COL, code, data)
                except AppwriteException:
                    db.create_document(DATABASE_ID, LEAGUES_CONFIG_COL, code, data)
                seeded += 1
            try:
                context.log(f"SEED DONE: {seeded} league updated to the collection")
            except:
                print(f"SEED DONE: {seeded} league updated to the collection")
            sys.stdout.flush()
            return {"seeded_leagues": seeded}

        if os.environ.get("RUN_PROCESSED_BACKFILL", "false").lower() == "true":
            chunk_size = int(os.environ.get("BACKFILL_CHUNK_SIZE", "500"))
            time_budget_s = int(os.environ.get("BACKFILL_TIME_BUDGET_S", "800"))
            started_at = time.monotonic()

            total_updated = 0
            rounds = 0
            
            while True:
                if time.monotonic() - started_at > time_budget_s:
                    try:
                        context.log(f"BACKFILL: timeout, {total_updated} fixed, run again!")
                    except:
                        print(f"BACKFILL: timeout, {total_updated} fixed, run again!")
                    sys.stdout.flush()
                    break

                try:
                    result = db.update_documents(
                        DATABASE_ID, FIXTURES_COL,
                        data={"processed": False},
                        queries=[Query.is_null("processed"), Query.limit(chunk_size)],
                    )
                    
                    if isinstance(result, dict):
                        updated_this_round = result.get("total", 0)
                    else:
                        updated_this_round = getattr(result, "total", 0)
                    
                    rounds += 1
                    total_updated += updated_this_round
                    try:
                        context.log(f"BACKFILL: #{rounds}: {updated_this_round} fixed ({total_updated} total)")
                    except:
                        print(f"BACKFILL: #{rounds}: {updated_this_round} fixed ({total_updated} total)")
                    sys.stdout.flush()

                    if updated_this_round == 0:
                        try:
                            context.log(f"BACKFILL Done: {total_updated} fixed, done")
                        except:
                            print(f"BACKFILL Done: {total_updated} fixed, done")
                        sys.stdout.flush()
                        break
                        
                except Exception as e:
                    try:
                        context.error(f"BACKFILL error: {e}")
                    except:
                        print(f"BACKFILL error: {e}")
                    sys.stdout.flush()
                    break

            return {"backfill_total_updated": total_updated, "rounds": rounds}

        leagues = get_leagues_to_track(creds)
        try:
            context.log(f"LEAGUES TO FOLLOW: {len(leagues)} db -> {[l['competition_code'] for l in leagues]}")
        except:
            print(f"LEAGUES TO FOLLOW: {len(leagues)} db -> {[l['competition_code'] for l in leagues]}")
        sys.stdout.flush()

        if not leagues:
            try:
                context.log("No active leagues")
            except:
                print("No active leagues")
            sys.stdout.flush()
            return {
                "total_fixtures_found": 0, "total_fixtures_written": 0,
                "total_results_found": 0, "total_results_inserted": 0,
                "state_processed": 0, "leagues_processed": 0, "leagues_failed": [],
            }

        try:
            team_mapping = load_team_mapping(TEAM_MAPPING_PATH)
        except FileNotFoundError:
            error_msg = f"team_mapping.csv not found here: {TEAM_MAPPING_PATH}"
            try:
                context.error(error_msg)
            except:
                print(f"ERROR: {error_msg}")
            sys.stdout.flush()
            return {"error": "team_mapping.csv not found"}

        run_started_at = time.monotonic()
        deadline = run_started_at + TIME_BUDGET_S
        all_fixture_rows = []
        all_result_rows = []
        failed_leagues = []
        time_budget_hit = False

        for league in leagues:
            if time.monotonic() > deadline:
                remaining = [l["competition_code"] for l in leagues[leagues.index(league):]]
                try:
                    context.log(f"Timeout ({TIME_BUDGET_S}s) - skipped leagues: {remaining}")
                except:
                    print(f"Timeout ({TIME_BUDGET_S}s) - skipped leagues: {remaining}")
                sys.stdout.flush()
                time_budget_hit = True
                break

            code = league["competition_code"]
            name = league["competition_name"]
            try:
                all_fixture_rows.extend(fetch_upcoming_fixtures(context, code, name, deadline=deadline))
                all_result_rows.extend(fetch_recent_results(context, code, name, deadline=deadline))
            except Exception as exc:
                try:
                    context.log(f"  ERROR ({code}): {type(exc).__name__}: {exc}")
                    context.error(f"Querying league {code} failed: {exc}")
                except:
                    print(f"  ERROR ({code}): {type(exc).__name__}: {exc}")
                sys.stdout.flush()
                failed_leagues.append(code)

        fixtures_written = 0
        for row in all_fixture_rows:
            try:
                upsert_document(db, UPCOMING_FIXTURES_COL, str(row["match_id"]), row)
                fixtures_written += 1
            except Exception as exc:
                try:
                    context.log(f"  ERROR during writing fixture (match_id={row['match_id']}): {type(exc).__name__}: {exc}")
                    context.error(f"Saving fixture {row['match_id']} failed: {exc}")
                except:
                    print(f"  ERROR during writing fixture (match_id={row['match_id']}): {type(exc).__name__}: {exc}")
                sys.stdout.flush()

        new_fixture_docs = result_rows_to_fixture_docs(context, all_result_rows, team_mapping)
        results_inserted = 0
        for doc in new_fixture_docs:
            try:
                if insert_fixture_if_new(db, doc["doc_id"], doc["data"]):
                    results_inserted += 1
            except Exception as exc:
                try:
                    context.log(f"  ERROR during inserting result (match_id={doc['doc_id']}): {type(exc).__name__}: {exc}")
                    context.error(f"Inserting result {doc['doc_id']} failed: {exc}")
                except:
                    print(f"  ERROR during inserting result (match_id={doc['doc_id']}): {type(exc).__name__}: {exc}")
                sys.stdout.flush()

        try:
            context.log(f"RESULTS: {results_inserted}/{len(new_fixture_docs)}new rows to fictures collection ")
        except:
            print(f"RESULTS: {results_inserted}/{len(new_fixture_docs)}new rows to fictures collection ")
        sys.stdout.flush()

        pending = raw_list_documents(
            creds, DATABASE_ID, FIXTURES_COL,
            queries=[
                Query.equal("processed", False),
                Query.is_not_null("home_goals"),
                Query.is_not_null("away_goals"),
                Query.order_asc("date"),
                Query.order_asc("$id"),
                Query.limit(STATE_BATCH_SIZE),
            ],
        )
        
        pending_total = pending.get("total", 0)
        pending_docs = pending.get("documents", [])
        
        try:
            context.log(f"PENDING TOTAL: {pending_total}, IN THIS BATCH: {len(pending_docs)}")
        except:
            print(f"PENDING TOTAL: {pending_total}, IN THIS BATCH: {len(pending_docs)}")
        sys.stdout.flush()

        state_processed = 0
        for fixture in pending_docs:
            if time.monotonic() > deadline:
                try:
                    context.log(f"TIMEOUT ({TIME_BUDGET_S}s)"
                                f"{state_processed}/{len(pending_docs)}")
                except:
                    print(f"TIMEOUT ({TIME_BUDGET_S}s)"
                          f"{state_processed}/{len(pending_docs)}")
                sys.stdout.flush()
                time_budget_hit = True
                break
            try:
                process_one_fixture(creds, db, fixture)
                state_processed += 1
            except Exception as exc:
                fixture_ref = fixture.get("$id") or "???"
                try:
                    context.log(f"ERROR fixture {fixture_ref}: {type(exc).__name__}: {exc}")
                    context.error(f"Fixture {fixture_ref} processing wrror: {exc}")
                except:
                    print(f"ERROR fixture {fixture_ref}: {type(exc).__name__}: {exc}")
                sys.stdout.flush()
                break

        response_data = {
            "total_fixtures_found": len(all_fixture_rows),
            "total_fixtures_written": fixtures_written,
            "total_results_found": len(all_result_rows),
            "total_results_inserted": results_inserted,
            "state_pending_total": pending_total,
            "state_processed": state_processed,
            "leagues_processed": len(leagues) - len(failed_leagues),
            "leagues_failed": failed_leagues,
            "time_budget_hit": time_budget_hit,
        }
        
        try:
            context.log(f"DONE: {response_data}")
        except:
            print(f"DONE: {response_data}")
        sys.stdout.flush()
            
        return response_data

    except Exception as e:
        error_msg = f"Fatal error in main: {type(e).__name__}: {e}"
        try:
            context.error(error_msg)
        except:
            print(f"ERROR: {error_msg}")
        sys.stdout.flush()
        return {"error": str(e)}