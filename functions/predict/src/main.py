import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
from appwrite.client import Client
from appwrite.exception import AppwriteException
from appwrite.query import Query
from appwrite.services.databases import Databases

from predict_live import (
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    TEAM_FORM_STATE_PATH,
    TEAM_MAPPING_PATH,
    UPCOMING_FIXTURES_PATH,
    build_model_features,
    infer_season,
    load_next_round_fixtures,
    load_team_form_state,
    load_team_mapping,
    team_features,
)
from train_bivariate import FEATURE_COLUMNS, predict_match

DATABASE_ID = os.environ["DATABASE_ID"]
PREDICTIONS_COLLECTION_ID = os.environ["PREDICTIONS_COLLECTION_ID"]


def upsert_document(databases: Databases, doc_id: str, data: dict) -> None:
    try:
        databases.update_document(
            database_id=DATABASE_ID,
            collection_id=PREDICTIONS_COLLECTION_ID,
            document_id=doc_id,
            data=data,
        )
    except AppwriteException as e:
        if e.code == 404:
            databases.create_document(
                database_id=DATABASE_ID,
                collection_id=PREDICTIONS_COLLECTION_ID,
                document_id=doc_id,
                data=data,
            )
        else:
            raise


def delete_stale_predictions(databases: Databases, context, keep_ids: set) -> int:
    """Deletes prediction documents whose $id (=match_id) is not among the
    freshly written ones - i.e. matches that dropped out of the next round
    (postponed, already played, etc.)."""
    deleted = 0
    cursor = None
    while True:
        queries = [Query.limit(100)]
        if cursor:
            queries.append(Query.cursor_after(cursor))
        result = databases.list_documents(
            database_id=DATABASE_ID,
            collection_id=PREDICTIONS_COLLECTION_ID,
            queries=queries,
        )
        docs = result["documents"] if isinstance(result, dict) else result.documents
        if not docs:
            break
        for doc in docs:
            doc_id = doc["$id"] if isinstance(doc, dict) else doc.id
            if doc_id not in keep_ids:
                databases.delete_document(
                    database_id=DATABASE_ID,
                    collection_id=PREDICTIONS_COLLECTION_ID,
                    document_id=doc_id,
                )
                deleted += 1
        if len(docs) < 100:
            break
        cursor = docs[-1]["$id"] if isinstance(docs[-1], dict) else docs[-1].id
    context.log(f"Stale predictions removed: {deleted} document(s) deleted.")
    return deleted


def main(context):
    client = (
        Client()
        .set_endpoint(os.environ["APPWRITE_FUNCTION_API_ENDPOINT"])
        .set_project(os.environ["APPWRITE_FUNCTION_PROJECT_ID"])
        .set_key(os.environ.get("APPWRITE_API_KEY", context.req.headers.get("x-appwrite-key", "")))
    )
    databases = Databases(client)

    saved_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    if saved_feature_columns != FEATURE_COLUMNS:
        return context.res.json({"ok": False, "error": "feature_columns mismatch"}, 500)

    team_form_state = load_team_form_state(TEAM_FORM_STATE_PATH)
    team_mapping = load_team_mapping(TEAM_MAPPING_PATH)
    fixtures = load_next_round_fixtures(UPCOMING_FIXTURES_PATH)
    model = joblib.load(MODEL_PATH)

    context.log(f"{len(fixtures)} match in the next rounds.")

    written, skipped = 0, 0
    kept_ids = set()
    for fx in fixtures:
        code = fx["competition_code"]
        home_map = team_mapping.get((code, int(fx["home_team_id"])))
        away_map = team_mapping.get((code, int(fx["away_team_id"])))

        if not home_map or not home_map["api_football_team_id"] or \
           not away_map or not away_map["api_football_team_id"]:
            context.log(f"  [skipped] {fx['home_team_name']} vs {fx['away_team_name']} ({code}) - not mapped")
            skipped += 1
            continue

        league_id = int(home_map["league_id"])
        match_date = datetime.fromisoformat(fx["utc_date"].replace("Z", "+00:00"))
        season = infer_season(match_date)

        home_feats = team_features(
            team_form_state, int(home_map["api_football_team_id"]), league_id, season, match_date
        )
        away_feats = team_features(
            team_form_state, int(away_map["api_football_team_id"]), league_id, season, match_date
        )
        home_model_feats, away_model_feats = build_model_features(home_feats, away_feats)

        probs = predict_match(model, home_model_feats, away_model_feats)
        outcome_probs = {"H": probs["home_win_prob"], "D": probs["draw_prob"], "A": probs["away_win_prob"]}
        decision = max(outcome_probs, key=outcome_probs.get)

        doc_id = str(fx["match_id"])
        upsert_document(
            databases, doc_id,
            data={
                "competition_code": code,
                "league_id": league_id,
                "season": season,
                "match_id": int(fx["match_id"]),
                "utc_date": fx["utc_date"],
                "matchday": int(fx["matchday"]) if fx["matchday"] else None,
                "home_team_id": int(fx["home_team_id"]),
                "away_team_id": int(fx["away_team_id"]),
                "api_football_home_id": int(home_map["api_football_team_id"]),
                "api_football_away_id": int(away_map["api_football_team_id"]),
                "home_team_name": fx["home_team_name"],
                "away_team_name": fx["away_team_name"],
                "home_team_crest": f"https://media.api-sports.io/football/teams/{home_map['api_football_team_id']}.png",
                "away_team_crest": f"https://media.api-sports.io/football/teams/{away_map['api_football_team_id']}.png",
                "exp_home_goals": probs["expected_home_goals"],
                "exp_away_goals": probs["expected_away_goals"],
                "home_win_prob": probs["home_win_prob"],
                "draw_prob": probs["draw_prob"],
                "away_win_prob": probs["away_win_prob"],
                "decision": decision,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        kept_ids.add(doc_id)
        written += 1

    delete_stale_predictions(databases, context, kept_ids)

    context.log(f"Done: {written} prediction saved, {skipped} match skipped.")
    return context.res.json({"ok": True, "written": written, "skipped": skipped})