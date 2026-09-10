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

FOOTBALL_DATA_ORG_LEAGUES = {code: name for code, (_, name) in LEAGUE_ID_MAP.items()}


def api_football_league_id(competition_code: str) -> int | None:
    entry = LEAGUE_ID_MAP.get(competition_code)
    return entry[0] if entry else None
