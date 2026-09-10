import pandas as pd


def get_fixture_info(fixtures: pd.DataFrame, fixture_id: int) -> dict:
    row = fixtures.loc[fixtures["fixture_id"] == fixture_id]

    if row.empty:
        raise ValueError(f"Fixture_id not found: {fixture_id}")

    row = row.iloc[0]
    return {
        "team1": row["home_team_id"],
        "team2": row["away_team_id"],
        "team1_goals": row["home_goals"],
        "team2_goals": row["away_goals"],
        "round": str(row["round"]),
        "league_id": row["league_id"],
        "season": row["season"],
        "date": pd.to_datetime(row["date"]),
    }


NEUTRAL_GOALS_AVG = 1.3


def get_running_attack_defense(team_goal_stats: dict, team_id, league_id, season) -> dict:
    entry = team_goal_stats.get((team_id, league_id, season))

    if entry is None or entry["for_count"] == 0:
        attack = NEUTRAL_GOALS_AVG
    else:
        attack = entry["for_sum"] / entry["for_count"]

    if entry is None or entry["against_count"] == 0:
        defense = NEUTRAL_GOALS_AVG
    else:
        defense = entry["against_sum"] / entry["against_count"]

    return {"attack": attack, "defense": defense}


def get_running_attack_defense_carryover(team_goal_stats: dict, team_id, league_id, season,
                                          max_seasons_back: int = 3) -> dict:
    """Mint get_running_attack_defense, de ha az aktualis szezonra meg nincs
    adat (pl. szezon eleje elo predikciohoz), visszanyul az adott csapat
    legutobbi olyan szezonjara, amihez van lejatszott meccs-statisztika,
    ahelyett hogy NEUTRAL_GOALS_AVG-ra "cold startolna". Csak elo
    predikcional (predict_live.py) hasznald - a tanitasi dataset-epitesnel
    (build_dataset) a szezononkenti cold start szandekos, ott marad
    get_running_attack_defense."""
    for offset in range(max_seasons_back + 1):
        entry = team_goal_stats.get((team_id, league_id, season - offset))
        if entry is not None and entry["for_count"] > 0:
            attack = entry["for_sum"] / entry["for_count"]
            defense = (
                entry["against_sum"] / entry["against_count"]
                if entry["against_count"] > 0 else NEUTRAL_GOALS_AVG
            )
            return {"attack": attack, "defense": defense}

    return {"attack": NEUTRAL_GOALS_AVG, "defense": NEUTRAL_GOALS_AVG}


def update_running_attack_defense(team_goal_stats: dict, team_id, league_id, season,
                                   goals_scored: float, goals_conceded: float) -> None:
    key = (team_id, league_id, season)
    entry = team_goal_stats.setdefault(
        key, {"for_sum": 0.0, "for_count": 0, "against_sum": 0.0, "against_count": 0}
    )
    entry["for_sum"] += goals_scored
    entry["for_count"] += 1
    entry["against_sum"] += goals_conceded
    entry["against_count"] += 1


def get_running_league_avg(league_goal_stats: dict, league_id, season) -> float:
    entry = league_goal_stats.get((league_id, season))
    if entry is None or entry["match_count"] == 0:
        return NEUTRAL_GOALS_AVG
    return entry["goals_sum"] / (2 * entry["match_count"])


def update_running_league_avg(league_goal_stats: dict, league_id, season,
                               home_goals: float, away_goals: float) -> None:
    key = (league_id, season)
    entry = league_goal_stats.setdefault(key, {"goals_sum": 0.0, "match_count": 0})
    entry["goals_sum"] += home_goals + away_goals
    entry["match_count"] += 1


def expected_goals(team_attack: float, opp_defense: float, league_avg) -> float | None:
    if league_avg is None or pd.isna(league_avg) or league_avg <= 0:
        return None
    exp = (team_attack * opp_defense) / league_avg
    return max(exp, 0.05)


def update_ewma(previous: float | None, new_ratio: float, alpha: float = 0.35) -> float:
    if previous is None or pd.isna(previous):
        return new_ratio
    return alpha * new_ratio + (1 - alpha) * previous



def build_prematch_standings_lookup(standings_path: str = "output/standings.csv") -> dict:
    standings = pd.read_csv(standings_path)
    table = standings[standings["stage_type"] == "table"].copy()

    lookup = {}

    for (league_id, season, group), grp in table.groupby(["league_id", "season", "group"], dropna=False, sort=False):
        rounds_in_order = list(dict.fromkeys(grp["round"]))

        round_snapshots = {}
        for round_label, round_grp in grp.groupby("round", sort=False):
            round_snapshots[round_label] = round_grp.sort_values("rank")[["team_id", "points"]].values.tolist()

        for i, round_label in enumerate(rounds_in_order):
            prev_round_label = rounds_in_order[i - 1] if i > 0 else None
            if prev_round_label is None:
                continue

            prev_snapshot = round_snapshots[prev_round_label]
            group_size = len(prev_snapshot)

            for rank_idx, (team_id, points) in enumerate(prev_snapshot):
                points_above = prev_snapshot[rank_idx - 1][1] if rank_idx > 0 else None
                points_below = prev_snapshot[rank_idx + 1][1] if rank_idx < group_size - 1 else None
                lookup[(league_id, season, round_label, team_id)] = {
                    "points": points,
                    "rank": rank_idx + 1,
                    "points_above": points_above,
                    "points_below": points_below,
                    "group_size": group_size,
                }

    return lookup


def get_prematch_motivation(lookup: dict, league_id, season, round_label, team_id) -> float:
    entry = lookup.get((league_id, season, round_label, team_id))
    if entry is None:
        return 0.5

    own_points = entry["points"]
    gap_above = 0 if entry["points_above"] is None else own_points - entry["points_above"]
    gap_below = 0 if entry["points_below"] is None else own_points - entry["points_below"]

    motivation = 1 / (1 + abs(gap_above)) + 1 / (1 + abs(gap_below))
    return round(motivation, 4)




def get_knockout_rounds(standings_path: str = "output/standings.csv") -> set:
    standings = pd.read_csv(standings_path)
    ko = standings[standings["stage_type"] == "knockout"]
    return set(zip(ko["league_id"], ko["season"], ko["round"].astype(str)))


def build_knockout_leg_counts(fixtures: pd.DataFrame, knockout_rounds: set) -> dict:
    counts = {}
    for _, fx in fixtures.iterrows():
        key3 = (fx["league_id"], fx["season"], str(fx["round"]))
        if key3 not in knockout_rounds:
            continue
        pair = frozenset((fx["home_team_id"], fx["away_team_id"]))
        key = key3 + (pair,)
        counts[key] = counts.get(key, 0) + 1
    return counts


def get_knockout_motivation(tie_progress: dict, leg_counts: dict,
                             league_id, season, round_label, pair_key,
                             team_id, opp_id, sensitivity: float = 0.5) -> float:
    total_legs = leg_counts.get((league_id, season, round_label, pair_key), 1)

    if total_legs <= 1:
        return 2.0

    prior = tie_progress.get((league_id, season, round_label, pair_key))
    if prior is None:
        return 1.5

    own_agg = prior.get(team_id, 0)
    opp_agg = prior.get(opp_id, 0)
    diff = abs(own_agg - opp_agg)

    return round(2.0 / (1 + sensitivity * diff), 4)


def update_knockout_tie_progress(tie_progress: dict, league_id, season, round_label, pair_key,
                                  home_id, away_id, home_goals, away_goals) -> None:
    key = (league_id, season, round_label, pair_key)
    entry = tie_progress.setdefault(key, {})
    entry[home_id] = entry.get(home_id, 0) + home_goals
    entry[away_id] = entry.get(away_id, 0) + away_goals




def normalize_motivation_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()

    for col, raw_col in [("home_motivation", "home_motivation_raw"),
                          ("away_motivation", "away_motivation_raw")]:
        dataset[raw_col] = dataset[col]

    for is_ko in [True, False]:
        mask = dataset["is_knockout"] == is_ko
        for col in ["home_motivation", "away_motivation"]:
            dataset.loc[mask, col] = dataset.loc[mask, col].rank(pct=True)

    return dataset


def build_dataset(fixtures: pd.DataFrame, standings_lookup: dict,
                   knockout_rounds: set, knockout_leg_counts: dict,
                   alpha: float = 0.35) -> tuple[pd.DataFrame, dict]:
    played = fixtures.dropna(subset=["home_goals", "away_goals"]).copy()
    played["date"] = pd.to_datetime(played["date"])
    played = played.sort_values("date").reset_index(drop=True)

    ewma_attack: dict[int, float] = {}
    ewma_defense: dict[int, float] = {}
    last_match_date: dict[int, pd.Timestamp] = {}
    team_goal_stats: dict = {}
    league_goal_stats: dict = {}
    knockout_tie_progress: dict = {}

    NEUTRAL_FORM = 1.0
    NEUTRAL_DAYS_SINCE = 365

    rows = []

    for _, fx in played.iterrows():
        league_id = fx["league_id"]
        season = fx["season"]
        home_id = fx["home_team_id"]
        away_id = fx["away_team_id"]
        match_date = fx["date"]
        round_label = str(fx["round"])
        pair_key = frozenset((home_id, away_id))

        home_stats = get_running_attack_defense(team_goal_stats, home_id, league_id, season)
        away_stats = get_running_attack_defense(team_goal_stats, away_id, league_id, season)
        league_avg = get_running_league_avg(league_goal_stats, league_id, season)

        home_attack_form = ewma_attack.get(home_id, NEUTRAL_FORM)
        home_defense_form = ewma_defense.get(home_id, NEUTRAL_FORM)
        away_attack_form = ewma_attack.get(away_id, NEUTRAL_FORM)
        away_defense_form = ewma_defense.get(away_id, NEUTRAL_FORM)

        home_days_since = (
            (match_date - last_match_date[home_id]).total_seconds() / 86400
            if home_id in last_match_date else NEUTRAL_DAYS_SINCE
        )
        away_days_since = (
            (match_date - last_match_date[away_id]).total_seconds() / 86400
            if away_id in last_match_date else NEUTRAL_DAYS_SINCE
        )

        is_knockout = (league_id, season, round_label) in knockout_rounds

        if is_knockout:
            home_motivation = get_knockout_motivation(
                knockout_tie_progress, knockout_leg_counts,
                league_id, season, round_label, pair_key, home_id, away_id
            )
            away_motivation = get_knockout_motivation(
                knockout_tie_progress, knockout_leg_counts,
                league_id, season, round_label, pair_key, away_id, home_id
            )
        else:
            home_motivation = get_prematch_motivation(standings_lookup, league_id, season, round_label, home_id)
            away_motivation = get_prematch_motivation(standings_lookup, league_id, season, round_label, away_id)

        rows.append({
            "fixture_id": fx["fixture_id"],
            "league_id": league_id,
            "season": season,
            "round": round_label,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_attack": home_stats["attack"],
            "home_defense": home_stats["defense"],
            "away_attack": away_stats["attack"],
            "away_defense": away_stats["defense"],
            "home_goals_scored_lastN": home_attack_form,
            "home_goals_conceded_lastN": home_defense_form,
            "away_goals_scored_lastN": away_attack_form,
            "away_goals_conceded_lastN": away_defense_form,
            "home_days_since_last_match": home_days_since,
            "away_days_since_last_match": away_days_since,
            "home_motivation": home_motivation,
            "away_motivation": away_motivation,
            "is_knockout": is_knockout,
            "home_goals": fx["home_goals"],
            "away_goals": fx["away_goals"],
        })

        exp_home_goals = expected_goals(home_stats["attack"], away_stats["defense"], league_avg)
        exp_away_goals = expected_goals(away_stats["attack"], home_stats["defense"], league_avg)

        if exp_home_goals is not None:
            ratio = fx["home_goals"] / exp_home_goals
            ewma_attack[home_id] = update_ewma(ewma_attack.get(home_id), ratio, alpha)
            ewma_defense[away_id] = update_ewma(ewma_defense.get(away_id), ratio, alpha)

        if exp_away_goals is not None:
            ratio = fx["away_goals"] / exp_away_goals
            ewma_attack[away_id] = update_ewma(ewma_attack.get(away_id), ratio, alpha)
            ewma_defense[home_id] = update_ewma(ewma_defense.get(home_id), ratio, alpha)

        last_match_date[home_id] = match_date
        last_match_date[away_id] = match_date


        update_running_attack_defense(team_goal_stats, home_id, league_id, season,
                                       fx["home_goals"], fx["away_goals"])
        update_running_attack_defense(team_goal_stats, away_id, league_id, season,
                                       fx["away_goals"], fx["home_goals"])
        update_running_league_avg(league_goal_stats, league_id, season,
                                   fx["home_goals"], fx["away_goals"])

        if is_knockout:
            update_knockout_tie_progress(
                knockout_tie_progress, league_id, season, round_label, pair_key,
                home_id, away_id, fx["home_goals"], fx["away_goals"]
            )


        home_stats_after = get_running_attack_defense(team_goal_stats, home_id, league_id, season)
        away_stats_after = get_running_attack_defense(team_goal_stats, away_id, league_id, season)

        rows[-1]["home_attack_after"] = home_stats_after["attack"]
        rows[-1]["home_defense_after"] = home_stats_after["defense"]
        rows[-1]["away_attack_after"] = away_stats_after["attack"]
        rows[-1]["away_defense_after"] = away_stats_after["defense"]
        rows[-1]["home_goals_scored_lastN_after"] = ewma_attack.get(home_id, NEUTRAL_FORM)
        rows[-1]["home_goals_conceded_lastN_after"] = ewma_defense.get(home_id, NEUTRAL_FORM)
        rows[-1]["away_goals_scored_lastN_after"] = ewma_attack.get(away_id, NEUTRAL_FORM)
        rows[-1]["away_goals_conceded_lastN_after"] = ewma_defense.get(away_id, NEUTRAL_FORM)

    team_form_state = {
        "ewma_attack": ewma_attack,
        "ewma_defense": ewma_defense,
        "last_match_date": last_match_date,
        "team_goal_stats": team_goal_stats,
        "league_goal_stats": league_goal_stats,
        "knockout_tie_progress": knockout_tie_progress,
    }

    return pd.DataFrame(rows), team_form_state


def _serialize_team_goal_stats(team_goal_stats: dict) -> dict:
    return {
        f"{team_id}:{league_id}:{season}": entry
        for (team_id, league_id, season), entry in team_goal_stats.items()
    }


def _serialize_league_goal_stats(league_goal_stats: dict) -> dict:
    return {
        f"{league_id}:{season}": entry
        for (league_id, season), entry in league_goal_stats.items()
    }


def _serialize_knockout_tie_progress(knockout_tie_progress: dict) -> dict:
    return {
        f"{league_id}:{season}:{round_label}:{'-'.join(str(t) for t in sorted(pair_key))}": entry
        for (league_id, season, round_label, pair_key), entry in knockout_tie_progress.items()
    }


def main():
    import json
    import os

    fixtures = pd.read_csv("output/fixtures.csv")
    standings_lookup = build_prematch_standings_lookup("output/standings.csv")
    knockout_rounds = get_knockout_rounds("output/standings.csv")
    knockout_leg_counts = build_knockout_leg_counts(fixtures, knockout_rounds)

    dataset, team_form_state = build_dataset(
        fixtures, standings_lookup, knockout_rounds, knockout_leg_counts, alpha=0.35
    )
    dataset = normalize_motivation_columns(dataset)

    print(f"Dataset size: {dataset.shape}")
    print(dataset.head())
    dataset.to_csv("output/dataset.csv", index=False)
    print("\nSaved: output/dataset.csv")

    os.makedirs("output", exist_ok=True)
    serializable_state = {
        "ewma_attack": team_form_state["ewma_attack"],
        "ewma_defense": team_form_state["ewma_defense"],
        "last_match_date": {
            team_id: str(date) for team_id, date in team_form_state["last_match_date"].items()
        },
        "team_goal_stats": _serialize_team_goal_stats(team_form_state["team_goal_stats"]),
        "league_goal_stats": _serialize_league_goal_stats(team_form_state["league_goal_stats"]),
        "knockout_tie_progress": _serialize_knockout_tie_progress(team_form_state["knockout_tie_progress"]),
    }
    with open("output/team_form_state.json", "w") as f:
        json.dump(serializable_state, f, indent=2)
    print("Saved: output/team_form_state.json")


if __name__ == "__main__":
    main()