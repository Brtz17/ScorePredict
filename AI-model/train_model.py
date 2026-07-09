import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error


FEATURE_COLUMNS = [
    "team_attack",
    "team_defense",
    "opp_attack",
    "opp_defense",
    "team_goals_lastN",
    "team_conceded_lastN",
    "opp_goals_lastN",
    "opp_conceded_lastN",
    "is_home",
    "days_since_last_match",
]
TARGET_COLUMN = "team_goals"


def to_team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["fixture_id", "league_id", "season", "round"]

    home_rows = pd.DataFrame({
        "fixture_id": df["fixture_id"],
        "league_id": df["league_id"],
        "season": df["season"],
        "round": df["round"],
        "team_attack": df["home_attack"],
        "team_defense": df["home_defense"],
        "opp_attack": df["away_attack"],
        "opp_defense": df["away_defense"],
        "team_goals_lastN": df["home_goals_scored_lastN"],
        "team_conceded_lastN": df["home_goals_conceded_lastN"],
        "opp_goals_lastN": df["away_goals_scored_lastN"],
        "opp_conceded_lastN": df["away_goals_conceded_lastN"],
        "is_home": 1,
        "days_since_last_match": df["home_days_since_last_match"],
        "team_goals": df["home_goals"],
    })

    away_rows = pd.DataFrame({
        "fixture_id": df["fixture_id"],
        "league_id": df["league_id"],
        "season": df["season"],
        "round": df["round"],
        "team_attack": df["away_attack"],
        "team_defense": df["away_defense"],
        "opp_attack": df["home_attack"],
        "opp_defense": df["home_defense"],
        "team_goals_lastN": df["away_goals_scored_lastN"],
        "team_conceded_lastN": df["away_goals_conceded_lastN"],
        "opp_goals_lastN": df["home_goals_scored_lastN"],
        "opp_conceded_lastN": df["home_goals_conceded_lastN"],
        "is_home": 0,
        "days_since_last_match": df["away_days_since_last_match"],
        "team_goals": df["away_goals"],
    })

    long_df = pd.concat([home_rows, away_rows], ignore_index=True)
    return long_df.sort_values(["season", "round", "fixture_id"]).reset_index(drop=True)


def load_dataset(path: str = "output/dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    required_wide_cols = [
        "fixture_id", "league_id", "season", "round",
        "home_attack", "home_defense", "away_attack", "away_defense",
        "home_goals_scored_lastN", "home_goals_conceded_lastN",
        "away_goals_scored_lastN", "away_goals_conceded_lastN",
        "home_days_since_last_match", "away_days_since_last_match",
        "home_goals", "away_goals",
    ]
    missing = [col for col in required_wide_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"dataset.csv is missing expected columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return to_team_perspective(df)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    return X


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):

    fixture_order = (
        df[["fixture_id", "season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round", "fixture_id"])
    )
    split_idx = int(len(fixture_order) * (1 - test_size))
    train_fixtures = set(fixture_order.iloc[:split_idx]["fixture_id"])
    test_fixtures = set(fixture_order.iloc[split_idx:]["fixture_id"])

    train_df = df[df["fixture_id"].isin(train_fixtures)].reset_index(drop=True)
    test_df = df[df["fixture_id"].isin(test_fixtures)].reset_index(drop=True)
    return train_df, test_df


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> PoissonRegressor:
    X_train = prepare_features(train_df)
    y_train = train_df[TARGET_COLUMN]
    X_test = prepare_features(test_df)
    y_test = test_df[TARGET_COLUMN]

    model = PoissonRegressor(alpha=0.01, max_iter=1000)
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 1e-6, None)
    deviance = mean_poisson_deviance(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    baseline_preds = np.clip(test_df["team_goals_lastN"].fillna(train_df["team_goals_lastN"].median()), 1e-6, None)
    baseline_deviance = mean_poisson_deviance(y_test, baseline_preds)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    print(f"Trained model   - Poisson deviance: {deviance:.4f}, MAE: {mae:.4f}")
    print(f"Naive baseline  - Poisson deviance: {baseline_deviance:.4f}, MAE: {baseline_mae:.4f}")
    print("(baseline = team's own recent goal average, no opponent/context)")

    coef_table = pd.Series(model.coef_, index=FEATURE_COLUMNS).sort_values(key=abs, ascending=False)
    print("\nTop features (by coefficient magnitude):")
    print(coef_table.to_string())
    print(f"\nIntercept (baseline goal level): {model.intercept_:.4f}")

    return model


def save_model(model: PoissonRegressor, output_dir: str = "output/models"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/team_goals_model.pkl")
    joblib.dump(FEATURE_COLUMNS, f"{output_dir}/feature_columns.pkl")
    print(f"\nModel saved: {output_dir}/team_goals_model.pkl")


def predict_match(model: PoissonRegressor, home_features: dict, away_features: dict) -> dict:
    home_row = {**home_features, "is_home": 1}
    away_row = {**away_features, "is_home": 0}

    X = pd.DataFrame([home_row, away_row])[FEATURE_COLUMNS]
    preds = np.clip(model.predict(X), 1e-6, None)

    return {"expected_home_goals": preds[0], "expected_away_goals": preds[1]}


def dixon_coles_tau(x: int, y: int, lambda_: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lambda_ * mu * rho
    if x == 0 and y == 1:
        return 1 + lambda_ * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def fit_dixon_coles_rho(model: PoissonRegressor, train_df: pd.DataFrame, bounds: tuple = (-0.2, 0.2)) -> float:
    from scipy.stats import poisson
    from scipy.optimize import minimize_scalar

    observed = []
    for fixture_id, group in train_df.groupby("fixture_id"):
        home_row = group[group["is_home"] == 1].iloc[0]
        away_row = group[group["is_home"] == 0].iloc[0]

        home_features = home_row[[c for c in FEATURE_COLUMNS if c != "is_home"]].to_dict()
        away_features = away_row[[c for c in FEATURE_COLUMNS if c != "is_home"]].to_dict()
        pred = predict_match(model, home_features, away_features)

        observed.append((
            int(home_row["team_goals"]),
            int(away_row["team_goals"]),
            pred["expected_home_goals"],
            pred["expected_away_goals"],
        ))

    def negative_log_likelihood(rho: float) -> float:
        total = 0.0
        for x, y, lam, mu in observed:
            tau = dixon_coles_tau(x, y, lam, mu, rho)
            prob = max(tau * poisson.pmf(x, lam) * poisson.pmf(y, mu), 1e-10)
            total -= np.log(prob)
        return total

    result = minimize_scalar(negative_log_likelihood, bounds=bounds, method="bounded")
    return result.x


def score_matrix_probabilities(expected_home: float, expected_away: float, max_goals: int = 6, rho: float = 0.0) -> dict:
    from scipy.stats import poisson

    home_probs = [poisson.pmf(i, expected_home) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, expected_away) for i in range(max_goals + 1)]

    matrix = np.outer(home_probs, away_probs)

    if rho != 0.0:
        for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            matrix[x, y] *= dixon_coles_tau(x, y, expected_home, expected_away, rho)
        matrix = matrix / matrix.sum()

    return {
        "home_win_prob": np.sum(np.tril(matrix, -1)),
        "draw_prob": np.sum(np.diag(matrix)),
        "away_win_prob": np.sum(np.triu(matrix, 1)),
        "score_matrix": matrix,
    }


def main():
    long_df = load_dataset()
    print(f"Team-level dataset size: {long_df.shape} (original matches: {long_df['fixture_id'].nunique()})")

    train_df, test_df = time_based_split(long_df, test_size=0.2)
    print(f"Train: {len(train_df)} rows ({train_df['fixture_id'].nunique()} matches), "
          f"Test: {len(test_df)} rows ({test_df['fixture_id'].nunique()} matches)\n")

    model = train_and_evaluate(train_df, test_df)
    save_model(model)

    last_fixture_id = test_df.sort_values(["season", "round"]).iloc[-1]["fixture_id"]
    match_rows = test_df[test_df["fixture_id"] == last_fixture_id]
    home_row = match_rows[match_rows["is_home"] == 1].iloc[0]
    away_row = match_rows[match_rows["is_home"] == 0].iloc[0]

    home_features = home_row[[c for c in FEATURE_COLUMNS if c != "is_home"]].to_dict()
    away_features = away_row[[c for c in FEATURE_COLUMNS if c != "is_home"]].to_dict()

    prediction = predict_match(model, home_features, away_features)
    probs = score_matrix_probabilities(prediction["expected_home_goals"], prediction["expected_away_goals"])

    print("\nSample prediction (last test match):")
    print(f"  Expected goals: home {prediction['expected_home_goals']:.2f} - "
          f"away {prediction['expected_away_goals']:.2f}")
    print(f"  Actual score: {home_row['team_goals']:.0f} - {away_row['team_goals']:.0f}")
    print(f"  Home/Draw/Away probability: {probs['home_win_prob']:.1%} / "
          f"{probs['draw_prob']:.1%} / {probs['away_win_prob']:.1%}")


if __name__ == "__main__":
    main()