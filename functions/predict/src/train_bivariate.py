import os

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
from sklearn.preprocessing import StandardScaler



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
NON_HOME_COLS = [c for c in FEATURE_COLUMNS if c != "is_home"]


def to_team_perspective(df: pd.DataFrame) -> pd.DataFrame:
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


def load_dataset(path: str = "../dataset.csv") -> pd.DataFrame:
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


def wide_from_long(long_df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    non_home_cols = [c for c in feature_columns if c != "is_home"]
    records = []
    for fixture_id, group in long_df.groupby("fixture_id"):
        home_row = group[group["is_home"] == 1].iloc[0]
        away_row = group[group["is_home"] == 0].iloc[0]
        rec = {
            "fixture_id": fixture_id,
            "season": home_row["season"],
            "round": home_row["round"],
            "home_goals": home_row["team_goals"],
            "away_goals": away_row["team_goals"],
        }
        for c in non_home_cols:
            rec[f"home_{c}"] = home_row[c]
            rec[f"away_{c}"] = away_row[c]
        records.append(rec)
    wide = pd.DataFrame(records).sort_values(["season", "round", "fixture_id"]).reset_index(drop=True)
    return wide




def _log_term(x: int, y: int, i: int, l1: float, l2: float, l3: float) -> float:
    return (
        (x - i) * np.log(l1) - gammaln(x - i + 1)
        + (y - i) * np.log(l2) - gammaln(y - i + 1)
        + i * np.log(l3) - gammaln(i + 1)
    )


def bp_pmf(x: int, y: int, l1: float, l2: float, l3: float) -> float:
    m = min(x, y)
    log_terms = np.array([_log_term(x, y, i, l1, l2, l3) for i in range(m + 1)])
    max_log = log_terms.max()
    total = np.exp(max_log) * np.sum(np.exp(log_terms - max_log))
    return float(np.exp(-(l1 + l2 + l3)) * total)


def expected_shared_component(x: int, y: int, l1: float, l2: float, l3: float) -> float:
    m = min(x, y)
    if m == 0:
        return 0.0
    log_terms = np.array([_log_term(x, y, i, l1, l2, l3) for i in range(m + 1)])
    max_log = log_terms.max()
    weights = np.exp(log_terms - max_log)
    weights_sum = weights.sum()
    if weights_sum <= 0:
        return 0.0
    idx = np.arange(m + 1)
    return float(np.sum(idx * weights) / weights_sum)


def dixon_coles_tau(x: int, y: int, l1: float, l2: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - (l1 * l2 * rho)
    if x == 0 and y == 1:
        return 1.0 + (l1 * rho)
    if x == 1 and y == 0:
        return 1.0 + (l2 * rho)
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def build_score_matrix_bp(l1: float, l2: float, l3: float, max_goals: int = 6) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            matrix[x, y] = bp_pmf(x, y, l1, l2, l3)
    matrix = matrix / matrix.sum()
    return matrix


def build_score_matrix_bp_dc(l1: float, l2: float, l3: float, rho: float, max_goals: int = 6) -> np.ndarray:
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    marg1 = l1 + l3
    marg2 = l2 + l3
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            base = bp_pmf(x, y, l1, l2, l3)
            tau = dixon_coles_tau(x, y, marg1, marg2, rho)
            matrix[x, y] = max(base * tau, 0.0)
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    return matrix


def outcome_probs_from_matrix(matrix: np.ndarray) -> dict:
    return {
        "home_win_prob": float(np.sum(np.tril(matrix, -1))),
        "draw_prob": float(np.sum(np.diag(matrix))),
        "away_win_prob": float(np.sum(np.triu(matrix, 1))),
    }


class BivariatePoissonModel:
    def __init__(self, feature_columns: list, alpha: float = 0.0001, max_iter_em: int = 50,
                 tol: float = 1e-4, use_dixon_coles: bool = True, rho_bounds: tuple = (-0.4, 0.4)):
        self.feature_columns = feature_columns
        self.l1_features = [f"home_{c}" for c in feature_columns]
        self.l2_features = [f"away_{c}" for c in feature_columns]
        self.alpha = alpha
        self.max_iter_em = max_iter_em
        self.tol = tol
        self.use_dixon_coles = use_dixon_coles
        self.rho_bounds = rho_bounds
        self.model1_ = None
        self.model2_ = None
        self.scaler1_ = None
        self.scaler2_ = None
        self.lambda3_const_ = None
        self.rho_ = 0.0
        self.history_ = []

    def _predict_l1(self, X1_raw):
        X1 = self.scaler1_.transform(X1_raw)
        return np.clip(self.model1_.predict(X1), 1e-6, None)

    def _predict_l2(self, X2_raw):
        X2 = self.scaler2_.transform(X2_raw)
        return np.clip(self.model2_.predict(X2), 1e-6, None)

    def _fit_rho(self, X1_raw, X2_raw, x, y) -> float:
        l1 = self._predict_l1(X1_raw)
        l2 = self._predict_l2(X2_raw)
        l3 = self.lambda3_const_
        marg1 = l1 + l3
        marg2 = l2 + l3
        x_int = x.astype(int)
        y_int = y.astype(int)

        def neg_log_likelihood(rho: float) -> float:
            total = 0.0
            for i in range(len(x_int)):
                base = bp_pmf(int(x_int[i]), int(y_int[i]), l1[i], l2[i], l3)
                tau = dixon_coles_tau(int(x_int[i]), int(y_int[i]), marg1[i], marg2[i], rho)
                p = max(base * tau, 1e-12)
                total += np.log(p)
            return -total

        result = minimize_scalar(
            neg_log_likelihood, bounds=self.rho_bounds, method="bounded"
        )
        return float(result.x)

    def fit(self, wide_df: pd.DataFrame):
        X1_raw = wide_df[self.l1_features].values
        X2_raw = wide_df[self.l2_features].values
        x = wide_df["home_goals"].values.astype(int)
        y = wide_df["away_goals"].values.astype(int)
        n = len(wide_df)

        self.scaler1_ = StandardScaler().fit(X1_raw)
        self.scaler2_ = StandardScaler().fit(X2_raw)
        X1 = self.scaler1_.transform(X1_raw)
        X2 = self.scaler2_.transform(X2_raw)

        self.model1_ = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(X1, x)
        self.model2_ = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(X2, y)
        self.lambda3_const_ = max(np.minimum(x, y).mean() * 0.5, 1e-3)

        prev_ll = -np.inf
        for iteration in range(self.max_iter_em):
            l1 = self._predict_l1(X1_raw)
            l2 = self._predict_l2(X2_raw)
            l3 = np.full(n, self.lambda3_const_)

            # E-lepes
            z = np.array([
                expected_shared_component(x[i], y[i], l1[i], l2[i], l3[i])
                for i in range(n)
            ])
            z = np.clip(z, 0, np.minimum(x, y))

            resid_x = np.clip(x - z, 0, None)
            resid_y = np.clip(y - z, 0, None)
            self.model1_ = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(X1, resid_x)
            self.model2_ = PoissonRegressor(alpha=self.alpha, max_iter=500).fit(X2, resid_y)
            self.lambda3_const_ = max(z.mean(), 1e-3)

            l1_new = self._predict_l1(X1_raw)
            l2_new = self._predict_l2(X2_raw)
            ll = sum(
                np.log(max(bp_pmf(int(x[i]), int(y[i]), l1_new[i], l2_new[i], self.lambda3_const_), 1e-12))
                for i in range(n)
            )
            self.history_.append(ll)

            if abs(ll - prev_ll) < self.tol * abs(prev_ll if prev_ll != -np.inf else 1):
                break
            prev_ll = ll

        if self.use_dixon_coles:
            self.rho_ = self._fit_rho(X1_raw, X2_raw, x, y)
        else:
            self.rho_ = 0.0

        return self

    def predict_lambdas(self, home_features: dict, away_features: dict) -> tuple:
        X1_raw = np.array([[home_features[c] for c in self.feature_columns]])
        X2_raw = np.array([[away_features[c] for c in self.feature_columns]])
        l1 = float(self._predict_l1(X1_raw)[0])
        l2 = float(self._predict_l2(X2_raw)[0])
        return l1, l2, self.lambda3_const_

    def predict_match_probs(self, home_features: dict, away_features: dict, max_goals: int = 6) -> dict:
        l1, l2, l3 = self.predict_lambdas(home_features, away_features)
        if self.use_dixon_coles:
            matrix = build_score_matrix_bp_dc(l1, l2, l3, self.rho_, max_goals=max_goals)
        else:
            matrix = build_score_matrix_bp(l1, l2, l3, max_goals=max_goals)
        probs = outcome_probs_from_matrix(matrix)
        probs["expected_home_goals"] = l1 + l3
        probs["expected_away_goals"] = l2 + l3
        probs["lambda3"] = l3
        probs["rho"] = self.rho_
        probs["score_matrix"] = matrix
        return probs

    def feature_importance(self) -> pd.DataFrame:
        rows = []
        for i, feat in enumerate(self.feature_columns):
            coef_home = self.model1_.coef_[i]
            coef_away = self.model2_.coef_[i]
            rows.append({
                "feature": feat,
                "coef_home_goals (lambda1)": coef_home,
                "coef_away_goals (lambda2)": coef_away,
                "effect_home_per_1sd (x)": np.exp(coef_home),
                "effect_away_per_1sd (x)": np.exp(coef_away),
            })
        df = pd.DataFrame(rows)
        df["_abs_total"] = df["coef_home_goals (lambda1)"].abs() + df["coef_away_goals (lambda2)"].abs()
        return (
            df.sort_values("_abs_total", ascending=False)
            .drop(columns="_abs_total")
            .reset_index(drop=True)
        )


def predict_match(model: BivariatePoissonModel, home_features: dict, away_features: dict) -> dict:
    return model.predict_match_probs(home_features, away_features)



def actual_outcome(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def majority_class_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    train_outcomes = []
    for fixture_id, group in train_df.groupby("fixture_id"):
        home_row = group[group["is_home"] == 1].iloc[0]
        away_row = group[group["is_home"] == 0].iloc[0]
        train_outcomes.append(actual_outcome(home_row["team_goals"], away_row["team_goals"]))

    majority = pd.Series(train_outcomes).mode()[0]

    test_outcomes = []
    for fixture_id, group in test_df.groupby("fixture_id"):
        home_row = group[group["is_home"] == 1].iloc[0]
        away_row = group[group["is_home"] == 0].iloc[0]
        test_outcomes.append(actual_outcome(home_row["team_goals"], away_row["team_goals"]))

    return (pd.Series(test_outcomes) == majority).mean()


def evaluate_outcome_accuracy(model: BivariatePoissonModel, df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    wide_df = wide_from_long(df, FEATURE_COLUMNS)
    records = []
    for _, row in wide_df.iterrows():
        home_features = {c: row[f"home_{c}"] for c in NON_HOME_COLS}
        away_features = {c: row[f"away_{c}"] for c in NON_HOME_COLS}
        probs = model.predict_match_probs(home_features, away_features)
        outcome_probs = {"H": probs["home_win_prob"], "D": probs["draw_prob"], "A": probs["away_win_prob"]}
        predicted = max(outcome_probs, key=outcome_probs.get)

        actual = actual_outcome(row["home_goals"], row["away_goals"])
        records.append({
            "fixture_id": row["fixture_id"],
            "predicted": predicted,
            "actual": actual,
            "correct": predicted == actual,
            "home_win_prob": probs["home_win_prob"],
            "draw_prob": probs["draw_prob"],
            "away_win_prob": probs["away_win_prob"],
        })
    result_df = pd.DataFrame(records)
    return result_df["correct"].mean(), result_df



def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        alpha: float = 0.01, max_iter_em: int = 50) -> BivariatePoissonModel:
    """Betanitja a modellt, kiirja a fobb metrikakat, es visszaadja a modellt."""
    wide_train = wide_from_long(train_df, FEATURE_COLUMNS)
    wide_test = wide_from_long(test_df, FEATURE_COLUMNS)

    model = BivariatePoissonModel(feature_columns=NON_HOME_COLS, alpha=alpha, max_iter_em=max_iter_em)
    model.fit(wide_train)

    records = []
    for _, row in wide_test.iterrows():
        home_features = {c: row[f"home_{c}"] for c in NON_HOME_COLS}
        away_features = {c: row[f"away_{c}"] for c in NON_HOME_COLS}
        probs = model.predict_match_probs(home_features, away_features)
        records.append({
            "exp_home": probs["expected_home_goals"],
            "exp_away": probs["expected_away_goals"],
            "true_home": row["home_goals"],
            "true_away": row["away_goals"],
        })
    pred_df = pd.DataFrame(records)

    true_all = pd.concat([pred_df["true_home"], pred_df["true_away"]])
    pred_all = np.clip(pd.concat([pred_df["exp_home"], pred_df["exp_away"]]), 1e-6, None)
    deviance = mean_poisson_deviance(true_all, pred_all)
    mae = mean_absolute_error(true_all, pred_all)

    baseline_preds = np.clip(
        test_df["team_goals_lastN"].fillna(train_df["team_goals_lastN"].median()), 1e-6, None
    )
    baseline_deviance = mean_poisson_deviance(test_df["team_goals"], baseline_preds)
    baseline_mae = mean_absolute_error(test_df["team_goals"], baseline_preds)

    print(f"EM iterations:               {len(model.history_)}")
    print(f"Estimated lambda3 (train):    {model.lambda3_const_:.4f}")
    print(f"Estimated Dixon-Coles rho:    {model.rho_:.4f}")
    print(f"Bivariate modell - Poisson deviance: {deviance:.4f}, MAE: {mae:.4f}")
    print(f"Naive baseline    - Poisson deviance: {baseline_deviance:.4f}, MAE: {baseline_mae:.4f}")
    print("(baseline = team's avg goals against a neutral enemy)")

    argmax_accuracy, _ = evaluate_outcome_accuracy(model, test_df)
    print(f"\nOutcome accuracy (argmax on score-matrix with DC-korrection): {argmax_accuracy:.1%}")

    print("\nFeature-weighing:")
    print(model.feature_importance().to_string(index=False))

    return model


def save_model(model: BivariatePoissonModel, output_dir: str = "output/models"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/bivariate_goals_model.pkl")
    joblib.dump(FEATURE_COLUMNS, f"{output_dir}/feature_columns.pkl")
    print(f"\nModel saved: {output_dir}/bivariate_goals_model.pkl")


def main():
    long_df = load_dataset()
    print(f"Team-level dataset size: {long_df.shape} (original matches: {long_df['fixture_id'].nunique()})")

    train_df, test_df = time_based_split(long_df, test_size=0.2)
    print(f"Train: {len(train_df)} sor ({train_df['fixture_id'].nunique()} meccs), "
          f"Test: {len(test_df)} sor ({test_df['fixture_id'].nunique()} meccs)\n")

    model = train_and_evaluate(train_df, test_df)
    save_model(model)

    argmax_accuracy, argmax_detail_df = evaluate_outcome_accuracy(model, test_df)
    baseline_acc = majority_class_baseline(train_df, test_df)

    print(f"\nBivariate Poisson + Dixon-Coles H/D/V accuracy (argmax): {argmax_accuracy:.1%}")
    print(f"Naive baseline:                          {baseline_acc:.1%}")
    print("\nMistake matrix - argmax (sor=real, oszlop=estimated):")
    print(pd.crosstab(argmax_detail_df["actual"], argmax_detail_df["predicted"]))

    last_fixture_id = test_df.sort_values(["season", "round"]).iloc[-1]["fixture_id"]
    match_rows = test_df[test_df["fixture_id"] == last_fixture_id]
    home_row = match_rows[match_rows["is_home"] == 1].iloc[0]
    away_row = match_rows[match_rows["is_home"] == 0].iloc[0]

    home_features = home_row[NON_HOME_COLS].to_dict()
    away_features = away_row[NON_HOME_COLS].to_dict()

    probs = predict_match(model, home_features, away_features)
    outcome_probs = {"H": probs["home_win_prob"], "D": probs["draw_prob"], "A": probs["away_win_prob"]}
    argmax_prediction = max(outcome_probs, key=outcome_probs.get)
    print("\nPattern prediction (last test match):")
    print(f"  Expected goals: home {probs['expected_home_goals']:.2f} - "
          f"away {probs['expected_away_goals']:.2f} "
          f"(common component lambda3={probs['lambda3']:.3f}, Dixon-Coles rho={probs['rho']:.4f})")
    print(f"  Real score: {home_row['team_goals']:.0f} - {away_row['team_goals']:.0f}")
    print(f"  H/D/A likelihood: {probs['home_win_prob']:.1%} / "
          f"{probs['draw_prob']:.1%} / {probs['away_win_prob']:.1%}")
    print(f"  Argmax decision: {argmax_prediction}")


if __name__ == "__main__":
    main()