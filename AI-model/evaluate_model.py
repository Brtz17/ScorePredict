import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error

from train_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
    prepare_features,
    predict_match,
    score_matrix_probabilities,
    fit_dixon_coles_rho,
)


def actual_outcome(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if away_goals > home_goals:
        return "A"
    return "D"


def evaluate_outcome_accuracy(model: PoissonRegressor, test_df: pd.DataFrame, rho: float = 0.0) -> tuple[float, pd.DataFrame]:
    records = []
    for fixture_id, group in test_df.groupby("fixture_id"):
        home_row = group[group["is_home"] == 1].iloc[0]
        away_row = group[group["is_home"] == 0].iloc[0]

        home_features = home_row[[c for c in FEATURE_COLUMNS if c != "is_home"]].to_dict()
        away_features = away_row[[c for c in FEATURE_COLUMNS if c != "is_home"]].to_dict()

        pred = predict_match(model, home_features, away_features)
        probs = score_matrix_probabilities(pred["expected_home_goals"], pred["expected_away_goals"], rho=rho)

        outcome_probs = {"H": probs["home_win_prob"], "D": probs["draw_prob"], "A": probs["away_win_prob"]}
        predicted = max(outcome_probs, key=outcome_probs.get)
        actual = actual_outcome(home_row["team_goals"], away_row["team_goals"])

        records.append({
            "fixture_id": fixture_id,
            "predicted": predicted,
            "actual": actual,
            "correct": predicted == actual,
            "home_win_prob": probs["home_win_prob"],
            "draw_prob": probs["draw_prob"],
            "away_win_prob": probs["away_win_prob"],
        })

    result_df = pd.DataFrame(records)
    accuracy = result_df["correct"].mean( )
    return accuracy, result_df


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


def get_fold_boundaries(df: pd.DataFrame, n_splits: int):
    fixtures = (
        df[["fixture_id", "season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round", "fixture_id"])
        .reset_index(drop=True)
    )
    chunks = np.array_split(fixtures["fixture_id"].values, n_splits + 1)
    return chunks


def walk_forward_cv(df: pd.DataFrame, n_splits: int = 4) -> pd.DataFrame:
    chunks = get_fold_boundaries(df, n_splits)
    results = []

    train_fixtures = list(chunks[0])
    for i in range(1, len(chunks)):
        test_fixtures = list(chunks[i])

        train_df = df[df["fixture_id"].isin(train_fixtures)].reset_index(drop=True)
        test_df = df[df["fixture_id"].isin(test_fixtures)].reset_index(drop=True)

        X_train = prepare_features(train_df)
        y_train = train_df[TARGET_COLUMN]
        X_test = prepare_features(test_df)
        y_test = test_df[TARGET_COLUMN]

        model = PoissonRegressor(alpha=1.0, max_iter=1000)
        model.fit(X_train, y_train)

        preds = np.clip(model.predict(X_test), 1e-6, None)
        deviance = mean_poisson_deviance(y_test, preds)
        mae = mean_absolute_error(y_test, preds)


        rho = fit_dixon_coles_rho(model, train_df)
        accuracy, _ = evaluate_outcome_accuracy(model, test_df, rho=rho)
        baseline_acc = majority_class_baseline(train_df, test_df)

        results.append({
            "fold": i,
            "train_matches": train_df["fixture_id"].nunique(),
            "test_matches": test_df["fixture_id"].nunique(),
            "poisson_deviance": deviance,
            "mae": mae,
            "rho": rho,
            "outcome_accuracy": accuracy,
            "majority_baseline_accuracy": baseline_acc,
        })

        # A következő fold train halmaza bővül a most tesztelt blokkal
        train_fixtures = train_fixtures + test_fixtures

    return pd.DataFrame(results)


def get_match_predictions(model: PoissonRegressor, df: pd.DataFrame, rho: float = 0.0) -> pd.DataFrame:

    records = []
    for fixture_id, group in df.groupby("fixture_id"):
        home_row = group[group["is_home"] == 1].iloc[0]
        away_row = group[group["is_home"] == 0].iloc[0]

        home_features = home_row[[c for c in FEATURE_COLUMNS if c != "is_home"]].to_dict()
        away_features = away_row[[c for c in FEATURE_COLUMNS if c != "is_home"]].to_dict()

        pred = predict_match(model, home_features, away_features)
        probs = score_matrix_probabilities(pred["expected_home_goals"], pred["expected_away_goals"], rho=rho)

        records.append({
            "fixture_id": fixture_id,
            "home_win_prob": probs["home_win_prob"],
            "draw_prob": probs["draw_prob"],
            "away_win_prob": probs["away_win_prob"],
            "actual": actual_outcome(home_row["team_goals"], away_row["team_goals"]),
        })
    return pd.DataFrame(records)


def decide_outcome(row: pd.Series, draw_margin: float = 0.0) -> str:
    scores = {
        "H": row["home_win_prob"],
        "D": row["draw_prob"] + draw_margin,
        "A": row["away_win_prob"],
    }
    return max(scores, key=scores.get)


def tune_draw_margin(predictions_df: pd.DataFrame, margins=None) -> tuple[float, pd.DataFrame]:
    if margins is None:
        margins = np.arange(0.0, 0.35, 0.01)

    results = []
    for margin in margins:
        predicted = predictions_df.apply(decide_outcome, axis=1, draw_margin=margin)
        accuracy = (predicted == predictions_df["actual"]).mean()
        draw_share = (predicted == "D").mean()
        results.append({"margin": margin, "accuracy": accuracy, "predicted_draw_share": draw_share})

    results_df = pd.DataFrame(results)
    best_margin = results_df.loc[results_df["accuracy"].idxmax(), "margin"]
    return best_margin, results_df


def alpha_rho_sensitivity(train_df: pd.DataFrame, test_df: pd.DataFrame, alphas=None) -> pd.DataFrame:


    from train_model import FEATURE_COLUMNS as _FC  # csak olvashatosaghoz

    if alphas is None:
        alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    X_train = prepare_features(train_df)
    y_train = train_df[TARGET_COLUMN]

    results = []
    for alpha in alphas:
        model = PoissonRegressor(alpha=alpha, max_iter=1000)
        model.fit(X_train, y_train)

        rho = fit_dixon_coles_rho(model, train_df)
        accuracy, _ = evaluate_outcome_accuracy(model, test_df, rho=rho)
        accuracy_no_rho, _ = evaluate_outcome_accuracy(model, test_df, rho=0.0)

        results.append({
            "alpha": alpha,
            "rho": rho,
            "accuracy_no_correction": accuracy_no_rho,
            "accuracy_with_dixon_coles": accuracy,
        })

    return pd.DataFrame(results)


def main():
    long_df = load_dataset()

    print("=" * 60)
    print("1) OUTCOME ACCURACY (based on last 20% split)")
    print("=" * 60)

    from train_model import time_based_split, train_and_evaluate

    train_df, test_df = time_based_split(long_df, test_size=0.2)
    model = train_and_evaluate(train_df, test_df)

    accuracy, detail_df = evaluate_outcome_accuracy(model, test_df)
    baseline_acc = majority_class_baseline(train_df, test_df)

    print(f"\nModel H/D/A accuracy:          {accuracy:.1%}")
    print(f"Naive baseline (majority class): {baseline_acc:.1%}")
    print("\nConfusion matrix (row=actual, col=predicted):")
    print(pd.crosstab(detail_df["actual"], detail_df["predicted"]))

    print("\n" + "=" * 60)
    print("1b) DIXON-COLES CORRECTION (low scores / draws)")
    print("=" * 60)

    rho = fit_dixon_coles_rho(model, train_df)
    dc_accuracy, dc_detail_df = evaluate_outcome_accuracy(model, test_df, rho=rho)

    print(f"Fitted rho (on TRAIN data): {rho:.4f}")
    print(f"\nTest accuracy without correction: {accuracy:.1%}")
    print(f"Test accuracy with Dixon-Coles:    {dc_accuracy:.1%}")
    print("\nConfusion matrix with Dixon-Coles correction (row=actual, col=predicted):")
    print(pd.crosstab(dc_detail_df["actual"], dc_detail_df["predicted"]))

    print("\n" + "=" * 60)
    print("1c) DRAW-MARGIN TRICK (draw bonus, for comparison)")
    print("=" * 60)

    train_predictions = get_match_predictions(model, train_df)
    best_margin, grid_df = tune_draw_margin(train_predictions)
    print(f"Best margin on TRAIN data: {best_margin:.2f}")
    print("\nHow train accuracy changes with margin (sample):")
    print(grid_df.iloc[::5].to_string(index=False))  # every 5th row to keep it short

    test_predictions = get_match_predictions(model, test_df)
    plain_accuracy = (test_predictions.apply(decide_outcome, axis=1, draw_margin=0.0) == test_predictions["actual"]).mean()
    tuned_predicted = test_predictions.apply(decide_outcome, axis=1, draw_margin=best_margin)
    tuned_accuracy = (tuned_predicted == test_predictions["actual"]).mean()

    print(f"\nTest accuracy with plain argmax:       {plain_accuracy:.1%}")
    print(f"Test accuracy with draw_margin={best_margin:.2f}:  {tuned_accuracy:.1%}")
    print("\nConfusion matrix with draw_margin (row=actual, col=predicted):")
    print(pd.crosstab(test_predictions["actual"], tuned_predicted))

    print("\n" + "=" * 60)
    print("1d) EFFECT OF ALPHA ON RHO AND ACCURACY")
    print("=" * 60)
    print("(the goal is accuracy, not the size of rho - see docstring)")

    sensitivity_df = alpha_rho_sensitivity(train_df, test_df)
    print(sensitivity_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("2) WALK-FORWARD CROSS-VALIDATION")
    print("=" * 60)

    cv_results = walk_forward_cv(long_df, n_splits=4)
    print(cv_results.to_string(index=False))

    print("\nSummary (mean ± std across folds):")
    print(f"  Poisson deviance:  {cv_results['poisson_deviance'].mean():.4f} ± {cv_results['poisson_deviance'].std():.4f}")
    print(f"  MAE:               {cv_results['mae'].mean():.4f} ± {cv_results['mae'].std():.4f}")
    print(f"  Dixon-Coles rho:   {cv_results['rho'].mean():.4f} ± {cv_results['rho'].std():.4f}")
    print(f"  Outcome accuracy:  {cv_results['outcome_accuracy'].mean():.1%} ± {cv_results['outcome_accuracy'].std():.1%}")
    print(f"  Majority baseline: {cv_results['majority_baseline_accuracy'].mean():.1%} ± {cv_results['majority_baseline_accuracy'].std():.1%}")


if __name__ == "__main__":
    main()