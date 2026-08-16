import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error

from train_bivariate import (
    FEATURE_COLUMNS,
    NON_HOME_COLS,
    load_dataset,
    time_based_split,
    wide_from_long,
    BivariatePoissonModel,
    train_and_evaluate,
    evaluate_outcome_accuracy,
    majority_class_baseline,
    save_model,
)


def ranked_probability_score(home_prob: float, draw_prob: float, away_prob: float, actual: str) -> float:
    pred = np.array([home_prob, draw_prob, away_prob])
    actual_vec = np.array([
        1.0 if actual == "H" else 0.0,
        1.0 if actual == "D" else 0.0,
        1.0 if actual == "A" else 0.0,
    ])
    cum_pred = np.cumsum(pred)
    cum_actual = np.cumsum(actual_vec)
    return float(np.sum((cum_pred[:-1] - cum_actual[:-1]) ** 2) / (len(pred) - 1))


def mean_rps(detail_df: pd.DataFrame) -> float:
    return detail_df.apply(
        lambda row: ranked_probability_score(
            row["home_win_prob"], row["draw_prob"], row["away_win_prob"], row["actual"]
        ),
        axis=1,
    ).mean()


def get_fold_boundaries(df: pd.DataFrame, n_splits: int):
    fixtures = (
        df[["fixture_id", "season", "round"]]
        .drop_duplicates()
        .sort_values(["season", "round", "fixture_id"])
        .reset_index(drop=True)
    )
    chunks = np.array_split(fixtures["fixture_id"].values, n_splits + 1)
    return chunks


def walk_forward_cv(df: pd.DataFrame, n_splits: int = 4, alpha: float = 0.0001) -> pd.DataFrame:
    chunks = get_fold_boundaries(df, n_splits)
    results = []

    train_fixtures = list(chunks[0])
    for i in range(1, len(chunks)):
        test_fixtures = list(chunks[i])

        train_df = df[df["fixture_id"].isin(train_fixtures)].reset_index(drop=True)
        test_df = df[df["fixture_id"].isin(test_fixtures)].reset_index(drop=True)

        wide_train = wide_from_long(train_df, FEATURE_COLUMNS)
        wide_test = wide_from_long(test_df, FEATURE_COLUMNS)

        model = BivariatePoissonModel(feature_columns=NON_HOME_COLS, alpha=alpha, max_iter_em=50)
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

        argmax_accuracy, argmax_detail_df = evaluate_outcome_accuracy(model, test_df)
        baseline_acc = majority_class_baseline(train_df, test_df)
        rps = mean_rps(argmax_detail_df)

        results.append({
            "fold": i,
            "alpha": alpha,
            "train_matches": train_df["fixture_id"].nunique(),
            "test_matches": test_df["fixture_id"].nunique(),
            "poisson_deviance": deviance,
            "mae": mae,
            "lambda3": model.lambda3_const_,
            "rho": model.rho_,
            "outcome_accuracy": argmax_accuracy,
            "majority_baseline_accuracy": baseline_acc,
            "rps": rps,
        })

        train_fixtures = train_fixtures + test_fixtures

    return pd.DataFrame(results)


def tune_alpha(df: pd.DataFrame, alphas: list, n_splits: int = 4) -> pd.DataFrame:
    summaries = []
    for alpha in alphas:
        print(f"\n--- alpha = {alpha} ---")
        cv_results = walk_forward_cv(df, n_splits=n_splits, alpha=alpha)
        print(cv_results.to_string(index=False))
        summaries.append({
            "alpha": alpha,
            "poisson_deviance": cv_results["poisson_deviance"].mean(),
            "mae": cv_results["mae"].mean(),
            "outcome_accuracy": cv_results["outcome_accuracy"].mean(),
            "majority_baseline_accuracy": cv_results["majority_baseline_accuracy"].mean(),
            "rps": cv_results["rps"].mean(),
        })

    summary_df = pd.DataFrame(summaries).sort_values("rps").reset_index(drop=True)
    return summary_df


def main():
    long_df = load_dataset()

    print("=" * 60)
    print("1) OUTCOME ACCURACY (utolso 20%-os split alapjan, bivariate Poisson + Dixon-Coles)")
    print("=" * 60)

    train_df, test_df = time_based_split(long_df, test_size=0.2)
    model = train_and_evaluate(train_df, test_df)
    save_model(model)

    argmax_accuracy, argmax_detail_df = evaluate_outcome_accuracy(model, test_df)
    baseline_acc = majority_class_baseline(train_df, test_df)

    rps = mean_rps(argmax_detail_df)

    print(f"\nModell H/D/V pontossag (argmax):                       {argmax_accuracy:.1%}")
    print(f"Naive baseline (tobbsegi osztaly):                      {baseline_acc:.1%}")
    print(f"Becsult Dixon-Coles rho:                                {model.rho_:.4f}")
    print(f"RPS (argmax valoszinusegeken, minel kisebb annal jobb): {rps:.4f}")
    print("\nTevesztesi matrix - argmax (sor=valos, oszlop=becsult):")
    print(pd.crosstab(argmax_detail_df["actual"], argmax_detail_df["predicted"]))

    print("\n" + "=" * 60)
    print("2) WALK-FORWARD CROSS-VALIDATION (bivariate Poisson + Dixon-Coles, alpha=0.0001)")
    print("=" * 60)

    cv_results = walk_forward_cv(long_df, n_splits=4, alpha=0.0001)
    print(cv_results.to_string(index=False))

    print("\nOsszefoglalo (atlag ± szoras a foldok kozott):")
    print(f"  Poisson deviance:  {cv_results['poisson_deviance'].mean():.4f} ± {cv_results['poisson_deviance'].std():.4f}")
    print(f"  MAE:               {cv_results['mae'].mean():.4f} ± {cv_results['mae'].std():.4f}")
    print(f"  lambda3 (kozos komponens): {cv_results['lambda3'].mean():.4f} ± {cv_results['lambda3'].std():.4f}")
    print(f"  Dixon-Coles rho:           {cv_results['rho'].mean():.4f} ± {cv_results['rho'].std():.4f}")
    print(f"  Outcome accuracy (argmax):  {cv_results['outcome_accuracy'].mean():.1%} ± {cv_results['outcome_accuracy'].std():.1%}")
    print(f"  Majority baseline:          {cv_results['majority_baseline_accuracy'].mean():.1%} ± {cv_results['majority_baseline_accuracy'].std():.1%}")
    print(f"  RPS (argmax, minel kisebb annal jobb): {cv_results['rps'].mean():.4f} ± {cv_results['rps'].std():.4f}")

    print("\n" + "=" * 60)
    print("3) ALPHA HYPERPARAMETER-HANGOLAS (walk-forward CV, RPS szerint rendezve)")
    print("=" * 60)

    alphas_to_try = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    alpha_summary = tune_alpha(long_df, alphas_to_try, n_splits=4)

    print("\nOsszesito tabla (RPS szerint novekvo sorrendben, minel kisebb annal jobb):")
    print(alpha_summary.to_string(index=False))

    best_alpha = alpha_summary.iloc[0]["alpha"]
    print(f"\nLegjobb alpha (RPS alapjan): {best_alpha}")


if __name__ == "__main__":
    main()