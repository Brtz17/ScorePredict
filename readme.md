# ScorePredict

A Python-based football match prediction system using a bivariate Poisson model to forecast exact scorelines.

## Overview

ScorePredict's goal isn't just predicting the match outcome (home win / draw / away win) — exact scoreline prediction is a core part of the product. It relies on a bivariate Poisson model with a Dixon-Coles rho correction.

## Main Components

- **`build_dataset.py`** — Builds the dataset. Computes causal, cumulative running averages for team statistics to eliminate future data leakage. It also computes a causal "motivation" feature:
  - For league phases: table-based points gap to the teams above/below (`build_prematch_standings_lookup`)
  - For knockout rounds: pair-aggregate goal-difference based metric
  - Both are normalized per-type via percentile ranking
  - Exports `team_form_state.json` for reuse in future live prediction
- **`train_bivariate.py`** — Trains the bivariate Poisson model; fully self-contained with no external module dependencies.
- **`evaluate_bivariate.py`** — Evaluates the model; also self-contained.

## Modeling Details

- **Form metric**: opponent-strength-adjusted EWMA (Exponentially Weighted Moving Average, α = 0.35), replacing the earlier naive last-N-match averaging
  - EWMA state is keyed by team_id and carries across season boundaries
  - Cold-start defaults: 1.0 for form ratios, 365 days for rest
- **Dixon-Coles rho correction**: implemented, fits near-zero across folds
- **Evaluation metrics**: RPS (Ranked Probability Score), plus a custom XGBoost eval metric

## Results

Walk-forward cross-validation showed **54–56% outcome accuracy** vs. a ~46% baseline.

## Known Limitations / Next Steps

- Draw underestimation is still an issue — region-level score matrix rescaling is the next planned approach to address it
- History: the pipeline was originally built as `build_features.py`, `train_model.py`, and `evaluate_model.py` in Google Colab; it has since been refactored into standalone, modular scripts

## Data Layer

- **Historical data**: API-Football (RapidAPI) — league IDs: PL=39, La Liga=140, Serie A=135, Bundesliga=78, Ligue 1=61; UEFA CL=2, EL=3, ECL=4
- **Upcoming fixtures**: football-data.org (free tier) — covers ~9-12 major competitions vs. the full ~30-league historical list, a deliberate tradeoff for staying free
- OpenLigaDB was ruled out since it only covers German leagues
- A Node.js data-fetching script with rate limiting and checkpoint/resume logic
- Team mapping between API-Football and football-data.org is mostly complete; one team couldn't be matched because it isn't in API-Football's teams.csv and is hard to obtain due to API limitations

## Planned Architecture (Appwrite backend)

1. An Appwrite table stores which league/season data to fetch
2. A daily job reruns the existing fetch + dataset-build code to refresh `dataset.csv`
3. Fetches each league's upcoming fixtures via a (not-yet-written) API integration
4. Runs the AI model on them
5. Exports predictions to an Appwrite table

**Build approach**: everything is built and tested locally first; the move to "automatic" (Appwrite-scheduled) operation happens only after tests pass. The Appwrite Function itself will also be deployed manually first ("Execute now"), with the daily cron schedule enabled only after a successful manual run.