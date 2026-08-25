# Methodology

## Data source (current)

[football-data.co.uk](https://www.football-data.co.uk) — free CSV downloads
of match results for major European leagues, updated regularly during the
season, no API key or signup required. Provides: date, teams, full-time and
half-time score, shots, shots on target, corners, cards, referee. It also
includes bookmaker odds columns, which we intentionally do NOT use as model
inputs (they'd leak sharp market information that trivially predicts
outcomes and isn't "our own" signal) — Bronze preserves them in case they're
useful later for calibration comparison, but Silver drops them.

Configured leagues/seasons live in `config/config.yaml` under
`active.leagues` / `active.seasons`. Extending coverage = adding entries
there and re-running `run_pipeline.py`.

## Data layers

- **Bronze**: raw CSVs, byte-for-byte as downloaded. Never edited in place.
- **Silver**: standardized columns/types/team-names, one row per match,
  de-duplicated by a derived `match_id` (date + home team + away team).
- **Gold**: model-ready feature tables.
  - `team_features/team_match_history.csv`: one row per (team, match) with
    that team's rolling form **entering** the match (values are computed
    with `.shift(1)` before the rolling window so a match's own result never
    leaks into its own features).
  - `match_features/match_features.csv`: one row per match with home-team
    and away-team form joined side by side, ready for modeling.

## Baseline model: Poisson regression

Two independent `PoissonRegressor` models (scikit-learn):

- `home_goals ~ home_recent_attack_form + away_recent_defense_form`
- `away_goals ~ away_recent_attack_form + home_recent_defense_form`

From the two predicted rates (λ_home, λ_away) we build a full scoreline
probability grid (`P(home=i, away=j)` for i, j up to `max_goals_simulated`)
assuming independence, then derive:

- Most likely exact scoreline (argmax of the grid)
- Win/draw/loss probabilities (sum of the lower-triangle / diagonal /
  upper-triangle of the grid)
- "Confidence" = probability mass on the single most likely scoreline (a
  simple, conservative confidence metric — a flat grid means low confidence
  even if one cell is technically the argmax)

**Why start here:** it's simple, fast, well-understood, and every
coefficient is directly interpretable (attack/defense strength). It sets the
bar that more complex models (logistic regression, random forest, XGBoost)
must clear on held-out data before being adopted — per the project's
explicit "don't assume more complex = better" principle.

## Known limitations (current stage)

- No player-level information yet — a team missing key attackers/defenders
  isn't reflected in predictions.
- No head-to-head or tactical-matchup features yet.
- Home advantage is only implicit (via separate home/away goal columns), not
  an explicit modeled term yet.
- Rolling form window defaults to last 5 matches (`config.yaml ->
  features.form_windows`) — this is a reasonable starting point, not a
  validated optimum. Should be swept as part of model comparison.
- Independence assumption between home and away goals is a simplification;
  real matches have some correlation (e.g. game state effects). A
  bivariate-Poisson or Dixon-Coles adjustment is a natural next step.
- Evaluation currently only covers match outcome/score, not player
  predictions (no player layer yet).

## Model comparison protocol (for future models)

1. Use a fixed, chronological train/test split (never randomly shuffled —
   this is time-series data and shuffling leaks the future into training).
2. Compare candidates on the same metrics tracked in
   `src/evaluation/evaluate_predictions.py`: winner accuracy, exact-score
   accuracy, goal MAE, Brier score.
3. A challenger model replaces the production baseline only if it wins on
   Brier score (calibration) AND doesn't meaningfully regress on winner
   accuracy, evaluated on a held-out set it never trained on.
4. Log the comparison itself (which models, which features, which metrics,
   which split) in `docs/` so model selection is reproducible, not just the
   final choice.
