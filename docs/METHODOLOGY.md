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

## Data source: player stats

[Wikipedia club-season articles](https://en.wikipedia.org) (e.g. "2025-26
Arsenal F.C. season"), fetched via the MediaWiki API (`action=parse`) rather
than scraping rendered pages -- a documented, stable contract. Two free
alternatives were evaluated and rejected: FBref sits behind a Cloudflare JS
challenge (unreachable via a plain request), and Understat no longer inlines
its player data in the page source (loads it dynamically after page load).
Wikipedia was the only source that's both free and reliably scrapable with a
plain HTTP client.

Provides per-player, per-season Premier League appearances (starts + sub
appearances) and goals -- real performance data, not video-game ratings.
Article formatting isn't fully standardized across ~100 club-season articles
written by different editors: separate "Appearances" + "Goals"/"Goalscorers"
tables; one combined "Appearances and goals" table with a two- or
three-level header (the third level is a positional-group row like
"Goalkeepers"); Apps/Starts-only tables whose goals live in a separate
goalscorers table; and icon-only column headers. All are handled (see
`src/cleaning/clean_player_stats.py`); a file matching none is skipped with
a logged message rather than crashing the run. Footer rows ("Total",
"Own goals") are dropped so they can't be counted as players.

**Coverage and validation (2026-10-05):** 79 of the 80 PL team-seasons in
2022-23..2025-26 parse (Wolves 2022-23 is the one miss). Each parsed
team-season's summed player league goals was checked against the team's
actual goals from match results: the ratio is 0.93-1.00 for almost all of
them, the shortfall being opponent own goals, which no player is credited
with. Two outliers are gaps in the Wikipedia source itself, not the parser,
and are left as-is (Bronze is never hand-edited): Bournemouth 2024-25's
table omits Antoine Semenyo (11 PL goals; ratio 0.79), and Luton 2023-24's
Apps column under-reports appearances (11.6 per match vs. a normal ~14-16).
Ipswich 2025-26 (a Championship season) is correctly skipped.

Player stats feed the squad-quality **challenger** feature (see below) and
the app's squad view. The production baseline model doesn't use them.

Wikipedia's anonymous API has a modest rate limit; ingestion sleeps between
requests and backs off on HTTP 429, and skips files that already exist so
re-runs only fetch what's missing.

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

## Alternative model: R Dixon-Coles-style GLM

`src/models/dixon_coles_model.R` (base R only, no CRAN packages) fits a
genuinely different model rather than reimplementing the Python one: a
Poisson GLM with **fixed team-specific attack/defense parameters**, the
standard Dixon & Coles (1997) parameterization --

```
goals ~ home_advantage + attack(team) + defense(opponent)
```

fit on a long reshape of the full match history (two rows per match: each
side's scoring performance contributes to its own attack coefficient and its
opponent's defense coefficient). Where the Python model asks "how has this
team played in its last 5 matches," the R model asks "how strong is this
team across the whole dataset" -- a different signal, so disagreement
between the two is expected, not a bug.

Run it with `Rscript src/models/dixon_coles_model.R` (needs R; on macOS,
`brew install r`). It writes:

- `models/dixon_coles.rds` -- the fitted model.
- `data/gold/prediction_features/dixon_coles_historical.csv` -- a prediction
  for every historical match, shaped so `evaluate_predictions.compute_metrics()`
  can score it directly, the same as the Python model.
- `data/gold/prediction_features/dixon_coles_matchups.csv` -- a prediction
  for every pairing of current-season teams, including 80%/95% Poisson
  confidence intervals, which the app reads to show both models side by side.

This is not currently wired into `run_pipeline.py` -- it's an optional,
separately-run comparison, the same way Wikipedia player-stats ingestion is.
Not yet implemented: the low-score correlation (`tau`/`rho`) adjustment from
the original Dixon-Coles paper, which corrects for the independence
assumption below on 0-0/1-0/0-1/1-1 scorelines specifically -- the team
attack/defense parameterization is the part that's done.

## Squad-quality feature (2026-10-10 experiment)

`squad_prior_goals_per_match` for team T in season s = (sum of the
PREVIOUS season's Premier League goals, at any club, of every player on T's
season-s roster) / 38. Built by `build_squad_quality()` in
`src/features/build_match_features.py`; tested as the rolling-form baseline
plus this feature (`build_squad_feature_lists`).

- **Why the previous season:** Wikipedia gives end-of-season totals only,
  so any same-season player stat contains the goals of the match being
  predicted and every later one. Last season's totals are known before the
  season starts, and summing over the current roster carries transfers in
  both directions.
- **Why a sum, not a per-appearance rate:** a rate lets a 1-goal,
  1-appearance player dominate; in a sum his weight is his (small) goal
  total, so no minimum-appearances floor is needed.
- **Missing data stays missing:** NaN (not 0) when the team-season has no
  roster, when the previous season isn't in the data at all (2022-23), or
  when the team was in the PL last season but its own article didn't parse.
  Players with no PL goals last season (signings from abroad, most of a
  promoted squad) contribute 0, so promoted squads score near 0 by design.
- **Known look-ahead, roster composition only:** rosters come from
  end-of-season articles, so a January signing counts toward the August-
  December matches too, and a January departure toward February-May.
  Results never leak.
- **Control arm:** `prev_season_goals_per_match` is the team's own goals per
  match last season from match results (0 if promoted). It has no player
  data and no look-ahead. Squad quality must beat this, not only the
  baseline, before a gain can be credited to player information.

Because squad quality doesn't exist for the earliest season, every arm of
this experiment is trained on the same reduced rows (2023-24 + 2024-25).
Results are in `docs/PROGRESS.md` (2026-10-05).

## Known limitations (current stage)

- Player information enters only through the squad-quality challenger
  (previous-season goals). There is no availability, injury, or lineup data,
  so a team missing key players on the day isn't reflected.
- The Python model's L2 penalty (`config.yaml -> model.poisson_alpha`,
  1.0 in every recorded experiment) acts on **unstandardized** features. It
  roughly halves the coefficients and penalizes smooth, low-variance
  features (whole-history averages) more than noisy ones (5-match form),
  which biases feature comparisons against long-horizon features. Setting
  it to 0.01 changes some conclusions (see PROGRESS.md 2026-10-05). It
  should be chosen on a validation split inside the training seasons, with
  features standardized first.
- The `season_avg_*` and `*_by_venue` columns are expanding means over each
  team's whole history in the data. They are NOT reset each season, despite
  the names.
- No head-to-head or tactical-matchup features yet.
- Home advantage is only implicit (via separate home/away goal columns), not
  an explicit modeled term yet.
- Rolling form window is 5 matches (`config.yaml -> features.form_windows`,
  first entry). 3 and 10 were tested against it on the held-out season
  (2026-10-05) with no distinguishable difference at either regularization
  setting.
- Independence assumption between home and away goals is a simplification;
  real matches have some correlation (e.g. game state effects). The R model
  adds team-specific attack/defense parameters but not yet the Dixon-Coles
  low-score correlation adjustment itself -- still a natural next step.
- Evaluation currently only covers match outcome/score, not player
  predictions (no player layer yet).

## Model comparison protocol (for future models)

1. Use a fixed, chronological train/test split (never randomly shuffled —
   this is time-series data and shuffling leaks the future into training).
2. Compare candidates on the same metrics tracked in
   `src/evaluation/evaluate_predictions.py`: winner accuracy, exact-score
   accuracy, goal MAE, Brier score, log loss -- on the identical held-out
   matches, with a paired bootstrap (`backtest.paired_bootstrap`) giving a
   95% interval for every difference. If a challenger can only train on a
   subset of rows, refit the baseline on exactly that subset.
   Hyperparameters (e.g. `poisson_alpha`) must be chosen without looking at
   the holdout season.
3. A challenger model replaces the production baseline only if it wins on
   Brier score (calibration) AND doesn't meaningfully regress on winner
   accuracy, evaluated on a held-out set it never trained on.
4. Log the comparison itself (which models, which features, which metrics,
   which split) in `docs/` so model selection is reproducible, not just the
   final choice.
