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
  - `prediction_features/`: held-out and walk-forward predictions from both
    models (the recorded alpha=1.0 experiments and R).
  - `derby_analysis/`, `final_evaluation/`: every interval, metric and
    per-match score behind the two research questions (see the sections
    below).

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

**Penalty and scaling.** `PoissonRegressor` applies an L2 penalty `alpha`.
Every recorded experiment (`backtest.py`, `derby_analysis.py`, and the app's
model) uses `config.yaml -> model.poisson_alpha: 1.0` on unstandardized
features, kept so those results stay reproducible. The final evaluation
instead standardizes features (`PoissonMatchModel(standardize=True)`, a
scikit-learn pipeline whose scaler sees the training rows only) and chooses
alpha per feature set inside the training seasons. It chose alpha=0 (no
penalty) for every set but one. See "Final evaluation" below.

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
- `data/gold/prediction_features/dixon_coles_holdout_backtest.csv` -- the
  held-out 2025-26 predictions from a fit on earlier seasons only.
- `data/gold/prediction_features/dixon_coles_walkforward.csv` -- walk-forward
  predictions for 2023-24..2025-26, each season from a fit on the seasons
  before it (used by the derby analysis).

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

## Derby / rivalry analysis (2026-10-17 experiment)

`src/evaluation/derby_analysis.py` asks whether derbies are systematically
harder to forecast. Results are in `docs/PROGRESS.md` (item 16).

**Definitions** (`config.yaml -> derbies`, fixed before any result was
computed, so they couldn't be tuned to the answer). EPL pairings only,
unordered, with team names checked against the match data so a typo can't
silently drop a derby:

- **local (primary):** both clubs in the same metropolitan area (Greater
  London, Greater Manchester, Merseyside, West Midlands, Tyne and Wear). A
  geographic rule, not a judgment call, and the largest sample: 25
  pairings, 194 of the 1,520 matches.
- **classic:** the Manchester, North London and Merseyside derbies (24
  matches).
- **rivalry:** 9 named rivalries, including between-city ones (North-West,
  M23, East Midlands, Roses, plus Tyne-Wear, West Midlands and the classic
  three; 58 matches). This is the one definition that needs judgment, so
  it's a sensitivity check only.

**Sample: walk-forward, not one holdout season.** A pairing meets at most
twice a season, so the 2025-26 holdout alone holds ~46 local derbies and 6
classic ones. Each of 2023-24, 2024-25 and 2025-26
(`config.yaml -> evaluation.walk_forward_test_seasons`) is predicted by a
model trained only on the seasons before it, and the three are pooled. Every
prediction is still out-of-sample. The 2025-26 fold is the same fit as the
existing holdout backtest and reproduces its numbers exactly. Two models:
the Python rolling-form baseline (`walk_forward_poisson`) and R
Dixon-Coles (section 4 of the R script, `dixon_coles_walkforward.csv`).

**"Harder" split into two questions.** A model's loss on derbies can be
higher because derbies are close matchups (which even a perfect model finds
uncertain) or because the model misjudges them. So two quantities are
compared:

- *raw loss*: per-match Brier / log loss;
- *excess loss*: raw loss minus the loss the model expects from its own
  probabilities p. Expected Brier = sum_k p_k * Brier(outcome k) =
  1 - sum p_k^2; expected log loss = entropy -sum p_k ln p_k. Excess
  averages zero for a calibrated model however close the matchup is.

raw difference = expected difference + excess difference, exactly. Also
compared: winner accuracy and calibration-in-the-large by outcome (mean of
outcome indicator minus predicted probability for H/D/A, for the
favorite's win, and actual minus expected goal difference).

**Inference.** Derby minus non-derby difference in means, with a 95%
bootstrap interval (10,000 resamples, each group resampled independently,
since the groups are disjoint, unlike the paired feature comparisons), plus
the minimum detectable difference at 80% power. Pre-specified primary test:
raw and excess Brier, local definition, pooled sample. A claim is made only
if the interval excludes zero for both models. Everything else is
secondary: about 70 intervals are reported, so ~3-4 excluding zero by
chance is expected.

**Robustness checks:** the other two definitions; each test season alone;
a pair-cluster bootstrap (resample whole pairings, in case meetings of the
same two teams are correlated); both models on their common matches;
Python at alpha=0.01; leave-one-pair-out.

**Home advantage** uses actual results for all four seasons, with no model,
so nothing can leak. Every pairing plays both legs each season, so team
strength cancels out of each group's average home-minus-away goal
difference, which makes the derby vs. non-derby comparison fair without
any adjustment.

**Limitations specific to this analysis:**
- Power. ~145 out-of-sample local derbies give a minimum detectable Brier
  difference of 0.05-0.08, larger than the models' whole skill over a
  uniform guess. A null result here can't rule out a real but moderate
  derby effect.
- 86% of local derbies are London derbies, many of them low-intensity. The
  high-intensity definitions (classic, rivalry) have 18-42 matches.
- At alpha=1.0 the Python baseline's probabilities barely vary between
  matches, so R is the more informative of the two tests.
- R skips matches involving promoted teams (no training history), so its
  sample is smaller and excludes e.g. 2025-26 Tyne-Wear derbies.
- EPL only. The plan's non-EPL examples (Milan-Inter, Real
  Madrid-Atletico) aren't in the data.
- Neither model has a derby term. Given the null result, there's no
  evidence-based reason to add one.

## Final evaluation (2026-10-24 freeze)

`src/evaluation/final_evaluation.py`. Outputs: `data/gold/final_evaluation/`.
Results: README.md and `docs/PROGRESS.md` item 18.

**Why it exists.** Every recorded Python experiment used one L2 penalty,
alpha=1.0, on unstandardized features. That penalty shrank the forecasts
toward the base rate. It also penalized smooth whole-history features harder
than noisy short-term form, and a diagnostic run at alpha=0.01 changed
which features looked useful (PROGRESS.md item 15). Adopting 0.01 because
it scored well on the holdout would have tuned to the holdout, so the final
evaluation chooses the penalty without it.

**Procedure** (fixed in `config.yaml -> final_evaluation` before the run):

1. Standardize every feature using the training rows' mean and sd only.
2. For each feature set separately, choose alpha from {0, 0.001, 0.003,
   0.01, 0.03, 0.1, 0.3, 1, 3} by walk-forward validation inside the
   training seasons (`select_alpha`):
   - 2023-24 is predicted by a fit on 2022-23, and 2024-25 by a fit on
     2022-24.
   - The predictions are pooled; the lowest Brier wins, ties going to the
     larger alpha.
   - Every season from the test season onward is dropped before this step.
   - The squad-quality experiment has a single validation fold (2024-25 from
     2023-24), because its features start in 2023-24.
3. Refit each set on all training seasons at its own alpha and score the
   2025-26 holdout once, on the same matches as `backtest.py` (374 for the
   form sets, 377 for the squad sets).
4. Run paired bootstraps for:
   - each challenger vs. its own baseline;
   - each model vs. the base rate;
   - R vs. each Python model, on the matches both scored;
   - each set vs. its own recorded alpha=1.0 run.

**Decision rule** (fixed beforehand). A factor improves out-of-sample
predictions only if its holdout Brier 95% interval against its own baseline
excludes zero in the improving direction. Bonferroni intervals for the 8
challengers (99.375%) are reported alongside. Whether a challenger then
becomes the frozen Python model follows protocol rule 3 below.

**Replication: nested walk-forward.** For each season in
`final_evaluation.nested_walk_forward_test_seasons` (2024-25, 2025-26):
- alpha is re-chosen only from validation seasons strictly before it (for
  2024-25, from 2023-24 alone);
- the model is fit on all earlier seasons;
- the season is predicted.

The script checks on every run that the 2025-26 fold is identical to step 3.

2024-25 is the replication season: no earlier experiment had scored any
feature set on it. It checks direction; it isn't a second chance to pass
the rule. What isn't in the replication:
- 2023-24 can't be a test season: there's no season before 2022-23 to
  validate on.
- Squad quality: it has no validation season before 2025-26.

**What it found** (numbers in README.md):
- **The penalty.**
  - alpha=0 won validation for 8 of 9 sets; the 6-feature combined-window
    set chose 0.03.
  - Validation Brier rises steadily with alpha for every unpenalized set.
    With 2-6 features and about a thousand training rows, there's little
    variance to trade for bias.
  - An unpenalized GLM is invariant to rescaling its features, so
    standardization only matters for the combined-window set.
- **Venue-split team averages pass the rule** and replicate on 2024-25. The
  holdout interval alone doesn't survive Bonferroni. Venue-split
  (unpenalized) is the frozen Python model.
- **10-match form** beats 5 in 2024-25 only (suggestive). **3-match form** is
  worse in 2024-25. **Combined windows, SOS, the control and squad quality**
  are not established or show no effect.
- **The recorded "wins" don't survive.** Squad quality and combined windows
  looked significant at alpha=1.0 but not with the chosen penalty. A heavy
  penalty shrinks the baseline's forecasts, and any extra strength feature
  lets some signal back through.

**Fitted effects of the frozen model** (standardized; the multiplier on
expected goals per +1 sd of each feature):

| Goals model | Feature | Multiplier |
|---|---|---:|
| Home goals | Home side's home scoring average | ×1.24 |
| Home goals | Away side's away conceding average | ×1.14 |
| Away goals | Away side's away scoring average | ×1.18 |
| Away goals | Home side's home conceding average | ×1.18 |

At average feature values the model expects 1.59 home goals and 1.35 away
goals. For comparison, the 5-match-form baseline's multipliers are
×1.17 / ×1.10 / ×1.14 / ×1.13.

**What it can't fix:**
- The holdout was reported on throughout development, so it influenced which
  features were tried. That's why the 2024-25 replication matters.
- The derby analysis still uses the alpha=1.0 baseline. Its alpha=0.01
  robustness run is effectively unpenalized and gives the same answer.
- The app still serves the alpha=1.0 5-match-form model.

## Known limitations (research version)

- Player information enters only through the squad-quality challenger
  (previous-season goals). There is no availability, injury, or lineup data,
  so a team missing key players on the day isn't reflected.
- Sample size. One season is ~375 matches, enough to resolve a Brier
  difference of ~0.015 but not the ~0.005 effects most features have. The
  data covers four seasons, so at most two test seasons with a validation
  season before them.
- The 2025-26 holdout was reported on by every development experiment, so it
  isn't a pristine test of which features to try. The final evaluation
  leans on the 2024-25 replication for that reason, and 8 challengers share
  the holdout (Bonferroni reported).
- Regularization: resolved at the freeze (see "Final evaluation"). The
  recorded experiments, the derby analysis and the app's model still use the
  old alpha=1.0 on unstandardized features. That setting shrinks forecasts
  toward the base rate and biases comparisons against smooth, long-horizon
  features, so read their numbers with that in mind.
- The `season_avg_*` and `*_by_venue` columns are expanding means over each
  team's whole history in the data. They are NOT reset each season, despite
  the names.
- No head-to-head or tactical-matchup features yet.
- Home advantage is only implicit (via separate home/away goal columns), not
  an explicit modeled term yet.
- Rolling form window is 5 matches (`config.yaml -> features.form_windows`,
  first entry). With the penalty chosen properly, 10 beats 5 and 3 is worse
  than 5 in 2024-25, with neither difference established on the holdout;
  combining 3/5/10 does no better than 10 alone. The whole-history
  venue-split averages beat all of them.
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
   95% interval for every difference. Report the no-information base rate
   (`backtest.base_rate_forecast`: training-season H/D/A frequencies for
   every match, whose accuracy equals always picking home) alongside, so
   every number has a "knows nothing about the teams" reference. If a challenger can only train on a
   subset of rows, refit the baseline on exactly that subset.
   Hyperparameters (e.g. `poisson_alpha`) must be chosen without looking at
   the holdout season -- `final_evaluation.select_alpha` does this by
   walk-forward validation inside the training seasons. When several
   challengers share one holdout, report a multiple-comparison-corrected
   interval too (`paired_bootstrap(level=...)`). For a comparison between two disjoint groups of
   matches (e.g. derbies vs. other matches) rather than two models on the
   same matches, use `derby_analysis.compare_groups`, which resamples each
   group independently.
3. A challenger model replaces the production baseline only if it wins on
   Brier score (calibration) AND doesn't meaningfully regress on winner
   accuracy, evaluated on a held-out set it never trained on.
4. Log the comparison itself (which models, which features, which metrics,
   which split) in `docs/` so model selection is reproducible, not just the
   final choice.
