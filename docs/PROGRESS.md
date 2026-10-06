# Progress Log

A handoff doc for picking this project back up in a new session (Claude or
otherwise) — what's been done, what state it's in, and what's next. For
architecture/roadmap context, see [ABOUT.md](../ABOUT.md); for how the data
and models actually work, see [METHODOLOGY.md](METHODOLOGY.md).

## What's been done

**1. Closed the loop on the Poisson baseline** (stage 1 of the roadmap)
- Fixed `src/models/poisson_model.py --predict`, which was a stub that just
  printed instructions instead of predicting. Added `predict_from_rows()` /
  `predict_matchup()` / `latest_team_row()`, shared by the CLI and
  `generate_predictions.py`.
- Ran the prediction → evaluation loop for real for the first time
  (`fixtures.csv` → `prediction_log.csv` → `evaluate_predictions.py`).
- Backfilled unit tests for previously-untested modules (model, predictions,
  evaluation, ingestion). Suite went from 6 tests to 21.

**2. Built the Streamlit app** (`app.py`)
- Iterated through a few UI designs based on feedback: season/team dropdowns
  → a card grid of fixtures → one unified searchable dropdown covering past
  results, real scheduled fixtures, and hypothetical matchups.
- Home/away color scheme, primary/secondary stat cards on each team, rounded
  predicted scores (was showing the raw argmax score, which skewed toward
  1-1 for most matchups).
- "Dev View" — for a past match, shows what the model would have predicted
  beforehand vs. what actually happened.
- An in-sample backtested accuracy summary.

**3. Player stats** (`src/ingestion/wikipedia_player_stats.py`,
`src/cleaning/clean_player_stats.py`)
- Evaluated and rejected FBref (Cloudflare-blocked, no plain-request access)
  and Understat (no longer inlines data in the page source) as scrape
  targets. Settled on Wikipedia club-season articles via the MediaWiki API.
- Real-world Wikipedia articles turned out to use **three different table
  layouts** across clubs/seasons (separate Appearances/Goals tables, one
  combined table, and an icon-header-only table) — all three are handled.
- Fetched all 4 seasons × ~20-25 teams (81 team-season articles); 57 of 81
  parse cleanly (1,865+ player-season rows). The other 24 use a still-
  different (4th, unhandled) layout, concentrated in the 2022-23/2023-24
  seasons.
- Added a per-team "Squad" view to the app.

**4. Real upcoming fixtures** (`src/cleaning/clean_fixtures.py`)
- Found that football-data.co.uk (already the match-results source) also
  publishes a live `fixtures.csv` of near-term scheduled matches.
- Cleaning cross-checks team names against known league teams, which caught
  a real data-quality bug in the source (a Championship match mislabeled
  with the Premier League's division code).
- Wired into the app's search dropdown alongside past results and
  hypothetical (unscheduled) pairings, each labeled distinctly.

**5. R alternative model** (`src/models/dixon_coles_model.R`)
- Installed R via Homebrew (wasn't present before).
- Built a genuinely different model from the Python baseline: a Poisson GLM
  with fixed team-specific attack/defense parameters (the classic Dixon &
  Coles 1997 parameterization) fit on the full match history, vs. Python's
  rolling-5-match-form regression.
- Outputs backtest predictions (for comparison via the existing
  `compute_metrics()`) and predictions for every current-team pairing,
  including 80%/95% Poisson confidence intervals.
- In-sample result: R's winner accuracy (55%) beats Python's (46%) —
  expected, since R sees each team's full multi-season record, not just
  recent form. Not a real "R wins" claim; see caveats in METHODOLOGY.md.

**6. App: model comparison + confidence-interval charts, and simplification**
- Added matplotlib charts showing each team's goal-count probability
  distribution with 80%/95% CI shading, plus the R model's expected value
  marked for comparison.
- Decluttered the page: the accuracy summary and match-detail views (stats
  table, Dev View, R comparison, CI charts) all moved from always-visible
  into collapsed expanders. Trimmed team-card secondary stats from 3 to 2.

**7. Repo hygiene**
- Added `.gitignore` (previously `.DS_Store` and `__pycache__/*.pyc` were
  tracked).
- Added `.gitattributes` (`*.html linguist-detectable=false`) so the ~80
  raw Wikipedia HTML files in `data/bronze/players/` don't dominate GitHub's
  language-detection stats.
- Audited tracked files and full git history for secrets/credentials/PII —
  none found.

**8. Real chronological train/test backtest** (2026-09-20)
- Added `evaluation.holdout_season: 2526` to `config/config.yaml` — the most
  recent complete season, held out entirely from training for both models.
- `src/evaluation/backtest.py`: refits the Python Poisson model on the
  non-holdout seasons only and scores it on the holdout season, reusing
  `evaluate_predictions.compute_metrics()` so numbers are directly
  comparable to live prediction-log evaluation. Writes
  `data/gold/prediction_features/poisson_holdout_backtest.csv`.
- Added an equivalent held-out section to `src/models/dixon_coles_model.R`:
  refits the GLM on train seasons only, predicts the holdout season, and
  prints/saves the same metrics
  (`dixon_coles_holdout_backtest.csv`). Matches involving a team with zero
  training-season appearances (newly promoted for the holdout season) are
  skipped rather than guessed at — a real limitation of the whole-history
  parameterization, not a bug.
- Real (out-of-sample) results: Python 43.8% winner accuracy / 0.646 Brier;
  R 47.7% winner accuracy / 0.618 Brier (on 342 matches, 38 skipped for
  unseen teams). R still edges out Python, but the gap is much smaller than
  the old in-sample comparison (55% vs 46%) suggested — that comparison was
  optimistic for R specifically, since it trained on the full history
  including the seasons it was being scored on.
- Added `tests/test_backtest.py` (2 tests: holdout-only scoring, empty-split
  error) — 41 tests pass as of this session.

**9. Team-level venue-split features (2026-09-26 research milestone)**
- Research question: does better team-level information improve
  predictions over the rolling-form baseline? Also the first milestone of
  the reframed research plan (see `CLAUDE.md` — this project is now scoped
  around two specific research questions with weekly deadlines through
  2026-10-24, not an open-ended feature backlog).
- The blended season-to-date averages (`season_avg_goals_for/against`)
  already existed. What was missing was the **home/away split**: a team's
  average goals scored/conceded computed ONLY from its own past matches at
  the same venue. Added `add_venue_split_averages()` to
  `src/features/build_match_features.py` — new columns
  `avg_goals_for_by_venue` / `avg_goals_against_by_venue`, expanding-mean
  shifted by 1 *within each venue-specific subsequence* (a team's 3rd home
  game looks back at its first 2 home games only, not at away games played
  in between).
- Verified no leakage by hand (Arsenal's 4th home match's feature value
  exactly equals the mean of its prior 3 home goals) and with two new unit
  tests in `tests/test_features.py`.
- Added `build_team_level_feature_lists()` to `src/models/poisson_model.py`
  as a named alternative to the existing `build_feature_lists()` (rolling
  form) — both still exist side by side, nothing deleted.
- Extended `src/evaluation/backtest.py` to run named feature sets
  head-to-head and print a comparison table. Also fixed a fairness issue:
  the two feature sets drop slightly different rows to NaN, so evaluation
  is now restricted to the intersection of valid holdout-season matches
  across all feature sets (374 matches for both, not 377 vs. 374).
- **Result: suggestive, not established.** Team-level venue-split scored
  higher on winner accuracy (45.7% vs. 44.1%) and lower on Brier (0.640 vs.
  0.645) on the identical 374-match held-out sample; exact-score accuracy
  tied, goal MAE mixed. (First written up as "yes, modestly" — corrected
  after testing the gap for noise.)
- Added `paired_bootstrap()` to `src/evaluation/backtest.py` (paired
  resampling over matches, 10,000 draws) and print it after every
  comparison. Brier difference −0.0051, 95% CI [−0.0104, +0.0003];
  accuracy difference +1.6 pts (6 matches: 13 gained, 7 lost), 95% CI
  [−0.8, +4.0] pts. Both include zero, so one season can't separate this
  from noise. Keep both feature sets as reference points; a multi-season
  walk-forward check is the natural way to settle it. 3 tests added
  (46 pass).
- R wasn't changed this round and its numbers are unchanged (reconfirmed by
  rerunning) — its fixed-effects design already encodes team-level
  attack/defense strength by construction, so "rolling-form vs. team-level"
  isn't a natural experiment on the R side the way it is for Python.

**10. Log loss added to evaluation (2026-09-28, step 1 of the 2026-10-03 milestone)**
- `per_match_log_loss()` in `src/evaluation/evaluate_predictions.py`:
  −ln(probability assigned to the actual H/D/A outcome), clipped at 1e-15
  so a stored 0.0 probability gives a large finite penalty rather than inf.
  `compute_metrics` now reports `log_loss`; the backtest prints it in the
  comparison table, and `paired_bootstrap` reports `log_loss_diff`.
- Measurement-only change — Brier/accuracy/MAE numbers are identical to
  before. Held-out log loss: rolling-form baseline 1.068, venue-split
  1.062 (diff −0.0060, 95% CI [−0.0132, +0.0013] — includes zero, same
  verdict as Brier), R Dixon-Coles 1.028 (computed from its existing
  holdout CSV, n=342). Reference: uniform 1/3 guess = ln 3 ≈ 1.099.
- 3 tests added (49 pass).

**11. Strength-of-schedule-adjusted form (2026-09-29, step 2 of the 2026-10-03 milestone)**
- Hypothesis: raw rolling form over-rates teams that just played weak
  opponents; adjusting each past match for opponent strength should help.
- `add_sos_adjusted_form()` in `src/features/build_match_features.py`:
  `adj_goals_for = goals_for − (opp_avg_goals_against − league_avg)` and
  the mirror for goals against, where the opponent's averages are its
  expanding mean over matches strictly before this one and `league_avg`
  uses only strictly earlier dates. An opponent with < `min_matches_for_form`
  matches of history (e.g. newly promoted) counts as league-average, so no
  adjustment. Then a shifted rolling mean → `avg_sos_goals_{for,against}_last{w}`.
  Challenger feature list: `build_sos_feature_lists()` in `poisson_model.py`.
- Held-out result (n=374, same matches as the other sets): Brier 0.6437 vs.
  0.6446 baseline (diff −0.0008, 95% CI [−0.0023, +0.0006]); log loss
  1.0669 vs. 1.0678 (−0.0009, CI [−0.0029, +0.0010]); accuracy 44.4% vs.
  44.1% (+1 match, CI [−0.8, +1.6] pts). **Negative result**: the
  adjustment barely moves predictions and the effect is indistinguishable
  from zero. Plausible reason: over 5 matches most teams face a roughly
  average mix of opponents, so the correction mostly averages out.
- 3 tests added (52 pass): hand-calculated adjustment, no-history
  opponent fallback, and no leakage of the current match's score.

**12. Wikipedia player-data parser fixed and validated (2026-10-05, prerequisite for the 2026-10-10 milestone)**
- A squad-quality feature is only as good as the player table under it, and
  that table had three problems: 23 PL team-seasons didn't parse; 8 parsed
  with **zero** goals (all four Chelsea seasons, Arsenal 2024-25, Villa
  2023-24/2024-25, Liverpool 2022-23); and Man City's "Totals"/"Own goals"
  footer rows were parsed as players, doubling the team's goal sum.
- Causes and fixes in `src/cleaning/clean_player_stats.py`: three-level
  headers (a "Goalkeepers"/"Defenders" group row) broke the column lookup
  (now matches on header levels 0/1); goals tables live under "Goals",
  "Goalscorers" or "Goals & Assists" (all tried, via a shared
  `clean_goals_table`); Apps/Starts-only tables now take goals from that
  separate table instead of silently writing 0, and use the real Starts
  column; "Statistics" heading and "Apps." alias added; summary rows
  dropped; `#` footnote marker stripped.
- Coverage 57/80 -> **79/80** PL team-seasons (Wolves 2022-23 still
  missing; Ipswich 2025-26 is a Championship season and correctly skipped).
- Validated against match results: summed player league goals / actual
  team goals is 0.93-1.00 for nearly every team-season (the gap is opponent
  own goals). Two outliers are source gaps, left as-is since Bronze is never
  hand-edited: Bournemouth 2024-25's table omits Semenyo (11 goals; ratio
  0.79) and Luton 2023-24 under-reports appearances.
- 3 tests added (three-level header, Apps/Starts + Goalscorers fallback,
  summary rows not counted).

**13. Form window comparison: 3 vs. 5 vs. 10 matches (2026-10-05, completes the 2026-10-03 milestone)**
- `config.yaml -> features.form_windows` is now `[5, 3, 10]`; the first
  entry stays the model's window, the rest are backtest challengers
  (`rolling_form_last3`, `rolling_form_last10`).
- Held-out (n=374, same matches as every other Experiment 1 set), vs. the
  5-match baseline: last3 Brier +0.0011 (95% CI [-0.0022, +0.0043]), log
  loss +0.0016 ([-0.0028, +0.0060]), accuracy -1.1 pts ([-2.7, +0.3]);
  last10 Brier -0.0002 ([-0.0036, +0.0032]), log loss +0.0002
  ([-0.0043, +0.0047]), accuracy -0.8 pts ([-2.4, +0.8]).
- **Negative result**: window length doesn't matter in this range. Same
  verdict at alpha=0.01 (see 15).

**14. Squad-quality feature (2026-10-05, the 2026-10-10 milestone)**
- Hypothesis: a team's current roster's proven PL goal output carries
  information that 5-match form doesn't.
- `squad_prior_goals_per_match` = sum of the PREVIOUS season's PL goals (any
  club) of every player on this season's roster, / 38. Previous season,
  because Wikipedia's end-of-season totals would leak the predicted match
  and all later ones. A sum, not a rate, so low-appearance players can't
  dominate without needing a floor. NaN, not 0, for missing data. Full
  definition and the one known look-ahead (end-of-season rosters include
  January signings) in METHODOLOGY.md. Sanity check: correlation 0.65 with
  the same season's actual goals per match across 59 team-seasons.
- Design: no previous season exists for 2022-23, so this experiment trains
  every arm on 2023-24 + 2024-25 only, the baseline included, so the
  training set isn't a confound. A **control arm**
  (`prev_season_goals_per_match`: the team's own last-season goals per
  match from match results, 0 if promoted, no player data) separates
  "player information helps" from "any longer-horizon strength helps".
- Held-out (n=377), alpha=1.0, vs. baseline refit on the same rows:
  - squad quality: Brier 0.6380 vs. 0.6460, **-0.0080 [-0.0129, -0.0030]**;
    log loss **-0.0112 [-0.0177, -0.0046]**; accuracy 48.8% vs. 43.2%,
    **+5.6 pts [+2.4, +9.0]**. The first experiment in the project with
    all three intervals excluding zero.
  - control: Brier -0.0053 [-0.0107, +0.0001]; log loss -0.0078
    [-0.0149, -0.0005]; accuracy +1.9 pts [-1.6, +5.3].
  - squad vs. control (what the roster adds): Brier -0.0027
    [-0.0060, +0.0005]; log loss -0.0035 [-0.0079, +0.0008]; accuracy
    +3.7 pts [+1.6, +6.1].
  - Larger gain on matches involving a promoted team (Brier -0.0135, n=105)
    than on the rest (-0.0059, n=272), but present in both.
- vs. R Dixon-Coles on R's own 342 matches: squad model Brier 0.6350 vs. R
  0.6177 (+0.0173 [-0.0059, +0.0403]), accuracy 49.7% vs. 47.7% (+2.0 pts
  [-2.3, +6.4]). R still leads on the proper scoring rules; neither gap is
  distinguishable from noise.
- **Verdict: the most promising feature so far, but not established.** See
  15: the accuracy gain doesn't survive weaker regularization, and the
  Brier/log-loss gain keeps its direction and size but loses significance.
- 5 tests added: previous-season-only (no leakage of same-season goals),
  transfer credited to the new club, NaN-not-zero for missing data,
  promoted team computed, control arm values.

**15. Regularization robustness check (2026-10-05)**
- `PoissonRegressor(alpha=1.0)` penalizes **unstandardized** features. On
  the baseline it roughly halves the coefficients (home: 0.105/0.066 vs.
  0.223/0.151 unpenalized) and shrinks the spread of predicted goal rates
  from sd 0.33 to 0.15. Every recorded experiment used alpha=1.0, so all of
  them were run under heavy shrinkage. The penalty hits smooth,
  low-variance features (whole-history averages) harder than noisy 5-match
  form, which biases comparisons against long-horizon features.
- Made it configurable (`config.yaml -> model.poisson_alpha`, default
  1.0, so every number above is unchanged and reproducible) and re-ran
  everything at 0.01 (diagnostic only, default not changed):
  - baseline itself: Brier 0.6446 -> 0.6372 (-0.0074 [-0.0141, -0.0009]),
    log loss -0.0114 [-0.0204, -0.0025]. Similar in size to any feature
    gain so far.
  - **venue-split vs. baseline: Brier -0.0161 [-0.0309, -0.0014]**,
    excluding zero (it was "not distinguishable" at 1.0). Venue-split
    Brier 0.6211, close to R's 0.6177 (different samples).
  - squad vs. baseline: Brier -0.0092 [-0.0200, +0.0021], log loss -0.0141
    [-0.0286, +0.0012], accuracy -0.8 pts. The point estimates hold but
    the intervals now include zero, and the +5.6-pt accuracy gain at 1.0
    disappears: it was shrinkage relief (a second strength feature lets
    more signal through the penalty), not information.
  - squad vs. control: Brier -0.0070 [-0.0143, +0.0005]. At 1.0 the control
    captured ~2/3 of the squad gain; at 0.01 only ~1/4. So "how much is
    roster info vs. longer horizon" isn't stable across settings.
  - SOS and windows 3/10: still null.
- Also found: `season_avg_*` and `*_by_venue` are expanding means over each
  team's **whole history** in the data, not reset each season (grouped by
  team only). The venue-split "team-level" features were always a
  long-horizon signal. Docstrings corrected; column names left unchanged
  because the app uses them.
- Emerging pattern across all experiments: the features that help
  (whole-history venue split, previous-season squad/control, whole-history
  Dixon-Coles) are long-horizon strength signals; variations on short-term
  form (SOS adjustment, window 3/10) don't help.
- **Caveat:** alpha=0.01 was picked here only as a diagnostic, and it has
  now been seen to score well on the holdout. Adopting it on that basis
  would tune to the holdout. The defensible fix is to standardize features
  and choose alpha by walk-forward validation inside the training seasons
  (train 2022-23 -> validate 2023-24; train 2022-24 -> validate 2024-25),
  then run the holdout once.
- Also: ~8 challengers have now been compared against the same baseline,
  so a single 95% interval that excludes zero deserves less weight than it
  would alone. The squad Brier result at alpha=1.0 is far enough from zero
  to survive a Bonferroni correction for 8 comparisons (bootstrap 99.375%
  interval [-0.0149, -0.0011]), but its alpha sensitivity is the bigger
  issue.

## Current state / known gaps

- **Player-stats coverage**: 79/80 PL team-seasons parse and are
  validated against match goals (Wolves 2022-23 missing; Bournemouth
  2024-25 and Luton 2023-24 have source-level gaps). See item 12.
- **Regularization**: every recorded Python result uses alpha=1.0 on
  unstandardized features, which changes some conclusions (item 15). Not
  yet chosen properly. This is the most important open methodological
  issue before the freeze.
- **R model**: doesn't yet include the Dixon-Coles low-score correlation
  (`tau`/`rho`) adjustment from the original paper — just the team
  attack/defense parameterization.
- **Held-out backtest now exists** (season 2526) — see item 8 above for real
  out-of-sample numbers. Both models still trained/evaluated only once (no
  cross-validation across multiple holdout seasons), and R still can't score
  newly promoted teams it has zero training history for.
- **Fixtures feed** only covers the next matchweek or two (a limitation of
  the free source, not something this project controls).
- **Player stats** feed only the squad-quality challenger (item 14), not the
  production baseline. No injury/lineup/availability data at all.
- `run_pipeline.py` covers only the original match pipeline (ingest → clean
  → features → train). Wikipedia ingestion and the R model are separate,
  manually-run steps, not wired in.
- Nothing in this project is committed automatically — ask before
  committing per the user's standing preference.

## Suggested next steps

- **Decide the regularization question before the freeze** (item 15): it
  changes which features look useful. Defensible version: standardize
  features, choose alpha by walk-forward validation inside the training
  seasons only, then re-run every experiment on the holdout once. A
  decision for the user; nothing has been changed yet.
- 2026-10-17 milestone: derby/rivalry analysis (EPL pairings only).
- More evaluation power: one holdout season (n~375) can't resolve effects
  of ~0.005 Brier. A walk-forward over 2024-25 and 2025-26 as two test
  seasons would roughly double n.
- Add the Dixon-Coles `tau`/`rho` low-score adjustment to the R model
  (post-freeze unless needed).
- Consider wiring Wikipedia ingestion + the R model into `run_pipeline.py`
  once both are stable enough to run unattended (not needed for either
  research question).
