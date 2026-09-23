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
- **Result: yes, modestly.** Team-level venue-split beat rolling-form on
  winner accuracy (45.7% vs. 44.1%) and Brier score (0.640 vs. 0.645) on
  the identical 374-match held-out sample; exact-score accuracy tied,
  goal MAE mixed (home slightly better, away slightly worse). This is now
  the Python baseline to beat going forward.
- R wasn't changed this round and its numbers are unchanged (reconfirmed by
  rerunning) — its fixed-effects design already encodes team-level
  attack/defense strength by construction, so "rolling-form vs. team-level"
  isn't a natural experiment on the R side the way it is for Python.

## Current state / known gaps

- **Player-stats coverage**: 24/81 team-seasons still unparsed (a 4th
  Wikipedia table layout, not yet reverse-engineered). Concentrated in
  2022-23/2023-24.
- **R model**: doesn't yet include the Dixon-Coles low-score correlation
  (`tau`/`rho`) adjustment from the original paper — just the team
  attack/defense parameterization.
- **Held-out backtest now exists** (season 2526) — see item 8 above for real
  out-of-sample numbers. Both models still trained/evaluated only once (no
  cross-validation across multiple holdout seasons), and R still can't score
  newly promoted teams it has zero training history for.
- **Fixtures feed** only covers the next matchweek or two (a limitation of
  the free source, not something this project controls).
- **Player stats are display-only** — not feeding into either forecasting
  model yet.
- `run_pipeline.py` covers only the original match pipeline (ingest → clean
  → features → train). Wikipedia ingestion and the R model are separate,
  manually-run steps, not wired in.
- Nothing in this project is committed automatically — ask before
  committing per the user's standing preference.

## Suggested next steps

- Extend `clean_player_stats.py` to handle the 4th Wikipedia layout (the
  remaining 24 team-seasons).
- Feed player/squad-availability data into the forecasting models
  (roadmap stage 3 — this is the biggest remaining lift).
- Add the Dixon-Coles `tau`/`rho` low-score adjustment to the R model.
- Real chronological train/test split for both models, per the model
  comparison protocol already documented in METHODOLOGY.md.
- Consider wiring Wikipedia ingestion + the R model into `run_pipeline.py`
  once both are stable enough to run unattended.
