# Soccer Match Forecaster — Project Memory

A personal quantitative-research project (not a general-purpose app) asking:
**how accurately can EPL match outcomes be forecast using historical team
and player information, and which factors actually improve out-of-sample
predictions?** — plus a secondary question, **are derby/rivalry matches
systematically harder to forecast?** The user is a CS/Data Science freshman
(math minor) aiming at quant research/finance, so this should read as a
research writeup (methodology, findings, limitations) with a hard freeze
date, not an open-ended feature-building app. See "Research plan &
deadlines" below.

Full context lives in three docs — read them, don't duplicate them here:

- [ABOUT.md](ABOUT.md) — goal, non-goals, full roadmap
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — how the data and models work
- [docs/PROGRESS.md](docs/PROGRESS.md) — session-by-session handoff log (the
  most detailed history — update it at the end of any substantial session)

This file is the fast-orientation summary: current state, active plan, and
working conventions.

**How to help on this project:** statistical validity over flashy features;
watch for data leakage / look-ahead bias in every new feature; chronological
/ walk-forward evaluation only, never a random split; don't reach for
Random Forest/XGBoost/etc. without a clear reason; frame work as hypotheses
+ experiments, not just code; say plainly when something isn't needed for
the two research questions above rather than building it anyway.

## Architecture (Bronze → Silver → Gold → Model → Prediction → Evaluation)

```
Bronze (raw)  →  Silver (cleaned)  →  Gold (features)  →  Model  →  Prediction  →  Evaluation
```

- **Bronze**: raw, byte-for-byte as downloaded (`data/bronze/`). Never edited in place.
- **Silver**: standardized, one row per match/player-season (`data/silver/`).
- **Gold**: model-ready feature tables (`data/gold/`).
- Config lives in `config/config.yaml` — leagues, seasons, paths, feature windows, model candidates. Change behavior there, not by hardcoding in scripts.

**Data sources**: football-data.co.uk (match results + near-term fixtures,
free, no key) and Wikipedia club-season articles via the MediaWiki API
(player appearances/goals). FBref and Understat were evaluated and rejected
(Cloudflare-blocked / no static data). Bookmaker odds are ingested but
deliberately never used as model inputs (would leak sharp-market signal).

**Models** (see METHODOLOGY.md for full detail):
1. **Python Poisson baseline** (`src/models/poisson_model.py`) — two independent `PoissonRegressor`s (home goals, away goals) from each team's rolling 5-match form. The baseline every other model must beat on held-out data.
2. **R Dixon-Coles model** (`src/models/dixon_coles_model.R`) — Poisson GLM with fixed team-specific attack/defense parameters fit on full match history. A genuinely different signal (whole-history strength vs. recent form), not a port of the Python model. Not wired into `run_pipeline.py` — run manually.

**App**: `app.py` (Streamlit) — searchable dropdown over past results, real
upcoming fixtures, and hypothetical matchups; predicted scores, win/draw/loss
probabilities, per-team squad view, Dev View (pre-match prediction vs.
actual), model comparison, and confidence-interval charts, with detail views
collapsed into expanders.

## Commands

```bash
python run_pipeline.py                                   # ingest -> clean -> features -> train (Python model only)
python -m src.predictions.generate_predictions --fixtures fixtures.csv
python -m src.evaluation.evaluate_predictions
python -m src.evaluation.backtest                         # real held-out backtest, Python model (see below)
Rscript src/models/dixon_coles_model.R                    # also runs its own held-out backtest — separate, manual, not in run_pipeline.py
streamlit run app.py
pytest                                                    # 46 tests as of last count
```

## Current state (see docs/PROGRESS.md for full detail)

Working end-to-end: ingestion → cleaning → features → two competing
forecasting models → evaluation loop → Streamlit app. Player stats are
scraped and displayed (squad view) but **do not yet feed either model**.

**Held-out backtest results** (season 2526 held out entirely from training —
see `src/evaluation/backtest.py` and the holdout section of
`src/models/dixon_coles_model.R`; both write their held-out predictions to
`data/gold/prediction_features/*_holdout_backtest.csv`):

| model / feature set | n | winner acc. | exact score | MAE (H/A) | Brier |
|---|---|---|---|---|---|
| Python Poisson — rolling 5-match form (original baseline) | 374 | 44.1% | 12.3% | 1.00 / 0.89 | 0.645 |
| Python Poisson — team-level venue-split (2026-09-26) | 374 | 45.7% | 12.3% | 0.98 / 0.91 | 0.640 |
| R Dixon-Coles (whole-history team fixed effects) | 342 (38 skipped — newly promoted teams never seen in training) | 47.7% | 12.0% | 0.94 / 0.86 | 0.618 |

R still leads on all metrics — smaller gap than the old in-sample numbers
(55% vs 46%) suggested, but the whole-history signal still looks real.
Python's venue-split team-level features (see 2026-09-26 below) score
slightly better than its rolling-form baseline on an identical 374-match
held-out sample (Brier −0.0051, 95% CI [−0.0104, +0.0003]; accuracy +6
matches, 95% CI [−0.8, +4.0] pts), but both intervals include zero — one
season is not enough to distinguish it from noise. Treat it as a leading
candidate, not an established improvement; keep both as reference points
until a multi-season/walk-forward check or a larger effect settles it. R wasn't re-engineered
this round: its fixed-effects design already encodes team-level strength
by construction, so this specific experiment (rolling-form vs. team-level)
doesn't have a natural R-side counterpart; R's numbers above are simply
reconfirmed unchanged. Re-run both backtests after each change and update
this table.

**Known gaps**
- R model missing the Dixon-Coles `tau`/`rho` low-score correlation adjustment.
- 24/81 team-seasons of Wikipedia player data still fail to parse (a 4th, unhandled table layout).
- No injury/lineup data at all — `data/bronze/injuries/` and `data/silver/injuries_clean/` are empty placeholders; no ingestion source identified yet (free real-time injury data is harder to source than match results — most options are paywalled or JS-rendered).
- `run_pipeline.py` only runs the Python pipeline; Wikipedia ingestion and the R model are separate manual steps.
- Fixtures feed only covers the next matchweek or two (source limitation).

## Research plan & deadlines (set 2026-09-23 — this supersedes the old open-ended roadmap below)

Hard freeze target: **2026-10-24**, a finished research version answering
both questions in the header. After that date, xG, injuries, Random
Forest/XGBoost are *future extensions*, not requirements — don't pull them
forward into this scope. Build and evaluate one feature at a time against
the held-out backtest (`src/evaluation/backtest.py` / the R holdout
section) — never a batch — so each change's effect is attributable.

- [x] **Chronological train/test split** (done 2026-09-20) — see results table above.
- [x] **2026-09-26 — team-level features** (done 2026-09-26). Research question: *does better team-level information improve predictions over the current rolling-form baseline?* Answer: **suggestive but not established** — venue-split scored slightly better, but a paired bootstrap on the single 374-match holdout gives 95% intervals that include zero for both Brier and accuracy.
  - [x] Average goals scored (blended) — already existed (`season_avg_goals_for`), unchanged.
  - [x] Average goals conceded (blended) — already existed (`season_avg_goals_against`), unchanged.
  - [x] Home goals scored/conceded — new `avg_goals_for_by_venue` / `avg_goals_against_by_venue`, computed only from a team's own past HOME matches (`src/features/build_match_features.py`'s `add_venue_split_averages`).
  - [x] Away goals scored/conceded — same function, away-only subset.
  - [x] Verified no future-information leakage — by hand (Arsenal's 4th home match's feature = mean of exactly its prior 3 home goals, not including its own result) and by unit test (`tests/test_features.py::test_venue_split_no_leakage_of_current_match`, `test_venue_split_ignores_other_venue_matches`).
  - [x] Re-ran both models — Python via `src/evaluation/backtest.py` (now compares named feature sets head-to-head on an identical held-out sample); R via `Rscript src/models/dixon_coles_model.R` (numbers unchanged, as expected — see note below the table).
  - [x] Added a paired bootstrap (`paired_bootstrap` in `src/evaluation/backtest.py`) so every feature comparison reports a noise estimate, not just point differences — use it for all later milestones.
  - [x] Compared against baseline — see table above. `build_team_level_feature_lists()` in `src/models/poisson_model.py` is the new candidate; `build_feature_lists()` (rolling form) remains the old baseline, both still available for comparison, not one deleted in favor of the other.
- [ ] **2026-10-03** — strength-of-schedule adjustment; compare 3/5/10-match form windows against each other. Record Brier score, log loss, and accuracy for each variant (log loss isn't computed anywhere yet — add it alongside Brier in `evaluate_predictions.compute_metrics`).
- [ ] **2026-10-10** — squad-quality feature from existing Wikipedia player data, aggregated to team level; guard against low-appearance players dominating the metric (e.g. minimum-minutes/appearances floor before a player's per-90 rate counts, per `config.yaml`'s existing `min_matches_for_form` pattern). Test whether it actually moves the backtest numbers — a negative result here is a valid, reportable finding, not a failure.
- [ ] **2026-10-17** — derby/rivalry research: define derby matches **using EPL pairings only** (e.g. Man Utd–Man City, Arsenal–Tottenham, Liverpool–Everton) — the user's example derbies (Milan–Inter, Real Madrid–Atlético) aren't in this dataset, which is EPL-only (`config.yaml -> active.leagues: ["E0"]`); expanding leagues is a separate scope decision, not assumed here. Compare derby vs. non-derby calibration/accuracy, analyze home advantage, test stability, and explicitly document the small-sample caveat (each derby pairing has at most ~8 meetings across the 4 seasons of data).
- [ ] **2026-10-24** — freeze feature set, final backtest, final model comparison, final derby analysis, graphs/tables, updated README + METHODOLOGY, reproducible/clean repo.

**Blocked, not scheduled before the freeze** (research these only if time allows, don't let them block the plan above):
- xG/xGA — both free sources evaluated (FBref, Understat) are dead ends; would need a new source found first.
- Player availability/injuries — no free source identified at all (`data/bronze/injuries/` still empty).
- Random Forest / XGBoost — deliberately deferred; a new algorithm on the same features being tested pre-freeze won't answer either research question.

**Original open-ended roadmap** (ABOUT.md, condensed — kept for reference, now subordinate to the plan above):
1. ~~Framework~~ — done.
2. More match-level signal — xG, head-to-head, strength-of-schedule.
3. Player layer — form, goal contributions, minutes, injuries/lineups.
4. Model comparison — logistic regression / random forest / XGBoost vs. Poisson baseline.
5. Explainability layer, 6. Automation, 7. Dashboard polish.

## Working conventions

- **Never commit automatically** — ask first, every time (standing user preference; nothing in this project has been committed without being asked).
- Explainability over accuracy: a slightly weaker but interpretable model beats a black box (ABOUT.md non-goal).
- New models must beat the Poisson baseline on a **held-out chronological split**, not in-sample, before being adopted (see METHODOLOGY.md's model comparison protocol) — never shuffle time-series data randomly.
- No paid data feeds or APIs — must stay free to run.
- Odds columns are ingested (Bronze) but must never be used as model features.
- One feature/change at a time, re-backtest after each, before moving to the next — batching changes makes it impossible to attribute an improvement (or regression) to a specific cause.
- A feature that doesn't move the backtest numbers is a valid documented finding for the research writeup, not wasted work — don't discard negative results.
