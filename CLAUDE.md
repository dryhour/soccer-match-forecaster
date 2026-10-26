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
2. **R Dixon-Coles model** (`src/models/dixon_coles_model.R`) — Poisson GLM with fixed team-specific attack/defense parameters fit on full match history. A genuinely different signal (whole-history strength vs. recent form), not a port of the Python model. Run manually or via `run_pipeline.py --reproduce`.

**Frozen research result (2026-10-26):** the frozen Python model is **venue-split team averages, unpenalized** (`build_team_level_feature_lists`, alpha=0, standardized) — chosen by `src/evaluation/final_evaluation.py`. It ties R. The app still serves the original 5-match alpha=1.0 model (not switched at the freeze).

**App**: `app.py` (Streamlit) — searchable dropdown over past results, real
upcoming fixtures, and hypothetical matchups; predicted scores, win/draw/loss
probabilities, per-team squad view, Dev View (pre-match prediction vs.
actual), model comparison, and confidence-interval charts, with detail views
collapsed into expanders.

## Commands

```bash
python run_pipeline.py --reproduce                       # offline: rebuild EVERY research result from committed Bronze (needs R)
python run_pipeline.py                                   # live: download -> clean -> features -> train (Python model only)
python -m src.predictions.generate_predictions --fixtures fixtures.csv
python -m src.evaluation.evaluate_predictions
python -m src.evaluation.backtest                         # real held-out backtest, Python model (see below)
Rscript src/models/dixon_coles_model.R                    # also runs its own held-out backtest + walk-forward — separate, manual, not in run_pipeline.py
python -m src.evaluation.derby_analysis                   # derby research question; run the R script first (reads its walk-forward CSV)
python -m src.evaluation.final_evaluation                 # FINAL: penalty chosen inside training seasons, holdout scored once (needs backtest + R CSVs)
python -m src.evaluation.figures                          # docs/figures/*.png from the saved CSVs (no refitting)
streamlit run app.py
pytest                                                    # 80 tests as of last count
```

## Current state (see docs/PROGRESS.md for full detail)

**Research version FROZEN** (2026-10-24 milestone, final run 2026-10-26).
README.md is now the research writeup (summary, methods, results, failed
hypotheses, limitations, figures). `python run_pipeline.py --reproduce`
regenerates every derived file byte-for-byte from committed Bronze
(verified in a clean copy). Post-freeze work is optional; see PROGRESS.md
"Suggested next steps".

**Final results (authoritative — `final_evaluation.py`, PROGRESS.md item 18):**
standardized features, each set's alpha chosen by walk-forward validation
inside the training seasons (alpha=0 won for 8 of 9 sets; combined windows
0.03), 2025-26 holdout scored once, 2024-25 replication from a nested
walk-forward. Brier vs. own baseline, holdout / replication:
venue-split **−0.0163 [−0.0313, −0.0012] / −0.0257 [−0.0426, −0.0092]**
(established; not Bonferroni-significant on the holdout alone); 10-match
−0.0049 / **−0.0142** (suggestive); 3-match +0.0045 / **+0.0104** (worse);
combined 3+5+10 −0.0056 / −0.0083 (no); SOS null; squad quality −0.0091
[−0.0201, +0.0023] (not established — its alpha=1.0 win was shrinkage
relief); control null. Vs. base rate (holdout): R −0.038, venue-split
−0.036, 5-match −0.019 (−0.012 at alpha=1.0). R vs. venue-split −0.0006
[−0.0142, +0.0130]: tied.

The table below is the **development record** (alpha=1.0 on unstandardized
features unless noted) — kept because it's reproducible and documents how
conclusions changed, but superseded by the final results above.

Working end-to-end: ingestion → cleaning → features → two competing
forecasting models → evaluation loop → Streamlit app. Player stats (79/80
PL team-seasons, validated against match goals) are displayed in the squad
view and feed the squad-quality **challenger** feature only, not the
production baseline.

**Held-out backtest results** (season 2526 held out entirely from training —
see `src/evaluation/backtest.py` and the holdout section of
`src/models/dixon_coles_model.R`; both write their held-out predictions to
`data/gold/prediction_features/*_holdout_backtest.csv`):

| model / feature set | n | winner acc. | exact score | MAE (H/A) | Brier | log loss |
|---|---|---|---|---|---|---|
| *Reference:* base rate — training-season H/D/A frequencies for every match, no team information (2026-10-16) | 374 | 42.5% | 12.3% | 1.00 / 0.91 | 0.656 | 1.085 |
| Python Poisson — rolling 5-match form (original baseline) | 374 | 44.1% | 12.3% | 1.00 / 0.89 | 0.645 | 1.068 |
| Python Poisson — team-level venue-split (2026-09-26) | 374 | 45.7% | 12.3% | 0.98 / 0.91 | 0.640 | 1.062 |
| Python Poisson — SOS-adjusted rolling 5-match form (2026-09-29) | 374 | 44.4% | 12.3% | 0.99 / 0.90 | 0.644 | 1.067 |
| Python Poisson — rolling 3-match form (2026-10-05) | 374 | 43.0% | 12.0% | 1.00 / 0.90 | 0.646 | 1.069 |
| Python Poisson — rolling 10-match form (2026-10-05) | 374 | 43.3% | 12.3% | 0.99 / 0.89 | 0.644 | 1.068 |
| *Squad experiment — all arms trained on 2324+2425 only (squad feature undefined for 2223):* | | | | | | |
| Python Poisson — rolling 5-match form, same rows | 377 | 43.2% | 12.5% | 1.00 / 0.91 | 0.646 | 1.070 |
| Python Poisson — + last season's team goals/match (control, no player data) | 377 | 45.1% | 11.9% | 0.99 / 0.90 | 0.641 | 1.063 |
| Python Poisson — + squad quality (2026-10-05) | 377 | 48.8% | 11.9% | 0.98 / 0.90 | 0.638 | 1.059 |
| R Dixon-Coles (whole-history team fixed effects) | 342 (38 skipped — newly promoted teams never seen in training) | 47.7% | 12.0% | 0.94 / 0.86 | 0.618 | 1.028 |

Log loss reference point: a uniform 1/3-1/3-1/3 guess scores ln 3 ≈ 1.099.
The tougher reference is the base-rate row (its accuracy = always picking
home). Both models beat it on the proper scoring rules — Python baseline
Brier −0.0118 [−0.0176, −0.0061], R −0.0383 [−0.0660, −0.0100] on R's
342 matches — but neither beats it on accuracy beyond noise, and Python's
whole edge over "know nothing about the teams" is about a third of R's.
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

**All Python rows in this table use `config.yaml -> model.poisson_alpha: 1.0`**,
an L2 penalty on unstandardized features that roughly halves the
coefficients and penalizes smooth whole-history features hardest. That
setting stays in config so `backtest.py` / `derby_analysis.py` reproduce
these numbers; the properly-chosen penalty lives in `config.yaml ->
final_evaluation` (resolved at the freeze, see above). Also: despite their
names, `season_avg_*` and `*_by_venue` are whole-history expanding means,
not reset each season.

**Derby question (2026-10-15):** answered as a well-documented null. On
pooled walk-forward predictions (2023-24..2025-26, ~145 local derbies per
model), neither model forecasts derbies measurably worse, in raw or in
excess-over-expected loss. The test can only detect a Brier gap of 0.05-0.08
— see the 2026-10-17 checklist below and PROGRESS.md item 16.

**Known gaps**
- R model missing the Dixon-Coles `tau`/`rho` low-score correlation adjustment.
- Regularization: resolved at the freeze (final_evaluation.py chooses it inside the training seasons). The recorded experiments, the derby analysis and the app's model still use alpha=1.0 on purpose (reproducibility); don't read their Python numbers as the final ones.
- The app serves the alpha=1.0 5-match model, not the frozen venue-split model.
- The 2025-26 holdout has been seen by every development experiment; the 2024-25 replication is the cleaner test, 2026-27 would be the first untouched one.
- Wikipedia player data: 1/80 PL team-seasons unparsed (Wolves 2022-23); Bournemouth 2024-25 (Semenyo missing) and Luton 2023-24 (apps under-reported) are source-level gaps.
- No injury/lineup data at all — `data/bronze/injuries/` and `data/silver/injuries_clean/` are empty placeholders; no ingestion source identified yet (free real-time injury data is harder to source than match results — most options are paywalled or JS-rendered).
- `run_pipeline.py` default (live) mode only runs the Python match pipeline; `--reproduce` runs every research step (incl. R) offline from committed Bronze. Wikipedia *ingestion* remains manual.
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
- [x] **2026-10-03** — strength-of-schedule adjustment; compare 3/5/10-match form windows against each other. Record Brier score, log loss, and accuracy for each variant. (done 2026-10-05)
  - [x] Log loss added (2026-09-28) — `per_match_log_loss` in `evaluate_predictions.py`, reported by `compute_metrics`, the backtest table, and `paired_bootstrap` (`log_loss_diff`). Measurement only; no feature or model changed.
  - [x] Strength-of-schedule-adjusted rolling form (window 5) vs. raw baseline (2026-09-29) — `add_sos_adjusted_form` / `build_sos_feature_lists`. **Negative result**: Brier −0.0008 (95% CI [−0.0023, +0.0006]), log loss −0.0009 (CI [−0.0029, +0.0010]), +1 correct match — indistinguishable from zero.
  - [x] 3/5/10-match window comparison (2026-10-05) — `form_windows: [5, 3, 10]` (first = model window). **Negative result**: vs. 5, last3 Brier +0.0011 (CI [−0.0022, +0.0043]), last10 −0.0002 (CI [−0.0036, +0.0032]); same at alpha=0.01.
- [x] **2026-10-10** — squad-quality feature from existing Wikipedia player data, aggregated to team level; guard against low-appearance players dominating the metric. (done 2026-10-05) Answer: **most promising feature so far, not established.**
  - [x] Fixed the player-data parser first: coverage 57/80 → 79/80 PL team-seasons, 8 zero-goal team-seasons and a doubled Man City total fixed, validated against actual team goals (PROGRESS.md item 12).
  - [x] `squad_prior_goals_per_match` = current roster's PREVIOUS-season PL goals / 38 (`build_squad_quality`). Previous season because same-season Wikipedia totals would leak; a sum instead of a rate so low-appearance players can't dominate (no floor needed); NaN, not 0, for missing data. One known look-ahead: end-of-season rosters include January signings (composition, not results).
  - [x] Control arm with no player data (`prev_season_goals_per_match`, last season's team goals) and all arms refit on identical rows.
  - [x] Result at alpha=1.0 (n=377): vs. baseline Brier −0.0080 [−0.0129, −0.0030], log loss −0.0112 [−0.0177, −0.0046], accuracy +5.6 pts [+2.4, +9.0], all excluding zero. vs. the control the increment is Brier −0.0027 [−0.0060, +0.0005] (includes 0).
  - [x] Robustness at alpha=0.01: the Brier/log-loss point estimates hold (−0.0092) but the intervals include zero, and the accuracy gain vanishes (shrinkage artifact). See PROGRESS.md items 14–15.
- [x] **2026-10-17** — derby/rivalry research, EPL pairings only. (done 2026-10-15) Answer: **no evidence derbies are harder to forecast, and the test is underpowered.** Full detail: PROGRESS.md item 16, METHODOLOGY.md "Derby / rivalry analysis".
  - [x] Definitions fixed in `config.yaml -> derbies` before any result was computed: primary `local` (same metro area, 25 pairings, 194 matches), sensitivity `classic` (Manchester / North London / Merseyside, 24) and `rivalry` (9 named rivalries incl. between-city, 58). Milan–Inter / Real–Atlético aren't in the data (EPL-only); expanding leagues was not done.
  - [x] Walk-forward sample (2324 ← 2223, 2425 ← 2223–24, 2526 ← 2223–25) for both models, ~3x one holdout season: Python `walk_forward_poisson`, R section 4 (holdout CSV byte-identical, 2526 fold reproduces the existing holdout exactly).
  - [x] Derby vs. non-derby, split into raw loss and *excess* loss (observed − the loss the model expects from its own probabilities), so "close matchups" is separated from "model misjudges derbies". Local, pooled: Python Brier +0.013 [−0.022, +0.048], R +0.001 [−0.058, +0.059]; excess Brier Python +0.013, R −0.019, both intervals include 0.
  - [x] Calibration by outcome: draws, favorite win rate, goal-difference residual; no interval excludes 0 and the favorite effect has opposite signs in the two models.
  - [x] Home advantage (actual results, all 1,520 matches): local derbies 0.13 home−away goals vs. 0.30 elsewhere, −0.17 [−0.44, +0.10]. About half the usual size by point estimate, not distinguishable from 0.
  - [x] Stability: other definitions, per season (sign flips), pair-cluster bootstrap, common matches, alpha=0.01, leave-one-pair-out. 4 of 70 intervals exclude 0 (≈3.5 expected by chance), none in the "harder" direction.
  - [x] Small-sample caveat quantified: minimum detectable Brier difference 0.050 (Python) / 0.084 (R), larger than each model's whole skill over a uniform guess. Even zero skill on derbies would be detected only ~50% of the time.
- [x] **2026-10-24** — freeze feature set, final backtest, final model comparison, final derby analysis, graphs/tables, updated README + METHODOLOGY, reproducible/clean repo. (done 2026-10-26) Full detail: PROGRESS.md item 18.
  - [x] Regularization settled without the holdout: standardized features, alpha per feature set by walk-forward validation inside the training seasons (`config.yaml -> final_evaluation`, `final_evaluation.select_alpha`). alpha=0 for 8 of 9 sets.
  - [x] Holdout scored once with pre-stated decision rule + Bonferroni; 2024-25 replication via nested walk-forward (alpha re-chosen from 2023-24 only).
  - [x] README's unchecked 10/03 item "combining windows" run as one more pre-planned challenger (`build_combined_window_feature_lists`): not established.
  - [x] Frozen Python model: venue-split, unpenalized (wins under protocol rule 3; ties R). App not switched.
  - [x] Derby analysis finalized as-is (its alpha=0.01 robustness run ≈ unpenalized, same answer); figures added.
  - [x] Figures (`src/evaluation/figures.py` → `docs/figures/`, light + dark), README rewritten as the writeup, METHODOLOGY "Final evaluation" section.
  - [x] `run_pipeline.py --reproduce`; clean-copy run regenerated all 47 derived files byte-for-byte. Unused xgboost/jupyter deps and `model.candidates` config removed.

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
