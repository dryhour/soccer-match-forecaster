# Soccer Match Forecaster — Project Memory

A free, explainable, reproducible soccer match forecasting system. Full
context lives in three docs — read them, don't duplicate them here:

- [ABOUT.md](ABOUT.md) — goal, non-goals, full roadmap
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — how the data and models work
- [docs/PROGRESS.md](docs/PROGRESS.md) — session-by-session handoff log (the
  most detailed history — update it at the end of any substantial session)

This file is the fast-orientation summary: current state, active plan, and
working conventions.

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
Rscript src/models/dixon_coles_model.R                    # separate, manual — not in run_pipeline.py
streamlit run app.py
pytest                                                    # 21 tests as of last count
```

## Current state (see docs/PROGRESS.md for full detail)

Working end-to-end: ingestion → cleaning → features → two competing
forecasting models → evaluation loop → Streamlit app. Player stats are
scraped and displayed (squad view) but **do not yet feed either model**.

**Known gaps**
- No true held-out backtest for either model — both scored in-sample (R more so).
- R model missing the Dixon-Coles `tau`/`rho` low-score correlation adjustment.
- 24/81 team-seasons of Wikipedia player data still fail to parse (a 4th, unhandled table layout).
- No injury/lineup data at all — `data/bronze/injuries/` and `data/silver/injuries_clean/` are empty placeholders; no ingestion source identified yet (free real-time injury data is harder to source than match results — most options are paywalled or JS-rendered).
- `run_pipeline.py` only runs the Python pipeline; Wikipedia ingestion and the R model are separate manual steps.
- Fixtures feed only covers the next matchweek or two (source limitation).

## Roadmap (from ABOUT.md, condensed)

1. ~~Framework~~ — done (ingestion, cleaning, features, Poisson baseline, prediction logging, evaluation).
2. More match-level signal — xG, head-to-head history, strength-of-schedule.
3. **Player layer** (biggest remaining lift, not started) — player-level features (form, goal contributions, minutes), injuries/suspensions/lineups → squad-availability-adjusted team strength.
4. Model comparison — logistic regression / random forest / XGBoost challengers vs. Poisson baseline, promoted only if they win on held-out metrics (`config.yaml` already lists these as `model.candidates`).
5. Explainability layer — per-prediction factor breakdown.
6. Automation — scheduled pipeline runs (e.g. GitHub Actions).
7. Dashboard polish — league/team selectors, prediction history, model comparison view.

## Active TODO / next steps

Prioritized 2026-09-13 by expected impact on prediction quality (feature/eval
work first, more complex models later — a Random Forest can't be evaluated
meaningfully without a real held-out split, and won't outperform the Poisson
baseline just by existing if the input features are the same). Build and
evaluate one item at a time against the held-out split, not in a batch —
otherwise you can't tell which change actually helped.

**Do first (blocks everything else):**
- [ ] **Real chronological train/test split** for both models, per METHODOLOGY.md's comparison protocol. Nothing below this line can be trusted without it — right now "improvements" are unfalsifiable.

**High-impact, buildable now (existing data, no new sources needed):**
- [ ] Team average goals scored / conceded, with explicit home vs. away splits (current model only has blended 5-match rolling form).
- [ ] Strength-of-schedule adjustment — a good recent record against weak teams shouldn't score the same as against strong teams.
- [ ] Team attacking/defensive strength features (rolling goals, shots, shots on target as a bundle, not goals alone).
- [ ] Multiple form windows (3/5/10-match) evaluated against each other — only 5 is used today.
- [ ] Squad-quality feature from existing Wikipedia data (weighted goal contribution of top scorers) — cheapest player-layer step available today.

**High-impact, blocked on a data source (research before building):**
- [ ] **xG / xGA** — ranked as the single biggest missing signal, but both free sources this project evaluated are dead ends (FBref: Cloudflare-blocked; Understat: no longer static in page source). Needs a new source investigated before this is buildable at all — don't schedule the feature work until a source is found.
- [ ] **Player availability (injuries/suspensions)** — same blocker: no free source identified yet (`data/bronze/injuries/` is still empty). High priority but high implementation difficulty for the same reason as xG.
- [ ] Predicted/starting lineups — depends on finding a reliable free source; likely harder than injuries alone.

**Player layer, once squad-quality is in place:**
- [ ] Player minutes played, goals/assists per 90, goal contributions per 90 (rate stats, not raw totals — accounts for playing time).
- [ ] Player match ratings, aggregated into team-level "current squad form."

**Lower priority (real but smaller expected impact):**
- [ ] Head-to-head history (stale as squads/managers turn over — deprioritized vs. xG/team-strength).
- [ ] Rolling goal difference, shots/shots-on-target per match, possession.
- [ ] Manager changes, rest days, fixture congestion, travel distance, weather — later experimentation only.

**Model work (deliberately last):**
- [ ] Add Random Forest / XGBoost as challengers to the Poisson baseline (`config.yaml` already lists them as candidates) — only once the held-out split exists and the feature set above has moved, so the comparison is meaningful rather than same-features-different-algorithm.
- [ ] Add Dixon-Coles `tau`/`rho` low-score adjustment to the R model.

**Housekeeping:**
- [ ] Extend `clean_player_stats.py` for the 4th Wikipedia table layout (remaining 24 team-seasons).
- [ ] Consider wiring Wikipedia ingestion + R model into `run_pipeline.py` once stable enough to run unattended.

**Top 5 missing signals overall** (for quick reference): xG/xGA → team
attacking/defensive strength → strength of schedule → player availability →
player form/contribution.

## Working conventions

- **Never commit automatically** — ask first, every time (standing user preference; nothing in this project has been committed without being asked).
- Explainability over accuracy: a slightly weaker but interpretable model beats a black box (ABOUT.md non-goal).
- New models must beat the Poisson baseline on a **held-out chronological split**, not in-sample, before being adopted (see METHODOLOGY.md's model comparison protocol) — never shuffle time-series data randomly.
- No paid data feeds or APIs — must stay free to run.
- Odds columns are ingested (Bronze) but must never be used as model features.
