# Soccer Match Forecaster

A free, explainable, reproducible system for forecasting **English Premier League match outcomes** using historical team and player data.

The project focuses on **statistical forecasting, out-of-sample evaluation, and quantitative research methodology** rather than simply building the most complex model possible.

## Research Question

> **How accurately can soccer match outcomes be forecast using historical team and player information, and which factors actually improve out-of-sample predictions?**

A secondary research question is:

> **Are derby and rivalry matches systematically harder to forecast than ordinary matches?**

The project will test these questions using chronological backtesting and statistical evaluation.

---

## Hypotheses

### Primary Hypothesis

Adding relevant information to a baseline team-strength model will improve the quality of match predictions.

Potential information includes:

- Team scoring and conceding rates
- Home/away performance
- Recent form
- Strength of schedule
- Shots and shots on target
- Squad quality
- Player performance

Features will only be used when the information would have been available **before the match being predicted**.

### Derby Hypothesis

Derby and rivalry matches may be more difficult to forecast than ordinary matches because historical and contextual factors may make standard team-strength assumptions less reliable.

The project will compare model performance on derby matches with performance on the broader dataset.

---

## Project Pipeline

```text
Bronze
  ↓
Silver
  ↓
Gold
  ↓
Features
  ↓
Statistical Model
  ↓
Prediction
  ↓
Out-of-Sample Evaluation
  ↓
Research Findings
```

### Bronze

Raw data collected from external sources.

```text
Raw Matches
Raw Players
Raw Teams
Raw Lineups
Raw League Data
```

### Silver

Cleaned and standardized data.

```text
Cleaned Matches
Cleaned Players
Cleaned Teams
Cleaned Lineups
```

### Gold

Model-ready datasets containing engineered features.

```text
Match Features
Team Features
Player Features
Prediction Features
```

---

## Data Sources

All project data must be available for free.

### football-data.co.uk

Historical Premier League match results from the 2022-23 through 2025-26 seasons, containing approximately 1,520 matches, along with near-term scheduled fixtures.

### Wikipedia / MediaWiki API

Club-season pages are used for player appearances and goals.

This data currently exists mainly as squad information and is being incorporated into the forecasting features.

### xG Data

FBref and Understat were investigated as free xG sources but are currently blocked or unsuitable for the project's automated pipeline.

Finding a reliable free xG/xGA source remains a future research task.

### Injury Data

No reliable free injury/availability source has been identified yet.

---

## Current Models

### Python Poisson Model

The baseline model uses two independent Poisson regressions to estimate:

- Home goals
- Away goals

The current baseline uses each team's rolling 5-match form.

### R Dixon-Coles Model

The second model uses a Poisson GLM with team-specific attack and defense parameters based on the classic Dixon-Coles approach.

The current implementation does not include the original low-score correlation adjustment (`tau` / `rho`).

The two implementations are intentionally different statistical approaches.

Run the Dixon-Coles model with:

```bash
Rscript src/models/dixon_coles_model.R
```

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the current methodology.

---

## Current Out-of-Sample Results

The project previously evaluated both models in-sample, which made the comparison unreliable because models were evaluated on matches they had already seen.

The current evaluation uses a **chronological train/test split**:

```text
Earlier Seasons
      ↓
   Training
      ↓
Most Recent Season
      ↓
     Test
      ↓
Compare Predictions
     vs.
Actual Results
```

Current results:

| Model          | Winner Accuracy | Exact Score | Brier Score |
| -------------- | --------------: | ----------: | ----------: |
| Python Poisson |           43.8% |       12.5% |       0.646 |
| R Dixon-Coles  |           47.7% |       12.0% |       0.618 |

The current results show a smaller difference between the models than the previous in-sample comparison.

The Dixon-Coles model also currently cannot score matches involving teams with no appearances in the training data. This caused 38 test matches involving such teams to be skipped.

---

# Features

The project will investigate the predictive value of:

- Team strength
- Goals scored
- Goals conceded
- Home/away performance
- Recent form
- Strength of schedule
- Shots
- Shots on target
- Squad quality
- Player performance
- Historical matchups
- Derby/rivalry status

Features will be tested rather than automatically assumed to improve predictions.

---

# Evaluation

The models will primarily be evaluated using **unseen historical matches**.

Metrics include:

- Win / draw / loss accuracy
- Brier score
- Log loss
- Probability calibration
- Goal prediction error
- Exact score accuracy
- Performance over time
- Performance by team
- Performance on derby matches

The main focus is on **probabilistic forecasting quality**, not exact-score accuracy alone.

---

# Research Timeline

The project is currently in development. The target is to complete the main research version by **10/24/2026**.

## 09/26/2026 — Team-Level Features

Complete:

- [ ] Average goals scored
- [ ] Average goals conceded
- [ ] Home goals scored/conceded
- [ ] Away goals scored/conceded
- [ ] Verify no future information is used
- [ ] Re-run both models
- [ ] Compare results against the current baseline

**Research question:**

> Does better team-level information improve predictions over the current rolling-form baseline?

---

## 10/03/2026 — Strength of Schedule & Form

Complete:

- [ ] Strength-of-schedule adjustment
- [ ] 3-match form
- [ ] 5-match form
- [ ] 10-match form
- [ ] Compare different form windows
- [ ] Test whether combining windows helps
- [ ] Record accuracy, Brier score, and log loss

**Research question:**

> Which representation of recent team performance provides the most useful predictive information?

---

## 10/10/2026 — Squad Quality

Complete:

- [ ] Build squad-quality feature
- [ ] Use existing Wikipedia player data
- [ ] Account for appearances / sample size
- [ ] Aggregate player information into team-level features
- [ ] Add squad quality to the model
- [ ] Compare against the previous best model

**Research question:**

> Does information about squad quality improve predictions beyond team-level match history?

---

## 10/17/2026 — Derby Analysis & Robustness

Complete:

- [ ] Define derby/rivalry matches
- [ ] Create derby subset
- [ ] Test examples such as AC Milan vs Inter, Manchester United vs Manchester City, and Real Madrid vs Atlético Madrid
- [ ] Compare derby vs non-derby predictions
- [ ] Compare calibration
- [ ] Analyze home advantage
- [ ] Perform robustness checks
- [ ] Document limitations caused by the smaller derby sample

**Research question:**

> Are derby/rivalry matches systematically harder for the forecasting models to predict?

---

## 10/24/2026 — Final Research Version

Complete:

- [ ] Freeze final feature set
- [ ] Run final chronological backtest
- [ ] Finalize model comparison
- [ ] Finalize derby analysis
- [ ] Create final tables and visualizations
- [ ] Document major findings
- [ ] Document failed hypotheses
- [ ] Document limitations
- [ ] Finish `README.md`
- [ ] Finish `docs/METHODOLOGY.md`
- [ ] Clean repository
- [ ] Verify reproducibility

**Final goal:**

Produce a complete quantitative research project that can clearly explain:

1. How accurately the models forecast EPL matches
2. Which features improve predictions
3. How the statistical models compare
4. Whether player/squad information adds predictive value
5. Whether derby matches behave differently
6. Where the models fail
7. What should be investigated next

---

# Future Extensions

These are **not required for the 10/24/2026 research version**.

Potential future work:

- xG / xGA
- Player availability and injuries
- Player-level rate statistics
- Random Forest
- XGBoost
- Additional leagues
- More advanced calibration
- Player performance forecasting
- Live or pre-match prediction updates

Complex ML models will only be added if they provide a useful research comparison.

---

# Streamlit App

The project also includes a Streamlit interface for exploring:

- Historical predictions
- Upcoming fixtures
- Hypothetical matchups
- Team squads
- Model comparisons
- Confidence intervals
- Prediction history

The application is intended as an interface for the research system rather than the primary focus of the project.

---

# Goals

- Build a reproducible soccer forecasting pipeline
- Apply statistical modeling to a real-world forecasting problem
- Test hypotheses using historical data
- Avoid data leakage and look-ahead bias
- Evaluate models using unseen data
- Understand which information provides predictive value
- Investigate derby/rivalry forecasting
- Document both successful and unsuccessful experiments
- Keep the project free and reproducible

---

# Status

**In Development**

The core forecasting models and chronological backtesting system are implemented.

The current phase focuses on feature research, out-of-sample evaluation, derby analysis, and robustness testing.

**Target research completion: 10/24/2026**
