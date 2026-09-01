# Soccer Match Forecasting

A data-driven soccer forecasting system that predicts **match outcomes, scores, and player performances** using historical data, team statistics, player statistics, squad availability, and matchup-specific factors.

## Overview

The project uses a **Bronze / Silver / Gold data architecture** to transform raw soccer data into forecasting-ready features.

```text
Bronze
  ↓
Silver
  ↓
Gold
  ↓
Forecasting Model
  ↓
Prediction
  ↓
Evaluation
```

The goal is to build a forecasting system that is **accurate, explainable, reproducible, and continuously improvable**.

## Features

* Match score predictions
* Win / draw / loss probabilities
* Expected goals
* Prediction confidence
* Player performance predictions
* Player ratings
* Goal and assist probabilities
* Team form analysis
* Player form analysis
* Home / away performance
* Head-to-head analysis
* Tactical matchup analysis
* Injury and squad analysis
* Prediction history
* Model accuracy tracking

## Data Architecture

### Bronze

Raw data collected from external sources.

```text
Raw Matches
Raw Players
Raw Teams
Raw Lineups
Raw Injuries
Raw League Data
```

### Silver

Cleaned and standardized data.

```text
Cleaned Matches
Cleaned Players
Cleaned Teams
Cleaned Lineups
Cleaned Injuries
```

### Gold

Model-ready datasets containing engineered features.

```text
Match Features
Team Features
Player Features
Prediction Features
```

## Forecasting

The project will experiment with statistical and machine learning models such as:

* Poisson Regression
* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Neural Networks
* Ensemble Models

Models will be evaluated against historical results to determine what approaches perform best.

Alongside the Python baseline, `src/models/dixon_coles_model.R` fits a
genuinely different model in R -- team-specific attack/defense strength
parameters, the classic Dixon-Coles GLM parameterization -- rather than a
port of the Python approach. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md)
for details. Requires R (`brew install r` on macOS); run with:

```bash
Rscript src/models/dixon_coles_model.R
```

## Prediction Tracking

Predictions will be saved and compared with actual results.

Metrics will include:

* Match outcome accuracy
* Exact score accuracy
* Goal prediction error
* Player prediction accuracy
* Probability calibration
* Accuracy by league
* Accuracy by team
* Performance over time

## Methodology

Predictions may consider:

* Team strength
* Recent form
* xG / xGA
* Home advantage
* Player performance
* Injuries
* Suspensions
* Expected lineups
* Tactical matchups
* Opponent strength
* Historical performance

The methodology will be documented so predictions can be understood and reproduced.

## Goals

* Build an accurate soccer forecasting model
* Understand which factors influence predictions
* Track performance over time
* Compare different models
* Make predictions explainable
* Automate data updates
* Keep the project free
* Make the system reproducible

## Status

**In Development**

The data pipeline, features, models, and forecasting methodology will evolve as the project is tested and evaluated.
