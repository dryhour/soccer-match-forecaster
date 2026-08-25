# About This Project

## Goal

Build a soccer forecasting system that doesn't just spit out a prediction,
but provides a complete, documented pipeline:

```
Bronze → Silver → Gold → Model → Prediction → Evaluation → Improvement
```

## Core questions the system should eventually answer

- Who is most likely to win?
- What is the predicted score?
- How confident is the prediction?
- Which players are expected to perform well?
- What factors are driving the prediction?
- How accurate has the model been historically?
- Which models and features perform best?

## Non-goals

- No paid data feeds or APIs — the project should always be runnable for
  free.
- No black-box-only models — every prediction should be explainable, even if
  that means favoring a slightly less accurate but interpretable model.
- No one-off scripts — everything should be re-runnable via
  `config/config.yaml` + the pipeline scripts, not manual notebook surgery.

## Roadmap (rough order)

1. **Framework (this stage)** — directory structure, config, ingestion of
   match results, cleaning, basic rolling-form features, baseline Poisson
   model, prediction logging, basic evaluation.
2. **More match-level signal** — xG (from a free source if available),
   head-to-head history, strength-of-schedule adjustment.
3. **Player layer** — player stats ingestion, player-level features (form,
   goal contributions, minutes), injuries/suspensions/lineups feeding into
   "expected starting XI" and squad-availability-adjusted team strength.
4. **Model comparison** — logistic regression / random forest / XGBoost
   challengers evaluated against the Poisson baseline on the same held-out
   splits; only promote a challenger if it wins on out-of-sample metrics.
5. **Explainability layer** — per-prediction factor breakdown (e.g. "Arsenal
   strong home record", "Chelsea missing 2 starting defenders").
6. **Automation** — scheduled pipeline runs (e.g. GitHub Actions) so new
   results/fixtures flow through Bronze → Silver → Gold → predictions
   without manual work.
7. **Dashboard** — league/team selectors, prediction history, model
   comparison view.

See `docs/METHODOLOGY.md` for the technical approach at the current stage.
