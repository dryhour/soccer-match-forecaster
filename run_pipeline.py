"""
Run the Bronze -> Silver -> Gold -> Model pipeline end to end.

Two modes:

    python run_pipeline.py              # refresh: download the latest data, rebuild, retrain
    python run_pipeline.py --reproduce  # rebuild every research result from the committed data

--reproduce never downloads anything. It starts from the Bronze files
committed to the repo (football-data.co.uk CSVs + Wikipedia HTML), so it
regenerates exactly the numbers, tables and figures in README.md and
docs/. Re-downloading can change Bronze (the source revises files), so
the default refresh mode is for live use, not for reproducing results.
--reproduce needs R (`Rscript` on PATH) for the Dixon-Coles model.

Each stage is idempotent -- safe to re-run any time.
"""

import argparse
import shutil
import subprocess
import sys

PY = sys.executable

REFRESH_STAGES = [
    ("Ingestion  (Bronze)", [PY, "-m", "src.ingestion.football_data_co_uk"]),
    ("Cleaning   (Silver)", [PY, "-m", "src.cleaning.clean_matches"]),
    ("Features   (Gold)", [PY, "-m", "src.features.build_match_features"]),
    ("Model      (train)", [PY, "-m", "src.models.poisson_model"]),
]

REPRODUCE_STAGES = [
    ("Cleaning: matches     (Silver)", [PY, "-m", "src.cleaning.clean_matches"]),
    ("Cleaning: players     (Silver)", [PY, "-m", "src.cleaning.clean_player_stats"]),
    ("Cleaning: fixtures    (Silver)", [PY, "-m", "src.cleaning.clean_fixtures"]),
    ("Features              (Gold)", [PY, "-m", "src.features.build_match_features"]),
    ("Model: app baseline   (train)", [PY, "-m", "src.models.poisson_model"]),
    ("Recorded experiments  (alpha=1.0 holdout backtest)", [PY, "-m", "src.evaluation.backtest"]),
    ("R Dixon-Coles         (fit, holdout, walk-forward)", ["Rscript", "src/models/dixon_coles_model.R"]),
    ("Derby analysis", [PY, "-m", "src.evaluation.derby_analysis"]),
    ("Final evaluation      (chosen penalty, holdout once)", [PY, "-m", "src.evaluation.final_evaluation"]),
    ("Figures", [PY, "-m", "src.evaluation.figures"]),
]


def main():
    parser = argparse.ArgumentParser(description="Run the forecasting pipeline.")
    parser.add_argument("--reproduce", action="store_true",
                        help="Rebuild every research result from the committed Bronze data (no download).")
    args = parser.parse_args()

    stages = REPRODUCE_STAGES if args.reproduce else REFRESH_STAGES
    if args.reproduce and shutil.which("Rscript") is None:
        print("--reproduce needs R for the Dixon-Coles model (Rscript not found on PATH). "
              "On macOS: brew install r")
        sys.exit(1)

    for name, cmd in stages:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStage '{name}' failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}\nPipeline complete.\n{'=' * 60}")
    if args.reproduce:
        print("Every table and figure in README.md / docs/ has been regenerated. "
              "`git status` should show no changes.")
        return
    print("Next steps:")
    print("  1. Create a fixtures.csv (date,home_team,away_team) for upcoming matches.")
    print("  2. python -m src.predictions.generate_predictions --fixtures fixtures.csv")
    print("  3. After matches are played, re-run ingestion + cleaning, then:")
    print("     python -m src.evaluation.evaluate_predictions")


if __name__ == "__main__":
    main()
