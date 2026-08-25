"""
Run the full Bronze -> Silver -> Gold -> Model pipeline end to end.

This is the "one command to rebuild everything" entry point referenced in
the README. Each stage is idempotent -- safe to re-run any time new data
shows up.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys

STAGES = [
    ("Ingestion  (Bronze)", ["python", "-m", "src.ingestion.football_data_co_uk"]),
    ("Cleaning   (Silver)", ["python", "-m", "src.cleaning.clean_matches"]),
    ("Features   (Gold)", ["python", "-m", "src.features.build_match_features"]),
    ("Model      (train)", ["python", "-m", "src.models.poisson_model"]),
]


def main():
    for name, cmd in STAGES:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nStage '{name}' failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}\nPipeline complete.\n{'=' * 60}")
    print("Next steps:")
    print("  1. Create a fixtures.csv (date,home_team,away_team) for upcoming matches.")
    print("  2. python -m src.predictions.generate_predictions --fixtures fixtures.csv")
    print("  3. After matches are played, re-run ingestion + cleaning, then:")
    print("     python -m src.evaluation.evaluate_predictions")


if __name__ == "__main__":
    main()
