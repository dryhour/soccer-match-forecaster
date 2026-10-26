"""
Final figures (2026-10-24 freeze) for README.md.

Drawn only from CSVs the analysis scripts already wrote -- nothing is refit
here, so every figure matches the numbers in the tables. Each figure is
saved in a light and a dark version (README.md picks one per the reader's
GitHub theme).

Reads:
    data/gold/final_evaluation/          (python -m src.evaluation.final_evaluation)
    data/gold/derby_analysis/            (python -m src.evaluation.derby_analysis)
    data/gold/prediction_features/       (walk-forward CSVs from the two scripts above + the R script)
Writes:
    docs/figures/<name>-light.png, docs/figures/<name>-dark.png

Usage:
    python -m src.evaluation.figures
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Chart tokens: two validated categorical slots (blue, orange) plus a gray
# for context series, with surface/ink/grid per mode.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "s1": "#2a78d6", "s2": "#eb6834", "context": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "s1": "#3987e5", "s2": "#d95926", "context": "#898781"},
}
plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def comparison(comps: pd.DataFrame, analysis: str, challenger: str, baseline: str = None,
               metric: str = "brier_diff", flip: bool = True):
    """(mean, low, high) of one paired-bootstrap row, or None if absent.
    flip=True turns "challenger - baseline" (negative = better) into an
    improvement (positive = better)."""
    rows = comps[(comps["analysis"] == analysis) & (comps["challenger"] == challenger)
                 & (comps["metric"] == metric)]
    if baseline is not None:
        rows = rows[rows["baseline"] == baseline]
    if rows.empty:
        return None
    r = rows.iloc[0]
    return (-r["mean"], -r["ci_high"], -r["ci_low"]) if flip else (r["mean"], r["ci_low"], r["ci_high"])


def style_axes(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=t["muted"], labelsize=9)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def forest(ax, labels: list, series: list, t: dict, xlabel: str) -> None:
    """Dot + 95% CI per row. series: [(color, [(mean, low, high) or None per row])],
    rows listed top to bottom."""
    style_axes(ax, t)
    ax.grid(axis="y", visible=False)
    n = len(labels)
    offsets = np.linspace(0.18, -0.18, len(series)) if len(series) > 1 else [0.0]
    ax.axvline(0, color=t["axis"], linewidth=1.2, zorder=1)
    for (color, values), off in zip(series, offsets):
        for i, v in enumerate(values):
            if v is None:
                continue
            y = n - 1 - i + off
            ax.plot([v[1], v[2]], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(v[0], y, "o", markersize=7.5, color=color, markeredgecolor=t["surface"],
                    markeredgewidth=1.5, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], color=t["ink"], fontsize=9.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9)


def header(fig, t: dict, title: str, subtitle: str) -> None:
    fig.text(0.015, 0.975, title, ha="left", va="top", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.015, 0.915, subtitle, ha="left", va="top", fontsize=9.5, color=t["ink2"], linespacing=1.4)


def legend(fig, t: dict, entries: list, y: float, extra: list = (), ncol: int = None) -> None:
    """entries get a line + dot key; extra = [(color, label)] get a plain hairline key."""
    handles = [Line2D([], [], color=c, linewidth=2, marker="o", markersize=7, markeredgecolor=t["surface"])
               for c, _ in entries] + [Line2D([], [], color=c, linewidth=1.2) for c, _ in extra]
    entries = list(entries) + list(extra)
    fig.legend(handles, [label for _, label in entries], loc="upper left", bbox_to_anchor=(0.008, y),
               ncol=ncol or len(entries), frameon=False, fontsize=9, labelcolor=t["ink2"], handlelength=1.8,
               columnspacing=1.6)


def save(fig, out_dir: Path, name: str, mode: str) -> None:
    path = out_dir / f"{name}-{mode}.png"
    # No Software/date metadata, so re-running gives the same bytes.
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), metadata={"Software": None})
    plt.close(fig)


def fig_model_skill(comps: pd.DataFrame, t: dict):
    rows = [
        ("R Dixon-Coles (whole-history team strength)", "r_dixon_coles"),
        ("Python: venue-split team averages", "team_level_venue_split"),
        ("Python: 5-match form + squad quality", "squad_quality"),
        ("Python: 5-match form", "rolling_form_baseline"),
        ("Python: 5-match form, as recorded (α = 1.0)", "rolling_form_baseline (alpha=1.0)"),
    ]
    fig = plt.figure(figsize=(9.2, 4.6), facecolor=t["surface"])
    header(fig, t, "Every model beats the no-information base rate; long-horizon strength roughly doubles the edge",
           "Brier score improvement over the base rate (training-season H/D/A frequencies for every match), "
           "95% paired bootstrap CI.\nRight = better. Python rows use the validation-chosen penalty unless "
           "marked; R and the squad model skip some matches, so their n differs.")
    legend(fig, t, [(t["s1"], "2025-26 holdout (primary, n = 342-377)"),
                    (t["s2"], "2024-25 + 2025-26 nested walk-forward (n = 684-748)")], 0.82)
    ax = fig.add_axes([0.34, 0.13, 0.63, 0.6])
    forest(ax, [r[0] for r in rows], [
        (t["s1"], [comparison(comps, "holdout_vs_base_rate", key) for _, key in rows]),
        (t["s2"], [comparison(comps, "nested_wf_vs_base_rate", key) for _, key in rows]),
    ], t, "Brier score improvement over the base rate (higher = better)")
    return fig


def fig_feature_experiments(comps: pd.DataFrame, t: dict):
    form, squad_rows = "rolling_form_baseline", "rolling_form_baseline_squad_rows"
    rows = [
        ("Venue-split team averages (vs. 5-match form)", "team_level_venue_split", form),
        ("10-match form (vs. 5)", "rolling_form_last10", form),
        ("3-match form (vs. 5)", "rolling_form_last3", form),
        ("3 + 5 + 10-match form combined (vs. 5)", "rolling_form_combined_windows", form),
        ("Strength-of-schedule-adjusted form", "sos_adjusted_form", form),
        ("Squad quality (vs. 5-match form)", "squad_quality", squad_rows),
        ("Last season's team goals (control)", "prev_season_team_goals_control", squad_rows),
        ("Squad quality vs. the control", "squad_quality", "prev_season_team_goals_control"),
    ]
    fig = plt.figure(figsize=(9.2, 5.9), facecolor=t["surface"])
    header(fig, t, "Only venue-split team averages beat 5-match form in both test seasons",
           "Brier score improvement of each feature set over its own baseline, 95% paired bootstrap CI. "
           "Right = better.\nSquad-quality rows exist only for 2025-26 (no earlier season to validate on).")
    legend(fig, t, [(t["s1"], "2025-26 holdout, chosen penalty (primary)"),
                    (t["s2"], "2024-25 replication, chosen penalty"),
                    (t["context"], "2025-26, as recorded (α = 1.0)")], 0.835)
    ax = fig.add_axes([0.31, 0.1, 0.66, 0.68])
    forest(ax, [r[0] for r in rows], [
        (t["s1"], [comparison(comps, "holdout_tuned", c, b) for _, c, b in rows]),
        (t["s2"], [comparison(comps, "nested_wf_season_2425", c, b) for _, c, b in rows]),
        (t["context"], [comparison(comps, "holdout_recorded_alpha1", c, b) for _, c, b in rows]),
    ], t, "Brier score improvement over the baseline (higher = better)")
    return fig


def reliability(preds: pd.DataFrame, edges: np.ndarray, min_n: int = 15) -> pd.DataFrame:
    """One-vs-rest reliability over H/D/A: each match contributes three
    (forecast probability, did it happen) pairs, binned by probability."""
    p = np.concatenate([preds[c].to_numpy(float) for c in ("home_win_prob", "draw_prob", "away_win_prob")])
    y = np.concatenate([(preds["actual_result"] == o).to_numpy(float) for o in "HDA"])
    bins = pd.cut(p, edges, include_lowest=True)
    out = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin", observed=True) \
        .agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    out = out[out["n"] >= min_n].copy()
    out["se"] = np.sqrt(out["y"] * (1 - out["y"]) / out["n"])
    return out


def fig_calibration(series: list, t: dict):
    fig = plt.figure(figsize=(7.6, 6.6), facecolor=t["surface"])
    header(fig, t, "Forecasts are roughly calibrated; the α = 1.0 penalty squeezed them together",
           "Predicted probability vs. observed frequency, home/draw/away pooled, 2024-25 + 2025-26 "
           "walk-forward.\nBins of width 0.1 with at least 15 forecasts; bars are 95% binomial intervals.")
    legend(fig, t, [(color, label) for color, label, _ in series], 0.835,
           extra=[(t["ink2"], "perfectly calibrated")], ncol=2)
    ax = fig.add_axes([0.1, 0.09, 0.86, 0.63])
    style_axes(ax, t)
    ax.plot([0, 1], [0, 1], color=t["ink2"], linewidth=1, zorder=1)
    edges = np.linspace(0, 1, 11)
    for color, _, preds in series:
        rel = reliability(preds, edges)
        ax.vlines(rel["p"], rel["y"] - 1.96 * rel["se"], rel["y"] + 1.96 * rel["se"], color=color,
                  linewidth=1.2, alpha=0.6, zorder=2)
        ax.plot(rel["p"], rel["y"], color=color, linewidth=2, solid_joinstyle="round", zorder=3)
        ax.plot(rel["p"], rel["y"], "o", markersize=7, color=color, markeredgecolor=t["surface"],
                markeredgewidth=1.5, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability", color=t["ink2"], fontsize=9)
    ax.set_ylabel("Observed frequency", color=t["ink2"], fontsize=9)
    return fig


def fig_derby(derby: pd.DataFrame, t: dict):
    defs = [("Local derbies (primary, n = 144-146)", "local"), ("Classic three derbies (n = 18)", "classic"),
            ("Named rivalries (n = 40-42)", "rivalry")]
    models = [(t["s1"], "python_poisson_baseline"), (t["s2"], "r_dixon_coles")]

    def value(model, definition, metric):
        r = derby[(derby["model"] == model) & (derby["definition"] == definition) & (derby["metric"] == metric)
                  & (derby["sample"] == "pooled") & (derby["bootstrap"] == "match")
                  & derby["analysis"].isin(["primary", "robustness"])].iloc[0]
        return r["diff"], r["ci_low"], r["ci_high"]

    fig = plt.figure(figsize=(9.6, 4.8), facecolor=t["surface"])
    header(fig, t, "No evidence derbies are harder to forecast, but only large effects were detectable",
           "Derby minus non-derby, pooled walk-forward 2023-24 to 2025-26, 95% bootstrap CI. Right = worse on "
           "derbies.\nMinimum detectable Brier difference for local derbies: 0.050 (Python), 0.084 (R).")
    legend(fig, t, [(t["s1"], "Python 5-match form (α = 1.0)"), (t["s2"], "R Dixon-Coles")], 0.815)
    for i, (metric, title) in enumerate([("brier", "Brier score"),
                                         ("excess_brier", "Excess Brier (observed − model-expected)")]):
        ax = fig.add_axes([0.25 + i * 0.38, 0.12, 0.34, 0.54])
        forest(ax, [d[0] for d in defs],
               [(color, [value(model, d, metric) for _, d in defs]) for color, model in models],
               t, "derby − non-derby (higher = worse on derbies)")
        ax.set_title(title, color=t["ink"], fontsize=10, loc="left")
        ax.set_xlim(-0.32, 0.2)
        if i:
            ax.set_yticklabels([])
    return fig


def main():
    cfg = load_config()
    final_dir = PROJECT_ROOT / cfg["paths"]["gold_final_evaluation"]
    derby_dir = PROJECT_ROOT / cfg["paths"]["gold_derby_analysis"]
    pred_dir = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"]
    out_dir = PROJECT_ROOT / cfg["paths"]["figures_dir"]
    test_seasons = cfg["final_evaluation"]["nested_walk_forward_test_seasons"]

    paths = {"comparisons": final_dir / "final_comparisons.csv",
             "nested": final_dir / "nested_walkforward_predictions.csv",
             "derby": derby_dir / "derby_comparisons.csv",
             "r_wf": pred_dir / "dixon_coles_walkforward.csv",
             "py_wf_alpha1": pred_dir / "poisson_walkforward_rolling_form_baseline.csv"}
    missing = [p.relative_to(PROJECT_ROOT) for p in paths.values() if not p.exists()]
    if missing:
        print(f"Missing {', '.join(map(str, missing))}. Run the R script, derby_analysis and final_evaluation first.")
        return

    comps = pd.read_csv(paths["comparisons"])
    nested = pd.read_csv(paths["nested"])
    derby = pd.read_csv(paths["derby"])
    r_wf = pd.read_csv(paths["r_wf"])
    py_alpha1 = pd.read_csv(paths["py_wf_alpha1"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode, t in THEMES.items():
        save(fig_model_skill(comps, t), out_dir, "model_skill", mode)
        save(fig_feature_experiments(comps, t), out_dir, "feature_experiments", mode)
        save(fig_calibration([
            (t["s1"], "Python venue-split (chosen penalty)", nested[nested["model"] == "team_level_venue_split"]),
            (t["s2"], "R Dixon-Coles", r_wf[r_wf["season"].isin(test_seasons)]),
            (t["context"], "Python 5-match form (α = 1.0)", py_alpha1[py_alpha1["season"].isin(test_seasons)]),
        ], t), out_dir, "calibration", mode)
        save(fig_derby(derby, t), out_dir, "derby", mode)
    print(f"Saved 4 figures x 2 themes -> {out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
