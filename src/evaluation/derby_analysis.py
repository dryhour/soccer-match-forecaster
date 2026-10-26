"""
Derby / rivalry analysis (2026-10-17 milestone): are derby matches
systematically harder to forecast than ordinary matches?

The design below was fixed before any derby result was computed.

Sample. Walk-forward out-of-sample predictions: each season in
config.yaml -> evaluation.walk_forward_test_seasons is predicted by a model
trained only on the seasons before it. Pooling them gives ~3x the derbies of
the single holdout season, and every prediction is still made without having
seen the match. Two models: the Python rolling-form Poisson baseline (fit
here) and the R Dixon-Coles model (section 4 of
src/models/dixon_coles_model.R writes its walk-forward CSV -- run it first).

Derby definitions. config.yaml -> derbies. Primary: "local" (both clubs in
the same metropolitan area). Sensitivity: "classic" (the Manchester, North
London and Merseyside derbies) and "rivalry" (named rivalries, including
some between different cities).

"Harder to forecast" is split in two, because a model can score worse on
derbies for two different reasons:
  1. Raw loss -- is Brier / log loss higher on derbies? A yes could just
     mean derbies tend to be close matchups, which even a perfect model
     would find uncertain.
  2. Excess loss = observed loss minus the loss the model EXPECTS from its
     own probabilities (expected Brier = 1 - sum(p^2); expected log loss =
     entropy of p). It averages zero for a calibrated model however close
     the matchup is, so higher excess on derbies means the model misjudges
     them. raw difference = expected difference + excess difference, exactly.

Primary test: derby minus non-derby difference in raw and in excess Brier,
pooled sample, local definition, bootstrap 95% CI. A claim ("derbies are
harder" / "the models misjudge derbies") is made only if the interval
excludes zero for BOTH models. Everything else (log loss, accuracy,
calibration by outcome, home advantage, the other definitions, per season,
pair-cluster bootstrap, leave-one-pair-out, alpha=0.01) is secondary: this
script prints ~70 intervals, so a few will exclude zero by chance alone and
are read as patterns, not as separate findings.

Home advantage uses actual results for every season (nothing is predicted,
so nothing can leak). Every pairing plays both legs each season, so team
strength cancels out of each group's average home-minus-away goal difference.

Usage:
    Rscript src/models/dixon_coles_model.R      # writes the R walk-forward CSV
    python -m src.evaluation.derby_analysis
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.evaluation.backtest import per_match_brier, per_match_correct, run_poisson_backtest
from src.evaluation.evaluate_predictions import per_match_log_loss
from src.models.poisson_model import build_feature_lists

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

DEFINITIONS = ["local", "classic", "rivalry"]
PROB_COLS = ["home_win_prob", "draw_prob", "away_win_prob"]
# Minimum detectable difference at 80% power, two-sided 5% test: z_0.975 + z_0.80.
MDE_Z = 1.96 + 0.84
PRIMARY_METRICS = ["brier", "expected_brier", "excess_brier",
                   "log_loss", "expected_log_loss", "excess_log_loss", "correct"]
CALIBRATION_METRICS = ["resid_home_win", "resid_draw", "resid_away_win",
                       "resid_favorite_win", "resid_goal_diff"]
HOME_ADVANTAGE_METRICS = ["goal_diff", "home_win", "draw", "away_win"]


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def derby_pairs(derby_cfg: dict, definition: str) -> dict:
    """Maps each unordered derby pairing (a frozenset of two team names) to
    its name, for one definition in config.yaml -> derbies."""
    if definition == "local":
        return {frozenset(pair): f"{area} derby"
                for area, teams in derby_cfg["metro_areas"].items()
                for pair in itertools.combinations(teams, 2)}
    return {frozenset(teams): name for name, teams in derby_cfg[definition].items()}


def check_team_names(derby_cfg: dict, known_teams) -> None:
    """A misspelled team name in the config would silently drop that derby
    from every result, so fail loudly instead."""
    named = set()
    for teams in derby_cfg["metro_areas"].values():
        named.update(teams)
    for definition in ("classic", "rivalry"):
        for teams in derby_cfg[definition].values():
            named.update(teams)
    unknown = named - set(known_teams)
    if unknown:
        raise ValueError(f"config.yaml -> derbies names team(s) not in the match data: {sorted(unknown)}")


def label_derbies(df: pd.DataFrame, derby_cfg: dict) -> pd.DataFrame:
    """Adds an order-independent `pairing` key and one boolean
    `derby_<definition>` column per definition."""
    out = df.copy()
    pairs = [frozenset(p) for p in zip(out["home_team"], out["away_team"])]
    out["pairing"] = [" vs ".join(sorted(p)) for p in zip(out["home_team"], out["away_team"])]
    for definition in DEFINITIONS:
        lookup = derby_pairs(derby_cfg, definition)
        out[f"derby_{definition}"] = [p in lookup for p in pairs]
    return out


def walk_forward_poisson(df: pd.DataFrame, home_features: list, away_features: list,
                          test_seasons: list, max_goals: int, alpha: float = 1.0,
                          standardize: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Out-of-sample predictions for each test season from a model trained only
    on the seasons before it. Later seasons are dropped before each fit, so
    run_poisson_backtest (which trains on every season except the held-out
    one) never sees the future.
    """
    frames = []
    for season in test_seasons:
        fold = run_poisson_backtest(df[df["season"] <= season], home_features, away_features,
                                    season, max_goals, alpha=alpha, standardize=standardize,
                                    verbose=verbose)
        fold.insert(0, "season", season)
        frames.append(fold)
    return pd.concat(frames, ignore_index=True)


def per_match_scores(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-match loss, the loss the model expects from its own probabilities,
    and observed-minus-predicted calibration residuals."""
    out = preds.reset_index(drop=True).copy()
    probs = out[PROB_COLS].to_numpy(dtype=float)
    actual = out["actual_result"]

    out["brier"] = per_match_brier(out)
    out["log_loss"] = per_match_log_loss(out)
    out["correct"] = per_match_correct(out)
    # What the loss averages to if outcomes really occur with the model's
    # probabilities: sum_k p_k * loss(outcome k).
    out["expected_brier"] = 1.0 - (probs ** 2).sum(axis=1)
    out["expected_log_loss"] = -(probs * np.log(np.clip(probs, 1e-15, 1.0))).sum(axis=1)
    out["excess_brier"] = out["brier"] - out["expected_brier"]
    out["excess_log_loss"] = out["log_loss"] - out["expected_log_loss"]

    # Calibration-in-the-large: outcome indicator minus predicted probability.
    for outcome, col, name in zip("HDA", PROB_COLS, ["home_win", "draw", "away_win"]):
        out[f"resid_{name}"] = (actual == outcome).astype(float) - out[col]
    # Favorite = the side with the higher win probability. "The form book goes
    # out the window in derbies" predicts a negative residual here.
    home_fav = out["home_win_prob"] >= out["away_win_prob"]
    fav_prob = np.where(home_fav, out["home_win_prob"], out["away_win_prob"])
    fav_won = np.where(home_fav, actual == "H", actual == "A")
    out["resid_favorite_win"] = fav_won.astype(float) - fav_prob
    # Positive = the home side did better than the model expected.
    out["resid_goal_diff"] = ((out["actual_home_goals"] - out["actual_away_goals"])
                              - (out["expected_home_goals"] - out["expected_away_goals"]))
    return out


def home_advantage_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Per-match home-advantage measures from actual results (no model)."""
    out = matches.copy()
    out["goal_diff"] = out["home_goals"] - out["away_goals"]
    out["home_win"] = (out["result"] == "H").astype(float)
    out["draw"] = (out["result"] == "D").astype(float)
    out["away_win"] = (out["result"] == "A").astype(float)
    return out


def compare_groups(values, is_derby, clusters=None, n_boot: int = 10000, seed: int = 0) -> dict:
    """
    Derby minus non-derby difference in means, with a bootstrap 95% CI.
    The groups are disjoint sets of matches, so (unlike
    backtest.paired_bootstrap) each is resampled independently. With
    `clusters` (the pairing of each match), whole pairings are resampled
    instead of single matches -- more conservative if meetings of the same
    two teams are correlated. Also returns the minimum detectable
    difference (80% power, 5% two-sided) for the match-level design.
    """
    values = np.asarray(values, dtype=float)
    is_derby = np.asarray(is_derby, dtype=bool)
    derby, other = values[is_derby], values[~is_derby]
    if len(derby) < 2 or len(other) < 2:
        raise ValueError(f"Need at least 2 matches per group (got {len(derby)} derby, {len(other)} other).")

    rng = np.random.default_rng(seed)
    boot_means = {}
    for flag in (True, False):
        v = values[is_derby == flag]
        if clusters is None:
            idx = rng.integers(0, len(v), size=(n_boot, len(v)))
            boot_means[flag] = v[idx].mean(axis=1)
        else:
            codes, _ = pd.factorize(np.asarray(clusters)[is_derby == flag])
            sums = np.bincount(codes, weights=v)
            counts = np.bincount(codes)
            idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
            boot_means[flag] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.percentile(boot_means[True] - boot_means[False], [2.5, 97.5])

    return {
        "n_derby": len(derby),
        "n_other": len(other),
        "derby_mean": float(derby.mean()),
        "other_mean": float(other.mean()),
        "diff": float(derby.mean() - other.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
        "mde": float(MDE_Z * np.sqrt(derby.var(ddof=1) / len(derby) + other.var(ddof=1) / len(other))),
    }


def detection_power(diff: float, se: float) -> float:
    """Approximate power of a two-sided 5% test to detect a true difference
    `diff` when the estimate's standard error is `se` (normal approximation)."""
    z = abs(diff) / se
    return float(norm.cdf(z - 1.96) + norm.cdf(-z - 1.96))


def leave_one_pair_out(scores: pd.DataFrame, definition: str, metric: str) -> pd.DataFrame:
    """Derby minus non-derby difference recomputed with each derby pairing
    dropped in turn -- shows whether one pairing drives the result."""
    col = f"derby_{definition}"
    rows = []
    for pairing in sorted(scores.loc[scores[col], "pairing"].unique()):
        kept = scores[scores["pairing"] != pairing]
        rows.append({
            "dropped_pairing": pairing,
            "n_dropped": int((scores["pairing"] == pairing).sum()),
            "diff": float(kept.loc[kept[col], metric].mean() - kept.loc[~kept[col], metric].mean()),
        })
    return pd.DataFrame(rows)


def pairing_table(scores: pd.DataFrame, derby_cfg: dict) -> pd.DataFrame:
    """Descriptive per-rivalry and per-metro-area breakdown (too small for
    inference: a pairing has at most 2 meetings per season)."""
    groups = {name: frozenset(teams) for name, teams in derby_cfg["rivalry"].items()}
    local = derby_pairs(derby_cfg, "local")
    keys = [frozenset(p) for p in zip(scores["home_team"], scores["away_team"])]
    rows = []
    for label, mask in (
        [(name, [k == pair for k in keys]) for name, pair in groups.items()]
        + [(f"{area} (all local pairs)", [local.get(k) == f"{area} derby" for k in keys])
           for area in derby_cfg["metro_areas"]]
    ):
        sub = scores[np.asarray(mask, dtype=bool)]
        if sub.empty:
            continue
        rows.append({
            "group": label, "n": len(sub),
            "home_wins": int((sub["actual_result"] == "H").sum()),
            "draws": int((sub["actual_result"] == "D").sum()),
            "away_wins": int((sub["actual_result"] == "A").sum()),
            "mean_pred_home": round(sub["home_win_prob"].mean(), 3),
            "mean_pred_draw": round(sub["draw_prob"].mean(), 3),
            "mean_pred_away": round(sub["away_win_prob"].mean(), 3),
            "accuracy": round(sub["correct"].mean(), 3),
            "brier": round(sub["brier"].mean(), 4),
            "excess_brier": round(sub["excess_brier"].mean(), 4),
        })
    return pd.DataFrame(rows)


def run_comparison(rows: list, frame: pd.DataFrame, metric: str, definition: str, *,
                   analysis: str, model: str, sample: str, cluster: bool = False) -> dict:
    res = compare_groups(frame[metric], frame[f"derby_{definition}"],
                         clusters=frame["pairing"] if cluster else None)
    row = {"analysis": analysis, "model": model, "definition": definition, "sample": sample,
           "metric": metric, "bootstrap": "pair-cluster" if cluster else "match", **res}
    rows.append(row)
    return row


def print_header():
    print(f"    {'metric':<22}{'n_derby':>8}{'n_other':>8}{'derby':>9}{'other':>9}{'diff':>9}"
          f"   {'95% CI':<20}")


def print_row(row: dict, label: str = None):
    verdict = "excludes 0" if row["excludes_zero"] else ""
    print(f"    {label or row['metric']:<22}{row['n_derby']:>8}{row['n_other']:>8}"
          f"{row['derby_mean']:>9.4f}{row['other_mean']:>9.4f}{row['diff']:>+9.4f}"
          f"   [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  {verdict}")


def main():
    cfg = load_config()
    derby_cfg = cfg["derbies"]
    primary = derby_cfg["primary_definition"]
    test_seasons = cfg["evaluation"]["walk_forward_test_seasons"]
    windows = cfg["features"]["form_windows"]
    max_goals = cfg["model"]["max_goals_simulated"]
    alpha = cfg["model"]["poisson_alpha"]
    pred_dir = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"]
    out_dir = PROJECT_ROOT / cfg["paths"]["gold_derby_analysis"]
    match_feat_path = PROJECT_ROOT / cfg["paths"]["gold_match_features"] / "match_features.csv"
    matches_path = PROJECT_ROOT / cfg["paths"]["silver_matches"] / "matches.csv"
    r_path = pred_dir / "dixon_coles_walkforward.csv"

    for path, fix in [(match_feat_path, "python -m src.features.build_match_features"),
                      (matches_path, "python -m src.cleaning.clean_matches")]:
        if not path.exists():
            print(f"Missing {path.relative_to(PROJECT_ROOT)}. Run: {fix}")
            return

    matches = pd.read_csv(matches_path, parse_dates=["date"])
    check_team_names(derby_cfg, set(matches["home_team"]) | set(matches["away_team"]))
    match_features = pd.read_csv(match_feat_path, parse_dates=["date"])
    home_features, away_features = build_feature_lists(windows)

    # ---- Walk-forward predictions ------------------------------------------
    print(f"Walk-forward Python Poisson baseline (alpha={alpha}), test seasons {test_seasons}:")
    py_preds = walk_forward_poisson(match_features, home_features, away_features, test_seasons,
                                    max_goals, alpha=alpha)
    py_preds.to_csv(pred_dir / "poisson_walkforward_rolling_form_baseline.csv", index=False)
    print("Robustness refit at alpha=0.01 (diagnostic only; see PROGRESS.md item 15):")
    py_preds_low_alpha = walk_forward_poisson(match_features, home_features, away_features, test_seasons,
                                              max_goals, alpha=0.01)
    py_preds_low_alpha.to_csv(pred_dir / "poisson_walkforward_rolling_form_baseline_alpha0.01.csv", index=False)

    models = {"python_poisson_baseline": py_preds}
    if r_path.exists():
        models["r_dixon_coles"] = pd.read_csv(r_path)
    else:
        print(f"\n[warn] {r_path.relative_to(PROJECT_ROOT)} not found -- run "
              f"`Rscript src/models/dixon_coles_model.R` first. Continuing with the Python model only.")

    scores = {}
    for name, preds in models.items():
        s = label_derbies(per_match_scores(preds), derby_cfg)
        s["match_key"] = (pd.to_datetime(s["match_date"]).dt.strftime("%Y-%m-%d")
                          + "|" + s["home_team"] + "|" + s["away_team"])
        scores[name] = s
    low_alpha_scores = label_derbies(per_match_scores(py_preds_low_alpha), derby_cfg)
    ha = label_derbies(home_advantage_frame(matches), derby_cfg)

    rows = []

    print("\n=== 0. Derby sample sizes ===")
    print(f"    {'definition':<12}{'pairings':>9}" + "".join(f"{m:>26}" for m in scores)
          + f"{'all seasons, actual':>22}")
    for d in DEFINITIONS:
        n_pairs = len(derby_pairs(derby_cfg, d))
        print(f"    {d:<12}{n_pairs:>9}"
              + "".join(f"{int(s[f'derby_{d}'].sum()):>14} of {len(s):<9}" for s in scores.values())
              + f"{int(ha[f'derby_{d}'].sum()):>12} of {len(ha):<7}")

    print(f"\n=== 1. PRIMARY: forecast loss on derbies ({primary} definition, pooled walk-forward, "
          f"derby - non-derby) ===")
    print("    Brier/log loss: positive diff = worse on derbies. excess = observed - model-expected loss.")
    for model, s in scores.items():
        print(f"  {model}")
        print_header()
        for metric in PRIMARY_METRICS:
            row = run_comparison(rows, s, metric, primary, analysis="primary", model=model, sample="pooled")
            print_row(row)
        brier_row = next(r for r in rows if r["model"] == model and r["metric"] == "brier"
                         and r["analysis"] == "primary")
        excess_mde = next(r["mde"] for r in rows if r["model"] == model and r["metric"] == "excess_brier"
                          and r["analysis"] == "primary")
        print(f"    minimum detectable difference (80% power): Brier {brier_row['mde']:.4f}, "
              f"excess Brier {excess_mde:.4f}")
        # Benchmark for that number: a uniform 1/3 forecast scores Brier 2/3 on
        # every match, so "no skill at all on derbies" is a derby penalty of
        # (2/3 - the model's non-derby Brier).
        no_skill_gap = 2 / 3 - brier_row["other_mean"]
        print(f"    if the model had NO skill on derbies (uniform-guess Brier 0.6667, a penalty of "
              f"{no_skill_gap:+.4f}), power to detect it: "
              f"{detection_power(no_skill_gap, brier_row['mde'] / MDE_Z):.0%}")

    print(f"\n=== 2. Calibration by outcome ({primary}, pooled): mean(observed - predicted), derby - non-derby ===")
    print("    resid_draw > 0 = more draws than predicted; resid_favorite_win < 0 = favorites underperform;")
    print("    resid_goal_diff < 0 = home side did worse than the model expected (less home advantage).")
    for model, s in scores.items():
        print(f"  {model}")
        print_header()
        for metric in CALIBRATION_METRICS:
            print_row(run_comparison(rows, s, metric, primary, analysis="calibration", model=model,
                                     sample="pooled"))

    print(f"\n=== 3. Home advantage in actual results (all {len(ha)} matches, no model) ===")
    print("    goal_diff = home goals - away goals. Both legs of every pairing are in each group, "
          "so team strength cancels.")
    for d in DEFINITIONS:
        print(f"  definition: {d}")
        print_header()
        for metric in HOME_ADVANTAGE_METRICS:
            print_row(run_comparison(rows, ha, metric, d, analysis="home_advantage", model="actual_results",
                                     sample="all_seasons"))
    print(f"  definition: {primary}, pair-cluster bootstrap")
    print_header()
    for metric in HOME_ADVANTAGE_METRICS:
        print_row(run_comparison(rows, ha, metric, primary, analysis="home_advantage", model="actual_results",
                                 sample="all_seasons", cluster=True))

    print("\n=== 4. Robustness of the primary metrics ===")
    lopo = []
    for model, s in scores.items():
        print(f"  {model}")
        print_header()
        for metric in ("brier", "excess_brier"):
            for d in DEFINITIONS:
                if d != primary:
                    print_row(run_comparison(rows, s, metric, d, analysis="robustness", model=model,
                                             sample="pooled"), label=f"{metric} [{d}]")
            for season in test_seasons:
                print_row(run_comparison(rows, s[s["season"] == season], metric, primary,
                                         analysis="robustness", model=model, sample=f"season_{season}"),
                          label=f"{metric} [{season}]")
            print_row(run_comparison(rows, s, metric, primary, analysis="robustness", model=model,
                                     sample="pooled", cluster=True), label=f"{metric} [pair-cluster]")
            if len(scores) > 1:
                common = set.intersection(*(set(x["match_key"]) for x in scores.values()))
                print_row(run_comparison(rows, s[s["match_key"].isin(common)], metric, primary,
                                         analysis="robustness", model=model, sample="common_matches"),
                          label=f"{metric} [common n]")
            if model == "python_poisson_baseline":
                print_row(run_comparison(rows, low_alpha_scores, metric, primary, analysis="robustness",
                                         model="python_poisson_baseline_alpha0.01", sample="pooled"),
                          label=f"{metric} [alpha=0.01]")
            loo = leave_one_pair_out(s, primary, metric).assign(model=model, metric=metric)
            lopo.append(loo)
            most = loo.loc[(loo["diff"] - (s.loc[s[f"derby_{primary}"], metric].mean()
                                          - s.loc[~s[f"derby_{primary}"], metric].mean())).abs().idxmax()]
            print(f"    {metric + ' [drop 1 pair]':<22} diff ranges {loo['diff'].min():+.4f} to "
                  f"{loo['diff'].max():+.4f} over {len(loo)} pairings; most influential: "
                  f"{most['dropped_pairing']} -> {most['diff']:+.4f}")

    print("\n=== 5. Named rivalries and local areas (pooled walk-forward, descriptive only) ===")
    pairings = []
    for model, s in scores.items():
        table = pairing_table(s, derby_cfg)
        print(f"  {model}")
        print(table.to_string(index=False))
        pairings.append(table.assign(model=model))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(6).to_csv(out_dir / "derby_comparisons.csv", index=False)
    pd.concat(lopo, ignore_index=True).to_csv(out_dir / "derby_leave_one_pair_out.csv", index=False)
    pd.concat(pairings, ignore_index=True).to_csv(out_dir / "derby_pairings.csv", index=False)
    pd.concat([s.assign(model=m) for m, s in scores.items()], ignore_index=True) \
        .to_csv(out_dir / "derby_match_scores.csv", index=False)
    print(f"\nSaved comparisons, leave-one-pair-out, pairing tables and per-match scores -> "
          f"{out_dir.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
