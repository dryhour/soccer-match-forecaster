"""
Head-to-head team lookup.

One search box lists every match -- past results and hypothetical
"upcoming" pairings of current teams -- as a single searchable dropdown.
Pick one to see a head-to-head view: each team's stats on its own side,
and in the middle either the real result (for a played match) or the
baseline Poisson model's prediction (for a hypothetical one), using the
same underlying logic as generate_predictions.py. Past matches also get
a "Dev View" showing what the model would have predicted beforehand,
plus an overall backtested accuracy summary at the top of the page.

Usage:
    streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from scipy.stats import poisson as poisson_dist

from src.evaluation.evaluate_predictions import compute_metrics
from src.models.poisson_model import PoissonMatchModel, build_feature_lists, predict_from_rows, predict_matchup

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

HOME_COLOR = "#2563eb"
HOME_BG = "#eff6ff"
AWAY_COLOR = "#ea580c"
AWAY_BG = "#fff7ed"
DRAW_COLOR = "#64748b"

PRIMARY_STATS = [
    ("season_avg_goals_for", "Avg Goals Scored"),
    ("season_avg_goals_against", "Avg Goals Conceded"),
]
SECONDARY_STATS = [
    ("avg_points_last5", "Form (pts/game, last 5)"),
    ("avg_goals_for_last5", "Scoring, last 5"),
]

MODEL_COLOR = "#2563eb"
R_MODEL_COLOR = "#059669"

MATCH_STAT_PAIRS = [
    ("Shots", "home_shots", "away_shots"),
    ("Shots on target", "home_shots_on_target", "away_shots_on_target"),
    ("Corners", "home_corners", "away_corners"),
    ("Yellow cards", "home_yellow_cards", "away_yellow_cards"),
    ("Red cards", "home_red_cards", "away_red_cards"),
]

CSS = f"""
<style>
.team-card {{
    border-radius: 14px;
    padding: 1.25rem 1.25rem 0.75rem 1.25rem;
    height: 100%;
}}
.team-card.home {{ background: {HOME_BG}; border-top: 6px solid {HOME_COLOR}; }}
.team-card.away {{ background: {AWAY_BG}; border-top: 6px solid {AWAY_COLOR}; }}
.side-badge {{
    display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em;
    padding: 0.15rem 0.6rem; border-radius: 999px; color: white; margin-bottom: 0.4rem;
}}
.side-badge.home {{ background: {HOME_COLOR}; }}
.side-badge.away {{ background: {AWAY_COLOR}; }}
.team-card h3 {{ margin: 0 0 0.75rem 0; font-size: 1.4rem; }}
.primary-stats {{ display: flex; gap: 1rem; margin-bottom: 1rem; }}
.primary-stat {{ flex: 1; }}
.primary-stat .value {{ font-size: 2.1rem; font-weight: 800; line-height: 1.1; }}
.primary-stat.home .value {{ color: {HOME_COLOR}; }}
.primary-stat.away .value {{ color: {AWAY_COLOR}; }}
.primary-stat .label {{ font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.03em; }}
.secondary-stats {{ border-top: 1px solid rgba(0,0,0,0.08); padding-top: 0.5rem; }}
.secondary-row {{
    display: flex; justify-content: space-between; font-size: 0.85rem;
    padding: 0.3rem 0; color: #334155;
}}
.secondary-row .label {{ color: #64748b; }}
.center-card {{ text-align: center; padding-top: 0.5rem; }}
.center-score {{ font-size: 3rem; font-weight: 800; margin: 0.25rem 0; }}
.center-outcome {{
    display: inline-block; padding: 0.2rem 0.9rem; border-radius: 999px;
    color: white; font-weight: 700; font-size: 0.85rem; margin-bottom: 0.5rem;
}}
</style>
"""


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_matches(cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["paths"]["silver_matches"] / "matches.csv"
    return pd.read_csv(path, parse_dates=["date"], dtype={"season": str})


@st.cache_data
def load_team_history(cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["paths"]["gold_team_features"] / "team_match_history.csv"
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_fixtures(cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["paths"]["silver_fixtures"] / "fixtures.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "home_team", "away_team"])
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_player_stats(cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["paths"]["silver_players"] / "player_stats.csv"
    if not path.exists():
        return pd.DataFrame(columns=["team", "season", "player", "position", "league_apps", "league_goals"])
    return pd.read_csv(path, dtype={"season": str})


@st.cache_resource
def load_model(cfg: dict) -> PoissonMatchModel:
    path = PROJECT_ROOT / cfg["paths"]["models_dir"] / "poisson_baseline.pkl"
    return PoissonMatchModel.load(path)


@st.cache_data
def load_dixon_coles_matchups(cfg: dict) -> pd.DataFrame:
    """Predictions for every current-team pairing from the R model (see
    src/models/dixon_coles_model.R) -- empty if that script hasn't been run."""
    path = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"] / "dixon_coles_matchups.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_dixon_coles_historical(cfg: dict) -> pd.DataFrame:
    path = PROJECT_ROOT / cfg["paths"]["gold_prediction_features"] / "dixon_coles_historical.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def dixon_coles_accuracy(cfg: dict) -> dict:
    historical = load_dixon_coles_historical(cfg)
    if historical.empty:
        return {}
    return compute_metrics(historical)


@st.cache_data
def backtest_accuracy(_model: PoissonMatchModel, cfg: dict) -> dict:
    """Model accuracy across every match with complete pre-match form. Each
    row already carries its own team's form as of BEFORE that match (no
    leakage) -- but the model was also trained on these same rows, so this
    is an in-sample check, not a true held-out backtest."""
    match_feat_path = PROJECT_ROOT / cfg["paths"]["gold_match_features"] / "match_features.csv"
    df = pd.read_csv(match_feat_path, parse_dates=["date"])
    home_features, away_features = build_feature_lists(cfg["features"]["form_windows"])
    usable = df.dropna(subset=home_features + away_features)

    records = []
    for _, row in usable.iterrows():
        pred = _model.predict_match(row)
        records.append({
            "home_win_prob": pred["home_win_prob"],
            "draw_prob": pred["draw_prob"],
            "away_win_prob": pred["away_win_prob"],
            "predicted_score": pred["predicted_score"],
            "expected_home_goals": pred["expected_home_goals"],
            "expected_away_goals": pred["expected_away_goals"],
            "actual_home_goals": row["home_goals"],
            "actual_away_goals": row["away_goals"],
            "actual_result": row["result"],
        })
    return compute_metrics(pd.DataFrame(records))


def season_label(code) -> str:
    code = str(code)
    if len(code) == 4 and code.isdigit():
        return f"20{code[:2]}/{code[2:]}"
    return code


def format_score(home_goals, away_goals) -> str:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return "-"
    return f"{int(home_goals)}:{int(away_goals)}"


def team_form_row(team_history: pd.DataFrame, team: str, match_id):
    if match_id is not None:
        rows = team_history[(team_history["team"] == team) & (team_history["match_id"] == match_id)]
        if not rows.empty:
            return rows.iloc[0]
    rows = team_history[team_history["team"] == team].sort_values("date")
    if rows.empty:
        return None
    return rows.iloc[-1]


@st.cache_data
def build_search_options(matches: pd.DataFrame, fixtures: pd.DataFrame, current_teams: list) -> dict:
    """key -> display label, for a single searchable dropdown covering past
    results, real scheduled fixtures (from football-data.co.uk's fixtures
    feed), and hypothetical pairings of current teams that aren't
    scheduled."""
    options = {}
    for row in matches.sort_values("date", ascending=False).itertuples():
        key = f"MATCH::{row.match_id}"
        options[key] = f"{row.home_team} vs {row.away_team} — {row.date.date()} — {format_score(row.home_goals, row.away_goals)}"

    scheduled_pairs = set()
    for row in fixtures.sort_values("date").itertuples():
        key = f"FIXTURE::{row.home_team}::{row.away_team}::{row.date.date()}"
        options[key] = f"{row.home_team} vs {row.away_team} — {row.date.date()} — -"
        scheduled_pairs.add((row.home_team, row.away_team))

    for home in current_teams:
        for away in current_teams:
            if home == away or (home, away) in scheduled_pairs:
                continue
            key = f"HYPOTHETICAL::{home}::{away}"
            options[key] = f"{home} vs {away} — Hypothetical — -"

    return options


def render_team_card(column, team: str, form_row, side: str):
    with column:
        badge = side.upper()
        if form_row is None:
            st.markdown(
                f"<div class='team-card {side}'><span class='side-badge {side}'>{badge}</span>"
                f"<h3>{team}</h3><p>No history yet.</p></div>",
                unsafe_allow_html=True,
            )
            return

        primary_html = ""
        for col, label in PRIMARY_STATS:
            value = form_row.get(col)
            value_str = f"{value:.2f}" if pd.notna(value) else "—"
            primary_html += (
                f"<div class='primary-stat {side}'><div class='value'>{value_str}</div>"
                f"<div class='label'>{label}</div></div>"
            )

        secondary_html = ""
        for col, label in SECONDARY_STATS:
            value = form_row.get(col)
            value_str = f"{value:.2f}" if pd.notna(value) else "—"
            secondary_html += (
                f"<div class='secondary-row'><span class='label'>{label}</span><span>{value_str}</span></div>"
            )

        st.markdown(
            f"<div class='team-card {side}'>"
            f"<span class='side-badge {side}'>{badge}</span>"
            f"<h3>{team}</h3>"
            f"<div class='primary-stats'>{primary_html}</div>"
            f"<div class='secondary-stats'>{secondary_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def r_matchup_row(dc_matchups: pd.DataFrame, home: str, away: str):
    if dc_matchups.empty:
        return None
    rows = dc_matchups[(dc_matchups["home_team"] == home) & (dc_matchups["away_team"] == away)]
    return rows.iloc[0] if not rows.empty else None


def r_historical_row(dc_historical: pd.DataFrame, match_id: str):
    if dc_historical.empty:
        return None
    rows = dc_historical[dc_historical["match_id"] == match_id]
    return rows.iloc[0] if not rows.empty else None


def render_ci_charts(lam_home: float, lam_away: float, home_team: str, away_team: str, r_row):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, lam, team, color, r_lam in [
        (axes[0], lam_home, home_team, HOME_COLOR, r_row["expected_home_goals"] if r_row is not None else None),
        (axes[1], lam_away, away_team, AWAY_COLOR, r_row["expected_away_goals"] if r_row is not None else None),
    ]:
        k = np.arange(0, 8)
        pmf = poisson_dist.pmf(k, lam)
        ci95_low, ci95_high = poisson_dist.interval(0.95, lam)
        ci80_low, ci80_high = poisson_dist.interval(0.80, lam)
        bar_colors = [color if ci95_low <= x <= ci95_high else "#e2e8f0" for x in k]
        ax.bar(k, pmf, color=bar_colors, width=0.7, zorder=2)
        ax.axvspan(ci80_low - 0.5, ci80_high + 0.5, color=color, alpha=0.18, zorder=1)
        if r_lam is not None:
            ax.axvline(r_lam, color=R_MODEL_COLOR, linestyle="--", linewidth=1.5, zorder=3)
        ax.set_title(f"{team} — expected {lam:.2f}", fontsize=10)
        ax.set_xlabel("Goals")
        ax.set_xticks(k)
        ax.set_ylabel("Probability")
        ax.set_xlim(-0.5, 7.5)

    fig.tight_layout()
    st.pyplot(fig)
    caption = (
        "Shaded band = 80% confidence interval; bars outside the lighter gray fall outside the 95% "
        "interval (both from the Python model's Poisson distribution)."
    )
    if r_row is not None:
        caption += " Dashed green line = R model's expected goals."
    st.caption(caption)


def render_squad(column, player_stats: pd.DataFrame, team: str, season: str, side: str):
    with column:
        squad = player_stats[(player_stats["team"] == team) & (player_stats["season"] == season)]
        label = f"👥 Squad — {season_label(season)} appearances & goals"
        with st.expander(label):
            if squad.empty:
                st.write(
                    "No player data for this team/season yet. Run: "
                    "`python -m src.ingestion.wikipedia_player_stats` then "
                    "`python -m src.cleaning.clean_player_stats`."
                )
                return
            display = squad.sort_values("league_apps", ascending=False)[
                ["player", "position", "league_apps", "league_goals"]
            ].rename(columns={
                "player": "Player", "position": "Pos", "league_apps": "Apps", "league_goals": "Goals",
            })
            st.dataframe(display, hide_index=True, use_container_width=True, height=350)


def render_actual_result(column, match_row: pd.Series):
    with column:
        with st.container(border=True):
            outcome_color = {"Home Win": HOME_COLOR, "Away Win": AWAY_COLOR, "Draw": DRAW_COLOR}
            color = outcome_color.get(match_row["result_label"], DRAW_COLOR)
            st.markdown(
                f"<div class='center-card'>"
                f"<div class='center-score'>{int(match_row['home_goals'])} : {int(match_row['away_goals'])}</div>"
                f"<div class='center-outcome' style='background:{color};'>{match_row['result_label']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Played {match_row['date'].date()} · Referee: {match_row.get('referee', 'n/a')}")


def render_prediction(column, pred: dict, note: str):
    with column:
        with st.container(border=True):
            home_rounded = round(pred["expected_home_goals"])
            away_rounded = round(pred["expected_away_goals"])

            probs = {"Home Win": pred["home_win_prob"], "Draw": pred["draw_prob"], "Away Win": pred["away_win_prob"]}
            outcome_color = {"Home Win": HOME_COLOR, "Away Win": AWAY_COLOR, "Draw": DRAW_COLOR}
            predicted_outcome = max(probs, key=probs.get)

            st.markdown(
                f"<div class='center-card'>"
                f"<div class='center-score'>{home_rounded} : {away_rounded}</div>"
                f"<div class='center-outcome' style='background:{outcome_color[predicted_outcome]};'>"
                f"Predicted {predicted_outcome}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(note)
            st.write("Home win")
            st.progress(pred["home_win_prob"], text=f"{pred['home_win_prob']:.0%}")
            st.write("Draw")
            st.progress(pred["draw_prob"], text=f"{pred['draw_prob']:.0%}")
            st.write("Away win")
            st.progress(pred["away_win_prob"], text=f"{pred['away_win_prob']:.0%}")


def render_model_details(model: PoissonMatchModel, team_history: pd.DataFrame, dc_matchups: pd.DataFrame,
                          dc_historical: pd.DataFrame, home_team: str, away_team: str, home_form, away_form, match_row):
    with st.expander("🔬 Prediction details & model comparison"):
        if match_row is not None:
            st.markdown("**Match stats**")
            stat_table = pd.DataFrame(
                {
                    home_team: [match_row[h] for _, h, _ in MATCH_STAT_PAIRS],
                    away_team: [match_row[a] for _, _, a in MATCH_STAT_PAIRS],
                },
                index=[label for label, _, _ in MATCH_STAT_PAIRS],
            )
            st.table(stat_table)
            st.divider()

            st.markdown("**Python vs R: what each model predicted beforehand**")
            if home_form is None or away_form is None:
                st.write("Not enough pre-match history to compute a Python prediction for this fixture.")
            else:
                py_pred = predict_from_rows(model, home_form, away_form)
                py_probs = {"H": py_pred["home_win_prob"], "D": py_pred["draw_prob"], "A": py_pred["away_win_prob"]}
                py_result = max(py_probs, key=py_probs.get)
                py_correct = "✅" if py_result == match_row["result"] else "❌"
                st.markdown(
                    f"- **Python** (form entering this match): {py_pred['predicted_score'].replace('-', ' - ')} "
                    f"(H {py_pred['home_win_prob']:.0%} / D {py_pred['draw_prob']:.0%} / A {py_pred['away_win_prob']:.0%}) {py_correct}"
                )
            r_row = r_historical_row(dc_historical, match_row["match_id"])
            if r_row is not None:
                r_probs = {"H": r_row["home_win_prob"], "D": r_row["draw_prob"], "A": r_row["away_win_prob"]}
                r_result = max(r_probs, key=r_probs.get)
                r_correct = "✅" if r_result == match_row["result"] else "❌"
                st.markdown(
                    f"- **R Dixon-Coles** (team strength, full dataset): {r_row['predicted_score'].replace('-', ' - ')} "
                    f"(H {r_row['home_win_prob']:.0%} / D {r_row['draw_prob']:.0%} / A {r_row['away_win_prob']:.0%}) {r_correct}"
                )
            else:
                st.write("R model prediction not available -- run `Rscript src/models/dixon_coles_model.R`.")
            st.caption(
                "Neither is a true out-of-sample test: Python used only this team's pre-match rolling "
                "form, but was trained on this match; R was trained on the full match history including "
                "this game. Treat both as sanity checks, not held-out accuracy."
            )
        else:
            py_pred = predict_matchup(model, team_history, home_team, away_team)
            r_row = r_matchup_row(dc_matchups, home_team, away_team)

            st.markdown("**Prediction distribution (Python model)**")
            render_ci_charts(py_pred["expected_home_goals"], py_pred["expected_away_goals"], home_team, away_team, r_row)
            st.divider()

            st.markdown("**Python vs R comparison**")
            comparison = {
                "Expected goals": {
                    "Python": f"{py_pred['expected_home_goals']} — {py_pred['expected_away_goals']}",
                    "R Dixon-Coles": f"{r_row['expected_home_goals']} — {r_row['expected_away_goals']}" if r_row is not None else "—",
                },
                "Predicted score": {
                    "Python": py_pred["predicted_score"].replace("-", " - "),
                    "R Dixon-Coles": r_row["predicted_score"].replace("-", " - ") if r_row is not None else "—",
                },
                "Home / Draw / Away": {
                    "Python": f"{py_pred['home_win_prob']:.0%} / {py_pred['draw_prob']:.0%} / {py_pred['away_win_prob']:.0%}",
                    "R Dixon-Coles": (
                        f"{r_row['home_win_prob']:.0%} / {r_row['draw_prob']:.0%} / {r_row['away_win_prob']:.0%}"
                        if r_row is not None else "—"
                    ),
                },
            }
            st.dataframe(pd.DataFrame(comparison).T, use_container_width=True)
            if r_row is None:
                st.caption("R model prediction not available -- run `Rscript src/models/dixon_coles_model.R`.")
            else:
                st.caption(
                    "Python uses each team's recent rolling form; R uses fixed team-strength ratings fit "
                    "on the full match history -- different philosophies, so disagreement between them is "
                    "expected, not a bug."
                )


def render_accuracy_expander(cfg: dict, model: PoissonMatchModel):
    py_metrics = backtest_accuracy(model, cfg)
    r_metrics = dixon_coles_accuracy(cfg)

    with st.expander("📈 Model accuracy (Python vs R)"):
        rows = {
            "Winner accuracy": ("winner_accuracy", "{:.0%}"),
            "Exact score accuracy": ("exact_score_accuracy", "{:.0%}"),
            "Brier score (lower = better)": ("brier_score", "{:.3f}"),
            "Matches evaluated": ("n_evaluated", "{:,}"),
        }
        table = {
            label: {
                "Python (rolling form)": fmt.format(py_metrics[key]),
                "R Dixon-Coles (team strength)": fmt.format(r_metrics[key]) if r_metrics else "—",
            }
            for label, (key, fmt) in rows.items()
        }
        st.dataframe(pd.DataFrame(table).T, use_container_width=True)
        st.caption(
            "Both evaluated in-sample (each trained on the data it's scored against), not a true "
            "held-out backtest -- and the R model is even more optimistic here, since it's fit on the "
            "FULL match history rather than each match's own pre-match form only, so it effectively "
            "already 'knows' each team's future results too. Treat this as a rough comparison of "
            "modeling approaches, not a real accuracy claim."
        )


def main():
    st.set_page_config(page_title="Soccer Match Forecaster", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Head-to-Head Lookup")

    cfg = load_config()
    matches = load_matches(cfg)
    team_history = load_team_history(cfg)
    player_stats = load_player_stats(cfg)
    fixtures = load_fixtures(cfg)
    model = load_model(cfg)
    dc_matchups = load_dixon_coles_matchups(cfg)
    dc_historical = load_dixon_coles_historical(cfg)

    render_accuracy_expander(cfg, model)
    st.divider()

    season_codes = sorted(matches["season"].unique())
    current_teams = sorted(
        set(matches.loc[matches["season"] == season_codes[-1], "home_team"])
        | set(matches.loc[matches["season"] == season_codes[-1], "away_team"])
    )
    options = build_search_options(matches, fixtures, current_teams)
    keys = list(options.keys())
    # Default to the soonest real scheduled fixture if there is one, else the
    # most recent played match.
    fixture_keys = [k for k in keys if k.startswith("FIXTURE::")]
    default_key = fixture_keys[0] if fixture_keys else keys[0]

    selected_key = st.selectbox(
        "Search matches (upcoming or past)",
        keys,
        index=keys.index(default_key),
        format_func=lambda k: options[k],
    )

    fixture_date = None
    if selected_key.startswith("MATCH::"):
        match_id = selected_key.split("::", 1)[1]
        match_row = matches[matches["match_id"] == match_id].iloc[0]
        home_team, away_team = match_row["home_team"], match_row["away_team"]
    elif selected_key.startswith("FIXTURE::"):
        _, home_team, away_team, fixture_date = selected_key.split("::")
        match_row = None
    else:  # HYPOTHETICAL::
        _, home_team, away_team = selected_key.split("::")
        match_row = None

    match_id = match_row["match_id"] if match_row is not None else None
    home_form = team_form_row(team_history, home_team, match_id)
    away_form = team_form_row(team_history, away_team, match_id)
    squad_season = match_row["season"] if match_row is not None else season_codes[-1]

    st.subheader(f"{home_team} vs {away_team}")
    col_home_stats, col_center, col_away_stats = st.columns([1, 1.1, 1])
    render_team_card(col_home_stats, home_team, home_form, "home")

    if match_row is not None:
        render_actual_result(col_center, match_row)
    else:
        if fixture_date is not None:
            note = f"Scheduled for {fixture_date} -- not yet played. Prediction uses each team's most recent form."
        else:
            note = "Hypothetical matchup (not on the schedule) -- prediction uses each team's most recent form."
        py_pred = predict_matchup(model, team_history, home_team, away_team)
        render_prediction(col_center, py_pred, note)

    render_team_card(col_away_stats, away_team, away_form, "away")

    render_model_details(model, team_history, dc_matchups, dc_historical,
                          home_team, away_team, home_form, away_form, match_row)

    col_home_squad, _, col_away_squad = st.columns([1, 1.1, 1])
    render_squad(col_home_squad, player_stats, home_team, squad_season, "home")
    render_squad(col_away_squad, player_stats, away_team, squad_season, "away")


if __name__ == "__main__":
    main()
