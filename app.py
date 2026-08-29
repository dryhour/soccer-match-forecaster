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

import pandas as pd
import streamlit as st
import yaml

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
    ("avg_goals_against_last5", "Defense, last 5"),
]

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


def render_dev_view(model: PoissonMatchModel, home_form, away_form, match_row: pd.Series):
    with st.expander("🛠️ Dev View — model's prediction vs. actual"):
        if home_form is None or away_form is None:
            st.write("Not enough pre-match history to compute a prediction for this fixture.")
            return
        pred = predict_from_rows(model, home_form, away_form)
        probs = {"H": pred["home_win_prob"], "D": pred["draw_prob"], "A": pred["away_win_prob"]}
        predicted_result = max(probs, key=probs.get)
        correct = predicted_result == match_row["result"]

        st.markdown(
            f"**Model predicted (using form entering this match):** "
            f"{pred['predicted_score'].replace('-', ' - ')}  "
            f"(H {pred['home_win_prob']:.0%} / D {pred['draw_prob']:.0%} / A {pred['away_win_prob']:.0%})"
        )
        st.markdown(
            f"**Actual result:** {int(match_row['home_goals'])} - {int(match_row['away_goals'])} "
            f"({match_row['result_label']})"
        )
        st.markdown("✅ Correct winner" if correct else "❌ Incorrect winner")
        st.caption(
            "Uses each team's rolling form as it stood right before this match, not hindsight -- "
            "but the model was also trained on this match, so treat this as a sanity check, not a "
            "true out-of-sample test."
        )


def render_actual_result(column, model: PoissonMatchModel, match_row: pd.Series,
                          home: str, away: str, home_form, away_form):
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
            st.divider()
            stat_table = pd.DataFrame(
                {
                    home: [match_row[h] for _, h, _ in MATCH_STAT_PAIRS],
                    away: [match_row[a] for _, _, a in MATCH_STAT_PAIRS],
                },
                index=[label for label, _, _ in MATCH_STAT_PAIRS],
            )
            st.table(stat_table)
        render_dev_view(model, home_form, away_form, match_row)


def render_prediction(column, model: PoissonMatchModel, team_history: pd.DataFrame, home: str, away: str, note: str):
    with column:
        with st.container(border=True):
            pred = predict_matchup(model, team_history, home, away)
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
            st.divider()
            st.markdown(
                f"Expected goals (unrounded): **{pred['expected_home_goals']}** — **{pred['expected_away_goals']}**"
            )
            st.write("Home win")
            st.progress(pred["home_win_prob"], text=f"{pred['home_win_prob']:.0%}")
            st.write("Draw")
            st.progress(pred["draw_prob"], text=f"{pred['draw_prob']:.0%}")
            st.write("Away win")
            st.progress(pred["away_win_prob"], text=f"{pred['away_win_prob']:.0%}")


def render_accuracy_bar(cfg: dict, model: PoissonMatchModel):
    metrics = backtest_accuracy(model, cfg)
    cols = st.columns(4)
    cols[0].metric("Winner accuracy", f"{metrics['winner_accuracy']:.0%}")
    cols[1].metric("Exact score accuracy", f"{metrics['exact_score_accuracy']:.0%}")
    cols[2].metric("Brier score (lower = better)", f"{metrics['brier_score']:.3f}")
    cols[3].metric("Matches evaluated", f"{metrics['n_evaluated']:,}")
    st.caption(
        "Baseline Poisson model, evaluated on every match with complete pre-match form. "
        "This is an in-sample check (the model trained on these same matches), not a held-out backtest."
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

    render_accuracy_bar(cfg, model)
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
        render_actual_result(col_center, model, match_row, home_team, away_team, home_form, away_form)
    elif fixture_date is not None:
        note = f"Scheduled for {fixture_date} -- not yet played. Prediction uses each team's most recent form."
        render_prediction(col_center, model, team_history, home_team, away_team, note)
    else:
        note = "Hypothetical matchup (not on the schedule) -- prediction uses each team's most recent form."
        render_prediction(col_center, model, team_history, home_team, away_team, note)

    render_team_card(col_away_stats, away_team, away_form, "away")

    col_home_squad, _, col_away_squad = st.columns([1, 1.1, 1])
    render_squad(col_home_squad, player_stats, home_team, squad_season, "home")
    render_squad(col_away_squad, player_stats, away_team, squad_season, "away")


if __name__ == "__main__":
    main()
