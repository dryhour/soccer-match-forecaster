#!/usr/bin/env Rscript
#
# R alternative model: Dixon-Coles style Poisson regression.
#
# Unlike the Python baseline (a regression on each team's recent ROLLING
# FORM), this fits fixed TEAM-SPECIFIC attack/defense strength parameters
# via a Poisson GLM over the full match history -- the standard Dixon &
# Coles (1997) parameterization:
#
#   goals ~ home_advantage + attack(team) + defense(opponent)
#
# fit on a "long" reshape of the match data (two rows per match: one row
# for each side's scoring performance, so every team's goals-for AND
# goals-against contribute to its attack and defense coefficients
# respectively). This is a genuinely different modeling approach from the
# Python model, not a reimplementation -- useful as a real second opinion.
#
# Deliberately base-R only (no CRAN packages) so this stays as easy to run
# as `Rscript path/to/this/file.R` with a stock R install.
#
# Usage:
#   Rscript src/models/dixon_coles_model.R
#
# Reads:
#   data/silver/matches_clean/matches.csv
# Writes:
#   models/dixon_coles.rds
#   data/gold/prediction_features/dixon_coles_historical.csv  (backtest predictions)
#   data/gold/prediction_features/dixon_coles_matchups.csv    (every current-team pairing)

project_root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE))), "..", ".."))
matches_path <- file.path(project_root, "data", "silver", "matches_clean", "matches.csv")
models_dir <- file.path(project_root, "models")
gold_dir <- file.path(project_root, "data", "gold", "prediction_features")
max_goals <- 10

if (!file.exists(matches_path)) {
  stop(sprintf("Missing %s. Run: python -m src.cleaning.clean_matches", matches_path))
}
dir.create(models_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(gold_dir, showWarnings = FALSE, recursive = TRUE)

matches <- read.csv(matches_path, stringsAsFactors = FALSE)
matches$season <- as.character(matches$season)

cat(sprintf("Fitting Dixon-Coles model on %d matches...\n", nrow(matches)))

# ---- Reshape to long format: one row per team-performance ----------------
home_rows <- data.frame(
  team = matches$home_team, opponent = matches$away_team,
  goals = matches$home_goals, home = 1L
)
away_rows <- data.frame(
  team = matches$away_team, opponent = matches$home_team,
  goals = matches$away_goals, home = 0L
)
long_df <- rbind(home_rows, away_rows)
long_df$team <- factor(long_df$team)
long_df$opponent <- factor(long_df$opponent, levels = levels(long_df$team))

model <- glm(goals ~ home + team + opponent, family = poisson(), data = long_df)
saveRDS(model, file.path(models_dir, "dixon_coles.rds"))
cat(sprintf("Saved model -> %s\n", file.path("models", "dixon_coles.rds")))

# ---- Prediction helpers ---------------------------------------------------
predict_lambdas <- function(model, home_team, away_team) {
  lam_home <- predict(model, newdata = data.frame(team = home_team, opponent = away_team, home = 1L), type = "response")
  lam_away <- predict(model, newdata = data.frame(team = away_team, opponent = home_team, home = 0L), type = "response")
  c(lam_home = unname(lam_home), lam_away = unname(lam_away))
}

predict_match <- function(lam_home, lam_away, max_goals) {
  gh <- 0:max_goals
  ga <- 0:max_goals
  p_home <- dpois(gh, lam_home)
  p_away <- dpois(ga, lam_away)
  grid <- outer(p_home, p_away)  # [home_goals, away_goals]

  home_win <- sum(grid[lower.tri(grid)])
  draw <- sum(diag(grid))
  away_win <- sum(grid[upper.tri(grid)])

  best <- which(grid == max(grid), arr.ind = TRUE)[1, ]
  predicted_score <- sprintf("%d-%d", best[1] - 1, best[2] - 1)
  confidence <- max(grid)

  ci80_home <- qpois(c(0.10, 0.90), lam_home)
  ci80_away <- qpois(c(0.10, 0.90), lam_away)
  ci95_home <- qpois(c(0.025, 0.975), lam_home)
  ci95_away <- qpois(c(0.025, 0.975), lam_away)

  list(
    expected_home_goals = round(lam_home, 2), expected_away_goals = round(lam_away, 2),
    predicted_score = predicted_score,
    home_win_prob = round(home_win, 4), draw_prob = round(draw, 4), away_win_prob = round(away_win, 4),
    prediction_confidence = round(confidence, 4),
    ci80_home_low = ci80_home[1], ci80_home_high = ci80_home[2],
    ci80_away_low = ci80_away[1], ci80_away_high = ci80_away[2],
    ci95_home_low = ci95_home[1], ci95_home_high = ci95_home[2],
    ci95_away_low = ci95_away[1], ci95_away_high = ci95_away[2]
  )
}

# ---- 1) Backtest predictions on every historical match --------------------
cat("Generating backtest predictions on historical matches...\n")
hist_rows <- vector("list", nrow(matches))
for (i in seq_len(nrow(matches))) {
  m <- matches[i, ]
  lam <- predict_lambdas(model, m$home_team, m$away_team)
  pred <- predict_match(lam["lam_home"], lam["lam_away"], max_goals)
  hist_rows[[i]] <- data.frame(
    match_id = m$match_id, home_team = m$home_team, away_team = m$away_team,
    expected_home_goals = pred$expected_home_goals, expected_away_goals = pred$expected_away_goals,
    predicted_score = pred$predicted_score,
    home_win_prob = pred$home_win_prob, draw_prob = pred$draw_prob, away_win_prob = pred$away_win_prob,
    actual_home_goals = m$home_goals, actual_away_goals = m$away_goals, actual_result = m$result
  )
}
historical_out <- do.call(rbind, hist_rows)
write.csv(historical_out, file.path(gold_dir, "dixon_coles_historical.csv"), row.names = FALSE)
cat(sprintf("Saved %d backtest prediction(s) -> %s\n", nrow(historical_out),
            file.path("data", "gold", "prediction_features", "dixon_coles_historical.csv")))

# ---- 2) Every pairing of current-season teams (for the app) ---------------
latest_season <- max(matches$season)
current_teams <- sort(unique(c(
  matches$home_team[matches$season == latest_season],
  matches$away_team[matches$season == latest_season]
)))
cat(sprintf("Generating matchup predictions for %d current teams (%d pairings)...\n",
            length(current_teams), length(current_teams) * (length(current_teams) - 1)))

pair_rows <- list()
idx <- 1
for (home_team in current_teams) {
  for (away_team in current_teams) {
    if (home_team == away_team) next
    lam <- predict_lambdas(model, home_team, away_team)
    pred <- predict_match(lam["lam_home"], lam["lam_away"], max_goals)
    pair_rows[[idx]] <- data.frame(
      home_team = home_team, away_team = away_team,
      expected_home_goals = pred$expected_home_goals, expected_away_goals = pred$expected_away_goals,
      predicted_score = pred$predicted_score,
      home_win_prob = pred$home_win_prob, draw_prob = pred$draw_prob, away_win_prob = pred$away_win_prob,
      ci80_home_low = pred$ci80_home_low, ci80_home_high = pred$ci80_home_high,
      ci80_away_low = pred$ci80_away_low, ci80_away_high = pred$ci80_away_high,
      ci95_home_low = pred$ci95_home_low, ci95_home_high = pred$ci95_home_high,
      ci95_away_low = pred$ci95_away_low, ci95_away_high = pred$ci95_away_high
    )
    idx <- idx + 1
  }
}
matchups_out <- do.call(rbind, pair_rows)
write.csv(matchups_out, file.path(gold_dir, "dixon_coles_matchups.csv"), row.names = FALSE)
cat(sprintf("Saved %d matchup prediction(s) -> %s\n", nrow(matchups_out),
            file.path("data", "gold", "prediction_features", "dixon_coles_matchups.csv")))

cat("\nDone.\n")
