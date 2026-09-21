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

# ---- 3) Real held-out backtest: fit on train seasons only, score on the ----
#         most recent complete season, which the model never saw. Everything
#         above this point (the full-history model, its "backtest" over every
#         historical match, and the current-team matchups) is in-sample --
#         this section is the only true out-of-sample evaluation in the repo.
#         Keep holdout_season in sync with config.yaml -> evaluation.holdout_season.
holdout_season <- "2526"
cat(sprintf("\nFitting a train-only model for the season-%s holdout backtest...\n", holdout_season))

train_matches <- matches[matches$season != holdout_season, ]
test_matches <- matches[matches$season == holdout_season, ]

train_home_rows <- data.frame(team = train_matches$home_team, opponent = train_matches$away_team,
                               goals = train_matches$home_goals, home = 1L)
train_away_rows <- data.frame(team = train_matches$away_team, opponent = train_matches$home_team,
                               goals = train_matches$away_goals, home = 0L)
train_long <- rbind(train_home_rows, train_away_rows)
train_long$team <- factor(train_long$team)
train_long$opponent <- factor(train_long$opponent, levels = levels(train_long$team))

holdout_model <- glm(goals ~ home + team + opponent, family = poisson(), data = train_long)
known_teams <- levels(train_long$team)

# A team with no matches in the training seasons (e.g. newly promoted for the
# holdout season) has no fitted attack/defense coefficient -- the model has
# literally never seen it play, so those matches are skipped rather than
# guessed at. This is a real, expected limitation of the whole-history
# parameterization, not a bug -- see docs/METHODOLOGY.md.
skipped <- 0
holdout_rows <- list()
idx <- 1
for (i in seq_len(nrow(test_matches))) {
  m <- test_matches[i, ]
  if (!(m$home_team %in% known_teams) || !(m$away_team %in% known_teams)) {
    skipped <- skipped + 1
    next
  }
  lam <- predict_lambdas(holdout_model, m$home_team, m$away_team)
  pred <- predict_match(lam["lam_home"], lam["lam_away"], max_goals)
  holdout_rows[[idx]] <- data.frame(
    match_date = m$date, home_team = m$home_team, away_team = m$away_team,
    expected_home_goals = pred$expected_home_goals, expected_away_goals = pred$expected_away_goals,
    predicted_score = pred$predicted_score,
    home_win_prob = pred$home_win_prob, draw_prob = pred$draw_prob, away_win_prob = pred$away_win_prob,
    actual_home_goals = m$home_goals, actual_away_goals = m$away_goals, actual_result = m$result
  )
  idx <- idx + 1
}
holdout_out <- do.call(rbind, holdout_rows)
write.csv(holdout_out, file.path(gold_dir, "dixon_coles_holdout_backtest.csv"), row.names = FALSE)
cat(sprintf("Saved %d held-out prediction(s) (skipped %d match(es) involving a team never seen in training) -> %s\n",
            nrow(holdout_out), skipped,
            file.path("data", "gold", "prediction_features", "dixon_coles_holdout_backtest.csv")))

holdout_out$predicted_result <- apply(
  holdout_out[, c("home_win_prob", "draw_prob", "away_win_prob")], 1,
  function(p) c("H", "D", "A")[which.max(p)]
)
winner_acc <- mean(holdout_out$predicted_result == holdout_out$actual_result)
score_parts <- do.call(rbind, strsplit(holdout_out$predicted_score, "-"))
exact_acc <- mean(as.integer(score_parts[, 1]) == holdout_out$actual_home_goals &
                     as.integer(score_parts[, 2]) == holdout_out$actual_away_goals)
mae_home <- mean(abs(holdout_out$expected_home_goals - holdout_out$actual_home_goals))
mae_away <- mean(abs(holdout_out$expected_away_goals - holdout_out$actual_away_goals))
onehot <- t(sapply(holdout_out$actual_result, function(r) as.numeric(c("H", "D", "A") == r)))
probs <- as.matrix(holdout_out[, c("home_win_prob", "draw_prob", "away_win_prob")])
brier <- mean(rowSums((probs - onehot) ^ 2))

cat(sprintf("\nR Dixon-Coles -- held-out backtest (season %s never seen in training):\n", holdout_season))
cat(sprintf("  n_evaluated: %d\n", nrow(holdout_out)))
cat(sprintf("  winner_accuracy: %.4f\n", winner_acc))
cat(sprintf("  exact_score_accuracy: %.4f\n", exact_acc))
cat(sprintf("  mae_home_goals: %.3f\n", mae_home))
cat(sprintf("  mae_away_goals: %.3f\n", mae_away))
cat(sprintf("  brier_score: %.4f\n", brier))
cat("\nCompare against src/evaluation/backtest.py's output (same season, same schema) for a like-for-like model comparison.\n")

cat("\nDone.\n")
