#' Copyright 2022 Google LLC
#
#' Licensed under the Apache License, Version 2.0 (the "License");
#' you may not use this file except in compliance with the License.
#' You may obtain a copy of the License at
#
#'      http://www.apache.org/licenses/LICENSE-2.0
#
#' Unless required by applicable law or agreed to in writing, software
#' distributed under the License is distributed on an "AS IS" BASIS,
#' WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#' See the License for the specific language governing permissions and
#' limitations under the License.
#' Purpose: Runs survival analysis to generate Figure 3g from Cosentino et al.
#' Nature Genetics 2023.
#' Updated: 2022-02-09
#'
#' Note: The input TSV, specified by `ML_VAL_FNAME`, is assumed to contain The
#' following required columns:
#'   - `eid`: A unique individual-level identifier.
#'   - `ml_based_copd`: Raw ML-based COPD liability scores.
#'   - `ml_based_copd_std`: ML-based COPD liability scores per standard
#'     deviation (i.e., `ml_based_copd/std(ml_based_copd)`).
#'   - `death_date`: Date of death in YYYY-MM-DD, if available. Defaults to
#'     `CUTOFF_DATE` if not provided for an individual.
#'   - `start_date`: The date of assessment.
#'   - `age`: The individual's age.
#'   - `sex`: The individual's sex.
#' The `ML_VAL_FNAME` path must be set prior to running the script.
suppressMessages({
  library(data.table)
  library(dplyr)
  library(survival)
  library(ggplot2)
  library(cowplot)
  library(tibble)
})

# Filepaths.
ML_VAL_FNAME <- "~/validation.tsv"

# Analysis cutoff date: last date for which death_date is potentially available.
DATE_FORMAT <- "%Y-%m-%d"
CUTOFF_DATE <- as.Date("2018-02-12", format = DATE_FORMAT)

process_dataframe <- function(dataframe, cutoff_date) {
  # Add censoring at the cutoff date.
  dataframe$status <- 1 * (!is.na(dataframe$death_date))
  dataframe$stop_date <- dataframe$death_date
  dataframe$stop_date[is.na(dataframe$stop_date)] <- cutoff_date

  # Format data.
  dataframe <- dataframe %>%
    dplyr::mutate(
      ## Format dates.
      start_date = as.Date(start_date, format = DATE_FORMAT),
      time = as.numeric(difftime(stop_date, start_date, units = "days"))
    )
  return(dataframe)
}


val_dataframe <- data.table::fread(file = ML_VAL_FNAME)
val_dataframe <- process_dataframe(val_dataframe, CUTOFF_DATE)

print_fit <- function(fit) {
  # Generate summary objects
  coxph_summary_result <- summary(fit)

  # Check proportional hazards. Ideally, none of the p-values are significant.
  cox_zph_result <- cox.zph(fit, global = FALSE)

  # Pull relevant tables.
  coxph_summary_coefs_table <- data.table(
    coxph_summary_result$coefficients,
    keep.rownames = TRUE
  )
  coxph_summary_conf_table <- data.table(
    coxph_summary_result$conf.int,
    keep.rownames = TRUE
  )
  cox_zph_table <- data.table(cox_zph_result$table, keep.rownames = TRUE)

  # Merge relevant tables.
  merged_table <- merge(
    coxph_summary_coefs_table,
    coxph_summary_conf_table,
    by = "rn",
    all = TRUE
  )
  merged_table <- merge(merged_table, cox_zph_table, by = "rn", all = TRUE)
  colnames(merged_table)[1] <- "term"

  # Print sheets-pastable result.
  cat(capture.output(print(coxph_summary_result$call$formula)), "\n")
  cat(
    "dataset =",
    paste0(coxph_summary_result$call$data),
    "\t",
    "n =",
    paste0(coxph_summary_result$n),
    "\t",
    "n_events =",
    paste0(coxph_summary_result$nevent),
    "\n"
  )
  write.table(merged_table, sep = "\t", col.names = TRUE, row.names = FALSE)
  cat("\n")
  return()
}

# Surv(time, status) ~ ml_based_copd_std + age + sex
val_fit <- coxph(
  Surv(time, status) ~ ml_based_copd_std + age + sex,
  data = val_dataframe
)
print_fit(val_fit)


km_ml_val_dataframe <- copy(val_dataframe)

# -----------------------------------------------------------------------------
# Categorize risk.
# -----------------------------------------------------------------------------

# Number of risk bins.
n_bins <- 4
bin_probs <- seq(from = 0, to = n_bins, by = 1) / n_bins
breaks <- quantile(x = km_ml_val_dataframe$ml_based_copd, probs = bin_probs)

labs <- names(breaks)
labs <- labs[2:length(labs)]
labs <- gsub(pattern = "%", replacement = "", x = labs) # nolint
labs <- paste0("p", labs)

km_ml_val_dataframe$ml_based_copd_categorized <- cut(
  x = km_ml_val_dataframe$resnet18_copd,
  breaks = breaks,
  labels = labs,
  include.lowest = TRUE,
  ordered = TRUE
)

# Check categorization:
table(km_ml_val_dataframe$ml_based_copd_categorized)

# -----------------------------------------------------------------------------
# Fit KM curves.
# -----------------------------------------------------------------------------

# Set the time period over which to plot.
min_time <- 0
max_time <- 1200
n_eval <- 1e3
time_grid <- seq(from = min_time, to = max_time, length.out = n_eval)

# Fit the curves.
split_km_ml_val_dataframe <- split(
  km_ml_val_dataframe,
  km_ml_val_dataframe$ml_based_copd_categorized
)
km_curves <- lapply(split_km_ml_val_dataframe, function(x) {
  fit <- survival::survfit(Surv(time, status) ~ 1, data = x)

  # Prob and SE step functions.
  g <- stats::stepfun(x = fit$time, y = c(1, fit$surv))
  h <- stats::stepfun(x = fit$time, y = c(0, fit$std.err))

  # Plotting frame.
  out <- data.frame(
    group = unique(x$ml_based_copd_categorized),
    time = time_grid,
    prob = g(time_grid),
    se = h(time_grid)
  )

  # Point-wise confidence intervals.
  z <- stats::qnorm(0.975)
  out$lower <- pmax(0, out$prob - z * out$se)
  out$upper <- pmin(1, out$prob + z * out$se)
  return(out)
})
km_output_dataframe <- do.call(rbind, km_curves)

# -----------------------------------------------------------------------------
# Plot KM curves.
# -----------------------------------------------------------------------------

x_lab <- "Days"
y_lab <- "Survival Probability"

q <- ggplot2::ggplot(data = km_output_dataframe) +
  ggplot2::theme_bw() +
  ggplot2::theme(
    panel.grid.minor = element_blank(),
    legend.position = "top"
  ) +
  ggplot2::geom_step(
    aes(x = time, y = prob, color = group),
    size = 1
  ) +
  ggplot2::geom_ribbon(
    aes(x = time, ymin = lower, ymax = upper, fill = group),
    alpha = 0.1,
    show.legend = FALSE
  ) +
  ggplot2::scale_x_continuous(
    name = x_lab
  ) +
  ggplot2::scale_y_continuous(
    name = y_lab
  ) +
  ggplot2::scale_color_brewer(
    name = "Risk group",
    palette = "Spectral"
  )

cowplot::ggsave2(
  file = "ml_based_copd_km_curves_val.pdf",
  device = "pdf",
  width = 7.5,
  height = 3.5,
  units = "in",
  dpi = 480
)

write.table(
  km_output_dataframe,
  file = "ml_based_copd_km_data_val.tsv",
  quote = FALSE,
  sep = "\t",
  row.names = FALSE
)
