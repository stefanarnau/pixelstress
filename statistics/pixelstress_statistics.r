# Load data
path_in <- "/mnt/data_dump/pixelstress/4_results"
df <- read.csv(file.path(path_in, "stats_table.csv"))

# Load necessary packages
library(afex)      # Used for aov_car(), which performs the mixed ANOVA
library(emmeans)   # Used for post-hoc pairwise comparisons via emmeans()
library(dplyr)     # Used for data manipulation (filter, group_by, summarise, mutate, etc.)

# Set factors for analysis (ensures proper treatment in models)
df$group <- factor(df$group)
df$feedback <- factor(df$feedback)
df$stage <- factor(df$stage)
df$id <- factor(df$id)

# List of dependent variables (DVs) to iterate over
dvs <- c("rt", "accuracy", "cnv_Fz", "cnv_Cz", "mft_target_cross", "posterior_alpha_cti")

# Function to summarize means and standard deviations for each DV
summarize_dv <- function(data, dv) {
  data %>%
    group_by(group, stage, feedback) %>%
    summarise(
      mean = mean(.data[[dv]], na.rm = TRUE),
      sd = sd(.data[[dv]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(group, stage, feedback)
}

# Function to cleanly print ANOVA results from aov_car()
print_anova_results <- function(aov_table) {
  # Convert ANOVA table to a data frame and preserve row names as 'Effect'
  atab <- as.data.frame(aov_table)
  atab$Effect <- rownames(atab)
  colnames(atab) <- make.names(colnames(atab))  # Clean up column names to make them syntactic

  # Expected columns
  base_cols <- c("Effect", "num.Df", "den.Df", "F", "Pr..F.", "pes")

  # Optional corrections (like Greenhouse-Geisser epsilon)
  optional_cols <- c("GGe")
  available_cols <- intersect(optional_cols, colnames(atab))
  selected_cols <- c(base_cols, available_cols)

  # Safety check for missing columns
  missing_base <- setdiff(base_cols, colnames(atab))
  if (length(missing_base) > 0) {
    stop("Missing required columns: ", paste(missing_base, collapse = ", "))
  }

  # Select and rename columns for readable output
  atab_clean <- atab %>%
    select(all_of(selected_cols)) %>%
    rename(
      `df1` = num.Df,
      `df2` = den.Df,
      `F` = F,
      `p (GG)` = Pr..F.,
      `partial eta²` = pes
    ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 3)))  # Round all numeric values

  # Optionally rename GGe column if present
  if ("GGe" %in% colnames(atab_clean)) {
    atab_clean <- atab_clean %>%
      rename(`GG epsilon` = GGe)
  }

  print(atab_clean, row.names = FALSE)
}

# Perform post-hoc tests for the 'feedback' factor if it has a significant main effect
perform_feedback_posthoc <- function(aov_result, factor_name = "feedback", adjust_method = "bonferroni") {
  # Extract p-value from ANOVA table
  p_val <- aov_result$anova_table[factor_name, "Pr(>F)"]

  # If significant, perform emmeans pairwise comparisons
  if (!is.na(p_val) && p_val < 0.05) {
    cat(paste0("\nPost hoc comparisons for ", factor_name, " (significant main effect):\n"))
    emms <- emmeans(aov_result, reformulate(factor_name))
    print(pairs(emms, adjust = adjust_method))
  } else {
    cat(paste0("\nNo significant main effect of ", factor_name, " — skipping post hoc tests.\n"))
  }
}

# Check for significant 2-way interactions and run follow-up ANOVAs within subsets
check_2way_interactions <- function(aov_result, df, dv) {
  for (int_term in c("group:stage", "group:feedback", "stage:feedback")) {
    pval <- aov_result$anova_table[int_term, "Pr(>F)"]

    if (!is.na(pval) && pval < 0.05) {
      cat(paste("\nSignificant 2-way interaction:", int_term, "\n"))

      # Subset by group if group-related interaction
      if (int_term == "group:stage" || int_term == "group:feedback") {
        for (g in levels(df$group)) {
          cat("\n-- Subset: group =", g, "--\n")
          df_group <- df %>% filter(group == g)
          aov_sub <- aov_car(
            as.formula(paste(dv, "~ stage * feedback + Error(id/(stage*feedback))")),
            data = df_group,
            anova_table = list(correction = "GG", es = "pes")
          )
          print_anova_results(aov_sub$anova_table)

          if (int_term == "group:feedback") {
            perform_feedback_posthoc(aov_sub)
          }
        }

      # Subset by stage if interaction is stage:feedback
      } else if (int_term == "stage:feedback") {
        for (s in levels(df$stage)) {
          cat("\n-- Subset: stage =", s, "--\n")
          df_stage <- df %>% filter(stage == s)

          # Adjust formula depending on group levels present
          if (length(unique(df_stage$group)) > 1) {
            formula_str <- paste(dv, "~ group * feedback + Error(id/feedback)")
          } else {
            formula_str <- paste(dv, "~ feedback + Error(id/feedback)")
          }

          aov_sub <- aov_car(
            as.formula(formula_str),
            data = df_stage,
            anova_table = list(correction = "GG", es = "pes")
          )
          print_anova_results(aov_sub$anova_table)

          perform_feedback_posthoc(aov_sub)
        }
      }
    }
  }
}

# Main loop over each dependent variable
for (dv in dvs) {
  cat("\n========== ANALYSIS FOR:", dv, "==========\n")

  # Print group-wise summary stats
  summary_table <- summarize_dv(df, dv)
  print(summary_table)

  # Run 3-way mixed ANOVA
  formula_str <- as.formula(paste(dv, "~ group * stage * feedback + Error(id/(stage*feedback))"))
  aov_result <- aov_car(formula_str, data = df, anova_table = list(correction = "GG", es = "pes"))
  print_anova_results(aov_result$anova_table)

  # Post-hoc for feedback if main effect is significant
  perform_feedback_posthoc(aov_result)

  # Check for 3-way interaction and do follow-ups if needed
  interaction_p <- aov_result$anova_table["group:stage:feedback", "Pr(>F)"]
  if (!is.na(interaction_p) && interaction_p < 0.05) {
    cat("\nSignificant 3-way interaction: running follow-up ANOVAs by group\n")
    for (g in levels(df$group)) {
      cat("\n-- Subset: group =", g, "--\n")
      df_group <- df %>% filter(group == g)
      aov_sub <- aov_car(
        as.formula(paste(dv, "~ stage * feedback + Error(id/(stage*feedback))")),
        data = df_group,
        anova_table = list(correction = "GG", es = "pes")
      )
      print(aov_sub$anova_table)

      # NEW: Also test 2-way interactions within this subset
      check_2way_interactions(aov_sub, df_group, dv)
    }

  } else {
    # If no 3-way interaction, just check 2-way ones
    check_2way_interactions(aov_result, df, dv)
  }
}
