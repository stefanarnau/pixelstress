# Load required packages
library(afex)
library(emmeans)
library(dplyr)

# Set path
path_in <- "/mnt/data_dump/pixelstress/4_results"

# Read CSV file
df <- read.csv(file.path(path_in, "stats_table.csv"))

# Make sure factor variables are set correctly
df <- df %>%
  mutate(across(c(id, stage, feedback, group), as.factor))

# List of dependent variables
dvs <- c("rt", "accuracy", "cnv_Fz", "cnv_Cz")

# Run mixed ANOVAs
results <- lapply(dvs, function(dv) {
  aov_ez(
    id = "id",  # subject identifier
    dv = dv,
    data = df,
    within = c("stage", "feedback"),
    between = "group",
    type = 3
  )
})
names(results) <- dvs

# Automatically run post-hoc tests if feedback effects are significant
posthoc_results <- list()

for (dv in dvs) {
  cat("=== Results for", dv, "===\n")
  res <- results[[dv]]
  
  # Fix: make 'Effect' a real column
  atab <- res$anova_table %>%
    tibble::rownames_to_column(var = "Effect")
  
  # Find significant effects involving 'feedback'
  sig_feedback <- atab %>%
    filter(grepl("feedback", Effect)) %>%
    filter(`Pr(>F)` < 0.05)
  
  if (nrow(sig_feedback) > 0) {
    cat("→ Significant effect(s) involving 'feedback' found. Running post-hoc tests...\n")
    
    em <- emmeans(res, pairwise ~ feedback)
    posthoc_results[[dv]] <- em$contrasts
    
    print(em$contrasts)
  } else {
    cat("→ No significant effect involving 'feedback'.\n")
  }
  
  cat("\n")
}
