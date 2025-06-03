# Load required packages
library(ez)
library(car)

# Set path
path_in <- "/mnt/data_dump/pixelstress/4_results"

# Read CSV file
data <- read.csv(file.path(path_in, "stats_table.csv"))

# === RT =============================================================

# Perform mixed ANOVA
anova_results_rt <- ezANOVA(
  data = data,
  dv = rt,  # Dependent variable
  wid = id,  # Replace with your subject ID column name
  between = group,  # Between-subjects factor
  within = .(feedback, stage),  # Within-subjects factors
  type = 3,
  return_aov = TRUE,
  detailed = FALSE,
)

# Check ANOVA assumptions
print(anova_results_rt$`Mauchly's Test for Sphericity`)
print(anova_results_rt$`Levene's Test`)

# View ANOVA results
print(anova_results_rt)


# === Accuracy =============================================================

# Perform mixed ANOVA
anova_results_accuracy <- ezANOVA(
  data = data,
  dv = accuracy,  # Dependent variable
  wid = id,  # Replace with your subject ID column name
  between = group,  # Between-subjects factor
  within = .(feedback, stage),  # Within-subjects factors
  type = 3,
  return_aov = TRUE,
  detailed = FALSE,
)

# Check ANOVA assumptions
print(anova_results_accuracy$`Mauchly's Test for Sphericity`)
print(anova_results_accuracy$`Levene's Test`)

# View ANOVA results
print(anova_results_accuracy)

# === Accuracy =============================================================

# Perform mixed ANOVA
anova_results_accuracy <- ezANOVA(
  data = data,
  dv = accuracy,  # Dependent variable
  wid = id,  # Replace with your subject ID column name
  between = group,  # Between-subjects factor
  within = .(feedback, stage),  # Within-subjects factors
  type = 3,
  return_aov = TRUE,
  detailed = FALSE,
)

# Check ANOVA assumptions
print(anova_results_accuracy$`Mauchly's Test for Sphericity`)
print(anova_results_accuracy$`Levene's Test`)

# View ANOVA results
print(anova_results_accuracy)