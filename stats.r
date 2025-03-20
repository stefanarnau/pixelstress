# Load required packages
library(ez)
library(car)

# Set path
path_in <- "/mnt/data_dump/pixelstress/3_st_data/"

# Read CSV file
data <- read.csv(file.path(path_in, "combined.csv"))

# Perform mixed ANOVA
anova_results <- ezANOVA(
  data = data,
  dv = cnv_Cz,  # Dependent variable
  wid = id,  # Replace with your subject ID column name
  between = group,  # Between-subjects factor
  within = .(stage, feedback),  # Within-subjects factors
  type = 3,
  return_aov = TRUE,
  detailed = FALSE,
)

# Check ANOVA assumptions
print(anova_results$`Mauchly's Test for Sphericity`)
print(anova_results$`Levene's Test`)

# View ANOVA results
print(anova_results)