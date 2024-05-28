library(dplyr)
library(ggpubr)
library(rstatix)
library(datarium)
library(readr)

# Path in
path_in <- "/mnt/data_dump/pixelstress/stats/"

# Load data
df_FCz <- read.csv("/mnt/data_dump/pixelstress/stats/stats_table_FCz.csv", header=TRUE)

# Compute Shapiro-Wilk test for each combinations of factor levels
df_FCz %>%
  group_by(group, trajectory, stage) %>%
  shapiro_test(V)

# Homogeneity of variance assumption using Levene’s test
df_FCz %>%
  group_by(trajectory, stage) %>%
  levene_test(V ~ group)

# Computation of ANOVA
aov_FCz <- anova_test(
  data = df_FCz, dv = V, wid = id,
  between = group, within = c(trajectory, stage)
)