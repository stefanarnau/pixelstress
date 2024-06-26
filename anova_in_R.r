library(dplyr)
library(ggpubr)
library(rstatix)
library(datarium)
library(readr)

# Path in
path_in <- "/mnt/data_dump/pixelstress/stats/"

# Load data
df_frontal_theta <- read.csv("/mnt/data_dump/pixelstress/stats/stats_table_frontal_theta.csv", header=TRUE)

df_cnv <- read.csv("/mnt/data_dump/pixelstress/stats/stats_table_FCz.csv", header=TRUE)


# Computation of ANOVA
aov_frontal_theta <- anova_test(
  data = df_frontal_theta, dv = dB, wid = id,
  between = group, within = c(trajectory, stage)
)

aov_frontal_theta <- aov_frontal_theta[[1]]

# ANOVA cnv
aov_cnv <- anova_test(
  data = df_cnv, dv = V, wid = id,
  between = group, within = c(trajectory, stage)
)