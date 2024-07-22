library(dplyr)
library(ggpubr)
library(rstatix)
library(datarium)
library(readr)

# Path in
path_in <- "/mnt/data_dump/pixelstress/stats/"

# Load data
df_cnv <- read.csv("/mnt/data_dump/pixelstress/stats/stats_table_frontal_electrodes.csv", header=TRUE)

# ANOVA cnv
aov_cnv <- anova_test(
  data = df_cnv, dv = V, wid = id,
  between = group, within = c(trajectory, stage)
)