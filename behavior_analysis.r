library(dplyr)
library(ggpubr)
library(rstatix)
library(datarium)
library(readr)

# install.packages("ggpubr")
# Path in
path_in <- "/mnt/data_dump/pixelstress/3_behavior/"

# Load data
df_behavior <- read.csv("/mnt/data_dump/pixelstress/3_behavior/pixelstress_behavioral_data.csv", header=TRUE)
df_cnv <- read.csv("/mnt/data_dump/pixelstress/3_behavior/cnv_table.csv", header=TRUE)

# Make variables categorical
df_behavior$id <- factor(df_behavior$id)
df_behavior$dist <- factor(df_behavior$dist)
df_behavior$outcome <- factor(df_behavior$outcome)
df_behavior$group <- factor(df_behavior$group)
df_cnv$id <- factor(df_cnv$id)
df_cnv$dist <- factor(df_cnv$dist)
df_cnv$outcome <- factor(df_cnv$outcome)
df_cnv$group <- factor(df_cnv$group)

# Compute Shapiro-Wilk test for each combinations of factor levels
df_behavior %>%
  group_by(group, dist, outcome) %>%
  shapiro_test(rt)

# Homogeneity of variance assumption using Levene’s test
df_behavior %>%
  group_by(dist, outcome) %>%
  levene_test(rt ~ group)

# Computation of ANOVA
aov_rt <- anova_test(
  data = df_behavior, dv = rt, wid = id,
  between = group, within = c(dist, outcome)
)
aov_acc <- anova_test(
  data = df_behavior, dv = acc, wid = id,
  between = group, within = c(dist, outcome)
)
aov_ie <- anova_test(
  data = df_behavior, dv = ie, wid = id,
  between = group, within = c(dist, outcome)
)
aov_cnv <- anova_test(
  data = df_cnv, dv = cnv_amp, wid = id,
  between = group, within = c(dist, outcome)
)