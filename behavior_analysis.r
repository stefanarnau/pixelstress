library(dplyr)
library(ggpubr)
library(rstatix)
library(datarium)
library(readr)

# install.packages("ggpubr")
# Path in
path_in <- "/mnt/data_dump/pixelstress/3_behavior/"

# Load data
df <- read.csv("/mnt/data_dump/pixelstress/3_behavior/pixelstress_behavioral_data.csv", header=TRUE)

# Make variables categorical
df$id <- factor(df$id)
df$time <- factor(df$time)
df$dist <- factor(df$dist)
df$outcome <- factor(df$outcome)
df$group <- factor(df$group)

# Compute Shapiro-Wilk test for each combinations of factor levels
df %>%
  group_by(group, dist, outcome) %>%
  shapiro_test(rt)

# Homogeneity of variance assumption using Levene’s test
df %>%
  group_by(dist, outcome) %>%
  levene_test(rt ~ group)

# Computation of ANOVA
aov_rt <- anova_test(
  data = df, dv = rt, wid = id,
  between = group, within = c(dist, outcome)
)
aov_acc <- anova_test(
  data = df, dv = acc, wid = id,
  between = group, within = c(dist, outcome)
)
aov_ie <- anova_test(
  data = df, dv = ie, wid = id,
  between = group, within = c(dist, outcome)
)