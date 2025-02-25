#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:52:54 2021

@author: Stefan Arnau
"""

# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_plot = "/mnt/data_dump/pixelstress/plots/"
path_stats = "/mnt/data_dump/pixelstress/stats/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp_trialinfo.csv")

# Collector bin for all trials
df = []

# Collect datasets
for dataset in datasets:
    df.append(pd.read_csv(dataset))
    
# Concatenate datasets
df = pd.concat(df).reset_index()

# Add new variable trajectory
df = df.assign(trajectory="none")
df.trajectory[df.block_wiggleroom == 0] = "close"
df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == -1)] = "below"
df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == 1)] = "above"

# Drop early sequences
df = df.drop(df[df.sequence_nr <= 4 ].index)

# Create df for correct only
df_correct_only = df.drop(df[df.accuracy != 1].index)

# Get rt for conditions
df_rt = (
    df_correct_only.groupby(["id", "trajectory"])["rt"]
    .mean()
    .reset_index(name="ms")
)

# Get accuracy for conditions
series_n_all = (
    df.groupby(["id", "trajectory"]).size().reset_index(name="acc")["acc"]
)
series_n_correct = (
    df_correct_only.groupby(["id", "trajectory"])
    .size()
    .reset_index(name="acc")["acc"]
)
series_accuracy = series_n_correct / series_n_all

# Get session condition for conditions
series_session = (
    df.groupby(["id", "trajectory"])["session_condition"]
    .mean()
    .reset_index(name="session")["session"]
)

# Compute inverse efficiency
series_ie = df_rt["ms"] / series_accuracy

# Combine
df_rt["acc"] = series_accuracy
df_rt["group"] = series_session
df_rt["ie"] = series_ie

# Rename group vars
df_rt.group[(df_rt.group == 1)] = "experimental"
df_rt.group[(df_rt.group == 2)] = "control"

# Make vars categorial
df_rt["group"] = df_rt["group"].astype("category")
df_rt["trajectory"] = df_rt["trajectory"].astype("category")


aov = pg.mixed_anova(data=df_rt, dv='ms', between='group', within='trajectory', subject='id')
print(aov)


# Plot
g = sns.catplot(
    data=df_rt,
    x="trajectory",
    y="ms",
    hue="group",
    palette="rocket",
    errorbar="se",
    kind="violin",
    height=6,
    aspect=0.75,
)
g.despine(left=True)


g = sns.catplot(
    data=df_rt,
    x="trajectory",
    y="acc",
    hue="group",
    capsize=0.2,
    palette="rocket",
    errorbar="se",
    kind="line",
    height=6,
    aspect=0.75,
)
g.despine(left=True)

g = sns.catplot(
    data=df_rt,
    x="trajectory",
    y="ie",
    hue="group",
    capsize=0.2,
    palette="rocket",
    errorbar="se",
    kind="line",
    height=6,
    aspect=0.75,
)
g.despine(left=True)


# Save dataframe
#df_rt.to_csv(os.path.join(path_stats, "behavioral data.csv"))





