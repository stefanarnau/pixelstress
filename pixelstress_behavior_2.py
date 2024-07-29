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
datasets = glob.glob(f"{path_in}/*erp.set")

# Collector bin for all trials
df = []

# Loop datasets
for dataset in datasets:

    # Load trialinfo
    df_trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "erp_trialinfo.csv")

    # Get trial indices
    early_sequences = 5
    late_sequences = 8
    idx_close_early = np.where(
        (df_trialinfo.sequence_nr < early_sequences)
        & (df_trialinfo.block_wiggleroom == 0)
    )[0]
    idx_below_early = np.where(
        (df_trialinfo.sequence_nr < early_sequences)
        & (df_trialinfo.block_wiggleroom == 1)
        & (df_trialinfo.block_outcome == -1)
    )[0]
    idx_above_early = np.where(
        (df_trialinfo.sequence_nr < early_sequences)
        & (df_trialinfo.block_wiggleroom == 1)
        & (df_trialinfo.block_outcome == 1)
    )[0]
    idx_close_late = np.where(
        (df_trialinfo.sequence_nr > late_sequences)
        & (df_trialinfo.block_wiggleroom == 0)
    )[0]
    idx_below_late = np.where(
        (df_trialinfo.sequence_nr > late_sequences)
        & (df_trialinfo.block_wiggleroom == 1)
        & (df_trialinfo.block_outcome == -1)
    )[0]
    idx_above_late = np.where(
        (df_trialinfo.sequence_nr > late_sequences)
        & (df_trialinfo.block_wiggleroom == 1)
        & (df_trialinfo.block_outcome == 1)
    )[0]

    df.append(df_trialinfo)

# Concatenate dfs
df = pd.concat(df).reset_index()

# Add new variable post cold
df = df.assign(post_cold="no")
df.post_cold[df["block_nr"].isin([1, 4, 5, 8])] = "yes"


# Add new variable stage
df = df.assign(stage="middle")
df.stage[df.sequence_nr <= 4] = "begin"
df.stage[df.sequence_nr >= 9] = "end"

# Add new variable trajectory
df = df.assign(trajectory="none")
df.trajectory[df.block_wiggleroom == 0] = "close"
df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == -1)] = "below"
df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == 1)] = "above"

# Add new variable trajectory2
df = df.assign(trajectory2="none")
df.trajectory2[(df.block_wiggleroom == 0) & (df.block_outcome == 1)] = "+1"
df.trajectory2[(df.block_wiggleroom == 0) & (df.block_outcome == -1)] = "-1"
df.trajectory2[(df.block_wiggleroom == 1) & (df.block_outcome == 1)] = "+2"
df.trajectory2[(df.block_wiggleroom == 1) & (df.block_outcome == -1)] = "-2"

# Drop middle trials
# df = df.drop(df[df.stage == "middle"].index).reset_index()

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

# Plot
g = sns.catplot(
    data=df_rt,
    x="group",
    y="ms",
    hue="trajectory",
    capsize=0.2,
    palette="rocket",
    errorbar="se",
    kind="point",
    height=6,
    aspect=0.75,
)
g.despine(left=True)


g = sns.catplot(
    data=df_rt,
    x="group",
    y="acc",
    hue="trajectory",
    capsize=0.2,
    palette="rocket",
    errorbar="se",
    kind="point",
    height=6,
    aspect=0.75,
)
g.despine(left=True)
