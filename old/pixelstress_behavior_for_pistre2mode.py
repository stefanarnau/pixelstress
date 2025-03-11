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

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_plot = "/mnt/data_dump/pixelstress/plots/"
path_stats = "/mnt/data_dump/pixelstress/stats/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp.set")

# Collector bin for all trials
df_behavior = []

# Loop datasets
for dataset in datasets:
    
    # Load trialinfo
    df_trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "erp_trialinfo.csv")
    
    # An empty dict. So sad...
    row = {}
    
    # Loop blocks
    for block_nr in range(8):
    
        # Get trial indices
        early_sequences = 5
        late_sequences = 8
        idx_early = np.where(
            (df_trialinfo.sequence_nr < early_sequences)
            & (df_trialinfo.block_nr == block_nr + 1)
        )[0]
        idx_late = np.where(
            (df_trialinfo.sequence_nr > late_sequences)
            & (df_trialinfo.block_nr == block_nr + 1)
        )[0]
        
        # Get early and late subsets
        df_early = df_trialinfo.iloc[idx_early]
        df_late = df_trialinfo.iloc[idx_late]
        
        # Get rts
        rt_early = df_early[df_early.accuracy == 1].rt.mean()
        rt_late = df_late[df_late.accuracy == 1].rt.mean()
        
        # Get accuracies
        acc_early = df_early[df_early.accuracy == 1].shape[0] / df_early.shape[0]
        acc_late = df_late[df_late.accuracy == 1].shape[0] / df_late.shape[0]
        
        # Fill dict
        dict_label = "bl" + str(block_nr + 1) + "_rt_early"
        row[dict_label] = rt_early
        dict_label = "bl" + str(block_nr + 1) + "_rt_late"
        row[dict_label] = rt_late
        dict_label = "bl" + str(block_nr + 1) + "_acc_early"
        row[dict_label] = acc_early
        dict_label = "bl" + str(block_nr + 1) + "_acc_late"
        row[dict_label] = acc_late

        # Append to list
        df_behavior.append(df_trialinfo)

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
df = df.drop(df[df.stage == "middle"].index).reset_index()

# Create df for correct only
df_correct_only = df.drop(df[df.accuracy != 1].index)

# Get rt for conditions
df_rt = (
    df_correct_only.groupby(["id", "trajectory", "stage"])["rt"]
    .mean()
    .reset_index(name="ms")
)

# Get accuracy for conditions
series_n_all = (
    df.groupby(["id", "trajectory", "stage"]).size().reset_index(name="acc")["acc"]
)
series_n_correct = (
    df_correct_only.groupby(["id", "trajectory", "stage"])
    .size()
    .reset_index(name="acc")["acc"]
)
series_accuracy = series_n_correct / series_n_all

# Get session condition for conditions
series_session = (
    df.groupby(["id", "trajectory", "stage"])["session_condition"]
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
    x="stage",
    y="ms",
    hue="trajectory",
    col="group",
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
    x="stage",
    y="acc",
    hue="trajectory",
    col="group",
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
    x="stage",
    y="ie",
    hue="trajectory",
    col="group",
    capsize=0.2,
    palette="rocket",
    errorbar="se",
    kind="point",
    height=6,
    aspect=0.75,
)
g.despine(left=True)


# Save dataframe
df_rt.to_csv(os.path.join(path_stats, "behavioral data.csv"))





