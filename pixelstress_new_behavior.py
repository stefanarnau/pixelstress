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
import statsmodels.formula.api as smf
import sklearn.linear_model

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_plot = "/mnt/data_dump/pixelstress/plots/"
path_stats = "/mnt/data_dump/pixelstress/stats/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp_trialinfo.csv")

# Collector bin for all trials
df_sequences = []
df_single_trial = []

# Collect datasets
for dataset in datasets:

    # Read data
    df_tmp = pd.read_csv(dataset)

    # Drop first sequences
    df_tmp = df_tmp.drop(df_tmp[df_tmp.sequence_nr <= 1].index)

    # Drop outliers
    z_scores = np.abs((df_tmp["rt"] - df_tmp["rt"].mean()) / df_tmp["rt"].std())
    df_tmp = df_tmp[z_scores < 2]

    # Remove trial difficulty confound using linear regression
    X = df_tmp[["trial_difficulty"]].values
    y = df_tmp["rt"].values
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    df_tmp["rt_detrended"] = y - y_pred

    # Add new variable sequence number total
    df_tmp = df_tmp.assign(sequence_nr_total="none")
    df_tmp.sequence_nr_total = df_tmp.sequence_nr + (df_tmp.block_nr - 1) * 12

    # Create df for correct only
    df_tmp_correct_only = df_tmp.drop(df_tmp[df_tmp.accuracy != 1].index)

    # Add new variable trajectory
    df_tmp_correct_only = df_tmp_correct_only.assign(trajectory=99)
    df_tmp_correct_only.trajectory[df_tmp_correct_only.block_wiggleroom == 0] = 0
    df_tmp_correct_only.trajectory[
        (df_tmp_correct_only.block_wiggleroom == 1)
        & (df_tmp_correct_only.block_outcome == -1)
    ] = -1
    df_tmp_correct_only.trajectory[
        (df_tmp_correct_only.block_wiggleroom == 1)
        & (df_tmp_correct_only.block_outcome == 1)
    ] = +1

    # Group variables by sequences and get rt for conditions
    df_tmp_ave = (
        df_tmp_correct_only.groupby(["sequence_nr_total"])[
            "id",
            "session_condition",
            "block_wiggleroom",
            "block_outcome",
            "trajectory",
            "last_feedback",
            "last_feedback_scaled",
            "block_nr",
            "sequence_nr",
            "rt",
            "rt_detrended",
        ]
        .mean()
        .reset_index()
    )

    # Iterate sequences and calculate accuracy
    df_tmp_ave = df_tmp_ave.assign(accuracy=99)
    for row_idx, seq_total in enumerate(df_tmp_ave["sequence_nr_total"].values):
        
        # Get indices of toal sequence number
        idx_seqtotal = df_tmp["sequence_nr_total"].values == seq_total
        
        # get accuracy
        df_tmp_ave.loc[row_idx, 'accuracy'] = sum(df_tmp["accuracy"].values[idx_seqtotal] == 1) / sum(idx_seqtotal)

    # Collect
    df_single_trial.append(df_tmp_correct_only)
    df_sequences.append(df_tmp_ave)

# Concatenate datasets
df_single_trial = pd.concat(df_single_trial).reset_index()
df_sequences = pd.concat(df_sequences).reset_index()

# Make categorial
df_sequences['trajectory'] = pd.Categorical(df_sequences['trajectory'])
df_sequences['session_condition'] = pd.Categorical(df_sequences['session_condition'])

# Select
dataset = df_sequences
depvar = "rt_detrended"

# Linear mixed model
model = smf.mixedlm(
    depvar + " ~ last_feedback_scaled*session_condition", data=dataset, groups="id"
)
results = model.fit()
results.summary()

# Plot
sns.set_style("whitegrid", {"axes.facecolor": "#F1F1F1"})
sns.lmplot(
    data=dataset,
    x="last_feedback_scaled",
    y=depvar,
    hue="session_condition",
    palette=["#003333", "#B90076"],
    scatter_kws={"s": 2},
)
