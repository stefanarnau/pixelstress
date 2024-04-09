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
import joblib
import scipy.io
import seaborn as sns
import pingouin as pg

# Define path
path_in = "/mnt/data_dump/pixelstress/3_behavior/"

# Load data
df = pd.read_csv(os.path.join(path_in, "behavior_all_tf.csv"))

# Add new variable block_phase
df = df.assign(block_phase="middle")
df.block_phase[df.sequence_nr <= 6] = "begin"
df.block_phase[df.sequence_nr >= 7] = "end"

# Add new variable trajectory
df = df.assign(trajectory="none")
df.trajectory[df.block_wiggleroom == 0] = "close"
df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == -1)] = "below"
df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == 1)] = "above"

# Drop non-end trials
df = df.drop(df[df.block_phase != "end"].index)

# Create df for correct only
df_correct_only = df.drop(df[df.accuracy != 1].index)

# Get rt for conditions
df_b = df_correct_only.groupby(["id", "trajectory"])["rt"].mean().reset_index(name='rt')

# Get accuracy for conditions
series_n_all = df.groupby(["id", "trajectory"]).size().reset_index(name='acc')["acc"]
series_n_correct = df_correct_only.groupby(["id", "trajectory"]).size().reset_index(name='acc')["acc"]
series_accuracy = series_n_correct / series_n_all

# Get session condition for conditions
series_session = df.groupby(["id", "trajectory"])["session_condition"].mean().reset_index(name='session')["session"]

# Compute inverse efficiency
series_ie = df_b["rt"] / series_accuracy

# Combine
df_b["acc"] = series_accuracy
df_b["group"] = series_session
df_b["ie"] = series_ie

# Make vars categorial
df_b["group"] = df_b["group"].astype("category")
df_b["trajectory"] = df_b["trajectory"].astype("category")

# Plot RT
sns.pointplot(data=df_b, x="trajectory", y="rt", hue="group")

# Plot accuracy
sns.pointplot(data=df_b, x="trajectory", y="acc", hue="group")

# Mixed anova
aov_rt = pg.mixed_anova(dv='rt', between='group', within='trajectory', subject='id', data=df_b)
aov_acc = pg.mixed_anova(dv='acc', between='group', within='trajectory', subject='id', data=df_b)










































