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

# Drop non-end trials
df = df.drop(df[df.block_phase != "end"].index)

# Create df for correct only
df_correct_only = df.drop(df[df.accuracy != 1].index)

# Get rt for conditions
df_b = df_correct_only.groupby(["id", "block_phase", "block_wiggleroom", "block_outcome"])["rt"].mean().reset_index(name='rt')

# Get accuracy for conditions
series_n_all = df.groupby(["id", "block_phase", "block_wiggleroom", "block_outcome"]).size().reset_index(name='acc')["acc"]
series_n_correct = df_correct_only.groupby(["id", "block_phase", "block_wiggleroom", "block_outcome"]).size().reset_index(name='acc')["acc"]
series_accuracy = series_n_correct / series_n_all

# Get session condition for conditions
series_session = df.groupby(["id", "block_phase", "block_wiggleroom", "block_outcome"])["session_condition"].mean().reset_index(name='session')["session"]

# Compute inverse efficiency
series_ie = df_b["rt"] / series_accuracy

# Combine
df_b["acc"] = series_accuracy
df_b["group"] = series_session
df_b["ie"] = series_ie

# Rename vars
df_b = df_b.rename(columns={"block_phase": "time", "block_wiggleroom": "dist", "block_outcome": "outcome"})

# Make vars categorial
df_b["group"] = df_b["group"].astype("category")
df_b["outcome"] = df_b["outcome"].astype("category")
df_b["dist"] = df_b["dist"].astype("category")

# Plot RT
g = sns.FacetGrid(df_b, col="group", hue="dist")
g.map(sns.pointplot, "outcome", "rt")
g.add_legend()

# Plot accuracy
g = sns.FacetGrid(df_b, col="group", hue="dist")
g.map(sns.pointplot, "outcome", "acc")
g.add_legend()

# Plot inverse efficiency
g = sns.FacetGrid(df_b, col="group", hue="dist")
g.map(sns.pointplot, "outcome", "ie")
g.add_legend()

# Save to csv for R
fn = os.path.join(path_in, "pixelstress_behavioral_data.csv")
df_b.to_csv(fn, index=False) 












































