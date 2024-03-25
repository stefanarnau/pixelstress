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
import sklearn.linear_model
import sklearn.preprocessing
import seaborn as sns

# Define path
path_in = "/mnt/data_dump/pixelstress/3_behavior/"

# Load data
df = pd.read_csv(os.path.join(path_in, "behavior_all_tf.csv"))

# Add new variable block_phase
df = df.assign(block_phase="middle")
df.block_phase[df.sequence_nr <= 4] = "begin"
df.block_phase[df.sequence_nr >= 9] = "end"

# Drop 'middle' trials
df = df.drop(df[df.block_phase == "middle"].index)

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

# Combine
df_b["acc"] = series_accuracy
df_b["group"] = series_session

# Rename vars
df_b = df_b.rename(columns={"block_phase": "time", "block_wiggleroom": "dist", "block_outcome": "outcome"})

# Make vars categorial
df_b["group"] = df_b["group"].astype("category")
df_b["outcome"] = df_b["outcome"].astype("category")
df_b["dist"] = df_b["dist"].astype("category")



# Plot RT
g = sns.FacetGrid(df_b, row="group", col="time", hue="outcome")
g.map(sns.pointplot, "dist", "rt")
g.add_legend()


# Plot accuracy
g = sns.FacetGrid(df_b, row="group", col="time", hue="outcome")
g.map(sns.pointplot, "dist", "acc")
g.add_legend()














































