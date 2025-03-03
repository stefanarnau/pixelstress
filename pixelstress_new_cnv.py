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
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp.set")

# Load channel labels
channel_labels = (
    open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r")
    .read()
    .split("\n")[:-1]
)

# Collector bin 
df_sequences = []

# Loop datasets
for dataset in datasets:

    # Read trialinfo
    df_tmp = pd.read_csv(dataset.split("_cleaned")[0] + "_erp_trialinfo.csv")

    # Load eeg data as trials x channel x times
    eeg_data = np.transpose(scipy.io.loadmat(dataset)["data"], [2, 0, 1])
    
    # Drop first sequences
    idx_not_first_sequences = (df_tmp.sequence_nr != 1).values
    df_tmp = df_tmp[idx_not_first_sequences]
    eeg_data = eeg_data[idx_not_first_sequences, :, :]
    
    # Drop outlier rt trials
    z_scores = np.abs((df_tmp["rt"] - df_tmp["rt"].mean()) / df_tmp["rt"].std())
    idx_rt_not_outliers = z_scores < 2
    df_tmp = df_tmp[idx_rt_not_outliers]
    eeg_data = eeg_data[idx_rt_not_outliers, :, :]
    
    # Drop incorrect trials
    idx_correct = (df_tmp.accuracy == 1).values
    df_tmp = df_tmp[idx_correct]
    eeg_data = eeg_data[idx_correct, :, :]
    
    # Add new variable sequence number total
    df_tmp = df_tmp.assign(sequence_nr_total="none")
    df_tmp.sequence_nr_total = df_tmp.sequence_nr + (df_tmp.block_nr - 1) * 12
    
    # Select channel and average eeg data to trial x time
    electrode_selection = ["Cz", "C1", "C2"]
    idx_channel = [
        idx
        for idx, element in enumerate(channel_labels)
        if element in electrode_selection
    ]
    eeg_data = eeg_data[:, idx_channel, :].mean(axis=1)
    
    # Get time vector
    erp_times = np.arange(-1.7, 1.2, 0.001)
    
    # Iterate all sequences
    erps = []
    for seq in set(df_tmp["sequence_nr_total"].values):
        
        # Get sequence indices
        idx_seq = df_tmp["sequence_nr_total"].values == seq
        
        # Collect erps
        erps.append(eeg_data[idx_seq, :].mean(axis=0));
    
    # Stack
    erps = np.stack(erps)
    
    # Average for time window
    time_idx = (erp_times >= -0.2) & (erp_times <= 0)
    cnv = erps[:, time_idx].mean(axis=1)
    
    # Group variables by sequences and get rt for conditions
    df_tmp_ave = (
        df_tmp.groupby(["sequence_nr_total"])[
            "id",
            "session_condition",
            "block_wiggleroom",
            "block_outcome",
            "last_feedback",
            "last_feedback_scaled",
            "block_nr",
            "sequence_nr",
        ]
        .mean()
        .reset_index()
    )
    
    # Add cnv to averaged dataframe
    df_tmp_ave = df_tmp_ave.assign(cnv=cnv)

    # Add erps to averaged dataframe
    df_tmp_ave = df_tmp_ave.assign(erp=pd.Series(erps.tolist())) 
    
    # Collect
    df_sequences.append(df_tmp_ave)

# Concatenate datasets
df_sequences = pd.concat(df_sequences).reset_index()

# Make categorial
df_sequences['session_condition'] = pd.Categorical(df_sequences['session_condition'])

# Select
dataset = df_sequences
depvar = "cnv"

# Linear mixed model
model = smf.mixedlm(
    depvar + " ~ last_feedback_scaled*session_condition", data=dataset, groups="id"
)
results = model.fit()
results.summary()








