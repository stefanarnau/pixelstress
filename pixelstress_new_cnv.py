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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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

    # Get time vector
    erp_times = np.arange(-1.7, 1.2, 0.001)

    # Iterate all sequences
    erps_Fz = []
    erps_F1 = []
    erps_F2 = []
    erps_FCz = []
    erps_FC1 = []
    erps_FC2 = []
    erps_Cz = []
    erps_C1 = []
    erps_C2 = []

    # Iterate sequences and average erps
    for seq in set(df_tmp["sequence_nr_total"].values):

        # Get sequence indices
        idx_seq = df_tmp["sequence_nr_total"].values == seq

        # Get midline erps
        erps_Fz.append(eeg_data[idx_seq, 4, :].mean(axis=0))
        erps_F1.append(eeg_data[idx_seq, 37, :].mean(axis=0))
        erps_F2.append(eeg_data[idx_seq, 38, :].mean(axis=0))
        erps_FCz.append(eeg_data[idx_seq, 64, :].mean(axis=0))
        erps_FC1.append(eeg_data[idx_seq, 8, :].mean(axis=0))
        erps_FC2.append(eeg_data[idx_seq, 9, :].mean(axis=0))
        erps_Cz.append(eeg_data[idx_seq, 13, :].mean(axis=0))
        erps_C1.append(eeg_data[idx_seq, 47, :].mean(axis=0))
        erps_C2.append(eeg_data[idx_seq, 48, :].mean(axis=0))

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

    # Save erps to dataframe
    df_tmp_ave = df_tmp_ave.assign(erp_Fz=pd.Series(erps_Fz))
    df_tmp_ave = df_tmp_ave.assign(erp_F1=pd.Series(erps_F1))
    df_tmp_ave = df_tmp_ave.assign(erp_F2=pd.Series(erps_F2))
    df_tmp_ave = df_tmp_ave.assign(erp_FCz=pd.Series(erps_FCz))
    df_tmp_ave = df_tmp_ave.assign(erp_FC1=pd.Series(erps_FC1))
    df_tmp_ave = df_tmp_ave.assign(erp_FC2=pd.Series(erps_FC2))
    df_tmp_ave = df_tmp_ave.assign(erp_Cz=pd.Series(erps_Cz))
    df_tmp_ave = df_tmp_ave.assign(erp_C1=pd.Series(erps_C1))
    df_tmp_ave = df_tmp_ave.assign(erp_C2=pd.Series(erps_C2))

    # Collect dataframes
    df_sequences.append(df_tmp_ave)


# Concatenate datasets
df_sequences = pd.concat(df_sequences).reset_index()

# Make categorial
df_sequences["session_condition"] = pd.Categorical(df_sequences["session_condition"])

# Select time window
time_idx = (erp_times >= -0.6) & (erp_times <= 0)

# Select electrodes
chans = ["erp_Cz", "erp_C1", "erp_C2"]

# Iterate rows and calculate erp averages for time window and defined channels
values = []
for i in range(len(df_sequences)):

    # Get row
    row = df_sequences.iloc[i]

    # Get erp average
    values.append(np.stack([row[ch][time_idx].mean() for ch in chans]).mean())

# Add to df
df_sequences = df_sequences.assign(cnv=values)

# Linear mixed model
model = smf.mixedlm(
    "cnv ~ last_feedback_scaled*session_condition*sequence_nr",
    data=df_sequences,
    groups="id",
)
results = model.fit()
results.summary()



# Plot
sns.set_style("whitegrid", {"axes.facecolor": "#F1F1F1"})
sns.lmplot(
    data=df_sequences,
    x="last_feedback_scaled",
    y="cnv",
    hue="session_condition",
    col="sequence_nr",
    palette=["#003333", "#B90076"],
    scatter_kws={"s": 2},
)





# Plot main effect sequence_nr

# Group by sequence nr
grouped = df_sequences.groupby("sequence_nr").agg({
    col: lambda x: np.mean(x) for col in chans
}).reset_index()

def average_vectors(row):
    vectors = [row[chan] for chan in chans]
    return np.mean(vectors, axis=0)

grouped['average'] = grouped.apply(average_vectors, axis=1)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Create a color map
cmap = plt.get_cmap('rocket')
norm = Normalize(vmin=grouped["sequence_nr"].min(), vmax=grouped["sequence_nr"].max())

# Plot each averaged vector
for _, row in grouped.iterrows():
    color = cmap(norm(row["sequence_nr"]))
    ax.plot(erp_times, row['average'], color=color, alpha=0.7, label=f"Fish: {row['sequence_nr']}")

# Add a colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('sequence nr')

# Set labels and title
ax.set_xlabel('ms')
ax.set_ylabel('mV')
ax.set_title('ERP by sequence nr')

plt.tight_layout()
plt.show()








