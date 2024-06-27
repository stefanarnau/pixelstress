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


# Function for plotting erps and calculate stats
def get_erpplot_and_stats(electrode_selection, stat_label, timewin_stats):

    idx_channel = [
        idx
        for idx, element in enumerate(eeg_epochs.ch_names)
        if element in electrode_selection
    ]
    df_rows = []
    for idx_id, id in enumerate(ids):

        for idx_t, t in enumerate(erp_times):

            if (t >= timewin_stats[0]) & (t <= timewin_stats[1]):
                in_statwin = 1
            else:
                in_statwin = 0

            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "close",
                    "stage": "early",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_close_early[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "below",
                    "stage": "early",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_below_early[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "above",
                    "stage": "early",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_above_early[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "close",
                    "stage": "late",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_close_late[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "below",
                    "stage": "late",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_below_late[idx_id, idx_channel, idx_t].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "above",
                    "stage": "late",
                    "time (s)": erp_times[idx_t],
                    "in_statwin": in_statwin,
                    "V": matrices_above_late[idx_id, idx_channel, idx_t].mean(),
                }
            )

    # Get dataframe
    df_frontal_erp = pd.DataFrame(df_rows)

    # Create ERP Lineplot
    sns.set_style("darkgrid")
    sns.relplot(
        data=df_frontal_erp,
        x="time (s)",
        y="V",
        hue="trajectory",
        col="stage",
        row="group",
        kind="line",
        height=3,
        aspect=1.8,
        errorbar=None,
        palette="rocket",
    )

    # Save plot
    plt.savefig(
        os.path.join(path_plot, "lineplot_" + stat_label + ".png"),
        dpi=300,
        transparent=True,
    )

    # Get dataframe statistical analysis (average is timewin)
    df_stats = df_frontal_erp.drop(df_frontal_erp[df_frontal_erp.in_statwin != 1].index)
    df_stats = (
        df_stats.groupby(["id", "group", "trajectory", "stage"])["V"]
        .mean()
        .reset_index()
    )

    # Save dataframe
    df_stats.to_csv(os.path.join(path_stats, "stats_table_" + stat_label + ".csv"))
    
    # Return stat dataframe
    return df_stats


# Collector lists
matrices_close_early = []
matrices_below_early = []
matrices_above_early = []
matrices_close_late = []
matrices_below_late = []
matrices_above_late = []
group = []
ids = []

# Loop datasets
for dataset in datasets:

    # Get id
    ids.append(int(dataset.split("/")[-1].split("_")[1]))

    # Load a dataset
    eeg_epochs = (
        mne.io.read_epochs_eeglab(dataset)
        .apply_baseline(baseline=(-1.2, -1))
        .resample(200)
        .crop(tmin=-1.2, tmax=0.5)
    )

    # Save times
    erp_times = eeg_epochs.times

    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "erp_trialinfo.csv")

    # Save group
    group.append(trialinfo.session_condition[0])

    # Get trial indices
    early_sequences = 5
    late_sequences = 8
    idx_close_early = np.where(
        (trialinfo.sequence_nr < early_sequences) & (trialinfo.block_wiggleroom == 0)
    )[0]
    idx_below_early = np.where(
        (trialinfo.sequence_nr < early_sequences)
        & (trialinfo.block_wiggleroom == 1)
        & (trialinfo.block_outcome == -1)
    )[0]
    idx_above_early = np.where(
        (trialinfo.sequence_nr < early_sequences)
        & (trialinfo.block_wiggleroom == 1)
        & (trialinfo.block_outcome == 1)
    )[0]
    idx_close_late = np.where(
        (trialinfo.sequence_nr > late_sequences) & (trialinfo.block_wiggleroom == 0)
    )[0]
    idx_below_late = np.where(
        (trialinfo.sequence_nr > late_sequences)
        & (trialinfo.block_wiggleroom == 1)
        & (trialinfo.block_outcome == -1)
    )[0]
    idx_above_late = np.where(
        (trialinfo.sequence_nr > late_sequences)
        & (trialinfo.block_wiggleroom == 1)
        & (trialinfo.block_outcome == 1)
    )[0]

    # Get evokeds
    evoked_close_early = eeg_epochs[idx_close_early].average()
    evoked_below_early = eeg_epochs[idx_below_early].average()
    evoked_above_early = eeg_epochs[idx_above_early].average()
    evoked_close_late = eeg_epochs[idx_close_late].average()
    evoked_below_late = eeg_epochs[idx_below_late].average()
    evoked_above_late = eeg_epochs[idx_above_late].average()

    # Collect as matrices
    matrices_close_early.append(evoked_close_early.data)
    matrices_below_early.append(evoked_below_early.data)
    matrices_above_early.append(evoked_above_early.data)
    matrices_close_late.append(evoked_close_late.data)
    matrices_below_late.append(evoked_below_late.data)
    matrices_above_late.append(evoked_above_late.data)

# Stack matrices
matrices_close_early = np.stack(matrices_close_early)
matrices_below_early = np.stack(matrices_below_early)
matrices_above_early = np.stack(matrices_above_early)
matrices_close_late = np.stack(matrices_close_late)
matrices_below_late = np.stack(matrices_below_late)
matrices_above_late = np.stack(matrices_above_late)


# Get ERP for Fz
df_stats_FCz = get_erpplot_and_stats(
    electrode_selection=["FCz"], stat_label="FCz", timewin_stats=(-0.2, 0)
)

# Draw a pointplot
g = sns.catplot(
    data=df_stats_FCz, x="stage", y="V", hue="trajectory", col="group",
    capsize=.2, palette="rocket", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)

























