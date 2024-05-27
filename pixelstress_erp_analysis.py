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
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_plot = "/mnt/data_dump/pixelstress/plots/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp.set")

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


# Create dataframe
electrode_selection = ["FC1", "FCz", "FC2", "Cz", "Fz"]
idx_channel = [
    idx
    for idx, element in enumerate(eeg_epochs.ch_names)
    if element in electrode_selection
]
timewin_stats = (-0.5, 0)
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


# ERP Lineplot
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
    ci=None,
    palette="rocket",
)

plt.savefig(
    os.path.join(path_plot, "lineplot_frontal_erp.png"),
    dpi=300,
    transparent=True,
)

aa = bb

# Get df for anovas
df_stat_frontal_erp = df_frontal_erp.drop(
    df_frontal_erp[df_frontal_erp.in_statwin != 1].index
)
df_stat_frontal_erp = (
    df_stat_frontal_erp.groupby(["id", "group", "trajectory", "stage"])["V"]
    .mean()
    .reset_index()
)

# Mixed anova
aov_frontal_erp = pg.mixed_anova(
    dv="theta (dB)",
    between="group",
    within="trajectory",
    subject="id",
    data=df_stat_theta,
)


# id group trajectory time value
timewin_stats = (-0.5, -0.1)
idx_channel = np.array([28, 29, 30, 60, 61, 62])
idx_freqs = (ersp_close.freqs >= 8) & (ersp_close.freqs <= 12)
df_posterior_alpha = pd.DataFrame(
    columns=["id", "group", "trajectory", "time (s)", "in_statwin", "alpha (dB)"]
)
for idx_id, id in enumerate(ids):

    for idx_t, t in enumerate(ersp_close.times):

        if (t >= timewin_stats[0]) & (t <= timewin_stats[1]):
            in_statwin = 1
        else:
            in_statwin = 0

        entry = pd.DataFrame(
            {
                "id": id,
                "group": ["experimental", "control"][group[idx_id] - 1],
                "trajectory": "close",
                "time (s)": ersp_close.times[idx_t],
                "in_statwin": in_statwin,
                "alpha (dB)": ersps_matrices_close[idx_id, idx_t, :, :][idx_freqs, :][
                    :, idx_channel
                ].mean(axis=(0, 1)),
            },
            index=[0],
        )
        df_posterior_alpha = pd.concat([df_posterior_alpha, entry], ignore_index=True)

        entry = pd.DataFrame(
            {
                "id": id,
                "group": ["experimental", "control"][group[idx_id] - 1],
                "trajectory": "below",
                "time (s)": ersp_close.times[idx_t],
                "in_statwin": in_statwin,
                "alpha (dB)": ersps_matrices_below[idx_id, idx_t, :, :][idx_freqs, :][
                    :, idx_channel
                ].mean(axis=(0, 1)),
            },
            index=[0],
        )
        df_posterior_alpha = pd.concat([df_posterior_alpha, entry], ignore_index=True)

        entry = pd.DataFrame(
            {
                "id": id,
                "group": ["experimental", "control"][group[idx_id] - 1],
                "trajectory": "above",
                "time (s)": ersp_close.times[idx_t],
                "in_statwin": in_statwin,
                "alpha (dB)": ersps_matrices_above[idx_id, idx_t, :, :][idx_freqs, :][
                    :, idx_channel
                ].mean(axis=(0, 1)),
            },
            index=[0],
        )
        df_posterior_alpha = pd.concat([df_posterior_alpha, entry], ignore_index=True)

# Lineplot posterior alpha
sns.lineplot(
    data=df_posterior_alpha,
    x="time (s)",
    y="alpha (dB)",
    hue="trajectory",
    style="group",
).set_title("posterior alpha [POz, PO1, PO2, O1, Oz, O2]")
plt.savefig(
    os.path.join(path_plot, "lineplot_alpha.png"),
    dpi=300,
    transparent=True,
)

# Get df for anovas
df_stat_alpha = df_posterior_alpha.drop(
    df_posterior_alpha[df_posterior_alpha.in_statwin != 1].index
)
df_stat_alpha = (
    df_stat_alpha.groupby(["id", "group", "trajectory"])["alpha (dB)"]
    .mean()
    .reset_index()
)

# Mixed anova
aov_posterior_alpha = pg.mixed_anova(
    dv="alpha (dB)",
    between="group",
    within="trajectory",
    subject="id",
    data=df_stat_alpha,
)
