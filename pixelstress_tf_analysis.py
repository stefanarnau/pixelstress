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
path_stats = "/mnt/data_dump/pixelstress/stats/"

# Define datasets
datasets = glob.glob(f"{path_in}/*tf.set")

# Load questionnaire
df_quest = pd.read_csv(os.path.join(path_in, "questionnaire.csv"))

# Reduce dataframe to exclude
#df_quest = df_quest[df_quest['influence'].isin(['4'])]

# Get indices to drop
#idx_drop = [idx for idx, e in enumerate([int(x.split("_")[-3]) for x in datasets]) if e in df_quest["id"].tolist()]

# Actually drop
#datasets = [e for idx, e in enumerate(datasets) if idx not in idx_drop]

# Create a montage
standard_1020_montage = mne.channels.make_standard_montage("standard_1020")

# Function for plotting erps and calculate stats
def get_erspplot_and_stats(
    electrode_selection, freq_selection, stat_label, timewin_stats
):

    idx_channel = np.isin(eeg_epochs.ch_names, electrode_selection)
    idx_freqs = (tf_freqs >= freq_selection[0]) & (tf_freqs <= freq_selection[1])
    df_rows = []
    for idx_id, id in enumerate(ids):

        for idx_t, t in enumerate(tf_times):

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
                    "time (s)": tf_times[idx_t],
                    "in_statwin": in_statwin,
                    "dB": matrices_close_early[idx_id, idx_t, idx_freqs, :][
                        :, idx_channel
                    ].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "below",
                    "stage": "early",
                    "time (s)": tf_times[idx_t],
                    "in_statwin": in_statwin,
                    "dB": matrices_below_early[idx_id, idx_t, idx_freqs, :][
                        :, idx_channel
                    ].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "above",
                    "stage": "early",
                    "time (s)": tf_times[idx_t],
                    "in_statwin": in_statwin,
                    "dB": matrices_above_early[idx_id, idx_t, idx_freqs, :][
                        :, idx_channel
                    ].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "close",
                    "stage": "late",
                    "time (s)": tf_times[idx_t],
                    "in_statwin": in_statwin,
                    "dB": matrices_close_late[idx_id, idx_t, idx_freqs, :][
                        :, idx_channel
                    ].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "below",
                    "stage": "late",
                    "time (s)": tf_times[idx_t],
                    "in_statwin": in_statwin,
                    "dB": matrices_below_late[idx_id, idx_t, idx_freqs, :][
                        :, idx_channel
                    ].mean(),
                }
            )
            df_rows.append(
                {
                    "id": id,
                    "group": ["experimental", "control"][group[idx_id] - 1],
                    "trajectory": "above",
                    "stage": "late",
                    "time (s)": tf_times[idx_t],
                    "in_statwin": in_statwin,
                    "dB": matrices_above_late[idx_id, idx_t, idx_freqs, :][
                        :, idx_channel
                    ].mean(),
                }
            )

    # Get dataframe
    df_ersp = pd.DataFrame(df_rows)

    # Create ERSP Lineplot
    sns.set_style("darkgrid")
    sns.relplot(
        data=df_ersp,
        x="time (s)",
        y="dB",
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
    df_stats = df_ersp.drop(df_ersp[df_ersp.in_statwin != 1].index)
    df_stats = (
        df_stats.groupby(["id", "group", "trajectory", "stage"])["dB"]
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
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-1.2, -1))

    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "tf_trialinfo.csv")

    # Save group
    group.append(trialinfo.session_condition[0])

    # Get trial indices
    early_sequences = 6
    late_sequences = 7
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

    # Perform time-frequency decomposition
    n_freqs = 25
    tf_freqs = np.linspace(3, 20, n_freqs)
    tf_cycles = np.linspace(6, 12, n_freqs)

    ersp_close_early = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_close_early],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )#.apply_baseline((-1.5, -1.2), mode="logratio")
    ersp_below_early = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_below_early],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )#.apply_baseline((-1.5, -1.2), mode="logratio")
    ersp_above_early = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_above_early],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )#.apply_baseline((-1.5, -1.2), mode="logratio")
    ersp_close_late = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_close_late],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )#.apply_baseline((-1.5, -1.2), mode="logratio")
    ersp_below_late = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_below_late],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )#.apply_baseline((-1.5, -1.2), mode="logratio")
    ersp_above_late = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_above_late],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )#.apply_baseline((-1.5, -1.2), mode="logratio")

    # Get baseline indices
    idx_bl = (ersp_close_early.times >= -1.5) & (ersp_close_early.times <= -1.2)

    # Average baseline values
    bl_values = (
        ersp_close_early._data
        + ersp_below_early._data
        + ersp_above_early._data
        + ersp_close_late._data
        + ersp_below_late._data
        + ersp_above_late._data
    ) / 3
    bl_values = bl_values[:, :, idx_bl].mean(axis=2)

    # Apply condition general dB baseline
    for ch in range(bl_values.shape[0]):
        for fr in range(bl_values.shape[1]):
            ersp_close_early._data[ch, fr, :] = 10 * np.log10(
                ersp_close_early._data[ch, fr, :].copy() / bl_values[ch, fr]
            )
            ersp_below_early._data[ch, fr, :] = 10 * np.log10(
                ersp_below_early._data[ch, fr, :].copy() / bl_values[ch, fr]
            )
            ersp_above_early._data[ch, fr, :] = 10 * np.log10(
                ersp_above_early._data[ch, fr, :].copy() / bl_values[ch, fr]
            )
            ersp_close_late._data[ch, fr, :] = 10 * np.log10(
                ersp_close_late._data[ch, fr, :].copy() / bl_values[ch, fr]
            )
            ersp_below_late._data[ch, fr, :] = 10 * np.log10(
                ersp_below_late._data[ch, fr, :].copy() / bl_values[ch, fr]
            )
            ersp_above_late._data[ch, fr, :] = 10 * np.log10(
                ersp_above_late._data[ch, fr, :].copy() / bl_values[ch, fr]
            )

    # Crop
    ersp_close_early.crop(tmin=-1.5, tmax=1)
    ersp_below_early.crop(tmin=-1.5, tmax=1)
    ersp_above_early.crop(tmin=-1.5, tmax=1)
    ersp_close_late.crop(tmin=-1.5, tmax=1)
    ersp_below_late.crop(tmin=-1.5, tmax=1)
    ersp_above_late.crop(tmin=-1.5, tmax=1)

    # Save times and freqs
    tf_times = ersp_close_early.times
    tf_freqs = ersp_close_early.freqs

    # Collect as matrices
    matrices_close_early.append(np.transpose(ersp_close_early.data, (2, 1, 0)))
    matrices_below_early.append(np.transpose(ersp_below_early.data, (2, 1, 0)))
    matrices_above_early.append(np.transpose(ersp_above_early.data, (2, 1, 0)))
    matrices_close_late.append(np.transpose(ersp_close_late.data, (2, 1, 0)))
    matrices_below_late.append(np.transpose(ersp_below_late.data, (2, 1, 0)))
    matrices_above_late.append(np.transpose(ersp_above_late.data, (2, 1, 0)))

# Stack matrices
matrices_close_early = np.stack(matrices_close_early)
matrices_below_early = np.stack(matrices_below_early)
matrices_above_early = np.stack(matrices_above_early)
matrices_close_late = np.stack(matrices_close_late)
matrices_below_late = np.stack(matrices_below_late)
matrices_above_late = np.stack(matrices_above_late)


# Get frontal theta
df_stats_frontal_theta = get_erspplot_and_stats(
    electrode_selection=[ "FCz", "FC1", "FC2", "Fz", "Cz"],
    freq_selection=(4, 6),
    stat_label="frontal_theta",
    timewin_stats=(-0.7, -0.3),
)

# Draw a pointplot to show theta as a function of three categorical factors
g = sns.catplot(
    data=df_stats_frontal_theta, x="stage", y="dB", hue="trajectory", col="group",
    capsize=.2, palette="rocket", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)

# Get frontal alpha
df_stats_frontal_alpha = get_erspplot_and_stats(
    electrode_selection=[ "FCz", "FC1", "FC2", "Fz", "Cz"],
    freq_selection=(10, 12),
    stat_label="frontal_alpha",
    timewin_stats=(-1.5, -1.2),
)

# Draw a pointplot to show theta as a function of three categorical factors
g = sns.catplot(
    data=df_stats_frontal_alpha, x="stage", y="dB", hue="trajectory", col="group",
    capsize=.2, palette="rocket", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)





# Get posterior alpha
df_stats_some_alpha = get_erspplot_and_stats(
    electrode_selection=[ "POz", "Oz", "Pz", "PO3", "PO4"],
    freq_selection=(8, 12),
    stat_label="posterior alpha",
    timewin_stats=(-0.8, -0.2),
)

# Draw a pointplot to show theta as a function of three categorical factors
g = sns.catplot(
    data=df_stats_some_alpha, x="stage", y="dB", hue="trajectory", col="group",
    capsize=.2, palette="rocket", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)

