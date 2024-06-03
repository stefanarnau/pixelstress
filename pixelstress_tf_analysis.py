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
datasets = glob.glob(f"{path_in}/*tf.set")

# Create a montage
standard_1020_montage = mne.channels.make_standard_montage("standard_1020")


# Function for plotting erps and calculate stats
def get_erspplot_and_stats(
    electrode_selection, freq_selection, stat_label, timewin_stats
):

    # Get channel indices
    idx_channel = [
        idx
        for idx, element in enumerate(eeg_epochs.ch_names)
        if element in electrode_selection
    ]
    
    # Get frequency range indices
    idx_freqs = (tf_freqs >= freq_selection[0]) & (tf_freqs <= freq_selection[1])
    
    # A list for dataframe rows
    df_rows = []
    
    # Loop subjects
    for idx_id, id in enumerate(ids):

        # Loop time points
        for idx_t, t in enumerate(tf_times):

            # Set stat-flag
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
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-1.2, -1))

    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "tf_trialinfo.csv")

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
    )
    ersp_below_early = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_below_early],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )
    ersp_above_early = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_above_early],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )
    ersp_close_late = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_close_late],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )
    ersp_below_late = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_below_late],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )
    ersp_above_late = mne.time_frequency.tfr_morlet(
        eeg_epochs[idx_above_late],
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )

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

# Get ERSP for Fz
df_stats_Fz = get_erspplot_and_stats(
    electrode_selection=["POz", "Pz", "PO1", "PO2", "P1", "P2"],
    freq_selection=(8, 12),
    stat_label="alpha",
    timewin_stats=(-0.4, 0),
)

aa=bb





# Grand averages considering group factor
ersps_close_exp = mne.grand_average(
    [x for i, x in enumerate(ersps_close) if group[i] == 1]
)
ersps_close_cnt = mne.grand_average(
    [x for i, x in enumerate(ersps_close) if group[i] == 2]
)
ersps_below_exp = mne.grand_average(
    [x for i, x in enumerate(ersps_below) if group[i] == 1]
)
ersps_below_cnt = mne.grand_average(
    [x for i, x in enumerate(ersps_below) if group[i] == 2]
)
ersps_above_exp = mne.grand_average(
    [x for i, x in enumerate(ersps_above) if group[i] == 1]
)
ersps_above_cnt = mne.grand_average(
    [x for i, x in enumerate(ersps_above) if group[i] == 2]
)


# Plot TF-plots
timefreqs = [(-0.8, 10), (-0.2, 10), (0.5, 10), (0.2, 5.5), (-0.6, 5.5)]
vmin, vmax = -4, 4
cmap = "Spectral_r"
topomap_args = dict(sensors=False, vmin=vmin, vmax=vmax)

ersps_close_exp.plot_joint(
    timefreqs=timefreqs,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    topomap_args=topomap_args,
    show=False,
)
plt.savefig(
    os.path.join(path_plot, "ersps_close_exp.png"),
    dpi=300,
    transparent=True,
)

ersps_close_cnt.plot_joint(
    timefreqs=timefreqs,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    topomap_args=topomap_args,
    show=False,
)
plt.savefig(
    os.path.join(path_plot, "ersps_close_cnt.png"),
    dpi=300,
    transparent=True,
)

ersps_below_exp.plot_joint(
    timefreqs=timefreqs,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    topomap_args=topomap_args,
    show=False,
)
plt.savefig(
    os.path.join(path_plot, "ersps_below_exp.png"),
    dpi=300,
    transparent=True,
)

ersps_below_cnt.plot_joint(
    timefreqs=timefreqs,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    topomap_args=topomap_args,
    show=False,
)
plt.savefig(
    os.path.join(path_plot, "ersps_below_cnt.png"),
    dpi=300,
    transparent=True,
)

ersps_above_exp.plot_joint(
    timefreqs=timefreqs,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    topomap_args=topomap_args,
    show=False,
)
plt.savefig(
    os.path.join(path_plot, "ersps_above_exp.png"),
    dpi=300,
    transparent=True,
)

ersps_above_cnt.plot_joint(
    timefreqs=timefreqs,
    vmin=vmin,
    vmax=vmax,
    cmap=cmap,
    topomap_args=topomap_args,
    show=False,
)
plt.savefig(
    os.path.join(path_plot, "ersps_above_cnt.png"),
    dpi=300,
    transparent=True,
)


# Avergae frontal theta
idx_subject = np.array(group) == 1
idx_freqs = (ersp_close.freqs >= 4) & (ersp_close.freqs <= 7)
idx_channel = np.array([7, 8, 64])  # FC1, FC2, FCz
frontal_theta_close_exp = ersps_matrices_close[idx_subject, :, :, :][
    :, :, idx_freqs, :
][:, :, :, idx_channel].mean(axis=(2, 3))
frontal_theta_close_cnt = ersps_matrices_close[idx_subject, :, :, :][
    :, :, idx_freqs, :
][:, :, :, idx_channel].mean(axis=(2, 3))
frontal_theta_below_exp = ersps_matrices_below[idx_subject, :, :, :][
    :, :, idx_freqs, :
][:, :, :, idx_channel].mean(axis=(2, 3))
frontal_theta_below_cnt = ersps_matrices_below[idx_subject, :, :, :][
    :, :, idx_freqs, :
][:, :, :, idx_channel].mean(axis=(2, 3))
frontal_theta_above_exp = ersps_matrices_above[idx_subject, :, :, :][
    :, :, idx_freqs, :
][:, :, :, idx_channel].mean(axis=(2, 3))
frontal_theta_above_cnt = ersps_matrices_above[idx_subject, :, :, :][
    :, :, idx_freqs, :
][:, :, :, idx_channel].mean(axis=(2, 3))


# id group trajectory time value
timewin_stats = (0.15, 0.25)
idx_channel = np.array([3, 4, 5, 8, 9, 64])
idx_freqs = (ersp_close.freqs >= 4) & (ersp_close.freqs <= 7)
df_frontal_theta = pd.DataFrame(
    columns=["id", "group", "trajectory", "time (s)", "in_statwin", "theta (dB)"]
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
                "theta (dB)": ersps_matrices_close[idx_id, idx_t, :, :][idx_freqs, :][
                    :, idx_channel
                ].mean(axis=(0, 1)),
            },
            index=[0],
        )
        df_frontal_theta = pd.concat([df_frontal_theta, entry], ignore_index=True)

        entry = pd.DataFrame(
            {
                "id": id,
                "group": ["experimental", "control"][group[idx_id] - 1],
                "trajectory": "below",
                "time (s)": ersp_close.times[idx_t],
                "in_statwin": in_statwin,
                "theta (dB)": ersps_matrices_below[idx_id, idx_t, :, :][idx_freqs, :][
                    :, idx_channel
                ].mean(axis=(0, 1)),
            },
            index=[0],
        )
        df_frontal_theta = pd.concat([df_frontal_theta, entry], ignore_index=True)

        entry = pd.DataFrame(
            {
                "id": id,
                "group": ["experimental", "control"][group[idx_id] - 1],
                "trajectory": "above",
                "time (s)": ersp_close.times[idx_t],
                "in_statwin": in_statwin,
                "theta (dB)": ersps_matrices_above[idx_id, idx_t, :, :][idx_freqs, :][
                    :, idx_channel
                ].mean(axis=(0, 1)),
            },
            index=[0],
        )
        df_frontal_theta = pd.concat([df_frontal_theta, entry], ignore_index=True)

# Set color palette
sns.set_palette("Set2")

# Lineplot frontal theta
sns.lineplot(
    data=df_frontal_theta, x="time (s)", y="theta (dB)", hue="trajectory", style="group"
).set_title("frontal theta [F3, Fz, F4, FC1, FCz, FC2]")
plt.savefig(
    os.path.join(path_plot, "lineplot_theta.png"),
    dpi=300,
    transparent=True,
)


# Get df for anovas
df_stat_theta = df_frontal_theta.drop(
    df_frontal_theta[df_frontal_theta.in_statwin != 1].index
)
df_stat_theta = (
    df_stat_theta.groupby(["id", "group", "trajectory"])["theta (dB)"]
    .mean()
    .reset_index()
)

# Mixed anova
aov_frontal_theta = pg.mixed_anova(
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
