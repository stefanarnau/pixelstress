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

# Collector lists
ersps_close = []
ersps_below = []
ersps_above = []
ersps_matrices_close = []
ersps_matrices_below = []
ersps_matrices_above = []
group = []
ids = []

# Loop datasets
for dataset in datasets:

    # Get id
    ids.append(int(dataset.split("/")[-1].split("_")[1]))

    # Load a dataset
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-1.2, -1))

    # Set montage
    eeg_epochs.set_montage(standard_1020_montage)

    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "tf_trialinfo.csv")

    # Get trial indices of conditions
    idx_close = np.where(
        (trialinfo.sequence_nr >= 7) & (trialinfo.block_wiggleroom == 0)
    )[0]
    idx_below = np.where(
        (trialinfo.sequence_nr >= 7)
        & (trialinfo.block_wiggleroom == 1)
        & (trialinfo.block_outcome == -1)
    )[0]
    idx_above = np.where(
        (trialinfo.sequence_nr >= 7)
        & (trialinfo.block_wiggleroom == 1)
        & (trialinfo.block_outcome == 1)
    )[0]

    # Random downsample trials
    min_n = np.min([len(idx_close), len(idx_above), len(idx_below)])

    # Get condition epochs
    epochs_close = eeg_epochs[idx_close]
    epochs_below = eeg_epochs[idx_below]
    epochs_above = eeg_epochs[idx_above]

    # Perform time-frequency decomposition
    n_freqs = 25
    tf_freqs = np.linspace(3, 20, n_freqs)
    tf_cycles = np.linspace(6, 12, n_freqs)

    ersp_close = mne.time_frequency.tfr_morlet(
        epochs_close,
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )

    ersp_below = mne.time_frequency.tfr_morlet(
        epochs_below,
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )

    ersp_above = mne.time_frequency.tfr_morlet(
        epochs_above,
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=False,
        n_jobs=-2,
        decim=2,
    )

    # Get baseline indices
    idx_bl = (ersp_close.times >= -1.5) & (ersp_close.times <= -1.2)

    # Average baseline values
    bl_values = (ersp_close._data + ersp_below._data + ersp_above._data) / 3
    bl_values = bl_values[:, :, idx_bl].mean(axis=2)

    # Apply condition general dB baseline
    for ch in range(bl_values.shape[0]):
        for fr in range(bl_values.shape[1]):
            ersp_close._data[ch, fr, :] = 10 * np.log10(
                ersp_close._data[ch, fr, :].copy() / bl_values[ch, fr]
            )
            ersp_below._data[ch, fr, :] = 10 * np.log10(
                ersp_below._data[ch, fr, :].copy() / bl_values[ch, fr]
            )
            ersp_above._data[ch, fr, :] = 10 * np.log10(
                ersp_above._data[ch, fr, :].copy() / bl_values[ch, fr]
            )

    # Apply baseline and crop
    ersp_close.crop(tmin=-1.5, tmax=1)
    ersp_below.crop(tmin=-1.5, tmax=1)
    ersp_above.crop(tmin=-1.5, tmax=1)

    # Collect
    ersps_close.append(ersp_close)
    ersps_below.append(ersp_below)
    ersps_above.append(ersp_above)
    group.append(trialinfo.session_condition[0])

    # Collect as matrices
    ersps_matrices_close.append(np.transpose(ersp_close.data, (2, 1, 0)))
    ersps_matrices_below.append(np.transpose(ersp_below.data, (2, 1, 0)))
    ersps_matrices_above.append(np.transpose(ersp_above.data, (2, 1, 0)))

# Stack matrices
ersps_matrices_close = np.stack(ersps_matrices_close)
ersps_matrices_below = np.stack(ersps_matrices_below)
ersps_matrices_above = np.stack(ersps_matrices_above)

# Define adjacency
adjacency, channel_names = mne.channels.find_ch_adjacency(
    ersps_close[0].info, ch_type="eeg"
)

# Plot adjacency
mne.viz.plot_ch_adjacency(ersps_close[0].info, adjacency, channel_names)

# Define adjacency in tf-sensor-space
tfs_adjacency = mne.stats.combine_adjacency(
    len(ersps_close[0].freqs), len(ersps_close[0].times), adjacency
)

# We are running an F test, so we look at the upper tail
# see also: https://stats.stackexchange.com/a/73993
tail = 1

# We want to set a critical test statistic (here: F), to determine when
# clusters are being formed. Using Scipy's percent point function of the F
# distribution, we can conveniently select a threshold that corresponds to
# some alpha level that we arbitrarily pick.
alpha_cluster_forming = 0.1

# For an F test we need the degrees of freedom for the numerator
# (number of conditions - 1) and the denominator (number of observations
# - number of conditions):
n_conditions = 2
n_observations = len(datasets)
df_effect = n_conditions - 1
df_error = n_observations - n_conditions

# Note: we calculate 1 - alpha_cluster_forming to get the critical value
# on the right tail
f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=df_effect, dfd=df_error)

# Run the cluster based permutation analysis
# cluster_stats_trajectory = mne.stats.spatio_temporal_cluster_test(
#     [ersps_matrices_close, ersps_matrices_below, ersps_matrices_above],
#     n_permutations=1000,
#     threshold=f_thresh,
#     tail=tail,
#     n_jobs=-2,
#     buffer_size=None,
#     adjacency=tfs_adjacency,
#     out_type="mask",
#     seed=4,
# )
# F_obs_trajectory, clusters_trajectory, p_values_trajectory, _ = cluster_stats_trajectory

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
idx_channel = np.array([3,4,5,8,9,64])
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
