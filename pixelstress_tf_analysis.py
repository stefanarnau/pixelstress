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

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*tf.set")

# Create a montage
standard_1020_montage = mne.channels.make_standard_montage("standard_1020")

# Collector lists
ersps_total = []
ersps_close = []
ersps_below = []
ersps_above = []
itpcs_total = []
itpcs_close = []
itpcs_below = []
itpcs_above = []

ersps_matrices_total = []
ersps_matrices_close = []
ersps_matrices_below = []
ersps_matrices_above = []
itpcs_matrices_total = []
itpcs_matrices_close = []
itpcs_matrices_below = []
itpcs_matrices_above = []

group = []

# Loop datasets
for dataset in datasets:

    # Load a dataset
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-1.2, -1.0))
    
    # Set montage
    eeg_epochs.set_montage(standard_1020_montage)
    
    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "tf_trialinfo.csv")
    
    # Get trial indices of conditions
    idx_total = np.where(trialinfo.sequence_nr >= 7)[0]
    idx_close = np.where((trialinfo.sequence_nr >= 7) & (trialinfo.block_wiggleroom == 0))[0]
    idx_below = np.where((trialinfo.sequence_nr >= 7) & (trialinfo.block_wiggleroom == 1) & (trialinfo.block_outcome == -1))[0]
    idx_above = np.where((trialinfo.sequence_nr >= 7) & (trialinfo.block_wiggleroom == 1) & (trialinfo.block_outcome == 1))[0]
    
    # Random downsample trials
    min_n = np.min([len(idx_close), len(idx_above), len(idx_below)])
    
    # Get condition epochs
    epochs_total = eeg_epochs[idx_total]
    epochs_close = eeg_epochs[idx_close]
    epochs_below = eeg_epochs[idx_below]
    epochs_above = eeg_epochs[idx_above]
    
    # Perform time-frequency decomposition
    n_freqs = 25
    tf_freqs = np.linspace(3, 20, n_freqs)
    tf_cycles = np.linspace(6, 12, n_freqs)
    
    ersp_total, itpc_total = mne.time_frequency.tfr_morlet(
        epochs_total,
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=True,
        n_jobs=-2,
        decim=2,
    )
        
    ersp_close, itpc_close = mne.time_frequency.tfr_morlet(
        epochs_close,
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=True,
        n_jobs=-2,
        decim=2,
    )
    
    ersp_below, itpc_below = mne.time_frequency.tfr_morlet(
        epochs_below,
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=True,
        n_jobs=-2,
        decim=2,
    )
        
    ersp_above, itpc_above = mne.time_frequency.tfr_morlet(
        epochs_above,
        tf_freqs,
        n_cycles=tf_cycles,
        average=True,
        return_itc=True,
        n_jobs=-2,
        decim=2,
    )
    
    # Apply baseline and crop
    ersp_total.apply_baseline((-1.5, -1.2), mode='logratio', verbose=None).crop(tmin=-1.5,tmax=1)
    ersp_close.apply_baseline((-1.5, -1.2), mode='logratio', verbose=None).crop(tmin=-1.5,tmax=1)
    ersp_below.apply_baseline((-1.5, -1.2), mode='logratio', verbose=None).crop(tmin=-1.5,tmax=1)
    ersp_above.apply_baseline((-1.5, -1.2), mode='logratio', verbose=None).crop(tmin=-1.5,tmax=1)
    itpc_total.crop(tmin=-1.5,tmax=1)
    itpc_close.crop(tmin=-1.5,tmax=1)
    itpc_below.crop(tmin=-1.5,tmax=1)
    itpc_above.crop(tmin=-1.5,tmax=1)
    
    # Collect
    ersps_total.append(ersp_total)
    ersps_close.append(ersp_close)
    ersps_below.append(ersp_below)
    ersps_above.append(ersp_above)
    itpcs_total.append(itpc_total)
    itpcs_close.append(itpc_close)
    itpcs_below.append(itpc_below)
    itpcs_above.append(itpc_above)
    group.append(trialinfo.session_condition[0])
    
    # Collect as matrices
    ersps_matrices_total.append(np.transpose(ersp_total.data, (2, 1, 0)))
    ersps_matrices_close.append(np.transpose(ersp_close.data, (2, 1, 0)))
    ersps_matrices_below.append(np.transpose(ersp_below.data, (2, 1, 0)))
    ersps_matrices_above.append(np.transpose(ersp_above.data, (2, 1, 0)))
    itpcs_matrices_total.append(np.transpose(itpc_total.data, (2, 1, 0)))
    itpcs_matrices_close.append(np.transpose(itpc_close.data, (2, 1, 0)))
    itpcs_matrices_below.append(np.transpose(itpc_below.data, (2, 1, 0)))
    itpcs_matrices_above.append(np.transpose(itpc_above.data, (2, 1, 0)))

# Stack matrices
ersps_matrices_total = np.stack(ersps_matrices_total)
ersps_matrices_close = np.stack(ersps_matrices_close)
ersps_matrices_below = np.stack(ersps_matrices_below)
ersps_matrices_above = np.stack(ersps_matrices_above)
itpcs_matrices_total = np.stack(itpcs_matrices_total)
itpcs_matrices_close = np.stack(itpcs_matrices_close)
itpcs_matrices_below = np.stack(itpcs_matrices_below)
itpcs_matrices_above = np.stack(itpcs_matrices_above)

# Define adjacency
adjacency, channel_names = mne.channels.find_ch_adjacency(ersps_close[0].info, ch_type="eeg")

# Plot adjacency
mne.viz.plot_ch_adjacency(ersps_close[0].info, adjacency, channel_names)

# Define adjacency in tf-sensor-space
tfs_adjacency = mne.stats.combine_adjacency(len(ersps_close[0].freqs), len(ersps_close[0].times), adjacency)

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
cluster_stats_trajectory = mne.stats.spatio_temporal_cluster_test(
    [itpcs_matrices_close, itpcs_matrices_below, itpcs_matrices_above],
    n_permutations=1000,
    threshold=f_thresh,
    tail=tail,
    n_jobs=-2,
    buffer_size=None,
    adjacency=tfs_adjacency,
    out_type="mask",
    seed=4,
)
F_obs_trajectory, clusters_trajectory, p_values_trajectory, _ = cluster_stats_trajectory



























a = mne.grand_average(itpcs_close)
a.plot_joint()

b = mne.grand_average(itpcs_below)
b.plot_joint()


c = mne.grand_average(itpcs_above)
c.plot_joint()























