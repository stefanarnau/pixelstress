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
import matplotlib.pyplot as plt

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*tf.set")

# Collectors for coefs
coefs_time_left = []
coefs_feedback = []
coefs_interaction = []

# Loop datasets
for dataset in datasets:

    # Load a dataset
    eeg_epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-1.2, -1.0))
    
    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "tf_trialinfo.csv")
    
    # Perform single trial time-frequency analysis
    n_freqs = 20
    tf_freqs = np.linspace(2, 20, n_freqs)
    tf_cycles = np.linspace(3, 12, n_freqs)
    tf_epochs = mne.time_frequency.tfr_morlet(
        eeg_epochs,
        tf_freqs,
        n_cycles=tf_cycles,
        average=False,
        return_itc=False,
        n_jobs=-2,
        decim=4,
    )

    # Apply baseline
    #tf_epochs.apply_baseline((-1.5, -1.2), mode='zscore', verbose=None)

    # Save info object for plotting topos
    info_object = tf_epochs.info

    # Prune in time
    to_keep_idx = (tf_epochs.times >= -1.5) & (tf_epochs.times <= 0.5)
    tf_times = tf_epochs.times[to_keep_idx]
    tf_data = tf_epochs.data[:, :, :, to_keep_idx]
    
    # Exclude trials of first sequence, as there is no previous feedback
    idx_drop = list(np.where(trialinfo["sequence_nr"] == 1)[0])
    trialinfo.drop(idx_drop, axis=0, inplace=True)
    trialinfo.reset_index(inplace=True)
    tf_data = np.delete(tf_data, idx_drop, axis=0)
    
    # Get dims
    n_trial, n_channel, n_freqs, n_time = tf_data.shape
    
    # Impose dependency tot
    #for t in range(trialinfo.shape[0]):
    #    tf_data[t, 0:5, 0:10, :] = trialinfo["trial_nr_total"][t]
        
    
    # Reshape to 2d
    X = tf_data.reshape((n_trial, n_channel * n_freqs * n_time))
    
    # Create design matrix
    y = trialinfo[["trial_nr_total",          # Control for time on task
                   "trial_nr",                # Control fer effects within sequence
                   "sequence_difficulty",     # Control for possible confound with difficulty
                   "sequence_nr",             # Time left until deadline
                   "last_feedback_scaled",    # Performance in relation to target
                   ]].to_numpy() 
    
    # A scaler
    scaler = sklearn.preprocessing.StandardScaler()
    
    # Scale data
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    
    # Add interaction as product to design matrix
    y = np.concatenate((y, np.multiply(trialinfo[["sequence_nr"]].to_numpy(), trialinfo[["last_feedback_scaled"]].to_numpy())), axis=1)
    
    # A regressor
    regr = sklearn.linear_model.ElasticNet()
    
    # Fit regressor
    regr.fit(X, y)
    
    # Get coefs
    coef = regr.coef_
    
    # Re-reshape coefs to 3d
    coefs_time_left.append(coef[3, :].reshape((n_channel, n_freqs, n_time)))
    coefs_feedback.append(coef[4, :].reshape((n_channel, n_freqs, n_time)))
    coefs_interaction.append(coef[5, :].reshape((n_channel, n_freqs, n_time)))


    aa=bb
    plt_data = np.stack(coefs_interaction).mean(axis=0)[64, :, :]
    plt.contourf(tf_times, tf_freqs, plt_data, vmin = -0.001, vmax = 0.001)
    plt.colorbar()


