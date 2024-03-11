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

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp.set")

# Loop datasets
for dataset in datasets:

    # Load a dataset
    epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-1.2, -1.0))
    
    # Load trialinfo
    trialinfo = pd.read_csv(dataset.split("cleaned_")[0] + "erp_trialinfo.csv")
    
    # Get data as trial x channel x times
    eeg_data_3d = epochs.get_data()
    
    # Exclude trials of first sequence, as there is no previous feedback
    idx_drop = list(np.where(trialinfo["sequence_nr"] == 1)[0])
    trialinfo.drop(idx_drop, axis=0, inplace=True)
    eeg_data_3d = np.delete(eeg_data_3d, idx_drop, axis=0)
    
    # Collectors
    X = []
    y = []
    
    # Iterate sequences n blocks
    for block in list(set(trialinfo["block_nr"])):
        for seq in list(set(trialinfo["sequence_nr"])):
            
            # Get indices for block-sequence combination
            idx = list(np.where((trialinfo["block_nr"] == block) & (trialinfo["sequence_nr"] == seq))[0])
            
            # Get erp for sequence
            X.append(eeg_data_3d[idx, :, :].mean(axis=0))
            
            #
            

    
    # Get dims
    n_trial, n_channel, n_time = eeg_data_3d.shape
    
    # Reshape to 2d
    X = eeg_data_3d.reshape((n_trial, n_channel * n_time))
    
    # Regression predictors:
    # block_nr (to control for time on task and order effects)
    # "sequence_nr" (to capture effects of deadline imminence)
    # "last_feedback_scaled" (to capture effects of distance to performance target)
    # "sequence nr" * "last feedback scaled" (to capture interaction of distance to performance target and deadline imminence)
    
    # Create design matrix
    y = trialinfo[["block_nr", "sequence_nr", "last_feedback_scaled"]].to_numpy() 
    
    # Add interaction as product to design matrix
    y = np.concatenate((y, np.multiply(trialinfo[["sequence_nr"]].to_numpy(), trialinfo[["last_feedback_scaled"]].to_numpy())), axis=1)
    
    # A scaler
    scaler = sklearn.preprocessing.StandardScaler()
    
    # Scale data
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    
    
    # A regressor
    regr = sklearn.linear_model.LinearRegression()
    
    # Fit regressor
    regr.fit(X, y)
    
    # Get coefs
    coef = regr.coef_
    
    # Re-reshape coefs to 3d
    coef_block_nr = coef[0, :].reshape((n_channel, n_time))
    coef_sequence_nr = coef[1, :].reshape((n_channel, n_time))
    coef_last_feedback_scaled = coef[2, :].reshape((n_channel, n_time))
    coef_interaction_feedback_sequence = coef[3, :].reshape((n_channel, n_time))
    
    # TODO: Scaler, Regression, re-reshape, cluster permutation test against zero.
    

    
    
    aa=bb