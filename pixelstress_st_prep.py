# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
import scipy.io
from joblib import dump
import sklearn.linear_model

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_out = "/mnt/data_dump/pixelstress/3_st_data/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp.set")

# Load channel labels
channel_labels = (
    open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r")
    .read()
    .split("\n")[:-1]
)

# Create info
info_tf = mne.create_info(channel_labels, 200, ch_types="eeg", verbose=None)
info_erp = mne.create_info(channel_labels, 1000, ch_types="eeg", verbose=None)

# Collect datasets
for dataset in datasets:

    # Load stuff =========================================================================================================

    # Read erp trialinfo
    df_erp = pd.read_csv(dataset.split("_cleaned")[0] + "_erp_trialinfo.csv")

    # Load erp data as trials x channel x times
    erp_data = np.transpose(scipy.io.loadmat(dataset)["data"], [2, 0, 1])

    # Get erp times
    erp_times = scipy.io.loadmat(dataset)["times"].ravel()

    # Read tf trialinfo
    df_tf = pd.read_csv(dataset.split("_cleaned")[0] + "_tf_trialinfo.csv")

    # Load tf eeg data
    tf_data = np.transpose(
        scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")["data"], [2, 0, 1]
    )

    # Load tf times
    tf_times = scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")[
        "times"
    ].ravel()

    # Drop first sequences from erp data
    idx_not_first_sequences = (df_erp.sequence_nr > 1).values
    df_erp = df_erp[idx_not_first_sequences]
    erp_data = erp_data[idx_not_first_sequences, :, :]

    # Drop first sequences from tf data
    idx_not_first_sequences = (df_tf.sequence_nr > 1).values
    df_tf = df_tf[idx_not_first_sequences]
    tf_data = tf_data[idx_not_first_sequences, :, :]

    # Get common trials
    to_keep = np.intersect1d(df_erp.trial_nr_total.values, df_tf.trial_nr_total.values)

    # Get df of common trials
    df = df_erp[df_erp["trial_nr_total"].isin(to_keep)]

    # Rename group column
    df.rename(columns={"session_condition": "group"}, inplace=True)

    # Reduce erp data to common trials
    mask = np.isin(df_erp["trial_nr_total"].values, to_keep)
    erp_data = erp_data[mask, :, :]

    # Reduce tf data to common trials
    mask = np.isin(df_tf["trial_nr_total"].values, to_keep)
    tf_data = tf_data[mask, :, :]

    # Get binned versions of feedback
    df["feedback_binned"] = pd.cut(
        df["last_feedback_scaled"],
        bins=3,
        labels=["below", "close", "above"],
    )

    # Add variable trajectory
    df = df.assign(trajectory="close")
    df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == -1)] = "below"
    df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == 1)] = "above"

    # Time-frequency parameters
    n_freqs = 30
    tf_freqs = np.linspace(4, 30, n_freqs)
    tf_cycles = np.linspace(6, 12, n_freqs)

    # Create epochs object for tf
    tf_data = mne.EpochsArray(tf_data, info_tf, tmin=-2.4)

    # tf-decomposition of data
    tf_data = (
        mne.time_frequency.tfr_morlet(
            tf_data,
            tf_freqs,
            n_cycles=tf_cycles,
            average=False,
            return_itc=False,
            n_jobs=-2,
        )
        .crop(tmin=-1.9, tmax=1)
        .decimate(decim=2)
    ).apply_baseline((-1.9, -1.6), mode="mean", verbose=None)

    # Create epochs object for erp
    erp_data = mne.EpochsArray(erp_data, info_erp, tmin=-1.7)

    # Get erp measures (CNV)
    time_idx = (erp_data.times >= -0.5) & (erp_data.times <= 0)
    cnv_F = (
        erp_data.copy()
        .pick(["Fz", "F1", "F2"])
        ._data.mean(axis=1)[:, time_idx]
        .mean(axis=1)
        .tolist()
    )
    cnv_FC = (
        erp_data.copy()
        .pick(["FCz", "FC1", "FC2"])
        ._data.mean(axis=1)[:, time_idx]
        .mean(axis=1)
        .tolist()
    )
    cnv_C = (
        erp_data.copy()
        .pick(["Cz", "C1", "C2"])
        ._data.mean(axis=1)[:, time_idx]
        .mean(axis=1)
        .tolist()
    )

    # Save to df
    df["cnv_F"] = cnv_F
    df["cnv_FC"] = cnv_FC
    df["cnv_C"] = cnv_C

    # Set tf indices
    time_idx = (tf_data.times >= -1) & (tf_data.times <= -0.2)
    theta_idx = (tf_data.freqs >= 4) & (tf_data.freqs <= 6)
    alpha_idx = (tf_data.freqs >= 8) & (tf_data.freqs <= 12)
    beta_idx = (tf_data.freqs >= 18) & (tf_data.freqs <= 30)

    # Get region tf data
    frontal_tf_data = (
        tf_data.copy().pick(["Fz", "FC1", "FC2", "FCz", "F1", "F2"])._data.mean(axis=1)
    )
    central_tf_data = tf_data.copy().pick(["Cz", "C1", "C2"])._data.mean(axis=1)
    posterior_tf_data = (
        tf_data.copy().pick(["Pz", "P1", "P2", "POz", "PO3", "PO4"])._data.mean(axis=1)
    )

    # Get tf measures
    frontal_theta = (
        frontal_tf_data[:, theta_idx, :].mean(axis=1)[:, time_idx].mean(axis=1).tolist()
    )
    frontal_alpha = (
        frontal_tf_data[:, alpha_idx, :].mean(axis=1)[:, time_idx].mean(axis=1).tolist()
    )
    frontal_beta = (
        frontal_tf_data[:, beta_idx, :].mean(axis=1)[:, time_idx].mean(axis=1).tolist()
    )
    central_theta = (
        central_tf_data[:, theta_idx, :].mean(axis=1)[:, time_idx].mean(axis=1).tolist()
    )
    central_alpha = (
        central_tf_data[:, alpha_idx, :].mean(axis=1)[:, time_idx].mean(axis=1).tolist()
    )
    central_beta = (
        central_tf_data[:, beta_idx, :].mean(axis=1)[:, time_idx].mean(axis=1).tolist()
    )
    posterior_theta = (
        posterior_tf_data[:, theta_idx, :]
        .mean(axis=1)[:, time_idx]
        .mean(axis=1)
        .tolist()
    )
    posterior_alpha = (
        posterior_tf_data[:, alpha_idx, :]
        .mean(axis=1)[:, time_idx]
        .mean(axis=1)
        .tolist()
    )
    posterior_beta = (
        posterior_tf_data[:, beta_idx, :]
        .mean(axis=1)[:, time_idx]
        .mean(axis=1)
        .tolist()
    )

    # Save to df
    df["frontal_theta"] = frontal_theta
    df["frontal_alpha"] = frontal_alpha
    df["frontal_beta"] = frontal_beta
    df["central_theta"] = central_theta
    df["central_alpha"] = central_alpha
    df["central_beta"] = central_beta
    df["posterior_theta"] = posterior_theta
    df["posterior_alpha"] = posterior_alpha
    df["posterior_beta"] = posterior_beta

    # Save result
    fn_out = os.path.join(
        path_out,
        "st_data_" + str(df.id.values[0]) + ".joblib",
    )
    dump(
        df,
        fn_out,
    )
