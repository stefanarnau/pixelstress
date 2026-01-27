# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
import scipy.io
from joblib import dump
import sklearn.linear_model
from sklearn.preprocessing import MinMaxScaler

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_out = "/mnt/data_dump/pixelstress/3_2fac_data/"

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

    # Set RTs of non-correct to nan
    df.loc[df["accuracy"] != 1, "rt"] = np.nan

    # Remove trial difficulty confound using linear regression
    scaler = MinMaxScaler()
    df["trial_difficulty_scaled"] = scaler.fit_transform(df[["trial_difficulty"]])
    X = df[["trial_difficulty_scaled"]].values
    y = df["rt"].values
    mask = ~np.isnan(y)
    model = sklearn.linear_model.LinearRegression()
    model.fit(X[mask], y[mask])
    y_pred = model.predict(X)
    df["rt_residuals"] = y - y_pred
    df["rt_resint"] = y - y_pred + model.intercept_

    # Rename group column
    df.rename(columns={"session_condition": "group"}, inplace=True)
    df["group"] = df["group"].replace({1: "experimental", 2: "control"})

    # Reduce erp data to common trials
    mask = np.isin(df_erp["trial_nr_total"].values, to_keep)
    erp_data = erp_data[mask, :, :]

    # Reduce tf data to common trials
    mask = np.isin(df_tf["trial_nr_total"].values, to_keep)
    tf_data = tf_data[mask, :, :]

    # Get binned versions of feedback
    df["feedback"] = pd.cut(
        df["last_feedback_scaled"],
        bins=3,
        labels=["below", "close", "above"],
    )

    # Add variable trajectory
    df = df.assign(trajectory="close")
    df.loc[(df.block_wiggleroom == 1) & (df.block_outcome == -1), "trajectory"] = (
        "below"
    )
    df.loc[(df.block_wiggleroom == 1) & (df.block_outcome == 1), "trajectory"] = "above"

    # Identify and drop non-correct
    mask = ~np.isnan(df["rt"].values)
    df_correct = df.dropna(subset=["rt"])
    tf_data = tf_data[mask, :, :]
    erp_data = erp_data[mask, :, :]

    # Make grouped df
    df_grouped = (
        df.groupby(["trajectory", "id", "group"], observed=True)[
            ["rt", "rt_resint", "rt_residuals"]
        ]
        .agg(
            {
                "rt": ["mean", "std"],  # mean and standard deviation for rt
                "rt_resint": "mean",  # mean only
                "rt_residuals": "mean",  # mean only
            }
        )
        .reset_index()
    )

    # Flatten columns
    df_grouped.columns = ["_".join(col).strip("_") for col in df_grouped.columns.values]

    # Time-frequency parameters
    n_freqs = 40
    tf_freqs = np.linspace(4, 30, n_freqs)
    tf_cycles = np.linspace(6, 18, n_freqs)

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
        .decimate(decim=4)
    )

    # Get baseline indices
    idx_bl = (tf_data.times >= -1.9) & (tf_data.times <= -1.6)

    # Get average baseline values
    bl_values = tf_data._data[:, :, :, idx_bl].mean(axis=3).mean(axis=0)

    # Iterate conditions
    n_trials = []
    accuracies = []
    erps = []
    tfrs = []
    for row_idx, row in df_grouped.iterrows():

        # get indices
        idx_df = ((df.trajectory == row.trajectory)).values
        idx_df_correct = (
            (df_correct.trajectory == row.trajectory)
        ).values

        # Get number of trials
        n_trials.append(sum(idx_df_correct))

        # If no trial in erp data for sequence remains...
        if sum(idx_df_correct) == 0:

            accuracies.append(np.nan)
            erps.append(np.nan)
            tfrs.append(np.nan)
            continue

        # Get accuracy for condition
        accuracies.append(sum(idx_df_correct) / sum(idx_df))

        # Create condition epochs object for erp
        epochs_erp = mne.EpochsArray(
            erp_data[idx_df_correct, :, :], info_erp, tmin=-1.7
        )

        # Get condition erps
        erps.append(epochs_erp.average().decimate(4))

        # Get condition tf data and apply condition-general dB baseline
        condition_tf = tf_data[idx_df_correct].average()
        tmp = condition_tf._data
        for ch in range(bl_values.shape[0]):
            for fr in range(bl_values.shape[1]):
                tmp[ch, fr, :] = 10 * np.log10(
                    tmp[ch, fr, :].copy() / bl_values[ch, fr]
                )
        condition_tf._data = tmp.copy()
        tfrs.append(condition_tf)

    # Add to grouped df
    df_grouped["n_trials"] = n_trials
    df_grouped["accuracy"] = accuracies
    df_grouped["erps"] = erps
    df_grouped["tfrs"] = tfrs

    # Save result
    fn_out = os.path.join(
        path_out,
        "fac2_data_" + str(df.id.values[0]) + ".joblib",
    )
    dump(
        df_grouped,
        fn_out,
    )
