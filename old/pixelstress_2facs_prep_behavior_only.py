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
path_out = "/mnt/data_dump/pixelstress/3_2fac_data_behavior/"

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
    idx_not_first_sequences = (df_erp.sequence_nr > 0).values
    df_erp = df_erp[idx_not_first_sequences]
    erp_data = erp_data[idx_not_first_sequences, :, :]

    # Drop first sequences from tf data
    idx_not_first_sequences = (df_tf.sequence_nr > 0).values
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
    df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == -1)] = "below"
    df.trajectory[(df.block_wiggleroom == 1) & (df.block_outcome == 1)] = "above"

    # Identify and drop non-correct
    mask = ~np.isnan(df["rt"].values)
    df_correct = df.dropna(subset=["rt"])
    tf_data = tf_data[mask, :, :]
    erp_data = erp_data[mask, :, :]

    # Make grouped df
    df_grouped = (
        df.groupby(["trajectory", "id", "group"])["rt", "rt_residuals", "rt_resint"]
        .mean()
        .reset_index()
    )
    
    # Save result
    fn_out = os.path.join(
        path_out,
        "fac2_data_behavior_" + str(df.id.values[0]) + ".joblib",
    )
    dump(
        df_grouped,
        fn_out,
    )
