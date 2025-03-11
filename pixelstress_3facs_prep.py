# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
import scipy.io
from joblib import dump

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_out = "/mnt/data_dump/pixelstress/3_condition_data/"

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

# Collector bin for all trials
df_sequences = []

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

    # Drop first x sequences
    x = 1
    idx_not_first_sequences = (df_erp.sequence_nr != x).values
    df_erp = df_erp[idx_not_first_sequences]
    erp_data = erp_data[idx_not_first_sequences, :, :]
    idx_not_first_sequences = (df_tf.sequence_nr != x).values
    df_tf = df_tf[idx_not_first_sequences]
    tf_data = tf_data[idx_not_first_sequences, :, :]

    # Get binned versions of feedback
    df_erp["feedback"] = pd.cut(
        df_erp["last_feedback_scaled"],
        bins=3,
        labels=["below", "close", "above"],
    )
    df_tf["feedback"] = pd.cut(
        df_tf["last_feedback_scaled"],
        bins=3,
        labels=["below", "close", "above"],
    )

    # Condition labels
    condition_labels = [
        "early_below",
        "early_close",
        "early_above",
        "late_below",
        "late_close",
        "late_above",
    ]

    # Get condition idx for erp
    idx_erp = [
        ((df_erp.feedback == "below") & (df_erp.block_nr <= 4)).values,
        ((df_erp.feedback == "close") & (df_erp.block_nr <= 4)).values,
        ((df_erp.feedback == "above") & (df_erp.block_nr <= 4)).values,
        ((df_erp.feedback == "below") & (df_erp.block_nr >= 5)).values,
        ((df_erp.feedback == "close") & (df_erp.block_nr >= 5)).values,
        ((df_erp.feedback == "above") & (df_erp.block_nr >= 5)).values,
    ]

    # Get condition idx for tf
    idx_tf = [
        ((df_tf.feedback == "below") & (df_tf.block_nr <= 4)).values,
        ((df_tf.feedback == "close") & (df_tf.block_nr <= 4)).values,
        ((df_tf.feedback == "above") & (df_tf.block_nr <= 4)).values,
        ((df_tf.feedback == "below") & (df_tf.block_nr >= 5)).values,
        ((df_tf.feedback == "close") & (df_tf.block_nr >= 5)).values,
        ((df_tf.feedback == "above") & (df_tf.block_nr >= 5)).values,
    ]
    
    # TODO: Check numbre of trials. What if zero?????

    # Time-frequency parameters
    n_freqs = 50
    tf_freqs = np.linspace(4, 30, n_freqs)
    tf_cycles = np.linspace(4, 12, n_freqs)

    # Iterate conditions
    for cond_nr, condition_label in enumerate(condition_labels):

        # Create condition epochs object for tf
        condition_epochs_tf = mne.EpochsArray(
            tf_data[idx_tf[cond_nr], :, :], info_tf, tmin=-2.4
        )

        # get dataframe for current condition for rt and accuracy
        df_behavior = df_tf[idx_tf[cond_nr]]

        # Get correct only
        idx_correct = (df_behavior.accuracy == 1).values
        df_correct = df_behavior[idx_correct]

        # get rt and accuracy
        rt = df_correct.rt.values.mean()
        acc = len(df_correct) / len(df_behavior)

        # tf-decomposition of all data (data is trial channels frequencies times)
        condition_tfr = (
            mne.time_frequency.tfr_morlet(
                condition_epochs_tf,
                tf_freqs,
                n_cycles=tf_cycles,
                average=True,
                return_itc=False,
                n_jobs=-2,
            )
            .apply_baseline((-1.9, -1.6), mode="logratio")
            .crop(tmin=-1.9, tmax=1)
            .decimate(decim=2)
        )

        # Create condition epochs object for tf
        condition_epochs_erp = mne.EpochsArray(
            erp_data[idx_erp[cond_nr], :, :], info_erp, tmin=-1.7
        )

        # Get erps
        condition_erp = condition_epochs_erp.average()

        # Get group variable
        if df_erp.session_condition.values[0] == 1:
            group = "experimental"
        else:
            group = "control"

        # Compile output
        condition_data = {
            "id": df_erp.id.values[0],
            "group": group,
            "stage": condition_label.split("_")[0],
            "feedback": condition_label.split("_")[1],
            "rt": rt,
            "accuracy": acc,
            "tfr": condition_tfr,
            "erp": condition_erp,
        }

        # Save result
        fn_out = os.path.join(
            path_out,
            "condition_data_"
            + str(df_erp.id.values[0])
            + "_"
            + condition_label
            + ".joblib",
        )
        dump(
            condition_data,
            fn_out,
        )
