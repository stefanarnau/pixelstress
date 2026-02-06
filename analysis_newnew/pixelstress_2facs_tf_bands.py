# Imports
import glob
import numpy as np
import pandas as pd
import mne
import scipy.io
import scipy.stats
from sklearn.preprocessing import StandardScaler
import mne.stats
import matplotlib.pyplot as plt
import seaborn as sns

# Path things
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
datasets = glob.glob(f"{path_in}/*erp.set")

ids_to_drop = {1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83}
min_trials_per_sequence = 3

# Load channel labels + create info + set montage + calculate adjacency
channel_labels = (
    open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r")
    .read()
    .split("\n")[:-1]
)
sfreq = 200.0
info_tf = mne.create_info(channel_labels, sfreq, ch_types="eeg", verbose=None)
montage = mne.channels.make_standard_montage("standard_1020")  # or "standard_1005"
info_tf.set_montage(montage, on_missing="warn", match_case=False)
adjacency, ch_names = mne.channels.find_ch_adjacency(info_tf, ch_type="eeg")


# Loop datasets
for dataset in datasets:
    base = dataset.split("_cleaned")[0]

    # Trialinfo
    df_erp = pd.read_csv(base + "_erp_trialinfo.csv")
    df_tf = pd.read_csv(base + "_tf_trialinfo.csv")

    subj_id = int(df_erp["id"].iloc[0])
    if subj_id in ids_to_drop:
        continue

    # Load tf eeg data as trials x channles x times
    tf_data = np.transpose(
        scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")["data"], [2, 0, 1]
    )

    # Load tf times
    tf_times = scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")[
        "times"
    ].ravel()

    # Determine time units for tmin (heuristic: EEGLAB typically ms)
    # If times look like [-1000..2000], that's ms; if [-1..2], that's seconds.
    if np.nanmax(np.abs(tf_times)) > 20:
        tmin = tf_times[0] / 1000.0
    else:
        tmin = tf_times[0]

    # Common trials between ERP and TF
    to_keep = np.intersect1d(
        df_erp["trial_nr_total"].values, df_tf["trial_nr_total"].values
    )

    # Reduce metadata to common trials (copy)
    df = df_tf[df_tf["trial_nr_total"].isin(to_keep)].copy()

    # Reduce TF data to those common trials
    mask_common = np.isin(df_tf["trial_nr_total"].values, to_keep)
    tf_data = tf_data[mask_common, :, :]

    # Basic checks
    if tf_data.shape[0] != len(df):
        raise RuntimeError(
            f"Trial alignment mismatch for subject {subj_id}: "
            f"tf_data {tf_data.shape[0]} vs df {len(df)}"
        )

    # Binarize accuracy
    df["accuracy"] = (df["accuracy"] == 1).astype(int)

    # Group coding
    df = df.rename(columns={"session_condition": "group"})
    df["group"] = df["group"].replace({1: "experimental", 2: "control"})

    # Remove first sequences
    mask = df["sequence_nr"] > 1
    df = df.loc[mask].reset_index(drop=True)
    tf_data = tf_data[mask.to_numpy(), :, :]

    # Keep only correct trials
    mask = df["accuracy"] == 1
    df = df.loc[mask].reset_index(drop=True)
    tf_data = tf_data[mask.to_numpy(), :, :]

    # Se tf-decomposition parameters
    freqs = np.arange(4, 31, 2)
    n_cycles = freqs / 2.0

    # Create epochs object
    epochs = mne.EpochsArray(tf_data, info_tf, tmin=tmin, verbose=False)
    epochs.metadata = df

    # TF decomposition
    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
        output="power",
        n_jobs=-1,
        verbose=False,
    )

    # -----------------------------
    # Sequence averaging for TFR: (block_nr, sequence_nr)
    # -----------------------------
    df = df.reset_index(drop=True)
    g = df.groupby(["block_nr", "sequence_nr"], sort=True)

    seq_tfr = []
    seq_meta = []

    X = power.data  # (n_trials, n_ch, n_freq, n_time)

    for (block_nr, seq_nr), idx in g.indices.items():
        idx = np.asarray(idx)
        ntr = len(idx)

        if ntr < min_trials_per_sequence:
            continue

        # Average TFR across trials in this sequence
        seq_avg = X[idx].mean(axis=0)  # (n_ch, n_freq, n_time)
        seq_tfr.append(seq_avg)

        df_sub = df.loc[idx]

        f = float(df_sub["last_feedback_scaled"].iloc[0])
        mean_difficulty = float(df_sub["trial_difficulty"].mean())
        half = "first" if int(block_nr) <= 4 else "second"
        group = df_sub["group"].iloc[0]

        seq_meta.append(
            {
                "id": subj_id,
                "group": group,
                "block_nr": int(block_nr),
                "sequence_nr": int(seq_nr),
                "n_trials": int(ntr),
                "mean_trial_difficulty": mean_difficulty,
                "f": f,
                "f2": f**2,
                "half": half,
            }
        )

    if len(seq_tfr) < 5:
        continue

    tfr_seq_data = np.stack(seq_tfr, axis=0)  # (n_seq, n_ch, n_freq, n_time)
    df_seq = pd.DataFrame(seq_meta)
    df_seq["half"] = df_seq["half"].astype("category")

    # Build a "sequence TFR" object by copying and overwriting
    n_seq = tfr_seq_data.shape[0]
    power_seq = power.copy()
    power_seq.data = tfr_seq_data

    # Create dummy events: (sample, 0, event_id)
    power_seq.events = np.c_[
        np.arange(n_seq), np.zeros(n_seq, int), np.ones(n_seq, int)
    ]

    # Also update selection
    power_seq.selection = np.arange(n_seq)

    # Setmetadata
    power_seq.metadata = df_seq.reset_index(drop=True)

    # Specify baseline
    baseline = (-1.5, -1.2)

    # Get baseline indices
    bl_idx = (power_seq.times >= baseline[0]) & (power_seq.times <= baseline[1])

    # Copy tf-power
    X = power_seq.data

    # Get baseline means(n_channels, n_freqs)
    bl_mean = X[..., bl_idx].mean(axis=(0, -1))

    # Avoid divide-by-zero
    eps = np.finfo(float).eps
    bl_mean = np.maximum(bl_mean, eps)

    # Calculate logratio of each trial
    Xc = 10 * np.log10(X / bl_mean[None, :, :, None])

    # BL-corrected data in a copy of power object
    power_seq_bl = power_seq.copy()
    power_seq_bl.data = Xc

    # Crop in time
    power_seq_bl.crop(tmin=-1.5, tmax=1)

    # Define freqbands (inclusive bounds)
    freqbands = {
        "theta": (4, 7),
        "alpha": (8, 13),
        "beta": (16, 30),
    }

    aa = bb
