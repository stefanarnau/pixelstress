# ============================================================
# Sequence-level ERP regression: compute beta maps per subject
# ============================================================

import glob
import numpy as np
import pandas as pd
import mne
import scipy.io
from sklearn.preprocessing import StandardScaler
import mne.stats 

# -----------------------------
# Settings
# -----------------------------
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
datasets = glob.glob(f"{path_in}/*erp.set")

ids_to_drop = {1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83}
min_trials_per_sequence = 3

# Load channel labels + create info
channel_labels = (
    open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r")
    .read()
    .split("\n")[:-1]
)
sfreq = 1000.0
info_erp = mne.create_info(channel_labels, sfreq, ch_types="eeg", verbose=None)

# Helper: build design matrix (within subject)
def build_design_matrix(df_seq: pd.DataFrame):
    X = df_seq[["f", "f2", "mean_trial_difficulty", "half"]].copy()
    X["half"] = (X["half"].astype(str) == "second").astype(int)
    X = X.astype(float)

    Xz = StandardScaler().fit_transform(X)  # within-subject z-scoring
    Xz = np.column_stack([np.ones(len(Xz)), Xz])  # intercept
    names = ["Intercept", "f", "f2", "difficulty", "half"]
    return Xz, names

# Outputs
beta_maps = []  # list of dicts: {id, group, betas{name: Evoked}}

for dataset in datasets:
    base = dataset.split("_cleaned")[0]

    # Trialinfo
    df_erp = pd.read_csv(base + "_erp_trialinfo.csv")
    df_tf  = pd.read_csv(base + "_tf_trialinfo.csv")

    subj_id = int(df_erp["id"].iloc[0])
    if subj_id in ids_to_drop:
        continue

    # Load ERP data: expects EEGLAB .set saved as .mat-like via scipy.io.loadmat
    mat = scipy.io.loadmat(dataset)
    erp_data = np.transpose(mat["data"], [2, 0, 1])  # (trials, channels, times)
    erp_times = mat["times"].ravel()

    # Determine time units for tmin (heuristic: EEGLAB typically ms)
    # If times look like [-1000..2000], that's ms; if [-1..2], that's seconds.
    if np.nanmax(np.abs(erp_times)) > 20:
        tmin = erp_times[0] / 1000.0
    else:
        tmin = erp_times[0]

    # Common trials between ERP and TF
    to_keep = np.intersect1d(df_erp["trial_nr_total"].values, df_tf["trial_nr_total"].values)

    # Reduce metadata to common trials (copy)
    df = df_erp[df_erp["trial_nr_total"].isin(to_keep)].copy()

    # Reduce ERP data to those common trials (mask based on ORIGINAL df_erp order)
    mask_common = np.isin(df_erp["trial_nr_total"].values, to_keep)
    erp_data = erp_data[mask_common, :, :]

    # Basic checks
    if erp_data.shape[0] != len(df):
        raise RuntimeError(f"Trial alignment mismatch for subject {subj_id}: "
                           f"erp_data {erp_data.shape[0]} vs df {len(df)}")

    # Binarize accuracy
    df["accuracy"] = (df["accuracy"] == 1).astype(int)

    # Group coding
    df = df.rename(columns={"session_condition": "group"})
    df["group"] = df["group"].replace({1: "experimental", 2: "control"})

    # Remove first sequences
    mask = df["sequence_nr"] > 1
    df = df.loc[mask].reset_index(drop=True)
    erp_data = erp_data[mask.to_numpy(), :, :]

    # Keep only correct trials
    mask = df["accuracy"] == 1
    df = df.loc[mask].reset_index(drop=True)
    erp_data = erp_data[mask.to_numpy(), :, :]

    if len(df) == 0:
        continue

    # -----------------------------
    # Sequence averaging: (block_nr, sequence_nr)
    # -----------------------------
    df = df.reset_index(drop=True)
    g = df.groupby(["block_nr", "sequence_nr"], sort=True)

    seq_erp = []
    seq_meta = []

    for (block_nr, seq_nr), idx in g.indices.items():
        idx = np.asarray(idx)
        ntr = len(idx)

        if ntr < min_trials_per_sequence:
            continue

        # Average ERP across trials in this sequence
        seq_avg = erp_data[idx].mean(axis=0)  # (n_ch, n_time)
        seq_erp.append(seq_avg)

        df_sub = df.loc[idx]

        # Feedback should be constant within sequence; take first
        f = float(df_sub["last_feedback_scaled"].iloc[0])

        # Difficulty may vary; average within sequence (only correct trials)
        mean_difficulty = float(df_sub["trial_difficulty"].mean())

        # Half from block number
        half = "first" if int(block_nr) <= 4 else "second"

        # Group constant within subject; take first
        group = df_sub["group"].iloc[0]

        seq_meta.append({
            "id": subj_id,
            "group": group,
            "block_nr": int(block_nr),
            "sequence_nr": int(seq_nr),
            "n_trials": int(ntr),
            "mean_trial_difficulty": mean_difficulty,
            "f": f,
            "f2": f**2,
            "half": half
        })

    if len(seq_erp) < 5:
        # Not enough sequences to fit a regression stably
        continue

    erp_seq_data = np.stack(seq_erp, axis=0)  # (n_seq, n_ch, n_time)
    df_seq = pd.DataFrame(seq_meta)

    # Ensure types
    df_seq["half"] = df_seq["half"].astype("category")

    # -----------------------------
    # Build EpochsArray over sequences and fit regression
    # -----------------------------
    epochs_seq = mne.EpochsArray(erp_seq_data, info_erp, tmin=tmin, verbose=False)
    epochs_seq.metadata = df_seq

    X, names = build_design_matrix(df_seq)

    lm = mne.stats.linear_regression(epochs_seq, X, names=names)

    # Extract betas as Evoked
    betas = {name: lm[name].beta for name in names}

    beta_maps.append({
        "id": subj_id,
        "group": df_seq["group"].iloc[0],
        "n_sequences": int(len(df_seq)),
        "betas": betas
    })
    
