from pathlib import Path
import mne
import numpy as np
import pandas as pd
import scipy.io

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned_45/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/3_sequence_tfr/")
DATASETS = sorted(PATH_IN.glob("*tf.set"))

# -----------------------------------------------------------------------------
# Exclusions
# -----------------------------------------------------------------------------
IDS_TO_DROP = {1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83}

# -----------------------------------------------------------------------------
# Channel info
# -----------------------------------------------------------------------------
CHANNEL_LABELS = (
    Path("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt")
    .read_text()
    .splitlines()
)

INFO_TF = mne.create_info(CHANNEL_LABELS, sfreq=200, ch_types="eeg", verbose=None)
INFO_TF.set_montage(
    mne.channels.make_standard_montage("standard_1020"),
    on_missing="warn",
    match_case=False,
)

# -----------------------------------------------------------------------------
# Containers for sequence-level data only
# -----------------------------------------------------------------------------
all_seq_tfr = []
all_seq_meta = []

for dataset in DATASETS:
    
    base = str(dataset).split("_cleaned")[0]
    df_trials = pd.read_csv(base + "_tf_trialinfo.csv")
    subj_id = int(df_trials["id"].iloc[0])
    
    print("Processing subject:", subj_id)

    if subj_id in IDS_TO_DROP:
        continue

    mat = scipy.io.loadmat(dataset)
    data = np.transpose(mat["data"], [2, 0, 1])  # trials x channels x time

    tf_times = mat["times"].ravel().astype(float)
    tf_times_sec = tf_times / 1000 if np.nanmax(np.abs(tf_times)) > 20 else tf_times

    # -------------------------------------------------------------------------
    # Trial metadata cleanup
    # -------------------------------------------------------------------------
    df_trials["accuracy"] = (df_trials["accuracy"] == 1).astype(int)

    df_trials = df_trials.rename(columns={"session_condition": "group"})
    df_trials["group"] = df_trials["group"].replace({1: "experimental", 2: "control"})
    df_trials["subject"] = subj_id

    # analysis variables
    df_trials["f_lin"] = df_trials["last_feedback_scaled"].astype(float)
    df_trials["f_quad"] = df_trials["f_lin"] ** 2
    df_trials["mean_trial_difficulty"] = df_trials["trial_difficulty"].astype(float)

    # derive half from block number
    df_trials["half"] = np.where(df_trials["block_nr"].between(1, 4), "first", "second")

    # exclude sequences without prior feedback
    keep_mask = df_trials["sequence_nr"] > 1
    df_trials = df_trials.loc[keep_mask].reset_index(drop=True)
    data = data[keep_mask, :, :]

    epochs = mne.EpochsArray(
        data=data,
        info=INFO_TF.copy(),
        tmin=float(tf_times_sec[0]),
        baseline=None,
        verbose=False,
        metadata=df_trials,
    )

    freqs = np.arange(3, 31, 2)
    n_cycles = freqs / 2.0

    tfr = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
        output="power",
        n_jobs=-1,   
        verbose=False,
    )

    # shape: trials x channels x freqs x times
    X = tfr.data.astype(np.float32)
    meta = df_trials.copy()

    # -------------------------------------------------------------------------
    # Sequence-level averaging within subject
    # -------------------------------------------------------------------------
    seq_id_cols = ["block_nr", "sequence_nr"]

    grouped = meta.groupby(seq_id_cols).indices
    seq_keys = list(grouped.keys())

    X_seq_subj = np.empty(
        (len(seq_keys), X.shape[1], X.shape[2], X.shape[3]),
        dtype=np.float32,
    )

    for i, key in enumerate(seq_keys):
        idx = grouped[key]
        X_seq_subj[i] = X[idx].mean(axis=0)

    # safer than .first() for difficulty
    seq_meta_subj = (
        meta.groupby(seq_id_cols, as_index=False)
        .agg(
            subject=("subject", "first"),
            group=("group", "first"),
            half=("half", "first"),
            f=("f_lin", "first"),
            f_quad=("f_quad", "first"),
            mean_trial_difficulty=("mean_trial_difficulty", "mean"),
        )
    )

    # align metadata order with X_seq_subj order
    seq_meta_subj = pd.DataFrame(seq_keys, columns=seq_id_cols).merge(
        seq_meta_subj,
        on=seq_id_cols,
        how="left",
        validate="one_to_one",
    )

    all_seq_tfr.append(X_seq_subj)
    all_seq_meta.append(seq_meta_subj)

# -----------------------------------------------------------------------------
# Concatenate sequence-level data across subjects
# -----------------------------------------------------------------------------
X_seq = np.concatenate(all_seq_tfr, axis=0)   # sequences x channels x freqs x times
seq_meta = pd.concat(all_seq_meta, ignore_index=True)

# -----------------------------------------------------------------------------
# Save combined data
# -----------------------------------------------------------------------------
np.save(PATH_OUT / "X_seq.npy", X_seq)
np.save(PATH_OUT / "freqs.npy", freqs.astype(np.float32))
np.save(PATH_OUT / "times.npy", tf_times_sec.astype(np.float32))
np.save(PATH_OUT / "channel_labels.npy", np.array(CHANNEL_LABELS, dtype=object))
seq_meta.to_csv(PATH_OUT / "seq_meta.csv", index=False)

