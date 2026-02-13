# Imports
import glob
import numpy as np
import pandas as pd
import mne
import scipy.io
from sklearn.preprocessing import StandardScaler
import mne.stats
import re
from pathlib import Path

path_out = Path("/mnt/data_dump/pixelstress/3_sequence_data_plus_fitted/")
path_out.mkdir(parents=True, exist_ok=True)


# Path things
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
datasets = glob.glob(f"{path_in}/*erp.set")

# Exclusion list
ids_to_drop = {1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83}

# This is for re-ordering channels by rows and within rows left to right
ROW_ORDER = ["Fp", "AF", "F", "FC", "FT", "C", "CT", "T", "CP", "TP", "P", "PO", "O"]
ROW_RANK = {row: i for i, row in enumerate(ROW_ORDER)}
_num_re = re.compile(r"(\d+)$", re.IGNORECASE)


def _row_prefix(ch: str) -> str:
    ch = ch.strip()
    if ch.endswith("z") and len(ch) >= 2:
        return ch[:-1]
    m = _num_re.search(ch)
    if m:
        return ch[: m.start()]
    return ch


def _lr_rank_pure_lr(ch: str) -> tuple:
    """
    Pure left->right ordering:
      left odds: descending (9..1), then z, then right evens: ascending (2..10)
    """
    ch = ch.strip()

    if ch.endswith("z"):
        return (1, 0)  # middle

    m = _num_re.search(ch)
    if not m:
        return (3, 999)  # unknown last

    n = int(m.group(1))
    if n % 2 == 1:  # left (odd)
        return (0, -n)  # DESCENDING on left: 9,7,5,3,1  (use -n)
    else:  # right (even)
        return (2, n)  # ASCENDING on right: 2,4,6,8,10


def sort_1010_labels_front_back_pure_lr(labels):
    def key(ch):
        row = _row_prefix(ch)
        row_rank = ROW_RANK.get(row, 999)
        lr_group, lr_val = _lr_rank_pure_lr(ch)
        return (row_rank, lr_group, lr_val, ch)

    return sorted(labels, key=key)


# Load channel labels + create info + set montage + calculate adjacency
channel_labels = (
    open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r")
    .read()
    .split("\n")[:-1]
)

# Get new channel order
sorted_channel_names = sort_1010_labels_front_back_pure_lr(channel_labels)
new_channel_order = [channel_labels.index(ch) for ch in sorted_channel_names]

# Create erp info object
info_erp = mne.create_info(channel_labels, sfreq=1000, ch_types="eeg", verbose=None)
montage = mne.channels.make_standard_montage("standard_1020")  # or "standard_1005"
info_erp.set_montage(montage, on_missing="warn", match_case=False)

# Create tf info object
info_tf = mne.create_info(channel_labels, sfreq=200, ch_types="eeg", verbose=None)
montage = mne.channels.make_standard_montage("standard_1020")  # or "standard_1005"
info_tf.set_montage(montage, on_missing="warn", match_case=False)

# Reorder info objects
info_erp = mne.pick_info(
    info_erp, [info_erp.ch_names.index(ch) for ch in sorted_channel_names]
)
info_tf = mne.pick_info(
    info_tf, [info_tf.ch_names.index(ch) for ch in sorted_channel_names]
)

# Define adjacencies
adjacency, ch_names = mne.channels.find_ch_adjacency(info_erp, ch_type="eeg")

# index rows for all subjects
rows = []

# Loop datasets
for dataset in datasets:

    # Name base
    base = dataset.split("_cleaned")[0]

    # Trialinfo
    df_erp = pd.read_csv(base + "_erp_trialinfo.csv")
    df_tf = pd.read_csv(base + "_tf_trialinfo.csv")

    # Get id and check exclusion list
    subj_id = int(df_erp["id"].iloc[0])
    if subj_id in ids_to_drop:
        continue

    # Load ERP data: expects EEGLAB .set saved as .mat-like via scipy.io.loadmat
    mat = scipy.io.loadmat(dataset)
    erp_data = np.transpose(mat["data"], [2, 0, 1])  # (trials, channels, times)
    erp_times = mat["times"].ravel()

    # Re-order channels
    erp_data = erp_data[:, new_channel_order, :]

    # Load tf eeg data as trials x channles x times
    tf_data = np.transpose(
        scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")["data"], [2, 0, 1]
    )
    tf_times = scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")[
        "times"
    ].ravel()

    # Re-order channels
    tf_data = tf_data[:, new_channel_order, :]

    # Determine time units. Is it ms or s
    tmin_erp = (
        erp_times[0] / 1000.0 if np.nanmax(np.abs(erp_times)) > 20 else erp_times[0]
    )
    tmin_tf = tf_times[0] / 1000.0 if np.nanmax(np.abs(tf_times)) > 20 else tf_times[0]

    # Common trials between ERP and TF
    to_keep = np.intersect1d(
        df_erp["trial_nr_total"].values, df_tf["trial_nr_total"].values
    )

    # Reduce metadata to common trials (copy)
    df = df_erp[df_erp["trial_nr_total"].isin(to_keep)].copy()

    # Reduce ERP data to those common trials (mask based on ORIGINAL df_erp order)
    mask_common = np.isin(df_erp["trial_nr_total"].values, to_keep)
    erp_data = erp_data[mask_common, :, :]

    # Reduce TF data to those common trials
    mask_common = np.isin(df_tf["trial_nr_total"].values, to_keep)
    tf_data = tf_data[mask_common, :, :]

    # Binarize accuracy
    df["accuracy"] = (df["accuracy"] == 1).astype(int)

    # Group coding
    df = df.rename(columns={"session_condition": "group"})
    df["group"] = df["group"].replace({1: "experimental", 2: "control"})

    # Remove first sequences
    mask = df["sequence_nr"] > 1
    df = df.loc[mask].reset_index(drop=True)
    erp_data = erp_data[mask.to_numpy(), :, :]
    tf_data = tf_data[mask.to_numpy(), :, :]

    # Keep only correct trials
    mask = df["accuracy"] == 1
    df = df.loc[mask].reset_index(drop=True)
    erp_data = erp_data[mask.to_numpy(), :, :]
    tf_data = tf_data[mask.to_numpy(), :, :]

    # Se tf-decomposition parameters
    freqs = np.linspace(4, 30, 15)
    n_cycles = np.linspace(4, 10, len(freqs))
    
    # Create Epochs for erp data to apply CSD
    #epochs_tmp = mne.EpochsArray(erp_data, info_erp, tmin=tmin_erp, baseline=None)
    #epochs_tmp = compute_current_source_density(epochs_tmp)
    #erp_data = epochs_tmp.get_data()

    # Create epochs object for tfr
    epochs = mne.EpochsArray(tf_data, info_tf, tmin=tmin_tf, verbose=False)
    epochs.metadata = df
    
    # Apply CSD to tfr data
    #epochs = compute_current_source_density(epochs)

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

    # Groupby block and sequence to get trial indices of individual sequences
    df = df.reset_index(drop=True)
    g = df.groupby(["block_nr", "sequence_nr"], sort=True)

    # Collectors
    seq_tfr = []
    seq_erp = []
    seq_meta = []

    # Get power data as (n_trials, n_ch, n_freq, n_time)
    power_data = power.data

    # Iterate sequences
    for (block_nr, seq_nr), idx in g.indices.items():

        # Get trial indices
        idx = np.asarray(idx)

        # Get number of trials
        n_trials = len(idx)

        # Check if sufficient number of trials
        if n_trials < 3:
            continue

        # Average ERP across trials in this sequence and collect
        seq_avg = erp_data[idx].mean(axis=0)
        seq_erp.append(seq_avg)

        # Average TFR across trials in this sequence and collect
        seq_avg = power_data[idx].mean(axis=0)
        seq_tfr.append(seq_avg)

        # Get df of sequence
        df_sub = df.loc[idx]

        # Get feedback of sequence
        f = float(df_sub["last_feedback_scaled"].iloc[0])

        # Get average difficulty of sequence
        mean_difficulty = float(df_sub["trial_difficulty"].mean())

        # Get experimental half from block number
        half = "first" if int(block_nr) <= 4 else "second"

        # Get group of subject
        group = df_sub["group"].iloc[0]

        # Collect metadata in dict
        seq_meta.append(
            {
                "id": subj_id,
                "group": group,
                "block_nr": int(block_nr),
                "sequence_nr": int(seq_nr),
                "n_trials": int(n_trials),
                "mean_trial_difficulty": mean_difficulty,
                "f": f,
                "f2": f**2,
                "half": half,
            }
        )

    # Check if at least 5 sequences
    if len(seq_erp) < 5:
        continue

    # Stack sequence erps as (n_seq, n_ch, n_time) array
    erp_seq_data = np.stack(seq_erp, axis=0)

    # Stack sequence tfr as (n_seq, n_ch, n_freq, n_time)) array
    tfr_seq_data = np.stack(seq_tfr, axis=0)

    # Create df of metadata
    df_seq = pd.DataFrame(seq_meta)

    # Make half caterogial
    df_seq["half"] = df_seq["half"].astype("category")

    # Build EpochsArray of sequences for erp data
    epochs_seq = mne.EpochsArray(erp_seq_data, info_erp, tmin=tmin_erp, verbose=False)
    epochs_seq.metadata = df_seq

    # Crop in time
    epochs_seq.crop(tmin=-1.5, tmax=1)

    # Downsample to 500 Hz (includes anti-alias filtering)
    epochs_seq.resample(500, npad="auto")

    # Sanity check
    assert len(df_seq) == len(epochs_seq)

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

    # Set metadata
    power_seq.metadata = df_seq.reset_index(drop=True)

    # Specify baseline
    baseline = (-1.5, -1.2)

    # Get baseline indices
    bl_idx = (power_seq.times >= baseline[0]) & (power_seq.times <= baseline[1])
    if not bl_idx.any():
        raise RuntimeError(
            f"Baseline {baseline} has no samples. "
            f"Time range is [{power_seq.times[0]:.3f}, {power_seq.times[-1]:.3f}] s."
        )

    # Apply logratio baseline
    tfr_seq = power_seq.data  # do NOT call this X
    bl_mean = tfr_seq[..., bl_idx].mean(axis=(0, -1))
    eps = np.finfo(tfr_seq.dtype).eps
    bl_mean = np.maximum(bl_mean, eps)
    tfr_seq_bl = 10 * np.log10(tfr_seq / bl_mean[None, :, :, None])

    # BL-corrected data in a copy of power object
    power_seq_bl = power_seq.copy()
    power_seq_bl.data = tfr_seq_bl

    # Crop in time
    power_seq_bl.crop(tmin=-1.5, tmax=1)

    # Sanity check
    assert power_seq_bl.data.shape[0] == len(df_seq)

    # Get lm predictors from df for design matrix
    desmat = df_seq[["f", "f2", "mean_trial_difficulty", "half"]].copy()

    # Make categorical variable integer
    desmat["half"] = (desmat["half"].astype(str) == "second").astype(int)

    # Cast design matrix to float
    desmat = desmat.astype(float)

    # Within-subject z-scoring
    desmat = StandardScaler().fit_transform(desmat)

    # Add intercept
    desmat = np.column_stack([np.ones(len(desmat)), desmat])

    # Sanity check
    assert desmat.shape[0] == len(df_seq)

    # Specify column names for model
    names = ["Intercept", "f", "f2", "difficulty", "half"]

    # Fit model for erp
    lm_erp = mne.stats.linear_regression(epochs_seq, desmat, names=names)

    # Extract betas as Evoked
    betas_erp = {name: lm_erp[name].beta for name in names}
    
    
    # Container for betas of each regressor in electrode x frequency x time -space
    betas_eft = {
        name: np.zeros(
            (power_seq_bl.shape[1], power_seq_bl.shape[2], power_seq_bl.shape[3])
        )
        for name in names
    }

    # Loop over frequencies: run regression on (n_seq, n_ch, n_time) at each freq
    for fi in range(power_seq_bl.shape[2]):

        # Slice one frequency as (n_seq, n_ch, n_time)
        seq_f = power_seq_bl.data[:, :, fi, :]

        # Create epochsArray from that
        epochs_seq_f = mne.EpochsArray(seq_f, power_seq_bl.info, tmin=power_seq_bl.tmin, verbose=False)
        epochs_seq_f.metadata = df_seq
        lm = mne.stats.linear_regression(epochs_seq_f, desmat, names=names)

        # Extract betas (Evoked) and store into arrays
        for name in names:
            betas_eft[name][:, fi, :] = lm[name].beta.data  # (n_ch, n_time)
            
            

    lm_by_band = {}
    betas_by_band = {}

    # Define freqbands (inclusive bounds)
    freqbands = {
        "theta": (4, 7),
        "alpha": (8, 13),
        "beta": (16, 30),
    }

    # Iterate freqbands
    for band, (fmin, fmax) in freqbands.items():

        # Get frequency mask
        freqmask = (power_seq_bl.freqs >= fmin) & (power_seq_bl.freqs <= fmax)

        # Collapse freq dimension to (n_seq, n_ch, n_time)
        freqband_data = power_seq_bl.data[:, :, freqmask, :].mean(axis=2)

        # wrap as EpochsArray (this is what linear_regression needs)
        epochs_band = mne.EpochsArray(
            freqband_data,
            info_tf,
            tmin=power_seq_bl.times[0],
            verbose=False,
        )
        epochs_band.metadata = df_seq.reset_index(drop=True)

        # optional time crop (do it here, not on the 4D object)
        epochs_band.crop(tmin=-1.5, tmax=1.0)

        # regression
        lm = mne.stats.linear_regression(epochs_band, desmat, names=names)

        # store betas (Evoked objects, exactly like ERP)
        betas = {name: lm[name].beta for name in names}

        # Collect betas for frequency bands
        betas_by_band[band] = betas

    # Check channel consistency
    assert epochs_seq.ch_names == sorted_channel_names
    assert power_seq_bl.ch_names == sorted_channel_names

    # Create subject directory
    sub_dir = path_out / f"sub-{subj_id:04d}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Save ERP sequences
    erp_epo_f = sub_dir / f"sub-{subj_id:04d}_seq-erp_epo.fif"
    epochs_seq.save(erp_epo_f, overwrite=True)

    # Save TFR sequences
    tfr_h5_f = sub_dir / f"sub-{subj_id:04d}_seq-tfr_bl-tfr.h5"
    power_seq_bl.save(tfr_h5_f, overwrite=True)

    # Save ERP betas
    erp_beta_f = sub_dir / f"sub-{subj_id:04d}_lm-erp_betas-ave.fif"
    mne.write_evokeds(erp_beta_f, [betas_erp[n] for n in names], overwrite=True)
    
    # Save electrode x freq x time betas (EFT)
    eft_beta_f = sub_dir / f"sub-{subj_id:04d}_lm-tf_eft_betas.npz"
    
    # Stack as (n_reg, n_ch, n_freq, n_time)
    reg_names = np.array(names, dtype="U") 
    eft_stack = np.stack([betas_eft[n] for n in names], axis=0).astype(np.float32)
    
    np.savez_compressed(
        eft_beta_f,
        betas=eft_stack,                    # (n_reg, n_ch, n_freq, n_time)
        regressor_names=reg_names,          # (n_reg,)
        ch_names=np.array(power_seq_bl.ch_names, dtype="U"),
        freqs=np.array(power_seq_bl.freqs, dtype=np.float32),
        times=np.array(power_seq_bl.times, dtype=np.float32),
        tmin=float(power_seq_bl.tmin),
        baseline=np.array(baseline, dtype=np.float32),
    )


    # Save TF band betas
    tf_beta_files = {}
    for band in ["theta", "alpha", "beta"]:
        f = sub_dir / f"sub-{subj_id:04d}_lm-tf-{band}_betas-ave.fif"
        mne.write_evokeds(f, [betas_by_band[band][n] for n in names], overwrite=True)
        tf_beta_files[band] = f

    # Collect index row
    rows.append(
        dict(
            id=subj_id,
            group=df_seq["group"].iloc[0],
            n_sequences=int(len(df_seq)),
            erp_epochs_path=str(erp_epo_f),
            tfr_epochs_path=str(tfr_h5_f),
            erp_betas_path=str(erp_beta_f),
            tf_theta_betas_path=str(tf_beta_files["theta"]),
            tf_alpha_betas_path=str(tf_beta_files["alpha"]),
            tf_beta_betas_path=str(tf_beta_files["beta"]),
            tf_eft_betas_path=str(eft_beta_f), 
        )
    )

# Save index
index = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
index.to_csv(path_out / "subject_index.csv", index=False)
