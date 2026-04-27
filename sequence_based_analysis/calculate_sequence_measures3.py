# -----------------------------------------------------------------------------
# Sequence-based FOOOF + RT + flattened oscillatory bandpower extraction
# CAR only
#
# Per subject x sequence x channel:
# 1) compute PSD trial-wise within sequence
# 2) aggregate PSD across trials within sequence
# 3) fit FOOOF to the aggregated PSD
# 4) extract exponent / offset / fit metrics / flattened bandpower
#
# The loaded data are assumed to already be CAR-referenced.
# No CSD logic is included.
# -----------------------------------------------------------------------------

from pathlib import Path

import fooof
import mne
import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
from mne.time_frequency import psd_array_multitaper


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned2/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

DATASETS = sorted(PATH_IN.glob("*erp.set"))


# -----------------------------------------------------------------------------
# Exclusions
# -----------------------------------------------------------------------------
IDS_TO_DROP = {1, 2, 3, 4, 5, 6, 13, 17, 25, 40, 49, 83}


# -----------------------------------------------------------------------------
# Channel info
# -----------------------------------------------------------------------------
CHANNEL_LABELS = (
    Path("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt")
    .read_text()
    .splitlines()
)

INFO_ERP = mne.create_info(CHANNEL_LABELS, sfreq=500, ch_types="eeg", verbose=None)
MONTAGE = mne.channels.make_standard_montage("standard_1020")
INFO_ERP.set_montage(MONTAGE, on_missing="warn", match_case=False)


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------
# If empty, all channels are used. Otherwise provide a list like:
# ["Cz", "CPz", "Pz"]
SELECT_CHANNELS = ["Cz", "Fz", "Pz"]

TIME_WINDOW = (-1.4, 0.0)
MIN_TRIALS_PER_SEQUENCE = 5

# PSD / FOOOF settings
DOWNSAMPLE_BEFORE_PSD = True
TARGET_SFREQ = 250

FMIN_FIT = 1.0
FMAX_FIT = 40.0
MT_BANDWIDTH = 1.5
MT_NORMALIZATION = "full"

FOOOF_KWARGS = dict(
    aperiodic_mode="fixed",
    peak_width_limits=(1, 8),
    max_n_peaks=8,
    min_peak_height=0.05,
    verbose=False,
)

BANDS = {
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta": (15, 30),
}

PSD_AGG_MODE = "mean"  # "mean" or "median"


# -----------------------------------------------------------------------------
# Channel selection
# -----------------------------------------------------------------------------
def get_selected_channel_indices(all_labels, selected_labels):
    if not selected_labels:
        idx = np.arange(len(all_labels), dtype=int)
        names = list(all_labels)
        return idx, names

    missing = [ch for ch in selected_labels if ch not in all_labels]
    if missing:
        raise ValueError(f"Selected channels not found: {missing}")

    idx = np.array([all_labels.index(ch) for ch in selected_labels], dtype=int)
    names = [all_labels[i] for i in idx]
    return idx, names


SELECTED_CH_IDX, SELECTED_CH_NAMES = get_selected_channel_indices(
    CHANNEL_LABELS,
    SELECT_CHANNELS,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def aggregate_psd(psd: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Aggregate PSDs across trials within a sequence.

    Input shape:
        trials x channels x freqs

    Output shape:
        channels x freqs
    """
    if mode == "mean":
        return psd.mean(axis=0)
    if mode == "median":
        return np.median(psd, axis=0)
    raise ValueError(f"Unknown PSD_AGG_MODE: {mode}")


def prepare_epochs_for_psd(epochs: mne.Epochs) -> mne.Epochs:
    if DOWNSAMPLE_BEFORE_PSD:
        return epochs.copy().resample(
            TARGET_SFREQ,
            npad="auto",
            verbose=False,
        )
    return epochs


def compute_flattened_bandpowers(
    psd_1d: np.ndarray,
    freqs: np.ndarray,
    offset: float,
    exponent: float,
    bands: dict,
):
    """
    Compute flattened bandpower from one PSD vector using a fixed aperiodic fit:

        flat = log10(psd) - (offset - exponent * log10(freqs))
    """
    eps = np.finfo(float).tiny
    log_psd = np.log10(np.maximum(psd_1d, eps))
    aperiodic_fit = offset - exponent * np.log10(freqs)
    flat = log_psd - aperiodic_fit

    out = {}
    for band_name, (lo, hi) in bands.items():
        band_mask = (freqs >= lo) & (freqs <= hi)
        out[f"{band_name}_flat"] = (
            float(np.mean(flat[band_mask])) if np.any(band_mask) else np.nan
        )

    return out


# -----------------------------------------------------------------------------
# Core extraction
# -----------------------------------------------------------------------------
def extract_sequence_measures(
    erp_data: np.ndarray,
    erp_times_sec: np.ndarray,
    df_trials: pd.DataFrame,
    subj_id: int,
    sfreq_psd: float,
):
    tidx = np.where(
        (erp_times_sec >= TIME_WINDOW[0]) & (erp_times_sec < TIME_WINDOW[1])
    )[0]

    if tidx.size == 0:
        raise ValueError("No samples found in requested PSD time window.")

    grouped = df_trials.groupby(["block_nr", "sequence_nr"], sort=True)

    rows = []
    seq_psd = []
    seq_psd_index = []
    freqs_out = None

    for (block_nr, seq_nr), idx in grouped.indices.items():
        idx = np.asarray(idx)
        n_trials = len(idx)

        if n_trials < MIN_TRIALS_PER_SEQUENCE:
            continue

        dseq = df_trials.loc[idx]

        group = dseq["group"].iloc[0]
        half = "first" if int(block_nr) <= 4 else "second"
        f = float(dseq["last_feedback_scaled"].iloc[0])

        mean_difficulty = float(dseq["trial_difficulty"].mean())
        mean_rt = float(dseq["rt"].mean())
        mean_log_rt = float(np.log(mean_rt)) if mean_rt > 0 else np.nan

        # trials x selected_channels x time-window
        x_psd = erp_data[idx][:, SELECTED_CH_IDX, :][:, :, tidx]

        psd, freqs = psd_array_multitaper(
            x_psd,
            sfreq=sfreq_psd,
            fmin=FMIN_FIT,
            fmax=FMAX_FIT,
            bandwidth=MT_BANDWIDTH,
            normalization=MT_NORMALIZATION,
            verbose=False,
            n_jobs=1,
        )
        # psd shape: trials x channels x freqs

        freqs_out = freqs

        psd_seq = aggregate_psd(psd, mode=PSD_AGG_MODE)  # channels x freqs
        seq_psd.append(psd_seq)

        seq_psd_index.append(
            {
                "id": subj_id,
                "group": group,
                "block_nr": int(block_nr),
                "sequence_nr": int(seq_nr),
                "half": half,
                "n_trials": int(n_trials),
                "mean_trial_difficulty": mean_difficulty,
                "f": f,
                "mean_rt": mean_rt,
                "mean_log_rt": mean_log_rt,
                "sfreq_psd": float(sfreq_psd),
                "psd_agg_mode": PSD_AGG_MODE,
            }
        )

        # FOOOF on sequence-aggregated PSD
        fg = fooof.FOOOFGroup(**FOOOF_KWARGS)
        fg.fit(freqs, psd_seq, [FMIN_FIT, FMAX_FIT])

        aperiodic = fg.get_params("aperiodic_params")
        r2_vals = fg.get_params("r_squared")
        err_vals = fg.get_params("error")

        offsets = aperiodic[:, 0]
        exponents = aperiodic[:, 1]

        for ch_ix_local in range(psd_seq.shape[0]):
            ch_ix_global = int(SELECTED_CH_IDX[ch_ix_local])
            ch_name = SELECTED_CH_NAMES[ch_ix_local]

            band_vals = compute_flattened_bandpowers(
                psd_1d=psd_seq[ch_ix_local],
                freqs=freqs,
                offset=float(offsets[ch_ix_local]),
                exponent=float(exponents[ch_ix_local]),
                bands=BANDS,
            )

            row = {
                "id": subj_id,
                "group": group,
                "block_nr": int(block_nr),
                "sequence_nr": int(seq_nr),
                "half": half,
                "n_trials": int(n_trials),
                "mean_trial_difficulty": mean_difficulty,
                "f": f,
                "mean_rt": mean_rt,
                "mean_log_rt": mean_log_rt,
                "ch_ix": ch_ix_global,
                "ch_name": ch_name,
                "offset": float(offsets[ch_ix_local]),
                "exponent": float(exponents[ch_ix_local]),
                "r2": float(r2_vals[ch_ix_local]),
                "error": float(err_vals[ch_ix_local]),
                "theta_flat": band_vals["theta_flat"],
                "alpha_flat": band_vals["alpha_flat"],
                "beta_flat": band_vals["beta_flat"],
                "fmin_fit": FMIN_FIT,
                "fmax_fit": FMAX_FIT,
                "mt_bandwidth": MT_BANDWIDTH,
                "sfreq_psd": float(sfreq_psd),
                "psd_agg_mode": PSD_AGG_MODE,
                "mt_normalization": MT_NORMALIZATION,
                "time_window_start": float(TIME_WINDOW[0]),
                "time_window_end": float(TIME_WINDOW[1]),
                "reference": "car",
            }

            rows.append(row)

    df_seq = pd.DataFrame(rows)

    if df_seq.empty:
        return None, None, None, None

    return df_seq, seq_psd, seq_psd_index, freqs_out


def run_pipeline(
    epochs: mne.Epochs,
    df_trials: pd.DataFrame,
    subj_id: int,
):
    epochs_psd = prepare_epochs_for_psd(epochs)

    erp_data_psd = epochs_psd.get_data(copy=True)
    erp_times_psd = epochs_psd.times.copy()
    sfreq_psd = float(epochs_psd.info["sfreq"])

    return extract_sequence_measures(
        erp_data=erp_data_psd,
        erp_times_sec=erp_times_psd,
        df_trials=df_trials,
        subj_id=subj_id,
        sfreq_psd=sfreq_psd,
    )


def save_outputs(
    df_seq: pd.DataFrame,
    seq_psd: list,
    seq_psd_index: list,
    freqs: np.ndarray,
    subj_id: int,
):
    subj_tag = f"sub-{subj_id:03d}"

    df_seq.to_csv(
        PATH_OUT / f"{subj_tag}_seq_fooof_rt_channelwise_long_car.csv",
        index=False,
    )

    if seq_psd and freqs is not None:
        np.savez_compressed(
            PATH_OUT / f"{subj_tag}_seq_psd_channelwise_car.npz",
            psd=np.stack(seq_psd),
            freqs=freqs,
            channels=np.array(SELECTED_CH_NAMES),
        )

        pd.DataFrame(seq_psd_index).to_csv(
            PATH_OUT / f"{subj_tag}_seq_psd_channelwise_index_car.csv",
            index=False,
        )


# -----------------------------------------------------------------------------
# Subject-level processing
# -----------------------------------------------------------------------------
def process_subject(dataset: Path):
    base = str(dataset).split("_cleaned")[0]

    df_trials = pd.read_csv(base + "_erp_trialinfo.csv")
    subj_id = int(df_trials["id"].iloc[0])

    if subj_id in IDS_TO_DROP:
        return None

    mat = scipy.io.loadmat(dataset)

    erp_data = np.transpose(mat["data"], [2, 0, 1])  # trials x channels x time
    erp_times = mat["times"].ravel().astype(float)
    erp_times_sec = erp_times / 1000 if np.nanmax(np.abs(erp_times)) > 20 else erp_times

    if erp_data.shape[1] != len(CHANNEL_LABELS):
        raise ValueError(
            f"Channel mismatch for subject {subj_id}: "
            f"{erp_data.shape[1]} vs {len(CHANNEL_LABELS)}"
        )

    df_trials["accuracy"] = (df_trials["accuracy"] == 1).astype(int)
    df_trials = df_trials.rename(columns={"session_condition": "group"})
    df_trials["group"] = df_trials["group"].replace({1: "experimental", 2: "control"})

    keep_mask = df_trials["sequence_nr"] > 1
    df_trials = df_trials.loc[keep_mask].reset_index(drop=True)
    erp_data = erp_data[keep_mask, :, :]

    # Optional accuracy-only analysis:
    # keep_mask = df_trials["accuracy"] == 1
    # df_trials = df_trials.loc[keep_mask].reset_index(drop=True)
    # erp_data = erp_data[keep_mask, :, :]

    if len(df_trials) != erp_data.shape[0]:
        raise ValueError(
            f"Trial/data mismatch for subject {subj_id}: "
            f"{len(df_trials)} rows vs {erp_data.shape[0]} trials"
        )

    epochs = mne.EpochsArray(
        data=erp_data,
        info=INFO_ERP.copy(),
        tmin=float(erp_times_sec[0]),
        baseline=None,
        verbose=False,
    )

    df_seq, seq_psd, seq_psd_index, freqs = run_pipeline(
        epochs=epochs,
        df_trials=df_trials,
        subj_id=subj_id,
    )

    if df_seq is None or df_seq.empty:
        return None

    save_outputs(
        df_seq=df_seq,
        seq_psd=seq_psd,
        seq_psd_index=seq_psd_index,
        freqs=freqs,
        subj_id=subj_id,
    )

    print(f"Saved subject {subj_id:03d} (car)")
    return df_seq


# -----------------------------------------------------------------------------
# Run extraction
# -----------------------------------------------------------------------------
seq_data = Parallel(
    n_jobs=12,
    backend="loky",
    verbose=10,
)(delayed(process_subject)(dataset) for dataset in DATASETS)

seq_data = [d for d in seq_data if d is not None and not d.empty]

if not seq_data:
    raise RuntimeError("No subject data returned.")

df_all = pd.concat(seq_data, ignore_index=True)


# -----------------------------------------------------------------------------
# Save combined dataframe
# -----------------------------------------------------------------------------
combined_file = PATH_OUT / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"
df_all.to_csv(combined_file, index=False)

print("Finished.")
print("Saved:", combined_file)
print("Rows:", len(df_all))
print("Subjects:", df_all["id"].nunique())
print("Electrodes:", df_all["ch_name"].nunique())
print("Reference:", sorted(df_all["reference"].unique()))
print("Selected channels:", SELECTED_CH_NAMES)
print(
    "Sequences:",
    df_all[["id", "block_nr", "sequence_nr"]].drop_duplicates().shape[0],
)