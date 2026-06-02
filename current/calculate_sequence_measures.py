# -----------------------------------------------------------------------------
# Sequence-based FOOOF + RT + flattened and unflattened bandpower extraction
# CAR only
#
# Per subject x sequence x channel:
# 1) compute PSD trial-wise within sequence
# 2) aggregate PSD across trials within sequence
# 3) fit FOOOF to the aggregated PSD
# 4) extract exponent / offset / fit metrics
# 5) extract flattened and unflattened bandpower
# 6) aggregate time-domain ERP across trials within sequence
# 7) extract CNV-like mean amplitude from -300 ms to 0 ms
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
from scipy.signal import butter, sosfiltfilt
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
SELECT_CHANNELS = []

TIME_WINDOW = (-1.4, 0.0)
CNV_WINDOW = (-0.3, 0.0)
MIN_TRIALS_PER_SEQUENCE = 6

# ERP/CNV branch settings.
# These affect only the saved ERP waveforms and cnv_mean, not PSD/FOOOF.
LOWPASS_ERP_FOR_SAVE_AND_CNV = True
ERP_LOWPASS_HZ = 30.0
ERP_LOWPASS_ORDER = 4

# PSD / FOOOF settings
DOWNSAMPLE_BEFORE_PSD = False
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

# Bandpower settings.
# delta is included to directly test whether unflattened low-frequency power
# accounts for the exponent effect.
BANDS = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta": (15, 30),
}

PSD_AGG_MODE = "mean"  # "mean" or "median"
RAW_BANDPOWER_LOG10 = False  # recommended for correlations with FOOOF exponent


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


def aggregate_erp(erp: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Aggregate time-domain data across trials within a sequence.

    Input shape:
        trials x channels x times

    Output shape:
        channels x times
    """
    if mode == "mean":
        return erp.mean(axis=0)
    if mode == "median":
        return np.median(erp, axis=0)
    raise ValueError(f"Unknown ERP aggregation mode: {mode}")


def lowpass_erp_array(
    x: np.ndarray,
    sfreq: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Low-pass filter ERP data along the last axis.

    This is intended only for the ERP/CNV branch. It should not be used for
    PSD/FOOOF input if the goal is to preserve broad-band spectral content.

    Input shape can be:
        trials x channels x times
        channels x times
    """
    if cutoff_hz is None or cutoff_hz <= 0:
        return x

    nyq = sfreq / 2.0
    if cutoff_hz >= nyq:
        return x

    sos = butter(
        N=order,
        Wn=cutoff_hz,
        btype="lowpass",
        fs=sfreq,
        output="sos",
    )
    return sosfiltfilt(sos, x, axis=-1)


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

    Output columns are named:
        <band>_flat
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


def compute_unflattened_bandpowers(
    psd_1d: np.ndarray,
    freqs: np.ndarray,
    bands: dict,
    log_transform: bool = True,
):
    """
    Compute unflattened bandpower from one PSD vector.

    If log_transform=True, returns mean log10(PSD) within each band.
    This is recommended for correlation with the FOOOF exponent because the
    exponent is fit in log10 power space.

    Output columns are named:
        <band>_raw
    """
    eps = np.finfo(float).tiny
    values = np.log10(np.maximum(psd_1d, eps)) if log_transform else psd_1d

    out = {}
    for band_name, (lo, hi) in bands.items():
        band_mask = (freqs >= lo) & (freqs <= hi)
        out[f"{band_name}_raw"] = (
            float(np.mean(values[band_mask])) if np.any(band_mask) else np.nan
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

    cnv_tidx = np.where(
        (erp_times_sec >= CNV_WINDOW[0]) & (erp_times_sec < CNV_WINDOW[1])
    )[0]

    if cnv_tidx.size == 0:
        raise ValueError("No samples found in requested CNV time window.")

    rows = []
    seq_psd = []
    seq_psd_index = []
    seq_erp = []
    seq_erp_index = []
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

        # trials x selected_channels x full epoch
        # Keep this broad-range copy for PSD/FOOOF.
        x_erp = erp_data[idx][:, SELECTED_CH_IDX, :]

        # ERP/CNV branch: optional low-pass copy for saved ERPs and cnv_mean.
        if LOWPASS_ERP_FOR_SAVE_AND_CNV:
            x_erp_for_cnv = lowpass_erp_array(
                x=x_erp,
                sfreq=sfreq_psd,
                cutoff_hz=ERP_LOWPASS_HZ,
                order=ERP_LOWPASS_ORDER,
            )
        else:
            x_erp_for_cnv = x_erp

        # trials x selected_channels x PSD time-window
        x_psd = x_erp[:, :, tidx]

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

        # Optional Welch robustness check. This is left commented to preserve
        # the multitaper primary analysis.
        # psd, freqs = psd_array_welch(
        #     x_psd,
        #     sfreq=sfreq_psd,
        #     fmin=FMIN_FIT,
        #     fmax=FMAX_FIT,
        #     n_fft=x_psd.shape[-1],
        #     n_per_seg=x_psd.shape[-1],
        #     n_overlap=0,
        #     average="mean",
        #     window="hamming",
        #     verbose=False,
        #     n_jobs=1,
        # )

        freqs_out = freqs

        psd_seq = aggregate_psd(psd, mode=PSD_AGG_MODE)  # channels x freqs
        seq_psd.append(psd_seq)

        erp_seq = aggregate_erp(x_erp_for_cnv, mode="mean")  # channels x times
        seq_erp.append(erp_seq)

        seq_index_row = {
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
            "erp_agg_mode": "mean",
            "erp_lowpass_for_save_and_cnv": LOWPASS_ERP_FOR_SAVE_AND_CNV,
            "erp_lowpass_hz": ERP_LOWPASS_HZ if LOWPASS_ERP_FOR_SAVE_AND_CNV else np.nan,
            "erp_lowpass_order": ERP_LOWPASS_ORDER if LOWPASS_ERP_FOR_SAVE_AND_CNV else np.nan,
        }
        seq_psd_index.append(seq_index_row.copy())
        seq_erp_index.append(seq_index_row.copy())

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

            flat_band_vals = compute_flattened_bandpowers(
                psd_1d=psd_seq[ch_ix_local],
                freqs=freqs,
                offset=float(offsets[ch_ix_local]),
                exponent=float(exponents[ch_ix_local]),
                bands=BANDS,
            )

            raw_band_vals = compute_unflattened_bandpowers(
                psd_1d=psd_seq[ch_ix_local],
                freqs=freqs,
                bands=BANDS,
                log_transform=RAW_BANDPOWER_LOG10,
            )

            cnv_mean = float(np.mean(erp_seq[ch_ix_local, cnv_tidx]))

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
                "delta_flat": flat_band_vals["delta_flat"],
                "theta_flat": flat_band_vals["theta_flat"],
                "alpha_flat": flat_band_vals["alpha_flat"],
                "beta_flat": flat_band_vals["beta_flat"],
                "delta_raw": raw_band_vals["delta_raw"],
                "theta_raw": raw_band_vals["theta_raw"],
                "alpha_raw": raw_band_vals["alpha_raw"],
                "beta_raw": raw_band_vals["beta_raw"],
                "cnv_mean": cnv_mean,
                "cnv_window_start": float(CNV_WINDOW[0]),
                "cnv_window_end": float(CNV_WINDOW[1]),
                "cnv_lowpass_for_save_and_cnv": LOWPASS_ERP_FOR_SAVE_AND_CNV,
                "cnv_lowpass_hz": ERP_LOWPASS_HZ if LOWPASS_ERP_FOR_SAVE_AND_CNV else np.nan,
                "cnv_lowpass_order": ERP_LOWPASS_ORDER if LOWPASS_ERP_FOR_SAVE_AND_CNV else np.nan,
                "raw_bandpower_log10": RAW_BANDPOWER_LOG10,
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
        return None, None, None, None, None, None, None

    return df_seq, seq_psd, seq_psd_index, freqs_out, seq_erp, seq_erp_index, erp_times_sec


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
    seq_erp: list,
    seq_erp_index: list,
    erp_times: np.ndarray,
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

    if seq_erp and erp_times is not None:
        np.savez_compressed(
            PATH_OUT / f"{subj_tag}_seq_erp_channelwise_car.npz",
            erp=np.stack(seq_erp),
            times=erp_times,
            channels=np.array(SELECTED_CH_NAMES),
        )

        pd.DataFrame(seq_erp_index).to_csv(
            PATH_OUT / f"{subj_tag}_seq_erp_channelwise_index_car.csv",
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

    (
        df_seq,
        seq_psd,
        seq_psd_index,
        freqs,
        seq_erp,
        seq_erp_index,
        erp_times,
    ) = run_pipeline(
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
        seq_erp=seq_erp,
        seq_erp_index=seq_erp_index,
        erp_times=erp_times,
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
