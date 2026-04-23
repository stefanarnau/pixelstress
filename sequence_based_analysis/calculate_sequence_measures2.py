# -----------------------------------------------------------------------------
# Sequence-based FOOOF + RT + flattened oscillatory bandpower + slow drift + TF
# CAR only
#
# Per subject x sequence x channel:
# 1) compute PSD trial-wise within sequence
# 2) aggregate PSD across trials within sequence
# 3) fit FOOOF to the aggregated PSD
# 4) extract exponent / offset / fit metrics / flattened bandpower
# 5) compute sequence-averaged ERP and slow-drift / CNV-like mean amplitude
# 6) compute sequence-averaged time-frequency power
# 7) extract summary TF measures from user-defined TF windows
#
# Key design:
# - PSD / FOOOF branch is separate from ERP branch
# - ERP branch can be low-pass filtered before ERP averaging
# - TF branch is separate and stores full sequence TF matrices
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
from mne.time_frequency import psd_array_multitaper, tfr_array_morlet


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned2/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/3_sequence_data2/")
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

# Time window from which sequence measures are extracted
TIME_WINDOW = (-1.4, 0.0)
MIN_TRIALS_PER_SEQUENCE = 5

# -------------------------------------------------------------------------
# ERP / slow-drift settings
# -------------------------------------------------------------------------
SLOW_DRIFT_WINDOW = (-0.5, 0.0)
LOWPASS_ERP_BEFORE_AVG = True
ERP_LOWPASS_HZ = 20.0

# -------------------------------------------------------------------------
# PSD / FOOOF settings
# -------------------------------------------------------------------------
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
ERP_AGG_MODE = "mean"  # "mean" or "median"

# -------------------------------------------------------------------------
# Time-frequency settings
# -------------------------------------------------------------------------
COMPUTE_TF = True

TF_METHOD = "morlet"
TF_FMIN = 2.0
TF_FMAX = 30.0
TF_N_FREQS = 15
TF_FREQS_LOG = True

# If "scaled": n_cycles = freqs * TF_N_CYCLES_FACTOR
# Else: a float, e.g. 5.0
TF_N_CYCLES = "scaled"
TF_N_CYCLES_FACTOR = 0.6

TF_OUTPUT = "power"  # power only
TF_AGG_MODE = "mean"  # "mean" or "median"

# User-defined summary windows:
# Each dict defines one dataframe column named "tf_<name>"
TF_WINDOWS = [
    {"name": "delta_pre", "fmin": 2.0, "fmax": 4.0, "tmin": -1, "tmax": -0.2},
    {"name": "theta_pre", "fmin": 4.0, "fmax": 7.0, "tmin": -1, "tmax": -0.2},
    {"name": "alpha_pre", "fmin": 8.0, "fmax": 13.0, "tmin": -1, "tmax": -0.2},
    {"name": "beta_pre", "fmin": 16.0, "fmax": 30.0, "tmin": -1, "tmax": -0.2},
]

TF_LOG_POWER = True
TF_LOG_POWER_SCALE = "db"  # "db" or "log10"


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
    if mode == "mean":
        return psd.mean(axis=0)
    if mode == "median":
        return np.median(psd, axis=0)
    raise ValueError(f"Unknown PSD_AGG_MODE: {mode}")


def aggregate_erp(x: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        return x.mean(axis=0)
    if mode == "median":
        return np.median(x, axis=0)
    raise ValueError(f"Unknown ERP_AGG_MODE: {mode}")


def aggregate_tf(tf: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Aggregate trial-level TF power across trials.
    Input shape: trials x channels x freqs x times
    Output shape: channels x freqs x times
    """
    if mode == "mean":
        return tf.mean(axis=0)
    if mode == "median":
        return np.median(tf, axis=0)
    raise ValueError(f"Unknown TF_AGG_MODE: {mode}")


def prepare_epochs_for_psd(epochs: mne.Epochs) -> mne.Epochs:
    if DOWNSAMPLE_BEFORE_PSD:
        return epochs.copy().resample(
            TARGET_SFREQ,
            npad="auto",
            verbose=False,
        )
    return epochs


def prepare_epochs_for_erp(epochs: mne.Epochs) -> mne.Epochs:
    ep = epochs.copy()

    if LOWPASS_ERP_BEFORE_AVG:
        ep = ep.filter(
            l_freq=None,
            h_freq=ERP_LOWPASS_HZ,
            picks="eeg",
            fir_design="firwin",
            phase="zero",
            verbose=False,
        )

    return ep


def prepare_epochs_for_tf(epochs: mne.Epochs) -> mne.Epochs:
    """
    Separate branch for TF.
    Currently uses original sampling and no extra filter by default.
    """
    return epochs.copy()


def make_tf_freqs():
    if TF_FREQS_LOG:
        return np.logspace(np.log10(TF_FMIN), np.log10(TF_FMAX), TF_N_FREQS)
    return np.linspace(TF_FMIN, TF_FMAX, TF_N_FREQS)


def make_tf_n_cycles(freqs: np.ndarray):
    if TF_N_CYCLES == "scaled":
        return freqs * TF_N_CYCLES_FACTOR
    if isinstance(TF_N_CYCLES, (int, float)):
        return float(TF_N_CYCLES)
    raise ValueError("TF_N_CYCLES must be 'scaled' or a numeric value.")


def compute_flattened_bandpowers(
    psd_1d: np.ndarray,
    freqs: np.ndarray,
    offset: float,
    exponent: float,
    bands: dict,
):
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


def compute_slow_drift_mean(
    erp_1d: np.ndarray,
    times_sec: np.ndarray,
    window: tuple[float, float],
) -> float:
    mask = (times_sec >= window[0]) & (times_sec < window[1])
    if not np.any(mask):
        return np.nan
    return float(np.mean(erp_1d[mask]))


def compute_tf_window_measures(
    tf_2d: np.ndarray,
    freqs: np.ndarray,
    times_sec: np.ndarray,
    tf_windows: list[dict],
):
    """
    Compute summary TF measures from one sequence-level TF matrix.

    Parameters
    ----------
    tf_2d : np.ndarray
        Shape: freqs x times
    freqs : np.ndarray
    times_sec : np.ndarray
    tf_windows : list of dicts
        Each dict needs: name, fmin, fmax, tmin, tmax

    Returns
    -------
    dict
        e.g. {"tf_alpha_pre": 0.123, ...}
    """
    out = {}

    for win in tf_windows:
        name = win["name"]
        fmin = float(win["fmin"])
        fmax = float(win["fmax"])
        tmin = float(win["tmin"])
        tmax = float(win["tmax"])

        fmask = (freqs >= fmin) & (freqs <= fmax)
        tmask = (times_sec >= tmin) & (times_sec < tmax)

        col = f"tf_{name}"

        if not np.any(fmask) or not np.any(tmask):
            out[col] = np.nan
            continue

        out[col] = float(np.mean(tf_2d[np.ix_(fmask, tmask)]))

    return out


def transform_tf_power(
    tf: np.ndarray, use_log: bool = True, scale: str = "db"
) -> np.ndarray:
    """
    Transform TF power to log scale.

    Parameters
    ----------
    tf : np.ndarray
        Power array, e.g. trials x channels x freqs x times
    use_log : bool
        Whether to apply log transform
    scale : str
        "db"   -> 10 * log10(power)
        "log10" -> log10(power)

    Returns
    -------
    np.ndarray
        Transformed TF array
    """
    if not use_log:
        return tf

    eps = np.finfo(float).tiny
    tf = np.maximum(tf, eps)

    if scale == "db":
        return 10.0 * np.log10(tf)
    if scale == "log10":
        return np.log10(tf)

    raise ValueError(f"Unknown TF_LOG_POWER_SCALE: {scale}")


# -----------------------------------------------------------------------------
# Core extraction
# -----------------------------------------------------------------------------
def extract_sequence_measures(
    erp_data_psd: np.ndarray,
    erp_times_psd_sec: np.ndarray,
    erp_data_erp: np.ndarray,
    erp_times_erp_sec: np.ndarray,
    erp_data_tf: np.ndarray,
    erp_times_tf_sec: np.ndarray,
    df_trials: pd.DataFrame,
    subj_id: int,
    sfreq_psd: float,
    sfreq_erp: float,
    sfreq_tf: float,
):
    tidx_psd = np.where(
        (erp_times_psd_sec >= TIME_WINDOW[0]) & (erp_times_psd_sec < TIME_WINDOW[1])
    )[0]

    tidx_erp = np.where(
        (erp_times_erp_sec >= TIME_WINDOW[0]) & (erp_times_erp_sec < TIME_WINDOW[1])
    )[0]

    tidx_tf = np.where(
        (erp_times_tf_sec >= TIME_WINDOW[0]) & (erp_times_tf_sec < TIME_WINDOW[1])
    )[0]

    if tidx_psd.size == 0:
        raise ValueError("No samples found in requested PSD time window.")
    if tidx_erp.size == 0:
        raise ValueError("No samples found in requested ERP time window.")
    if COMPUTE_TF and tidx_tf.size == 0:
        raise ValueError("No samples found in requested TF time window.")

    erp_times_win = erp_times_erp_sec[tidx_erp]
    tf_times_win = erp_times_tf_sec[tidx_tf]

    tf_freqs = make_tf_freqs() if COMPUTE_TF else None
    tf_n_cycles = make_tf_n_cycles(tf_freqs) if COMPUTE_TF else None

    grouped = df_trials.groupby(["block_nr", "sequence_nr"], sort=True)

    rows = []
    seq_psd = []
    seq_psd_index = []
    seq_erp = []
    seq_erp_index = []
    seq_tf = []
    seq_tf_index = []
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

        # ---------------------------------------------------------------------
        # PSD branch
        # ---------------------------------------------------------------------
        x_psd = erp_data_psd[idx][:, SELECTED_CH_IDX, :][:, :, tidx_psd]

        # ---------------------------------------------------------------------
        # ERP branch
        # ---------------------------------------------------------------------
        x_erp = erp_data_erp[idx][:, SELECTED_CH_IDX, :][:, :, tidx_erp]

        # ---------------------------------------------------------------------
        # TF branch
        # ---------------------------------------------------------------------
        if COMPUTE_TF:
            x_tf = erp_data_tf[idx][:, SELECTED_CH_IDX, :][:, :, tidx_tf]
        else:
            x_tf = None

        # ---------------------------------------------------------------------
        # Sequence ERP
        # ---------------------------------------------------------------------
        erp_seq = aggregate_erp(x_erp, mode=ERP_AGG_MODE)  # channels x time
        seq_erp.append(erp_seq)
        seq_erp_index.append(
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
                "sfreq_erp": float(sfreq_erp),
                "erp_agg_mode": ERP_AGG_MODE,
                "erp_lowpass_before_avg": bool(LOWPASS_ERP_BEFORE_AVG),
                "erp_lowpass_hz": (
                    float(ERP_LOWPASS_HZ) if LOWPASS_ERP_BEFORE_AVG else np.nan
                ),
                "time_window_start": float(TIME_WINDOW[0]),
                "time_window_end": float(TIME_WINDOW[1]),
                "slow_drift_window_start": float(SLOW_DRIFT_WINDOW[0]),
                "slow_drift_window_end": float(SLOW_DRIFT_WINDOW[1]),
            }
        )

        # ---------------------------------------------------------------------
        # Sequence PSD
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Sequence TF
        # ---------------------------------------------------------------------
        if COMPUTE_TF:
            if TF_METHOD != "morlet":
                raise ValueError(f"Unsupported TF_METHOD: {TF_METHOD}")

            tf = tfr_array_morlet(
                x_tf,
                sfreq=sfreq_tf,
                freqs=tf_freqs,
                n_cycles=tf_n_cycles,
                output=TF_OUTPUT,
                zero_mean=True,
                use_fft=True,
                decim=1,
                n_jobs=1,
                verbose=False,
            )
            # shape: trials x channels x freqs x times

            tf = transform_tf_power(
                tf,
                use_log=TF_LOG_POWER,
                scale=TF_LOG_POWER_SCALE,
            )

            tf_seq = aggregate_tf(tf, mode=TF_AGG_MODE)  # channels x freqs x times
            seq_tf.append(tf_seq)
            seq_tf_index.append(
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
                    "sfreq_tf": float(sfreq_tf),
                    "tf_method": TF_METHOD,
                    "tf_agg_mode": TF_AGG_MODE,
                    "tf_fmin": float(TF_FMIN),
                    "tf_fmax": float(TF_FMAX),
                    "tf_n_freqs": int(TF_N_FREQS),
                    "tf_log_power": bool(TF_LOG_POWER),
                    "tf_log_power_scale": TF_LOG_POWER_SCALE if TF_LOG_POWER else "none",
                    "time_window_start": float(TIME_WINDOW[0]),
                    "time_window_end": float(TIME_WINDOW[1]),
                }
            )
        else:
            tf_seq = None

        # ---------------------------------------------------------------------
        # FOOOF on sequence-aggregated PSD
        # ---------------------------------------------------------------------
        fg = fooof.FOOOFGroup(**FOOOF_KWARGS)
        fg.fit(freqs, psd_seq, [FMIN_FIT, FMAX_FIT])

        aperiodic = fg.get_params("aperiodic_params")
        r2_vals = fg.get_params("r_squared")
        err_vals = fg.get_params("error")

        offsets = aperiodic[:, 0]
        exponents = aperiodic[:, 1]

        # ---------------------------------------------------------------------
        # Build output rows
        # ---------------------------------------------------------------------
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

            slow_drift_mean = compute_slow_drift_mean(
                erp_1d=erp_seq[ch_ix_local],
                times_sec=erp_times_win,
                window=SLOW_DRIFT_WINDOW,
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
                "slow_drift_mean": slow_drift_mean,
                "fmin_fit": FMIN_FIT,
                "fmax_fit": FMAX_FIT,
                "mt_bandwidth": MT_BANDWIDTH,
                "sfreq_psd": float(sfreq_psd),
                "sfreq_erp": float(sfreq_erp),
                "sfreq_tf": float(sfreq_tf) if COMPUTE_TF else np.nan,
                "psd_agg_mode": PSD_AGG_MODE,
                "erp_agg_mode": ERP_AGG_MODE,
                "erp_lowpass_before_avg": bool(LOWPASS_ERP_BEFORE_AVG),
                "erp_lowpass_hz": (
                    float(ERP_LOWPASS_HZ) if LOWPASS_ERP_BEFORE_AVG else np.nan
                ),
                "mt_normalization": MT_NORMALIZATION,
                "time_window_start": float(TIME_WINDOW[0]),
                "time_window_end": float(TIME_WINDOW[1]),
                "slow_drift_window_start": float(SLOW_DRIFT_WINDOW[0]),
                "slow_drift_window_end": float(SLOW_DRIFT_WINDOW[1]),
                "reference": "car",
            }

            if COMPUTE_TF:
                tf_measures = compute_tf_window_measures(
                    tf_2d=tf_seq[ch_ix_local],
                    freqs=tf_freqs,
                    times_sec=tf_times_win,
                    tf_windows=TF_WINDOWS,
                )
                row.update(tf_measures)

            rows.append(row)

    df_seq = pd.DataFrame(rows)

    if df_seq.empty:
        return None, None, None, None, None, None, None, None, None

    return (
        df_seq,
        seq_psd,
        seq_psd_index,
        seq_erp,
        seq_erp_index,
        seq_tf,
        seq_tf_index,
        freqs_out,
        erp_times_win,
        tf_freqs,
        tf_times_win,
    )


def run_pipeline(
    epochs: mne.Epochs,
    df_trials: pd.DataFrame,
    subj_id: int,
):
    epochs_psd = prepare_epochs_for_psd(epochs)
    epochs_erp = prepare_epochs_for_erp(epochs)
    epochs_tf = prepare_epochs_for_tf(epochs)

    erp_data_psd = epochs_psd.get_data(copy=True)
    erp_data_erp = epochs_erp.get_data(copy=True)
    erp_data_tf = epochs_tf.get_data(copy=True)

    erp_times_psd = epochs_psd.times.copy()
    erp_times_erp = epochs_erp.times.copy()
    erp_times_tf = epochs_tf.times.copy()

    sfreq_psd = float(epochs_psd.info["sfreq"])
    sfreq_erp = float(epochs_erp.info["sfreq"])
    sfreq_tf = float(epochs_tf.info["sfreq"])

    return extract_sequence_measures(
        erp_data_psd=erp_data_psd,
        erp_times_psd_sec=erp_times_psd,
        erp_data_erp=erp_data_erp,
        erp_times_erp_sec=erp_times_erp,
        erp_data_tf=erp_data_tf,
        erp_times_tf_sec=erp_times_tf,
        df_trials=df_trials,
        subj_id=subj_id,
        sfreq_psd=sfreq_psd,
        sfreq_erp=sfreq_erp,
        sfreq_tf=sfreq_tf,
    )


def save_outputs(
    df_seq: pd.DataFrame,
    seq_psd: list,
    seq_psd_index: list,
    seq_erp: list,
    seq_erp_index: list,
    seq_tf: list,
    seq_tf_index: list,
    freqs: np.ndarray,
    erp_times: np.ndarray,
    tf_freqs: np.ndarray,
    tf_times: np.ndarray,
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

    if COMPUTE_TF and seq_tf and tf_freqs is not None and tf_times is not None:
        np.savez_compressed(
            PATH_OUT / f"{subj_tag}_seq_tf_channelwise_car.npz",
            tf=np.stack(seq_tf),  # sequences x channels x freqs x times
            freqs=tf_freqs,
            times=tf_times,
            channels=np.array(SELECTED_CH_NAMES),
        )
        pd.DataFrame(seq_tf_index).to_csv(
            PATH_OUT / f"{subj_tag}_seq_tf_channelwise_index_car.csv",
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
        seq_erp,
        seq_erp_index,
        seq_tf,
        seq_tf_index,
        freqs,
        erp_times_win,
        tf_freqs,
        tf_times_win,
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
        seq_erp=seq_erp,
        seq_erp_index=seq_erp_index,
        seq_tf=seq_tf,
        seq_tf_index=seq_tf_index,
        freqs=freqs,
        erp_times=erp_times_win,
        tf_freqs=tf_freqs,
        tf_times=tf_times_win,
        subj_id=subj_id,
    )

    print(f"Saved subject {subj_id:03d} (car)")
    return df_seq


# -----------------------------------------------------------------------------
# Run extraction
# -----------------------------------------------------------------------------
seq_data = Parallel(
    n_jobs=8,
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
print("ERP lowpass before avg:", LOWPASS_ERP_BEFORE_AVG)
print("ERP lowpass Hz:", ERP_LOWPASS_HZ if LOWPASS_ERP_BEFORE_AVG else "not applied")
print("TF enabled:", COMPUTE_TF)
if COMPUTE_TF:
    print("TF method:", TF_METHOD)
    print("TF freq range:", (TF_FMIN, TF_FMAX))
    print("TF n freqs:", TF_N_FREQS)
    print("TF windows:", [f"tf_{w['name']}" for w in TF_WINDOWS])
print(
    "Sequences:",
    df_all[["id", "block_nr", "sequence_nr"]].drop_duplicates().shape[0],
)
