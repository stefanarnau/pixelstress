# -----------------------------------------------------------------------------
# Inspect PSD + FOOOF for one subject / one sequence / one channel
# Compare:
#   1) Averaged PSD -> FOOOF
#   2) Trialwise FOOOF -> average extracted outputs
#
# Purpose
# -------
# Diagnostic script for evaluating whether multitaper PSD + FOOOF settings are
# appropriate, and for visualizing the difference between:
#
#   - average first, then FOOOF
#   - FOOOF first, then average
#
# This script:
#   - loads one subject
#   - selects one sequence (block_nr, sequence_nr)
#   - selects one channel
#   - computes trialwise multitaper PSDs
#   - compares the two approaches in a 2x2 figure
# -----------------------------------------------------------------------------

from pathlib import Path

import fooof
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.io
from mne.time_frequency import psd_array_multitaper


# =============================================================================
# SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned_45/")
CHANNEL_LABELS_FILE = Path("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt")

# -----------------------------------------------------------------------------
# Subject / sequence / channel selection
# -----------------------------------------------------------------------------
SUBJECT_ID = 18
BLOCK_NR = 2
SEQUENCE_NR = 8
CHANNEL_NAME = "FCz"

# -----------------------------------------------------------------------------
# Trial filtering: match main pipeline
# -----------------------------------------------------------------------------
DROP_INCORRECT_TRIALS = True
DROP_SEQUENCE_1 = True

# -----------------------------------------------------------------------------
# Exclusions: match main pipeline
# -----------------------------------------------------------------------------
IDS_TO_DROP = {1, 2, 3, 4, 5, 6, 13, 17, 25, 40, 49, 83}

# -----------------------------------------------------------------------------
# Reference scheme
# -----------------------------------------------------------------------------
# Loaded data are assumed already CAR-referenced.
REFERENCE_SCHEME = "car"

# -----------------------------------------------------------------------------
# Optional downsampling before PSD
# -----------------------------------------------------------------------------
DOWNSAMPLE_BEFORE_PSD = True
TARGET_SFREQ = 250

# -----------------------------------------------------------------------------
# Time window
# -----------------------------------------------------------------------------
WINDOW_START = -2.0
WINDOW_END = 0.0

# -----------------------------------------------------------------------------
# PSD settings
# -----------------------------------------------------------------------------
FMIN_FIT = 1.0
FMAX_FIT = 40.0
MT_BANDWIDTH = 1.5
PSD_AGG_MODE = "mean"   # "mean" or "median"
MT_NORMALIZATION = "full"

# -----------------------------------------------------------------------------
# FOOOF settings
# -----------------------------------------------------------------------------
FOOOF_KWARGS = dict(
    aperiodic_mode="fixed",
    peak_width_limits=(2, 12),
    max_n_peaks=8,
    min_peak_height=0.05,
    verbose=False,
)

# -----------------------------------------------------------------------------
# Bands for flattened summaries
# -----------------------------------------------------------------------------
BANDS = {
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta": (15, 30),
}

# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------
FIGSIZE = (14, 9)
SHOW_INDIVIDUAL_TRIAL_PSDS = True
SHOW_INDIVIDUAL_TRIAL_FOOOF_FITS = False
TRIAL_LINE_ALPHA = 0.20
N_DECIMALS = 4


# =============================================================================
# HELPERS
# =============================================================================

def aggregate_psd(psd: np.ndarray, mode: str = "mean") -> np.ndarray:
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


def get_window_sample_indices(times_sec: np.ndarray, window_start: float, window_end: float):
    return np.where((times_sec >= window_start) & (times_sec < window_end))[0]


def format_float(x, n=N_DECIMALS):
    if x is None:
        return "nan"
    try:
        if np.isnan(x):
            return "nan"
    except TypeError:
        pass
    return f"{x:.{n}f}"


def compute_flattened_bandpowers(psd_1d, freqs, offset, exponent, bands):
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


def fit_fooof_single_spectrum(psd_1d, freqs):
    fm = fooof.FOOOF(**FOOOF_KWARGS)
    fm.fit(freqs, psd_1d, [FMIN_FIT, FMAX_FIT])

    aperiodic = fm.get_params("aperiodic_params")
    offset = float(aperiodic[0])
    exponent = float(aperiodic[1])
    r2 = float(fm.get_params("r_squared"))
    err = float(fm.get_params("error"))

    band_vals = compute_flattened_bandpowers(
        psd_1d=psd_1d,
        freqs=freqs,
        offset=offset,
        exponent=exponent,
        bands=BANDS,
    )

    peak_params = fm.get_params("peak_params")
    if peak_params is None or len(np.atleast_2d(peak_params)) == 0:
        n_peaks = 0
    else:
        n_peaks = np.atleast_2d(peak_params).shape[0]

    return {
        "fm": fm,
        "offset": offset,
        "exponent": exponent,
        "r2": r2,
        "error": err,
        "n_peaks": n_peaks,
        "theta_flat": band_vals["theta_flat"],
        "alpha_flat": band_vals["alpha_flat"],
        "beta_flat": band_vals["beta_flat"],
    }


# =============================================================================
# LOAD CHANNELS / INFO
# =============================================================================

CHANNEL_LABELS = CHANNEL_LABELS_FILE.read_text().splitlines()

INFO_ERP = mne.create_info(CHANNEL_LABELS, sfreq=500, ch_types="eeg", verbose=None)
MONTAGE = mne.channels.make_standard_montage("standard_1020")
INFO_ERP.set_montage(MONTAGE, on_missing="warn", match_case=False)


# =============================================================================
# FIND DATASET FOR SUBJECT
# =============================================================================

DATASETS = sorted(PATH_IN.glob("*erp.set"))

dataset_match = None
trialinfo_match = None

for dataset in DATASETS:
    base = str(dataset).split("_cleaned")[0]
    trialinfo_file = Path(base + "_erp_trialinfo.csv")

    if not trialinfo_file.exists():
        continue

    df_tmp = pd.read_csv(trialinfo_file)
    subj_id_tmp = int(df_tmp["id"].iloc[0])

    if subj_id_tmp == SUBJECT_ID:
        dataset_match = dataset
        trialinfo_match = trialinfo_file
        break

if dataset_match is None:
    raise FileNotFoundError(f"Could not find dataset for SUBJECT_ID={SUBJECT_ID}")

if SUBJECT_ID in IDS_TO_DROP:
    raise ValueError(f"SUBJECT_ID={SUBJECT_ID} is in IDS_TO_DROP")


# =============================================================================
# LOAD SUBJECT DATA
# =============================================================================

df_trials = pd.read_csv(trialinfo_match)

mat = scipy.io.loadmat(dataset_match)

erp_data = np.transpose(mat["data"], [2, 0, 1])  # trials x channels x time
erp_times = mat["times"].ravel().astype(float)
erp_times_sec = erp_times / 1000 if np.nanmax(np.abs(erp_times)) > 20 else erp_times

if erp_data.shape[1] != len(CHANNEL_LABELS):
    raise ValueError(
        f"Channel mismatch: {erp_data.shape[1]} vs {len(CHANNEL_LABELS)}"
    )

df_trials["accuracy"] = (df_trials["accuracy"] == 1).astype(int)
df_trials = df_trials.rename(columns={"session_condition": "group"})
df_trials["group"] = df_trials["group"].replace({1: "experimental", 2: "control"})

if DROP_SEQUENCE_1:
    keep_mask = df_trials["sequence_nr"] > 1
    df_trials = df_trials.loc[keep_mask].reset_index(drop=True)
    erp_data = erp_data[keep_mask, :, :]

if DROP_INCORRECT_TRIALS:
    keep_mask = df_trials["accuracy"] == 1
    df_trials = df_trials.loc[keep_mask].reset_index(drop=True)
    erp_data = erp_data[keep_mask, :, :]

if len(df_trials) != erp_data.shape[0]:
    raise ValueError(
        f"Trial/data mismatch: {len(df_trials)} rows vs {erp_data.shape[0]} trials"
    )

epochs = mne.EpochsArray(
    data=erp_data,
    info=INFO_ERP.copy(),
    tmin=float(erp_times_sec[0]),
    baseline=None,
    verbose=False,
)

epochs_psd = prepare_epochs_for_psd(epochs)

erp_data_ref = epochs_psd.get_data(copy=True)
erp_times_sec_ref = epochs_psd.times.copy()
sfreq_psd = float(epochs_psd.info["sfreq"])


# =============================================================================
# SELECT SEQUENCE / WINDOW / CHANNEL
# =============================================================================

channel_idx = CHANNEL_LABELS.index(CHANNEL_NAME)

seq_mask = (
    (df_trials["block_nr"] == BLOCK_NR) &
    (df_trials["sequence_nr"] == SEQUENCE_NR)
)
seq_idx = np.where(seq_mask.to_numpy())[0]

if len(seq_idx) == 0:
    raise ValueError(
        f"No trials found for block_nr={BLOCK_NR}, sequence_nr={SEQUENCE_NR}"
    )

tidx = get_window_sample_indices(erp_times_sec_ref, WINDOW_START, WINDOW_END)
if tidx.size == 0:
    raise ValueError(
        f"No samples found in window [{WINDOW_START}, {WINDOW_END})"
    )

dt = np.diff(erp_times_sec_ref[:2])[0]
epoch_tmin = float(erp_times_sec_ref.min())
epoch_tmax = float(erp_times_sec_ref.max())
if WINDOW_START < epoch_tmin or WINDOW_END > (epoch_tmax + dt):
    raise ValueError(
        f"Window [{WINDOW_START}, {WINDOW_END}) exceeds epoch support "
        f"[{epoch_tmin}, {epoch_tmax + dt})"
    )

# trials x time for one channel
x = erp_data_ref[seq_idx][:, channel_idx, tidx]

if x.ndim != 2:
    raise RuntimeError(f"Unexpected shape for selected data: {x.shape}")

n_trials = x.shape[0]

print(f"Selected subject: {SUBJECT_ID}")
print(f"Selected sequence: block_nr={BLOCK_NR}, sequence_nr={SEQUENCE_NR}")
print(f"Selected channel: {CHANNEL_NAME}")
print(f"Reference scheme: {REFERENCE_SCHEME}")
print(f"Window: [{WINDOW_START}, {WINDOW_END}) s")
print(f"Trials in sequence after filtering: {n_trials}")
print(f"Sampling rate for PSD: {sfreq_psd} Hz")


# =============================================================================
# COMPUTE MULTITAPER PSD
# =============================================================================

psd, freqs = psd_array_multitaper(
    x,
    sfreq=sfreq_psd,
    fmin=FMIN_FIT,
    fmax=FMAX_FIT,
    bandwidth=MT_BANDWIDTH,
    normalization=MT_NORMALIZATION,
    verbose=False,
    n_jobs=1,
)
# psd shape: trials x freqs

psd_agg = aggregate_psd(psd, mode=PSD_AGG_MODE)


# =============================================================================
# METHOD 1: AVERAGED PSD -> FOOOF
# =============================================================================

fm_avg = fooof.FOOOF(**FOOOF_KWARGS)
fm_avg.fit(freqs, psd_agg, [FMIN_FIT, FMAX_FIT])

aperiodic_avg = fm_avg.get_params("aperiodic_params")
offset_avg = float(aperiodic_avg[0])
exponent_avg = float(aperiodic_avg[1])
r2_avg = float(fm_avg.get_params("r_squared"))
err_avg = float(fm_avg.get_params("error"))

peak_params_avg = fm_avg.get_params("peak_params")
if peak_params_avg is None or len(np.atleast_2d(peak_params_avg)) == 0:
    n_peaks_avg = 0
else:
    n_peaks_avg = np.atleast_2d(peak_params_avg).shape[0]

band_vals_avg = compute_flattened_bandpowers(
    psd_1d=psd_agg,
    freqs=freqs,
    offset=offset_avg,
    exponent=exponent_avg,
    bands=BANDS,
)

avg_obs_log = np.log10(np.maximum(psd_agg, np.finfo(float).tiny))
avg_model_log = fm_avg.fooofed_spectrum_.copy()
avg_ap_log = fm_avg._ap_fit.copy()


# =============================================================================
# METHOD 2: FOOOF PER TRIAL -> AVERAGE OUTPUTS
# =============================================================================

trial_results = []
trial_obs_log = []
trial_model_log = []
trial_ap_log = []

for trial_ix in range(psd.shape[0]):
    res = fit_fooof_single_spectrum(psd[trial_ix], freqs)
    fm = res["fm"]

    trial_results.append(
        {
            "offset": res["offset"],
            "exponent": res["exponent"],
            "r2": res["r2"],
            "error": res["error"],
            "n_peaks": res["n_peaks"],
            "theta_flat": res["theta_flat"],
            "alpha_flat": res["alpha_flat"],
            "beta_flat": res["beta_flat"],
        }
    )

    trial_obs_log.append(np.log10(np.maximum(psd[trial_ix], np.finfo(float).tiny)))
    trial_model_log.append(fm.fooofed_spectrum_.copy())
    trial_ap_log.append(fm._ap_fit.copy())

df_trial = pd.DataFrame(trial_results)

trialavg_offset = float(df_trial["offset"].mean())
trialavg_exponent = float(df_trial["exponent"].mean())
trialavg_r2 = float(df_trial["r2"].mean())
trialavg_error = float(df_trial["error"].mean())
trialavg_n_peaks = float(df_trial["n_peaks"].mean())

trialavg_theta = float(df_trial["theta_flat"].mean())
trialavg_alpha = float(df_trial["alpha_flat"].mean())
trialavg_beta = float(df_trial["beta_flat"].mean())

mean_trial_obs_log = np.mean(np.vstack(trial_obs_log), axis=0)
mean_trial_model_log = np.mean(np.vstack(trial_model_log), axis=0)
mean_trial_ap_log = np.mean(np.vstack(trial_ap_log), axis=0)


# =============================================================================
# PLOT
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)

fig.suptitle(
    f"sub={SUBJECT_ID} | block={BLOCK_NR} seq={SEQUENCE_NR} | ch={CHANNEL_NAME} | "
    f"ref={REFERENCE_SCHEME} | window=[{WINDOW_START}, {WINDOW_END}) | "
    f"MT bandwidth={MT_BANDWIDTH}",
    fontsize=14,
)

# -------------------------------------------------------------------------
# Top-left: linear PSD, averaged first
# -------------------------------------------------------------------------
ax = axes[0, 0]
if SHOW_INDIVIDUAL_TRIAL_PSDS:
    for i in range(psd.shape[0]):
        ax.plot(freqs, psd[i], alpha=TRIAL_LINE_ALPHA, linewidth=1)

ax.plot(freqs, psd_agg, linewidth=2.5, label=f"Aggregated PSD ({PSD_AGG_MODE})")
ax.set_title("Average first: linear PSD")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.legend(loc="best")

# -------------------------------------------------------------------------
# Bottom-left: log10 PSD + FOOOF on aggregated PSD
# -------------------------------------------------------------------------
ax = axes[1, 0]
ax.plot(freqs, avg_obs_log, linewidth=2.5, label="Observed log10 PSD")
ax.plot(freqs, avg_model_log, linewidth=2.5, label="FOOOF full model")
ax.plot(freqs, avg_ap_log, linestyle="--", linewidth=2.5, label="Aperiodic fit")

fit_text = (
    f"R² = {format_float(r2_avg)}\n"
    f"Error = {format_float(err_avg)}\n"
    f"Offset = {format_float(offset_avg)}\n"
    f"Exponent = {format_float(exponent_avg)}\n"
    f"N peaks = {n_peaks_avg}\n"
    f"Theta flat = {format_float(band_vals_avg['theta_flat'])}\n"
    f"Alpha flat = {format_float(band_vals_avg['alpha_flat'])}\n"
    f"Beta flat = {format_float(band_vals_avg['beta_flat'])}"
)

ax.text(
    0.98, 0.98, fit_text,
    transform=ax.transAxes,
    ha="right", va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)

ax.set_title("Average first: FOOOF on aggregated PSD")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("log10(Power)")
ax.legend(loc="best")

# -------------------------------------------------------------------------
# Top-right: linear PSDs + average trialwise model
# -------------------------------------------------------------------------
ax = axes[0, 1]
if SHOW_INDIVIDUAL_TRIAL_PSDS:
    for i in range(psd.shape[0]):
        ax.plot(freqs, psd[i], alpha=TRIAL_LINE_ALPHA, linewidth=1)

# mean trialwise model back-transformed to linear power
ax.plot(freqs, 10 ** mean_trial_model_log, linewidth=2.5, label="Mean trialwise FOOOF model")
ax.plot(freqs, 10 ** mean_trial_ap_log, linestyle="--", linewidth=2.5, label="Mean trialwise aperiodic")

ax.set_title("FOOOF first: mean trialwise models (linear)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.legend(loc="best")

# optional individual trial fooof fits in linear power
if SHOW_INDIVIDUAL_TRIAL_FOOOF_FITS:
    for arr in trial_model_log:
        ax.plot(freqs, 10 ** arr, alpha=0.12, linewidth=1)

# -------------------------------------------------------------------------
# Bottom-right: mean trial log PSD + mean trialwise model
# -------------------------------------------------------------------------
ax = axes[1, 1]
ax.plot(freqs, mean_trial_obs_log, linewidth=2.5, label="Mean trial log10 PSD")
ax.plot(freqs, mean_trial_model_log, linewidth=2.5, label="Mean trialwise FOOOF model")
ax.plot(freqs, mean_trial_ap_log, linestyle="--", linewidth=2.5, label="Mean trialwise aperiodic")

fit_text = (
    f"Mean R² = {format_float(trialavg_r2)}\n"
    f"Mean error = {format_float(trialavg_error)}\n"
    f"Mean offset = {format_float(trialavg_offset)}\n"
    f"Mean exponent = {format_float(trialavg_exponent)}\n"
    f"Mean N peaks = {format_float(trialavg_n_peaks)}\n"
    f"Mean theta flat = {format_float(trialavg_theta)}\n"
    f"Mean alpha flat = {format_float(trialavg_alpha)}\n"
    f"Mean beta flat = {format_float(trialavg_beta)}"
)

ax.text(
    0.98, 0.98, fit_text,
    transform=ax.transAxes,
    ha="right", va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)

ax.set_title("FOOOF first: average extracted trialwise outputs")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("log10(Power)")
ax.legend(loc="best")

plt.show()


# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("\n--- Summary: average first ---")
print(f"Offset: {offset_avg}")
print(f"Exponent: {exponent_avg}")
print(f"R²: {r2_avg}")
print(f"Error: {err_avg}")
print(f"N peaks: {n_peaks_avg}")
print(f"Theta flat: {band_vals_avg['theta_flat']}")
print(f"Alpha flat: {band_vals_avg['alpha_flat']}")
print(f"Beta flat: {band_vals_avg['beta_flat']}")

print("\n--- Summary: FOOOF first then average ---")
print(f"Mean offset: {trialavg_offset}")
print(f"Mean exponent: {trialavg_exponent}")
print(f"Mean R²: {trialavg_r2}")
print(f"Mean error: {trialavg_error}")
print(f"Mean N peaks: {trialavg_n_peaks}")
print(f"Mean theta flat: {trialavg_theta}")
print(f"Mean alpha flat: {trialavg_alpha}")
print(f"Mean beta flat: {trialavg_beta}")