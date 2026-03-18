# -----------------------------------------------------------------------------
# Sequence-based FOOOF + RT + flattened oscillatory bandpower extraction
# for both:
#   - CAR (already present in loaded preprocessed data)
#   - CSD (computed from epochs)
#
# Optional downsampling is applied AFTER reference choice and BEFORE PSD
# estimation, identically for CAR and CSD.
# -----------------------------------------------------------------------------

from pathlib import Path

import fooof
import mne
import numpy as np
import pandas as pd
import scipy.io
from joblib import Parallel, delayed
from mne.preprocessing import compute_current_source_density
from mne.time_frequency import psd_array_multitaper


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned_45/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
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
TIME_WINDOW = (-1.7, 0.0)
MIN_TRIALS_PER_SEQUENCE = 5

# PSD / FOOOF settings
DOWNSAMPLE_BEFORE_PSD = True
TARGET_SFREQ = 250

FMIN_FIT = 1.0
FMAX_FIT = 40.0
MT_BANDWIDTH = 2.0

FOOOF_KWARGS = dict(
    aperiodic_mode="fixed",
    peak_width_limits=(2, 12),
    max_n_peaks=8,
    min_peak_height=0.05,
    verbose=False,
)

BANDS = {
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta": (15, 30),
}

CSD_KWARGS = dict(
    stiffness=4,
    lambda2=1e-5,
    n_legendre_terms=50,
)

# Aggregation across trials within sequence
PSD_AGG_MODE = "mean"  # "mean" or "median"

# Reference schemes to extract
REFERENCE_SCHEMES = ("car", "csd")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def aggregate_psd(psd: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        return psd.mean(axis=0)
    if mode == "median":
        return np.median(psd, axis=0)
    raise ValueError(f"Unknown PSD_AGG_MODE: {mode}")


def get_epochs_for_reference(epochs: mne.Epochs, reference: str) -> mne.Epochs:
    """
    Return epochs for the requested reference scheme.

    Notes
    -----
    - 'car': assumes loaded preprocessed data already use common average reference.
             No rereferencing is performed here.
    - 'csd': computes current source density from the epochs.
    """
    if reference == "car":
        return epochs.copy()

    if reference == "csd":
        return compute_current_source_density(epochs.copy(), **CSD_KWARGS)

    raise ValueError(f"Unknown reference scheme: {reference}")


def prepare_epochs_for_psd(epochs_ref: mne.Epochs) -> mne.Epochs:
    """
    Optionally downsample after reference selection and before PSD estimation.
    """
    if DOWNSAMPLE_BEFORE_PSD:
        return epochs_ref.copy().resample(
            TARGET_SFREQ,
            npad="auto",
            verbose=False,
        )
    return epochs_ref


# -----------------------------------------------------------------------------
# Core extraction
# -----------------------------------------------------------------------------
def extract_sequence_measures(
    erp_data: np.ndarray,
    erp_times_sec: np.ndarray,
    df_trials: pd.DataFrame,
    subj_id: int,
    sfreq_psd: float,
    reference: str,
):
    tidx = np.where(
        (erp_times_sec >= TIME_WINDOW[0]) & (erp_times_sec < TIME_WINDOW[1])
    )[0]

    if tidx.size == 0:
        raise ValueError("No samples found in requested time window.")

    grouped = df_trials.groupby(["block_nr", "sequence_nr"], sort=True)

    rows = []
    seq_psd = []
    seq_index = []
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

        # trials x channels x time-window
        x = erp_data[idx][:, :, tidx]

        psd, freqs = psd_array_multitaper(
            x,
            sfreq=sfreq_psd,
            fmin=FMIN_FIT,
            fmax=FMAX_FIT,
            bandwidth=MT_BANDWIDTH,
            normalization="full",
            verbose=False,
            n_jobs=1,
        )

        psd_seq = aggregate_psd(psd, mode=PSD_AGG_MODE)
        freqs_out = freqs

        seq_psd.append(psd_seq)
        seq_index.append(
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
                "reference": reference,
            }
        )

        fg = fooof.FOOOFGroup(**FOOOF_KWARGS)
        fg.fit(freqs, psd_seq, [FMIN_FIT, FMAX_FIT])

        aperiodic = fg.get_params("aperiodic_params")
        r2 = fg.get_params("r_squared")
        err = fg.get_params("error")

        offsets = aperiodic[:, 0]
        exponents = aperiodic[:, 1]

        for ch_ix in range(psd_seq.shape[0]):
            log_psd = np.log10(psd_seq[ch_ix])
            aperiodic_fit = offsets[ch_ix] - exponents[ch_ix] * np.log10(freqs)
            flat = log_psd - aperiodic_fit

            band_vals = {}
            for band_name, (lo, hi) in BANDS.items():
                band_mask = (freqs >= lo) & (freqs <= hi)
                band_vals[f"{band_name}_flat"] = (
                    float(np.mean(flat[band_mask])) if np.any(band_mask) else np.nan
                )

            rows.append(
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
                    "reference": reference,
                    "ch_ix": int(ch_ix),
                    "ch_name": CHANNEL_LABELS[ch_ix],
                    "offset": float(offsets[ch_ix]),
                    "exponent": float(exponents[ch_ix]),
                    "r2": float(r2[ch_ix]),
                    "error": float(err[ch_ix]),
                    "theta_flat": band_vals["theta_flat"],
                    "alpha_flat": band_vals["alpha_flat"],
                    "beta_flat": band_vals["beta_flat"],
                    "fmin_fit": FMIN_FIT,
                    "fmax_fit": FMAX_FIT,
                    "mt_bandwidth": MT_BANDWIDTH,
                    "sfreq_psd": float(sfreq_psd),
                    "psd_agg_mode": PSD_AGG_MODE,
                }
            )

    df_seq = pd.DataFrame(rows)

    if df_seq.empty:
        return None, None, None, None

    return df_seq, seq_psd, seq_index, freqs_out


def run_reference_pipeline(
    epochs: mne.Epochs,
    df_trials: pd.DataFrame,
    subj_id: int,
    reference: str,
):
    """
    Run the full sequence-extraction pipeline for one reference scheme.
    """
    epochs_ref = get_epochs_for_reference(epochs, reference=reference)
    epochs_psd = prepare_epochs_for_psd(epochs_ref)

    erp_data_ref = epochs_psd.get_data(copy=True)
    erp_times_sec_ref = epochs_psd.times.copy()
    sfreq_psd = float(epochs_psd.info["sfreq"])

    return extract_sequence_measures(
        erp_data=erp_data_ref,
        erp_times_sec=erp_times_sec_ref,
        df_trials=df_trials,
        subj_id=subj_id,
        sfreq_psd=sfreq_psd,
        reference=reference,
    )


def save_reference_outputs(
    df_seq: pd.DataFrame,
    seq_psd: list,
    seq_index: list,
    freqs: np.ndarray,
    subj_id: int,
    reference: str,
):
    subj_tag = f"sub-{subj_id:03d}"

    df_seq.to_csv(
        PATH_OUT / f"{subj_tag}_seq_fooof_rt_channelwise_long_{reference}.csv",
        index=False,
    )

    if seq_psd and freqs is not None:
        np.savez_compressed(
            PATH_OUT / f"{subj_tag}_seq_psd_channelwise_{reference}.npz",
            psd=np.stack(seq_psd),
            freqs=freqs,
            channels=np.array(CHANNEL_LABELS),
        )
        pd.DataFrame(seq_index).to_csv(
            PATH_OUT / f"{subj_tag}_seq_psd_channelwise_index_{reference}.csv",
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

    keep_mask = df_trials["accuracy"] == 1
    df_trials = df_trials.loc[keep_mask].reset_index(drop=True)
    erp_data = erp_data[keep_mask, :, :]

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

    # IMPORTANT:
    # The loaded data are assumed to already be CAR-referenced.
    # Therefore:
    #   - reference='car' uses epochs as loaded
    #   - reference='csd' computes CSD from those epochs
    out = []

    for reference in REFERENCE_SCHEMES:
        df_seq, seq_psd, seq_index, freqs = run_reference_pipeline(
            epochs=epochs,
            df_trials=df_trials,
            subj_id=subj_id,
            reference=reference,
        )

        if df_seq is None or df_seq.empty:
            continue

        save_reference_outputs(
            df_seq=df_seq,
            seq_psd=seq_psd,
            seq_index=seq_index,
            freqs=freqs,
            subj_id=subj_id,
            reference=reference,
        )

        out.append(df_seq)

    if not out:
        return None

    print(f"Saved subject {subj_id:03d} ({', '.join(REFERENCE_SCHEMES)})")
    return pd.concat(out, ignore_index=True)


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
combined_file = PATH_OUT / "all_subjects_seq_fooof_rt_channelwise_long_car_csd.csv"
df_all.to_csv(combined_file, index=False)

print("Finished.")
print("Saved:", combined_file)
print("Rows:", len(df_all))
print("Subjects:", df_all["id"].nunique())
print("Electrodes:", df_all["ch_name"].nunique())
print("References:", sorted(df_all["reference"].unique()))
print(
    "Sequences:",
    df_all[["id", "reference", "block_nr", "sequence_nr"]].drop_duplicates().shape[0],
)