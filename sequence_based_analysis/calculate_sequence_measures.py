# -----------------------------------------------------------------------------
# Sequence-based FOOOF + RT + flattened oscillatory bandpower extraction
# -----------------------------------------------------------------------------

import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io
import fooof
from joblib import Parallel, delayed
from mne.time_frequency import psd_array_multitaper


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_out = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
path_out.mkdir(parents=True, exist_ok=True)

path_in = "/mnt/data_dump/pixelstress/2_autocleaned_45/"
datasets = glob.glob(f"{path_in}/*erp.set")


# -----------------------------------------------------------------------------
# Exclusion list
# -----------------------------------------------------------------------------
ids_to_drop = {1,2,3,4,5,6,13,17,18,25,40,49,83}


# -----------------------------------------------------------------------------
# Channel labels
# -----------------------------------------------------------------------------
channel_labels = (
    open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r")
    .read()
    .split("\n")[:-1]
)

info_erp = mne.create_info(channel_labels, sfreq=1000, ch_types="eeg", verbose=None)
montage = mne.channels.make_standard_montage("standard_1020")
info_erp.set_montage(montage, on_missing="warn", match_case=False)


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------
window_name = "pre_target"
window = (-1.7, 0)

fmin_fit, fmax_fit = 1.0, 30.0
mt_bandwidth = 3.0
min_trials_per_sequence = 5

fooof_kwargs = dict(
    aperiodic_mode="fixed",
    peak_width_limits=(2,12),
    max_n_peaks=8,
    min_peak_height=0.05,
    verbose=False,
)

# frequency bands for flattened power
bands = {
    "theta": (4,7),
    "alpha": (8,13),
    "beta":  (15,30),
}


# -----------------------------------------------------------------------------
# Subject processing
# -----------------------------------------------------------------------------
def process_subject(dataset):

    base = dataset.split("_cleaned")[0]

    df_erp = pd.read_csv(base + "_erp_trialinfo.csv")
    subj_id = int(df_erp["id"].iloc[0])

    if subj_id in ids_to_drop:
        return None


    # -------------------------------------------------------------------------
    # Load ERP data
    # -------------------------------------------------------------------------
    mat = scipy.io.loadmat(dataset)

    erp_data = np.transpose(mat["data"], [2,0,1])
    erp_times = mat["times"].ravel().astype(float)

    erp_times_sec = erp_times/1000 if np.nanmax(np.abs(erp_times)) > 20 else erp_times

    if erp_data.shape[1] != len(channel_labels):
        raise ValueError(
            f"Channel mismatch subject {subj_id}: "
            f"{erp_data.shape[1]} vs {len(channel_labels)}"
        )


    # -------------------------------------------------------------------------
    # Coding
    # -------------------------------------------------------------------------
    df_erp["accuracy"] = (df_erp["accuracy"] == 1).astype(int)

    df_erp = df_erp.rename(columns={"session_condition":"group"})
    df_erp["group"] = df_erp["group"].replace({1:"experimental",2:"control"})


    # -------------------------------------------------------------------------
    # Remove first sequences
    # -------------------------------------------------------------------------
    mask = df_erp["sequence_nr"] > 1
    df_erp = df_erp.loc[mask].reset_index(drop=True)
    erp_data = erp_data[mask,:,:]


    # -------------------------------------------------------------------------
    # Keep only correct trials
    # -------------------------------------------------------------------------
    mask = df_erp["accuracy"] == 1
    df_erp = df_erp.loc[mask].reset_index(drop=True)
    erp_data = erp_data[mask,:,:]


    if len(df_erp) != erp_data.shape[0]:
        raise ValueError(
            f"Mismatch subject {subj_id}: "
            f"{len(df_erp)} rows vs {erp_data.shape[0]} trials"
        )


    # -------------------------------------------------------------------------
    # Window indices
    # -------------------------------------------------------------------------
    tidx = np.where(
        (erp_times_sec >= window[0]) &
        (erp_times_sec <  window[1])
    )[0]

    if tidx.size == 0:
        raise ValueError(f"No samples for window {window_name}")


    # -------------------------------------------------------------------------
    # Sequence grouping
    # -------------------------------------------------------------------------
    g = df_erp.groupby(["block_nr","sequence_nr"], sort=True)

    seq_rows = []
    seq_psd = []
    seq_index = []


    # -------------------------------------------------------------------------
    # Sequence loop
    # -------------------------------------------------------------------------
    for (block_nr, seq_nr), idx in g.indices.items():

        idx = np.asarray(idx)
        n_trials = len(idx)

        if n_trials < min_trials_per_sequence:
            continue


        df_sub = df_erp.loc[idx]

        group = df_sub["group"].iloc[0]
        half = "first" if int(block_nr) <= 4 else "second"

        f = float(df_sub["last_feedback_scaled"].iloc[0])
        f2 = f**2

        mean_difficulty = float(df_sub["trial_difficulty"].mean())

        mean_rt = float(df_sub["rt"].mean())
        mean_log_rt = float(np.log(mean_rt)) if mean_rt > 0 else np.nan


        # -----------------------------------------------------------------
        # PSD
        # -----------------------------------------------------------------
        x = erp_data[idx][:,:,tidx]

        psd, freqs = psd_array_multitaper(
            x,
            sfreq=info_erp["sfreq"],
            fmin=fmin_fit,
            fmax=fmax_fit,
            bandwidth=mt_bandwidth,
            normalization="full",
            verbose=False,
            n_jobs=1
        )

        psd_seq = psd.mean(axis=0)

        seq_psd.append(psd_seq)

        seq_index.append(
            dict(
                id=subj_id,
                group=group,
                block_nr=int(block_nr),
                sequence_nr=int(seq_nr),
                window=window_name,
                half=half,
                n_trials=n_trials,
                mean_trial_difficulty=mean_difficulty,
                f=f,
                f2=f2,
                mean_rt=mean_rt,
                mean_log_rt=mean_log_rt
            )
        )


        # -----------------------------------------------------------------
        # FOOOF
        # -----------------------------------------------------------------
        fg = fooof.FOOOFGroup(**fooof_kwargs)
        fg.fit(freqs, psd_seq, [fmin_fit,fmax_fit])
        
        aperiodic = fg.get_params("aperiodic_params")
        r2 = fg.get_params("r_squared")
        err = fg.get_params("error")
        
        offsets = aperiodic[:, 0]
        exponents = aperiodic[:, 1]
        
        for ci in range(psd_seq.shape[0]):
        
            # manual flattened spectrum from PSD and fitted aperiodic line
            log_psd = np.log10(psd_seq[ci])
            aperiodic_fit = offsets[ci] - exponents[ci] * np.log10(freqs)
            flat = log_psd - aperiodic_fit
        
            band_vals = {}
            for b, (lo, hi) in bands.items():
                idx_band = (freqs >= lo) & (freqs <= hi)
        
                if not np.any(idx_band):
                    band_vals[f"{b}_flat"] = np.nan
                else:
                    band_vals[f"{b}_flat"] = float(np.mean(flat[idx_band]))
        
            seq_rows.append(
                {
                    "id": subj_id,
                    "group": group,
                    "block_nr": int(block_nr),
                    "sequence_nr": int(seq_nr),
                    "window": window_name,
                    "half": half,
                    "n_trials": int(n_trials),
                    "mean_trial_difficulty": mean_difficulty,
                    "f": f,
                    "f2": f2,
                    "mean_rt": mean_rt,
                    "mean_log_rt": mean_log_rt,
                    "ch_ix": int(ci),
                    "ch_name": channel_labels[ci],
                    "offset": float(offsets[ci]),
                    "exponent": float(exponents[ci]),
                    "r2": float(r2[ci]),
                    "error": float(err[ci]),
                    "theta_flat": band_vals["theta_flat"],
                    "alpha_flat": band_vals["alpha_flat"],
                    "beta_flat": band_vals["beta_flat"],
                    "fmin_fit": fmin_fit,
                    "fmax_fit": fmax_fit,
                    "mt_bandwidth": mt_bandwidth,
                }
            )


    df_seq = pd.DataFrame(seq_rows)

    if len(df_seq) == 0:
        return None


    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------
    df_seq = df_seq[
        (df_seq["r2"] >= 0.8) &
        (df_seq["error"] <= 0.2)
    ]


    # -------------------------------------------------------------------------
    # Save subject outputs
    # -------------------------------------------------------------------------
    out_csv = path_out / f"sub-{subj_id:03d}_seq_fooof_rt_channelwise.csv"
    df_seq.to_csv(out_csv, index=False)


    if len(seq_psd) > 0:

        psd_arr = np.stack(seq_psd)

        np.savez_compressed(
            path_out / f"sub-{subj_id:03d}_seq_psd_channelwise.npz",
            psd=psd_arr,
            freqs=freqs,
            channels=np.array(channel_labels)
        )

        pd.DataFrame(seq_index).to_csv(
            path_out / f"sub-{subj_id:03d}_seq_psd_channelwise_index.csv",
            index=False
        )

    print(f"Saved subject {subj_id:03d}")

    return df_seq


# -----------------------------------------------------------------------------
# Run subjects in parallel
# -----------------------------------------------------------------------------
seq_data = Parallel(
    n_jobs=-1,
    backend="loky",
    verbose=10
)(
    delayed(process_subject)(dataset) for dataset in datasets
)


seq_data = [d for d in seq_data if d is not None and len(d)>0]

if len(seq_data) == 0:
    raise RuntimeError("No subject data returned")


df_all = pd.concat(seq_data, ignore_index=True)


# -----------------------------------------------------------------------------
# Save combined dataframe
# -----------------------------------------------------------------------------
df_all.to_csv(
    path_out / "all_subjects_seq_fooof_rt_channelwise_long.csv",
    index=False
)


print("Finished.")
print("Rows:",len(df_all))
print("Subjects:",df_all["id"].nunique())
print("Electrodes:",df_all["ch_name"].nunique())
print("Sequences:",df_all[["id","block_nr","sequence_nr"]].drop_duplicates().shape[0])