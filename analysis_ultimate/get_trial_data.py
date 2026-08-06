# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import scipy.io


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/3_trial_data/")

DATASETS = sorted(PATH_IN.glob("*erp.set"))

FILE_EEG_OUT = PATH_OUT / "trial_level_eeg.h5"
FILE_METADATA_OUT = PATH_OUT / "trial_level_metadata.csv"


# -----------------------------------------------------------------------------
# Exclusions
# -----------------------------------------------------------------------------
IDS_TO_DROP = {1, 2, 3, 4, 5, 6, 13, 17, 25, 40, 49, 83}


# -----------------------------------------------------------------------------
# EEG parameters
# -----------------------------------------------------------------------------
SFREQ = 200

TMIN_SAVE = -1.8
TMAX_SAVE = 1.0

# Save all datasets at 100 Hz.
# Set to 1 to retain the original 500-Hz resolution.
DECIM = 2
SFREQ_OUT = SFREQ / DECIM

N_JOBS = -1


# -----------------------------------------------------------------------------
# Time-frequency parameters
# -----------------------------------------------------------------------------
FREQUENCY_BANDS = {
    "theta": np.arange(4, 8),       # 4–7 Hz
    "alpha": np.arange(8, 14),      # 8–13 Hz
    "beta": np.arange(14, 31),      # 14–30 Hz
}

# Frequency-dependent wavelet lengths:
# 4 Hz -> 2 cycles
# 10 Hz -> 5 cycles
# 30 Hz -> 15 cycles
#
# This gives a constant nominal wavelet duration of 0.5 s.
N_CYCLES_DIVISOR = 2


# -----------------------------------------------------------------------------
# Channel information
# -----------------------------------------------------------------------------
CHANNEL_LABELS = (
    Path("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt")
    .read_text()
    .splitlines()
)

INFO_ERP = mne.create_info(
    CHANNEL_LABELS,
    sfreq=SFREQ,
    ch_types="eeg",
    verbose=None,
)

MONTAGE = mne.channels.make_standard_montage("standard_1020")

INFO_ERP.set_montage(
    MONTAGE,
    on_missing="warn",
    match_case=False,
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def compute_band_power(
    data: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """
    Compute single-trial Morlet power and average across frequencies.

    Parameters
    ----------
    data
        EEG data with shape:
        trials x channels x time

    frequencies
        Frequencies included in the band.

    Returns
    -------
    band_power
        Linear power with shape:
        trials x channels x time
    """
    n_cycles = frequencies / N_CYCLES_DIVISOR

    power = mne.time_frequency.tfr_array_morlet(
        data,
        sfreq=SFREQ,
        freqs=frequencies,
        n_cycles=n_cycles,
        output="power",
        use_fft=True,
        zero_mean=True,
        decim=DECIM,
        n_jobs=N_JOBS,
        verbose=False,
    )

    # trials x channels x frequencies x time
    # -> trials x channels x time
    return power.mean(axis=2).astype(np.float32)


def append_to_dataset(
    h5_dataset: h5py.Dataset,
    values: np.ndarray,
) -> None:
    """Append trials to an extendable HDF5 dataset."""
    start = h5_dataset.shape[0]
    stop = start + values.shape[0]

    h5_dataset.resize(stop, axis=0)
    h5_dataset[start:stop] = values


# -----------------------------------------------------------------------------
# Output setup
# -----------------------------------------------------------------------------
PATH_OUT.mkdir(parents=True, exist_ok=True)

all_metadata = []

# Close a handle left behind by an interrupted run.
try:
    h5_file.close()
except (NameError, ValueError):
    pass

# Remove an incomplete file from a previous failed run.
if FILE_EEG_OUT.exists():
    FILE_EEG_OUT.unlink()

h5_file = h5py.File(FILE_EEG_OUT, mode="w")

h5_datasets = None
saved_times = None

# -----------------------------------------------------------------------------
# Subject loop
# -----------------------------------------------------------------------------
for dataset in DATASETS:

    # -------------------------------------------------------------------------
    # Load trial metadata
    # -------------------------------------------------------------------------
    base = str(dataset).split("_cleaned")[0]

    df_trials = pd.read_csv(base + "_erp_trialinfo.csv")

    subj_id = int(df_trials["id"].iloc[0])

    if subj_id in IDS_TO_DROP:
        continue

    print(f"Processing subject {subj_id}")

    # -------------------------------------------------------------------------
    # Load EEG data
    # -------------------------------------------------------------------------
    mat = scipy.io.loadmat(dataset)

    # MATLAB:
    # channels x time x trials
    #
    # Python:
    # trials x channels x time
    erp_data = np.transpose(
        mat["data"],
        [2, 0, 1],
    )

    erp_times = mat["times"].ravel().astype(float)

    erp_times_sec = (
        erp_times / 1000
        if np.nanmax(np.abs(erp_times)) > 20
        else erp_times
    )

    # -------------------------------------------------------------------------
    # Prepare trial metadata
    # -------------------------------------------------------------------------
    df_trials["accuracy"] = (
        df_trials["accuracy"] == 1
    ).astype(int)

    df_trials = df_trials.rename(
        columns={
            "session_condition": "group",
            "trial_nr": "trial_nr_sequence",
            "trial_nr_total": "trial_nr_global",
            "last_feedback_scaled": "feedback",
        }
    )

    df_trials["group"] = df_trials["group"].replace(
        {
            1: "experimental",
            2: "control",
        }
    )

    df_trials["feedback2"] = df_trials["feedback"] ** 2

    df_trials["half"] = (
        df_trials["block_nr"] > 4
    ).astype(int)

    df_trials["log_rt"] = np.log(df_trials["rt"])

    df_trials["sequence_uid"] = (
        df_trials["id"].astype(str)
        + "_"
        + df_trials["block_nr"].astype(str)
        + "_"
        + df_trials["sequence_nr"].astype(str)
    )

    # -------------------------------------------------------------------------
    # Exclude first sequence of each block
    # -------------------------------------------------------------------------
    keep_mask = (
        df_trials["sequence_nr"] > 1
    ).to_numpy()

    df_trials = (
        df_trials.loc[keep_mask]
        .reset_index(drop=True)
    )

    erp_data = erp_data[keep_mask]

    # -------------------------------------------------------------------------
    # Compute TF power from the uncropped epochs
    # -------------------------------------------------------------------------
    band_data = {}

    for band_name, frequencies in FREQUENCY_BANDS.items():

        print(
            f"  Computing {band_name}: "
            f"{frequencies[0]}–{frequencies[-1]} Hz"
        )

        band_data[band_name] = compute_band_power(
            erp_data,
            frequencies,
        )

    # -------------------------------------------------------------------------
    # Downsample ERP voltage onto the same time grid
    # -------------------------------------------------------------------------
    erp_data = mne.filter.resample(
        erp_data.astype(np.float64, copy=False),
        down=DECIM,
        axis=-1,
    ).astype(np.float32)
    
    times_out = (
        erp_times_sec[0]
        + np.arange(erp_data.shape[-1]) / SFREQ_OUT
    )
        
    # -------------------------------------------------------------------------
    # Crop all four datasets to -1.5 to +1.0 s
    # -------------------------------------------------------------------------
    time_mask = (
        (times_out >= TMIN_SAVE)
        & (times_out <= TMAX_SAVE)
    )

    times_save = times_out[time_mask]

    erp_data = erp_data[:, :, time_mask]

    for band_name in band_data:
        band_data[band_name] = band_data[band_name][:, :, time_mask]

    # -------------------------------------------------------------------------
    # Create HDF5 datasets after the first included subject
    # -------------------------------------------------------------------------
    if h5_datasets is None:

        n_channels = erp_data.shape[1]
        n_times = erp_data.shape[2]

        # Chunk over trials so that downstream trial subsets can be read
        # efficiently without loading the complete array.
        chunks = (
            1,
            n_channels,
            n_times,
        )

        h5_datasets = {
            "erp": h5_file.create_dataset(
                "erp",
                shape=(0, n_channels, n_times),
                maxshape=(None, n_channels, n_times),
                dtype=np.float32,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
            ),
            "theta": h5_file.create_dataset(
                "theta",
                shape=(0, n_channels, n_times),
                maxshape=(None, n_channels, n_times),
                dtype=np.float32,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
            ),
            "alpha": h5_file.create_dataset(
                "alpha",
                shape=(0, n_channels, n_times),
                maxshape=(None, n_channels, n_times),
                dtype=np.float32,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
            ),
            "beta": h5_file.create_dataset(
                "beta",
                shape=(0, n_channels, n_times),
                maxshape=(None, n_channels, n_times),
                dtype=np.float32,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
            ),
        }

        saved_times = times_save.copy()

        h5_file.create_dataset(
            "times",
            data=saved_times.astype(np.float64),
        )

        string_dtype = h5py.string_dtype(
            encoding="utf-8"
        )

        h5_file.create_dataset(
            "channels",
            data=np.asarray(
                CHANNEL_LABELS,
                dtype=object,
            ),
            dtype=string_dtype,
        )

        h5_file.attrs["sfreq_original"] = SFREQ
        h5_file.attrs["sfreq_saved"] = SFREQ_OUT
        h5_file.attrs["tmin"] = TMIN_SAVE
        h5_file.attrs["tmax"] = TMAX_SAVE
        h5_file.attrs["tf_method"] = "morlet"
        h5_file.attrs["tf_output"] = "linear_power"
        h5_file.attrs["n_cycles"] = "frequency / 2"

    # -------------------------------------------------------------------------
    # Add global EEG row index to metadata
    # -------------------------------------------------------------------------
    eeg_row_start = h5_datasets["erp"].shape[0]

    df_trials["eeg_row"] = np.arange(
        eeg_row_start,
        eeg_row_start + len(df_trials),
    )

    # -------------------------------------------------------------------------
    # Append subject data
    # -------------------------------------------------------------------------
    append_to_dataset(
        h5_datasets["erp"],
        erp_data,
    )

    append_to_dataset(
        h5_datasets["theta"],
        band_data["theta"],
    )

    append_to_dataset(
        h5_datasets["alpha"],
        band_data["alpha"],
    )

    append_to_dataset(
        h5_datasets["beta"],
        band_data["beta"],
    )

    all_metadata.append(df_trials)


# -----------------------------------------------------------------------------
# Save metadata
# -----------------------------------------------------------------------------
h5_file.close()

df_all = pd.concat(
    all_metadata,
    ignore_index=True,
)

df_all.to_csv(
    FILE_METADATA_OUT,
    index=False,
)

print()
print(f"Saved metadata: {FILE_METADATA_OUT}")
print(f"Saved EEG data: {FILE_EEG_OUT}")
print(f"Number of trials: {len(df_all)}")