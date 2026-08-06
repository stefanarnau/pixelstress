"""
Looped ROI analysis for ERP, alpha, and theta.

Starting point
--------------
Refactored from the current single-measure theta script.

What this script does
---------------------
1. Loads metadata, channel names, and time vector once.
2. Loops over measure-specific configurations.
3. For each measure:
   - selects trials;
   - extracts the configured ROI and analysis window;
   - applies optional participant-level dB baseline correction;
   - adds the trial-level ROI average to one shared metadata DataFrame;
   - fits the EEG mixed model;
   - fits the single-measure brain-behavior model;
   - creates the same composite figure as the original script.
4. Saves one enriched trial-level CSV containing all ROI columns.
5. Optionally fits a combined RT model containing ERP, alpha, and theta.

The enriched CSV preserves every metadata row. Trials not used by a particular
measure receive NaN in that measure's ROI column.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.ndimage import gaussian_filter1d
from scipy.stats import chi2


# =============================================================================
# PATHS
# =============================================================================

PATH_IN = Path("/mnt/data_dump/pixelstress/3_trial_data/")
PATH_OUT = PATH_IN / "roi_mlm_results"
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_EEG = PATH_IN / "trial_level_eeg.h5"
FILE_METADATA = PATH_IN / "trial_level_metadata.csv"
FILE_ENRICHED = PATH_IN / "trial_level_metadata_with_roi_values.csv"


# =============================================================================
# MEASURE CONFIGURATIONS
# =============================================================================

MEASURE_CONFIGS = {
    "erp": {
        "dataset": "erp",
        "roi_column": "erp_roi",
        "channels": ["Cz", "FCz", "C1", "C2", "CPz"],
        "analysis_window": (-0.100, 0.000),
        "timecourse_window": (-1.300, 0.800),
        "topo_times": [-1.0, -0.8, -0.6, -0.4, -0.2],
        "correct_only": True,
        "apply_power_baseline": False,
        "baseline_window": None,
    },
    "alpha": {
        "dataset": "alpha",
        "roi_column": "alpha_roi_db",
        "channels": ["POz", "PO7", "PO8", "Oz", "O1", "O2"],
        "analysis_window": (-1.000, 0.000),
        "timecourse_window": (-1.300, 0.800),
        "topo_times": [-1.0, -0.8, -0.6, -0.4, -0.2],
        "correct_only": True,
        "apply_power_baseline": True,
        "baseline_window": (-1.800, -1.400),
    },
    "theta": {
        "dataset": "theta",
        "roi_column": "theta_roi_db",
        "channels": ["FCz", "FC1", "FC2", "Cz", "C1", "C2"],
        "analysis_window": (0.150, 0.350),
        "timecourse_window": (-0.200, 0.600),
        "topo_times": [0.0, 0.1, 0.2, 0.3, 0.4],
        "correct_only": True,
        "apply_power_baseline": True,
        "baseline_window": (-1.800, -1.400),
    },
}

COVARIATES_TO_CENTER = [
    "sequence_difficulty",
    # "half",
]

INCLUDE_SEQUENCE_RANDOM_INTERCEPT = False
OPTIMIZERS = ["bfgs", "lbfgs", "powell"]
MAXITER = 2000

RUN_SINGLE_MEASURE_BRAIN_BEHAVIOR = True
RUN_COMBINED_BRAIN_BEHAVIOR = True
BEHAVIOR_OUTCOME = "rt"
BEHAVIOR_REML = True

# Figure settings
N_BINS = 5
BIN_LABELS = None
TIMECOURSE_SMOOTHING_MS = 20
SHOW_TIMECOURSE_CI = False
TIMECOURSE_CI_LEVEL = 0.95

TOPO_HALF_WINDOW = 0.025
TOPO_MONTAGE = "standard_1020"
TOPO_CONTOURS = 6
TOPO_VLIM = None

PREDICTION_POINTS = 200
INTERACTION_CI_LEVEL = 0.95
FIGURE_DPI = 600

GROUP_ORDER = ["control", "experimental"]
GROUP_TITLES = {
    "control": "Control",
    "experimental": "Experimental",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normal_critical_value(ci_level: float) -> float:
    lookup = {
        0.90: 1.644854,
        0.95: 1.959964,
        0.99: 2.575829,
    }
    if ci_level not in lookup:
        raise ValueError("CI level must be 0.90, 0.95, or 0.99.")
    return lookup[ci_level]


def validate_nonempty_indices(indices: np.ndarray, label: str) -> None:
    if len(indices) == 0:
        raise ValueError(f"No samples found for {label}.")


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"Cannot standardize {series.name}: SD={sd}.")
    return (series - series.mean()) / sd


def fit_mixed_model(
    formula: str,
    data: pd.DataFrame,
    re_formula: str,
    reml: bool,
    vc_formula=None,
):
    """Fit a MixedLM while trying a short optimizer sequence."""
    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data["id"],
        re_formula=re_formula,
        vc_formula=vc_formula,
    )

    last_result = None
    last_exception = None

    for optimizer in OPTIMIZERS:
        try:
            result = model.fit(
                reml=reml,
                method=optimizer,
                maxiter=MAXITER,
                disp=False,
            )
            last_result = result
            if result.converged:
                return result, optimizer
        except Exception as exc:
            last_exception = exc

    if last_result is not None:
        warnings.warn(
            f"Returning a non-converged model for formula:\n{formula}",
            RuntimeWarning,
        )
        return last_result, OPTIMIZERS[-1]

    raise RuntimeError(
        f"Model failed for formula:\n{formula}\n"
        f"Last exception: {last_exception}"
    )


def read_eeg_block(
    dataset,
    selected_rows: np.ndarray,
    channel_idx: np.ndarray,
    time_idx: np.ndarray,
) -> np.ndarray:
    """Read trial × channel × time data while satisfying h5py index rules."""
    selected_rows = np.asarray(selected_rows, dtype=int)
    channel_idx = np.asarray(channel_idx, dtype=int)
    time_idx = np.asarray(time_idx, dtype=int)

    start = int(time_idx.min())
    stop = int(time_idx.max()) + 1
    local_idx = time_idx - start

    read_order = np.argsort(channel_idx)
    sorted_channel_idx = channel_idx[read_order]

    if np.any(np.diff(sorted_channel_idx) <= 0):
        raise ValueError("Channel indices must be unique.")

    restore_order = np.argsort(read_order)

    block = np.asarray(dataset[:, sorted_channel_idx, start:stop])
    block = block[selected_rows]
    block = block[:, restore_order, :]

    return block[:, :, local_idx]


def participant_baseline_references(
    baseline_data: np.ndarray,
    participant_ids: np.ndarray,
) -> dict:
    """Return participant-specific channel baselines in dB."""
    log_baseline = 10 * np.log10(baseline_data.astype(np.float64))
    references = {}

    for participant in np.unique(participant_ids):
        participant_mask = participant_ids == participant
        references[participant] = log_baseline[participant_mask].mean(
            axis=(0, 2)
        )

    return references


def transform_power_by_participant(
    data: np.ndarray,
    baseline_reference: dict | None,
    participant_ids: np.ndarray,
    apply_baseline: bool,
) -> np.ndarray:
    transformed = 10 * np.log10(data.astype(np.float64))

    if not apply_baseline:
        return transformed

    if baseline_reference is None:
        raise ValueError("Baseline requested but no baseline reference supplied.")

    for participant in np.unique(participant_ids):
        participant_mask = participant_ids == participant
        transformed[participant_mask] -= (
            baseline_reference[participant][None, :, None]
        )

    return transformed


def make_feedback_bins(feedback: pd.Series):
    feedback = np.asarray(feedback, dtype=float)
    edges = np.linspace(
        np.nanmin(feedback),
        np.nanmax(feedback),
        N_BINS + 1,
    )
    edges[0] -= np.finfo(float).eps
    edges[-1] += np.finfo(float).eps

    if BIN_LABELS is None:
        labels = [
            f"{edges[index]:.2f} to {edges[index + 1]:.2f}"
            for index in range(N_BINS)
        ]
    else:
        if len(BIN_LABELS) != N_BINS:
            raise ValueError("BIN_LABELS must contain N_BINS labels.")
        labels = BIN_LABELS

    bins = pd.cut(
        feedback,
        bins=edges,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    return bins, edges, labels


def fixed_effect_prediction(
    result,
    feedback_values: np.ndarray,
    experimental: int,
    feedback_mean: float,
    centered_covariates: list[str],
):
    fixed_names = list(result.fe_params.index)
    beta = result.fe_params.loc[fixed_names].to_numpy()
    covariance = result.cov_params().loc[
        fixed_names, fixed_names
    ].to_numpy()

    rows = []
    feedback_centered = feedback_values - feedback_mean

    for feedback_c in feedback_centered:
        values = {
            "Intercept": 1.0,
            "experimental": float(experimental),
            "feedback_c": feedback_c,
            "feedback2_c": feedback_c ** 2,
            "experimental:feedback_c": float(experimental) * feedback_c,
            "experimental:feedback2_c": (
                float(experimental) * feedback_c ** 2
            ),
        }
        values.update({name: 0.0 for name in centered_covariates})
        rows.append([values.get(name, 0.0) for name in fixed_names])

    design = np.asarray(rows)
    prediction = design @ beta
    standard_error = np.sqrt(
        np.einsum("ij,jk,ik->i", design, covariance, design)
    )
    critical = normal_critical_value(INTERACTION_CI_LEVEL)

    return (
        prediction,
        prediction - critical * standard_error,
        prediction + critical * standard_error,
    )


def save_text_result(result, filepath: Path, optimizer: str) -> None:
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(f"Optimizer: {optimizer}\n")
        file.write(f"Converged: {result.converged}\n\n")
        file.write(str(result.summary()))


def fixed_effect_table(result) -> pd.DataFrame:
    names = result.fe_params.index
    ci = result.conf_int().loc[names]
    return pd.DataFrame({
        "beta": result.fe_params,
        "se": result.bse_fe,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p": result.pvalues.loc[names],
    })


# =============================================================================
# DATA PREPARATION
# =============================================================================

metadata = pd.read_csv(FILE_METADATA)
enriched_df = metadata.copy()

with h5py.File(FILE_EEG, "r") as h5:
    times = h5["times"][:]
    channels = np.asarray(h5["channels"].asstr()[:])

if len(metadata) == 0:
    raise ValueError("Metadata table is empty.")

with h5py.File(FILE_EEG, "r") as h5:
    for config_name, config in MEASURE_CONFIGS.items():
        if config["dataset"] not in h5:
            raise KeyError(
                f"HDF5 dataset '{config['dataset']}' not found for {config_name}."
            )
        if h5[config["dataset"]].shape[0] != len(metadata):
            raise ValueError(
                f"{config_name}: HDF5 trials "
                f"({h5[config['dataset']].shape[0]}) do not match metadata rows "
                f"({len(metadata)})."
            )

print("Loaded shared data")
print("------------------")
print(f"Metadata rows: {len(metadata)}")
print(f"Participants: {metadata['id'].nunique()}")
print(f"Measures:     {list(MEASURE_CONFIGS)}")


# =============================================================================
# SINGLE-MEASURE LOOP
# =============================================================================

individual_results = {}
individual_brain_behavior_results = {}

for measure_name, config in MEASURE_CONFIGS.items():
    print("\n" + "=" * 78)
    print(f"ANALYZING {measure_name.upper()}")
    print("=" * 78)

    dataset_name = config["dataset"]
    roi_column = config["roi_column"]
    roi_channels = config["channels"]
    time_tmin, time_tmax = config["analysis_window"]
    tc_tmin, tc_tmax = config["timecourse_window"]
    topo_times = config["topo_times"]
    correct_only = config["correct_only"]
    apply_baseline = config["apply_power_baseline"]
    baseline_window = config["baseline_window"]

    missing_channels = sorted(set(roi_channels) - set(channels))
    if missing_channels:
        raise ValueError(
            f"{measure_name}: ROI channels not found: {missing_channels}"
        )

    channel_indices = np.array([
        np.where(channels == channel)[0][0]
        for channel in roi_channels
    ])
    all_channel_indices = np.arange(len(channels))

    time_indices = np.where(
        (times >= time_tmin) & (times <= time_tmax)
    )[0]
    timecourse_indices = np.where(
        (times >= tc_tmin) & (times <= tc_tmax)
    )[0]

    validate_nonempty_indices(
        time_indices,
        f"{measure_name} analysis window",
    )
    validate_nonempty_indices(
        timecourse_indices,
        f"{measure_name} time-course window",
    )

    if measure_name != "erp" and apply_baseline:
        if baseline_window is None:
            raise ValueError(
                f"{measure_name}: baseline enabled but no window provided."
            )
        baseline_tmin, baseline_tmax = baseline_window
        baseline_indices = np.where(
            (times >= baseline_tmin) & (times < baseline_tmax)
        )[0]
        validate_nonempty_indices(
            baseline_indices,
            f"{measure_name} baseline window",
        )
    else:
        baseline_indices = None

    topo_indices = []
    for topo_time in topo_times:
        indices = np.where(
            (times >= topo_time - TOPO_HALF_WINDOW)
            & (times <= topo_time + TOPO_HALF_WINDOW)
        )[0]
        validate_nonempty_indices(
            indices,
            f"{measure_name} topography at {topo_time:.3f} s",
        )
        topo_indices.append(indices)

    # ---------------------------------------------------------
    # Trial selection
    # ---------------------------------------------------------
    selection_mask = np.ones(len(metadata), dtype=bool)

    if correct_only:
        selection_mask &= metadata["accuracy"].to_numpy() == 1

    selected_indices = np.where(selection_mask)[0]
    model_df = metadata.loc[selection_mask].copy()
    model_df["_metadata_row"] = selected_indices
    model_df = model_df.reset_index(drop=True)

    participant_ids = model_df["id"].to_numpy()

    # ---------------------------------------------------------
    # EEG extraction
    # ---------------------------------------------------------
    with h5py.File(FILE_EEG, "r") as h5:
        dataset = h5[dataset_name]

        roi_timecourse_data = read_eeg_block(
            dataset,
            selected_indices,
            channel_indices,
            timecourse_indices,
        )
        roi_analysis_data = read_eeg_block(
            dataset,
            selected_indices,
            channel_indices,
            time_indices,
        )
        topo_data_blocks = [
            read_eeg_block(
                dataset,
                selected_indices,
                all_channel_indices,
                indices,
            )
            for indices in topo_indices
        ]

        if measure_name == "erp":
            roi_timecourses = (
                roi_timecourse_data.astype(np.float64).mean(axis=1)
            )
            eeg_value = (
                roi_analysis_data.astype(np.float64).mean(axis=(1, 2))
            )
            topo_values = np.stack([
                block.astype(np.float64).mean(axis=2)
                for block in topo_data_blocks
            ], axis=1)

        else:
            if apply_baseline:
                roi_baseline_data = read_eeg_block(
                    dataset,
                    selected_indices,
                    channel_indices,
                    baseline_indices,
                )
                roi_baselines = participant_baseline_references(
                    roi_baseline_data,
                    participant_ids,
                )
            else:
                roi_baselines = None

            roi_timecourse_transformed = transform_power_by_participant(
                roi_timecourse_data,
                roi_baselines,
                participant_ids,
                apply_baseline,
            )
            roi_timecourses = roi_timecourse_transformed.mean(axis=1)

            analysis_transformed = transform_power_by_participant(
                roi_analysis_data,
                roi_baselines,
                participant_ids,
                apply_baseline,
            )
            eeg_value = analysis_transformed.mean(axis=(1, 2))

            if apply_baseline:
                topo_baseline_data = read_eeg_block(
                    dataset,
                    selected_indices,
                    all_channel_indices,
                    baseline_indices,
                )
                topo_baselines = participant_baseline_references(
                    topo_baseline_data,
                    participant_ids,
                )
            else:
                topo_baselines = None

            topo_values_list = []
            for topo_power in topo_data_blocks:
                topo_transformed = transform_power_by_participant(
                    topo_power,
                    topo_baselines,
                    participant_ids,
                    apply_baseline,
                )
                topo_values_list.append(topo_transformed.mean(axis=2))

            topo_values = np.stack(topo_values_list, axis=1)

    model_df["eeg_value"] = eeg_value
    model_df["_array_row"] = np.arange(len(model_df))

    # Add the ROI scalar to the shared metadata table using original row indices.
    enriched_df[roi_column] = np.nan
    enriched_df.loc[
        model_df["_metadata_row"].to_numpy(),
        roi_column,
    ] = model_df["eeg_value"].to_numpy()

    # Save after each iteration as a checkpoint.
    enriched_df.to_csv(FILE_ENRICHED, index=False)

    # ---------------------------------------------------------
    # Predictors
    # ---------------------------------------------------------
    model_df["experimental"] = (
        model_df["group"] == "experimental"
    ).astype(int)

    model_df["feedback"] = pd.to_numeric(
        model_df["feedback"],
        errors="coerce",
    )
    feedback_mean = model_df["feedback"].mean()
    model_df["feedback_c"] = model_df["feedback"] - feedback_mean
    model_df["feedback2_c"] = model_df["feedback_c"] ** 2

    model_df["sequence_uid"] = (
        model_df["id"].astype(str)
        + "_b"
        + model_df["block_nr"].astype(str)
        + "_s"
        + model_df["sequence_nr"].astype(str)
    )

    centered_covariates = []
    for covariate in COVARIATES_TO_CENTER:
        centered_name = f"{covariate}_c"
        numeric_covariate = pd.to_numeric(
            model_df[covariate],
            errors="coerce",
        )
        model_df[centered_name] = (
            numeric_covariate - numeric_covariate.mean()
        )
        centered_covariates.append(centered_name)

    model_columns = [
        "eeg_value",
        "id",
        "group",
        "experimental",
        "feedback",
        "feedback_c",
        "feedback2_c",
        "sequence_uid",
    ] + centered_covariates

    valid_mask = (
        model_df[model_columns].notna().all(axis=1).to_numpy()
    )
    valid_array_rows = model_df.loc[
        valid_mask, "_array_row"
    ].to_numpy()

    model_df = (
        model_df.loc[valid_mask].copy().reset_index(drop=True)
    )
    roi_timecourses = roi_timecourses[valid_array_rows]
    topo_values = topo_values[valid_array_rows]
    model_df["_array_row"] = np.arange(len(model_df))

    (
        model_df["feedback_bin"],
        feedback_bin_edges,
        feedback_bin_labels,
    ) = make_feedback_bins(model_df["feedback"])

    fixed_terms = [
        "experimental",
        "feedback_c",
        "feedback2_c",
        "experimental:feedback_c",
        "experimental:feedback2_c",
    ] + centered_covariates

    formula = "eeg_value ~ " + " + ".join(fixed_terms)

    vc_formula = None
    if INCLUDE_SEQUENCE_RANDOM_INTERCEPT:
        vc_formula = {"sequence": "0 + C(sequence_uid)"}

    result, optimizer = fit_mixed_model(
        formula=formula,
        data=model_df,
        re_formula="1 + feedback_c",
        reml=False,
        vc_formula=vc_formula,
    )

    individual_results[measure_name] = result

    # ---------------------------------------------------------
    # Output names and model output
    # ---------------------------------------------------------
    channel_label = "-".join(roi_channels)
    trial_label = "correct" if correct_only else "all"

    if measure_name == "erp":
        baseline_label = ""
    elif apply_baseline:
        baseline_label = "_grandbaseline_db"
    else:
        baseline_label = "_absolute_log"

    analysis_stem = (
        f"{measure_name}"
        f"_{channel_label}"
        f"_{time_tmin:+.3f}_{time_tmax:+.3f}"
        f"_{trial_label}"
        f"{baseline_label}"
        "_participant_feedback_slope_RE"
    ).replace(".", "p")

    print("\nAnalysis definition")
    print("-------------------")
    print(f"Measure:           {measure_name}")
    print(f"ROI column:        {roi_column}")
    print(f"Channels:          {roi_channels}")
    print(f"Time window:       {time_tmin:.3f} to {time_tmax:.3f} s")
    print(f"Correct only:      {correct_only}")
    if measure_name != "erp":
        print(f"Power baseline:    {apply_baseline}")
        if apply_baseline:
            print(
                f"Baseline window:   "
                f"{baseline_window[0]:.3f} to {baseline_window[1]:.3f} s"
            )
    print(f"Trials:            {len(model_df)}")
    print(f"Participants:      {model_df['id'].nunique()}")
    print(f"Sequences:         {model_df['sequence_uid'].nunique()}")
    print(f"Formula:           {formula}")
    print(f"Optimizer:         {optimizer}")
    print(f"Converged:         {result.converged}")
    print()
    print(result.summary())

    save_text_result(
        result,
        PATH_OUT / f"{analysis_stem}_eeg_model.txt",
        optimizer,
    )
    fixed_effect_table(result).to_csv(
        PATH_OUT / f"{analysis_stem}_eeg_fixed_effects.csv"
    )

    # ---------------------------------------------------------
    # Single-measure brain-behavior model
    # ---------------------------------------------------------
    brain_behavior_result = None

    if RUN_SINGLE_MEASURE_BRAIN_BEHAVIOR:
        behavior_df = model_df.copy()
        behavior_df[BEHAVIOR_OUTCOME] = pd.to_numeric(
            behavior_df[BEHAVIOR_OUTCOME],
            errors="coerce",
        )

        behavior_df["eeg_between"] = (
            behavior_df.groupby("id")["eeg_value"].transform("mean")
        )
        behavior_df["eeg_within"] = (
            behavior_df["eeg_value"] - behavior_df["eeg_between"]
        )

        behavior_columns = [
            BEHAVIOR_OUTCOME,
            "id",
            "experimental",
            "feedback_c",
            "feedback2_c",
            "eeg_within",
            "eeg_between",
        ] + centered_covariates

        behavior_df = behavior_df.dropna(
            subset=behavior_columns
        ).reset_index(drop=True)

        behavior_fixed_terms = fixed_terms + [
            "eeg_within",
            "eeg_between",
        ]
        behavior_formula = (
            f"{BEHAVIOR_OUTCOME} ~ "
            + " + ".join(behavior_fixed_terms)
        )

        brain_behavior_result, behavior_optimizer = fit_mixed_model(
            formula=behavior_formula,
            data=behavior_df,
            re_formula="1 + feedback_c",
            reml=BEHAVIOR_REML,
        )
        individual_brain_behavior_results[
            measure_name
        ] = brain_behavior_result

        print("\nBrain-behavior model")
        print("--------------------")
        print(f"Outcome:            {BEHAVIOR_OUTCOME}")
        print(f"Trials:             {len(behavior_df)}")
        print(f"Participants:       {behavior_df['id'].nunique()}")
        print(f"Formula:            {behavior_formula}")
        print("EEG decomposition:  within + between participant")
        print(f"Optimizer:          {behavior_optimizer}")
        print(f"Converged:          {brain_behavior_result.converged}")
        print()
        print(brain_behavior_result.summary())

        save_text_result(
            brain_behavior_result,
            PATH_OUT / f"{analysis_stem}_brain_behavior_model.txt",
            behavior_optimizer,
        )
        fixed_effect_table(brain_behavior_result).to_csv(
            PATH_OUT
            / f"{analysis_stem}_brain_behavior_fixed_effects.csv"
        )

    # ---------------------------------------------------------
    # Composite figure
    # ---------------------------------------------------------
    cmap = plt.colormaps.get_cmap("plasma").resampled(N_BINS)
    bin_colors = [cmap(index) for index in range(N_BINS)]
    timecourse_times = times[timecourse_indices]

    if TIMECOURSE_SMOOTHING_MS > 0:
        sampling_interval_ms = (
            np.median(np.diff(timecourse_times)) * 1000
        )
        smoothing_sigma = (
            TIMECOURSE_SMOOTHING_MS / sampling_interval_ms
        )
    else:
        smoothing_sigma = 0

    timecourse_records = []
    for group in GROUP_ORDER:
        for bin_index, bin_label in enumerate(
            feedback_bin_labels
        ):
            subset = model_df[
                (model_df["group"] == group)
                & (model_df["feedback_bin"] == bin_label)
            ]

            for participant, participant_df in subset.groupby("id"):
                rows = participant_df["_array_row"].to_numpy()
                participant_curve = roi_timecourses[rows].mean(axis=0)

                if smoothing_sigma > 0:
                    participant_curve = gaussian_filter1d(
                        participant_curve,
                        sigma=smoothing_sigma,
                    )

                timecourse_records.append({
                    "group": group,
                    "feedback_bin": bin_label,
                    "bin_index": bin_index,
                    "id": participant,
                    "curve": participant_curve,
                })

    participant_topographies = []
    for participant, participant_df in model_df.groupby("id"):
        rows = participant_df["_array_row"].to_numpy()
        participant_topographies.append(
            topo_values[rows].mean(axis=0)
        )
    grand_topographies = np.mean(
        participant_topographies,
        axis=0,
    )

    montage = mne.channels.make_standard_montage(TOPO_MONTAGE)
    info = mne.create_info(
        ch_names=channels.tolist(),
        sfreq=1.0,
        ch_types="eeg",
    )
    info.set_montage(montage, on_missing="warn")
    roi_mask = np.isin(channels, roi_channels)[:, None]

    if TOPO_VLIM is None:
        topo_absmax = np.nanmax(np.abs(grand_topographies))
        topo_vlim = (-topo_absmax, topo_absmax)
    else:
        topo_vlim = TOPO_VLIM

    n_topographies = len(topo_times)
    n_grid_columns = 2 * n_topographies
    figure_width = max(12.0, 2.25 * n_topographies)

    figure_combined = plt.figure(
        figsize=(figure_width, 11.5)
    )
    grid = figure_combined.add_gridspec(
        3,
        n_grid_columns,
        height_ratios=[1.0, 1.85, 1.65],
        hspace=0.55,
        wspace=0.45,
    )

    topo_axes = [
        figure_combined.add_subplot(
            grid[0, 2 * index:2 * index + 2]
        )
        for index in range(n_topographies)
    ]
    time_axes = [
        figure_combined.add_subplot(
            grid[1, :n_topographies]
        ),
        figure_combined.add_subplot(
            grid[1, n_topographies:]
        ),
    ]
    interaction_axis = figure_combined.add_subplot(
        grid[2, :]
    )

    topo_image = None
    for axis, topo_time, topo_data in zip(
        topo_axes,
        topo_times,
        grand_topographies,
    ):
        topo_image, _ = mne.viz.plot_topomap(
            topo_data,
            info,
            axes=axis,
            show=False,
            cmap="RdBu_r",
            vlim=topo_vlim,
            contours=TOPO_CONTOURS,
            mask=roi_mask,
            mask_params={
                "marker": "o",
                "markerfacecolor": "none",
                "markeredgecolor": "black",
                "linewidth": 1,
                "markersize": 5,
            },
        )
        axis.set_title(f"{topo_time * 1000:.0f} ms")

    colorbar_axis = figure_combined.add_axes(
        [0.92, 0.755, 0.015, 0.145]
    )
    colorbar = figure_combined.colorbar(
        topo_image,
        cax=colorbar_axis,
    )
    colorbar.set_label(
        "Voltage (µV)"
        if measure_name == "erp"
        else "Power (dB)"
    )

    tc_critical = normal_critical_value(
        TIMECOURSE_CI_LEVEL
    )

    for axis, group in zip(time_axes, GROUP_ORDER):
        for bin_index, bin_label in enumerate(
            feedback_bin_labels
        ):
            curves = np.asarray([
                record["curve"]
                for record in timecourse_records
                if record["group"] == group
                and record["feedback_bin"] == bin_label
            ])
            if len(curves) == 0:
                continue

            mean_curve = curves.mean(axis=0)
            axis.plot(
                timecourse_times,
                mean_curve,
                color=bin_colors[bin_index],
                linewidth=1.9,
                label=bin_label,
            )

            if SHOW_TIMECOURSE_CI and len(curves) > 1:
                sem_curve = curves.std(
                    axis=0,
                    ddof=1,
                ) / np.sqrt(len(curves))
                axis.fill_between(
                    timecourse_times,
                    mean_curve - tc_critical * sem_curve,
                    mean_curve + tc_critical * sem_curve,
                    color=bin_colors[bin_index],
                    alpha=0.14,
                    linewidth=0,
                )

        axis.axvspan(
            time_tmin,
            time_tmax,
            color="0.85",
            zorder=0,
        )
        axis.axvline(
            0,
            color="black",
            linewidth=0.9,
            linestyle="--",
        )
        axis.axhline(
            0,
            color="0.5",
            linewidth=0.7,
        )
        axis.set_title(GROUP_TITLES[group])
        axis.set_xlabel("Time from target onset (s)")
        axis.set_xlim(tc_tmin, tc_tmax)

    ylabel = (
        "ROI voltage (µV)"
        if measure_name == "erp"
        else "ROI power (dB)"
    )
    time_axes[0].set_ylabel(ylabel)

    y_limits = [axis.get_ylim() for axis in time_axes]
    shared_ylim = (
        min(limit[0] for limit in y_limits),
        max(limit[1] for limit in y_limits),
    )
    for axis in time_axes:
        axis.set_ylim(shared_ylim)

    handles, labels = time_axes[1].get_legend_handles_labels()
    figure_combined.legend(
        handles,
        labels,
        title="Feedback bin",
        loc="center",
        ncol=N_BINS,
        frameon=False,
        bbox_to_anchor=(0.5, 0.355),
    )

    group_line_styles = {
        "control": "--",
        "experimental": "-",
    }
    group_colors = {
        "control": "C0",
        "experimental": "C1",
    }

    feedback_grid = np.linspace(
        model_df["feedback"].min(),
        model_df["feedback"].max(),
        PREDICTION_POINTS,
    )

    for group in GROUP_ORDER:
        experimental_value = int(group == "experimental")
        prediction, lower, upper = fixed_effect_prediction(
            result,
            feedback_grid,
            experimental_value,
            feedback_mean,
            centered_covariates,
        )

        interaction_axis.plot(
            feedback_grid,
            prediction,
            color=group_colors[group],
            linestyle=group_line_styles[group],
            linewidth=2.2,
            label=GROUP_TITLES[group],
            zorder=3,
        )
        interaction_axis.fill_between(
            feedback_grid,
            lower,
            upper,
            color=group_colors[group],
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )

    interaction_axis.axhline(
        0,
        color="0.5",
        linewidth=0.7,
    )
    interaction_axis.set_xlabel("Feedback")
    interaction_axis.set_ylabel(ylabel)
    interaction_axis.set_title(
        f"Model-predicted "
        f"{time_tmin:.2f}–{time_tmax:.2f} s ROI values"
    )
    interaction_axis.legend(frameon=False)

    figure_combined.suptitle(
        f"{measure_name.capitalize()} ROI time course, "
        f"scalp distribution, and model prediction",
        y=0.985,
    )

    figure_file = PATH_OUT / (
        f"{analysis_stem}_combined_figure.png"
    )
    figure_combined.savefig(
        figure_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure_combined)

    print(f"\nAdded '{roi_column}' to enriched trial table.")
    print(f"Saved checkpoint: {FILE_ENRICHED}")
    print(f"Saved figure:     {figure_file}")


# =============================================================================
# FINAL ENRICHED DATASET
# =============================================================================

enriched_df.to_csv(FILE_ENRICHED, index=False)

print("\n" + "=" * 78)
print("ENRICHED TRIAL-LEVEL DATASET")
print("=" * 78)
print(f"Rows:         {len(enriched_df)}")
print(f"Participants: {enriched_df['id'].nunique()}")
print(f"Saved to:     {FILE_ENRICHED}")

for config in MEASURE_CONFIGS.values():
    column = config["roi_column"]
    print(
        f"{column:16s}: "
        f"{enriched_df[column].notna().sum()} non-missing values"
    )


# =============================================================================
# COMBINED BRAIN-BEHAVIOR MODELS
# =============================================================================

if RUN_COMBINED_BRAIN_BEHAVIOR:
    combined_df = enriched_df.copy()

    combined_df["experimental"] = (
        combined_df["group"] == "experimental"
    ).astype(int)

    combined_df["feedback"] = pd.to_numeric(
        combined_df["feedback"],
        errors="coerce",
    )
    combined_feedback_mean = combined_df["feedback"].mean()
    combined_df["feedback_c"] = (
        combined_df["feedback"] - combined_feedback_mean
    )
    combined_df["feedback2_c"] = (
        combined_df["feedback_c"] ** 2
    )

    combined_centered_covariates = []
    for covariate in COVARIATES_TO_CENTER:
        centered_name = f"{covariate}_c"
        numeric_covariate = pd.to_numeric(
            combined_df[covariate],
            errors="coerce",
        )
        combined_df[centered_name] = (
            numeric_covariate - numeric_covariate.mean()
        )
        combined_centered_covariates.append(centered_name)

    roi_names = list(MEASURE_CONFIGS)
    roi_columns = {
        name: config["roi_column"]
        for name, config in MEASURE_CONFIGS.items()
    }

    combined_required = [
        BEHAVIOR_OUTCOME,
        "id",
        "experimental",
        "feedback_c",
        "feedback2_c",
        *combined_centered_covariates,
        *roi_columns.values(),
    ]
    combined_df = combined_df.dropna(
        subset=combined_required
    ).copy().reset_index(drop=True)

    # Standardized within- and between-participant components.
    for measure_name, roi_column in roi_columns.items():
        participant_mean = combined_df.groupby(
            "id"
        )[roi_column].transform("mean")

        within_raw = (
            combined_df[roi_column] - participant_mean
        )
        between_by_id = combined_df.groupby(
            "id"
        )[roi_column].mean()

        combined_df[f"{measure_name}_within"] = zscore(
            within_raw
        )
        combined_df[f"{measure_name}_between"] = (
            combined_df["id"].map(zscore(between_by_id))
        )

    base_terms = [
        "experimental",
        "feedback_c",
        "feedback2_c",
        "experimental:feedback_c",
        "experimental:feedback2_c",
    ] + combined_centered_covariates

    def combined_formula(selected_measures):
        terms = list(base_terms)
        for selected_measure in selected_measures:
            terms.extend([
                f"{selected_measure}_within",
                f"{selected_measure}_between",
            ])
        return (
            f"{BEHAVIOR_OUTCOME} ~ "
            + " + ".join(terms)
        )

    model_specs = [("base", tuple())]
    for measure_name in roi_names:
        model_specs.append(
            (measure_name, (measure_name,))
        )
    for pair in combinations(roi_names, 2):
        model_specs.append(
            ("+".join(pair), pair)
        )
    model_specs.append(("all", tuple(roi_names)))

    ml_results = {}
    fit_rows = []

    print("\n" + "=" * 78)
    print("COMBINED BRAIN-BEHAVIOR MODELS")
    print("=" * 78)
    print(f"Complete-case trials: {len(combined_df)}")
    print(f"Participants:         {combined_df['id'].nunique()}")

    for model_name, selected_measures in model_specs:
        formula = combined_formula(selected_measures)
        result, optimizer = fit_mixed_model(
            formula=formula,
            data=combined_df,
            re_formula="1 + feedback_c",
            reml=False,
        )
        ml_results[model_name] = result

        fit_rows.append({
            "model": model_name,
            "measures": (
                "+".join(selected_measures)
                if selected_measures
                else "none"
            ),
            "n_obs": int(result.nobs),
            "n_parameters": len(result.params),
            "log_likelihood": result.llf,
            "aic": result.aic,
            "bic": result.bic,
            "optimizer": optimizer,
            "converged": result.converged,
        })

    fit_summary = pd.DataFrame(
        fit_rows
    ).sort_values("aic")
    fit_summary.to_csv(
        PATH_OUT / "combined_rt_model_fit_summary_ml.csv",
        index=False,
    )

    print("\nML model comparison")
    print("-------------------")
    print(fit_summary.to_string(index=False))

    def likelihood_ratio_test(smaller, larger):
        lr = 2 * (larger.llf - smaller.llf)
        df_difference = (
            len(larger.params) - len(smaller.params)
        )
        p_value = chi2.sf(lr, df_difference)
        return lr, df_difference, p_value

    lrt_rows = []

    # Each measure relative to the behavioral base.
    for measure_name in roi_names:
        lr, df_difference, p_value = likelihood_ratio_test(
            ml_results["base"],
            ml_results[measure_name],
        )
        lrt_rows.append({
            "smaller_model": "base",
            "larger_model": measure_name,
            "added_measure": measure_name,
            "lr_chi2": lr,
            "df": df_difference,
            "p": p_value,
        })

    # Unique contribution of each measure after the other two.
    for held_out in roi_names:
        other_measures = tuple(
            name for name in roi_names
            if name != held_out
        )
        smaller_name = "+".join(other_measures)

        lr, df_difference, p_value = likelihood_ratio_test(
            ml_results[smaller_name],
            ml_results["all"],
        )
        lrt_rows.append({
            "smaller_model": smaller_name,
            "larger_model": "all",
            "added_measure": held_out,
            "lr_chi2": lr,
            "df": df_difference,
            "p": p_value,
        })

    lrt_summary = pd.DataFrame(lrt_rows)
    lrt_summary.to_csv(
        PATH_OUT / "combined_rt_nested_likelihood_ratio_tests.csv",
        index=False,
    )

    print("\nNested likelihood-ratio tests")
    print("-----------------------------")
    print(lrt_summary.to_string(index=False))

    # Full model refitted with REML for reporting.
    full_formula = combined_formula(tuple(roi_names))
    full_result, full_optimizer = fit_mixed_model(
        formula=full_formula,
        data=combined_df,
        re_formula="1 + feedback_c",
        reml=BEHAVIOR_REML,
    )

    print("\nFull combined model")
    print("-------------------")
    print(f"Formula:    {full_formula}")
    print(f"Optimizer:  {full_optimizer}")
    print(f"Converged:  {full_result.converged}")
    print()
    print(full_result.summary())

    save_text_result(
        full_result,
        PATH_OUT / "combined_rt_full_model_reml.txt",
        full_optimizer,
    )
    fixed_effect_table(full_result).to_csv(
        PATH_OUT / "combined_rt_full_model_reml_fixed_effects.csv"
    )

print("\nDone.")
