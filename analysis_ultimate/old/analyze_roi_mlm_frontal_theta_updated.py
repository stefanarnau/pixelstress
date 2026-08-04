from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.cm import get_cmap
from scipy.ndimage import gaussian_filter1d


# Settings
PATH_IN = Path("/mnt/data_dump/pixelstress/3_trial_data/")
PATH_OUT = PATH_IN / "roi_mlm_results"
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_EEG = PATH_IN / "trial_level_eeg.h5"
FILE_METADATA = PATH_IN / "trial_level_metadata.csv"

MEASURE = "theta"  # "erp", "theta", "alpha", or "beta"
CHANNELS = ["FCz", "Fz", "FC1", "FC2", "Cz"]
TIME_TMIN, TIME_TMAX = 0.15, 0.35

CORRECT_ONLY = False

APPLY_POWER_BASELINE = True
BASELINE_TMIN, BASELINE_TMAX = -1.800, -1.400

COVARIATES_TO_CENTER = [
    "sequence_difficulty",
    # "half",
]

INCLUDE_SEQUENCE_RANDOM_INTERCEPT = False

OPTIMIZER = "bfgs"
MAXITER = 2000
SAVE_TRIAL_VALUES = True

# Visualization settings
CREATE_FIGURES = True
N_BINS = 7
BIN_LABELS = None  # None creates labels from the numerical bin boundaries

TIMECOURSE_TMIN, TIMECOURSE_TMAX = -0.20, 0.60
TIMECOURSE_SMOOTHING_MS = 20
SHOW_TIMECOURSE_CI = False
TIMECOURSE_CI_LEVEL = 0.95

TOPO_TIMES = [0.0, 0.1, 0.2, 0.3, 0.4]
TOPO_HALF_WINDOW = 0.025
TOPO_MONTAGE = "standard_1020"
TOPO_CONTOURS = 6
TOPO_VLIM = None  # None uses one symmetric scale across all topographies

PREDICTION_POINTS = 200
SHOW_PARTICIPANT_BIN_VALUES = True
INTERACTION_CI_LEVEL = 0.95
FIGURE_DPI = 600
SAVE_FIGURE_PDF = True

GROUP_ORDER = ["control", "experimental"]
GROUP_TITLES = {
    "control": "Control",
    "experimental": "Experimental",
}


# Helpers
def normal_critical_value(ci_level):
    """Return a normal-theory critical value for common confidence levels."""
    lookup = {
        0.90: 1.644854,
        0.95: 1.959964,
        0.99: 2.575829,
    }
    if ci_level not in lookup:
        raise ValueError(
            "CI level must currently be one of 0.90, 0.95, or 0.99."
        )
    return lookup[ci_level]


def validate_nonempty_indices(indices, label):
    if len(indices) == 0:
        raise ValueError(f"No samples found for {label}.")


def transform_power_by_participant(
    data,
    baseline_reference,
    participant_ids,
):
    """Convert power to dB and optionally subtract participant-level baselines."""
    transformed = 10 * np.log10(data.astype(np.float64))

    if not APPLY_POWER_BASELINE:
        return transformed

    for participant in np.unique(participant_ids):
        participant_mask = participant_ids == participant
        transformed[participant_mask] -= baseline_reference[participant][None, :, None]

    return transformed


def read_eeg_block(dataset, selected_rows, channel_idx, time_idx):
    """Read a trial × channel × time block without loading the full dataset.

    h5py requires fancy indices to be strictly increasing. Channel indices are
    therefore sorted for the HDF5 read and then restored to the order requested
    by ``channel_idx``.
    """
    selected_rows = np.asarray(selected_rows, dtype=int)
    channel_idx = np.asarray(channel_idx, dtype=int)
    time_idx = np.asarray(time_idx, dtype=int)

    start = int(time_idx.min())
    stop = int(time_idx.max()) + 1
    local_idx = time_idx - start

    channel_read_order = np.argsort(channel_idx)
    sorted_channel_idx = channel_idx[channel_read_order]

    if np.any(np.diff(sorted_channel_idx) <= 0):
        raise ValueError("channel_idx must contain unique channel indices")

    restore_channel_order = np.argsort(channel_read_order)

    block = np.asarray(dataset[:, sorted_channel_idx, start:stop])
    block = block[selected_rows]
    block = block[:, restore_channel_order, :]

    return block[:, :, local_idx]


def participant_baseline_references(
    baseline_data,
    participant_ids,
):
    """Compute one baseline spectrum per participant and channel."""
    log_baseline = 10 * np.log10(baseline_data.astype(np.float64))
    references = {}

    for participant in np.unique(participant_ids):
        participant_mask = participant_ids == participant
        references[participant] = log_baseline[participant_mask].mean(axis=(0, 2))

    return references


def make_feedback_bins(feedback):
    """Create common, equal-width feedback bins across both groups."""
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
            raise ValueError("BIN_LABELS must contain exactly N_BINS labels.")
        labels = BIN_LABELS

    bins = pd.cut(
        feedback,
        bins=edges,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    return bins, edges, labels


def fixed_effect_prediction(result, feedback_values, experimental):
    """Predict fixed effects and their normal-theory confidence interval."""
    fixed_names = list(result.fe_params.index)
    beta = result.fe_params.loc[fixed_names].to_numpy()
    covariance = result.cov_params().loc[fixed_names, fixed_names].to_numpy()

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

    return prediction, prediction - critical * standard_error, prediction + critical * standard_error


def save_figure(fig, stem):
    png_file = PATH_OUT / f"{stem}.png"
    fig.savefig(png_file, dpi=FIGURE_DPI, bbox_inches="tight")
    saved = [png_file]

    if SAVE_FIGURE_PDF:
        pdf_file = PATH_OUT / f"{stem}.pdf"
        fig.savefig(pdf_file, bbox_inches="tight")
        saved.append(pdf_file)

    return saved


# Load metadata and EEG information
df = pd.read_csv(FILE_METADATA)

with h5py.File(FILE_EEG, "r") as h5:
    times = h5["times"][:]
    channels = np.asarray(h5["channels"].asstr()[:])

missing_channels = sorted(set(CHANNELS) - set(channels))
if missing_channels:
    raise ValueError(f"ROI channels not found: {missing_channels}")

channel_indices = np.array([
    np.where(channels == channel)[0][0]
    for channel in CHANNELS
])
all_channel_indices = np.arange(len(channels))

time_indices = np.where(
    (times >= TIME_TMIN) & (times <= TIME_TMAX)
)[0]
baseline_indices = np.where(
    (times >= BASELINE_TMIN) & (times < BASELINE_TMAX)
)[0]
timecourse_indices = np.where(
    (times >= TIMECOURSE_TMIN) & (times <= TIMECOURSE_TMAX)
)[0]

topo_indices = []
for topo_time in TOPO_TIMES:
    indices = np.where(
        (times >= topo_time - TOPO_HALF_WINDOW)
        & (times <= topo_time + TOPO_HALF_WINDOW)
    )[0]
    validate_nonempty_indices(indices, f"topography at {topo_time:.3f} s")
    topo_indices.append(indices)

validate_nonempty_indices(time_indices, "analysis time window")
validate_nonempty_indices(timecourse_indices, "time-course window")
if MEASURE != "erp" and APPLY_POWER_BASELINE:
    validate_nonempty_indices(baseline_indices, "power baseline window")


# Select trials
selection_mask = np.ones(len(df), dtype=bool)

if CORRECT_ONLY:
    selection_mask &= df["accuracy"].to_numpy() == 1

selected_indices = np.where(selection_mask)[0]
model_df = df.loc[selection_mask].copy().reset_index(drop=True)
participant_ids = model_df["id"].to_numpy()


# Extract ROI scalar, ROI time course, and narrow all-channel topography windows
with h5py.File(FILE_EEG, "r") as h5:
    dataset = h5[MEASURE]

    roi_timecourse_data = read_eeg_block(
        dataset, selected_indices, channel_indices, timecourse_indices
    )
    roi_analysis_data = read_eeg_block(
        dataset, selected_indices, channel_indices, time_indices
    )
    topo_data_blocks = [
        read_eeg_block(
            dataset, selected_indices, all_channel_indices, indices
        )
        for indices in topo_indices
    ]

    if MEASURE == "erp":
        roi_timecourses = roi_timecourse_data.astype(np.float64).mean(axis=1)
        eeg_value = roi_analysis_data.astype(np.float64).mean(axis=(1, 2))
        topo_values = np.stack([
            block.astype(np.float64).mean(axis=2)
            for block in topo_data_blocks
        ], axis=1)

    else:
        if APPLY_POWER_BASELINE:
            roi_baseline_data = read_eeg_block(
                dataset, selected_indices, channel_indices, baseline_indices
            )
            roi_baselines = participant_baseline_references(
                roi_baseline_data, participant_ids
            )
        else:
            roi_baselines = None

        roi_timecourse_transformed = transform_power_by_participant(
            roi_timecourse_data, roi_baselines, participant_ids
        )
        roi_timecourses = roi_timecourse_transformed.mean(axis=1)

        analysis_transformed = transform_power_by_participant(
            roi_analysis_data, roi_baselines, participant_ids
        )
        eeg_value = analysis_transformed.mean(axis=(1, 2))

        if APPLY_POWER_BASELINE:
            topo_baseline_data = read_eeg_block(
                dataset, selected_indices, all_channel_indices, baseline_indices
            )
            topo_baselines = participant_baseline_references(
                topo_baseline_data, participant_ids
            )
        else:
            topo_baselines = None

        topo_values = []
        for topo_power in topo_data_blocks:
            topo_transformed = transform_power_by_participant(
                topo_power, topo_baselines, participant_ids
            )
            topo_values.append(topo_transformed.mean(axis=2))
        topo_values = np.stack(topo_values, axis=1)

model_df["eeg_value"] = eeg_value
model_df["_array_row"] = np.arange(len(model_df))


# Prepare predictors
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
    model_df[centered_name] = numeric_covariate - numeric_covariate.mean()
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

valid_mask = model_df[model_columns].notna().all(axis=1).to_numpy()
valid_array_rows = model_df.loc[valid_mask, "_array_row"].to_numpy()
model_df = model_df.loc[valid_mask].copy().reset_index(drop=True)
roi_timecourses = roi_timecourses[valid_array_rows]
topo_values = topo_values[valid_array_rows]
model_df["_array_row"] = np.arange(len(model_df))

model_df["feedback_bin"], feedback_bin_edges, feedback_bin_labels = (
    make_feedback_bins(model_df["feedback"])
)


# Mixed-effects model
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
    vc_formula = {
        "sequence": "0 + C(sequence_uid)"
    }

model = smf.mixedlm(
    formula=formula,
    data=model_df,
    groups=model_df["id"],
    re_formula="1 + feedback_c",
    vc_formula=vc_formula,
)

result = model.fit(
    reml=False,
    method=OPTIMIZER,
    maxiter=MAXITER,
)


# Output
print("\nAnalysis definition")
print("-------------------")
print(f"Measure:           {MEASURE}")
print(f"Channels:          {CHANNELS}")
print(f"Time window:       {TIME_TMIN:.3f} to {TIME_TMAX:.3f} s")
print(f"Correct only:      {CORRECT_ONLY}")

if MEASURE != "erp":
    print(f"Power baseline:    {APPLY_POWER_BASELINE}")
    if APPLY_POWER_BASELINE:
        print(
            f"Baseline window:   "
            f"{BASELINE_TMIN:.3f} to {BASELINE_TMAX:.3f} s"
        )

print(f"Trials:            {len(model_df)}")
print(f"Participants:      {model_df['id'].nunique()}")
print(f"Sequences:         {model_df['sequence_uid'].nunique()}")
print(f"Formula:           {formula}")
print(
    "Random effects:    participant intercept + feedback slope"
    + (
        " + sequence intercept"
        if INCLUDE_SEQUENCE_RANDOM_INTERCEPT
        else ""
    )
)
print()
print(result.summary())


# Save model outputs
channel_label = "-".join(CHANNELS)
trial_label = "correct" if CORRECT_ONLY else "all"

if MEASURE == "erp":
    baseline_label = ""
elif APPLY_POWER_BASELINE:
    baseline_label = "_grandbaseline_db"
else:
    baseline_label = "_absolute_log"

random_label = (
    "participant_feedback_slope_sequence_RE"
    if INCLUDE_SEQUENCE_RANDOM_INTERCEPT
    else "participant_feedback_slope_RE"
)

analysis_stem = (
    f"{MEASURE}"
    f"_{channel_label}"
    f"_{TIME_TMIN:+.3f}_{TIME_TMAX:+.3f}"
    f"_{trial_label}"
    f"{baseline_label}"
    f"_{random_label}"
).replace(".", "p")

summary_file = PATH_OUT / f"{analysis_stem}_model_summary.txt"
coefficient_file = PATH_OUT / f"{analysis_stem}_coefficients.csv"

analysis_definition = f"""Analysis definition
-------------------
Measure:           {MEASURE}
Channels:          {CHANNELS}
Time window:       {TIME_TMIN:.3f} to {TIME_TMAX:.3f} s
Correct only:      {CORRECT_ONLY}
Power baseline:    {APPLY_POWER_BASELINE if MEASURE != "erp" else "n/a"}
Baseline window:   {f"{BASELINE_TMIN:.3f} to {BASELINE_TMAX:.3f} s" if MEASURE != "erp" and APPLY_POWER_BASELINE else "n/a"}
Trials:            {len(model_df)}
Participants:      {model_df["id"].nunique()}
Sequences:         {model_df["sequence_uid"].nunique()}
Formula:           {formula}
Random effects:    participant intercept + feedback slope{" + sequence intercept" if INCLUDE_SEQUENCE_RANDOM_INTERCEPT else ""}
Feedback bins:     {N_BINS}, edges={np.round(feedback_bin_edges, 4).tolist()}

"""

summary_file.write_text(
    analysis_definition + result.summary().as_text(),
    encoding="utf-8",
)

fixed_effect_names = result.fe_params.index
confidence_intervals = result.conf_int().loc[fixed_effect_names]

coefficient_table = pd.DataFrame({
    "term": fixed_effect_names,
    "estimate": result.fe_params,
    "standard_error": result.bse_fe,
    "z_value": result.tvalues.loc[fixed_effect_names],
    "p_value": result.pvalues.loc[fixed_effect_names],
    "ci_low": confidence_intervals[0],
    "ci_high": confidence_intervals[1],
})
coefficient_table.to_csv(coefficient_file, index=False)

saved_files = [
    summary_file,
    coefficient_file,
]

if SAVE_TRIAL_VALUES:
    trial_file = PATH_OUT / f"{analysis_stem}_trial_values.csv"
    model_df.drop(columns="_array_row").to_csv(trial_file, index=False)
    saved_files.append(trial_file)


# Figures
if CREATE_FIGURES:
    cmap = get_cmap("viridis", N_BINS)
    bin_colors = [cmap(index) for index in range(N_BINS)]
    timecourse_times = times[timecourse_indices]

    if TIMECOURSE_SMOOTHING_MS > 0:
        sampling_interval_ms = np.median(np.diff(timecourse_times)) * 1000
        smoothing_sigma = TIMECOURSE_SMOOTHING_MS / sampling_interval_ms
    else:
        smoothing_sigma = 0

    # Participant-level ROI time courses for each feedback bin
    timecourse_records = []
    for group in GROUP_ORDER:
        for bin_index, bin_label in enumerate(feedback_bin_labels):
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

    # Participant-weighted topographies collapsed across group and feedback
    participant_topographies = []
    for participant, participant_df in model_df.groupby("id"):
        rows = participant_df["_array_row"].to_numpy()
        participant_topographies.append(topo_values[rows].mean(axis=0))
    grand_topographies = np.mean(participant_topographies, axis=0)

    montage = mne.channels.make_standard_montage(TOPO_MONTAGE)
    info = mne.create_info(
        ch_names=channels.tolist(),
        sfreq=1.0,
        ch_types="eeg",
    )
    info.set_montage(montage, on_missing="warn")
    roi_mask = np.isin(channels, CHANNELS)[:, None]

    if TOPO_VLIM is None:
        topo_absmax = np.nanmax(np.abs(grand_topographies))
        topo_vlim = (-topo_absmax, topo_absmax)
    else:
        topo_vlim = TOPO_VLIM

    n_topographies = len(TOPO_TIMES)
    if n_topographies == 0:
        raise ValueError("TOPO_TIMES must contain at least one time point")
    
    n_grid_columns = 2 * n_topographies
    figure_width = max(12.0, 2.25 * n_topographies)
    
    figure_temporal = plt.figure(figsize=(figure_width, 7.5))
    grid = figure_temporal.add_gridspec(
        2,
        n_grid_columns,
        height_ratios=[1.0, 2.2],
        hspace=0.35,
        wspace=0.45,
    )
    
    topo_axes = [
        figure_temporal.add_subplot(grid[0, 2 * index:2 * index + 2])
        for index in range(n_topographies)
    ]
    
    time_axes = [
        figure_temporal.add_subplot(grid[1, :n_topographies]),
        figure_temporal.add_subplot(grid[1, n_topographies:]),
    ]
    
    topo_image = None
    for axis, topo_time, topo_data in zip(
        topo_axes,
        TOPO_TIMES,
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

    colorbar_axis = figure_temporal.add_axes([0.92, 0.68, 0.015, 0.18])
    colorbar = figure_temporal.colorbar(topo_image, cax=colorbar_axis)
    colorbar.set_label("Voltage (µV)" if MEASURE == "erp" else "Power (dB)")

    timecourse_critical = normal_critical_value(TIMECOURSE_CI_LEVEL)

    for axis, group in zip(time_axes, GROUP_ORDER):
        for bin_index, bin_label in enumerate(feedback_bin_labels):
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
                linewidth=1.8,
                label=bin_label,
            )

            if SHOW_TIMECOURSE_CI and len(curves) > 1:
                sem_curve = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
                axis.fill_between(
                    timecourse_times,
                    mean_curve - timecourse_critical * sem_curve,
                    mean_curve + timecourse_critical * sem_curve,
                    color=bin_colors[bin_index],
                    alpha=0.14,
                    linewidth=0,
                )

        axis.axvspan(TIME_TMIN, TIME_TMAX, color="0.85", zorder=0)
        axis.axvline(0, color="black", linewidth=0.9, linestyle="--")
        axis.axhline(0, color="0.5", linewidth=0.7)
        axis.set_title(GROUP_TITLES[group])
        axis.set_xlabel("Time from target onset (s)")
        axis.set_xlim(TIMECOURSE_TMIN, TIMECOURSE_TMAX)

    ylabel = "ROI voltage (µV)" if MEASURE == "erp" else "ROI power (dB)"
    time_axes[0].set_ylabel(ylabel)

    y_limits = [axis.get_ylim() for axis in time_axes]
    shared_ylim = (
        min(limit[0] for limit in y_limits),
        max(limit[1] for limit in y_limits),
    )
    for axis in time_axes:
        axis.set_ylim(shared_ylim)

    handles, labels = time_axes[1].get_legend_handles_labels()
    figure_temporal.legend(
        handles,
        labels,
        title="Feedback bin",
        loc="lower center",
        ncol=N_BINS,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure_temporal.suptitle(
        f"{MEASURE.capitalize()} ROI time course and scalp distribution",
        y=0.98,
    )

    temporal_stem = f"{analysis_stem}_timecourse_topographies"
    saved_files.extend(save_figure(figure_temporal, temporal_stem))
    plt.close(figure_temporal)

    # Interaction plot: participant-bin values, descriptive means, model curves
    participant_bins = (
        model_df
        .groupby(["id", "group", "feedback_bin"], observed=True)
        .agg(
            eeg_value=("eeg_value", "mean"),
            feedback=("feedback", "mean"),
        )
        .reset_index()
    )

    figure_interaction, interaction_axis = plt.subplots(figsize=(7.5, 5.5))
    group_line_styles = {
        "control": "--",
        "experimental": "-",
    }
    group_markers = {
        "control": "o",
        "experimental": "s",
    }

    feedback_grid = np.linspace(
        model_df["feedback"].min(),
        model_df["feedback"].max(),
        PREDICTION_POINTS,
    )

    for group in GROUP_ORDER:
        group_bins = participant_bins[participant_bins["group"] == group]
        experimental_value = int(group == "experimental")
        group_color = "C0" if group == "control" else "C1"

        if SHOW_PARTICIPANT_BIN_VALUES:
            interaction_axis.scatter(
                group_bins["feedback"],
                group_bins["eeg_value"],
                s=12,
                alpha=0.16,
                color=group_color,
                linewidths=0,
            )

        descriptive = (
            group_bins
            .groupby("feedback_bin", observed=True)
            .agg(
                feedback=("feedback", "mean"),
                mean=("eeg_value", "mean"),
                sd=("eeg_value", "std"),
                n=("eeg_value", "count"),
            )
            .reset_index()
        )
        descriptive["sem"] = descriptive["sd"] / np.sqrt(descriptive["n"])
        descriptive_critical = normal_critical_value(INTERACTION_CI_LEVEL)

        interaction_axis.errorbar(
            descriptive["feedback"],
            descriptive["mean"],
            yerr=descriptive_critical * descriptive["sem"],
            fmt=group_markers[group],
            markersize=6,
            capsize=3,
            color=group_color,
            linestyle="none",
            label=f"{GROUP_TITLES[group]}: binned data",
            zorder=4,
        )

        prediction, lower, upper = fixed_effect_prediction(
            result,
            feedback_grid,
            experimental_value,
        )
        interaction_axis.plot(
            feedback_grid,
            prediction,
            color=group_color,
            linestyle=group_line_styles[group],
            linewidth=2.2,
            label=f"{GROUP_TITLES[group]}: model",
            zorder=3,
        )
        interaction_axis.fill_between(
            feedback_grid,
            lower,
            upper,
            color=group_color,
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )

    interaction_axis.axhline(0, color="0.5", linewidth=0.7)
    interaction_axis.set_xlabel("Feedback")
    interaction_axis.set_ylabel(ylabel)
    interaction_axis.set_title(
        f"Feedback × group effect in the {TIME_TMIN:.2f}–{TIME_TMAX:.2f} s ROI"
    )
    interaction_axis.legend(frameon=False)
    figure_interaction.tight_layout()

    interaction_stem = f"{analysis_stem}_interaction_predictions"
    saved_files.extend(save_figure(figure_interaction, interaction_stem))
    plt.close(figure_interaction)

    participant_bin_file = PATH_OUT / f"{analysis_stem}_participant_bins.csv"
    participant_bins.to_csv(participant_bin_file, index=False)
    saved_files.append(participant_bin_file)

print("\nSaved:")
for saved_file in saved_files:
    print(saved_file)
