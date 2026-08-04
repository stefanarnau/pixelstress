"""
Plot full-epoch ERP and band-power time courses by group and binned feedback.

Averaging hierarchy:
    1. Average selected channels within each trial.
    2. Average trials within participant × group × feedback bin.
    3. Compute grand mean and SEM across participants.

The script produces:
    - one combined 4 × 2 figure: measures × groups
    - one 1 × 2 figure for each measure
    - a CSV containing the participant-level time courses used for plotting

Expected HDF5 datasets:
    erp, theta, alpha, beta : trials × channels × time
    channels               : channel labels
    times                  : time points in seconds

Expected metadata columns:
    id
    group
    last_feedback_scaled   (change FEEDBACK_COLUMN below if needed)
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR = Path("/mnt/data_dump/pixelstress/3_trial_data")
H5_FILE = DATA_DIR / "trial_level_eeg.h5"
METADATA_FILE = DATA_DIR / "trial_level_metadata.csv"

OUTPUT_DIR = DATA_DIR / "feedback_bin_timecourse_plots"

MEASURES = ["erp", "theta", "alpha", "beta"]

# One shared channel selection for all measures.
# This can be replaced by measure-specific selections below.
CHANNELS = ["FCz"]

# Optional measure-specific channel selections.
# Leave as None to use CHANNELS for every measure.
CHANNELS_BY_MEASURE = {
    "erp": None,
    "theta": None,
    "alpha": None,
    "beta": None,
}

SUBJECT_COLUMN = "id"
GROUP_COLUMN = "group"
FEEDBACK_COLUMN = "last_feedback"

# Explicit plotting order. Change labels here if the metadata uses other values.
GROUP_ORDER = ["control", "experimental"]

# Fixed-width bins preserve the scale of the manipulated feedback variable.
# Five bins over [-1, 1]:
FEEDBACK_BIN_EDGES = np.linspace(-1.0, 1.0, 4)

# Alternative, for seven bins:
# FEEDBACK_BIN_EDGES = np.linspace(-1.0, 1.0, 8)

CORRECT_ONLY = False
ACCURACY_COLUMN = "accuracy"

# Power transformation:
# True  -> participant-specific grand-baseline correction:
#          10 * log10(power / participant grand baseline)
# False -> absolute log power:
#          10 * log10(power)
APPLY_POWER_BASELINE = True
BASELINE_TMIN = -1.5
BASELINE_TMAX = -1.0

# ERP is plotted exactly as stored.
# Set this to True only if an additional participant-specific ERP baseline
# subtraction is desired.
APPLY_ERP_BASELINE = False
ERP_BASELINE_TMIN = -1.5
ERP_BASELINE_TMAX = -1.0

TIME_PLOT_TMIN = -1.5
TIME_PLOT_TMAX = 1.0

SHOW_SEM = True
SEM_ALPHA = 0.18

FIGURE_DPI = 180
SAVE_PDF = True
SHOW_FIGURES = True


# =============================================================================
# HELPERS
# =============================================================================

def decode_labels(values):
    """Convert HDF5 byte-string labels to ordinary strings."""
    labels = []
    for value in values:
        if isinstance(value, bytes):
            labels.append(value.decode("utf-8"))
        else:
            labels.append(str(value))
    return labels


def resolve_group_order(observed_groups):
    """Use requested group order when possible, then append any other groups."""
    observed_groups = list(pd.unique(observed_groups))
    ordered = [group for group in GROUP_ORDER if group in observed_groups]
    ordered.extend(group for group in observed_groups if group not in ordered)
    return ordered


def get_channel_indices(all_channels, requested_channels):
    """Return indices for requested channels and fail clearly if any are absent."""
    missing = [channel for channel in requested_channels if channel not in all_channels]
    if missing:
        raise ValueError(
            f"Channels not found in HDF5 file: {missing}\n"
            f"Available channels:\n{all_channels}"
        )
    return np.sort(
        np.array([all_channels.index(channel) for channel in requested_channels])
    )


def assign_feedback_bins(metadata):
    """Add fixed-width feedback bins and readable labels."""
    metadata = metadata.copy()

    feedback = pd.to_numeric(metadata[FEEDBACK_COLUMN], errors="coerce")
    metadata[FEEDBACK_COLUMN] = feedback

    edges = np.asarray(FEEDBACK_BIN_EDGES, dtype=float)
    if edges.ndim != 1 or len(edges) < 3 or not np.all(np.diff(edges) > 0):
        raise ValueError("FEEDBACK_BIN_EDGES must be a strictly increasing 1D array.")

    labels = [
        f"{left:.2f} to {right:.2f}"
        for left, right in zip(edges[:-1], edges[1:])
    ]

    metadata["feedback_bin"] = pd.cut(
        metadata[FEEDBACK_COLUMN],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
        ordered=True,
    )

    metadata["feedback_bin_center"] = pd.cut(
        metadata[FEEDBACK_COLUMN],
        bins=edges,
        labels=(edges[:-1] + edges[1:]) / 2,
        include_lowest=True,
        right=True,
        ordered=True,
    ).astype(float)

    return metadata, labels


def participant_grand_baseline_power(data, subject_ids, baseline_mask):
    """
    Apply participant-specific grand-baseline correction to power.

    For each participant and selected channel-average time course:
        baseline = mean power over all participant trials and baseline time points
        transformed trial = 10 * log10(trial power / baseline)
    """
    transformed = np.full(data.shape, np.nan, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny

    for subject in pd.unique(subject_ids):
        subject_mask = subject_ids == subject
        subject_data = data[subject_mask]

        baseline = np.nanmean(subject_data[:, baseline_mask])
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError(
                f"Invalid power baseline for participant {subject}: {baseline}"
            )

        transformed[subject_mask] = 10.0 * np.log10(
            np.maximum(subject_data, tiny) / baseline
        )

    return transformed


def participant_grand_baseline_erp(data, subject_ids, baseline_mask):
    """Subtract each participant's grand ERP baseline."""
    transformed = np.full(data.shape, np.nan, dtype=np.float64)

    for subject in pd.unique(subject_ids):
        subject_mask = subject_ids == subject
        subject_data = data[subject_mask]
        baseline = np.nanmean(subject_data[:, baseline_mask])
        transformed[subject_mask] = subject_data - baseline

    return transformed


def load_measure_timecourses(h5_file, measure, channel_indices, metadata, times):
    """Load one measure, average selected channels, and apply requested scaling."""
    dataset = h5_file[measure]

    if dataset.shape[0] != len(metadata):
        raise ValueError(
            f"{measure}: HDF5 has {dataset.shape[0]} trials but metadata has "
            f"{len(metadata)} rows."
        )

    # Reading only selected channels reduces memory use.
    # Result: trials × selected channels × time
    selected = np.asarray(dataset[:, channel_indices, :], dtype=np.float64)

    # Average channels within each trial: trials × time
    trial_timecourses = np.nanmean(selected, axis=1)
    del selected

    subject_ids = metadata[SUBJECT_COLUMN].to_numpy()

    if measure == "erp":
        if APPLY_ERP_BASELINE:
            baseline_mask = (
                (times >= ERP_BASELINE_TMIN)
                & (times <= ERP_BASELINE_TMAX)
            )
            if not np.any(baseline_mask):
                raise ValueError("No ERP time points fall inside the baseline window.")
            trial_timecourses = participant_grand_baseline_erp(
                trial_timecourses,
                subject_ids,
                baseline_mask,
            )
    else:
        if np.nanmin(trial_timecourses) < 0:
            raise ValueError(
                f"{measure} contains negative values before log transformation. "
                "The script expects linear power in the HDF5 file."
            )

        if APPLY_POWER_BASELINE:
            baseline_mask = (
                (times >= BASELINE_TMIN)
                & (times <= BASELINE_TMAX)
            )
            if not np.any(baseline_mask):
                raise ValueError("No power time points fall inside the baseline window.")
            trial_timecourses = participant_grand_baseline_power(
                trial_timecourses,
                subject_ids,
                baseline_mask,
            )
        else:
            tiny = np.finfo(np.float64).tiny
            trial_timecourses = 10.0 * np.log10(
                np.maximum(trial_timecourses, tiny)
            )

    return trial_timecourses


def make_participant_averages(trial_timecourses, metadata, times):
    """
    Average trials within participant × group × feedback bin.

    Returns:
        participant_array:
            rows × time
        participant_info:
            one row per participant × group × feedback bin
    """
    valid = (
        metadata[SUBJECT_COLUMN].notna()
        & metadata[GROUP_COLUMN].notna()
        & metadata["feedback_bin"].notna()
    ).to_numpy()

    metadata_valid = metadata.loc[valid].reset_index(drop=True)
    data_valid = trial_timecourses[valid]

    grouping = metadata_valid.groupby(
        [SUBJECT_COLUMN, GROUP_COLUMN, "feedback_bin"],
        observed=True,
        sort=False,
    ).indices

    participant_timecourses = []
    participant_rows = []

    for (subject, group, feedback_bin), row_indices in grouping.items():
        row_indices = np.asarray(row_indices)
        participant_timecourses.append(
            np.nanmean(data_valid[row_indices], axis=0)
        )
        participant_rows.append(
            {
                SUBJECT_COLUMN: subject,
                GROUP_COLUMN: group,
                "feedback_bin": str(feedback_bin),
                "n_trials": len(row_indices),
            }
        )

    if not participant_timecourses:
        raise ValueError("No valid participant × group × feedback-bin cells remain.")

    participant_array = np.vstack(participant_timecourses)
    participant_info = pd.DataFrame(participant_rows)

    return participant_array, participant_info


def summarize_across_participants(
    participant_array,
    participant_info,
    group_order,
    feedback_labels,
):
    """Compute mean, SEM, and participant count for every plotted curve."""
    summaries = {}

    for group in group_order:
        for feedback_label in feedback_labels:
            mask = (
                participant_info[GROUP_COLUMN].eq(group)
                & participant_info["feedback_bin"].eq(feedback_label)
            ).to_numpy()

            values = participant_array[mask]
            if values.shape[0] == 0:
                continue

            mean = np.nanmean(values, axis=0)

            if values.shape[0] > 1:
                sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            else:
                sem = np.full(values.shape[1], np.nan)

            summaries[(group, feedback_label)] = {
                "mean": mean,
                "sem": sem,
                "n": values.shape[0],
            }

    return summaries


def get_measure_ylabel(measure):
    if measure == "erp":
        return "Amplitude"
    if APPLY_POWER_BASELINE:
        return "Power (dB relative to baseline)"
    return "Log power (dB)"


def plot_measure_on_axes(
    axes,
    measure,
    times,
    summaries,
    group_order,
    feedback_labels,
    colors,
    channel_names,
):
    """Plot one measure in two group-specific axes."""
    for column, group in enumerate(group_order):
        axis = axes[column]

        for color, feedback_label in zip(colors, feedback_labels):
            key = (group, feedback_label)
            if key not in summaries:
                continue

            summary = summaries[key]
            label = f"{feedback_label} (n={summary['n']})"

            axis.plot(
                times,
                summary["mean"],
                color=color,
                linewidth=1.5,
                label=label,
            )

            if SHOW_SEM:
                axis.fill_between(
                    times,
                    summary["mean"] - summary["sem"],
                    summary["mean"] + summary["sem"],
                    color=color,
                    alpha=SEM_ALPHA,
                    linewidth=0,
                )

        axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        axis.set_xlim(TIME_PLOT_TMIN, TIME_PLOT_TMAX)
        axis.set_title(str(group))
        axis.set_xlabel("Time (s)")
        axis.grid(False)

        if column == 0:
            axis.set_ylabel(get_measure_ylabel(measure))

    axes[0].text(
        0.01,
        0.98,
        f"{measure.upper()} | {', '.join(channel_names)}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )


def export_participant_timecourses(
    output_file,
    measure,
    participant_array,
    participant_info,
    times,
):
    """Save the participant-level curves underlying the grand averages."""
    time_columns = [f"time_{time:.3f}" for time in times]
    timecourse_df = pd.DataFrame(participant_array, columns=time_columns)
    output = pd.concat(
        [
            participant_info.reset_index(drop=True),
            timecourse_df,
        ],
        axis=1,
    )
    output.insert(0, "measure", measure)
    output.to_csv(output_file, index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_FILE)

    required_columns = {
        SUBJECT_COLUMN,
        GROUP_COLUMN,
        FEEDBACK_COLUMN,
    }
    if CORRECT_ONLY:
        required_columns.add(ACCURACY_COLUMN)

    missing_columns = sorted(required_columns.difference(metadata.columns))
    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}\n"
            f"Available columns:\n{metadata.columns.tolist()}"
        )

    if CORRECT_ONLY:
        keep = pd.to_numeric(
            metadata[ACCURACY_COLUMN],
            errors="coerce",
        ).eq(1)
        metadata = metadata.loc[keep].reset_index(drop=True)

    metadata, feedback_labels = assign_feedback_bins(metadata)

    observed_groups = metadata[GROUP_COLUMN].dropna().unique()
    group_order = resolve_group_order(observed_groups)

    if len(group_order) != 2:
        raise ValueError(
            "The plotting layout expects exactly two groups. "
            f"Observed groups: {group_order}"
        )

    # One stable color per feedback bin across all measures.
    color_map = plt.get_cmap("viridis")
    colors = color_map(np.linspace(0.08, 0.92, len(feedback_labels)))

    all_results = {}

    with h5py.File(H5_FILE, "r") as h5_file:
        required_datasets = set(MEASURES + ["channels", "times"])
        missing_datasets = sorted(required_datasets.difference(h5_file.keys()))
        if missing_datasets:
            raise ValueError(
                f"Missing HDF5 datasets: {missing_datasets}\n"
                f"Available datasets: {list(h5_file.keys())}"
            )

        all_channels = decode_labels(h5_file["channels"][:])
        times = np.asarray(h5_file["times"][:], dtype=float)

        # Metadata was filtered above, so retain matching HDF5 rows.
        # This assumes metadata originally aligned 1:1 with the HDF5 trials.
        if CORRECT_ONLY:
            original_metadata = pd.read_csv(METADATA_FILE)
            h5_keep = pd.to_numeric(
                original_metadata[ACCURACY_COLUMN],
                errors="coerce",
            ).eq(1).to_numpy()
        else:
            h5_keep = np.ones(h5_file[MEASURES[0]].shape[0], dtype=bool)

        if h5_keep.sum() != len(metadata):
            raise ValueError(
                "Metadata/HDF5 alignment failed after trial filtering: "
                f"{h5_keep.sum()} HDF5 rows versus {len(metadata)} metadata rows."
            )

        for measure in MEASURES:
            requested_channels = CHANNELS_BY_MEASURE.get(measure) or CHANNELS
            channel_indices = get_channel_indices(
                all_channels,
                requested_channels,
            )

            # Load one channel at a time. This avoids h5py fancy-indexing
            # restrictions and keeps memory use moderate.
            dataset = h5_file[measure]
            trial_timecourses = np.zeros(
                (int(h5_keep.sum()), dataset.shape[2]),
                dtype=np.float64,
            )

            for channel_index in channel_indices:
                channel_data = np.asarray(
                    dataset[:, int(channel_index), :],
                    dtype=np.float64,
                )
                trial_timecourses += channel_data[h5_keep]

            trial_timecourses /= len(channel_indices)

            subject_ids = metadata[SUBJECT_COLUMN].to_numpy()

            if measure == "erp":
                if APPLY_ERP_BASELINE:
                    baseline_mask = (
                        (times >= ERP_BASELINE_TMIN)
                        & (times <= ERP_BASELINE_TMAX)
                    )
                    if not np.any(baseline_mask):
                        raise ValueError(
                            "No ERP time points fall inside the baseline window."
                        )
                    trial_timecourses = participant_grand_baseline_erp(
                        trial_timecourses,
                        subject_ids,
                        baseline_mask,
                    )
            else:
                if np.nanmin(trial_timecourses) < 0:
                    raise ValueError(
                        f"{measure} contains negative values before log transformation."
                    )

                if APPLY_POWER_BASELINE:
                    baseline_mask = (
                        (times >= BASELINE_TMIN)
                        & (times <= BASELINE_TMAX)
                    )
                    if not np.any(baseline_mask):
                        raise ValueError(
                            "No power time points fall inside the baseline window."
                        )
                    trial_timecourses = participant_grand_baseline_power(
                        trial_timecourses,
                        subject_ids,
                        baseline_mask,
                    )
                else:
                    tiny = np.finfo(np.float64).tiny
                    trial_timecourses = 10.0 * np.log10(
                        np.maximum(trial_timecourses, tiny)
                    )

            participant_array, participant_info = make_participant_averages(
                trial_timecourses,
                metadata,
                times,
            )

            summaries = summarize_across_participants(
                participant_array,
                participant_info,
                group_order,
                feedback_labels,
            )

            all_results[measure] = {
                "summaries": summaries,
                "channels": requested_channels,
            }

            export_participant_timecourses(
                OUTPUT_DIR / f"{measure}_participant_feedback_bin_timecourses.csv",
                measure,
                participant_array,
                participant_info,
                times,
            )

    # -------------------------------------------------------------------------
    # Combined 4 × 2 figure
    # -------------------------------------------------------------------------
    figure, axes = plt.subplots(
        nrows=len(MEASURES),
        ncols=2,
        figsize=(14, 12),
        sharex=True,
        constrained_layout=True,
    )

    for row, measure in enumerate(MEASURES):
        plot_measure_on_axes(
            axes[row, :],
            measure,
            times,
            all_results[measure]["summaries"],
            group_order,
            feedback_labels,
            colors,
            all_results[measure]["channels"],
        )

    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title=f"{FEEDBACK_COLUMN} bins",
        loc="outside upper center",
        ncol=len(feedback_labels),
        frameon=False,
    )

    figure.suptitle(
        "Full-epoch EEG time courses by group and feedback bin",
        fontsize=14,
    )

    combined_png = OUTPUT_DIR / "all_measures_feedback_bins_by_group.png"
    figure.savefig(combined_png, dpi=FIGURE_DPI, bbox_inches="tight")

    if SAVE_PDF:
        figure.savefig(
            OUTPUT_DIR / "all_measures_feedback_bins_by_group.pdf",
            bbox_inches="tight",
        )

    # -------------------------------------------------------------------------
    # Separate 1 × 2 figures
    # -------------------------------------------------------------------------
    for measure in MEASURES:
        measure_figure, measure_axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(14, 4.5),
            sharex=True,
            constrained_layout=True,
        )

        plot_measure_on_axes(
            measure_axes,
            measure,
            times,
            all_results[measure]["summaries"],
            group_order,
            feedback_labels,
            colors,
            all_results[measure]["channels"],
        )

        handles, labels = measure_axes[1].get_legend_handles_labels()
        measure_figure.legend(
            handles,
            labels,
            title=f"{FEEDBACK_COLUMN} bins",
            loc="outside upper center",
            ncol=len(feedback_labels),
            frameon=False,
        )

        measure_figure.suptitle(
            f"{measure.upper()} by group and feedback bin",
            fontsize=13,
        )

        measure_figure.savefig(
            OUTPUT_DIR / f"{measure}_feedback_bins_by_group.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )

        if SAVE_PDF:
            measure_figure.savefig(
                OUTPUT_DIR / f"{measure}_feedback_bins_by_group.pdf",
                bbox_inches="tight",
            )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")

    print(f"Saved plots and participant-level time courses to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()