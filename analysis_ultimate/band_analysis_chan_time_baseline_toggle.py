# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from scipy.stats import t
from scipy.stats import ttest_ind

from mne.stats import (
    combine_adjacency,
    permutation_cluster_1samp_test,
    permutation_cluster_test,
)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_trial_data/")

FILE_EEG = PATH_IN / "trial_level_eeg.h5"
FILE_METADATA = PATH_IN / "trial_level_metadata.csv"


# -----------------------------------------------------------------------------
# Analysis definition
# -----------------------------------------------------------------------------
TMIN = -1.800
TMAX = 1.000

BASELINE_TMIN = -1.800
BASELINE_TMAX = -1.400

# True: participant-specific grand pre-cue baseline correction in dB
# False: absolute log power in dB, without baseline correction
APPLY_BASELINE_CORRECTION = False

SEED = 42

# Frequency band stored in the HDF5 file: "theta", "alpha", or "beta"
BAND = "theta"
VALID_BANDS = {"theta", "alpha", "beta"}

if BAND not in VALID_BANDS:
    raise ValueError(
        f"BAND must be one of {sorted(VALID_BANDS)}, got {BAND!r}."
    )


CLUSTER_ALPHA = 0.05              # cluster-level significance
CLUSTER_FORMING_P = 0.05          # uncorrected voxel-wise threshold
CLUSTER_TAIL = 0                  # two-sided
N_PERMUTATIONS = 100

# -----------------------------------------------------------------------------
# Load metadata
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_METADATA)

df["experimental"] = (
    df["group"] == "experimental"
).astype(int)

participants = np.sort(
    df["id"].unique()
)

n_participants = len(participants)


# -----------------------------------------------------------------------------
# Load selected frequency-band data
#
# The full analysis epoch is loaded so the same time range can be used with
# or without participant-specific grand-baseline correction.
# -----------------------------------------------------------------------------
with h5py.File(FILE_EEG, mode="r") as h5_file:

    all_times = h5_file["times"][:]

    channels = np.asarray(
        h5_file["channels"].asstr()[:]
    )

    time_mask = (
        (all_times >= TMIN)
        & (all_times <= TMAX)
    )

    time_indices = np.where(time_mask)[0]

    if len(time_indices) == 0:
        raise ValueError(
            "No samples found in the requested analysis interval."
        )

    time_start = time_indices[0]
    time_stop = time_indices[-1] + 1

    times = all_times[
        time_start:time_stop
    ]

    # The upper bound is exclusive so that a cue occurring exactly at
    # -1.4 s cannot enter the baseline.
    baseline_mask = (
        (times >= BASELINE_TMIN)
        & (times < BASELINE_TMAX)
    )

    if APPLY_BASELINE_CORRECTION and not baseline_mask.any():
        raise ValueError(
            "No samples found in the requested baseline interval."
        )

    # Trial × channel × time
    band_data = h5_file[BAND][
        :,
        :,
        time_start:time_stop,
    ]


n_channels = len(channels)
n_times = len(times)


# -----------------------------------------------------------------------------
# Check trial alignment
# -----------------------------------------------------------------------------
if band_data.shape[0] != len(df):
    raise ValueError(
        "The number of selected-band trials does not match the number of metadata rows."
    )


# -----------------------------------------------------------------------------
# Participant-wise OLS
#
# Each participant receives one coefficient map per predictor:
#
# participant × coefficient × channel × time
#
# Coefficients:
# 0 = intercept
# 1 = feedback
# 2 = feedback²
# 3 = trial difficulty
# 4 = half
# -----------------------------------------------------------------------------
coefficient_names = [
    "intercept",
    "feedback",
    "feedback2",
    "sequence_difficulty",
    #"trial_difficulty",
    "half",
]

participant_betas = np.zeros(
    (
        n_participants,
        len(coefficient_names),
        n_channels,
        n_times,
    ),
    dtype=np.float32,
)

participant_groups = np.zeros(
    n_participants,
    dtype=int,
)

participant_ids = df["id"].to_numpy()

for participant_idx, participant in enumerate(participants):

    print(
        f"OLS participant "
        f"{participant_idx + 1}/{n_participants}: "
        f"{participant}"
    )

    participant_mask = (
        participant_ids == participant
    )

    participant_df = df.loc[
        participant_mask
    ]

    # Trial × channel × time
    participant_band = band_data[
        participant_mask,
        :,
        :,
    ].astype(np.float64, copy=False)

    n_trials = participant_band.shape[0]

    if (
        not np.all(np.isfinite(participant_band))
        or np.any(participant_band <= 0)
    ):
        raise ValueError(
            f"{BAND.capitalize()} power contains non-finite or non-positive values for "
            f"participant {participant}."
        )

    if APPLY_BASELINE_CORRECTION:

        # Participant-specific grand pre-cue reference:
        # one value per channel, averaged across all trials and all samples in
        # the baseline interval. This preserves trial-to-trial tonic variation
        # while removing stable between-participant differences in absolute power.
        baseline_reference = participant_band[
            :,
            :,
            baseline_mask,
        ].mean(axis=(0, 2))

        if (
            not np.all(np.isfinite(baseline_reference))
            or np.any(baseline_reference <= 0)
        ):
            raise ValueError(
                f"Invalid {BAND} baseline reference for participant {participant}."
            )

        # dB change relative to the participant's grand pre-cue baseline.
        participant_band = 10.0 * np.log10(
            participant_band
            / baseline_reference[np.newaxis, :, np.newaxis]
        )

    else:

        # Absolute log power in dB. No participant-specific baseline is removed,
        # so tonic between-participant and between-group differences are retained.
        participant_band = 10.0 * np.log10(
            participant_band
        )

    # Trial × (channel × time)
    participant_band_flat = participant_band.reshape(
        n_trials,
        n_channels * n_times,
    )

    design_matrix = np.column_stack(
        [
            np.ones(n_trials),
            participant_df["feedback"].to_numpy(),
            participant_df["feedback2"].to_numpy(),
            participant_df["sequence_difficulty"].to_numpy(),
            #participant_df["trial_difficulty"].to_numpy(),
            participant_df["half"].to_numpy(),
        ]
    )

    betas, _, rank, _ = np.linalg.lstsq(
        design_matrix,
        participant_band_flat,
        rcond=None,
    )

    if rank < design_matrix.shape[1]:
        raise ValueError(
            f"Rank-deficient design matrix for participant {participant}."
        )

    # Coefficient × channel × time
    betas = betas.reshape(
        len(coefficient_names),
        n_channels,
        n_times,
    )

    participant_betas[
        participant_idx,
        :,
        :,
        :,
    ] = betas

    participant_groups[
        participant_idx
    ] = participant_df["experimental"].iloc[0]


# Free trial-level data before permutation testing
del band_data


# -----------------------------------------------------------------------------
# Create EEG information and channel adjacency
# -----------------------------------------------------------------------------
info = mne.create_info(
    ch_names=channels.tolist(),
    sfreq=100,
    ch_types="eeg",
)

montage = mne.channels.make_standard_montage(
    "standard_1020"
)

info.set_montage(
    montage,
    match_case=False,
    on_missing="raise",
)

channel_adjacency, adjacency_channels = (
    mne.channels.find_ch_adjacency(
        info,
        ch_type="eeg",
    )
)

if adjacency_channels != channels.tolist():
    raise ValueError(
        "Channel order in the adjacency matrix does not match the EEG data."
    )


# -----------------------------------------------------------------------------
# Combined channel × time adjacency
#
# The participant data are ordered:
# channel × time
#
# Therefore, the adjacency dimensions must use the same order.
# -----------------------------------------------------------------------------
spatiotemporal_adjacency = combine_adjacency(
    channel_adjacency,
    n_times,
)


# -----------------------------------------------------------------------------
# Cluster-forming thresholds
# -----------------------------------------------------------------------------
def get_t_threshold(
    p_value,
    degrees_of_freedom,
    tail,
):
    """
    Convert an uncorrected cluster-forming p-value into a t threshold.

    tail:
    -1 = negative one-sided test
     0 = two-sided test
     1 = positive one-sided test
    """

    if not 0 < p_value < 1:
        raise ValueError(
            "CLUSTER_FORMING_P must be between 0 and 1."
        )

    if tail == 0:
        return t.ppf(
            1.0 - p_value / 2.0,
            df=degrees_of_freedom,
        )

    if tail == 1:
        return t.ppf(
            1.0 - p_value,
            df=degrees_of_freedom,
        )

    if tail == -1:
        return t.ppf(
            p_value,
            df=degrees_of_freedom,
        )

    raise ValueError(
        "CLUSTER_TAIL must be -1, 0, or 1."
    )


n_control = np.sum(
    participant_groups == 0
)

n_experimental = np.sum(
    participant_groups == 1
)

one_sample_threshold = get_t_threshold(
    p_value=CLUSTER_FORMING_P,
    degrees_of_freedom=n_participants - 1,
    tail=CLUSTER_TAIL,
)

group_threshold = get_t_threshold(
    p_value=CLUSTER_FORMING_P,
    degrees_of_freedom=n_control + n_experimental - 2,
    tail=CLUSTER_TAIL,
)

print(
    f"Cluster-forming p: {CLUSTER_FORMING_P}"
)
print(
    f"Cluster tail:      {CLUSTER_TAIL}"
)
print(
    f"One-sample t:      {one_sample_threshold:.4f}"
)
print(
    f"Group-test t:      {group_threshold:.4f}"
)

# -----------------------------------------------------------------------------
# Independent-groups statistic
# -----------------------------------------------------------------------------
def independent_t_statistic(
    experimental_data,
    control_data,
):
    """
    Pooled-variance independent-samples t statistic.

    Input dimensions:
    participants × features
    """

    return ttest_ind(
        experimental_data,
        control_data,
        axis=0,
        equal_var=True,
        nan_policy="raise",
    ).statistic


# -----------------------------------------------------------------------------
# Helper: convert cluster to channel × time mask
# -----------------------------------------------------------------------------
def reshape_cluster_mask(cluster):
    """
    Ensure that an MNE cluster mask has shape:
    channel × time
    """

    cluster = np.asarray(cluster)

    if cluster.shape == (n_channels, n_times):
        return cluster.astype(bool)

    if cluster.size == n_channels * n_times:
        return cluster.reshape(
            n_channels,
            n_times,
        ).astype(bool)

    raise ValueError(
        f"Unexpected cluster shape: {cluster.shape}"
    )


# -----------------------------------------------------------------------------
# Helper: collect significant clusters
# -----------------------------------------------------------------------------
def combine_significant_clusters(
    clusters,
    cluster_p_values,
    alpha=CLUSTER_ALPHA,
):
    """
    Combine all clusters below alpha into one channel × time mask.
    """

    significant_mask = np.zeros(
        (
            n_channels,
            n_times,
        ),
        dtype=bool,
    )

    for cluster, cluster_p in zip(
        clusters,
        cluster_p_values,
    ):

        if cluster_p < alpha:

            significant_mask |= reshape_cluster_mask(
                cluster
            )

    return significant_mask


# -----------------------------------------------------------------------------
# Helper: store cluster-test result
# -----------------------------------------------------------------------------
def create_result(
    effect,
    coefficient_map,
    statistic,
    clusters,
    cluster_p_values,
):
    """
    Create a standardized result dictionary.
    """

    cluster_p_values = np.asarray(
        cluster_p_values
    )

    if len(cluster_p_values) == 0:
        minimum_cluster_p = 1.0
    else:
        minimum_cluster_p = float(
            np.min(cluster_p_values)
        )

    return {
        "effect": effect,
        "map": coefficient_map,
        "statistic": statistic,
        "clusters": clusters,
        "cluster_p_values": cluster_p_values,
        "minimum_cluster_p": minimum_cluster_p,
        "significant_mask": combine_significant_clusters(
            clusters,
            cluster_p_values,
        ),
    }


# -----------------------------------------------------------------------------
# Masks and coefficient indices
# -----------------------------------------------------------------------------
control_mask = (
    participant_groups == 0
)

experimental_mask = (
    participant_groups == 1
)

coefficient_indices = {
    name: idx
    for idx, name in enumerate(
        coefficient_names
    )
}


# -----------------------------------------------------------------------------
# Spatiotemporal cluster tests
# -----------------------------------------------------------------------------
results = {}


# -----------------------------------------------------------------------------
# Feedback main effect
# -----------------------------------------------------------------------------
effect_data = participant_betas[
    :,
    coefficient_indices["feedback"],
    :,
    :,
]

t_values, clusters, cluster_p_values, _ = (
    permutation_cluster_1samp_test(
        effect_data,
        threshold=one_sample_threshold,
        adjacency=spatiotemporal_adjacency,
        n_permutations=N_PERMUTATIONS,
        tail=CLUSTER_TAIL,
        out_type="mask",
        seed=SEED,
        verbose=True,
    )
)

results["feedback"] = create_result(
    effect="feedback",
    coefficient_map=effect_data.mean(axis=0),
    statistic=t_values,
    clusters=clusters,
    cluster_p_values=cluster_p_values,
)


# -----------------------------------------------------------------------------
# Feedback² main effect
# -----------------------------------------------------------------------------
effect_data = participant_betas[
    :,
    coefficient_indices["feedback2"],
    :,
    :,
]

t_values, clusters, cluster_p_values, _ = (
    permutation_cluster_1samp_test(
        effect_data,
        threshold=one_sample_threshold,
        adjacency=spatiotemporal_adjacency,
        n_permutations=N_PERMUTATIONS,
        tail=CLUSTER_TAIL,
        out_type="mask",
        seed=SEED,
        verbose=True,
    )
)

results["feedback2"] = create_result(
    effect="feedback2",
    coefficient_map=effect_data.mean(axis=0),
    statistic=t_values,
    clusters=clusters,
    cluster_p_values=cluster_p_values,
)


# -----------------------------------------------------------------------------
# Group main effect
#
# This tests the difference between participant-specific intercept maps.
# -----------------------------------------------------------------------------
experimental_data = participant_betas[
    experimental_mask,
    coefficient_indices["intercept"],
    :,
    :,
]

control_data = participant_betas[
    control_mask,
    coefficient_indices["intercept"],
    :,
    :,
]

t_values, clusters, cluster_p_values, _ = (
    permutation_cluster_test(
        [
            experimental_data,
            control_data,
        ],
        stat_fun=independent_t_statistic,
        threshold=group_threshold,
        adjacency=spatiotemporal_adjacency,
        n_permutations=N_PERMUTATIONS,
        tail=CLUSTER_TAIL,
        out_type="mask",
        seed=SEED,
        verbose=True,
    )
)

results["group"] = create_result(
    effect="group",
    coefficient_map=(
        experimental_data.mean(axis=0)
        - control_data.mean(axis=0)
    ),
    statistic=t_values,
    clusters=clusters,
    cluster_p_values=cluster_p_values,
)


# -----------------------------------------------------------------------------
# Group × feedback
# -----------------------------------------------------------------------------
experimental_data = participant_betas[
    experimental_mask,
    coefficient_indices["feedback"],
    :,
    :,
]

control_data = participant_betas[
    control_mask,
    coefficient_indices["feedback"],
    :,
    :,
]

t_values, clusters, cluster_p_values, _ = (
    permutation_cluster_test(
        [
            experimental_data,
            control_data,
        ],
        stat_fun=independent_t_statistic,
        threshold=group_threshold,
        adjacency=spatiotemporal_adjacency,
        n_permutations=N_PERMUTATIONS,
        tail=CLUSTER_TAIL,
        out_type="mask",
        seed=SEED,
        verbose=True,
    )
)

results["group_feedback"] = create_result(
    effect="group_feedback",
    coefficient_map=(
        experimental_data.mean(axis=0)
        - control_data.mean(axis=0)
    ),
    statistic=t_values,
    clusters=clusters,
    cluster_p_values=cluster_p_values,
)


# -----------------------------------------------------------------------------
# Group × feedback²
# -----------------------------------------------------------------------------
experimental_data = participant_betas[
    experimental_mask,
    coefficient_indices["feedback2"],
    :,
    :,
]

control_data = participant_betas[
    control_mask,
    coefficient_indices["feedback2"],
    :,
    :,
]

t_values, clusters, cluster_p_values, _ = (
    permutation_cluster_test(
        [
            experimental_data,
            control_data,
        ],
        stat_fun=independent_t_statistic,
        threshold=group_threshold,
        adjacency=spatiotemporal_adjacency,
        n_permutations=N_PERMUTATIONS,
        tail=CLUSTER_TAIL,
        out_type="mask",
        seed=SEED,
        verbose=True,
    )
)

results["group_feedback2"] = create_result(
    effect="group_feedback2",
    coefficient_map=(
        experimental_data.mean(axis=0)
        - control_data.mean(axis=0)
    ),
    statistic=t_values,
    clusters=clusters,
    cluster_p_values=cluster_p_values,
)


# -----------------------------------------------------------------------------
# Print inferential summary
# -----------------------------------------------------------------------------
effect_order = [
    "group",
    "feedback",
    "feedback2",
    "group_feedback",
    "group_feedback2",
]

effect_titles = {
    "group": "Group",
    "feedback": "Feedback",
    "feedback2": "Feedback²",
    "group_feedback": "Group × feedback",
    "group_feedback2": "Group × feedback²",
}

summary_rows = []

for effect in effect_order:

    result = results[effect]

    significant_cluster_count = np.sum(
        result["cluster_p_values"]
        < CLUSTER_ALPHA
    )

    summary_rows.append(
        {
            "effect": effect_titles[effect],
            "minimum_cluster_p": (
                result["minimum_cluster_p"]
            ),
            "significant_clusters": (
                significant_cluster_count
            ),
        }
    )

results_df = pd.DataFrame(
    summary_rows
)

print()
print(results_df.to_string(index=False))
print()


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def get_plot_definition(result):
    """
    Determine the channels and time points used for descriptive plotting.

    If significant clusters exist:
    - topography is averaged over all time points occupied by a significant
      cluster;
    - time course is averaged over all channels occupied by a significant
      cluster.

    If no significant cluster exists:
    - topography is averaged over the full analysis interval;
    - time course is averaged over all channels;
    - no significance markers are drawn.
    """

    cluster_mask = result["significant_mask"]

    if cluster_mask.any():

        channel_mask = cluster_mask.any(axis=1)
        temporal_mask = cluster_mask.any(axis=0)

    else:

        channel_mask = np.ones(
            n_channels,
            dtype=bool,
        )

        temporal_mask = np.ones(
            n_times,
            dtype=bool,
        )

    topography = result["map"][
        :,
        temporal_mask,
    ].mean(axis=1)

    time_course = result["map"][
        channel_mask,
        :,
    ].mean(axis=0)

    significant_channels = (
        cluster_mask.any(axis=1)
    )

    significant_times = (
        cluster_mask.any(axis=0)
    )

    return (
        topography,
        time_course,
        significant_channels,
        significant_times,
    )


def add_significant_time_regions(
    ax,
    significant_times,
):
    """
    Shade contiguous time intervals occupied by significant clusters.
    """

    padded_mask = np.concatenate(
        [
            [False],
            significant_times,
            [False],
        ]
    )

    transitions = np.diff(
        padded_mask.astype(int)
    )

    starts = np.where(
        transitions == 1
    )[0]

    stops = np.where(
        transitions == -1
    )[0]

    for start, stop in zip(
        starts,
        stops,
    ):

        region_start = times[start]
        region_stop = times[stop - 1]

        ax.axvspan(
            region_start,
            region_stop,
            alpha=0.20,
        )


# -----------------------------------------------------------------------------
# Plot spatiotemporal results
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    len(effect_order),
    figsize=(20, 8),
)

all_topographies = {}

for effect in effect_order:

    (
        topography,
        time_course,
        significant_channels,
        significant_times,
    ) = get_plot_definition(
        results[effect]
    )

    all_topographies[effect] = topography


# Use one common topographic scale across the five selected-band effects
coefficient_limit = max(
    np.max(
        np.abs(topography)
    )
    for topography in all_topographies.values()
)

for effect_idx, effect in enumerate(
    effect_order
):

    result = results[effect]

    (
        topography,
        time_course,
        significant_channels,
        significant_times,
    ) = get_plot_definition(
        result
    )

    # -------------------------------------------------------------------------
    # Topography
    # -------------------------------------------------------------------------
    topography_ax = axes[
        0,
        effect_idx,
    ]

    image, _ = mne.viz.plot_topomap(
        topography,
        info,
        axes=topography_ax,
        show=False,
        cmap="RdBu_r",
        vlim=(
            -coefficient_limit,
            coefficient_limit,
        ),
        contours=6,
        sensors=True,
        mask=significant_channels,
        mask_params={
            "marker": "o",
            "markerfacecolor": "yellow",
            "markeredgecolor": "black",
            "linewidth": 0,
            "markersize": 6,
        },
    )

    topography_ax.set_title(
        f"{effect_titles[effect]}\n"
        f"minimum cluster p = "
        f"{result['minimum_cluster_p']:.3f}"
    )

    fig.colorbar(
        image,
        ax=topography_ax,
        shrink=0.65,
    )

    # -------------------------------------------------------------------------
    # Coefficient time course
    # -------------------------------------------------------------------------
    time_course_ax = axes[
        1,
        effect_idx,
    ]

    time_course_ax.plot(
        times,
        time_course,
    )

    time_course_ax.axhline(
        0,
        linewidth=1,
    )

    time_course_ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    add_significant_time_regions(
        time_course_ax,
        significant_times,
    )

    time_course_ax.set_xlim(
        TMIN,
        TMAX,
    )

    time_course_ax.set_xlabel(
        "Time relative to target (s)"
    )

    if effect_idx == 0:
        time_course_ax.set_ylabel(
            "Coefficient (dB)"
        )

    time_course_ax.set_title(
        "Cluster-channel average"
        if significant_channels.any()
        else "Whole-scalp average"
    )


power_scale_label = (
    "grand-baseline dB"
    if APPLY_BASELINE_CORRECTION
    else "absolute log power (dB)"
)

fig.suptitle(
    f"Participant-wise {BAND} coefficient maps ({power_scale_label}): "
    "channel × time cluster permutation"
)

fig.tight_layout()

plt.show()