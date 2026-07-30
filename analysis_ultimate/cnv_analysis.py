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
# CNV definition
# -----------------------------------------------------------------------------
CNV_TMIN = -0.300
CNV_TMAX = 0.000

N_PERMUTATIONS = 5000


CLUSTER_THRESHOLD = t.ppf(
    1 - 0.025,
    df=69,
)


# -----------------------------------------------------------------------------
# Load metadata
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_METADATA)

df["experimental"] = (
    df["group"] == "experimental"
).astype(int)


# -----------------------------------------------------------------------------
# Load CNV data
# -----------------------------------------------------------------------------
with h5py.File(FILE_EEG, mode="r") as h5_file:

    times = h5_file["times"][:]

    channels = np.asarray(
        h5_file["channels"].asstr()[:]
    )

    time_mask = (
        (times >= CNV_TMIN)
        & (times <= CNV_TMAX)
    )

    time_indices = np.where(time_mask)[0]

    time_start = time_indices[0]
    time_stop = time_indices[-1] + 1

    erp_data = h5_file["erp"][
        :,
        :,
        time_start:time_stop,
    ]

    # Trial × channel
    cnv_data = erp_data.mean(axis=2)


# -----------------------------------------------------------------------------
# Participant-wise OLS
# -----------------------------------------------------------------------------
#
# Each participant receives one coefficient per predictor and electrode.
#
# Design matrix:
# intercept, feedback, feedback², difficulty, half
# -----------------------------------------------------------------------------
participants = np.sort(
    df["id"].unique()
)

effects = [
    "feedback",
    "feedback2",
    "trial_difficulty",
    "half",
]

participant_betas = np.zeros(
    (
        len(participants),
        len(effects),
        len(channels),
    )
)

participant_groups = np.zeros(
    len(participants),
    dtype=int,
)

for participant_idx, participant in enumerate(participants):

    participant_mask = (
        df["id"].to_numpy() == participant
    )

    participant_df = df.loc[
        participant_mask
    ]

    participant_cnv = cnv_data[
        participant_mask,
        :
    ]

    design_matrix = np.column_stack(
        [
            np.ones(len(participant_df)),
            participant_df["feedback"].to_numpy(),
            participant_df["feedback2"].to_numpy(),
            participant_df["trial_difficulty"].to_numpy(),
            participant_df["half"].to_numpy(),
        ]
    )

    betas, _, _, _ = np.linalg.lstsq(
        design_matrix,
        participant_cnv,
        rcond=None,
    )

    # Drop intercept
    participant_betas[
        participant_idx,
        :,
        :
    ] = betas[1:, :]

    participant_groups[
        participant_idx
    ] = participant_df["experimental"].iloc[0]


# -----------------------------------------------------------------------------
# Create channel adjacency
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

adjacency, adjacency_channels = (
    mne.channels.find_ch_adjacency(
        info,
        ch_type="eeg",
    )
)


# -----------------------------------------------------------------------------
# Cluster-permutation helper
# -----------------------------------------------------------------------------
def combine_significant_clusters(
    clusters,
    cluster_p_values,
    alpha=0.05,
):

    significant_mask = np.zeros(
        len(channels),
        dtype=bool,
    )

    for cluster, cluster_p in zip(
        clusters,
        cluster_p_values,
    ):
        if cluster_p < alpha:
            significant_mask |= cluster

    return significant_mask


# -----------------------------------------------------------------------------
# One-sample sign-flip tests
# -----------------------------------------------------------------------------
one_sample_results = {}

for effect_idx, effect in enumerate(effects):

    effect_data = participant_betas[
        :,
        effect_idx,
        :
    ]

    t_values, clusters, cluster_p_values, _ = (
        permutation_cluster_1samp_test(
            effect_data,
            threshold=CLUSTER_THRESHOLD,
            adjacency=adjacency,
            n_permutations=N_PERMUTATIONS,
            tail=0,
            out_type="mask",
            seed=42,
            verbose=False,
        )
    )

    one_sample_results[effect] = {
        "map": effect_data.mean(axis=0),
        "mask": combine_significant_clusters(
            clusters,
            cluster_p_values,
        ),
    }


# -----------------------------------------------------------------------------
# Group-difference tests
#
# These test:
# experimental × feedback
# experimental × feedback²
# -----------------------------------------------------------------------------
control_mask = participant_groups == 0
experimental_mask = participant_groups == 1


def independent_t_statistic(
    experimental_data,
    control_data,
):

    return ttest_ind(
        experimental_data,
        control_data,
        axis=0,
        equal_var=False,
    ).statistic


interaction_results = {}

for effect_idx, effect in enumerate(
    ["feedback", "feedback2"]
):

    experimental_data = participant_betas[
        experimental_mask,
        effect_idx,
        :
    ]

    control_data = participant_betas[
        control_mask,
        effect_idx,
        :
    ]

    t_values, clusters, cluster_p_values, _ = (
        permutation_cluster_test(
            [
                experimental_data,
                control_data,
            ],
            stat_fun=independent_t_statistic,
            threshold=CLUSTER_THRESHOLD,
            adjacency=adjacency,
            n_permutations=N_PERMUTATIONS,
            tail=0,
            out_type="mask",
            seed=42,
            verbose=False,
        )
    )

    interaction_results[effect] = {
        "map": (
            experimental_data.mean(axis=0)
            - control_data.mean(axis=0)
        ),
        "mask": combine_significant_clusters(
            clusters,
            cluster_p_values,
        ),
    }


# -----------------------------------------------------------------------------
# Collect maps
# -----------------------------------------------------------------------------
maps = [
    one_sample_results["feedback"]["map"],
    one_sample_results["feedback2"]["map"],
    interaction_results["feedback"]["map"],
    interaction_results["feedback2"]["map"],
    one_sample_results["trial_difficulty"]["map"],
    one_sample_results["half"]["map"],
]

masks = [
    one_sample_results["feedback"]["mask"],
    one_sample_results["feedback2"]["mask"],
    interaction_results["feedback"]["mask"],
    interaction_results["feedback2"]["mask"],
    one_sample_results["trial_difficulty"]["mask"],
    one_sample_results["half"]["mask"],
]

titles = [
    "Feedback",
    "Feedback²",
    "Group × feedback",
    "Group × feedback²",
    "Trial difficulty",
    "Half",
]


# -----------------------------------------------------------------------------
# Plot coefficient maps
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8),
)

for ax, values, significant_mask, title in zip(
    axes.flat,
    maps,
    masks,
    titles,
):

    coefficient_limit = np.max(
        np.abs(values)
    )

    image, _ = mne.viz.plot_topomap(
        values,
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        vlim=(
            -coefficient_limit,
            coefficient_limit,
        ),
        contours=6,
        sensors=True,
        mask=significant_mask,
        mask_params={
            "marker": "o",
            "markerfacecolor": "yellow",
            "markeredgecolor": "black",
            "linewidth": 0,
            "markersize": 7,
        },
    )

    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.70,
    )


fig.suptitle(
    f"Participant-wise CNV coefficients, "
    f"{CNV_TMIN * 1000:.0f} to "
    f"{CNV_TMAX * 1000:.0f} ms"
)

fig.tight_layout()

plt.show()