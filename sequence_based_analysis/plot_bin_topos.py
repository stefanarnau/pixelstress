# -----------------------------------------------------------------------------
# Topographies of sequence-level measures by feedback bin and group
#
# Input:
#   all_subjects_seq_fooof_rt_channelwise_long_car.csv
#
# Expected granularity:
#   one row per subject x sequence x channel
#
# Output:
#   one topomap figure per measure:
#       rows    = group: control / experimental
#       columns = feedback bins
#
# Notes:
# - This script bins signed feedback f globally.
# - Within each group x bin x channel, it first averages repeated sequence rows
#   within subject, then averages across subjects. This avoids overweighting
#   subjects with more available sequences.
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mne
import statsmodels.formula.api as smf

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/topographies/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
MEASURES = [
    "exponent",
    "theta_flat",
    "alpha_flat",
    "cnv_mean",
]

N_BINS = 9
GROUP_ORDER = ["control", "experimental"]

# If None, uses all channels present in the dataframe.
# Otherwise provide a list, e.g. ["Fz", "Cz", "Pz"]
CHANNELS_TO_PLOT = None

# Topomap settings
MONTAGE_NAME = "standard_1020"
CMAP = "RdBu_r"
CONTOURS = 6
SHOW_NAMES = False
SENSORS = True

# Color scaling.
# "global"  = one scale per measure across all groups/bins.
# "group"   = one scale per measure and group.
# "bin"     = one scale per individual map. Not recommended for comparison.
# "symzero" = symmetric scale around zero per measure across all groups/bins.
COLOR_SCALE_MODE = "global"

# Optional manual limits per measure. Values override COLOR_SCALE_MODE.
# Example: MANUAL_LIMITS = {"exponent": (0.8, 1.8), "cnv_mean": (-4, 2)}
MANUAL_LIMITS = {}

# Difference maps are useful for condition effects.
PLOT_EXPERIMENTAL_MINUS_CONTROL = True
DIFF_CMAP = "RdBu_r"
DIFF_SYMMETRIC = True

SAVE_DPI = 200
SHOW_FIGURES = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def make_feedback_bins(df, n_bins):
    edges = np.linspace(df["f"].min(), df["f"].max(), n_bins + 1)
    df = df.copy()
    df["f_bin"] = pd.cut(df["f"], bins=edges, include_lowest=True)
    df["f_mid"] = df["f_bin"].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    return df, edges


def make_info_for_channels(ch_names, sfreq=500.0, montage_name="standard_1020"):
    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, on_missing="warn", match_case=False)
    return info


def compute_subject_balanced_bin_means(df, measure):
    """
    Average sequence rows within subject x group x bin x channel,
    then average subjects within group x bin x channel.
    """
    subj_level = (
        df.groupby(["id", "group", "f_bin", "f_mid", "ch_name"], observed=True)[measure]
        .mean()
        .reset_index(name="value")
    )

    group_level = (
        subj_level.groupby(["group", "f_bin", "f_mid", "ch_name"], observed=True)
        .agg(
            value=("value", "mean"),
            sem=("value", "sem"),
            n_subjects=("id", "nunique"),
        )
        .reset_index()
    )

    return subj_level, group_level


def values_for_map(group_level, group_name, f_bin, ch_names):
    dg = group_level[
        (group_level["group"] == group_name)
        & (group_level["f_bin"].astype(str) == str(f_bin))
    ]
    s = dg.set_index("ch_name")["value"]
    return np.array([s.get(ch, np.nan) for ch in ch_names], dtype=float)


def get_limits(values_by_key, measure, mode="global", manual_limits=None, group_name=None):
    if manual_limits is not None and measure in manual_limits:
        return manual_limits[measure]

    if mode == "bin":
        return None

    arrays = []
    for key, arr in values_by_key.items():
        key_group, _ = key
        if mode == "group" and group_name is not None and key_group != group_name:
            continue
        arrays.append(arr)

    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if np.any(np.isfinite(a))])
    if vals.size == 0:
        return None

    if mode == "symzero":
        vmax = np.nanmax(np.abs(vals))
        return -vmax, vmax

    return np.nanmin(vals), np.nanmax(vals)


def plot_measure_topomaps(
    group_level,
    ch_names,
    info,
    measure,
    group_order,
    f_bins,
    f_mids,
    path_out,
):
    values_by_key = {}
    for group_name in group_order:
        for f_bin in f_bins:
            values_by_key[(group_name, f_bin)] = values_for_map(
                group_level=group_level,
                group_name=group_name,
                f_bin=f_bin,
                ch_names=ch_names,
            )

    fig, axes = plt.subplots(
        nrows=len(group_order),
        ncols=len(f_bins),
        figsize=(2.1 * len(f_bins), 2.5 * len(group_order)),
        squeeze=False,
    )

    im_for_cbar = None

    for r, group_name in enumerate(group_order):
        if COLOR_SCALE_MODE == "group":
            vlim = get_limits(
                values_by_key,
                measure=measure,
                mode="group",
                manual_limits=MANUAL_LIMITS,
                group_name=group_name,
            )
        else:
            vlim = get_limits(
                values_by_key,
                measure=measure,
                mode=COLOR_SCALE_MODE,
                manual_limits=MANUAL_LIMITS,
            )

        for c, f_bin in enumerate(f_bins):
            ax = axes[r, c]
            vals = values_by_key[(group_name, f_bin)]

            if COLOR_SCALE_MODE == "bin" and measure not in MANUAL_LIMITS:
                finite = vals[np.isfinite(vals)]
                if finite.size > 0:
                    vlim_this = (np.nanmin(finite), np.nanmax(finite))
                else:
                    vlim_this = None
            else:
                vlim_this = vlim

            im, _ = mne.viz.plot_topomap(
                vals,
                info,
                axes=ax,
                show=False,
                cmap=CMAP,
                contours=CONTOURS,
                names=ch_names if SHOW_NAMES else None,
                sensors=SENSORS,
                vlim=vlim_this,
            )
            im_for_cbar = im

            if r == 0:
                ax.set_title(f"f={f_mids[c]:.2f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(group_name, fontsize=12)

    fig.suptitle(f"{measure}: topography by feedback bin", y=0.98, fontsize=14)

    if im_for_cbar is not None:
        cbar = fig.colorbar(
            im_for_cbar,
            ax=axes.ravel().tolist(),
            shrink=0.75,
            pad=0.01,
        )
        cbar.set_label(measure)

    fig.savefig(
        path_out / f"topomap_{measure}_by_group_feedback_bin.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_difference_topomaps(
    group_level,
    ch_names,
    info,
    measure,
    f_bins,
    f_mids,
    path_out,
):
    if not set(["control", "experimental"]).issubset(set(group_level["group"].astype(str))):
        return

    diff_values = []
    for f_bin in f_bins:
        exp_vals = values_for_map(group_level, "experimental", f_bin, ch_names)
        ctl_vals = values_for_map(group_level, "control", f_bin, ch_names)
        diff_values.append(exp_vals - ctl_vals)

    all_vals = np.concatenate([v[np.isfinite(v)] for v in diff_values if np.any(np.isfinite(v))])
    if all_vals.size == 0:
        return

    if DIFF_SYMMETRIC:
        vmax = np.nanmax(np.abs(all_vals))
        vlim = (-vmax, vmax)
    else:
        vlim = (np.nanmin(all_vals), np.nanmax(all_vals))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(f_bins),
        figsize=(2.1 * len(f_bins), 2.4),
        squeeze=False,
    )

    im_for_cbar = None
    for c, vals in enumerate(diff_values):
        ax = axes[0, c]
        im, _ = mne.viz.plot_topomap(
            vals,
            info,
            axes=ax,
            show=False,
            cmap=DIFF_CMAP,
            contours=CONTOURS,
            names=ch_names if SHOW_NAMES else None,
            sensors=SENSORS,
            vlim=vlim,
        )
        im_for_cbar = im
        ax.set_title(f"f={f_mids[c]:.2f}", fontsize=10)

    fig.suptitle(f"{measure}: experimental - control", y=1.03, fontsize=14)

    if im_for_cbar is not None:
        cbar = fig.colorbar(
            im_for_cbar,
            ax=axes.ravel().tolist(),
            shrink=0.75,
            pad=0.01,
        )
        cbar.set_label(f"Δ {measure}")

    fig.savefig(
        path_out / f"topomap_{measure}_experimental_minus_control_by_feedback_bin.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_IN)

df["id"] = df["id"].astype(str)
df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER)
df["ch_name"] = df["ch_name"].astype(str)

# -----------------------------------------------------------------------------
# Residualize measures channelwise
# Removes nuisance variance from:
#   - half
#   - mean_trial_difficulty
#
# Keeps:
#   - group
#   - feedback effects (f)
#   - all interactions
#
# Residuals are shifted back by the channelwise grand mean so that:
#   - units remain interpretable
#   - topomaps are easier to read
# -----------------------------------------------------------------------------



RESIDUALIZE_MEASURES = [
    "exponent",
    "offset",
    "delta_raw",
    "theta_raw",
    "alpha_raw",
    "beta_raw",
    "delta_flat",
    "theta_flat",
    "alpha_flat",
    "beta_flat",
    "cnv_mean",
]

RESIDUALIZE_FORMULA = (
    "{measure} ~ half + mean_trial_difficulty"
)

print("\nCreating residualized measures...")

for measure in RESIDUALIZE_MEASURES:

    resid_col = f"{measure}_resid"
    df[resid_col] = np.nan

    for ch_name in sorted(df["ch_name"].dropna().unique()):

        dch = df.loc[
            df["ch_name"] == ch_name,
            [
                measure,
                "half",
                "mean_trial_difficulty",
            ],
        ].copy()

        valid_idx = dch.dropna().index

        if len(valid_idx) < 10:
            continue

        dfit = df.loc[valid_idx].copy()

        try:
            model = smf.ols(
                RESIDUALIZE_FORMULA.format(measure=measure),
                data=dfit,
            ).fit()

            resid_shifted = (
                model.resid
                + dfit[measure].mean()
            )

            df.loc[valid_idx, resid_col] = resid_shifted.values

        except Exception as exc:
            print(
                f"Residualization failed for "
                f"{measure} | {ch_name}: {exc}"
            )

print("Finished residualization.")

required = ["id", "group", "block_nr", "sequence_nr", "ch_name", "f"]
missing_required = [c for c in required if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

missing_measures = [m for m in MEASURES if m not in df.columns]
if missing_measures:
    raise ValueError(f"Missing requested measures: {missing_measures}")

for col in ["f"] + MEASURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if CHANNELS_TO_PLOT is None:
    ch_names = sorted(df["ch_name"].dropna().unique().tolist())
else:
    ch_names = list(CHANNELS_TO_PLOT)
    missing_channels = sorted(set(ch_names) - set(df["ch_name"].unique()))
    if missing_channels:
        raise ValueError(f"Requested channels not present in dataframe: {missing_channels}")

df = df[df["ch_name"].isin(ch_names)].copy()
df = df.dropna(subset=["f", "group", "ch_name"]).copy()

# Use MNE info/montage for topographic coordinates.
info = make_info_for_channels(ch_names, montage_name=MONTAGE_NAME)

# Feedback bins are common across all measures and groups.
df, bin_edges = make_feedback_bins(df, N_BINS)
f_bins = list(df["f_bin"].cat.categories)
f_mids = np.array([(iv.left + iv.right) / 2 for iv in f_bins], dtype=float)

print("Rows:", len(df))
print("Subjects:", df["id"].nunique())
print("Channels:", len(ch_names))
print("Measures:", MEASURES)
print("Feedback bin edges:", bin_edges)


# -----------------------------------------------------------------------------
# Plot measures
# -----------------------------------------------------------------------------
summary_rows = []

for measure in MEASURES:
    d = df.dropna(subset=[measure]).copy()

    if d.empty:
        print(f"Skipping {measure}: no non-missing values")
        continue

    subj_level, group_level = compute_subject_balanced_bin_means(d, measure)

    group_level.to_csv(
        PATH_OUT / f"topomap_input_{measure}_group_bin_channel_means.csv",
        index=False,
    )

    subj_level.to_csv(
        PATH_OUT / f"topomap_input_{measure}_subject_bin_channel_means.csv",
        index=False,
    )

    plot_measure_topomaps(
        group_level=group_level,
        ch_names=ch_names,
        info=info,
        measure=measure,
        group_order=GROUP_ORDER,
        f_bins=f_bins,
        f_mids=f_mids,
        path_out=PATH_OUT,
    )

    if PLOT_EXPERIMENTAL_MINUS_CONTROL:
        plot_difference_topomaps(
            group_level=group_level,
            ch_names=ch_names,
            info=info,
            measure=measure,
            f_bins=f_bins,
            f_mids=f_mids,
            path_out=PATH_OUT,
        )

    summary_rows.append(
        {
            "measure": measure,
            "n_rows": len(d),
            "n_subjects": d["id"].nunique(),
            "n_channels": d["ch_name"].nunique(),
            "min": float(d[measure].min()),
            "max": float(d[measure].max()),
            "mean": float(d[measure].mean()),
            "sd": float(d[measure].std()),
        }
    )

pd.DataFrame(summary_rows).to_csv(
    PATH_OUT / "topomap_measure_summary.csv",
    index=False,
)

print("Finished. Saved topographies to:", PATH_OUT)
