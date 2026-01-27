# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scipy.special
import pingouin as pg

# Some global settings
colormap = "rainbow"
lineplot_palette = ["#888888", "#bb11bb", "#11bbbb"]


# Function for ez-diffusion
def ez_diffusion(MRT, VRT, Pc, n_trials, s=0.1):
    """
    Translates the R function get.vaTer to Python.

    Parameters:
    - Pc: Proportion correct (0 < Pc < 1)
    - VRT: Variance of RT (typically correct trials only)
    - MRT: Mean RT (in seconds, correct trials only)
    - s: Scaling parameter for DDM noise (default: 0.1)

    Returns:
    - v: Drift rate
    - a: Boundary separation
    - Ter: Non-decision time
    """

    # Replace Pc if accuracy is 1
    if Pc == 1:
        Pc = 1 - (0.5 / n_trials)

    s2 = s**2

    # Basic validation
    if Pc <= 0 or Pc == 0.5 or Pc >= 1:
        raise ValueError(
            "Pc must be between 0 and 1 and not equal to 0.5 for this method to work."
        )

    # Logit of Pc
    L = scipy.special.logit(Pc)  # same as qlogis in R

    # Compute intermediate term
    x = L * (L * Pc**2 - L * Pc + Pc - 0.5) / VRT

    # Drift rate
    v = np.sign(Pc - 0.5) * s * (x**0.25)

    # Boundary separation
    a = s2 * L / v

    # y = -v * a / s²
    y = -v * a / s2

    # Mean decision time
    MDT = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))

    # Non-decision time
    Ter = MRT - MDT

    return a, v, Ter


# Function for parameterizing and plotting ERP =========================================================================
def get_erp(erp_label, erp_timewin, channel_selection):

    # Get time idx
    erp_times = df_data["erps"][0].times
    erp_win = (erp_times >= erp_timewin[0]) & (erp_times <= erp_timewin[1])

    # Build df for topo
    topo_df = []
    for row_idx, row in df_data.iterrows():
        topovals = row["erps"].copy().data[:, erp_win].mean(axis=1)
        topo_df.append(
            {
                "id": row["id"],
                "group": row["group"],
                "feedback": row["feedback"],
                "μV": topovals,
            }
        )

    # Average topo df across ids
    topo_df = (
        pd.DataFrame(topo_df).groupby(["feedback", "group"])["μV"].mean().reset_index()
    )

    # Re-order topo df
    # new_order = [7, 9, 11, 1, 3, 5, 6, 8, 10, 0, 2, 4]
    # topo_df = topo_df.reindex(new_order).reset_index()

    # Subplot grid for topos
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

    # Find indices of the channels to highlight
    highlight_idx = [
        info["ch_names"].index(ch) for ch in ["Cz", "Fz"] if ch in info["ch_names"]
    ]

    # Create mask: True for highlighted channels, False otherwise
    mask = np.zeros(len(info["ch_names"]), dtype=bool)
    mask[highlight_idx] = True

    # Set mask_params for color and size
    mask_params = dict(marker="o", markersize=6, markerfacecolor="#ff00ff")

    for i, ax in enumerate(axes.flat):
        plot_data = topo_df["μV"][i]
        condition_label = topo_df["group"][i] + " " + topo_df["feedback"][i]
        plot_topo = mne.viz.plot_topomap(
            plot_data,
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap=colormap,
            res=600,
            size=5,
            vlim=(-3.5, 3.5),
            mask=mask,
            mask_params=mask_params,
        )
        ax.set_title(condition_label)

    # Make space for the colorbar
    fig.subplots_adjust(right=0.85)

    # Create colorbar axis
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

    # Create a ScalarMappable for the colorbar
    norm = Normalize(vmin=-3.5, vmax=3.5)
    sm = ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])

    # Add colorbar with label
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("μV", rotation=90, labelpad=10)

    plot_topo = fig

    # Save
    fn = os.path.join(path_out, erp_label + "_topos.png")
    plot_topo.savefig(fn, dpi=300, bbox_inches="tight")

    # Iterate df and create long df including time points as rows
    new_df = []
    for row_idx, row in df_data.iterrows():

        # Get selected channel data
        erp_ts = row["erps"].copy().pick(channel_selection).get_data().mean(axis=0)

        # Save average for statistics
        df.at[row_idx, erp_label] = erp_ts[erp_win].mean()

        # Build df for plotting
        for tidx, t in enumerate(erp_times):

            new_df.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "feedback": row["feedback"],
                    "s": t,
                    "μV": erp_ts[tidx],
                }
            )

    # Average plotting df across ids
    new_df = (
        pd.DataFrame(new_df)
        .groupby(["feedback", "group", "s"])["μV"]
        .mean()
        .reset_index()
    )

    # Set seaborn params
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Plot parameters
    df_plot = df.copy()
    df_plot = df_plot.rename(columns={erp_label: "μV"})

    palette = lineplot_palette

    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 10), constrained_layout=False
    )

    # Line plot (top row, 2 columns)

    ax = axes[0, 0]
    sns.lineplot(
        data=new_df,
        x="s",
        y="μV",
        hue="feedback",
        style="group",
        palette=palette,
        ax=ax,
    )
    ax.axvspan(erp_timewin[0], erp_timewin[1], color="silver", alpha=0.5)
    ax.invert_yaxis()
    axes[0, 0].legend(loc="best")

    # Boxen plot (bottom row, 2 columns)

    ax = axes[1, 0]
    sns.boxenplot(
        data=df_plot,
        x="group",
        y="μV",
        hue="feedback",
        palette=palette,
        k_depth=4,
        ax=ax,
    )
    axes[1, 0].legend(loc="best")

    # Adjust spacing between subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.2)

    # Add row titles
    fig.text(
        0.5,
        0.95,
        "ERP at " + " ".join(channel_selection),
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        0.47,
        "Mean Amplitude",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )

    # Save combined figure
    fn = os.path.join(path_out, erp_label + "_combined.png")
    fig.savefig(fn, dpi=600)
    plt.show()

    return None


# Function for parameterizing and plotting a frequency band ======================================================================
def get_freqband(tf_label, tf_timewin, tf_freqwin, channel_selection, vminmax):

    # Get idx
    tf_times = df_data["tfrs"][0].times
    tf_freqs = df_data["tfrs"][0].freqs

    tf_time_idx = (tf_times >= tf_timewin[0]) & (tf_times <= tf_timewin[1])
    tf_freq_idx = (tf_freqs >= tf_freqwin[0]) & (tf_freqs <= tf_freqwin[1])

    # Build df for topo
    topo_df = []
    for row_idx, row in df_data.iterrows():
        tmp = row["tfrs"].copy().data
        tmp = tmp[:, :, tf_time_idx].mean(axis=2)
        topovals = tmp[:, tf_freq_idx].mean(axis=1)
        topo_df.append(
            {
                "id": row["id"],
                "group": row["group"],
                "stage": row["stage"],
                "feedback": row["feedback"],
                "dB": topovals,
            }
        )

    # Average topo df across ids
    topo_df = (
        pd.DataFrame(topo_df)
        .groupby(["stage", "feedback", "group"])["dB"]
        .mean()
        .reset_index()
    )

    # Re-order topo df
    new_order = [7, 9, 11, 1, 3, 5, 6, 8, 10, 0, 2, 4]
    topo_df = topo_df.reindex(new_order).reset_index()

    # Subplot grid for topos
    nrows, ncols = 2, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

    # Find indices of the channels to highlight
    highlight_idx = [
        info["ch_names"].index(ch) for ch in channel_selection if ch in info["ch_names"]
    ]

    # Create mask: True for highlighted channels, False otherwise
    mask = np.zeros(len(info["ch_names"]), dtype=bool)
    mask[highlight_idx] = True

    # Set mask_params for color and size
    mask_params = dict(marker="o", markersize=6, markerfacecolor="#ff00ff")

    for i, ax in enumerate(axes.flat):
        plot_data = topo_df["dB"][i]
        condition_label = (
            topo_df["group"][i]
            + " "
            + topo_df["feedback"][i]
            + " "
            + topo_df["stage"][i]
        )
        mne.viz.plot_topomap(
            plot_data,
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap=colormap,
            res=300,
            size=5,
            vlim=vminmax,
            mask=mask,
            mask_params=mask_params,
        )
        ax.set_title(condition_label)

    # Make space for the colorbar
    fig.subplots_adjust(right=0.85)

    # Create colorbar axis
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])

    # Create a ScalarMappable for the colorbar
    norm = Normalize(vmin=vminmax[0], vmax=vminmax[1])
    sm = ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])

    # Add colorbar with label
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("dB", rotation=90, labelpad=10)

    # Save
    fn = os.path.join(path_out, tf_label + "_topos.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight")

    # Iterate df
    new_df = []
    for row_idx, row in df_data.iterrows():

        # Get selected channel data
        tfr = row["tfrs"].copy().pick(channel_selection)._data.mean(axis=0)

        # Save average for statistics
        df.at[row_idx, tf_label] = (
            tfr[tf_freq_idx, :].mean(axis=0)[tf_time_idx].mean(axis=0)
        )

        # Build df for plotting
        for tidx, t in enumerate(tf_times):
            for fidx, f in enumerate(tf_freqs):

                new_df.append(
                    {
                        "id": row["id"],
                        "group": row["group"],
                        "stage": row["stage"],
                        "feedback": row["feedback"],
                        "combined": row["combined"],
                        "s": t,
                        "Hz": f,
                        "dB": tfr[fidx, tidx],
                    }
                )

    # Average plotting df across ids
    new_df = (
        pd.DataFrame(new_df)
        .groupby(["stage", "feedback", "group", "combined", "s", "Hz"])["dB"]
        .mean()
        .reset_index()
    )

    # Get freq-specific df
    freq_df = (
        new_df[new_df["Hz"].between(tf_freqwin[0], tf_freqwin[1])]
        .groupby(["stage", "feedback", "group", "s"], observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )

    # Set seaborn style
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # Setup
    stages = ["start", "end"]
    palette = lineplot_palette

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(16, 10), constrained_layout=False
    )

    # Top row: line plots
    for i, stage in enumerate(stages):
        ax = axes[0, i]
        sns.lineplot(
            data=freq_df[freq_df["stage"] == stage],
            x="s",
            y="dB",
            hue="feedback",
            style="group",
            palette=palette,
            ax=ax,
        )
        ax.axvspan(tf_timewin[0], tf_timewin[1], color="silver", alpha=0.5)
        ax.set_title(stage)

    axes[0, 0].legend(loc="best")
    axes[0, 1].get_legend().remove()

    # Bottom row: boxen plots
    df_plot = df.copy()
    df_plot = df_plot.rename(columns={tf_label: "dB"})

    for i, stage in enumerate(stages):
        ax = axes[1, i]
        sns.boxenplot(
            data=df_plot[df_plot["stage"] == stage],
            x="group",
            y="dB",
            hue="feedback",
            palette=palette,
            k_depth=4,
            ax=ax,
        )
        ax.set_title(stage)
        # Remove individual legends
        ax.legend_.remove()

    # Shared legend (bottom center)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    # Add row titles
    fig.text(
        0.5,
        0.95,
        f"{'-'.join(map(str, tf_freqwin))} Hz power at {' '.join(channel_selection)}",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5, 0.47, "Mean Power", ha="center", va="center", fontsize=14, weight="bold"
    )

    # Adjust layout
    fig.subplots_adjust(hspace=0.4, wspace=0.2, bottom=0.08)

    # Save
    fn = os.path.join(path_out, tf_label + "_combined.png")
    fig.savefig(fn, dpi=600, bbox_inches="tight")
    plt.show()

    # Get the unique levels of your factor
    factor_levels = [
        "exp above start",
        "exp close start",
        "exp below start",
        "exp above end",
        "exp close end",
        "exp below end",
        "ctrl above start",
        "ctrl close start",
        "ctrl below start",
        "ctrl above end",
        "ctrl close end",
        "ctrl below end",
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle(f"ERSP at {' '.join(channel_selection)}", fontsize=24)

    axes = axes.flatten()

    for i, level in enumerate(factor_levels):
        ax = axes[i]
        subset = new_df[new_df["combined"] == level]
        pivot_data = subset.pivot(index="Hz", columns="s", values="dB")

        sns.heatmap(
            pivot_data,
            ax=ax,
            cbar=False,
            vmin=vminmax[0],
            vmax=vminmax[1],
            cmap=colormap,
        )

        ax.set_title(str(level))

        # Invert y-axis
        ax.invert_yaxis()

        # Get actual axis values
        x_vals = pivot_data.columns.values.astype(float)
        y_vals = pivot_data.index.values.astype(float)

        # Map: real values -> matrix position
        x_ticks = [i for i, val in enumerate(x_vals) if round(val % 0.5, 2) == 0]
        x_ticklabels = [f"{x_vals[i]:.1f}" for i in x_ticks]

        y_ticks = [i for i, val in enumerate(y_vals) if val % 4 == 0]
        y_ticklabels = [f"{y_vals[i]:.0f}" for i in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticklabels)

        # Y-axis: only show ticks and label on left column
        if i % 3 != 0:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")  # Remove y-axis label
        else:
            ax.set_ylabel("Hz")

        # X-axis: only show ticks and label on bottom row
        if i < 9:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")  # Remove x-axis label
        else:
            ax.set_xlabel("s")

        # Draw vertical dashed lines at x = 0 and x = -1.6
        for xpos in [0, -1.6]:
            if xpos in x_vals:
                xpos_index = (
                    np.where(x_vals == xpos)[0][0] + 0.5
                )  # Center of the column
                ax.axvline(x=xpos_index, color="black", linestyle="--", linewidth=1)

        # Map time (x) and frequency (y) ranges to matrix indices
        # Find the closest matching column index for tf_timewin
        x_start = np.argmin(np.abs(x_vals - tf_timewin[0]))
        x_end = np.argmin(np.abs(x_vals - tf_timewin[1]))

        # Same for frequency (y-axis is row index)
        y_start = np.argmin(np.abs(y_vals - tf_freqwin[0]))
        y_end = np.argmin(np.abs(y_vals - tf_freqwin[1]))

        # Rectangle parameters (in heatmap matrix coordinates)
        x_pos = min(x_start, x_end)
        y_pos = min(y_start, y_end)
        width = abs(x_end - x_start) + 1
        height = abs(y_end - y_start) + 1

        # Add the rectangle
        rect = Rectangle(
            (x_pos, y_pos),  # (x, y)
            width,
            height,
            linewidth=1.5,
            edgecolor="black",
            facecolor="none",
            linestyle="-",
        )
        ax.add_patch(rect)

        # Set subplot title font size (slightly larger than default)
        ax.set_title(str(level), fontsize=16)  # or your preferred size

        # Increase axis label font size (x and y labels)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

        # Increase tick label font size (x and y ticks)
        ax.tick_params(axis="both", labelsize=16)

    # Shrink the main grid to make room on the right
    fig.subplots_adjust(right=0.85)  # Leave space for colorbar

    # Add a new axis for the colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=vminmax[0], vmax=vminmax[1])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="dB")

    # Remove unused axes
    for j in range(len(factor_levels), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Leave space for colorbar and title

    # Save
    fn = os.path.join(path_out, tf_label + "_ersp.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight")

    return None


# =================================================================================================================

# Define paths
path_in = "/mnt/data_dump/pixelstress/3_2fac_data/"
path_out = "/mnt/data_dump/pixelstress/4_results_2fac/"

# Define datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Collector bin
data_in = []

# Collect datasets
for dataset in datasets:

    # Read data
    data = load(os.path.join(path_in, dataset))

    # Check if less than minimum trials in any condition
    if (data["n_trials"] < 5).any():
        continue

    # If not... Collect!
    else:
        data_in.append(data)

# Concatenate
df_data = pd.concat(data_in).reset_index()

# Rename trajectory column
df_data = df_data.rename(columns={"trajectory": "feedback"})

# Rename rt column
df_data = df_data.rename(columns={"rt": "ms"})

# Rename groups
df_data["group"].replace({"experimental": "exp", "control": "ctrl"}, inplace=True)

# Get an info object
info = df_data["erps"][0].info
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

# Cretae a combined factor variable
df_data["combined"] = (
    df_data["group"].astype(str) + " " + df_data["feedback"].astype(str)
)

# Drop eeg data for parameterized df
df = df_data.drop(["erps", "tfrs"], axis=1)

# Drop participants =======================================================================================

# IDs to exclude
ids_to_drop = [1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 32, 40, 48, 49, 50, 52, 83, 88]

# Remove from dataframes
df = df[~df["id"].isin(ids_to_drop)].reset_index()
df_data = df_data[~df_data["id"].isin(ids_to_drop)].reset_index()

# Calculate EZ-diffusion parameters ========================================================================================================
df["drift_rate"] = 0
df["boundary_seperation"] = 0
df["non_decision_time"] = 0

for idx, row in df.iterrows():

    a, v, t0 = ez_diffusion(
        row["rt_mean"], row["rt_std"] ** 2, row["accuracy"], row["n_trials"]
    )
    df.at[idx, "drift_rate"] = v
    df.at[idx, "boundary_seperation"] = a
    df.at[idx, "non_decision_time"] = t0

aa = bb

# Run mixed ANOVA
aov = pg.mixed_anova(
    dv='rt_mean', 
    between='group', 
    within='feedback', 
    subject='id', 
    data=df
)

print(aov)

# Set seaborn params
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Dependent variables (top to bottom)
dvs = [
    ("rt_mean", "Response Times"),
    ("accuracy", "Accuracy"),
    ("drift_rate", "Drift Rate"),
    ("boundary_seperation", "Boundary Separation"),
    ("non_decision_time", "Non-Decision Time"),
]

# Safety check
missing = [dv for dv, _ in dvs if dv not in df.columns]
if missing:
    raise ValueError(f"Missing DV columns in df: {missing}")

# Orders
group_order = sorted(df["group"].unique())

# Figure: rows = DVs, cols = groups
fig_width = max(5 * len(group_order), 8)
fig_height = max(3.5 * len(dvs), 8)
fig, axes = plt.subplots(
    nrows=len(dvs),
    ncols=len(group_order),
    figsize=(fig_width, fig_height),
    sharex=True,
    sharey="row",  # Share y-axis across each DV row
)

# Ensure 2D indexing
axes = np.atleast_2d(axes)

# Plot
for row_idx, (dv, row_title) in enumerate(dvs):
    for col_idx, group in enumerate(group_order):
        ax = axes[row_idx, col_idx]
        df_sub = df[df["group"] == group]

        sns.pointplot(
            data=df_sub,
            x="stage",
            y=dv,
            hue="feedback",
            estimator="mean",
            errorbar="se",
            dodge=0.25,
            join=True,
            markers="o",
            linestyles="-",
            ax=ax,
            palette=lineplot_palette,
            markersize=8,  # Slightly larger markers
        )

        # Only left column gets y-label
        if col_idx == 0:
            ax.set_ylabel(row_title)
        else:
            ax.set_ylabel("")

        # Only bottom row gets x-label
        if row_idx == len(dvs) - 1:
            ax.set_xlabel("Stage")
        else:
            ax.set_xlabel("")

        # Enforce stage order
        ax.set_xticks(range(len(stage_order)))
        ax.set_xticklabels(stage_order)

        # Clean legends (shared later)
        if ax.get_legend() is not None:
            ax.legend_.remove()

        ax.set_title(group)

# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Feedback",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    ncol=len(labels),
    frameon=False,
)

# Adjust spacing
plt.subplots_adjust(
    hspace=0.35,
    wspace=0.2,
    top=0.95,
    bottom=0.12,
)

# Save figure
fn = os.path.join(path_out, "behavior_and_ddm_params_means_se.png")
fig.savefig(fn, dpi=600, bbox_inches="tight")
plt.close(fig)


aa = bb

# Plot ERP ======================================================================================================================

get_erp(erp_label="cnv_Fz", erp_timewin=(-0.3, 0), channel_selection=["Fz"])
get_erp(erp_label="cnv_Cz", erp_timewin=(-0.3, 0), channel_selection=["Cz"])


# Plot ERSP ======================================================================================================================

get_freqband(
    tf_label="mft_target_cross",
    tf_timewin=(0.1, 0.4),
    tf_freqwin=(4, 7),
    channel_selection=["FCz", "Cz", "Fz", "FC1", "FC2"],
    vminmax=(-3, 3),
)

get_freqband(
    tf_label="posterior_alpha_cti",
    tf_timewin=(-1, -0.2),
    tf_freqwin=(8, 14),
    channel_selection=["PO7", "PO8", "O1", "O2"],
    vminmax=(-5, 5),
)

# Save to csv
fn = os.path.join(path_out, "stats_table.csv")
df.to_csv(fn, index=False)
