# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
import scipy.io
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import statsmodels.formula.api as smf
import itertools

# Some global settings
colormap = "rainbow"
lineplot_palette = ["#888888", "#bb11bb", "#11bbbb"]


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
                "stage": row["stage"],
                "feedback": row["feedback"],
                "μV": topovals,
            }
        )

    # Average topo df across ids
    topo_df = (
        pd.DataFrame(topo_df)
        .groupby(["stage", "feedback", "group"])["μV"]
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
        info["ch_names"].index(ch) for ch in ["Cz", "Fz"] if ch in info["ch_names"]
    ]

    # Create mask: True for highlighted channels, False otherwise
    mask = np.zeros(len(info["ch_names"]), dtype=bool)
    mask[highlight_idx] = True

    # Set mask_params for color and size
    mask_params = dict(marker="o", markersize=6, markerfacecolor="#ff00ff")

    for i, ax in enumerate(axes.flat):
        plot_data = topo_df["μV"][i]
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

    # Save
    fn = os.path.join(path_out, erp_label + "_topos.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight")

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
                    "stage": row["stage"],
                    "feedback": row["feedback"],
                    "s": t,
                    "μV": erp_ts[tidx],
                }
            )

    # Average plotting df across ids
    new_df = (
        pd.DataFrame(new_df)
        .groupby(["stage", "feedback", "group", "s"])["μV"]
        .mean()
        .reset_index()
    )

    # Set seaborn params
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # plot erp
    g = sns.relplot(
        data=new_df,
        x="s",
        y="μV",
        hue="feedback",
        style="group",
        col="stage",
        kind="line",
        palette=lineplot_palette,
        col_order=["start", "end"],
    )

    # Highlight x-axis range
    for ax in g.axes.flat:
        ax.axvspan(erp_timewin[0], erp_timewin[1], color="silver", alpha=0.5)
        ax.invert_yaxis()

    # Main title
    g.fig.suptitle("ERP at " + " ".join(channel_selection), y=1.05, fontsize=16)

    # Save
    fn = os.path.join(path_out, erp_label + "_erp.png")
    g.savefig(fn, dpi=300)

    # Plot parameters
    df_plot = df.copy()
    df_plot = df_plot.rename(columns={erp_label: "μV"})
    g = sns.catplot(
        data=df_plot,
        x="group",
        y="μV",
        hue="feedback",
        kind="boxen",
        col="stage",
        k_depth=4,
        palette=lineplot_palette,
        col_order=["start", "end"],
    )

    # Save
    fn = os.path.join(path_out, erp_label + "_boxen.png")
    g.savefig(fn, dpi=300)

    return df


# Function for parameterizing and plotting a frequency band ======================================================================
def get_freqband(tf_label, tf_timewin, tf_freqwin, channel_selection):

    # Get idx
    tf_times = df_data["tfrs"][0].times
    tf_freqs = df_data["tfrs"][0].freqs

    tf_time_idx = (tf_times >= tf_timewin[0]) & (tf_times <= tf_timewin[1])
    tf_freq_idx = (tf_freqs >= tf_freqwin[0]) & (tf_freqs <= tf_freqwin[1])

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
        .groupby(["stage", "feedback", "group", "s"])
        .mean()
        .reset_index()
    )

    # Set seaborn params
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    # plot freqband
    g = sns.relplot(
        data=freq_df,
        x="s",
        y="dB",
        hue="feedback",
        style="group",
        col="stage",
        kind="line",
        palette=lineplot_palette,
        col_order=["start", "end"],
    )

    # Highlight x-axis range
    for ax in g.axes.flat:
        ax.axvspan(tf_timewin[0], tf_timewin[1], color="silver", alpha=0.5)
        ax.invert_yaxis()

    # Main title
    g.fig.suptitle(
        "-".join([str(x) for x in tf_freqwin])
        + " Hz power at "
        + " ".join(channel_selection),
        y=1.05,
        fontsize=16,
    )

    # Save
    fn = os.path.join(path_out, tf_label + "_lineplot.png")
    g.savefig(fn, dpi=300)

    # Plot parameters
    df_plot = df.copy()
    df_plot = df_plot.rename(columns={tf_label: "dB"})
    g = sns.catplot(
        data=df_plot,
        x="group",
        y="dB",
        hue="feedback",
        kind="boxen",
        col="stage",
        k_depth=4,
        palette=lineplot_palette,
        col_order=["start", "end"],
    )

    # Save
    fn = os.path.join(path_out, tf_label + "_boxen.png")
    g.savefig(fn, dpi=300)

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

    # Create a 3x4 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(40, 30), dpi=300, sharex=True, sharey=True)

    plt.subplots_adjust(
        left=0.25,
        right=0.89,
        top=0.97,
        bottom=0.08,
        wspace=0.1,
        hspace=0.2,
    )

    # Flatten the axes array for easier iteration
    axes_flat = axes.flatten()

    # Iterate through your factor levels
    for i, level in enumerate(factor_levels):

        subset = new_df[new_df["combined"] == level]
        pivot_data = subset.pivot("Hz", "s", "dB")

        sns.heatmap(
            pivot_data, ax=axes_flat[i], cbar=False, vmin=-4, vmax=4, cmap=colormap
        )

        axes_flat[i].set_title(f"{level}", fontsize=32)
        axes_flat[i].invert_yaxis()

        # Plot lines at cue onset and at target onset
        idx_onset_cue = np.abs(tf_times - (-1.7)).argmin()
        idx_onset_target = np.abs(tf_times).argmin()
        axes_flat[i].axvline(
            x=idx_onset_cue, linewidth=1.5, linestyle="dashed", color="k"
        )
        axes_flat[i].axvline(
            x=idx_onset_target, linewidth=1.5, linestyle="dashed", color="k"
        )

        # Highlight roi
        idx_onset_roi_time = np.abs(tf_times - tf_timewin[0]).argmin()
        idx_onset_roi_freq = np.abs(tf_freqs - tf_freqwin[0]).argmin()
        idx_offset_roi_time = np.abs(tf_times - tf_timewin[1]).argmin()
        idx_offset_roi_freq = np.abs(tf_freqs - tf_freqwin[1]).argmin()

        # Get with and height
        rect_width, rect_height = (
            idx_offset_roi_time - idx_onset_roi_time,
            idx_offset_roi_freq - idx_onset_roi_freq,
        )

        # Create a rectangle
        rect = patches.Rectangle(
            (idx_onset_roi_time, idx_onset_roi_freq),
            rect_width,
            rect_height,
            fill=False,
            edgecolor="black",
            lw=2,
        )

        # Add the rectangle to the heatmap
        axes_flat[i].add_patch(rect)

        # x-ticks (keep your existing logic)
        xticks = pivot_data.columns.values
        show_xticks = [
            j for j, x in enumerate(xticks) if np.isclose(x % 0.5, 0, atol=1e-6)
        ]
        axes_flat[i].set_xticks(show_xticks)
        axes_flat[i].set_xticklabels(
            [f"{xticks[j]:.1f}" for j in show_xticks], fontsize=36
        )

        # y-ticks (set AFTER heatmap)
        ytick_positions = np.arange(len(pivot_data.index))
        ytick_labels = [f"{freq:.0f}" for freq in pivot_data.index]
        show_positions = [
            pos for pos, freq in enumerate(pivot_data.index) if freq % 4 == 0
        ]
        show_labels = [ytick_labels[pos] for pos in show_positions]

        if i % 3 != 0:
            axes_flat[i].set_ylabel("")
            axes_flat[i].set_yticklabels([])
        else:
            axes_flat[i].set_ylabel("Hz", fontsize=36)
            axes_flat[i].set_yticks(show_positions)
            axes_flat[i].set_yticklabels(show_labels, fontsize=36)

            # Force tick visibility
            axes_flat[i].tick_params(
                axis="y", which="both", labelleft=True, labelcolor="black", length=10
            )
            axes_flat[i].yaxis.set_visible(True)
            for label in axes_flat[i].get_yticklabels():
                label.set_visible(True)
                label.set_color("black")

        # Ensure ticks and labels show
        axes_flat[i].tick_params(axis="y", which="both", labelleft=(i % 3 == 0))

        if i < 9:
            axes_flat[i].set_xlabel("")
            axes_flat[i].set_xticklabels([])
        else:
            axes_flat[i].set_xlabel("s", fontsize=36)

    # Use the last heatmap's QuadMesh for the colorbar (or any, since vmin/vmax are fixed)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(axes_flat[0].collections[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label("dB", fontsize=32)

    # Main title
    fig.suptitle(f"ERSP at {' '.join(channel_selection)}", fontsize=40, y=1.02)

    # Save
    fn = os.path.join(path_out, tf_label + "_ersp.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight")

    return None


# =================================================================================================================

# Define paths
path_in = "/mnt/data_dump/pixelstress/3_3fac_data/"
path_out = "/mnt/data_dump/pixelstress/4_results/"

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

# Rename groups
df_data["group"].replace({"experimental": "exp", "control": "ctrl"}, inplace=True)

# Get an info object
info = df_data["erps"][0].info
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

# Cretae a combined factor variable
df_data["combined"] = (
    df_data["group"].astype(str)
    + " "
    + df_data["feedback"].astype(str)
    + " "
    + df_data["stage"].astype(str)
)

# Drop eeg data for parameterized df
df = df_data.drop(["erps", "tfrs"], axis=1)

# Drpo participants =======================================================================================

# IDs to exclude
ids_to_drop = [1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 32, 40, 48, 49, 50, 52, 83, 88]

# Remove from dataframes
df = df[~df["id"].isin(ids_to_drop)].reset_index()
df_data = df_data[~df_data["id"].isin(ids_to_drop)].reset_index()

# Plot behavior ========================================================================================================

# Set seaborn params
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

g = sns.catplot(
    data=df,
    x="group",
    y="rt",
    hue="feedback",
    kind="boxen",
    col="stage",
    k_depth=4,
    palette=lineplot_palette,
    col_order=["start", "end"],
)

# Save
fn = os.path.join(path_out, "rt_boxen.png")
g.savefig(fn, dpi=300)

g = sns.catplot(
    data=df,
    x="group",
    y="accuracy",
    hue="feedback",
    kind="boxen",
    col="stage",
    k_depth=4,
    palette=lineplot_palette,
    col_order=["start", "end"],
)

# Save
fn = os.path.join(path_out, "acc_boxen.png")
g.savefig(fn, dpi=300)

# Plot ERP ======================================================================================================================

# get_erp(erp_label="cnv_Fz", erp_timewin=(-0.3, 0), channel_selection=["Fz"])
# get_erp(erp_label="cnv_Cz", erp_timewin=(-0.3, 0), channel_selection=["Cz"])

# Plot ERSP ======================================================================================================================

get_freqband(
    tf_label="frontal_theta_target_FCz",
    tf_timewin=(0.1, 0.4),
    tf_freqwin=(4, 7),
    channel_selection=["FCz"],
)

# get_freqband(
#     tf_label="frontal_theta_target_Fz",
#     tf_timewin=(0.1, 0.4),
#     tf_freqwin=(4, 7),
#     channel_selection=["Fz", "FCz", "FC1", "FC2", "F1", "F2"],
# )

# get_freqband(
#     tf_label="alpha_cti",
#     tf_timewin=(-1.2, -0),
#     tf_freqwin=(8, 14),
#     channel_selection=["POz"],
# )

# get_freqband(
#     tf_label="central_beta_cti",
#     tf_timewin=(-1.2, -0.3),
#     tf_freqwin=(16, 20),
#     channel_selection=["C3", "C4", "C1", "C2", "CP3", "CP4"],
# )

# Save to csv
fn = os.path.join(path_out, "stats_table.csv")
df.to_csv(fn, index=False)

aa = bb
