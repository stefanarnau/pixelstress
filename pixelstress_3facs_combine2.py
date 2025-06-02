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
from matplotlib.patches import Rectangle
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
    highlight_idx = [info['ch_names'].index(ch) for ch in ['Cz', 'Fz'] if ch in info['ch_names']]
    
    # Create mask: True for highlighted channels, False otherwise
    mask = np.zeros(len(info['ch_names']), dtype=bool)
    mask[highlight_idx] = True
    
    # Set mask_params for color and size
    mask_params = dict(marker='o', markersize=6, markerfacecolor='#ff00ff')
        
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
            mask_params=mask_params
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
    cbar.set_label('μV', rotation=90, labelpad=10)
    
    # Save
    fn = os.path.join(path_out, erp_label + "_topos.png")
    fig.savefig(fn, dpi=300, bbox_inches='tight')

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
    g.fig.suptitle("ERP at " + ' '.join(channel_selection), y=1.05,
    fontsize=16)

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
    g.fig.suptitle('-'.join([str(x) for x in tf_freqwin]) + ' Hz power at ' + ' '.join(channel_selection), y=1.05,
    fontsize=16)
    
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



    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle(f"ERSP at {' '.join(channel_selection)}", fontsize=24)
    
    axes = axes.flatten()
    
    for i, level in enumerate(factor_levels):
        ax = axes[i]
        subset = new_df[new_df["combined"] == level]
        pivot_data = subset.pivot(index="Hz", columns="s", values="dB")
    
        sns.heatmap(pivot_data, ax=ax, cbar=False, vmin=-4, vmax=4, cmap=colormap)
    
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
            ax.set_ylabel('Hz')
        
        # X-axis: only show ticks and label on bottom row
        if i < 9:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")  # Remove x-axis label
        else:
            ax.set_xlabel('s')
    
        # Draw vertical dashed lines at x = 0 and x = -1.6
        for xpos in [0, -1.6]:
            if xpos in x_vals:
                xpos_index = np.where(x_vals == xpos)[0][0] + 0.5  # Center of the column
                ax.axvline(x=xpos_index, color='black', linestyle='--', linewidth=1)
                
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
            edgecolor='black',
            facecolor='none',
            linestyle='-'
        )
        ax.add_patch(rect)
        
        # Set subplot title font size (slightly larger than default)
        ax.set_title(str(level), fontsize=16)  # or your preferred size
        
        # Increase axis label font size (x and y labels)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
    
        # Increase tick label font size (x and y ticks)
        ax.tick_params(axis='both', labelsize=16)
                
    # Shrink the main grid to make room on the right
    fig.subplots_adjust(right=0.85)  # Leave space for colorbar
    
    # Add a new axis for the colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=-4, vmax=4)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='dB')
                                
    # Remove unused axes
    for j in range(len(factor_levels), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Leave space for colorbar and title
    
    # Save
    fn = os.path.join(path_out, tf_label + "_ersp.png")
    fig.savefig(fn, dpi=300, bbox_inches='tight')

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
