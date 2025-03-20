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
import statsmodels.formula.api as smf
import itertools


# Function for parameterizing and plotting ERP =========================================================================
def get_erp(erp_label, erp_timewin, channel_selection):

    # Get time idx
    erp_times = df_data["erps"][0].times
    erp_win = (erp_times >= erp_timewin[0]) & (erp_times <= erp_timewin[1])

    # Iterate df
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
                    "mV": erp_ts[tidx],
                }
            )

    # Average plotting df across ids
    new_df = (
        pd.DataFrame(new_df)
        .groupby(["stage", "feedback", "group", "s"])["mV"]
        .mean()
        .reset_index()
    )

    # plot erp
    sns.relplot(
        data=new_df,
        x="s",
        y="mV",
        hue="group",
        style="stage",
        col="feedback",
        kind="line",
    )
    plt.show()
    
    # Plot parameters
    sns.relplot(data=df, x="feedback", y=erp_label, hue="stage", style="group", kind="line")
    plt.show()

    return None


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

    # plot freqband
    sns.relplot(
        data=freq_df,
        x="s",
        y="dB",
        hue="group",
        style="stage",
        col="feedback",
        kind="line",
    )
    plt.show()
    
    # Plot parameters
    sns.relplot(data=df, x="feedback", y=tf_label, hue="stage", style="group", kind="line")
    plt.show()
    
    return None


# =================================================================================================================

# Define paths
path_in = "/mnt/data_dump/pixelstress/3_st_data/"

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

# Cretae a combined factor variable
df_data["combined"] = (
    df_data["feedback"].astype(str)
    + " "
    + df_data["stage"].astype(str)
    + " "
    + df_data["group"].astype(str)
)

# Drop eeg data for parameterized df
df = df_data.drop(["erps", "tfrs"], axis=1)

# Plot number of trials ========================================================================================================

# Plot
sns.set_style("whitegrid")
fig, (ax) = plt.subplots(1, 1, figsize=(24, 12))
sns.boxplot(x="combined", y="n_trials", data=df, ax=ax)
ax.set_title("n trials")
plt.tight_layout()
plt.show()

# Plot ERP ======================================================================================================================
get_erp(erp_label="cnv_Fz", erp_timewin=(-0.5, 0), channel_selection=["Fz", "F1", "F2"])
get_erp(erp_label="cnv_FCz", erp_timewin=(-0.5, 0), channel_selection=["FCz", "FC1", "FC2"])
get_erp(erp_label="cnv_Cz", erp_timewin=(-0.5, 0), channel_selection=["Cz", "C1", "C2"])

get_freqband(
    tf_label="frontal_theta_cue_target",
    tf_timewin=(-0.8, -0.2),
    tf_freqwin=(4, 6),
    channel_selection=["FCz", "Fz", "FC1", "FC2", "F1", "F2"],
)

get_freqband(
    tf_label="frontal_alpha",
    tf_timewin=(-0.8, -0.2),
    tf_freqwin=(8, 14),
    channel_selection=["FCz", "Fz", "FC1", "FC2", "F1", "F2"],
)

get_freqband(
    tf_label="posterior_alpha",
    tf_timewin=(-1.2, -0.2),
    tf_freqwin=(8, 14),
    channel_selection=["POz", "Pz", "PO3", "PO4"],
)

get_freqband(
    tf_label="central_beta",
    tf_timewin=(-1.2, -0.2),
    tf_freqwin=(18, 30),
    channel_selection=["Cz", "C1", "C2"],
)

# Save to csv
fn = os.path.join(path_in, "combined.csv")
df.to_csv(fn, index=False)


aa = bb
# Create a 3x4 grid of subplots
fig, axes = plt.subplots(4, 3, figsize=(20, 15), sharex=True, sharey=True)

# Flatten the axes array for easier iteration
axes_flat = axes.flatten()

# Get the unique levels of your factor
factor_levels = [
    "above start experimental",
    "close start experimental",
    "below start experimental",
    "above end experimental",
    "close end experimental",
    "below end experimental",
    "above start control",
    "close start control",
    "below start control",
    "above end control",
    "close end control",
    "below end control",
]

# Iterate through your factor levels
for i, level in enumerate(factor_levels):

    print(level)

    # Filter the dataframe based on the current factor level
    subset = new_df[new_df["combined"] == level]

    # Create a pivot table for the heatmap
    pivot_data = subset.pivot("Hz", "s", "dB")

    # Plot the heatmap in the current subplot
    sns.heatmap(
        pivot_data, ax=axes_flat[i], cbar=False, vmin=-5, vmax=5, cmap="Spectral_r"
    )

    # Set the title for each subplot
    axes_flat[i].set_title(f"{level}")
    axes_flat[i].invert_yaxis()

    axes_flat[i].axvline(x=10, linewidth=2, linestyle="dashed", color="k")

# Add a common colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(axes_flat[0].collections[0], cax=cbar_ax)

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()



# Plot behavior
sns.relplot(data=df, x="feedback", y="rt", hue="stage", style="group", kind="line")
plt.show()
sns.relplot(
    data=df, x="feedback", y="rt_detrended", hue="stage", style="group", kind="line"
)
plt.show()
sns.relplot(
    data=df, x="feedback", y="accuracy", hue="stage", style="group", kind="line"
)
plt.show()

sns.relplot(data=df, x="feedback", y="cnv_Fz", hue="stage", style="group", kind="line")
plt.show()
