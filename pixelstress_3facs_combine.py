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


# Function for parameterizing and plotting ERP
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

    return None


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
    df_data["group"].astype(str)
    + " "
    + df_data["stage"].astype(str)
    + " "
    + df_data["feedback"].astype(str)
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
get_erp(erp_label="cnv_Fz", erp_timewin=(-0.3, 0), channel_selection=["Fz"])

channel_selection = ["FCz", "Fz"]
tf_timewin = (-1.2, -0.2)
tf_freqwin = (4, 7)

# Get idx
tf_times = df_data["tfrs"][0].times
tf_freqs = df_data["tfrs"][0].freqs

tf_time_idx = (tf_times >= tf_timewin[0]) & (tf_times <= tf_timewin[1])
tf_freq_idx = (tf_freqs >= tf_freqwin[0]) & (tf_freqs <= tf_freqwin[1])

# Iterate df
new_df = []
for row_idx, row in df_data.iterrows():

    # Get selected channel data
    erp_ts = row["tfrs"].copy().pick(channel_selection).get_data()

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


# Save to csv
fn = os.path.join(path_in, "combined.csv")
df.to_csv(fn, index=False)

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

sns.relplot(
    data=df, x="feedback", y="cnv_Fz", hue="stage", style="group", kind="line"
)
plt.show()