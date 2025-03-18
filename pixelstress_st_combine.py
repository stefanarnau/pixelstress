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

# Define paths
path_in = "/mnt/data_dump/pixelstress/3_st_data/"

# Define datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Collector bin
data_in = []

# Collect datasets
for dataset in datasets:
    data_in.append(load(os.path.join(path_in, dataset)))


# Concatenate
df = pd.concat(data_in).reset_index()

# Add variable stage
df = df.assign(stage="start")
df.stage[(df.block_nr >= 5)] = "end"

# Drop non-correct
df = df.dropna(subset=['rt'])

# Drop first sequences
df = df[df['sequence_nr'] >= 1]

# Make grouped df
df_grouped = df.groupby(['stage', 'feedback_binned', 'id'])["rt", "rt_detrended", "group"].mean().reset_index()

# Identify ids where 'rt' is NaN
nan_ids = df_grouped.loc[df_grouped['rt'].isna(), 'id']

# Drop rows where 'id' matches any nan id
df_grouped = df_grouped[~df_grouped["id"].isin(nan_ids)]

# Save to csv
fn = os.path.join(path_in, "combined.csv")
df_grouped.to_csv(fn, index=False)

dv = "rt_detrended"

# Plot
sns.set(rc={'axes.facecolor': 'lightgrey', 'figure.facecolor': 'lightgrey'})
sns.relplot(
    data=df_grouped, x="feedback_binned", y=dv, 
    hue="group", kind="line", style="stage", palette=['darkcyan', 'darkmagenta']
)
   