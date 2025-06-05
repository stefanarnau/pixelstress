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

# Define paths
path_in = "/mnt/data_dump/pixelstress/3_2fac_data_behavior/"

# Define datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Collector bin
data_in = []

# Collect datasets
for dataset in datasets:

    # Read data
    data = load(os.path.join(path_in, dataset))


    data_in.append(data)

# Concatenate
df_data = pd.concat(data_in).reset_index()
df = df_data

# Drpo participants =======================================================================================

# IDs to exclude
ids_to_drop = [1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 32, 40, 48, 49, 83, 50, 52, 88]

# Remove from dataframes
df = df[~df["id"].isin(ids_to_drop)].reset_index()
df_data = df_data[~df_data["id"].isin(ids_to_drop)].reset_index()

# Plot rt ========================================================================================================

sns.relplot(data=df, x="trajectory", y="rt", style="group", kind="line")
plt.show()

sns.relplot(data=df, x="trajectory", y="rt_resint", style="group", kind="line")
plt.show()

sns.relplot(data=df, x="trajectory", y="rt_residuals", style="group", kind="line")
plt.show()

# Save to csv
fn = os.path.join(path_in, "combined.csv")
df.to_csv(fn, index=False)