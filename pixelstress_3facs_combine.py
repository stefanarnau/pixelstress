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
path_in = "/mnt/data_dump/pixelstress/3_condition_data/"

# Define datasets
datasets = glob.glob(f"{path_in}/*.joblib")

# Collector bin
data_in = []

# List of ids with less than minumum trials in some conditions
ids_to_exclude = []

# Collect datasets
for dataset in datasets:

    # Read data
    data = load(os.path.join(path_in, dataset))

    # Collect
    data_in.append(data)

    # Check minimum trialcount
    if data["rt"] == None:
        ids_to_exclude.append(data["id"])

    # Set a stricter criterium
    if (data["n_trials_erp"] < 20) | (data["n_trials_tf"] < 20):
        ids_to_exclude.append(data["id"])


# To set
ids_to_exclude = list(set(ids_to_exclude))

# Create dataframe
df = pd.DataFrame(data_in)

# Exclude ids with small number of trials
df = df[~df["id"].isin(ids_to_exclude)]

# Plot number of trials ========================================================================================================

# Cretae a combined factor variable
df["combined"] = (
    df["group"].astype(str)
    + " "
    + df["stage"].astype(str)
    + " "
    + df["feedback"].astype(str)
)

# Plot
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
sns.boxplot(x="combined", y="n_trials_erp", data=df, ax=ax1)
ax1.set_title("n trials erp")
sns.boxplot(x="combined", y="n_trials_tf", data=df, ax=ax2)
ax2.set_title("n trials tf")
plt.tight_layout()
plt.show()

# Plot ERP ======================================================================================================================

# Create CNV averages
for i, row in df.iterrows():
    
    # Get time idx
    time_idx = (row["erp"].times >= - 0.5) & (row["erp"].times <= 0)
    
    # Get average
    df.at[i, "cnv_C"] = row["erp"].copy().pick(["Cz"])._data.mean(axis=0)[time_idx].mean()
    df.at[i, "cnv_FC"] = row["erp"].copy().pick(["FCz"])._data.mean(axis=0)[time_idx].mean()
    df.at[i, "cnv_F"] = row["erp"].copy().pick(["Fz", "F1", "F2"])._data.mean(axis=0)[time_idx].mean()
    
# Select dv
dv="cnv_F"

# Linear mixed model
model = smf.mixedlm(
    dv + " ~ feedback*group*stage",
    data=df,
    groups="id",
)
results = model.fit()
results.summary()


sns.set(rc={'axes.facecolor': 'lightgrey', 'figure.facecolor': 'lightgrey'})
sns.relplot(
    data=df, x="feedback", y=dv, 
    hue="group", style="stage", kind="line", palette=['cyan', 'magenta']
)
   
    

df = df.assign(erp_C=erp_C)
grouped_vectors = df.groupby(['group', 'stage', 'feedback'])['erp_C'].apply(lambda x: np.mean(np.vstack(x), axis=0))

# Plot the averaged vectors
fig, ax = plt.subplots()
for category, vector in grouped_vectors.items():
    ax.plot(erp_times, vector, label=category)

ax.legend()
ax.set_xlabel('Vector Dimension')
ax.set_ylabel('Average Value')
ax.set_title('Averaged Vectors by Category')
plt.show()

aa=bb
    

    
    




# Select elctrodes to plot
sensors = ["Cz"]

# Iterate groups
for g, stage in enumerate(["early", "late"]):
    
    # Get subset
    df_subset = df[df["stage"] == stage]
        
    # Create evokeds dictionary
    evokeds_dict = {}
    for evoked, condition in zip(df_subset["erp"].tolist(), df_subset["combined"].tolist()):
        if condition not in evokeds_dict:
            evokeds_dict[condition] = []
        evokeds_dict[condition].append(evoked)
        
    
    # Set up dicts
    color_dict = {}
    style_dict = {}
    for condition in list(set(df_subset["combined"].tolist())):
        
        if condition.split(" ")[2] == "close":
            color_dict[condition] = "black"
        elif condition.split(" ")[2] == "above":
            color_dict[condition] = "cyan"
        elif condition.split(" ")[2] == "below":
            color_dict[condition] = "magenta"
        if condition.split(" ")[0] == "control":
            style_dict[condition] = ":"
        elif condition.split(" ")[0] == "experimental":
            style_dict[condition] = "-"
          
    # Plot for a specific channel (e.g., 'EEG 001')
    mne.viz.plot_compare_evokeds(
        evokeds_dict,
        picks=sensors,
        colors=color_dict,
        linestyles=style_dict,
        ci=False,
        combine="mean"
    
    )
    
