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
df = df.assign(stage=1)
df.stage[(df.block_nr >= 5)] = 2

dv = "posterior_alpha"

# Linear mixed model
model = smf.mixedlm(
    dv + " ~ trajectory*group",
    data=df,
    groups="id",
)
results = model.fit()
results.summary()




sns.set(rc={'axes.facecolor': 'lightgrey', 'figure.facecolor': 'lightgrey'})
sns.relplot(
    data=df, x="feedback_binned", y=dv, 
    hue="group", kind="line", style="stage", palette=['darkcyan', 'darkmagenta']
)
   

aa=bb    

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
    
