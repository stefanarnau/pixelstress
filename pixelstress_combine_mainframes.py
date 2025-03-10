# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import statsmodels.formula.api as smf
import sklearn.linear_model

# Define paths
path_in = "/mnt/data_dump/pixelstress/dataframe/"
path_cleaned = "/mnt/data_dump/pixelstress/2_autocleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*.pkl")

# Get erp times
erp_times = scipy.io.loadmat(os.path.join(path_cleaned, "vp_7_cleaned_erp.set"))["times"].ravel()

# Load tf times
tf_times = scipy.io.loadmat(os.path.join(path_cleaned, "vp_7_cleaned_tf.set"))["times"].ravel()

# Collector bin
df = []

# Loop datasets
for dataset in datasets:
    
    # Load dataframe
    df.append(pd.read_pickle(os.path.join(path_in, dataset)))
    
# Concatenate
df = pd.concat(df).reset_index()

# Exclude
idx_more_than_4_trials = (df.seq_trial_n_tf > 4).values
df = df[idx_more_than_4_trials]
idx_more_than_4_trials = (df.seq_trial_n_erp > 4).values
df = df[idx_more_than_4_trials]

# Get binned versions of feedback
df["feedback"] = pd.cut(
    df["last_feedback_scaled"],
    bins=3,
    labels=["low",  "mid",  "high"],
)

# Get binned versions of deadline
df["deadline"] = pd.cut(
    df["sequence_nr"],
    bins=2,
    labels=["far", "close"],
)


# Get binned versions of block
df["exp"] = pd.cut(
    df["block_nr"],
    bins=2,
    labels=["start", "end"],
)

dv = "rt"

# Linear mixed model
model = smf.mixedlm(
    dv + " ~ feedback*session_condition*exp",
    data=df,
    groups="id",
)
results = model.fit()
results.summary()


sns.set(rc={'axes.facecolor': 'lightgrey', 'figure.facecolor': 'lightgrey'})
sns.relplot(
    data=df, x="feedback", y=dv, 
    hue="session_condition", style="exp", kind="line", palette=['red', 'black']
)

aa=bb
# ====================================================================


grouped_vectors = df.groupby(['session_condition', 'feedback'])['erp_C'].apply(lambda x: np.mean(np.vstack(x), axis=0))

# Plot the averaged vectors
fig, ax = plt.subplots()
for category, vector in grouped_vectors.items():
    ax.plot(erp_times, vector, label=category)

ax.legend()
ax.set_xlabel('Vector Dimension')
ax.set_ylabel('Average Value')
ax.set_title('Averaged Vectors by Category')
plt.show()

# Group by sequence nr
grouped = df.groupby("sequence_nr")



def average_vectors(row):
    vectors = [row[chan] for chan in chans]
    return np.mean(vectors, axis=0)


grouped["average"] = grouped.apply(average_vectors, axis=1)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Create a color map
cmap = plt.get_cmap("rocket")
norm = Normalize(vmin=grouped["sequence_nr"].min(), vmax=grouped["sequence_nr"].max())

# Plot each averaged vector
for _, row in grouped.iterrows():
    color = cmap(norm(row["sequence_nr"]))
    ax.plot(
        erp_times,
        row["average"],
        color=color,
        alpha=0.7,
        label=f"Fish: {row['sequence_nr']}",
    )

# Add a colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("sequence nr")

# Set labels and title
ax.set_xlabel("ms")
ax.set_ylabel("mV")
ax.set_title("ERP by sequence nr")

plt.tight_layout()
plt.show()
