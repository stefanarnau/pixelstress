# Imports
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import seaborn as sns

# Pandas settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp.set")


# Collect datasets
dfs = []
for dataset in datasets:

    # Read erp trialinfo
    df_erp = pd.read_csv(dataset.split("_cleaned")[0] + "_erp_trialinfo.csv")

    # Read tf trialinfo
    df_tf = pd.read_csv(dataset.split("_cleaned")[0] + "_tf_trialinfo.csv")

    # Get common trials
    to_keep = np.intersect1d(df_erp.trial_nr_total.values, df_tf.trial_nr_total.values)

    # Reduce to common trials
    df = df_erp[df_erp["trial_nr_total"].isin(to_keep)].copy()
    
    # Binarize accuracy
    df["accuracy"] = (df["accuracy"] == 1).astype(int)
    
    # Rename group var
    df = df.rename(columns={"session_condition": "group"})
    df["group"] = df["group"].replace({1: "experimental", 2: "control"})
    
    # Collect
    dfs.append(df)
    
# combine into one dataframe
df_trials = pd.concat(dfs, ignore_index=True)

# IDs to exclude
ids_to_drop = [1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83]
df_trials = df_trials[~df_trials["id"].isin(ids_to_drop)].reset_index(drop=True)

# Remove first sequences (no feedback received yet)
df_trials = df_trials[df_trials.sequence_nr > 1]

# Scale control variables
controls = [
    "trial_difficulty",
]
scaler = StandardScaler()
df_trials[controls] = scaler.fit_transform(df_trials[controls])

# keep only correct trials
df_rt = df_trials[df_trials["accuracy"] == 1].copy()

# make sure these are categorical
df_rt["id"] = df_rt["id"].astype("category")
df_rt["group"] = df_rt["group"].astype("category")

# Aggregate
df_seq_rt = df_rt.groupby(["id", "group", "block_nr", "sequence_nr"]).agg(
    mean_rt=("rt", "mean"),
    mean_trial_difficulty=("trial_difficulty", "mean"),
    mean_feedback=("last_feedback_scaled", "mean"),
).reset_index()

# Soma datatypes
df_seq_rt["group"] = df_seq_rt["group"].astype("category")
df_seq_rt["group"] = df_seq_rt["group"].cat.reorder_categories(["control","experimental"], ordered=False)
df_seq_rt["block_nr"] = df_seq_rt["block_nr"].astype(int)

# log-transform after aggregation
df_seq_rt["mean_log_rt"] = np.log(df_seq_rt["mean_rt"])

# Drop nans
df_seq_rt = df_seq_rt.dropna(subset=[
    "mean_log_rt", 
    "mean_trial_difficulty", 
    "mean_feedback", 
    "block_nr", 
])

# Specify facors
df_seq_rt["f"] = df_seq_rt["mean_feedback"]
df_seq_rt["f2"] = df_seq_rt["f"] ** 2
df_seq_rt["half"] = np.where(df_seq_rt["block_nr"] <= 4, "first", "second")
df_seq_rt["half"] = df_seq_rt["half"].astype("category")


# Model formula
formula = """
mean_log_rt ~ group * f + group * f2
             + mean_trial_difficulty + half
"""

# Specify model
model = smf.mixedlm(
    formula,
    df_seq_rt,
    groups=df_seq_rt["id"],
    re_formula="1 + f + f2"
)

# Fit model
fitted_model = model.fit(method="lbfgs", reml=False, maxiter=4000, disp=False)

# Plot summary
print(fitted_model.summary())

# Some seaborn settings
sns.set_theme(style="whitegrid", context="paper")

# Plotting
df_plot = df_seq_rt.copy()

# Make sure group/half are categorical (or at least consistent)
df_plot["group"] = df_plot["group"].astype("category")
df_plot["half"] = df_plot["half"].astype("category")

# -----------------------------
# 1) Bin feedback for visualization (equal-count bins)
# -----------------------------
n_bins = 18
df_plot["f_bin"] = pd.qcut(df_plot["f"], q=n_bins, duplicates="drop")

# -----------------------------
# 2) Observed binned means with SE ACROSS SUBJECTS (robust)
# -----------------------------

# Ensure id is present/categorical
df_plot["id"] = df_plot["id"].astype("category")

# Bin centers computed from numeric f (safe)
# Use groupby on f_bin and take mean of f (numeric), then map back
bin_centers = df_plot.groupby("f_bin", observed=True)["f"].mean()
df_plot["f_center"] = df_plot["f_bin"].map(bin_centers).astype(float)

# Step A: within-subject mean per (group × bin)
subj_bin = (
    df_plot
    .groupby(["id", "group", "f_bin"], observed=True)["mean_log_rt"]
    .mean()
    .reset_index()
)

# Attach numeric bin centers (constant per bin)
subj_bin["f_center"] = subj_bin["f_bin"].map(bin_centers).astype(float)

# Step B: across-subject mean + SE across subject means
obs = (
    subj_bin
    .groupby(["group", "f_bin"], observed=True)["mean_log_rt"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "mean_log_rt", "std": "sd_log_rt", "count": "n_subjects"})
)

# Attach f_center WITHOUT aggregating it
# (use first; it's identical for all rows in the same bin)
centers_df = (
    subj_bin[["f_bin", "f_center"]]
    .drop_duplicates("f_bin")
    .reset_index(drop=True)
)
obs = obs.merge(centers_df, on="f_bin", how="left")

# SE across subjects
obs["se_log_rt"] = obs["sd_log_rt"] / np.sqrt(obs["n_subjects"])

# Back-transform to RT units (delta-method approx for SE)
obs["mean_rt"] = np.exp(obs["mean_log_rt"])
obs["se_rt"] = obs["mean_rt"] * obs["se_log_rt"]


# -----------------------------
# 3) Model-implied curves, averaged across halves
# -----------------------------
f_grid = np.linspace(df_plot["f"].min(), df_plot["f"].max(), 200)
difficulty_ref = df_plot["mean_trial_difficulty"].mean()

half_levels = list(df_plot["half"].cat.categories)
group_levels = list(df_plot["group"].cat.categories)

pred_rows = []
for g in group_levels:
    preds_half = []
    for h in half_levels:
        tmp = pd.DataFrame({
            "f": f_grid,
            "f2": f_grid**2,
            "group": g,
            "half": h,
            "mean_trial_difficulty": difficulty_ref
        })
        preds_half.append(fitted_model.predict(tmp).to_numpy())

    avg_pred_log_rt = np.mean(np.vstack(preds_half), axis=0)

    pred_rows.append(pd.DataFrame({
        "f": f_grid,
        "group": g,
        "pred_rt": np.exp(avg_pred_log_rt)
    }))

pred = pd.concat(pred_rows, ignore_index=True)

# -----------------------------
# 4) Plot with seaborn (clean color handling)
# -----------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))

# Fixed palette so curves/points/errorbars match
palette = sns.color_palette(n_colors=len(group_levels))
color_map = dict(zip(group_levels, palette))

# Model curves
sns.lineplot(
    data=pred,
    x="f",
    y="pred_rt",
    hue="group",
    palette=color_map,
    linewidth=2.5,
    ax=ax
)

# Observed binned points (no legend duplication)
sns.scatterplot(
    data=obs,
    x="f_center",
    y="mean_rt",
    hue="group",
    palette=color_map,
    s=60,
    edgecolor="none",
    ax=ax,
    legend=False
)

# Error bars per group (colors match)
for g, sub in obs.groupby("group", observed=True):
    ax.errorbar(
        sub["f_center"].to_numpy(),
        sub["mean_rt"].to_numpy(),
        yerr=sub["se_rt"].to_numpy(),
        fmt="none",
        ecolor=color_map[g],
        elinewidth=1.2,
        capsize=3,
        alpha=0.7
    )

# Decorations
ax.axvline(0, linestyle="--", color="black", linewidth=1)
ax.set_xlabel("Feedback relative to target (0 = target)")
ax.set_ylabel("Response time")
ax.set_title("RT vs feedback: model-implied curves with binned observed means (avg halves)")

# Legend (single, from the lineplot)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title="Group", frameon=True)

sns.despine(trim=True)
plt.tight_layout()
plt.show()
