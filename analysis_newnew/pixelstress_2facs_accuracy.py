# ============================================================
# Pixelstress ACCURACY analysis
# Sequence-level binomial GLM (cluster-robust by subject)
# + seaborn plot (model curve + binned means, SE across subjects)
# ============================================================

# Imports
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns

# Pandas settings (optional)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

sns.set_theme(style="whitegrid", context="paper")

# -----------------------------
# Paths + datasets
# -----------------------------
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
datasets = glob.glob(f"{path_in}/*erp.set")

# -----------------------------
# Load + align ERP/TF trial sets
# -----------------------------
dfs = []
for dataset in datasets:
    base = dataset.split("_cleaned")[0]

    df_erp = pd.read_csv(base + "_erp_trialinfo.csv")
    df_tf = pd.read_csv(base + "_tf_trialinfo.csv")

    # Keep trials present in both ERP and TF
    to_keep = np.intersect1d(
        df_erp["trial_nr_total"].values, df_tf["trial_nr_total"].values
    )

    # Reduce to common trials
    df = df_erp[df_erp["trial_nr_total"].isin(to_keep)].copy()

    # Binarize accuracy
    df["accuracy"] = (df["accuracy"] == 1).astype(int)

    # Group labels
    df = df.rename(columns={"session_condition": "group"})
    df["group"] = df["group"].replace({1: "experimental", 2: "control"})

    dfs.append(df)

df_trials = pd.concat(dfs, ignore_index=True)

# -----------------------------
# Exclusions + filters
# -----------------------------
ids_to_drop = [1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83]
df_trials = df_trials[~df_trials["id"].isin(ids_to_drop)].reset_index(drop=True)

# Remove first sequences (no feedback received yet)
df_trials = df_trials[df_trials["sequence_nr"] > 1].copy()

# -----------------------------
# Scale trial-level controls (applied before aggregation)
# -----------------------------
controls = ["trial_difficulty"]
scaler = StandardScaler()
df_trials[controls] = scaler.fit_transform(df_trials[controls])

# -----------------------------
# Aggregate to sequence level for accuracy
# -----------------------------
df_seq_acc = (
    df_trials.groupby(["id", "group", "block_nr", "sequence_nr"])
    .agg(
        n_trials=("accuracy", "size"),
        n_correct=("accuracy", "sum"),
        mean_trial_difficulty=("trial_difficulty", "mean"),
        mean_feedback=("last_feedback_scaled", "mean"),
    )
    .reset_index()
)

# Datatypes / reference levels
df_seq_acc["group"] = df_seq_acc["group"].astype("category")
df_seq_acc["group"] = df_seq_acc["group"].cat.reorder_categories(
    ["control", "experimental"], ordered=False
)
df_seq_acc["block_nr"] = df_seq_acc["block_nr"].astype(int)

# Predictors
df_seq_acc["f"] = df_seq_acc["mean_feedback"]
df_seq_acc["f2"] = df_seq_acc["f"] ** 2
df_seq_acc["half"] = np.where(df_seq_acc["block_nr"] <= 4, "first", "second")
df_seq_acc["half"] = df_seq_acc["half"].astype("category")


# Proportion accuracy for plotting
df_seq_acc["acc_rate"] = df_seq_acc["n_correct"] / df_seq_acc["n_trials"]

# Drop missings
df_seq_acc = df_seq_acc.dropna(
    subset=["n_trials", "n_correct", "mean_trial_difficulty", "f", "f2", "half"]
).copy()

# ============================================================
# Model: binomial GLM with cluster-robust SE by subject
# ============================================================

# Binomial GLM with frequency weights = number of trials per sequence
# (This effectively models the number of correct trials out of n_trials.)
formula_acc = """
acc_rate ~ group * f + group * f2
          + mean_trial_difficulty + half
"""

glm_acc = smf.glm(
    formula=formula_acc,
    data=df_seq_acc,
    family=sm.families.Binomial(),
    freq_weights=df_seq_acc["n_trials"],
)

res_acc = glm_acc.fit(cov_type="cluster", cov_kwds={"groups": df_seq_acc["id"]})
print(res_acc.summary())

# ============================================================
# Plot: model curve + binned means (SE across subjects)
# Averaged across halves (as in RT plot)
# ============================================================

df_plot = df_seq_acc.copy()
df_plot["id"] = df_plot["id"].astype("category")
df_plot["group"] = df_plot["group"].astype("category")
df_plot["half"] = df_plot["half"].astype("category")

# 1) Bin feedback for visualization (equal-count bins)
n_bins = 18
n_bins = min(n_bins, df_plot["f"].nunique())
df_plot["f_bin"] = pd.qcut(df_plot["f"], q=n_bins, duplicates="drop")

# Bin centers (numeric)
bin_centers = df_plot.groupby("f_bin", observed=True)["f"].mean()
df_plot["f_center"] = df_plot["f_bin"].map(bin_centers).astype(float)

# 2) Subject-level mean within (group × bin), then SE across subjects
subj_bin = (
    df_plot.groupby(["id", "group", "f_bin"], observed=True)
    .agg(f_center=("f_center", "mean"), acc_rate=("acc_rate", "mean"))
    .reset_index()
)

obs = (
    subj_bin.groupby(["group", "f_bin"], observed=True)
    .agg(
        f_center=("f_center", "mean"),
        mean_acc=("acc_rate", "mean"),
        sd_acc=("acc_rate", "std"),
        n_subjects=("id", "nunique"),
    )
    .reset_index()
)
obs["se_acc"] = obs["sd_acc"] / np.sqrt(obs["n_subjects"])

# 3) Model-implied curves on a grid, averaged across halves
f_grid = np.linspace(df_plot["f"].min(), df_plot["f"].max(), 200)
difficulty_ref = df_plot["mean_trial_difficulty"].mean()
half_levels = list(df_plot["half"].cat.categories)
group_levels = list(df_plot["group"].cat.categories)

pred_rows = []
for g in group_levels:
    preds_half = []
    for h in half_levels:
        tmp = pd.DataFrame(
            {
                "f": f_grid,
                "f2": f_grid**2,
                "group": g,
                "half": h,
                "mean_trial_difficulty": difficulty_ref,
            }
        )
        preds_half.append(res_acc.predict(tmp).to_numpy())

    pred_rows.append(
        pd.DataFrame(
            {
                "f": f_grid,
                "group": g,
                "pred_acc": np.mean(np.vstack(preds_half), axis=0),
            }
        )
    )

pred = pd.concat(pred_rows, ignore_index=True)

# 4) Plot
fig, ax = plt.subplots(figsize=(8.5, 5))

palette = sns.color_palette(n_colors=len(group_levels))
color_map = dict(zip(group_levels, palette))

sns.lineplot(
    data=pred, x="f", y="pred_acc", hue="group", palette=color_map, linewidth=2.5, ax=ax
)

sns.scatterplot(
    data=obs,
    x="f_center",
    y="mean_acc",
    hue="group",
    palette=color_map,
    s=60,
    edgecolor="none",
    ax=ax,
    legend=False,
)

for g, sub in obs.groupby("group", observed=True):
    ax.errorbar(
        sub["f_center"].to_numpy(),
        sub["mean_acc"].to_numpy(),
        yerr=sub["se_acc"].to_numpy(),
        fmt="none",
        ecolor=color_map[g],
        elinewidth=1.2,
        capsize=3,
        alpha=0.7,
    )

ax.axvline(0, linestyle="--", color="black", linewidth=1)
ax.set_xlabel("Feedback relative to target (0 = target)")
ax.set_ylabel("Accuracy (proportion correct)")
ax.set_ylim(0, 1)
ax.set_title("Accuracy vs feedback: model curve + binned means (SE across subjects)")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title="Group", frameon=True)

# Set ylims
ymin = (obs["mean_acc"] - obs["se_acc"]).min()
ymax = (obs["mean_acc"] + obs["se_acc"]).max()
pad = 0.02  # 2 percentage points padding
ax.set_ylim(
    max(0, ymin - pad),
    min(1, ymax + pad)
)

sns.despine(trim=True)
plt.tight_layout()
plt.show()
