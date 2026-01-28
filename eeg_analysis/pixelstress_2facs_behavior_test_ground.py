# Imports
import glob
import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pingouin as pg
import statsmodels.formula.api as smf
import seaborn as sns

# Define paths
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_out = "/mnt/data_dump/pixelstress/3_2fac_data/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp.set")


# Function for ez-diffusion
def ez_diffusion(MRT, VRT, Pc, n_trials, s=0.1):
    """
    Translates the R function get.vaTer to Python.

    Parameters:
    - Pc: Proportion correct (0 < Pc < 1)
    - VRT: Variance of RT (typically correct trials only)
    - MRT: Mean RT (in seconds, correct trials only)
    - s: Scaling parameter for DDM noise (default: 0.1)

    Returns:
    - v: Drift rate
    - a: Boundary separation
    - Ter: Non-decision time
    """

    # Replace Pc if accuracy is 1
    if Pc == 1:
        Pc = 1 - (0.5 / n_trials)

    s2 = s**2

    # Basic validation
    if Pc <= 0 or Pc == 0.5 or Pc >= 1:
        raise ValueError(
            "Pc must be between 0 and 1 and not equal to 0.5 for this method to work."
        )

    # Logit of Pc
    L = scipy.special.logit(Pc)  # same as qlogis in R

    # Compute intermediate term
    x = L * (L * Pc**2 - L * Pc + Pc - 0.5) / VRT

    # Drift rate
    v = np.sign(Pc - 0.5) * s * (x**0.25)

    # Boundary separation
    a = s2 * L / v

    # y = -v * a / s²
    y = -v * a / s2

    # Mean decision time
    MDT = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))

    # Non-decision time
    Ter = MRT - MDT

    return a, v, Ter


# Collect datasets
dfs = []
for dataset in datasets:

    # Read erp trialinfo
    df_erp = pd.read_csv(dataset.split("_cleaned")[0] + "_erp_trialinfo.csv")

    # Read tf trialinfo
    df_tf = pd.read_csv(dataset.split("_cleaned")[0] + "_tf_trialinfo.csv")

    # Get common trials
    to_keep = np.intersect1d(df_erp.trial_nr_total.values, df_tf.trial_nr_total.values)

    # Get df of common trials
    df = df_erp[df_erp["trial_nr_total"].isin(to_keep)]

    # Set accuracy of non-correct to 0
    df.loc[df["accuracy"] != 1, "accuracy"] = 0

    # Rename group column
    df.rename(columns={"session_condition": "group"}, inplace=True)
    df["group"] = df["group"].replace({1: "experimental", 2: "control"})

    # Add variable trajectory
    df = df.assign(trajectory="close")
    df.loc[(df.block_wiggleroom == 1) & (df.block_outcome == -1), "trajectory"] = (
        "below"
    )
    df.loc[(df.block_wiggleroom == 1) & (df.block_outcome == 1), "trajectory"] = "above"
    

    dfs.append(df)

# combine into one dataframe
df_trials = pd.concat(dfs, ignore_index=True)

# IDs to exclude
ids_to_drop = [1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83, 32, 48]
df_trials = df_trials[~df_trials["id"].isin(ids_to_drop)].reset_index()

# Remove first sequences
df_trials = df_trials[df_trials.sequence_nr > 1]

# Scale control variables
controls = [
    "trial_difficulty",
    "trial_nr_total",
]
scaler = StandardScaler()
df_trials[controls] = scaler.fit_transform(df_trials[controls])

# keep only correct trials
df_rt = df_trials[df_trials["accuracy"] == 1].copy()

# make sure these are categorical
df_rt["id"] = df_rt["id"].astype("category")
df_rt["group"] = df_rt["group"].astype("category")

# Columns to average
avg_cols = ["rt", "trial_difficulty", "last_feedback_scaled", "trial_nr_total"]

# Aggregate
df_seq_rt = df_rt.groupby(["id", "block_nr", "sequence_nr"]).agg(
    mean_rt=("rt", "mean"),
    mean_trial_difficulty=("trial_difficulty", "mean"),
    mean_feedback=("last_feedback_scaled", "mean"),
    mean_trial_nr_total=("trial_nr_total", "mean")
).reset_index()

# safe log-transform after aggregation
df_seq_rt["mean_log_rt"] = np.log(df_seq_rt["mean_rt"])

# Drop nans
df_seq_rt = df_seq_rt.dropna(subset=[
    "mean_log_rt", 
    "mean_trial_difficulty", 
    "mean_feedback", 
    "mean_trial_nr_total"
])

# Create mapping from id → group
id_to_group = df_rt.set_index("id")["group"].to_dict()

# Add group column
df_seq_rt["group"] = df_seq_rt["id"].map(id_to_group)

# Create quadratic feedback term
df_seq_rt["mean_feedback_sq"] = df_seq_rt["mean_feedback"]**2

# Model formula
formula_seq = """
mean_log_rt ~ group * mean_feedback + group * mean_feedback_sq
               + mean_trial_difficulty + mean_trial_nr_total
"""

# Build model
model_seq = smf.mixedlm(
    formula_seq,
    df_seq_rt,
    groups=df_seq_rt["id"],  # random intercept per subject
    re_formula="1"
)

# Fit model
result_seq = model_seq.fit(method="lbfgs")

# Print results
print(result_seq.summary())





aa=bb











# predicted RT
df_rt["pred_rt"] = np.exp(result.fittedvalues)

# create 5–10 bins of feedback
df_rt["feedback_bin"] = pd.qcut(df_rt["last_feedback_scaled"], q=15)

# compute mean predicted RT per bin and group
agg = df_rt.groupby(["group", "feedback_bin"]).agg(
    mean_pred_rt = ("pred_rt", "mean"),
    sem_pred_rt = ("pred_rt", lambda x: x.std()/np.sqrt(len(x)))
).reset_index()

# get bin centers for plotting
agg["bin_center"] = agg["feedback_bin"].apply(lambda x: x.mid)

plt.figure(figsize=(7,5))
sns.lineplot(data=agg, x="bin_center", y="mean_pred_rt", hue="group")
# add error bars for SEM
for g in agg["group"].unique():
    gdata = agg[agg["group"]==g]
    plt.errorbar(
        gdata["bin_center"], gdata["mean_pred_rt"], 
        yerr=gdata["sem_pred_rt"], fmt='o', capsize=3
    )

plt.xlabel("Scaled Feedback")
plt.ylabel("Predicted RT (ms)")
plt.title("Predicted RT by Feedback × Group (binned)")
plt.show()



aa=bb



#ids_to_drop = [32, 48, 50, 52, 88]

# Drop first sequences
results = []
within_factor = "trajectory"
for seq_to_drop in [1, 2, 3, 4, 5]:
    
    df_trials = df_trials[(df_trials.sequence_nr > seq_to_drop).values]
    
    # Group dataframe
    group_cols = ["id", "group", within_factor]
    
    # RT stats: only correct trials
    rt_df = (
        df_trials[df_trials["accuracy"] == 1]
        .groupby(group_cols, as_index=False)
        .agg(
            mean_rt=("rt", "mean"),
            var_rt=("rt", "var"),
            n_trials=("rt", "size"),  # number of correct trials
        )
    )
    
    # Accuracy: proportion correct
    acc_df = df_trials.groupby(group_cols, as_index=False).agg(acc=("accuracy", "mean"))
    
    # Merge
    df_summary = pd.merge(rt_df, acc_df, on=group_cols, how="outer")
    
    # Calculate EZ-diffusion parameters ========================================================================================================
    df_summary["drift_rate"] = 0
    df_summary["boundary_seperation"] = 0
    df_summary["non_decision_time"] = 0
    
    for idx, row in df_summary.iterrows():
    
        a, v, t0 = ez_diffusion(
            row["mean_rt"], row["var_rt"] ** 2, row["acc"], row["n_trials"]
        )
        df_summary.at[idx, "drift_rate"] = v
        df_summary.at[idx, "boundary_seperation"] = a
        df_summary.at[idx, "non_decision_time"] = t0
    
    # Run mixed ANOVA
    aov = pg.mixed_anova(
        dv="mean_rt", between="group", within=within_factor, subject="id", data=df_summary
    )
    
    # Get pvals
    pvals_df = (
        aov.set_index('Source')['p-unc']
           .loc[['group', within_factor, 'Interaction']]
           .rename({
               'group': 'p_group',
               within_factor: 'p_' + within_factor,
               'Interaction': 'p_interaction'
           })
           .to_frame()
           .T
    )
    pvals_df['seq_pos_excluded'] = seq_to_drop
    results.append(pvals_df)

# combine into one dataframe
pvals_df = pd.concat(results, ignore_index=True)
