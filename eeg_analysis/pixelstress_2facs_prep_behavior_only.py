# Imports
import mne
import glob
import os
import pandas as pd
import numpy as np
import scipy.io
from joblib import dump
import sklearn.linear_model
from sklearn.preprocessing import MinMaxScaler
import pingouin as pg

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

# Reduce info
cols_to_keep = ["id", "group", "sequence_nr", "rt", "accuracy", "trajectory"]
df_trials = df_trials[cols_to_keep]

# IDs to exclude
#ids_to_drop = [1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 32, 40, 48, 49, 50, 52, 83, 88]
#df_trials = df_trials[~df_trials["id"].isin(ids_to_drop)].reset_index()

# Drop first sequences
results = []
for seq_to_drop in range(7):
    
    df_trials = df_trials[(df_trials.sequence_nr > seq_to_drop).values]
    
    # Group dataframe
    group_cols = ["id", "group", "trajectory"]
    
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
        dv="mean_rt", between="group", within="trajectory", subject="id", data=df_summary
    )
    
    # Get pvals
    pvals_df = (
        aov.set_index('Source')['p-unc']
           .loc[['group', 'trajectory', 'Interaction']]
           .rename({
               'group': 'p_group',
               'trajectory': 'p_trajectory',
               'Interaction': 'p_interaction'
           })
           .to_frame()
           .T
    )
    pvals_df['seq_pos_excluded'] = seq_to_drop
    results.append(pvals_df)

# combine into one dataframe
pvals_df = pd.concat(results, ignore_index=True)
