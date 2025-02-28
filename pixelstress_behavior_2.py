#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 14:52:54 2021

@author: Stefan Arnau
"""

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
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
path_plot = "/mnt/data_dump/pixelstress/plots/"
path_stats = "/mnt/data_dump/pixelstress/stats/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp_trialinfo.csv")

# Collector bin for all trials
df_sequences = []
df_single_trial = []

# Collect datasets
for dataset in datasets:

    # Read data
    df_tmp = pd.read_csv(dataset)

    # Drop first sequences
    df_tmp = df_tmp.drop(df_tmp[df_tmp.sequence_nr <= 1].index)
    
    # Drop outliers
    z_scores = np.abs((df_tmp['rt'] - df_tmp['rt'].mean()) / df_tmp['rt'].std())
    df_tmp = df_tmp[z_scores < 2]
    
    # Assuming your DataFrame is called 'df'
    X = df_tmp[['trial_difficulty']].values
    y = df_tmp['rt'].values
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    df_tmp['rt_detrended'] = y - y_pred

    # Add new variable sequence number total
    df_tmp = df_tmp.assign(sequence_nr_total="none")
    df_tmp.sequence_nr_total = df_tmp.sequence_nr + (df_tmp.block_nr - 1) * 12

    # Create df for correct only
    df_tmp_correct_only = df_tmp.drop(df_tmp[df_tmp.accuracy != 1].index)

    # Add new variable trajectory
    df_tmp_correct_only = df_tmp_correct_only.assign(trajectory=99)
    df_tmp_correct_only.trajectory[df_tmp_correct_only.block_wiggleroom == 0] = 0
    df_tmp_correct_only.trajectory[
        (df_tmp_correct_only.block_wiggleroom == 1)
        & (df_tmp_correct_only.block_outcome == -1)
    ] = -1
    df_tmp_correct_only.trajectory[
        (df_tmp_correct_only.block_wiggleroom == 1)
        & (df_tmp_correct_only.block_outcome == 1)
    ] = +1

    # Group variables by sequences and get rt for conditions 
    df_tmp_ave = (
        df_tmp_correct_only.groupby(["sequence_nr_total"])[
            "id",
            "session_condition",
            "block_wiggleroom",
            "block_outcome",
            "trajectory",
            "last_feedback",
            "last_feedback_scaled",
            "block_nr",
            "sequence_nr",
            "rt",
            "rt_detrended",
        ]
        .mean()
        .reset_index()
    )

    # Get accuracy
    series_n_all = (
        df_tmp.groupby(["sequence_nr_total"]).size().reset_index(name="acc")["acc"]
    )
    series_n_correct = (
        df_tmp_correct_only.groupby(["sequence_nr_total"])
        .size()
        .reset_index(name="acc")["acc"]
    )
    series_accuracy = series_n_correct / series_n_all

    # Compute inverse efficiency
    series_ie = df_tmp_ave["rt"] / series_accuracy

    # Combine
    df_tmp_ave["acc"] = series_accuracy
    df_tmp_ave["ie"] = series_ie

    # Collect
    df_single_trial.append(df_tmp_ave)
    df_sequences.append(df_tmp_ave)

# Concatenate datasets
df_sequences = pd.concat(df_sequences).reset_index()


model = smf.mixedlm("rt_detrended ~ last_feedback_scaled*sequence_nr*session_condition", 
                    data=df, 
                    groups="id")


results = model.fit()

# Extract the fixed effects results
fe_results = results.fe_params
se = results.bse_fe

# Calculate t-statistics
t_stats = fe_results / se

# Calculate p-values
p_values = [2 * (1 - scipy.stats.t.cdf(abs(t), df=results.df_resid)) for t in t_stats]

# Create a DataFrame with coefficients and p-values
coef_df = pd.DataFrame(
    {
        "coef": fe_results,
        "std err": se,
        "t": t_stats,
        "P>|t|": p_values,
        #"[0.025": results.conf_int()[0],
        #"0.975]": results.conf_int()[1],
    }
)

jitter = np.random.normal(0, 0.3, size=len(df))
g = sns.relplot(
    data=df,
    x=df["sequence_nr"] + jitter,
    y="rt_detrended",
    hue="last_feedback_scaled",
    col="session_condition",
    kind="scatter",
    palette="viridis",
    height=5,
    aspect=1.2,
    col_wrap=2,
)

# Add a color bar
plt.colorbar(g.axes[0].collections[0], label="rt", ax=g.axes)

# Adjust labels and title
g.set_axis_labels("Continuous Variable 1", "Continuous Variable 2")
g.fig.suptitle("Relationship between Variables", y=1.02)

plt.show()

aa = bb

# Create df for correct only
df_correct_only = df.drop(df[df.accuracy != 1].index)

# Get rt for conditions
df_rt = (
    df_correct_only.groupby(["id", "trajectory"])["rt"].mean().reset_index(name="ms")
)

# Get accuracy for conditions
series_n_all = df.groupby(["id", "trajectory"]).size().reset_index(name="acc")["acc"]
series_n_correct = (
    df_correct_only.groupby(["id", "trajectory"]).size().reset_index(name="acc")["acc"]
)
series_accuracy = series_n_correct / series_n_all

# Get session condition for conditions
series_session = (
    df.groupby(["id", "trajectory"])["session_condition"]
    .mean()
    .reset_index(name="session")["session"]
)

# Compute inverse efficiency
series_ie = df_rt["ms"] / series_accuracy

# Combine
df_rt["acc"] = series_accuracy
df_rt["group"] = series_session
df_rt["ie"] = series_ie

# Rename group vars
df_rt.group[(df_rt.group == 1)] = "experimental"
df_rt.group[(df_rt.group == 2)] = "control"

# Make vars categorial
df_rt["group"] = df_rt["group"].astype("category")
df_rt["trajectory"] = df_rt["trajectory"].astype("category")


aov = pg.mixed_anova(
    data=df_rt, dv="ms", between="group", within="trajectory", subject="id"
)
print(aov)


# Plot
g = sns.catplot(
    data=df_rt,
    x="trajectory",
    y="ms",
    hue="group",
    palette="rocket",
    errorbar="se",
    kind="violin",
    height=6,
    aspect=0.75,
)
g.despine(left=True)


g = sns.catplot(
    data=df_rt,
    x="trajectory",
    y="acc",
    hue="group",
    capsize=0.2,
    palette="rocket",
    errorbar="se",
    kind="line",
    height=6,
    aspect=0.75,
)
g.despine(left=True)

g = sns.catplot(
    data=df_rt,
    x="trajectory",
    y="ie",
    hue="group",
    capsize=0.2,
    palette="rocket",
    errorbar="se",
    kind="line",
    height=6,
    aspect=0.75,
)
g.despine(left=True)


# Save dataframe
# df_rt.to_csv(os.path.join(path_stats, "behavioral data.csv"))
