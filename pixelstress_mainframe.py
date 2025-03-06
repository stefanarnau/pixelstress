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
datasets = glob.glob(f"{path_in}/*erp.set")

# Collector bin for all trials
df_sequences = []

# Collect datasets
for dataset in datasets:

    # Load stuff =========================================================================================================

    # Read erp trialinfo
    df_erp = pd.read_csv(dataset.split("_cleaned")[0] + "_erp_trialinfo.csv")

    # Load erp data as trials x channel x times
    erp_data = np.transpose(scipy.io.loadmat(dataset)["data"], [2, 0, 1])

    # Get erp times
    erp_times = scipy.io.loadmat(dataset)["times"]

    # Read tf trialinfo
    df_tf = pd.read_csv(dataset.split("_cleaned")[0] + "_tf_trialinfo.csv")

    # Load tf eeg data
    tf_data = np.transpose(
        scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")["data"], [2, 0, 1]
    )

    # Load tf times
    tf_times = scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")["times"]

    # Prepare behavior based on TF trialinfo =============================================================================

    # Drop first sequences
    idx_not_first_sequences = (df_tf.sequence_nr != 1).values
    df_tf = df_tf[idx_not_first_sequences]
    tf_data = tf_data[idx_not_first_sequences, :, :]

    # Remove trial difficulty confound using linear regression
    X = df_tf[["trial_difficulty"]].values
    y = df_tf["rt"].values
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    df_tf["rt_detrended"] = y - y_pred

    # Add new variable sequence number total
    df_tf = df_tf.assign(sequence_nr_total="none")
    df_tf.sequence_nr_total = df_tf.sequence_nr + (df_tf.block_nr - 1) * 12

    # Exclude trials belonging to sequences from which less than 5 trials are left
    trial_counts = df_tf["sequence_nr_total"].value_counts()
    
    # Count trials in each sequence
    df_tf['seq_trial_n_tf'] = df_tf.groupby('sequence_nr_total')['sequence_nr_total'].transform('count')
    
    # Create df for correct only
    df_tf_correct_only = df_tf.drop(df_tf[df_tf.accuracy != 1].index)

    # Group variables by sequences and get rt for conditions
    df_out = (
        df_tf_correct_only.groupby(["sequence_nr_total"])[
            "id",
            "session_condition",
            "block_wiggleroom",
            "block_outcome",
            "last_feedback",
            "last_feedback_scaled",
            "block_nr",
            "sequence_nr",
            "seq_trial_n_tf",
            "rt",
            "rt_detrended",
        ]
        .mean()
        .reset_index()
    )

    # Iterate sequences and calculate accuracy
    df_out = df_out.assign(accuracy=99)
    for row_idx, seq_total in enumerate(df_out["sequence_nr_total"].values):

        # Get indices of toal sequence number
        idx_seqtotal = df_tf["sequence_nr_total"].values == seq_total

        # get accuracy
        df_out.loc[row_idx, "accuracy"] = sum(
            df_tf["accuracy"].values[idx_seqtotal] == 1
        ) / sum(idx_seqtotal)

    # Add new variable trajectory
    df_out = df_out.assign(trajectory="close")
    df_out.trajectory[(df_out.block_wiggleroom == 1) & (df_out.block_outcome == -1)] = (
        "below"
    )
    df_out.trajectory[(df_out.block_wiggleroom == 1) & (df_out.block_outcome == 1)] = (
        "above"
    )

    # Iterate all sequences
    erps_F = []
    erps_FC = []
    erps_C = []

    # Iterate sequences and average erps
    for seq in set(df_out["sequence_nr_total"].values):

        # Get sequence erps
        eeg_seq = eeg_data[df_out["sequence_nr_total"].values == seq, :, :].mean(axis=0)

        # Get midline erps
        erps_F.append(eeg_data[idx_seq, [4, 37, 38], :].mean(axis=0))
        erps_FC.append(eeg_data[idx_seq, [64, 8, 9], :].mean(axis=0))
        erps_C.append(eeg_data[idx_seq, [13, 47, 48], :].mean(axis=0))

    # Save erps to dataframe
    df_tmp_ave = df_tmp_ave.assign(erp_F=pd.Series(erps_F))
    df_tmp_ave = df_tmp_ave.assign(erp_FC=pd.Series(erps_FC))
    df_tmp_ave = df_tmp_ave.assign(erp_C=pd.Series(erps_C))

    # Select time window and calculate cnv values
    time_idx = (erp_times >= -600) & (erp_times <= 0)
    values_F = []
    values_FC = []
    values_C = []
    for i in range(len(df_tmp_ave)):
        row = df_tmp_ave.iloc[i]
        values_F.append(row["erp_F"][time_idx].mean())
        values_FC.append(row["erp_FC"][time_idx].mean())
        values_C.append(row["erp_C"][time_idx].mean())

    # Add cnv values to df
    df_tmp_ave = df_tmp_ave.assign(cnv_F=values_F)
    df_tmp_ave = df_tmp_ave.assign(cnv_FC=values_FC)
    df_tmp_ave = df_tmp_ave.assign(cnv_C=values_C)

    aa=bb

    # Collect
    df_sequences.append(df_tmp_ave)

# Concatenate datasets
df_sequences = pd.concat(df_sequences).reset_index()

# Make categorial
df_sequences["trajectory"] = pd.Categorical(df_sequences["trajectory"])
df_sequences["session_condition"] = pd.Categorical(df_sequences["session_condition"])
