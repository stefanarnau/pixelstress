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
path_df = "/mnt/data_dump/pixelstress/dataframe/"

# Define datasets
datasets = glob.glob(f"{path_in}/*erp.set")

# Load channel labels
channel_labels = open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r").read().split("\n")[:-1]

# Create info
info = mne.create_info(channel_labels, 200, ch_types='eeg', verbose=None)

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
    erp_times = scipy.io.loadmat(dataset)["times"].ravel()

    # Read tf trialinfo
    df_tf = pd.read_csv(dataset.split("_cleaned")[0] + "_tf_trialinfo.csv")

    # Load tf eeg data
    tf_data = np.transpose(
        scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")["data"], [2, 0, 1]
    )

    # Load tf times
    tf_times = scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")["times"].ravel()
    
    # Drop first sequences
    idx_not_first_sequences = (df_erp.sequence_nr != 1).values
    df_erp = df_erp[idx_not_first_sequences]
    erp_data = erp_data[idx_not_first_sequences, :, :]
    idx_not_first_sequences = (df_tf.sequence_nr != 1).values
    df_tf = df_tf[idx_not_first_sequences]
    tf_data = tf_data[idx_not_first_sequences, :, :]
    
    # Add new variable sequence number total
    df_erp = df_erp.assign(sequence_nr_total="none")
    df_erp.sequence_nr_total = df_erp.sequence_nr + (df_erp.block_nr - 1) * 12
    df_tf = df_tf.assign(sequence_nr_total="none")
    df_tf.sequence_nr_total = df_tf.sequence_nr + (df_tf.block_nr - 1) * 12

    # Prepare behavior based on TF trialinfo =============================================================================

    # Remove trial difficulty confound using linear regression
    X = df_tf[["trial_difficulty"]].values
    y = df_tf["rt"].values
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    df_tf["rt_detrended"] = y - y_pred

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

    # Create erp columns
    df_out["erp_F"] = None
    df_out["erp_FC"] = None
    df_out["erp_C"] = None
    
    # Collector lists
    erps_F = []
    erps_FC = []
    erps_C = []

    # Iterate sequences and average erps
    for row_idx, seq_total in enumerate(df_out["sequence_nr_total"].values):
        
        # Get sequence indices in erp data
        seq_idx = df_erp["sequence_nr_total"].values == seq_total
        
        # If no trial in erp data for sequence remains...
        if sum(seq_idx) == 0:
            
            erps_F.append(np.nan)
            erps_FC.append(np.nan)
            erps_C.append(np.nan)
            continue
        
        # Get sequence erps
        erp_seq = erp_data[seq_idx, :, :].mean(axis=0)

        # Get midline erps
        erps_F.append(erp_seq[[4, 37, 38], :].mean(axis=0))
        erps_FC.append(erp_seq[[64, 8, 9], :].mean(axis=0))
        erps_C.append(erp_seq[[13, 47, 48], :].mean(axis=0))
        
    df_out["erp_F"] = erps_F
    df_out["erp_FC"] = erps_FC
    df_out["erp_C"] = erps_C

    # Select time window for cnv analysis
    cnv_times = (erp_times >= -600) & (erp_times <= 0)
    
    # Collector lists
    values_F = []
    values_FC = []
    values_C = []
    
    # Iterate rows
    for i in range(len(df_out)):
        
        # get row
        row = df_out.iloc[i]
        
        # Skip missing
        if np.isnan(erps_F[i]).any():
            values_F.append(np.nan)
            values_FC.append(np.nan)
            values_C.append(np.nan)
            continue
        
        # Collect parameterized CNV
        values_F.append(row["erp_F"][cnv_times].mean())
        values_FC.append(row["erp_FC"][cnv_times].mean())
        values_C.append(row["erp_C"][cnv_times].mean())

    # Add cnv values to df
    df_out = df_out.assign(cnv_F=values_F)
    df_out = df_out.assign(cnv_FC=values_FC)
    df_out = df_out.assign(cnv_C=values_C)

    # Create epochs object
    tf_data = mne.EpochsArray(tf_data, info)

    # Get baseline indices
    idx_bl = (tf_times >= -2200) & (tf_times <= -1700)

    # Time-frequency parameters
    n_freqs = 30
    tf_freqs = np.linspace(4, 15, n_freqs)
    tf_cycles = np.linspace(6, 10, n_freqs)

    # tf-decomposition of all data (data is trial channels frequencies times)
    ersp_all = mne.time_frequency.tfr_morlet(
        tf_data,
        tf_freqs,
        n_cycles=tf_cycles,
        average=False,
        return_itc=False,
        n_jobs=-2,
    )

    # Get average baseline values
    bl_values = ersp_all._data[:, :, :, idx_bl].mean(axis=3).mean(axis=0)
    
    # Create erp columns
    df_out["frontal_theta"] = None
    df_out["posterior_alpha"] = None
    
    # Collector lists
    fr_theta = []
    pos_alpha = []
    
    # Iterate sequences and get tf measures
    for row_idx, seq_total in enumerate(df_out["sequence_nr_total"].values):
        
        # Get sequence indices in tf data
        seq_idx = df_tf["sequence_nr_total"].values == seq_total

        # Get sequence epochs
        seq_tf_data = ersp_all._data[seq_idx, :, :, :].mean(axis=0)

        # Apply condition general dB baseline
        for ch in range(bl_values.shape[0]):
            for fr in range(bl_values.shape[1]):
                seq_tf_data[ch, fr, :] = 10 * np.log10(
                    seq_tf_data[ch, fr, :].copy() / bl_values[ch, fr]
                )
                
        # Get frontal theta
        tmp = seq_tf_data[:, (tf_freqs >= 4) & (tf_freqs <= 7), :].mean(axis=1)
        fr_theta.append(tmp[[4, 64, 8, 9, 37, 38], :].mean(axis=0))
        
        # Get posterior alpha
        tmp = seq_tf_data[:, (tf_freqs >= 8) & (tf_freqs <= 12), :].mean(axis=1)
        pos_alpha.append(tmp[[61, 59, 63], :].mean(axis=0))
      
    df_out["frontal_theta"] = fr_theta
    df_out["posterior_alpha"] = pos_alpha

    # Select time window for tf analysis
    theta_times = (tf_times >= -1200) & (tf_times <= -200)
    alpha_times = (tf_times >= -1200) & (tf_times <= -200)
    
    # Collector lists
    values_theta = []
    values_alpha = []
    
    # Iterate rows
    for i in range(len(df_out)):
        
        # get row
        row = df_out.iloc[i]
        
        # Collect parameterized CNV
        values_theta.append(row["frontal_theta"][theta_times].mean())
        values_alpha.append(row["posterior_alpha"][alpha_times].mean())

    # Add cnv values to df
    df_out = df_out.assign(frontal_theta_value=values_theta)
    df_out = df_out.assign(posterior_alpha_value=values_alpha)

    # Collect
    df_sequences.append(df_out)

# Concatenate datasets
df_sequences = pd.concat(df_sequences).reset_index()

# Make categorial
df_sequences["trajectory"] = pd.Categorical(df_sequences["trajectory"])
df_sequences["session_condition"] = pd.Categorical(df_sequences["session_condition"])

# Save dataframe
fn = "dataframe_pixelstress.pkl"
df_sequences.to_pickle(os.path.join(path_df, fn))
