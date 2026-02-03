# Imports
import glob
import numpy as np
import pandas as pd
import mne
import scipy.io
import scipy.stats
from sklearn.preprocessing import StandardScaler
import mne.stats
import matplotlib.pyplot as plt
import seaborn as sns

# Path things
path_in = "/mnt/data_dump/pixelstress/2_autocleaned/"
datasets = glob.glob(f"{path_in}/*erp.set")

ids_to_drop = {1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83}
min_trials_per_sequence = 3

# Load channel labels + create info + set montage + calculate adjacency
channel_labels = (
    open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r")
    .read()
    .split("\n")[:-1]
)
sfreq = 200.0
info_tf = mne.create_info(channel_labels, sfreq, ch_types="eeg", verbose=None)
montage = mne.channels.make_standard_montage("standard_1020")  # or "standard_1005"
info_tf.set_montage(montage, on_missing="warn", match_case=False)
adjacency, ch_names = mne.channels.find_ch_adjacency(info_tf, ch_type="eeg")


# Helper: build design matrix (within subject)
def build_design_matrix(df_seq: pd.DataFrame):
    X = df_seq[["f", "f2", "mean_trial_difficulty", "half"]].copy()
    X["half"] = (X["half"].astype(str) == "second").astype(int)
    X = X.astype(float)

    Xz = StandardScaler().fit_transform(X)  # within-subject z-scoring
    Xz = np.column_stack([np.ones(len(Xz)), Xz])  # intercept
    names = ["Intercept", "f", "f2", "difficulty", "half"]
    return Xz, names


# Helper to build X with correct shape
def stack_betas(beta_maps_tf, term):
    X = np.stack(
        [d["betas"][term] for d in beta_maps_tf], axis=0
    )  # (subj, ch, freq, time)
    X = np.transpose(X, (0, 3, 2, 1))  # -> (subj, time, freq, ch)
    return X


# Collectors
beta_maps_tf = []

# Loop datasets
for dataset in datasets:
    base = dataset.split("_cleaned")[0]

    # Trialinfo
    df_erp = pd.read_csv(base + "_erp_trialinfo.csv")
    df_tf = pd.read_csv(base + "_tf_trialinfo.csv")

    subj_id = int(df_erp["id"].iloc[0])
    if subj_id in ids_to_drop:
        continue

    # Load tf eeg data as trials x channles x times
    tf_data = np.transpose(
        scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")["data"], [2, 0, 1]
    )

    # Load tf times
    tf_times = scipy.io.loadmat(dataset.split("_erp.set")[0] + "_tf.set")[
        "times"
    ].ravel()

    # Determine time units for tmin (heuristic: EEGLAB typically ms)
    # If times look like [-1000..2000], that's ms; if [-1..2], that's seconds.
    if np.nanmax(np.abs(tf_times)) > 20:
        tmin = tf_times[0] / 1000.0
    else:
        tmin = tf_times[0]

    # Common trials between ERP and TF
    to_keep = np.intersect1d(
        df_erp["trial_nr_total"].values, df_tf["trial_nr_total"].values
    )

    # Reduce metadata to common trials (copy)
    df = df_tf[df_tf["trial_nr_total"].isin(to_keep)].copy()

    # Reduce TF data to those common trials
    mask_common = np.isin(df_tf["trial_nr_total"].values, to_keep)
    tf_data = tf_data[mask_common, :, :]

    # Basic checks
    if tf_data.shape[0] != len(df):
        raise RuntimeError(
            f"Trial alignment mismatch for subject {subj_id}: "
            f"tf_data {tf_data.shape[0]} vs df {len(df)}"
        )

    # Binarize accuracy
    df["accuracy"] = (df["accuracy"] == 1).astype(int)

    # Group coding
    df = df.rename(columns={"session_condition": "group"})
    df["group"] = df["group"].replace({1: "experimental", 2: "control"})

    # Remove first sequences
    mask = df["sequence_nr"] > 1
    df = df.loc[mask].reset_index(drop=True)
    tf_data = tf_data[mask.to_numpy(), :, :]

    # Keep only correct trials
    mask = df["accuracy"] == 1
    df = df.loc[mask].reset_index(drop=True)
    tf_data = tf_data[mask.to_numpy(), :, :]

    if len(df) == 0:
        continue

    # Se tf-decomposition parameters
    freqs = np.arange(4, 31, 2)
    n_cycles = freqs / 2.0

    # Create epochs object
    epochs = mne.EpochsArray(tf_data, info_tf, tmin=tmin, verbose=False)
    epochs.metadata = df

    # TF decomposition
    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
        output="power",
        decim=2,
        n_jobs=-2,
        verbose=False,
    )
    
    # power.data: (n_trials, n_ch, n_freq, n_time)
    P = power.data
    
    bmask = (power.times >= -1.8) & (power.times <= -1.5)
    
    # baseline reference per subject, per ch×freq
    B = P[..., bmask].mean(axis=(0, 3))  # (n_ch, n_freq)
    
    # dB conversion per trial
    power.data = 10.0 * np.log10((P + 1e-12) / (B[None, :, :, None] + 1e-12))  # (trials, ch, freq, time)
    
    # Crpo in time
    power.crop(tmin=-1.8, tmax=1.0)
    
    # Times...
    times = power.times
    
    # Create info with effective sfreq from time vector
    sfreq_eff = 1.0 / np.mean(np.diff(times))
    info_tf_eff = mne.create_info(channel_labels, sfreq_eff, ch_types="eeg", verbose=None)
    tmin_eff = float(times[0])
    info_tf_eff.set_montage(montage, on_missing="warn", match_case=False)
    
    # -----------------------------
    # Sequence averaging: (block_nr, sequence_nr)
    # -----------------------------
    df = df.reset_index(drop=True)
    g = df.groupby(["block_nr", "sequence_nr"], sort=True)

    seq_tf = []
    seq_meta = []

    for (block_nr, seq_nr), idx in g.indices.items():
        idx = np.asarray(idx)
        ntr = len(idx)

        if ntr < min_trials_per_sequence:
            continue

        # Average TF power across trials in this sequence
        # pow_data: (n_trials, n_ch, n_freq, n_time)
        seq_avg = power.data[idx].mean(axis=0)  # (n_ch, n_freq, n_time)
        seq_tf.append(seq_avg)

        df_sub = df.loc[idx]

        # Feedback constant within sequence (take first)
        f = float(df_sub["last_feedback_scaled"].iloc[0])

        # Difficulty may vary; average within sequence
        mean_difficulty = float(
            pd.to_numeric(df_sub["trial_difficulty"], errors="coerce").mean()
        )

        # Half from block
        half = "first" if int(block_nr) <= 4 else "second"

        # Group constant within subject
        group = df_sub["group"].iloc[0]

        seq_meta.append(
            {
                "id": subj_id,
                "group": group,
                "block_nr": int(block_nr),
                "sequence_nr": int(seq_nr),
                "n_trials": int(ntr),
                "mean_trial_difficulty": mean_difficulty,
                "f": f,
                "f2": f**2,
                "half": half,
            }
        )

    # Not enough sequences → skip subject
    if len(seq_tf) < 5:
        continue

    tf_seq_data = np.stack(seq_tf, axis=0)  # (n_seq, n_ch, n_freq, n_time)
    df_seq = pd.DataFrame(seq_meta)
    df_seq["half"] = df_seq["half"].astype("category")

    # Z-score data
    mu = tf_seq_data.mean(axis=0, keepdims=True)
    sd = tf_seq_data.std(axis=0, ddof=1, keepdims=True) + 1e-12
    tf_seq_data_z = (tf_seq_data - mu) / sd

    # Build design matrix (same function you used for ERP)
    X, names = build_design_matrix(df_seq)

    # Container for betas: each regressor -> (n_ch, n_freq, n_time)
    betas = {
        name: np.zeros(
            (tf_seq_data_z.shape[1], tf_seq_data_z.shape[2], tf_seq_data_z.shape[3])
        )
        for name in names
    }

    # Loop over frequencies: run regression on (n_seq, n_ch, n_time) at each freq
    for fi in range(tf_seq_data_z.shape[2]):

        # Slice one frequency -> (n_seq, n_ch, n_time)
        seq_f = tf_seq_data_z[:, :, fi, :]

        epochs_seq_f = mne.EpochsArray(
            seq_f, info_tf_eff, tmin=tmin_eff, verbose=False
        )
        epochs_seq_f.metadata = df_seq
        lm = mne.stats.linear_regression(epochs_seq_f, X, names=names)
        
        # Extract betas (Evoked) and store into arrays
        for name in names:
            betas[name][:, fi, :] = lm[name].beta.data  # (n_ch, n_time)

    # Store subject result
    beta_maps_tf.append(
        {
            "id": subj_id,
            "group": df_seq["group"].iloc[0],
            "n_sequences": int(len(df_seq)),
            "betas": betas,  # arrays (ch, freq, time)
            "times": epochs_seq_f.times,
            "freqs": freqs,
            "info": info_tf,
        }
    )

# Get adjacencies for clustering in time freq channel space
times = beta_maps_tf[0]["times"]
freqs = beta_maps_tf[0]["freqs"]
adj_tfs = mne.stats.combine_adjacency(len(times), len(freqs), adjacency)

# Cluster test for main effect of quadratic term
X_f2 = stack_betas(beta_maps_tf, "f2")

# Get cluster threshold of 0.001 (two-sided)
n_subj = X_f2.shape[0]
t_thresh = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_subj - 1)

# Cluster test main effect quadratic
T_obs, clusters, pvals, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
    X_f2,
    adjacency=adj_tfs,
    n_permutations=500,
    threshold=t_thresh,
    tail=0,
    n_jobs=-2,
)

# Cluster test interaction
X_f2_exp = stack_betas([d for d in beta_maps_tf if d["group"] == "experimental"], "f2")
X_f2_ctl = stack_betas([d for d in beta_maps_tf if d["group"] == "control"], "f2")

# Get cluster threshold of 0.05 (two-sided)
n_subj = X_f2.shape[0]
t_thresh = scipy.stats.t.ppf(1 - 0.05 / 2, df=n_subj - 1)

T_obs_g, clusters_g, pvals_g, H0_g = mne.stats.spatio_temporal_cluster_test(
    [X_f2_exp, X_f2_ctl],
    adjacency=adj_tfs,
    n_permutations=2000,
    #threshold=t_thresh,
    tail=0,
    n_jobs=-2,
)



# = plot =================================================================================

alpha = 0.05
good = np.where(pvals < alpha)[0]
print(f"{len(good)} significant clusters at alpha={alpha}")

times = beta_maps_tf[0]["times"]
freqs = beta_maps_tf[0]["freqs"]
info = beta_maps_tf[0]["info"]

def get_cluster_indices_tf(cluster, n_times, n_freqs, n_ch):
    """
    Return ti, fi, si (unique integer indices) for a TF cluster,
    robust to different MNE cluster return formats.
    """
    # Case 1: boolean mask array of shape (n_times, n_freqs, n_ch)
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
        if cluster.shape != (n_times, n_freqs, n_ch):
            raise ValueError(f"Unexpected cluster mask shape: {cluster.shape}")
        ti, fi, si = np.where(cluster)
        return np.unique(ti), np.unique(fi), np.unique(si)

    # Case 2: tuple of index arrays OR boolean 1D masks
    if isinstance(cluster, tuple):
        if len(cluster) != 3:
            raise ValueError(f"Expected 3 elements (time,freq,space), got {len(cluster)}")
        t, f, s = cluster

        # If boolean masks, convert to indices
        if getattr(t, "dtype", None) == bool:
            t = np.where(t)[0]
        if getattr(f, "dtype", None) == bool:
            f = np.where(f)[0]
        if getattr(s, "dtype", None) == bool:
            s = np.where(s)[0]

        return np.unique(t), np.unique(f), np.unique(s)

    raise TypeError(f"Unknown cluster type: {type(cluster)}")


def _cluster_inds(clu):
    ti, fi, si = clu
    return np.unique(ti), np.unique(fi), np.unique(si)

for ci in good:
    
    n_times, n_freqs, n_ch = X_f2.shape[1], X_f2.shape[2], X_f2.shape[3]

    for ci in np.where(pvals < 0.05)[0]:
        ti, fi, si = get_cluster_indices_tf(clusters[ci], n_times, n_freqs, n_ch)

        print(ci, len(ti), len(fi), len(si))


    tmin, tmax = times[ti[0]], times[ti[-1]]
    fmin, fmax = freqs[fi[0]], freqs[fi[-1]]

    print(f"\nCluster {ci}: {tmin:.3f}–{tmax:.3f}s, {fmin:.1f}–{fmax:.1f}Hz, p={pvals[ci]:.4f}")
    print(f"  n_times={len(ti)}, n_freqs={len(fi)}, n_ch={len(si)}")

    # ---------------------------------------------------------
    # 1) Time×Freq map with CLUSTER OUTLINE (contour)
    # ---------------------------------------------------------
    tf_map = X_f2[:, :, :, si].mean(axis=(0, 3))  # (time, freq)

    # build boolean mask in (time, freq) space for this cluster
    tf_mask = np.zeros((len(times), len(freqs)), dtype=bool)
    tf_mask[np.ix_(ti, fi)] = True

    plt.figure(figsize=(7.4, 4.2))
    plt.imshow(
        tf_map.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )
    plt.colorbar(label="β_f² (a.u.)")
    plt.axvline(0, color="k", linestyle="--", linewidth=1)

    # Cluster outline as contour (white + black for visibility)
    # Need x,y grids in data coordinates
    tt, ff = np.meshgrid(times, freqs, indexing="xy")  # shapes (freq, time) if indexing=xy
    # Our tf_mask is (time, freq) so transpose it for (freq, time)
    plt.contour(
        times,
        freqs,
        tf_mask.T.astype(int),
        levels=[0.5],
        colors="white",
        linewidths=2.0,
    )
    plt.contour(
        times,
        freqs,
        tf_mask.T.astype(int),
        levels=[0.5],
        colors="black",
        linewidths=0.8,
        alpha=0.8,
    )

    plt.title(f"β_f² TF map (avg subj + cluster channels) | cluster {ci}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 2) Topography with CLUSTER CHANNELS MARKED
    # ---------------------------------------------------------
    topo = X_f2[:, ti[:, None], fi[None, :], :].mean(axis=(0, 1, 2))  # (ch,)

    evk = mne.EvokedArray(topo[:, None], info, tmin=0.0, verbose=False)

    # mask: True for channels in the cluster
    ch_mask = np.zeros(len(info["ch_names"]), dtype=bool)
    ch_mask[si] = True

    fig = evk.plot_topomap(
        times=0,
        time_format="",
        cmap="RdBu_r",
        contours=0,
        show=False,
        mask=ch_mask[:, None],  # mask expects shape (n_ch, n_times); we have 1 time
        mask_params=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="k",
            linewidth=0,
            markersize=8,
        ),
    )
    fig.suptitle(f"β_f² topo (cluster {ci}) {tmin:.2f}–{tmax:.2f}s, {fmin:.1f}–{fmax:.1f}Hz")
    plt.show()

    # ---------------------------------------------------------
    # 3) Time course (avg over cluster freqs+channels; full time)
    # ---------------------------------------------------------
    tc = X_f2[:, :, fi, :][:, :, :, si].mean(axis=(2, 3))  # (subj, time)
    mean_tc = tc.mean(axis=0)
    sem_tc = tc.std(axis=0, ddof=1) / np.sqrt(tc.shape[0])

    plt.figure(figsize=(7.2, 3.0))
    plt.plot(times, mean_tc, color="k", linewidth=1.7)
    plt.fill_between(times, mean_tc - sem_tc, mean_tc + sem_tc, color="k", alpha=0.25)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvspan(tmin, tmax, color="gray", alpha=0.15)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.title(f"β_f² time course (avg cluster freqs+channels) | cluster {ci}")
    plt.xlabel("Time (s)")
    plt.ylabel("β_f² (a.u.)")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4) Frequency profile (avg over cluster times+channels; full freqs)
    # ---------------------------------------------------------
    fp = X_f2[:, ti, :, :][:, :, :, si].mean(axis=(1, 3))  # (subj, freq)
    mean_fp = fp.mean(axis=0)
    sem_fp = fp.std(axis=0, ddof=1) / np.sqrt(fp.shape[0])

    plt.figure(figsize=(6.0, 3.0))
    plt.plot(freqs, mean_fp, color="k", linewidth=1.7)
    plt.fill_between(freqs, mean_fp - sem_fp, mean_fp + sem_fp, color="k", alpha=0.25)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    plt.axvspan(fmin, fmax, color="gray", alpha=0.15)
    plt.title(f"β_f² frequency profile (avg cluster times+channels) | cluster {ci}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("β_f² (a.u.)")
    plt.tight_layout()
    plt.show()
