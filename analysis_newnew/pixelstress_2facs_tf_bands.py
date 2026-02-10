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

# Helper: build design matrix (within subject)
def build_design_matrix(df_seq: pd.DataFrame):
    X = df_seq[["f", "f2", "mean_trial_difficulty", "half"]].copy()
    X["half"] = (X["half"].astype(str) == "second").astype(int)
    X = X.astype(float)

    Xz = StandardScaler().fit_transform(X)  # within-subject z-scoring
    Xz = np.column_stack([np.ones(len(Xz)), Xz])  # intercept
    names = ["Intercept", "f", "f2", "difficulty", "half"]
    return Xz, names

def evokeds_to_array(evokeds):
    # input: list of Evoked (n_ch × n_time)
    # output: (n_subj × n_time × n_ch)
    data = np.stack([e.data for e in evokeds], axis=0)  # (n_subj, n_ch, n_time)
    return np.transpose(data, (0, 2, 1))


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

# Collectors
beta_maps = []  # list of dicts: {id, group, betas{name: Evoked}}

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

    # Se tf-decomposition parameters
    freqs = np.arange(4, 31, 2)
    n_cycles = freqs / 2.0

    # Create epochs object
    epochs = mne.EpochsArray(tf_data, info_tf, tmin=tmin, verbose=False)
    epochs.metadata = df

    # TF decomposition
    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=False,
        output="power",
        n_jobs=-1,
        verbose=False,
    )

    # -----------------------------
    # Sequence averaging for TFR: (block_nr, sequence_nr)
    # -----------------------------
    df = df.reset_index(drop=True)
    g = df.groupby(["block_nr", "sequence_nr"], sort=True)

    seq_tfr = []
    seq_meta = []

    X = power.data  # (n_trials, n_ch, n_freq, n_time)

    for (block_nr, seq_nr), idx in g.indices.items():
        idx = np.asarray(idx)
        ntr = len(idx)

        if ntr < min_trials_per_sequence:
            continue

        # Average TFR across trials in this sequence
        seq_avg = X[idx].mean(axis=0)  # (n_ch, n_freq, n_time)
        seq_tfr.append(seq_avg)

        df_sub = df.loc[idx]

        f = float(df_sub["last_feedback_scaled"].iloc[0])
        mean_difficulty = float(df_sub["trial_difficulty"].mean())
        half = "first" if int(block_nr) <= 4 else "second"
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

    if len(seq_tfr) < 5:
        continue

    tfr_seq_data = np.stack(seq_tfr, axis=0)  # (n_seq, n_ch, n_freq, n_time)
    df_seq = pd.DataFrame(seq_meta)
    df_seq["half"] = df_seq["half"].astype("category")

    # Build a "sequence TFR" object by copying and overwriting
    n_seq = tfr_seq_data.shape[0]
    power_seq = power.copy()
    power_seq.data = tfr_seq_data

    # Create dummy events: (sample, 0, event_id)
    power_seq.events = np.c_[
        np.arange(n_seq), np.zeros(n_seq, int), np.ones(n_seq, int)
    ]

    # Also update selection
    power_seq.selection = np.arange(n_seq)

    # Setmetadata
    power_seq.metadata = df_seq.reset_index(drop=True)

    # Specify baseline
    baseline = (-1.5, -1.2)

    # Get baseline indices
    bl_idx = (power_seq.times >= baseline[0]) & (power_seq.times <= baseline[1])

    # Copy tf-power
    X = power_seq.data

    # Get baseline means(n_channels, n_freqs)
    bl_mean = X[..., bl_idx].mean(axis=(0, -1))

    # Avoid divide-by-zero
    eps = np.finfo(power_seq.data.dtype).eps
    bl_mean = np.maximum(bl_mean, eps)

    # Calculate logratio of each trial
    Xc = 10 * np.log10(X / bl_mean[None, :, :, None])

    # BL-corrected data in a copy of power object
    power_seq_bl = power_seq.copy()
    power_seq_bl.data = Xc

    # Crop in time
    power_seq_bl.crop(tmin=-1.5, tmax=1)

    # Define freqbands (inclusive bounds)
    freqbands = {
        "theta": (4, 7),
        "alpha": (8, 13),
        "beta": (16, 30),
    }

    # power_seq_bl: (n_seq, n_ch, n_freq, n_time) with metadata df_seq
    # info_tf must have sfreq consistent with your time axis (it is, since you created epochs with sfreq=200)
    
    X, names = build_design_matrix(df_seq)  # same as ERP
    
    lm_by_band = {}
    betas_by_band = {}
    
    for band, (fmin, fmax) in freqbands.items():
        # band mask (inclusive bounds)
        fmask = (power_seq_bl.freqs >= fmin) & (power_seq_bl.freqs <= fmax)
        if not fmask.any():
            raise ValueError(f"No freqs fall into band {band}: {fmin}-{fmax} Hz")
    
        # collapse freq dimension -> (n_seq, n_ch, n_time)
        band_data = power_seq_bl.data[:, :, fmask, :].mean(axis=2)
    
        # wrap as EpochsArray (this is what linear_regression needs)
        epochs_band = mne.EpochsArray(
            band_data,
            info_tf,
            tmin=power_seq_bl.times[0],
            verbose=False,
        )
        epochs_band.metadata = df_seq.reset_index(drop=True)
    
        # optional time crop (do it here, not on the 4D object)
        epochs_band.crop(tmin=-1.5, tmax=1.0)
    
        # regression
        lm = mne.stats.linear_regression(epochs_band, X, names=names)
    
        # store betas (Evoked objects, exactly like ERP)
        betas = {name: lm[name].beta for name in names}
    
        lm_by_band[band] = lm
        betas_by_band[band] = betas

    beta_maps.append(
        {
            "id": subj_id,
            "group": df_seq["group"].iloc[0],
            "n_sequences": int(len(df_seq)),
            "betas_by_band": betas_by_band,
        }
    )

# ====== Main effect: f (per band) ======
alpha = 0.05
bands = ["theta", "alpha", "beta"]

for band in bands:
    
    betas_f_all = [d["betas_by_band"][band]["f"] for d in beta_maps]
    X = evokeds_to_array(betas_f_all)  # (n_subj, n_times, n_channels)

    T_obs, clusters, pvals, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_permutations=2000,
        tail=0,  # two-sided
        n_jobs=-1,
    )

    good_clusters = np.where(pvals < alpha)[0]

    times = betas_f_all[0].times
    info = betas_f_all[0].info

    print(f"\n[{band}] {len(good_clusters)} significant clusters at alpha={alpha}")

    for ci in good_clusters:
        time_inds, space_inds = clusters[ci]
        time_inds = np.unique(time_inds)
        space_inds = np.unique(space_inds)

        tmin = times[time_inds[0]]
        tmax = times[time_inds[-1]]

        print(f"Cluster {ci}: {tmin:.3f} – {tmax:.3f} s (p={pvals[ci]:.4f})")

        # 1) Topography: mean over subjects AND cluster time
        topo_allch = X[:, time_inds, :].mean(axis=(0, 1))  # (n_channels,)
        evoked = betas_f_all[0].copy()
        evoked.data[:] = topo_allch[:, None]  # (n_ch, 1)

        fig = evoked.plot_topomap(
            times=0,
            time_format="",
            cmap="RdBu_r",
            contours=0,
            show=False,
        )
        fig.suptitle(f"[{band}] β_f topography (cluster {ci}, {tmin:.2f}–{tmax:.2f}s)")
        plt.show()

        # 2) Time course: average over cluster channels, plot FULL time
        tc = X[:, :, space_inds].mean(axis=2)  # (n_subj, n_times)
        mean_tc = tc.mean(axis=0)
        sem_tc = tc.std(axis=0, ddof=1) / np.sqrt(tc.shape[0])

        plt.figure(figsize=(7, 3))
        plt.plot(times, mean_tc, color="k", linewidth=1.5)
        plt.fill_between(times, mean_tc - sem_tc, mean_tc + sem_tc, color="k", alpha=0.25)
        plt.axvspan(tmin, tmax, color="gray", alpha=0.2)
        plt.axhline(0, linestyle="--", color="gray", linewidth=1)
        plt.title(f"[{band}] β_f time course (cluster {ci} channels; window shaded)")
        plt.xlabel("Time (s)")
        plt.ylabel("Beta amplitude")
        plt.tight_layout()
        plt.show()
        
# ====== Main effect: f2 (per band) ======
alpha = 0.05
bands = ["theta", "alpha", "beta"]

for band in bands:
    
    betas_f2_all = [d["betas_by_band"][band]["f2"] for d in beta_maps]
    X = evokeds_to_array(betas_f2_all)  # (n_subj, n_times, n_channels)

    T_obs, clusters, pvals, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        adjacency=adjacency,
        n_permutations=2000,
        tail=0,  # two-sided
        n_jobs=1,
    )

    good_clusters = np.where(pvals < alpha)[0]

    times = betas_f2_all[0].times
    info = betas_f2_all[0].info

    print(f"\n[{band}] {len(good_clusters)} significant clusters at alpha={alpha}")

    for ci in good_clusters:
        time_inds, space_inds = clusters[ci]
        time_inds = np.unique(time_inds)
        space_inds = np.unique(space_inds)

        tmin = times[time_inds[0]]
        tmax = times[time_inds[-1]]

        print(f"Cluster {ci}: {tmin:.3f} – {tmax:.3f} s (p={pvals[ci]:.4f})")

        # 1) Topography: mean over subjects AND cluster time
        topo_allch = X[:, time_inds, :].mean(axis=(0, 1))  # (n_channels,)
        evoked = betas_f2_all[0].copy()
        evoked.data[:] = topo_allch[:, None]  # (n_ch, 1)

        fig = evoked.plot_topomap(
            times=0,
            time_format="",
            cmap="RdBu_r",
            contours=0,
            show=False,
        )
        fig.suptitle(f"[{band}] β_f² topography (cluster {ci}, {tmin:.2f}–{tmax:.2f}s)")
        plt.show()

        # 2) Time course: average over cluster channels, plot FULL time
        tc = X[:, :, space_inds].mean(axis=2)  # (n_subj, n_times)
        mean_tc = tc.mean(axis=0)
        sem_tc = tc.std(axis=0, ddof=1) / np.sqrt(tc.shape[0])

        plt.figure(figsize=(7, 3))
        plt.plot(times, mean_tc, color="k", linewidth=1.5)
        plt.fill_between(times, mean_tc - sem_tc, mean_tc + sem_tc, color="k", alpha=0.25)
        plt.axvspan(tmin, tmax, color="gray", alpha=0.2)
        plt.axhline(0, linestyle="--", color="gray", linewidth=1)
        plt.title(f"[{band}] β_f² time course (cluster {ci} channels; window shaded)")
        plt.xlabel("Time (s)")
        plt.ylabel("Beta amplitude")
        plt.tight_layout()
        plt.show()

# =========================== Group effect (intercept)
alpha = 0.05
bands = ["theta", "alpha", "beta"]

for band in bands:
    print(f"\n===== Group main effect: Intercept ({band}) (EXP vs CTL) =====")

    betas_int_exp = [
        d["betas_by_band"][band]["Intercept"]
        for d in beta_maps
        if d["group"] == "experimental"
    ]
    betas_int_ctl = [
        d["betas_by_band"][band]["Intercept"]
        for d in beta_maps
        if d["group"] == "control"
    ]

    print(f"[{band}] Experimental n={len(betas_int_exp)} | Control n={len(betas_int_ctl)}")
    if len(betas_int_exp) < 5 or len(betas_int_ctl) < 5:
        print(f"[{band}] Too few subjects in one group; skipping.")
        continue

    X_exp = evokeds_to_array(betas_int_exp)  # (n_subj, n_times, n_ch)
    X_ctl = evokeds_to_array(betas_int_ctl)
    
    # Determine cluster forming threshold
    n_exp = X_exp.shape[0]
    n_ctl = X_ctl.shape[0]
    degf = n_exp + n_ctl - 2  # degrees of freedom for two-sample t
    p_threshold = 0.05   # stricter than default 0.05
    t_threshold = scipy.stats.t.ppf(1 - p_threshold / 2, degf)

    T_obs, clusters, pvals, H0 = mne.stats.spatio_temporal_cluster_test(
        [X_exp, X_ctl],
        adjacency=adjacency,
        threshold=t_threshold,
        n_permutations=2000,
        tail=0,
        n_jobs=-1,
    )

    good_clusters = np.where(pvals < alpha)[0]

    times = betas_int_exp[0].times
    info = betas_int_exp[0].info

    print(f"[{band}] {len(good_clusters)} significant group-difference clusters at alpha={alpha}")

    for ci in good_clusters:
        time_inds, space_inds = clusters[ci]
        time_inds = np.unique(time_inds)
        space_inds = np.unique(space_inds)

        tmin = times[time_inds[0]]
        tmax = times[time_inds[-1]]

        print(f"\n[{band}] Cluster {ci}: {tmin:.3f} – {tmax:.3f} s (p={pvals[ci]:.4f})")

        # 1) Topography of group difference (EXP - CTL) in this window
        topo_exp = X_exp[:, time_inds, :].mean(axis=(0, 1))  # (n_ch,)
        topo_ctl = X_ctl[:, time_inds, :].mean(axis=(0, 1))  # (n_ch,)
        topo_diff = topo_exp - topo_ctl

        evoked = betas_int_exp[0].copy()
        evoked.data[:] = topo_diff[:, None]

        fig = evoked.plot_topomap(
            times=0,
            time_format="",
            cmap="RdBu_r",
            contours=0,
            show=False,
        )
        fig.suptitle(
            f"[{band}] Intercept group diff (EXP–CTL) cluster {ci} ({tmin:.2f}–{tmax:.2f}s)"
        )
        plt.show()

        # 2) Time course over cluster channels (plot both groups + diff)
        tc_exp = X_exp[:, :, space_inds].mean(axis=2)  # (n_subj, n_times)
        tc_ctl = X_ctl[:, :, space_inds].mean(axis=2)

        mean_exp = tc_exp.mean(axis=0)
        sem_exp = tc_exp.std(axis=0, ddof=1) / np.sqrt(tc_exp.shape[0])

        mean_ctl = tc_ctl.mean(axis=0)
        sem_ctl = tc_ctl.std(axis=0, ddof=1) / np.sqrt(tc_ctl.shape[0])

        mean_diff = mean_exp - mean_ctl

        plt.figure(figsize=(8, 3.2))
        plt.plot(times, mean_exp, label="experimental", linewidth=1.5)
        plt.fill_between(times, mean_exp - sem_exp, mean_exp + sem_exp, alpha=0.2)

        plt.plot(times, mean_ctl, label="control", linewidth=1.5)
        plt.fill_between(times, mean_ctl - sem_ctl, mean_ctl + sem_ctl, alpha=0.2)

        plt.plot(times, mean_diff, label="exp - ctl", linestyle="--", linewidth=1.5)

        plt.axvspan(tmin, tmax, color="gray", alpha=0.2)
        plt.axhline(0, linestyle="--", color="gray", linewidth=1)

        plt.title(f"[{band}] Intercept time course (cluster {ci}; window shaded)")
        plt.xlabel("Time (s)")
        plt.ylabel("Intercept beta amplitude")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()

# ====== Group comparison: f2 (experimental vs control) ==========================================

alpha = 0.05
bands = ["theta", "alpha", "beta"]

for band in bands:
    print(f"\n===== Group comparison: f2 ({band}) (experimental vs control) =====")

    # -----------------------------
    # 1) Split beta maps by group
    # -----------------------------
    betas_f2_exp = [
        d["betas_by_band"][band]["f2"]
        for d in beta_maps
        if d["group"] == "experimental"
    ]
    betas_f2_ctl = [
        d["betas_by_band"][band]["f2"]
        for d in beta_maps
        if d["group"] == "control"
    ]

    print(f"[{band}] Experimental n={len(betas_f2_exp)} | Control n={len(betas_f2_ctl)}")

    if len(betas_f2_exp) < 5 or len(betas_f2_ctl) < 5:
        print(f"[{band}] Too few subjects in one group; skipping.")
        continue

    X_exp = evokeds_to_array(betas_f2_exp)  # (n_subj, n_times, n_ch)
    X_ctl = evokeds_to_array(betas_f2_ctl)

    # -----------------------------
    # 2) 2-sample spatio-temporal cluster permutation test
    # -----------------------------
    T_obs, clusters, pvals, H0 = mne.stats.spatio_temporal_cluster_test(
        [X_exp, X_ctl],
        adjacency=adjacency,
        n_permutations=2000,
        tail=0,  # two-sided
        n_jobs=1,
    )

    # -----------------------------
    # 3) Quick inspection plots for significant clusters
    # -----------------------------
    good_clusters = np.where(pvals < alpha)[0]

    times = betas_f2_exp[0].times
    info = betas_f2_exp[0].info

    print(f"[{band}] {len(good_clusters)} significant group-difference clusters at alpha={alpha}")

    for ci in good_clusters:
        time_inds, space_inds = clusters[ci]
        time_inds = np.unique(time_inds)
        space_inds = np.unique(space_inds)

        tmin = times[time_inds[0]]
        tmax = times[time_inds[-1]]

        print(f"\n[{band}] Cluster {ci}: {tmin:.3f} – {tmax:.3f} s (p={pvals[ci]:.4f})")

        # ---------------------------------------------------------
        # 1) Topography of group difference (EXP - CTL) in this window
        # ---------------------------------------------------------
        topo_exp = X_exp[:, time_inds, :].mean(axis=(0, 1))  # (n_ch,)
        topo_ctl = X_ctl[:, time_inds, :].mean(axis=(0, 1))  # (n_ch,)
        topo_diff = topo_exp - topo_ctl

        evoked = betas_f2_exp[0].copy()
        evoked.data[:] = topo_diff[:, None]

        fig = evoked.plot_topomap(
            times=0,
            time_format="",
            cmap="RdBu_r",
            contours=0,
            show=False,
        )
        fig.suptitle(
            f"[{band}] β_f² group diff (EXP–CTL) cluster {ci} ({tmin:.2f}–{tmax:.2f}s)"
        )
        plt.show()

        # ---------------------------------------------------------
        # 2) Time course averaged over cluster channels, full epoch
        #    Plot both groups + difference, shade significant window
        # ---------------------------------------------------------
        tc_exp = X_exp[:, :, space_inds].mean(axis=2)  # (n_subj, n_times)
        tc_ctl = X_ctl[:, :, space_inds].mean(axis=2)

        mean_exp = tc_exp.mean(axis=0)
        sem_exp = tc_exp.std(axis=0, ddof=1) / np.sqrt(tc_exp.shape[0])

        mean_ctl = tc_ctl.mean(axis=0)
        sem_ctl = tc_ctl.std(axis=0, ddof=1) / np.sqrt(tc_ctl.shape[0])

        mean_diff = mean_exp - mean_ctl

        plt.figure(figsize=(8, 3.2))
        plt.plot(times, mean_exp, label="experimental", linewidth=1.5)
        plt.fill_between(times, mean_exp - sem_exp, mean_exp + sem_exp, alpha=0.2)

        plt.plot(times, mean_ctl, label="control", linewidth=1.5)
        plt.fill_between(times, mean_ctl - sem_ctl, mean_ctl + sem_ctl, alpha=0.2)

        plt.plot(times, mean_diff, label="exp - ctl", linestyle="--", linewidth=1.5)

        plt.axvspan(tmin, tmax, color="gray", alpha=0.2)
        plt.axhline(0, linestyle="--", color="gray", linewidth=1)

        plt.title(f"[{band}] β_f² time course (cluster {ci}; window shaded)")
        plt.xlabel("Time (s)")
        plt.ylabel("Beta amplitude")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.show()





