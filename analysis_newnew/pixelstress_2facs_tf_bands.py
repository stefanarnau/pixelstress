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


def evokeds_to_array(evokeds):
    # input: list of Evoked (n_ch × n_time)
    # output: (n_subj × n_time × n_ch)
    data = np.stack([e.data for e in evokeds], axis=0)  # (n_subj, n_ch, n_time)
    return np.transpose(data, (0, 2, 1))


# Collectors
beta_maps = []  # list of dicts: {id, group, betas{name: Evoked}}
subj_seq_data_list = []

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

    bl_mask = (power.times >= -1.8) & (power.times <= -1.5)

    # baseline reference per subject, per ch×freq
    B = P[..., bl_mask].mean(axis=(0, 3))  # (n_ch, n_freq)

    # dB conversion per trial
    power.data = 10.0 * np.log10((P + 1e-12) / (B[None, :, :, None] + 1e-12))

    # Crpo in time
    power.crop(tmin=-1.8, tmax=1.0)

    # Times...
    times = power.times

    # Create info with effective sfreq from time vector
    sfreq_eff = 1.0 / np.mean(np.diff(times))
    info_tf_eff = mne.create_info(
        channel_labels, sfreq_eff, ch_types="eeg", verbose=None
    )
    tmin_eff = float(times[0])
    info_tf_eff.set_montage(montage, on_missing="warn", match_case=False)

    # Define freqbands (inclusive bounds)
    freqbands = {
        "theta": (4, 7),
        "alpha": (8, 13),
        "beta": (16, 30),
    }

    # Prepare grouping ONCE (reuse across bands)
    df = df.reset_index(drop=True)
    g = df.groupby(["block_nr", "sequence_nr"], sort=True)

    # Store results per band
    beta_maps_by_band = {band: [] for band in freqbands}
    subj_seq_data_list = []  # if you want to keep epochs for plotting later

    # Use TF time axis + sfreq that matches power.times
    times = power.times
    tmin_eff = float(times[0])

    sfreq_eff = 1.0 / np.mean(np.diff(times))
    info_tf_eff = info_tf.copy()
    info_tf_eff["sfreq"] = sfreq_eff

    for band_name, (fmin, fmax) in freqbands.items():

        # Band mask from power.freqs
        band_mask = (power.freqs >= fmin) & (power.freqs <= fmax)
        if band_mask.sum() == 0:
            print(
                f"[WARN] {band_name}: no freqs in {fmin}-{fmax} Hz (available: {power.freqs.min()}-{power.freqs.max()})"
            )
            continue

        # Average over frequencies -> (n_trials, n_ch, n_time)
        tf_band_data = power.data[:, :, band_mask, :].mean(axis=2)

        # -----------------------------
        # Sequence averaging
        # -----------------------------
        seq_data = []
        seq_meta = []

        for (block_nr, seq_nr), idx in g.indices.items():
            idx = np.asarray(idx)
            ntr = len(idx)

            if ntr < min_trials_per_sequence:
                continue

            # Average across trials in sequence -> (n_ch, n_time)
            seq_avg = tf_band_data[idx].mean(axis=0)
            seq_data.append(seq_avg)

            df_sub = df.loc[idx]

            f = float(df_sub["last_feedback_scaled"].iloc[0])
            mean_difficulty = float(
                pd.to_numeric(df_sub["trial_difficulty"], errors="coerce").mean()
            )
            half = "first" if int(block_nr) <= 4 else "second"
            group = df_sub["group"].iloc[0]

            seq_meta.append(
                dict(
                    id=subj_id,
                    group=group,
                    block_nr=int(block_nr),
                    sequence_nr=int(seq_nr),
                    n_trials=int(ntr),
                    mean_trial_difficulty=mean_difficulty,
                    f=f,
                    f2=f**2,
                    half=half,
                    freqband=band_name,
                )
            )

        # Not enough sequences → skip this band for this subject
        if len(seq_data) < 5:
            continue

        seq_arr = np.stack(seq_data, axis=0)  # (n_seq, n_ch, n_time)
        df_seq = pd.DataFrame(seq_meta)
        df_seq["half"] = df_seq["half"].astype("category")

        # Build EpochsArray of sequences (NOTE: tmin must match power.times[0])
        epochs_seq = mne.EpochsArray(seq_arr, info_tf_eff, tmin=tmin_eff, verbose=False)
        epochs_seq.metadata = df_seq

        # Design matrix
        X, names = build_design_matrix(df_seq)

        # Regression
        lm = mne.stats.linear_regression(epochs_seq, X, names=names)

        # Betas as Evoked objects (ch × time)
        betas = {name: lm[name].beta for name in names}

        beta_maps_by_band[band_name].append(
            dict(
                id=subj_id,
                group=df_seq["group"].iloc[0],
                n_sequences=int(len(df_seq)),
                betas=betas,
                times=epochs_seq.times,
                info=epochs_seq.info,
            )
        )

        subj_seq_data_list.append(
            dict(
                id=subj_id,
                group=df_seq["group"].iloc[0],
                freqband=band_name,
                epochs=epochs_seq,
            )
        )


# ====== Main effect: f2 ===================================================================================

betas_f2_all = [d["betas"]["f2"] for d in beta_maps]
X = evokeds_to_array(betas_f2_all)


T_obs, clusters, pvals, H0 = mne.stats.spatio_temporal_cluster_1samp_test(
    X,
    adjacency=adjacency,
    n_permutations=2000,
    tail=0,  # two-sided
    n_jobs=1,
)


alpha = 0.05
good_clusters = np.where(pvals < alpha)[0]

times = betas_f2_all[0].times
info = betas_f2_all[0].info

print(f"{len(good_clusters)} significant clusters at alpha={alpha}")

for ci in good_clusters:
    time_inds, space_inds = clusters[ci]
    time_inds = np.unique(time_inds)
    space_inds = np.unique(space_inds)

    tmin = times[time_inds[0]]
    tmax = times[time_inds[-1]]

    print(f"\nCluster {ci}: {tmin:.3f} – {tmax:.3f} s (p={pvals[ci]:.4f})")

    # ---------------------------------------------------------
    # 1) Topography: mean over subjects AND cluster time,
    # but show scalp pattern across channels (optionally restrict)
    # ---------------------------------------------------------

    # Mean over subjects and cluster time for each channel
    topo_allch = X[:, time_inds, :].mean(axis=(0, 1))  # (n_channels,)

    # Option A: show full scalp topography (recommended)
    evoked = betas_f2_all[0].copy()
    evoked.data[:] = topo_allch[:, None]  # (n_ch, 1)

    fig = evoked.plot_topomap(
        times=0,
        time_format="",
        cmap="RdBu_r",
        contours=0,
        show=False,
    )
    fig.suptitle(f"β_f² topography (cluster {ci}, {tmin:.2f}–{tmax:.2f}s)")
    plt.show()

    # Option B (optional): print which channels are in the cluster
    # print("Cluster channels:", [info["ch_names"][k] for k in space_inds])

    # ---------------------------------------------------------
    # 2) Time course: average over cluster channels, plot FULL time
    # ---------------------------------------------------------
    tc = X[:, :, space_inds].mean(axis=2)  # (n_subj, n_times) avg over cluster channels
    mean_tc = tc.mean(axis=0)
    sem_tc = tc.std(axis=0, ddof=1) / np.sqrt(tc.shape[0])

    plt.figure(figsize=(7, 3))
    plt.plot(times, mean_tc, color="k", linewidth=1.5)
    plt.fill_between(times, mean_tc - sem_tc, mean_tc + sem_tc, color="k", alpha=0.25)

    # Shade cluster window
    plt.axvspan(tmin, tmax, color="gray", alpha=0.2)

    plt.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.title(f"β_f² time course (cluster {ci} channels; window shaded)")
    plt.xlabel("Time (s)")
    plt.ylabel("Beta amplitude")
    plt.tight_layout()
    plt.show()

# ====== Group main effect: Intercept (EXP vs CTL) ==========================================================================================================================

betas_int_exp = [
    d["betas"]["Intercept"] for d in beta_maps if d["group"] == "experimental"
]
betas_int_ctl = [d["betas"]["Intercept"] for d in beta_maps if d["group"] == "control"]

X_exp = evokeds_to_array(betas_int_exp)
X_ctl = evokeds_to_array(betas_int_ctl)

T_obs, clusters, pvals, H0 = mne.stats.spatio_temporal_cluster_test(
    [X_exp, X_ctl],
    adjacency=adjacency,
    n_permutations=2000,
    tail=0,
    n_jobs=1,
)

alpha = 0.05
good_clusters = np.where(pvals < alpha)[0]

times = betas_int_exp[0].times
info = betas_int_exp[0].info

print(f"{len(good_clusters)} significant group-difference clusters at alpha={alpha}")

for ci in good_clusters:
    time_inds, space_inds = clusters[ci]
    time_inds = np.unique(time_inds)
    space_inds = np.unique(space_inds)

    tmin = times[time_inds[0]]
    tmax = times[time_inds[-1]]

    print(f"\nCluster {ci}: {tmin:.3f} – {tmax:.3f} s (p={pvals[ci]:.4f})")

    # ---------------------------------------------------------
    # 1) Topography of group difference (EXP - CTL) in this window
    # ---------------------------------------------------------
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
        f"Intercept group diff (EXP–CTL) cluster {ci} ({tmin:.2f}–{tmax:.2f}s)"
    )
    plt.show()

    # ---------------------------------------------------------
    # 2) Time course over cluster channels (plot both groups + diff)
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

    plt.title(f"Intercept time course (cluster {ci} channels; window shaded)")
    plt.xlabel("Time (s)")
    plt.ylabel("Intercept beta amplitude")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


# ====== Group comparison: f2 (experimental vs control) ==========================================

# -----------------------------
# 1) Split beta maps by group
# -----------------------------
betas_f2_exp = [d["betas"]["f2"] for d in beta_maps if d["group"] == "experimental"]
betas_f2_ctl = [d["betas"]["f2"] for d in beta_maps if d["group"] == "control"]

print(f"Experimental n={len(betas_f2_exp)} | Control n={len(betas_f2_ctl)}")

if len(betas_f2_exp) < 5 or len(betas_f2_ctl) < 5:
    raise RuntimeError("Too few subjects in one group for a reliable cluster test.")

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
alpha = 0.05
good_clusters = np.where(pvals < alpha)[0]

times = betas_f2_exp[0].times  # same times for both groups
info = betas_f2_exp[0].info

print(f"{len(good_clusters)} significant group-difference clusters at alpha={alpha}")

for ci in good_clusters:
    time_inds, space_inds = clusters[ci]
    time_inds = np.unique(time_inds)
    space_inds = np.unique(space_inds)

    tmin = times[time_inds[0]]
    tmax = times[time_inds[-1]]

    print(f"\nCluster {ci}: {tmin:.3f} – {tmax:.3f} s (p={pvals[ci]:.4f})")

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
    fig.suptitle(f"β_f² group diff (EXP–CTL) cluster {ci} ({tmin:.2f}–{tmax:.2f}s)")
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

    plt.title(f"β_f² time course (cluster {ci} channels; window shaded)")
    plt.xlabel("Time (s)")
    plt.ylabel("Beta amplitude")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


# --- define ROI ---
roi_chs = ["FCz", "Cz"]  # extend if you want
tmin_roi, tmax_roi = -0.5, 0.0

# collect subject-level ROI means
rows = []
for d in beta_maps:
    evk = d["betas"]["f2"]
    times = evk.times

    # channel picks that exist
    picks = [evk.ch_names.index(ch) for ch in roi_chs if ch in evk.ch_names]
    if len(picks) == 0:
        raise RuntimeError("None of the ROI channels were found in evoked.ch_names")

    tmask = (times >= tmin_roi) & (times <= tmax_roi)

    roi_mean = evk.data[np.ix_(picks, tmask)].mean()

    rows.append({"id": d["id"], "group": d["group"], "beta_f2_roi": roi_mean})

df_roi = pd.DataFrame(rows)

# --- stats ---
exp = df_roi.loc[df_roi.group == "experimental", "beta_f2_roi"].to_numpy()
ctl = df_roi.loc[df_roi.group == "control", "beta_f2_roi"].to_numpy()

print("ROI means (β_f²): EXP n=", len(exp), "CTL n=", len(ctl))

# 1-sample across all
t_all, p_all = scipy.stats.ttest_1samp(df_roi["beta_f2_roi"].to_numpy(), 0.0)
print(f"All-subjects β_f² ROI vs 0: t={t_all:.3f}, p={p_all:.4g}")

# group comparison
t_gc, p_gc = scipy.stats.ttest_ind(exp, ctl, equal_var=False)
print(f"Group diff (EXP-CTL) β_f² ROI: t={t_gc:.3f}, p={p_gc:.4g}")


# ======================================================================================
# CNV plot (publication-style): binned feedback × group
# - Uses sequence-level ERP data (erp_seq_data) + metadata (df_seq) per subject
# - ROI: Cz + FCz (or add Fz if you like)
# - Time: full window -1.0 to 0.5 s (as requested)
# - Feedback bins: 5–7 (choose n_bins)
# - Plot: mean ± SEM across subjects, with seaborn
# ======================================================================================


sns.set_theme(style="whitegrid", context="paper")

# -----------------------------
# User settings
# -----------------------------
tmin_plot, tmax_plot = -1.0, 0.5
roi_chs = ["FCz", "Cz"]  # add "Fz" if desired
n_bins = 5  # set 5–7 as you want
binning = "quantile"  # "quantile" (equal-count) or "fixed" (equal-width)
y_units = "µV"  # label only

# -----------------------------
# Assumes you have per-subject stored sequence data.
# You need a list like this from your beta-extraction loop:
#   subj_seq_data_list = [
#       {"id": subj_id, "group": group, "epochs": epochs_seq}, ...
#   ]
#
# If you didn't store epochs_seq, store *cropped* EpochsArray per subject during beta calc:
#   epochs_seq.crop(tmin=tmin_plot, tmax=tmax_plot)
#   subj_seq_data_list.append({"id": subj_id, "group": group, "epochs": epochs_seq})
# -----------------------------


# ---------- helper: build long dataframe for plotting ----------
def build_cnv_long_df(
    subj_seq_data_list, roi_chs, tmin_plot, tmax_plot, n_bins=7, binning="quantile"
):
    rows = []
    for sub in subj_seq_data_list:
        subj_id = sub["id"]
        group = sub["group"]
        epochs = sub["epochs"].copy()

        # crop (safe if already cropped)
        epochs.crop(tmin=tmin_plot, tmax=tmax_plot)

        # metadata must include f (signed feedback), and ideally is sequence-level
        md = epochs.metadata.copy()
        if "f" not in md.columns:
            raise RuntimeError("epochs.metadata must contain column 'f' (feedback).")

        # pick ROI channels
        picks = [epochs.ch_names.index(ch) for ch in roi_chs if ch in epochs.ch_names]
        if len(picks) == 0:
            raise RuntimeError(f"None of ROI channels found: {roi_chs}")

        data = epochs.get_data()  # (n_seq, n_ch, n_time)
        roi = data[:, picks, :].mean(axis=1)  # (n_seq, n_time)
        times = epochs.times

        # Build bins within subject? For your stated goal (global intuition), bin across all sequences.
        # We'll bin globally later; here store sequences with their feedback values.
        for i_seq in range(roi.shape[0]):
            rows.append(
                {
                    "id": subj_id,
                    "group": group,
                    "f": float(md.iloc[i_seq]["f"]),
                    "times": times,
                    "cnv": roi[i_seq, :],
                }
            )

    # Expand into long format: one row per (sequence × time)
    out = []
    for r in rows:
        out.append(
            pd.DataFrame(
                {
                    "id": r["id"],
                    "group": r["group"],
                    "f": r["f"],
                    "time": r["times"],
                    "cnv": r["cnv"],
                }
            )
        )
    df_long = pd.concat(out, ignore_index=True)

    # Create feedback bins (global across all subjects/sequences)
    if binning == "quantile":
        df_long["f_bin"] = pd.qcut(df_long["f"], q=n_bins, duplicates="drop")
    elif binning == "fixed":
        df_long["f_bin"] = pd.cut(df_long["f"], bins=n_bins)
    else:
        raise ValueError("binning must be 'quantile' or 'fixed'")

    # Bin centers for labeling
    centers = df_long.groupby("f_bin", observed=True)["f"].mean().sort_values()
    df_long["f_center"] = df_long["f_bin"].map(centers).astype(float)

    # Ordered categorical by center
    ordered_bins = centers.index.tolist()
    df_long["f_bin"] = pd.Categorical(
        df_long["f_bin"], categories=ordered_bins, ordered=True
    )

    return df_long


# ---------- helper: subject-mean timecourse per bin ----------
def subject_mean_timecourses(df_long):
    # Average within subject for each (group × f_bin × time)
    subj = (
        df_long.groupby(["id", "group", "f_bin", "f_center", "time"], observed=True)[
            "cnv"
        ]
        .mean()
        .reset_index(name="cnv")
    )

    # Then mean ± SEM across subjects for each (group × f_bin × time)
    summary = (
        subj.groupby(["group", "f_bin", "f_center", "time"], observed=True)["cnv"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"])
    return summary


# ============================
# Build plotting dataframe
# ============================
# df_long = build_cnv_long_df(subj_seq_data_list, roi_chs, tmin_plot, tmax_plot, n_bins=n_bins, binning=binning)
# df_sum  = subject_mean_timecourses(df_long)

# Make sure group order stable (optional)
# df_sum["group"] = df_sum["group"].astype("category")
# df_sum["group"] = df_sum["group"].cat.reorder_categories(["control", "experimental"], ordered=True)


# ============================
# Plot (seaborn FacetGrid): rows=group, columns=feedback bin
# ============================
def plot_cnv_facet(df_sum, tmin_plot, tmax_plot, y_units="µV"):
    # Create a readable label for each bin using center value
    df_sum = df_sum.copy()
    df_sum["bin_label"] = df_sum["f_center"].map(lambda x: f"{x:+.2f}")

    # Keep bins ordered by f_center
    bin_order = (
        df_sum[["f_bin", "f_center"]]
        .drop_duplicates()
        .sort_values("f_center")["f_bin"]
        .tolist()
    )

    g = sns.FacetGrid(
        df_sum,
        row="group",
        col="f_bin",
        col_order=bin_order,
        sharey=True,
        sharex=True,
        height=2.0,
        aspect=1.2,
        margin_titles=True,
    )

    def _panel(data, **kws):
        ax = plt.gca()
        ax.plot(data["time"], data["mean"], linewidth=1.8)
        ax.fill_between(
            data["time"],
            data["mean"] - data["sem"],
            data["mean"] + data["sem"],
            alpha=0.25,
        )
        ax.axvline(0, linestyle="--", linewidth=1, color="gray")
        ax.axhline(0, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
        ax.set_xlim(tmin_plot, tmax_plot)

    g.map_dataframe(_panel)

    # Nicer column titles: show bin center (instead of interval text)
    # Build mapping from f_bin to center label
    centers = df_sum.groupby("f_bin", observed=True)["f_center"].mean().sort_values()
    for ax, fbin in zip(g.axes[0], bin_order):
        ax.set_title(f"f ≈ {centers.loc[fbin]:+.2f}")

    # Axis labels
    g.set_axis_labels("Time (s)", f"CNV amplitude ({y_units})")

    # Row titles nicer
    for ax, row_name in zip(g.axes[:, 0], g.row_names):
        ax.annotate(
            row_name,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - 18, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size=11,
            ha="right",
            va="center",
            rotation=90,
        )

    plt.subplots_adjust(top=0.88)
    g.fig.suptitle(f"CNV (ROI: {', '.join(roi_chs)}) by feedback bin and group")
    sns.despine(trim=True)
    plt.show()


# ============================
# OPTIONAL: Single-panel overlay (cleaner for paper)
# Shows mean timecourses per bin, separate line style per group
# ============================
def plot_cnv_overlay(df_sum, tmin_plot, tmax_plot, y_units="µV"):
    df = df_sum.copy()

    # Discretize bins to ordered labels
    centers = df.groupby("f_bin", observed=True)["f_center"].mean().sort_values()
    bin_order = centers.index.tolist()

    df["bin_label"] = pd.Categorical(
        df["f_bin"].map(lambda b: f"{centers.loc[b]:+.2f}"),
        ordered=True,
        categories=[f"{centers.loc[b]:+.2f}" for b in bin_order],
    )

    # Plot: hue=bin_label, style=group
    plt.figure(figsize=(8.5, 4.2))
    ax = plt.gca()

    sns.lineplot(
        data=df,
        x="time",
        y="mean",
        hue="bin_label",
        style="group",
        linewidth=2.0,
        ax=ax,
    )

    # Shaded SEM (manual, by group×bin)
    for (grp, bl), sub in df.groupby(["group", "bin_label"], observed=True):
        ax.fill_between(
            sub["time"].to_numpy(),
            (sub["mean"] - sub["sem"]).to_numpy(),
            (sub["mean"] + sub["sem"]).to_numpy(),
            alpha=0.12,
        )

    ax.axvline(0, linestyle="--", linewidth=1, color="gray")
    ax.axhline(0, linestyle="--", linewidth=0.8, color="gray", alpha=0.6)
    ax.set_xlim(tmin_plot, tmax_plot)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"CNV amplitude ({y_units})")
    ax.set_title(
        f"CNV by feedback bin (hue) and group (style) | ROI: {', '.join(roi_chs)}"
    )
    ax.legend(
        title="Feedback bin center (f)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()


# ======================================================================================
# USAGE
# ======================================================================================
df_long = build_cnv_long_df(
    subj_seq_data_list, ["Cz"], tmin_plot, tmax_plot, n_bins=n_bins, binning=binning
)
df_sum = subject_mean_timecourses(df_long)
plot_cnv_overlay(df_sum, tmin_plot, tmax_plot, y_units=y_units)  # single-panel view
