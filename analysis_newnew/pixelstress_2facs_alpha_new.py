# step2_alpha_plots_and_clusters_masks_topos_plus_roi_bins.py
#
# Theta-band version of your step2 script:
# - Uses TF alpha-band betas: index.tf_alpha_betas_path
# - Uses baseline-corrected sequence TFR (4D) for ROI bin plot: index.tfr_epochs_path
# - Heatmaps + cluster contours + topomaps work on alpha-band betas (Evoked)
# - ROI/bin plot uses alpha-band power (collapsed freq) averaged over CLUSTER CHANNELS
#
# Assumptions:
# - subject_index.csv contains:
#     tf_alpha_betas_path, tfr_epochs_path, group, id
# - tfr_epochs_path points to power_seq_bl saved via .save(... .h5)
# - power_seq_bl.metadata contains column 'f' and 'group'
# - power_seq_bl.freqs contains the original freqs (4..30 step 2), so alpha mask works.

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths / IO
# -----------------------------
path_out = Path("/mnt/data_dump/pixelstress/3_sequence_data_plus_fitted/")
index = pd.read_csv(path_out / "subject_index.csv")

# -----------------------------
# Settings
# -----------------------------
NAMES = ["Intercept", "f", "f2", "difficulty", "half"]
alpha = 0.05
n_perm = 2000

# alpha band definition (must match your regression step)
ALPHA = (8.0, 13.0)

# -----------------------------
# Helpers
# -----------------------------
def load_betas_evoked(fif_path: str, names=NAMES):
    evs = mne.read_evokeds(fif_path, condition=None, verbose=False)
    if len(evs) != len(names):
        raise RuntimeError(f"{fif_path}: expected {len(names)} evokeds, got {len(evs)}")
    return dict(zip(names, evs))


def stack_evokeds(evokeds):
    """Return (X, times, ch_names, info) where X is (n_subj, n_ch, n_time)."""
    data = np.stack([e.data for e in evokeds], axis=0)
    return data, evokeds[0].times, evokeds[0].ch_names, evokeds[0].info


def evokeds_to_X(evokeds):
    """For cluster tests: (n_subj, n_times, n_ch)."""
    data = np.stack([e.data for e in evokeds], axis=0)  # (n_subj, n_ch, n_time)
    return np.transpose(data, (0, 2, 1))  # (n_subj, n_time, n_ch)


def plot_beta_heatmap_with_clusters(
    beta_ch_time,
    times,
    ch_names,
    title,
    vlim=None,
    clusters=None,
    pvals=None,
    alpha=0.05,
    cmap="coolwarm",
    contour_lw=3.0,
):
    plt.figure(figsize=(12, 8))
    im = plt.imshow(
        beta_ch_time,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, len(ch_names)],
        cmap=cmap,
        vmin=None if vlim is None else -vlim,
        vmax=None if vlim is None else vlim,
    )
    plt.colorbar(im, label="Beta amplitude")
    plt.yticks(np.arange(len(ch_names)) + 0.5, ch_names, fontsize=7)
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.title(title)

    if clusters is not None and pvals is not None:
        good = np.where(pvals < alpha)[0]
        if len(good) > 0:
            x = times
            y = np.arange(len(ch_names)) + 0.5
            for ci in good:
                Z = clusters[ci].T.astype(float)  # (n_ch, n_time)
                plt.contour(x, y, Z, levels=[0.5], linewidths=contour_lw)

    plt.tight_layout()
    plt.show()


def plot_cluster_topomap(
    evoked_template,
    mean_beta_ch_time,
    times,
    cluster_mask_tc,
    title_prefix,
    mask_sensors=True,
):
    time_inds = np.where(cluster_mask_tc.any(axis=1))[0]
    if len(time_inds) == 0:
        print(f"[topomap] {title_prefix}: cluster has no time samples (skipping).")
        return

    tmin = float(times[time_inds[0]])
    tmax = float(times[time_inds[-1]])

    topo = mean_beta_ch_time[:, time_inds].mean(axis=1)  # (n_ch,)
    evk = mne.EvokedArray(topo[:, None], evoked_template.info, tmin=0.0, comment="")

    if mask_sensors:
        ch_mask = cluster_mask_tc.any(axis=0)  # (n_ch,)
        mask = ch_mask[:, None]               # (n_ch, 1)
        mask_params = dict(marker="o", markerfacecolor="none", markeredgewidth=1.5)
    else:
        mask = None
        mask_params = None

    fig = evk.plot_topomap(
        times=[0.0],
        time_format="",
        cmap="coolwarm",
        contours=0,
        show=False,
        mask=mask,
        mask_params=mask_params,
    )
    fig.suptitle(f"{title_prefix}\ncluster time-range: {tmin:.3f}–{tmax:.3f} s")
    plt.show()


# =====================================================================
# 1) Load ALPHA betas for all subjects
# =====================================================================
betas_all = []
for row in index.itertuples(index=False):
    betas = load_betas_evoked(row.tf_alpha_betas_path)  # <-- alpha betas
    betas_all.append(dict(id=row.id, group=row.group, betas=betas))

betas_exp = [d for d in betas_all if d["group"] == "experimental"]
betas_ctl = [d for d in betas_all if d["group"] == "control"]

print(
    f"Loaded ALPHA betas: total n={len(betas_all)} | "
    f"exp n={len(betas_exp)} | ctl n={len(betas_ctl)}"
)

# adjacency aligned to saved alpha evokeds
example_evoked = betas_all[0]["betas"]["f2"]
adjacency, ch_names = mne.channels.find_ch_adjacency(example_evoked.info, ch_type="eeg")

# =====================================================================
# 2) Cluster tests (alpha betas)
# =====================================================================
# 1) One-sample: beta_f
evs_f = [d["betas"]["f"] for d in betas_all]
X_f = evokeds_to_X(evs_f)
T_obs_f, clusters_f, pvals_f, H0_f = mne.stats.spatio_temporal_cluster_1samp_test(
    X_f,
    adjacency=adjacency,
    n_permutations=n_perm,
    tail=0,
    n_jobs=-1,
    out_type="mask",
)
print(f"[alpha beta_f] significant clusters: {np.sum(pvals_f < alpha)} (alpha={alpha})")

# 2) One-sample: beta_f2
evs_f2 = [d["betas"]["f2"] for d in betas_all]
X_f2 = evokeds_to_X(evs_f2)
T_obs_f2, clusters_f2, pvals_f2, H0_f2 = mne.stats.spatio_temporal_cluster_1samp_test(
    X_f2,
    adjacency=adjacency,
    n_permutations=n_perm,
    tail=0,
    n_jobs=-1,
    out_type="mask",
)
print(f"[alpha beta_f2] significant clusters: {np.sum(pvals_f2 < alpha)} (alpha={alpha})")

# 3) Two-sample: Intercept (EXP vs CTL)
clusters_int, pvals_int = None, None
diff_map_int = None
times_int = None
if len(betas_exp) < 5 or len(betas_ctl) < 5:
    print("[alpha Intercept EXP vs CTL] Too few subjects per group; skipping test.")
else:
    evs_int_exp = [d["betas"]["Intercept"] for d in betas_exp]
    evs_int_ctl = [d["betas"]["Intercept"] for d in betas_ctl]
    X_int_exp = evokeds_to_X(evs_int_exp)
    X_int_ctl = evokeds_to_X(evs_int_ctl)

    T_obs_int, clusters_int, pvals_int, H0_int = mne.stats.spatio_temporal_cluster_test(
        [X_int_exp, X_int_ctl],
        adjacency=adjacency,
        n_permutations=n_perm,
        tail=0,
        n_jobs=-1,
        out_type="mask",
    )
    print(
        f"[alpha Intercept EXP vs CTL] significant clusters: {np.sum(pvals_int < alpha)} (alpha={alpha})"
    )

# 4) Two-sample: f2 × group proxy (beta_f2 EXP vs CTL)
clusters_f2g, pvals_f2g = None, None
diff_map_f2g = None
times_f2g = None
if len(betas_exp) < 5 or len(betas_ctl) < 5:
    print("[alpha beta_f2 EXP vs CTL] Too few subjects per group; skipping test.")
else:
    evs_f2_exp = [d["betas"]["f2"] for d in betas_exp]
    evs_f2_ctl = [d["betas"]["f2"] for d in betas_ctl]
    X_f2_exp = evokeds_to_X(evs_f2_exp)
    X_f2_ctl = evokeds_to_X(evs_f2_ctl)

    T_obs_f2g, clusters_f2g, pvals_f2g, H0_f2g = mne.stats.spatio_temporal_cluster_test(
        [X_f2_exp, X_f2_ctl],
        adjacency=adjacency,
        n_permutations=n_perm,
        tail=0,
        n_jobs=-1,
        out_type="mask",
    )
    print(
        f"[alpha beta_f2 EXP vs CTL] significant clusters: {np.sum(pvals_f2g < alpha)} (alpha={alpha})"
    )

# =====================================================================
# 3) Heatmaps + topomaps (alpha betas)
# =====================================================================
# A) Grand-average beta_f
X, times, ch_names_plot, _ = stack_evokeds(evs_f)
mean_map_f = X.mean(axis=0)
vmax_f = np.nanpercentile(np.abs(mean_map_f), 99)

plot_beta_heatmap_with_clusters(
    mean_map_f,
    times,
    ch_names_plot,
    title=f"ALPHA β_f: mean across subjects (n={len(evs_f)})",
    vlim=vmax_f,
    clusters=clusters_f,
    pvals=pvals_f,
    alpha=alpha,
    cmap="coolwarm",
    contour_lw=3.0,
)

# B) Grand-average beta_f2
X, times, ch_names_plot, _ = stack_evokeds(evs_f2)
mean_map_f2 = X.mean(axis=0)
vmax_f2 = np.nanpercentile(np.abs(mean_map_f2), 99)

plot_beta_heatmap_with_clusters(
    mean_map_f2,
    times,
    ch_names_plot,
    title=f"ALPHA β_f²: mean across subjects (n={len(evs_f2)})",
    vlim=vmax_f2,
    clusters=clusters_f2,
    pvals=pvals_f2,
    alpha=alpha,
    cmap="coolwarm",
    contour_lw=3.0,
)

# C) Intercept group difference (EXP - CTL)
if len(betas_exp) >= 2 and len(betas_ctl) >= 2:
    Xexp, times_int, ch_names_plot_int, _ = stack_evokeds(
        [d["betas"]["Intercept"] for d in betas_exp]
    )
    Xctl, _, _, _ = stack_evokeds([d["betas"]["Intercept"] for d in betas_ctl])
    diff_map_int = Xexp.mean(axis=0) - Xctl.mean(axis=0)
    vmax_int = np.nanpercentile(np.abs(diff_map_int), 99)

    plot_beta_heatmap_with_clusters(
        diff_map_int,
        times_int,
        ch_names_plot_int,
        title="ALPHA Intercept group difference (EXP − CTL)",
        vlim=vmax_int,
        clusters=clusters_int,
        pvals=pvals_int,
        alpha=alpha,
        cmap="coolwarm",
        contour_lw=3.0,
    )
else:
    print("Not enough subjects per group for alpha intercept difference plot.")

# D) f2 × group proxy (EXP - CTL on beta_f2)
if len(betas_exp) >= 2 and len(betas_ctl) >= 2:
    Xexp, times_f2g, ch_names_plot_f2g, _ = stack_evokeds(
        [d["betas"]["f2"] for d in betas_exp]
    )
    Xctl, _, _, _ = stack_evokeds([d["betas"]["f2"] for d in betas_ctl])
    diff_map_f2g = Xexp.mean(axis=0) - Xctl.mean(axis=0)
    vmax_f2g = np.nanpercentile(np.abs(diff_map_f2g), 99)

    plot_beta_heatmap_with_clusters(
        diff_map_f2g,
        times_f2g,
        ch_names_plot_f2g,
        title="ALPHA f² × group interaction proxy (β_f² EXP − β_f² CTL)",
        vlim=vmax_f2g,
        clusters=clusters_f2g,
        pvals=pvals_f2g,
        alpha=alpha,
        cmap="coolwarm",
        contour_lw=3.0,
    )
else:
    print("Not enough subjects per group for alpha f2×group plot.")

# -----------------------------
# Topographies for significant clusters
# -----------------------------
topo_template = betas_all[0]["betas"]["f2"]

good_f = np.where(pvals_f < alpha)[0]
for ci in good_f:
    plot_cluster_topomap(
        topo_template, mean_map_f, times, clusters_f[ci],
        title_prefix=f"Topomap: ALPHA β_f (cluster {ci})",
        mask_sensors=True,
    )

good_f2 = np.where(pvals_f2 < alpha)[0]
for ci in good_f2:
    plot_cluster_topomap(
        topo_template, mean_map_f2, times, clusters_f2[ci],
        title_prefix=f"Topomap: ALPHA β_f² (cluster {ci})",
        mask_sensors=True,
    )

if clusters_int is not None and pvals_int is not None and diff_map_int is not None:
    good_int = np.where(pvals_int < alpha)[0]
    for ci in good_int:
        plot_cluster_topomap(
            topo_template, diff_map_int, times_int, clusters_int[ci],
            title_prefix=f"Topomap: ALPHA Intercept EXP−CTL (cluster {ci})",
            mask_sensors=True,
        )

if clusters_f2g is not None and pvals_f2g is not None and diff_map_f2g is not None:
    good_f2g = np.where(pvals_f2g < alpha)[0]
    for ci in good_f2g:
        plot_cluster_topomap(
            topo_template, diff_map_f2g, times_f2g, clusters_f2g[ci],
            title_prefix=f"Topomap: ALPHA β_f² EXP−CTL (cluster {ci})",
            mask_sensors=True,
        )

# =====================================================================
# 4) ALPHA ROI bin plot over CLUSTER CHANNELS from alpha f2 one-sample cluster
# subject-averaged first, feedback (7 QUANTILE bins) x group
# Uses baseline-corrected TFR sequences (4D): sub-XXXX_seq-tfr_bl-tfr.h5
# =====================================================================

import matplotlib.cm as cm
import matplotlib.colors as mcolors

roi_label = "alpha_beta_f2"
n_bins = 7
time_window = None  # e.g. (-0.2, 0.8) or None

# --- pick significant alpha f2 cluster (from betas) ---
good = np.where(pvals_f2 < alpha)[0]
if len(good) == 0:
    raise RuntimeError(f"No significant clusters for {roi_label} at alpha={alpha}.")
if len(good) > 1:
    print(f"Warning: {len(good)} significant clusters for {roi_label}; using smallest p.")
ci = good[np.argmin(pvals_f2[good])]

mask_tc = clusters_f2[ci]          # (n_times, n_ch) from alpha betas
roi_ch_mask = mask_tc.any(axis=0)
if not np.any(roi_ch_mask):
    raise RuntimeError("Cluster channel mask is empty (unexpected).")

roi_ch_names = np.array(topo_template.ch_names)[roi_ch_mask]
print(f"ROI: ALPHA f2 cluster {ci} p={pvals_f2[ci]:.4f}, n_channels={len(roi_ch_names)}")

def truncated_cmap(cmap_name="plasma", minval=0.05, maxval=0.8, n=256):
    base = cm.get_cmap(cmap_name)
    new_colors = base(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list("trunc_plasma", new_colors)

cmap = truncated_cmap("plasma", 0.05, 0.8)

# -------------------------------------------------
# 1) Global quantile edges for f (common across subjects/groups)
#    Pull from TFR metadata (same df_seq)
# -------------------------------------------------
all_f = []
for r in index.itertuples(index=False):
    tfr = mne.time_frequency.read_tfrs(r.tfr_epochs_path, verbose=False)[0]
    meta = tfr.metadata
    if meta is None or "f" not in meta.columns:
        raise RuntimeError("Expected column 'f' in TFR.metadata for all subjects.")
    all_f.append(meta["f"].to_numpy(dtype=float))
all_f = np.concatenate(all_f)

q = np.linspace(0, 1, n_bins + 1)
edges = np.quantile(all_f, q)
if len(np.unique(edges)) < len(edges):
    rng = np.random.RandomState(0)
    jitter = 1e-12 * rng.randn(all_f.size)
    edges = np.quantile(all_f + jitter, q)
edges = np.unique(edges)
if len(edges) < 3:
    raise RuntimeError("Quantile binning failed: too few unique bin edges.")
if len(edges) != (n_bins + 1):
    print(f"Warning: requested {n_bins} bins but got {len(edges)-1} after handling ties.")

# -------------------------------------------------
# 2) Per-subject bin means over ROI channels, using ALPHA power
# -------------------------------------------------
sub_bin_rows = []
times_roi = None

for r in index.itertuples(index=False):
    tfr = mne.time_frequency.read_tfrs(r.tfr_epochs_path, verbose=False)[0]

    if time_window is not None:
        tfr = tfr.copy().crop(tmin=time_window[0], tmax=time_window[1])

    # pick ROI channels (intersection)
    present = [ch for ch in roi_ch_names if ch in tfr.ch_names]
    if len(present) == 0:
        continue
    tfr = tfr.copy().pick_channels(present)

    # pick alpha freqs and collapse freqs -> (n_seq, n_ch, n_time)
    fmask = (tfr.freqs >= ALPHA[0]) & (tfr.freqs <= ALPHA[1])
    if not np.any(fmask):
        raise RuntimeError(f"No alpha freqs found in TFR for {r.tfr_epochs_path}")

    # data: (n_seq, n_ch, n_freq, n_time) -> mean over freq -> (n_seq, n_ch, n_time)
    data_alpha = tfr.data[:, :, fmask, :].mean(axis=2)

    # ROI average across channels -> (n_seq, n_time)
    y = data_alpha.mean(axis=1)
    times_roi = tfr.times

    meta = tfr.metadata.reset_index(drop=True)
    if "f" not in meta.columns or "group" not in meta.columns:
        raise RuntimeError("Expected columns 'f' and 'group' in TFR.metadata.")

    fvals = meta["f"].to_numpy(dtype=float)
    fbin = pd.cut(fvals, bins=edges, include_lowest=True, duplicates="drop")

    group = str(meta["group"].iloc[0])
    sid = int(meta["id"].iloc[0]) if "id" in meta.columns else int(r.id)

    for b in fbin.categories:
        sel = np.asarray(fbin == b)
        if not np.any(sel):
            continue
        sub_bin_rows.append(dict(id=sid, group=group, f_bin=b, y=y[sel].mean(axis=0)))

df_roi = pd.DataFrame(sub_bin_rows)
if len(df_roi) == 0:
    raise RuntimeError("No ROI alpha data computed. Check ROI channels and file paths.")

bins = sorted(df_roi["f_bin"].unique(), key=lambda itv: itv.left)

# -------------------------------------------------
# 3) Color mapping + labels
# -------------------------------------------------
bin_mid = {b: 0.5 * (b.left + b.right) for b in bins}
bin_dist = {b: abs(bin_mid[b]) for b in bins}
vals = np.array([bin_dist[b] for b in bins], dtype=float)
norm = mcolors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
bin_label = {b: f"bin{k} (mid={bin_mid[b]:.2f})" for k, b in enumerate(bins, start=1)}

# -------------------------------------------------
# 4) Group mean ± SEM across SUBJECTS (per bin)
# -------------------------------------------------
groups = ["experimental", "control"]
mean_tc = {g: {} for g in groups}
sem_tc = {g: {} for g in groups}
n_subs = {g: {} for g in groups}

for g in groups:
    df_g = df_roi[df_roi["group"] == g]
    for b in bins:
        df_gb = df_g[df_g["f_bin"] == b]
        if len(df_gb) == 0:
            continue
        ys = np.stack(df_gb["y"].values, axis=0)
        mean_tc[g][b] = ys.mean(axis=0)
        sem_tc[g][b] = (
            ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])
            if ys.shape[0] > 1
            else np.zeros_like(ys[0])
        )
        n_subs[g][b] = ys.shape[0]

# -------------------------------------------------
# 5) Plot ROI bin curves
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)

for ax, g in zip(axes, groups):
    for b in bins:
        if b not in mean_tc[g]:
            continue
        m = mean_tc[g][b]
        s = sem_tc[g][b]
        color = cmap(norm(bin_dist[b]))

        ax.plot(
            times_roi, m, color=color,
            label=f"{bin_label[b]} (n={n_subs[g][b]})",
            linewidth=1.8,
        )
        ax.fill_between(times_roi, m - s, m + s, color=color, alpha=0.25)

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title(f"ALPHA power ROI = f² cluster channels (p={pvals_f2[ci]:.3f}) — {g}")
    ax.set_xlabel("Time (s)")

axes[0].set_ylabel("Theta power (dB, logratio BL)")

handles, labels = axes[1].get_legend_handles_labels()
seen = set()
uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
handles_u, labels_u = zip(*uniq) if len(uniq) else ([], [])

ncol = min(4, max(1, len(labels_u)))
fig.legend(
    handles_u,
    labels_u,
    loc="lower center",
    ncol=ncol,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()
