# -----------------------------------------------------------------------------
# Cluster correction for EEG main effect of GROUP
#
# Workflow:
# 1) For each subject x electrode:
#       average EEG measure across all sequences
#
# 2) For each electrode:
#       subject_mean ~ group
#
# 3) Cluster-correct electrode t-map by shuffling group labels across subjects.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import statsmodels.formula.api as smf
import mne
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/cluster_group_main_effects/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"

MEASURE = "theta_flat"

N_PERM = 500
RANDOM_SEED = 123

GROUP_ORDER = ["control", "experimental"]
MONTAGE_NAME = "standard_1020"

CLUSTER_T_THRESHOLD = 1.8


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def prepare_subject_electrode_means(df, measure):
    df = df.copy()

    df["id"] = df["id"].astype(str)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER)
    df["ch_name"] = df["ch_name"].astype(str)
    df[measure] = pd.to_numeric(df[measure], errors="coerce")

    means = (
        df.groupby(["id", "group", "ch_name"], observed=True, as_index=False)[measure]
        .mean()
        .rename(columns={measure: "y_mean"})
        .dropna(subset=["y_mean", "group", "ch_name"])
    )

    means["group"] = pd.Categorical(means["group"], categories=GROUP_ORDER)

    return means


def make_info(ch_names):
    info = mne.create_info(ch_names=list(ch_names), sfreq=500, ch_types="eeg")
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    info.set_montage(montage, on_missing="ignore", match_case=False)
    return info


def electrode_group_tmap(means, ch_names):
    rows = []

    for ch in ch_names:
        ds = means[means["ch_name"] == ch].dropna(subset=["y_mean", "group"]).copy()

        n_control = int((ds["group"] == "control").sum())
        n_exp = int((ds["group"] == "experimental").sum())

        if n_control < 3 or n_exp < 3:
            rows.append(
                {
                    "ch_name": ch,
                    "beta_group": np.nan,
                    "t": np.nan,
                    "p": np.nan,
                    "n_control": n_control,
                    "n_experimental": n_exp,
                }
            )
            continue

        try:
            fit = smf.ols("y_mean ~ group", data=ds).fit()
            term = "group[T.experimental]"

            rows.append(
                {
                    "ch_name": ch,
                    "beta_group": fit.params.get(term, np.nan),
                    "t": fit.tvalues.get(term, np.nan),
                    "p": fit.pvalues.get(term, np.nan),
                    "n_control": n_control,
                    "n_experimental": n_exp,
                }
            )

        except Exception:
            rows.append(
                {
                    "ch_name": ch,
                    "beta_group": np.nan,
                    "t": np.nan,
                    "p": np.nan,
                    "n_control": n_control,
                    "n_experimental": n_exp,
                }
            )

    return pd.DataFrame(rows)


def find_clusters(stat_vals, adjacency, threshold):
    stat_vals = np.asarray(stat_vals, dtype=float)
    finite = np.isfinite(stat_vals)

    clusters = []

    for sign_name, mask in [
        ("positive", finite & (stat_vals >= threshold)),
        ("negative", finite & (stat_vals <= -threshold)),
    ]:
        if not np.any(mask):
            continue

        idx = np.where(mask)[0]
        sub_adj = adjacency[idx][:, idx]

        n_comp, labels = connected_components(
            sub_adj,
            directed=False,
            return_labels=True,
        )

        for k in range(n_comp):
            members_local = np.where(labels == k)[0]
            members = idx[members_local]
            mass = float(np.sum(np.abs(stat_vals[members])))

            clusters.append(
                {
                    "sign": sign_name,
                    "indices": members,
                    "mass": mass,
                    "size": int(len(members)),
                }
            )

    return clusters


def permute_group_labels(means, rng):
    subj_group = (
        means[["id", "group"]]
        .drop_duplicates()
        .sort_values("id")
        .reset_index(drop=True)
    )

    shuffled = subj_group.copy()
    shuffled["group"] = rng.permutation(shuffled["group"].values)

    out = means.drop(columns=["group"]).merge(
        shuffled,
        on="id",
        how="left",
    )

    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER)

    return out


def plot_tmap(
    tmap,
    ch_names,
    info,
    path_out,
    title,
    clusters=None,
    cluster_pvals=None,
    alpha=0.05,
):
    vals = (
        tmap.set_index("ch_name")
        .reindex(ch_names)["t"]
        .to_numpy(dtype=float)
    )

    vmax = np.nanmax(np.abs(vals))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    vlim = (-vmax, vmax)

    sig_mask = np.zeros(len(ch_names), dtype=bool)
    sig_cluster_labels = []

    if clusters is not None and cluster_pvals is not None:
        for k, (cl, pval) in enumerate(zip(clusters, cluster_pvals)):
            if np.isfinite(pval) and pval < alpha:
                sig_mask[cl["indices"]] = True
                sig_cluster_labels.append(
                    f"cluster {k}: p={pval:.3f}, {cl['sign']}, n={cl['size']}"
                )

    mask = sig_mask

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    im, _ = mne.viz.plot_topomap(
        vals,
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        sensors=True,
        contours=0,
        vlim=vlim,
        mask=mask,
        mask_params=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            linewidth=1.5,
            markersize=9,
        ),
    )

    if sig_cluster_labels:
        subtitle = "\n" + " | ".join(sig_cluster_labels)
    else:
        subtitle = "\nNo cluster-corrected significant electrodes"

    ax.set_title(title + subtitle, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("t")

    safe_title = (
        title.replace(" ", "_")
        .replace(":", "")
        .replace("/", "_")
        .replace("[", "")
        .replace("]", "")
    )

    fig.savefig(
        path_out / f"{safe_title}_tmap_with_cluster_mask.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
print("Loading data...")
df_raw = pd.read_csv(FILE_IN)

means = prepare_subject_electrode_means(df_raw, MEASURE)

ch_names = sorted(means["ch_name"].dropna().unique().tolist())
info = make_info(ch_names)

adjacency, adjacency_ch_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")

if list(adjacency_ch_names) != list(ch_names):
    order = [list(adjacency_ch_names).index(ch) for ch in ch_names]
    adjacency = adjacency[order][:, order]

adjacency = sparse.csr_matrix(adjacency)

print("Measure:", MEASURE)
print("Channels:", len(ch_names))
print("Subjects:", means["id"].nunique())
print("Permutations:", N_PERM)

means.to_csv(
    PATH_OUT / f"{MEASURE}_subject_electrode_means.csv",
    index=False,
)

print("\nComputing observed group t-map...")
obs_tmap = electrode_group_tmap(means, ch_names)

obs_tmap.to_csv(
    PATH_OUT / f"{MEASURE}_group_main_effect_observed_tmap.csv",
    index=False,
)

obs_t = obs_tmap["t"].to_numpy(dtype=float)
obs_clusters = find_clusters(obs_t, adjacency, CLUSTER_T_THRESHOLD)

print("Observed clusters:", len(obs_clusters))

rng = np.random.default_rng(RANDOM_SEED)
max_cluster_masses = np.zeros(N_PERM)

print("\nRunning group-label permutations...")
for i in range(N_PERM):
    if (i + 1) % 500 == 0:
        print(f"Permutation {i + 1}/{N_PERM}")

    means_perm = permute_group_labels(means, rng)
    perm_tmap = electrode_group_tmap(means_perm, ch_names)
    perm_t = perm_tmap["t"].to_numpy(dtype=float)

    perm_clusters = find_clusters(perm_t, adjacency, CLUSTER_T_THRESHOLD)

    if len(perm_clusters) == 0:
        max_cluster_masses[i] = 0.0
    else:
        max_cluster_masses[i] = max(cl["mass"] for cl in perm_clusters)

np.save(
    PATH_OUT / f"{MEASURE}_group_main_effect_max_cluster_null.npy",
    max_cluster_masses,
)

cluster_rows = []

for k, cl in enumerate(obs_clusters):
    p_cluster = (1 + np.sum(max_cluster_masses >= cl["mass"])) / (N_PERM + 1)

    cluster_rows.append(
        {
            "cluster": k,
            "sign": cl["sign"],
            "size": cl["size"],
            "mass": cl["mass"],
            "p_cluster": p_cluster,
            "channels": ",".join([ch_names[i] for i in cl["indices"]]),
        }
    )

cluster_df = pd.DataFrame(cluster_rows)

cluster_df.to_csv(
    PATH_OUT / f"{MEASURE}_group_main_effect_cluster_corrected_results.csv",
    index=False,
)

plot_tmap(
    tmap=obs_tmap,
    ch_names=ch_names,
    info=info,
    path_out=PATH_OUT,
    title=f"{MEASURE}_group_main_effect_tmap",
    clusters=obs_clusters,
    cluster_pvals=cluster_df["p_cluster"].to_numpy()
    if not cluster_df.empty else np.array([]),
    alpha=0.05,
)

print("\nCluster-corrected group main-effect results:")
print(cluster_df)

print("\nFinished.")
print("Saved to:", PATH_OUT)