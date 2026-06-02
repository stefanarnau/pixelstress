# -----------------------------------------------------------------------------
# Fast cluster correction for EEG main effects of within-subject predictors
#
# Target:
#   Main effect of f  OR  main effect of f2
#
# Workflow:
# 1) For each subject x electrode:
#       measure ~ f + f2 + mean_trial_difficulty_c + half
#    Extract subject's slope for SLOPE_TERM.
#
# 2) For each electrode:
#       one-sample t-test of slope_beta against 0
#
# 3) Cluster-correct electrode t-map by sign-flipping subject slope maps.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import ttest_1samp
import statsmodels.formula.api as smf
import mne
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/cluster_subject_slopes_main_effects/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"

MEASURE = "cnv_mean"

# Test main effect of "f" or "f2"
SLOPE_TERM = "f2"

N_PERM = 500
RANDOM_SEED = 123

GROUP_ORDER = ["control", "experimental"]
MONTAGE_NAME = "standard_1020"

# Approximately p < .05 two-sided for N~70
CLUSTER_T_THRESHOLD = 2.0

MIN_OBS_PER_SUBJECT_ELECTRODE = 20
RIDGE_ALPHA = 0.1


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def prepare_data(df, measure):
    df = df.copy()

    df["id"] = df["id"].astype(str)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER)
    df["half"] = pd.Categorical(df["half"])
    df["ch_name"] = df["ch_name"].astype(str)

    for col in [measure, "f", "mean_trial_difficulty"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    seq_meta = (
        df[
            [
                "id",
                "group",
                "block_nr",
                "sequence_nr",
                "half",
                "mean_trial_difficulty",
                "f",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    seq_meta["f_c"] = seq_meta["f"] - seq_meta["f"].mean()
    seq_meta["f2"] = seq_meta["f_c"] ** 2 - np.mean(seq_meta["f_c"] ** 2)

    seq_meta["mean_trial_difficulty_c"] = (
        seq_meta["mean_trial_difficulty"]
        - seq_meta["mean_trial_difficulty"].mean()
    )

    d = df[
        [
            "id",
            "block_nr",
            "sequence_nr",
            "ch_name",
            measure,
        ]
    ].copy()

    d = d.rename(columns={measure: "y"})

    d = seq_meta.merge(
        d,
        on=["id", "block_nr", "sequence_nr"],
        how="inner",
    )

    d = d.dropna(
        subset=[
            "id",
            "group",
            "half",
            "f",
            "f2",
            "mean_trial_difficulty_c",
            "ch_name",
            "y",
        ]
    ).copy()

    return d


def make_info(ch_names):
    info = mne.create_info(ch_names=list(ch_names), sfreq=500, ch_types="eeg")
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    info.set_montage(montage, on_missing="ignore", match_case=False)
    return info


def estimate_subject_slopes(d, slope_term, ridge_alpha=1.0):
    """
    Fits one ridge regression per subject x electrode:

        y ~ f + f2 + mean_trial_difficulty_c + half

    Returns:
        one row per subject x electrode with ridge-estimated slope_term beta.

    Predictors are standardized before ridge.
    y is not standardized.
    Returned beta is in y-units per 1 SD predictor change.
    """
    rows = []
    beta_col = f"{slope_term}_beta"

    group_lookup = (
        d[["id", "group"]]
        .drop_duplicates()
        .set_index("id")["group"]
        .to_dict()
    )

    grouped = d.groupby(["id", "ch_name"], observed=True)

    for (subj, ch), ds in grouped:
        ds = ds.dropna(
            subset=[
                "y",
                "f",
                "f2",
                "mean_trial_difficulty_c",
                "half",
            ]
        ).copy()

        if len(ds) < MIN_OBS_PER_SUBJECT_ELECTRODE:
            continue

        if ds["f"].nunique() < 3 or ds["f2"].nunique() < 2:
            continue

        try:
            X = pd.DataFrame(
                {
                    "f": ds["f"].astype(float),
                    "f2": ds["f2"].astype(float),
                    "mean_trial_difficulty_c": ds["mean_trial_difficulty_c"].astype(float),
                }
            )

            half_dummies = pd.get_dummies(
                ds["half"],
                prefix="half",
                drop_first=True,
                dtype=float,
            )

            X = pd.concat([X, half_dummies], axis=1)

            y = ds["y"].astype(float).to_numpy()

            model = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                Ridge(alpha=ridge_alpha, fit_intercept=True),
            )

            model.fit(X, y)

            ridge = model.named_steps["ridge"]
            coef = pd.Series(ridge.coef_, index=X.columns)

            beta = coef.get(slope_term, np.nan)

            rows.append(
                {
                    "id": subj,
                    "group": group_lookup[subj],
                    "ch_name": ch,
                    beta_col: beta,
                    "n_obs": len(ds),
                    "ridge_alpha": ridge_alpha,
                    "predictor_scale": "standardized_X_unstandardized_y",
                }
            )

        except Exception:
            rows.append(
                {
                    "id": subj,
                    "group": group_lookup.get(subj, np.nan),
                    "ch_name": ch,
                    beta_col: np.nan,
                    "n_obs": len(ds),
                    "ridge_alpha": ridge_alpha,
                    "predictor_scale": "standardized_X_unstandardized_y",
                }
            )

    slopes = pd.DataFrame(rows)
    slopes["group"] = pd.Categorical(slopes["group"], categories=GROUP_ORDER)

    return slopes


def electrode_main_effect_tmap(slopes, ch_names, slope_term):
    """
    For each electrode:
        one-sample t-test of subject slopes against zero.
    """
    rows = []
    beta_col = f"{slope_term}_beta"

    for ch in ch_names:
        ds = slopes[slopes["ch_name"] == ch].dropna(subset=[beta_col]).copy()
        vals = ds[beta_col].to_numpy(dtype=float)

        if len(vals) < 5:
            rows.append(
                {
                    "ch_name": ch,
                    "mean_slope": np.nan,
                    "t": np.nan,
                    "p": np.nan,
                    "n_subjects": len(vals),
                }
            )
            continue

        tval, pval = ttest_1samp(vals, popmean=0.0, nan_policy="omit")

        rows.append(
            {
                "ch_name": ch,
                "mean_slope": float(np.nanmean(vals)),
                "t": float(tval),
                "p": float(pval),
                "n_subjects": len(vals),
            }
        )

    return pd.DataFrame(rows)


def find_clusters(stat_vals, adjacency, threshold):
    """
    Positive and negative clusters separately.
    Cluster mass = sum(abs(t)) inside cluster.
    """
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


def sign_flip_subject_slopes(slopes, rng):
    """
    Randomly multiply each subject's full electrode slope map by +1 or -1.

    This tests the null:
        mean subject slope = 0

    while preserving:
        - spatial covariance
        - subject-level structure
        - electrode-wise dependencies
    """
    out = slopes.copy()
    beta_cols = [c for c in out.columns if c.endswith("_beta")]

    subj_ids = sorted(out["id"].dropna().unique())
    signs = pd.DataFrame(
        {
            "id": subj_ids,
            "flip": rng.choice([-1.0, 1.0], size=len(subj_ids)),
        }
    )

    out = out.merge(signs, on="id", how="left")

    for col in beta_cols:
        out[col] = out[col] * out["flip"]

    out = out.drop(columns=["flip"])

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

    mask = sig_mask[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    im, _ = mne.viz.plot_topomap(
        vals,
        info,
        axes=ax,
        show=False,
        cmap="RdBu_r",
        sensors=True,
        contours=6,
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

d = prepare_data(df_raw, MEASURE)

ch_names = sorted(d["ch_name"].dropna().unique().tolist())
info = make_info(ch_names)

adjacency, adjacency_ch_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")

if list(adjacency_ch_names) != list(ch_names):
    order = [list(adjacency_ch_names).index(ch) for ch in ch_names]
    adjacency = adjacency[order][:, order]

adjacency = sparse.csr_matrix(adjacency)

print("Measure:", MEASURE)
print("Main-effect subject slope:", SLOPE_TERM)
print("Channels:", len(ch_names))
print("Subjects:", d["id"].nunique())
print("Rows:", len(d))
print("Permutations:", N_PERM)

print("\nEstimating subject-level slopes...")
slopes = estimate_subject_slopes(
    d,
    SLOPE_TERM,
    ridge_alpha=RIDGE_ALPHA,
)

slopes.to_csv(
    PATH_OUT / f"{MEASURE}_{SLOPE_TERM}_main_effect_subject_electrode_slopes.csv",
    index=False,
)

print("Slope rows:", len(slopes))
print("Subjects with slopes:", slopes["id"].nunique())

print("\nComputing observed one-sample electrode t-map...")
obs_tmap = electrode_main_effect_tmap(slopes, ch_names, SLOPE_TERM)

obs_tmap.to_csv(
    PATH_OUT / f"{MEASURE}_{SLOPE_TERM}_main_effect_observed_tmap.csv",
    index=False,
)

obs_t = obs_tmap["t"].to_numpy(dtype=float)
obs_clusters = find_clusters(obs_t, adjacency, CLUSTER_T_THRESHOLD)

print("Observed clusters:", len(obs_clusters))

rng = np.random.default_rng(RANDOM_SEED)
max_cluster_masses = np.zeros(N_PERM)

print("\nRunning sign-flip permutations...")
for i in range(N_PERM):
    if (i + 1) % 500 == 0:
        print(f"Permutation {i + 1}/{N_PERM}")

    slopes_perm = sign_flip_subject_slopes(slopes, rng)
    perm_tmap = electrode_main_effect_tmap(slopes_perm, ch_names, SLOPE_TERM)
    perm_t = perm_tmap["t"].to_numpy(dtype=float)

    perm_clusters = find_clusters(perm_t, adjacency, CLUSTER_T_THRESHOLD)

    if len(perm_clusters) == 0:
        max_cluster_masses[i] = 0.0
    else:
        max_cluster_masses[i] = max(cl["mass"] for cl in perm_clusters)

np.save(
    PATH_OUT / f"{MEASURE}_{SLOPE_TERM}_main_effect_max_cluster_null.npy",
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
    PATH_OUT / f"{MEASURE}_{SLOPE_TERM}_main_effect_cluster_corrected_results.csv",
    index=False,
)

plot_tmap(
    tmap=obs_tmap,
    ch_names=ch_names,
    info=info,
    path_out=PATH_OUT,
    title=f"{MEASURE}_{SLOPE_TERM}_main_effect_tmap",
    clusters=obs_clusters,
    cluster_pvals=cluster_df["p_cluster"].to_numpy()
    if not cluster_df.empty else np.array([]),
    alpha=0.05,
)

print("\nCluster-corrected main-effect results:")
print(cluster_df)

print("\nFinished.")
print("Saved to:", PATH_OUT)