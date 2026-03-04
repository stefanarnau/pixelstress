
# step2_tf_eft_cluster_tests.py
#
# Cluster permutation tests in TF space (freq x time x channels) using saved EFT betas (.npz)
# Tests:
#   1) one-sample: beta_f   (all subjects)
#   2) one-sample: beta_f2  (all subjects)
#   3) two-sample: Intercept (EXP vs CTL)  [group main effect proxy]
#   4) two-sample: beta_f2  (EXP vs CTL)   [interaction proxy]
#
# Input requirements:
#   - subject_index.csv has column: tf_eft_betas_path
#   - each .npz contains:
#       betas: (n_reg, n_ch, n_freq, n_time)
#       regressor_names: (n_reg,)
#       ch_names: (n_ch,)
#       freqs: (n_freq,)
#       times: (n_time,)
#
# Output:
#   - prints #significant clusters (p < alpha) per test
#   - stores results dicts you can later use for plotting/masking

import numpy as np
import pandas as pd
import mne
import scipy.stats as stats
from pathlib import Path


# -----------------------------
# Paths / IO
# -----------------------------
path_out = Path("/mnt/data_dump/pixelstress/3_sequence_data_plus_fitted/")
index = pd.read_csv(path_out / "subject_index.csv")

# -----------------------------
# Params
# -----------------------------
alpha = 0.05
n_perm = 2000
cluster_alpha = 0.05  # cluster-forming threshold alpha (two-sided)
tail = 0              # two-sided

MIN_PER_GROUP = 5


# -----------------------------
# Helpers
# -----------------------------
def load_eft_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=False)
    betas = z["betas"]  # (n_reg, n_ch, n_freq, n_time)
    regressor_names = [str(x) for x in z["regressor_names"].tolist()]
    ch_names = [str(x) for x in z["ch_names"].tolist()]
    freqs = z["freqs"].astype(float)
    times = z["times"].astype(float)
    return betas, regressor_names, ch_names, freqs, times


def get_reg_index(regressor_names, target):
    if target not in regressor_names:
        raise RuntimeError(f"Regressor '{target}' not found. Available: {regressor_names}")
    return regressor_names.index(target)


def assert_same_grid(ref_freqs, ref_times, freqs, times, tol=1e-9):
    if len(ref_freqs) != len(freqs) or np.max(np.abs(ref_freqs - freqs)) > tol:
        raise RuntimeError("Mismatch in freqs across subjects.")
    if len(ref_times) != len(times) or np.max(np.abs(ref_times - times)) > tol:
        raise RuntimeError("Mismatch in times across subjects.")


def build_3d_adjacency(info, n_freq, n_time):
    """
    Feature order must match flattening order below.
    We'll flatten as (freq, time, ch) with ch fastest, then time, then freq.
    => combine_adjacency(n_freq, n_time, adjacency_ch)
    """
    adjacency_ch, _ = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    adj = mne.stats.combine_adjacency(n_freq, n_time, adjacency_ch)
    return adj


def flatten_ftc(beta_ch_freq_time):
    """
    Input: (n_ch, n_freq, n_time)
    We reorder to (n_freq, n_time, n_ch) then flatten in C-order.
    """
    x = np.transpose(beta_ch_freq_time, (1, 2, 0))  # (freq, time, ch)
    return x.reshape(-1)


def cluster_threshold_1samp(n_obs, cluster_alpha=0.05, tail=0):
    df = n_obs - 1
    if tail == 0:
        return stats.t.ppf(1 - cluster_alpha / 2, df)
    elif tail == 1:
        return stats.t.ppf(1 - cluster_alpha, df)
    else:  # tail == -1
        return stats.t.ppf(cluster_alpha, df)


def cluster_threshold_2samp(n1, n2, cluster_alpha=0.05, tail=0):
    df = n1 + n2 - 2
    if tail == 0:
        return stats.t.ppf(1 - cluster_alpha / 2, df)
    elif tail == 1:
        return stats.t.ppf(1 - cluster_alpha, df)
    else:
        return stats.t.ppf(cluster_alpha, df)


def run_1samp_cluster(X, adjacency, n_perm, alpha, cluster_alpha, tail):
    """
    X: (n_obs, n_features)
    """
    thr = cluster_threshold_1samp(X.shape[0], cluster_alpha=cluster_alpha, tail=tail)
    T_obs, clusters, pvals, H0 = mne.stats.permutation_cluster_1samp_test(
        X,
        n_permutations=n_perm,
        threshold=thr,
        tail=tail,
        adjacency=adjacency,
        out_type="mask",
        n_jobs=-1,
        seed=0,
    )
    good = np.where(pvals < alpha)[0]
    return dict(T_obs=T_obs, clusters=clusters, pvals=pvals, H0=H0, good=good, threshold=thr)


def run_2samp_cluster(X1, X2, adjacency, n_perm, alpha, cluster_alpha, tail):
    """
    X1, X2: (n_obs, n_features)
    """
    thr = cluster_threshold_2samp(X1.shape[0], X2.shape[0], cluster_alpha=cluster_alpha, tail=tail)
    T_obs, clusters, pvals, H0 = mne.stats.permutation_cluster_test(
        [X1, X2],
        n_permutations=n_perm,
        threshold=thr,
        tail=tail,
        adjacency=adjacency,
        out_type="mask",
        n_jobs=-1,
        seed=0,
    )
    good = np.where(pvals < alpha)[0]
    return dict(T_obs=T_obs, clusters=clusters, pvals=pvals, H0=H0, good=good, threshold=thr)


def reshape_cluster_mask(mask_1d, n_freq, n_time, n_ch):
    """
    mask_1d: boolean (n_features,)
    Return: boolean (n_freq, n_time, n_ch) matching our flatten order.
    """
    return mask_1d.reshape(n_freq, n_time, n_ch)


# -----------------------------
# Load TF EFT betas across subjects
# -----------------------------
rows = []
ref_freqs = ref_times = None
ref_ch_names = None
regressor_names_ref = None

# We also load one info object for channel adjacency (saved earlier by you)
info_tf_path = path_out / "info_tf.fif"
if not info_tf_path.exists():
    raise RuntimeError(f"Missing {info_tf_path}. You saved it in step 1; re-run that part.")
info_tf = mne.io.read_info(info_tf_path)

for r in index.itertuples(index=False):
    if not hasattr(r, "tf_eft_betas_path") or pd.isna(r.tf_eft_betas_path):
        raise RuntimeError("subject_index.csv missing column tf_eft_betas_path (or has NaNs).")

    betas, regressor_names, ch_names, freqs, times = load_eft_npz(r.tf_eft_betas_path)

    if ref_freqs is None:
        ref_freqs = freqs
        ref_times = times
        ref_ch_names = ch_names
        regressor_names_ref = regressor_names
    else:
        assert_same_grid(ref_freqs, ref_times, freqs, times)
        if ch_names != ref_ch_names:
            raise RuntimeError("Mismatch in channel names/order across subjects.")
        if regressor_names != regressor_names_ref:
            raise RuntimeError("Mismatch in regressor_names across subjects.")

    rows.append(dict(id=int(r.id), group=str(r.group), betas=betas))

print(f"Loaded EFT TF betas for n={len(rows)} subjects.")
print(f"Regressors available: {regressor_names_ref}")
print(f"TF grid: n_ch={len(ref_ch_names)} | n_freq={len(ref_freqs)} | n_time={len(ref_times)}")


# -----------------------------
# Build 3D adjacency for (freq, time, ch) features
# -----------------------------
n_ch = len(ref_ch_names)
n_freq = len(ref_freqs)
n_time = len(ref_times)

adj_3d = build_3d_adjacency(info_tf, n_freq=n_freq, n_time=n_time)


# -----------------------------
# Build design arrays (flattened)
# -----------------------------
def build_X_for_reg(rows, reg_name):
    ri = get_reg_index(regressor_names_ref, reg_name)
    X = []
    for d in rows:
        # d["betas"]: (n_reg, n_ch, n_freq, n_time)
        beta_ch_freq_time = d["betas"][ri, :, :, :]  # (n_ch, n_freq, n_time)
        X.append(flatten_ftc(beta_ch_freq_time))
    return np.asarray(X)  # (n_subj, n_features)


rows_exp = [d for d in rows if d["group"] == "experimental"]
rows_ctl = [d for d in rows if d["group"] == "control"]

print(f"Groups: exp n={len(rows_exp)} | ctl n={len(rows_ctl)}")


# -----------------------------
# 1) One-sample cluster: beta_f
# -----------------------------
X_f = build_X_for_reg(rows, "f")
res_f = run_1samp_cluster(X_f, adj_3d, n_perm=n_perm, alpha=alpha, cluster_alpha=cluster_alpha, tail=tail)
print(f"[TF β_f] threshold={res_f['threshold']:.3f} | sig clusters={len(res_f['good'])} (alpha={alpha})")


# -----------------------------
# 2) One-sample cluster: beta_f2
# -----------------------------
X_f2 = build_X_for_reg(rows, "f2")
res_f2 = run_1samp_cluster(X_f2, adj_3d, n_perm=n_perm, alpha=alpha, cluster_alpha=cluster_alpha, tail=tail)
print(f"[TF β_f2] threshold={res_f2['threshold']:.3f} | sig clusters={len(res_f2['good'])} (alpha={alpha})")


# -----------------------------
# 3) Two-sample cluster: Intercept (EXP vs CTL)
# -----------------------------
res_int = None
if len(rows_exp) < MIN_PER_GROUP or len(rows_ctl) < MIN_PER_GROUP:
    print("[TF Intercept EXP vs CTL] Too few subjects per group; skipping.")
else:
    X_int_exp = build_X_for_reg(rows_exp, "Intercept")
    X_int_ctl = build_X_for_reg(rows_ctl, "Intercept")
    res_int = run_2samp_cluster(X_int_exp, X_int_ctl, adj_3d, n_perm=n_perm, alpha=alpha, cluster_alpha=cluster_alpha, tail=tail)
    print(f"[TF Intercept EXP vs CTL] threshold={res_int['threshold']:.3f} | sig clusters={len(res_int['good'])} (alpha={alpha})")


# -----------------------------
# 4) Two-sample cluster: f2 x group proxy (beta_f2 EXP vs CTL)
# -----------------------------
res_f2g = None
if len(rows_exp) < MIN_PER_GROUP or len(rows_ctl) < MIN_PER_GROUP:
    print("[TF β_f2 EXP vs CTL] Too few subjects per group; skipping.")
else:
    X_f2_exp = build_X_for_reg(rows_exp, "f2")
    X_f2_ctl = build_X_for_reg(rows_ctl, "f2")
    res_f2g = run_2samp_cluster(X_f2_exp, X_f2_ctl, adj_3d, n_perm=n_perm, alpha=alpha, cluster_alpha=cluster_alpha, tail=tail)
    print(f"[TF β_f2 EXP vs CTL] threshold={res_f2g['threshold']:.3f} | sig clusters={len(res_f2g['good'])} (alpha={alpha})")


# -----------------------------
# Optional: convert significant cluster masks back to (freq, time, ch)
# -----------------------------
def extract_sig_masks(res, label):
    if res is None:
        return []
    masks = []
    for ci in res["good"]:
        mask_1d = res["clusters"][ci]           # boolean (n_features,)
        mask_ftc = reshape_cluster_mask(mask_1d, n_freq, n_time, n_ch)
        masks.append((ci, float(res["pvals"][ci]), mask_ftc))
    print(f"[{label}] extracted {len(masks)} significant masks in (freq,time,ch).")
    return masks

sig_masks_f   = extract_sig_masks(res_f,   "TF β_f")
sig_masks_f2  = extract_sig_masks(res_f2,  "TF β_f2")
sig_masks_int = extract_sig_masks(res_int, "TF Intercept EXP-CTL")
sig_masks_f2g = extract_sig_masks(res_f2g, "TF β_f2 EXP-CTL")

# At this point you have:
#  - ref_freqs, ref_times, ref_ch_names
#  - sig_masks_* lists: (cluster_index, p_value, mask_ftc)
# Use these for plotting (e.g., collapse over cluster channels, show TFR masks, etc.).