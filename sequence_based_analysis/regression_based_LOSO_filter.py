# -----------------------------------------------------------------------------
# LOSO supervised spatial filter + final MLM + binned filter-weighted spectra
#
# Overview
# --------
# For one chosen EEG measure and one or more chosen target effects:
#
#   1. Build sequence x electrode matrices from the long summary dataframe
#   2. For each held-out subject (LOSO):
#        - derive an effect-specific multivariate spatial filter on training data
#        - nuisance regression: half + mean_trial_difficulty
#        - within-subject centering
#        - ridge regression across all electrodes simultaneously
#        - apply the filter to held-out sequences
#   3. Fit a final MLM on the out-of-fold component scores
#   4. Load the saved PSD .npz + matching index CSV files
#   5. Apply the mean LOSO filter / mean LOSO Haufe pattern to the PSDs
#   6. Plot binned filter-weighted spectra as a function of f, split by group
#
# Notes
# -----
# - Regression weights are used for scoring.
# - Haufe patterns are for interpretation and spectral visualization.
# - Spectrum plots are descriptive.
# - This script assumes CAR data only.
# -----------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"

PATH_OUT = PATH_IN / "loso_supervised_spatial_filter"
PATH_OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
MEASURE = "beta_flat"   # "exponent", "theta_flat", "alpha_flat", "beta_flat"

TARGET_EFFECTS = [
    "f",
    "f2",
    "group[T.experimental]",
    "group[T.experimental]:f",
    "group[T.experimental]:f2",
]

FINAL_FORMULA = """
score ~ group * f + group * f2
        + mean_trial_difficulty_c + half
"""

FINAL_RE_FORMULAS = [
    "1 + f + f2",
    "1 + f",
    "1",
]

MIN_SUBJECTS = 8
MIN_SEQUENCES_PER_SUBJECT = 8

RIDGE_ALPHA = 1.0

N_JOBS = -1
PARALLEL_BACKEND = "loky"
PARALLEL_VERBOSE = 10

APPLY_QC = True
QC_MIN_R2 = 0.80
QC_MAX_ERROR = 0.30
QC_MIN_EXPONENT = 0.50
QC_MAX_EXPONENT = 3.50

MIN_RETAINED_FILTER_MASS = 0.80

N_BINS_PLOT = 9

# Spectral plotting options
PSD_USE_LOG10 = True
PSD_WEIGHT_SOURCE = "haufe"   # "haufe" or "weights"
PSD_SPECTRAL_WEIGHT_MODE = "squared"   # "abs" or "squared"


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
VALID_MEASURES = ["exponent", "theta_flat", "alpha_flat", "beta_flat"]

VALID_TARGET_EFFECTS = [
    "f",
    "f2",
    "group[T.experimental]",
    "group[T.experimental]:f",
    "group[T.experimental]:f2",
]

SEQ_ID_COLS = ["id", "block_nr", "sequence_nr"]
SEQ_META_COLS = [
    "id",
    "group",
    "block_nr",
    "sequence_nr",
    "half",
    "n_trials",
    "mean_trial_difficulty",
    "mean_trial_difficulty_c",
    "f",
    "f2",
    "mean_rt",
    "mean_log_rt",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_name(text: str) -> str:
    return (
        text.replace("[", "")
        .replace("]", "")
        .replace(":", "x")
        .replace(".", "p")
        .replace("(", "")
        .replace(")", "")
    )


def align_filter_table_to_anchor(
    df_filters: pd.DataFrame,
    electrodes: list[str],
    value_col: str,
) -> pd.DataFrame:
    """
    Align fold-specific filter vectors to a common sign anchor using only
    LOSO fold outputs.
    """
    if df_filters.empty:
        return df_filters.copy()

    wide = (
        df_filters.pivot(index="held_out_subject", columns="electrode", values=value_col)
        .reindex(columns=electrodes)
        .sort_index()
    )

    subjects = wide.index.tolist()
    X = wide.to_numpy(dtype=float)

    if X.shape[0] == 0:
        return df_filters.copy()

    X_aligned = X.copy()
    anchor = X_aligned[0, :].copy()

    for i in range(1, X_aligned.shape[0]):
        xi = X_aligned[i, :].copy()

        valid = np.isfinite(xi) & np.isfinite(anchor)
        if valid.sum() > 1:
            r = np.corrcoef(xi[valid], anchor[valid])[0, 1]
            if np.isfinite(r) and r < 0:
                xi = -xi

        X_aligned[i, :] = xi
        anchor = np.nanmean(X_aligned[: i + 1, :], axis=0)

    wide_aligned = pd.DataFrame(X_aligned, index=subjects, columns=electrodes)
    wide_aligned.index.name = "held_out_subject"

    df_out = (
        wide_aligned
        .reset_index()
        .melt(id_vars="held_out_subject", var_name="electrode", value_name=value_col)
    )

    return df_out


def make_spectral_weights(weight_vec: np.ndarray, mode: str = "squared") -> np.ndarray:
    w = np.asarray(weight_vec, dtype=float)
    w = np.nan_to_num(w, nan=0.0)

    if mode == "abs":
        w_spec = np.abs(w)
    elif mode == "squared":
        w_spec = w ** 2
    else:
        raise ValueError("mode must be 'abs' or 'squared'")

    s = w_spec.sum()
    if not np.isfinite(s) or s == 0:
        raise RuntimeError("Spectral weights sum to zero.")

    return w_spec / s


def fit_mixedlm_with_fallback(df_model: pd.DataFrame, formula: str, re_formulas: list[str]):
    fit = None
    used_re = None
    error_log = []

    for re_formula in re_formulas:
        try:
            model = smf.mixedlm(
                formula=formula,
                data=df_model,
                groups=df_model["id"],
                re_formula=re_formula,
            )

            fit_try = model.fit(
                method="lbfgs",
                reml=False,
                maxiter=4000,
                disp=False,
            )

            if bool(getattr(fit_try, "converged", False)):
                fit = fit_try
                used_re = re_formula
                break
            else:
                error_log.append(f"{re_formula}: converged=False")

        except Exception as exc:
            error_log.append(f"{re_formula}: {exc}")

    return fit, used_re, error_log


def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    qc_mask = (
        df["r2"].ge(QC_MIN_R2)
        & df["error"].le(QC_MAX_ERROR)
        & df["exponent"].between(QC_MIN_EXPONENT, QC_MAX_EXPONENT)
    )
    return df.loc[qc_mask].copy()


def add_centered_predictors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["f"] = pd.to_numeric(df["f"], errors="coerce")
    df["mean_trial_difficulty"] = pd.to_numeric(df["mean_trial_difficulty"], errors="coerce")

    f_mean = df["f"].mean()
    df["f_c"] = df["f"] - f_mean
    df["f2"] = df["f_c"] ** 2 - np.mean(df["f_c"] ** 2)

    df["mean_trial_difficulty_c"] = (
        df["mean_trial_difficulty"] - df["mean_trial_difficulty"].mean()
    )

    return df


def retain_subjects_with_min_sequences(df: pd.DataFrame, min_sequences: int) -> pd.DataFrame:
    seq_counts = (
        df[SEQ_ID_COLS]
        .drop_duplicates()
        .groupby("id", observed=True)
        .size()
        .rename("n_sequences")
        .reset_index()
    )

    keep_ids = seq_counts.loc[seq_counts["n_sequences"] >= min_sequences, "id"].astype(str)
    return df[df["id"].astype(str).isin(keep_ids)].copy()


def build_sequence_wide(df: pd.DataFrame, measure: str, electrodes: list[str]):
    needed = [measure, "ch_name"] + SEQ_META_COLS
    d = df.dropna(subset=needed).copy()

    seq_wide = d.pivot_table(
        index=SEQ_META_COLS,
        columns="ch_name",
        values=measure,
        aggfunc="first",
        observed=True,
    )

    seq_wide = seq_wide.reindex(columns=electrodes)

    if seq_wide.shape[0] == 0:
        return None, None, None

    meta = seq_wide.index.to_frame(index=False).reset_index(drop=True)
    X = seq_wide.to_numpy(dtype=float)
    return electrodes, meta, X


def build_target_vector(meta: pd.DataFrame, effect: str) -> np.ndarray:
    g = (meta["group"].astype(str) == "experimental").astype(float).to_numpy()
    f = pd.to_numeric(meta["f"], errors="coerce").to_numpy()
    f2 = pd.to_numeric(meta["f2"], errors="coerce").to_numpy()

    if effect == "f":
        y = f
    elif effect == "f2":
        y = f2
    elif effect == "group[T.experimental]":
        y = g
    elif effect == "group[T.experimental]:f":
        y = g * f
    elif effect == "group[T.experimental]:f2":
        y = g * f2
    else:
        raise ValueError(f"Unknown target effect: {effect}")

    return y.astype(float)


def build_nuisance_design(meta: pd.DataFrame) -> pd.DataFrame:
    covars = meta[["mean_trial_difficulty_c", "half"]].copy()
    covars["mean_trial_difficulty_c"] = pd.to_numeric(
        covars["mean_trial_difficulty_c"], errors="coerce"
    )
    covars["half"] = covars["half"].astype("category")
    covars = pd.get_dummies(covars, columns=["half"], drop_first=True)
    covars = covars.astype(float)
    return covars


def residualize_vector(y: np.ndarray, covars: pd.DataFrame) -> np.ndarray:
    valid = np.isfinite(y) & np.all(np.isfinite(covars.to_numpy()), axis=1)
    out = np.full_like(y, np.nan, dtype=float)

    if valid.sum() < 3:
        return out

    Xc = sm.add_constant(covars.loc[valid], has_constant="add")
    fit = sm.OLS(y[valid], Xc).fit()
    out[valid] = fit.resid
    return out


def residualize_matrix(X: np.ndarray, covars: pd.DataFrame) -> np.ndarray:
    out = np.full_like(X, np.nan, dtype=float)
    cov_ok = np.all(np.isfinite(covars.to_numpy()), axis=1)

    for j in range(X.shape[1]):
        xj = X[:, j]
        valid = np.isfinite(xj) & cov_ok

        if valid.sum() < 3:
            continue

        Xc = sm.add_constant(covars.loc[valid], has_constant="add")
        fit = sm.OLS(xj[valid], Xc).fit()
        out[valid, j] = fit.resid

    return out


def subject_center_vector(y: np.ndarray, ids: pd.Series) -> np.ndarray:
    out = np.full_like(y, np.nan, dtype=float)
    tmp = pd.DataFrame({"id": ids.astype(str).values, "y": y})

    for sid, idx in tmp.groupby("id").groups.items():
        vals = tmp.loc[idx, "y"].to_numpy(dtype=float)
        mu = np.nanmean(vals)
        out[idx] = vals - mu

    return out


def subject_center_matrix(X: np.ndarray, ids: pd.Series) -> np.ndarray:
    out = np.full_like(X, np.nan, dtype=float)
    tmp_ids = ids.astype(str).values

    for sid in pd.unique(tmp_ids):
        idx = np.where(tmp_ids == sid)[0]
        Xi = X[idx, :]
        mu = np.nanmean(Xi, axis=0, keepdims=True)
        out[idx, :] = Xi - mu

    return out


def fit_supervised_filter(X_train: np.ndarray, meta_train: pd.DataFrame, target_effect: str, alpha: float):
    y_raw = build_target_vector(meta_train, target_effect)
    covars = build_nuisance_design(meta_train)

    X_res = residualize_matrix(X_train, covars)
    y_res = residualize_vector(y_raw, covars)

    X_ws = subject_center_matrix(X_res, meta_train["id"])
    y_ws = subject_center_vector(y_res, meta_train["id"])

    valid_rows = np.isfinite(y_ws) & np.all(np.isfinite(X_ws), axis=1)

    if valid_rows.sum() < 10:
        raise RuntimeError("Too few valid training rows after residualization/centering.")

    X_fit = X_ws[valid_rows, :]
    y_fit = y_ws[valid_rows]

    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_fit, y_fit)

    w = ridge.coef_.astype(float)
    w_norm = np.linalg.norm(w)
    if not np.isfinite(w_norm) or w_norm == 0:
        raise RuntimeError("Filter has zero norm.")
    w = w / w_norm

    cov_X = np.cov(X_fit, rowvar=False)
    a = cov_X @ w
    a_norm = np.linalg.norm(a)
    if np.isfinite(a_norm) and a_norm > 0:
        a = a / a_norm

    return w, a


def compute_available_scores(X_test: np.ndarray, w: np.ndarray, min_filter_mass: float):
    scores = np.full(X_test.shape[0], np.nan, dtype=float)
    retained_mass = np.full(X_test.shape[0], np.nan, dtype=float)
    n_available = np.zeros(X_test.shape[0], dtype=int)

    total_mass = float(np.sum(w ** 2))
    if total_mass <= 0:
        return scores, retained_mass, n_available

    for i in range(X_test.shape[0]):
        xi = X_test[i, :]
        valid = np.isfinite(xi) & np.isfinite(w)
        n_available[i] = int(valid.sum())

        if valid.sum() == 0:
            continue

        mass_i = float(np.sum(w[valid] ** 2))
        frac = mass_i / total_mass
        retained_mass[i] = frac

        if not np.isfinite(frac) or frac < min_filter_mass:
            continue

        denom = np.sqrt(mass_i)
        if denom == 0:
            continue

        scores[i] = float(np.dot(xi[valid], w[valid]) / denom)

    return scores, retained_mass, n_available


def compute_filter_stability(df_filters: pd.DataFrame, electrodes: list[str], value_col: str):
    if df_filters.empty:
        return pd.DataFrame(), pd.DataFrame()

    wide = (
        df_filters.pivot(index="held_out_subject", columns="electrode", values=value_col)
        .reindex(columns=electrodes)
        .sort_index()
    )

    subjects = wide.index.tolist()
    X = wide.to_numpy(dtype=float)

    pair_rows = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            xi = X[i, :]
            xj = X[j, :]
            valid = np.isfinite(xi) & np.isfinite(xj)
            r = np.nan if valid.sum() <= 1 else np.corrcoef(xi[valid], xj[valid])[0, 1]
            pair_rows.append({"subject_1": subjects[i], "subject_2": subjects[j], "r": r})

    df_pair = pd.DataFrame(pair_rows)

    mean_vec = np.nanmean(X, axis=0)
    norm = np.linalg.norm(np.nan_to_num(mean_vec, nan=0.0))
    if norm > 0:
        mean_vec = mean_vec / norm

    to_mean_rows = []
    for i, sid in enumerate(subjects):
        xi = X[i, :]
        valid = np.isfinite(xi) & np.isfinite(mean_vec)
        r = np.nan if valid.sum() <= 1 else np.corrcoef(xi[valid], mean_vec[valid])[0, 1]
        to_mean_rows.append({"held_out_subject": sid, "r_to_mean": r})

    df_to_mean = pd.DataFrame(to_mean_rows)

    summary = pd.DataFrame(
        {
            "n_successful_folds": [len(subjects)],
            "mean_pairwise_r": [df_pair["r"].mean(skipna=True)],
            "median_pairwise_r": [df_pair["r"].median(skipna=True)],
            "sd_pairwise_r": [df_pair["r"].std(skipna=True)],
            "mean_r_to_mean": [df_to_mean["r_to_mean"].mean(skipna=True)],
            "median_r_to_mean": [df_to_mean["r_to_mean"].median(skipna=True)],
            "sd_r_to_mean": [df_to_mean["r_to_mean"].std(skipna=True)],
        }
    )

    return df_pair, summary


def save_topomap(values: np.ndarray, info: mne.Info, title: str, out_file: Path):
    vmax = np.nanmax(np.abs(values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig = plt.figure(figsize=(5, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.06])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    evoked = mne.EvokedArray(values[:, None], info, tmin=0.0, verbose=False)
    evoked.plot_topomap(
        times=[0],
        axes=[ax, cax],
        colorbar=True,
        time_format="",
        cmap="RdBu_r",
        vlim=(-vmax, vmax),
        scalings=1,
        show=False,
        sphere=None,
    )

    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_curves_with_counts(df_model: pd.DataFrame, fit, outcome_name: str, out_file: Path, n_bins: int = 8):
    d = df_model.copy()
    group_order = ["control", "experimental"]

    edges = np.linspace(d["f"].min(), d["f"].max(), n_bins + 1)
    d["f_bin"] = pd.cut(d["f"], bins=edges, include_lowest=True)

    agg = (
        d.groupby(["group", "f_bin"], observed=True)
        .agg(
            mean_score=("score", "mean"),
            sem_score=("score", "sem"),
            n=("score", "size"),
        )
        .reset_index()
    )

    agg["f_mid"] = agg["f_bin"].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    agg = agg.sort_values(["group", "f_mid"]).reset_index(drop=True)

    f_grid = np.linspace(d["f"].min(), d["f"].max(), 300)
    f_mean = float(d["f"].mean())
    f2_grid = (f_grid - f_mean) ** 2 - np.mean((d["f"] - f_mean) ** 2)

    half_ref = d["half"].mode().iloc[0]
    difficulty_ref = 0.0

    pred_rows = []
    for group_name in group_order:
        for f_val, f2_val in zip(f_grid, f2_grid):
            pred_rows.append(
                {
                    "group": group_name,
                    "f": f_val,
                    "f2": f2_val,
                    "mean_trial_difficulty_c": difficulty_ref,
                    "half": half_ref,
                }
            )

    pred = pd.DataFrame(pred_rows)
    pred["pred"] = fit.predict(pred)

    colors = {"control": "#1f77b4", "experimental": "#d62728"}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    ax = axes[0]
    for group_name in group_order:
        dg = agg[agg["group"] == group_name].copy()
        dg_pred = pred[pred["group"] == group_name].copy()
        color = colors[group_name]

        ax.errorbar(
            dg["f_mid"],
            dg["mean_score"],
            yerr=dg["sem_score"],
            fmt="o",
            linestyle="none",
            capsize=3,
            color=color,
            label=f"{group_name} observed",
        )

        ax.plot(
            dg_pred["f"],
            dg_pred["pred"],
            linewidth=3,
            color=color,
            label=f"{group_name} model",
        )

    ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
    ax.set_ylabel("filter_score")
    ax.set_title("Final MLM: filter score as a function of feedback")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    width = np.diff(edges).mean() * 0.4

    for group_name in group_order:
        dg = agg[agg["group"] == group_name].copy()
        offset = -width / 2 if group_name == "control" else width / 2
        color = colors[group_name]

        ax2.bar(
            dg["f_mid"] + offset,
            dg["n"],
            width=width,
            color=color,
            alpha=0.6,
        )

    ax2.axvline(0, color="k", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Feedback (f)")
    ax2.set_ylabel("n")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_loso_fold(
    held_out_subject,
    df_base: pd.DataFrame,
    measure: str,
    electrodes: list[str],
    target_effect: str,
):
    sid = str(held_out_subject)

    df_train = df_base[df_base["id"].astype(str) != sid].copy()
    df_test = df_base[df_base["id"].astype(str) == sid].copy()

    _, meta_train, X_train = build_sequence_wide(df_train, measure=measure, electrodes=electrodes)
    _, meta_test, X_test = build_sequence_wide(df_test, measure=measure, electrodes=electrodes)

    if X_train is None:
        return {"status": "no_training_data", "held_out_subject": sid}
    if X_test is None:
        return {"status": "no_test_data", "held_out_subject": sid}

    try:
        w, a = fit_supervised_filter(
            X_train=X_train,
            meta_train=meta_train,
            target_effect=target_effect,
            alpha=RIDGE_ALPHA,
        )
    except Exception as exc:
        return {"status": f"filter_fit_failed: {exc}", "held_out_subject": sid}

    scores, retained_mass, n_available = compute_available_scores(
        X_test=X_test,
        w=w,
        min_filter_mass=MIN_RETAINED_FILTER_MASS,
    )

    df_scores = meta_test.copy()
    df_scores["held_out_subject"] = sid
    df_scores["component_score"] = scores
    df_scores["retained_filter_mass"] = retained_mass
    df_scores["n_available_electrodes"] = n_available
    df_scores["n_total_electrodes"] = len(electrodes)

    df_scores = df_scores.dropna(subset=["component_score"]).reset_index(drop=True)

    if len(df_scores) == 0:
        return {"status": "no_scored_sequences_after_mass_threshold", "held_out_subject": sid}

    df_w = pd.DataFrame(
        {"held_out_subject": sid, "electrode": electrodes, "weight": w}
    )

    df_a = pd.DataFrame(
        {"held_out_subject": sid, "electrode": electrodes, "haufe_pattern": a}
    )

    return {
        "status": "ok",
        "held_out_subject": sid,
        "scores": df_scores,
        "weights": df_w,
        "patterns": df_a,
    }


# -----------------------------------------------------------------------------
# PSD loading + plotting
# -----------------------------------------------------------------------------
def load_all_subject_psds():
    npz_files = sorted(PATH_IN.glob("sub-*_seq_psd_channelwise_car.npz"))

    psd_blocks = []
    meta_blocks = []
    freqs_ref = None
    channels_ref = None

    for npz_file in npz_files:
        subj_tag = npz_file.stem.replace("_seq_psd_channelwise_car", "")
        index_file = PATH_IN / f"{subj_tag}_seq_psd_channelwise_index_car.csv"

        if not index_file.exists():
            continue

        arr = np.load(npz_file, allow_pickle=True)
        psd = arr["psd"]
        freqs = arr["freqs"]
        channels = arr["channels"].astype(str)

        meta = pd.read_csv(index_file)

        if psd.shape[0] != len(meta):
            print(f"Skipping {subj_tag}: PSD/index length mismatch.")
            continue

        if freqs_ref is None:
            freqs_ref = freqs.copy()
        else:
            if len(freqs_ref) != len(freqs) or not np.allclose(freqs_ref, freqs):
                raise RuntimeError(f"Frequency mismatch in {npz_file.name}")

        if channels_ref is None:
            channels_ref = channels.copy()
        else:
            if list(channels_ref) != list(channels):
                raise RuntimeError(f"Channel mismatch in {npz_file.name}")

        meta["id"] = meta["id"].astype(str)
        meta["group"] = meta["group"].astype(str)
        meta["half"] = meta["half"].astype(str)

        psd_blocks.append(psd)
        meta_blocks.append(meta)

    if len(psd_blocks) == 0:
        raise RuntimeError("No PSD files found.")

    psd_all = np.concatenate(psd_blocks, axis=0)
    meta_all = pd.concat(meta_blocks, ignore_index=True)

    return psd_all, freqs_ref, channels_ref, meta_all


def plot_binned_filter_weighted_spectra(
    psd_all: np.ndarray,
    freqs: np.ndarray,
    channels: np.ndarray,
    meta: pd.DataFrame,
    weight_vec: np.ndarray,
    out_file: Path,
    title: str,
    n_bins: int = 7,
    use_log10: bool = True,
    spectral_weight_mode: str = "squared",
):
    w_spec = make_spectral_weights(weight_vec, mode=spectral_weight_mode)
    weighted_psd = np.einsum("c,ncf->nf", w_spec, psd_all)

    if use_log10:
        weighted_psd = np.log10(np.maximum(weighted_psd, 1e-20))

    d = meta.copy().reset_index(drop=True)
    d["f"] = pd.to_numeric(d["f"], errors="coerce")
    d["weighted_spectrum"] = list(weighted_psd)

    d = d.dropna(subset=["f", "group"]).copy()

    edges = np.linspace(d["f"].min(), d["f"].max(), n_bins + 1)
    d["f_bin"] = pd.cut(d["f"], bins=edges, include_lowest=True)

    grouped = d.groupby(["group", "f_bin"], observed=True)

    rows = []
    for (group_name, f_bin), dg in grouped:
        if len(dg) == 0:
            continue

        spec = np.vstack(dg["weighted_spectrum"].values)
        mean_spec = np.mean(spec, axis=0)
        sem_spec = (
            np.std(spec, axis=0, ddof=1) / np.sqrt(len(spec))
            if len(spec) > 1
            else np.zeros_like(mean_spec)
        )

        rows.append(
            {
                "group": group_name,
                "f_mid": (f_bin.left + f_bin.right) / 2,
                "n": len(dg),
                "spectrum_mean": mean_spec,
                "spectrum_sem": sem_spec,
            }
        )

    df_plot = pd.DataFrame(rows)

    colors = {"control": "#1f77b4", "experimental": "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 6))
    max_n = max(df_plot["n"].max(), 1)

    for group_name in ["control", "experimental"]:
        dg = df_plot[df_plot["group"] == group_name].sort_values("f_mid")
        color = colors[group_name]

        for k, (_, row) in enumerate(dg.iterrows()):
            alpha = min(1.0, 0.25 + row["n"] / max_n)
            label = group_name if k == 0 else None

            ax.plot(
                freqs,
                row["spectrum_mean"],
                color=color,
                alpha=alpha,
                linewidth=2,
                label=label,
            )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log10 weighted PSD" if use_log10 else "Weighted PSD")
    ax.set_title(title + f"\n(nonnegative {spectral_weight_mode} weights)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binned_filter_weighted_spectra_heatmap(
    psd_all: np.ndarray,
    freqs: np.ndarray,
    channels: np.ndarray,
    meta: pd.DataFrame,
    weight_vec: np.ndarray,
    out_file: Path,
    title: str,
    n_bins: int = 7,
    use_log10: bool = True,
    spectral_weight_mode: str = "squared",
):
    w_spec = make_spectral_weights(weight_vec, mode=spectral_weight_mode)

    weighted_psd = np.einsum("c,ncf->nf", w_spec, psd_all)

    if use_log10:
        weighted_psd = np.log10(np.maximum(weighted_psd, 1e-20))

    d = meta.copy().reset_index(drop=True)
    d["f"] = pd.to_numeric(d["f"], errors="coerce")
    d["weighted_spectrum"] = list(weighted_psd)
    d = d.dropna(subset=["f", "group"]).copy()

    edges = np.linspace(d["f"].min(), d["f"].max(), n_bins + 1)
    d["f_bin"] = pd.cut(d["f"], bins=edges, include_lowest=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ims = []
    for ax, group_name in zip(axes, ["control", "experimental"]):
        dg = d[d["group"] == group_name].copy()

        rows = []
        mids = []

        for f_bin, db in dg.groupby("f_bin", observed=True):
            if len(db) == 0:
                continue
            mids.append((f_bin.left + f_bin.right) / 2)
            spec = np.vstack(db["weighted_spectrum"].values)
            rows.append(np.mean(spec, axis=0))

        ax.set_title(group_name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Feedback (f)")

        if len(rows) == 0:
            continue

        mat = np.vstack(rows)

        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            extent=[freqs.min(), freqs.max(), min(mids), max(mids)],
        )
        ims.append(im)

    if len(ims) > 0:
        cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label("Log10 weighted PSD" if use_log10 else "Weighted PSD")

    fig.suptitle(title + f"\n(nonnegative {spectral_weight_mode} weights)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------------------
if MEASURE not in VALID_MEASURES:
    raise ValueError(f"MEASURE must be one of: {VALID_MEASURES}")

bad_effects = [e for e in TARGET_EFFECTS if e not in VALID_TARGET_EFFECTS]
if bad_effects:
    raise ValueError(f"Unknown TARGET_EFFECTS: {bad_effects}")

df_raw = pd.read_csv(FILE_IN)

required_cols = [
    "id",
    "group",
    "ch_name",
    "block_nr",
    "sequence_nr",
    "half",
    "n_trials",
    "mean_trial_difficulty",
    "f",
    "mean_rt",
    "mean_log_rt",
    "r2",
    "error",
    "exponent",
] + VALID_MEASURES

missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in ["f", "mean_trial_difficulty", "mean_rt", "mean_log_rt", "r2", "error", "exponent"] + VALID_MEASURES:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

df_raw["id"] = df_raw["id"].astype(str)
df_raw["group"] = df_raw["group"].astype("category")
df_raw["group"] = df_raw["group"].cat.set_categories(["control", "experimental"])
df_raw["half"] = df_raw["half"].astype("category")
df_raw["ch_name"] = df_raw["ch_name"].astype(str)

if APPLY_QC:
    n_before = len(df_raw)
    df_raw = apply_qc(df_raw)
    print(f"QC applied. Rows before: {n_before}, after: {len(df_raw)}")

df_raw = add_centered_predictors(df_raw)
df_raw = retain_subjects_with_min_sequences(df_raw, MIN_SEQUENCES_PER_SUBJECT)

if df_raw["id"].nunique() < MIN_SUBJECTS:
    raise RuntimeError(
        f"Too few subjects after filtering: {df_raw['id'].nunique()} (need >= {MIN_SUBJECTS})"
    )

electrodes = sorted(df_raw["ch_name"].dropna().unique().tolist())

info = mne.create_info(
    ch_names=electrodes,
    sfreq=1000,
    ch_types="eeg",
    verbose=None,
)
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, on_missing="warn", match_case=False)

print(f"Measure: {MEASURE}")
print(f"Subjects: {df_raw['id'].nunique()}")
print(f"Electrodes: {len(electrodes)}")
print(f"Sequences: {df_raw[SEQ_ID_COLS].drop_duplicates().shape[0]}")


# -----------------------------------------------------------------------------
# Load PSDs once
# -----------------------------------------------------------------------------
psd_all, freqs_psd, channels_psd, psd_meta = load_all_subject_psds()
psd_meta["id"] = psd_meta["id"].astype(str)
psd_meta["group"] = psd_meta["group"].astype(str)

channel_to_idx = {ch: i for i, ch in enumerate(channels_psd)}
missing_channels = [ch for ch in electrodes if ch not in channel_to_idx]
if missing_channels:
    raise RuntimeError(f"Channels missing in PSD files: {missing_channels}")

psd_reorder_idx = [channel_to_idx[ch] for ch in electrodes]
psd_all = psd_all[:, psd_reorder_idx, :]


# -----------------------------------------------------------------------------
# Run target effects
# -----------------------------------------------------------------------------
all_final_results = []

for target_effect in TARGET_EFFECTS:
    print("\n" + "=" * 100)
    print(f"Running target effect: {target_effect}")

    effect_safe = safe_name(target_effect)
    combo_dir = PATH_OUT / MEASURE / effect_safe
    combo_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(df_raw["id"].unique().tolist())

    fold_results = Parallel(
        n_jobs=N_JOBS,
        backend=PARALLEL_BACKEND,
        verbose=PARALLEL_VERBOSE,
    )(
        delayed(run_loso_fold)(
            held_out_subject=sid,
            df_base=df_raw,
            measure=MEASURE,
            electrodes=electrodes,
            target_effect=target_effect,
        )
        for sid in subjects
    )

    score_list = []
    weight_list = []
    pattern_list = []
    failed_rows = []

    for res in fold_results:
        if res["status"] == "ok":
            score_list.append(res["scores"])
            weight_list.append(res["weights"])
            pattern_list.append(res["patterns"])
        else:
            failed_rows.append(
                {
                    "held_out_subject": res["held_out_subject"],
                    "status": res["status"],
                }
            )

    df_failed = pd.DataFrame(failed_rows)
    df_failed.to_csv(combo_dir / f"{MEASURE}_loso_failed_folds_{effect_safe}.csv", index=False)

    if len(score_list) == 0:
        print("No successful LOSO folds.")
        continue

    df_scores = pd.concat(score_list, ignore_index=True)
    df_weights = pd.concat(weight_list, ignore_index=True)
    df_patterns = pd.concat(pattern_list, ignore_index=True)

    df_weights = align_filter_table_to_anchor(
        df_filters=df_weights,
        electrodes=electrodes,
        value_col="weight",
    )

    df_patterns = align_filter_table_to_anchor(
        df_filters=df_patterns,
        electrodes=electrodes,
        value_col="haufe_pattern",
    )

    df_scores.to_csv(combo_dir / f"{MEASURE}_loso_scores_{effect_safe}.csv", index=False)
    df_weights.to_csv(combo_dir / f"{MEASURE}_loso_weights_{effect_safe}.csv", index=False)
    df_patterns.to_csv(combo_dir / f"{MEASURE}_loso_haufe_patterns_{effect_safe}.csv", index=False)

    df_pair_w, df_stab_w = compute_filter_stability(
        df_filters=df_weights,
        electrodes=electrodes,
        value_col="weight",
    )
    df_pair_a, df_stab_a = compute_filter_stability(
        df_filters=df_patterns,
        electrodes=electrodes,
        value_col="haufe_pattern",
    )

    df_pair_w.to_csv(combo_dir / f"{MEASURE}_weight_stability_pairwise_{effect_safe}.csv", index=False)
    df_pair_a.to_csv(combo_dir / f"{MEASURE}_haufe_stability_pairwise_{effect_safe}.csv", index=False)

    if not df_stab_w.empty:
        df_stab_w["kind"] = "weights"
        df_stab_w.to_csv(combo_dir / f"{MEASURE}_weight_stability_summary_{effect_safe}.csv", index=False)

    if not df_stab_a.empty:
        df_stab_a["kind"] = "haufe"
        df_stab_a.to_csv(combo_dir / f"{MEASURE}_haufe_stability_summary_{effect_safe}.csv", index=False)

    df_mean_w = (
        df_weights.groupby("electrode", as_index=False)["weight"]
        .mean()
        .set_index("electrode")
        .reindex(electrodes)
        .reset_index()
    )
    mean_w = df_mean_w["weight"].to_numpy(dtype=float)

    df_mean_a = (
        df_patterns.groupby("electrode", as_index=False)["haufe_pattern"]
        .mean()
        .set_index("electrode")
        .reindex(electrodes)
        .reset_index()
    )
    mean_a = df_mean_a["haufe_pattern"].to_numpy(dtype=float)

    save_topomap(
        values=mean_w,
        info=info,
        title=f"Mean LOSO regression filter\n{target_effect}",
        out_file=combo_dir / f"{MEASURE}_mean_loso_filter_{effect_safe}.png",
    )
    save_topomap(
        values=mean_a,
        info=info,
        title=f"Mean LOSO Haufe pattern\n{target_effect}",
        out_file=combo_dir / f"{MEASURE}_mean_loso_haufe_{effect_safe}.png",
    )

    d_final = df_scores.copy()
    d_final["id"] = d_final["id"].astype("category")
    d_final["group"] = d_final["group"].astype("category")
    d_final["group"] = d_final["group"].cat.set_categories(["control", "experimental"])
    d_final["half"] = d_final["half"].astype("category")

    for col in [
        "component_score",
        "f",
        "f2",
        "mean_trial_difficulty_c",
        "mean_rt",
        "mean_log_rt",
        "retained_filter_mass",
        "n_available_electrodes",
        "n_total_electrodes",
    ]:
        d_final[col] = pd.to_numeric(d_final[col], errors="coerce")

    d_model = d_final.dropna(
        subset=[
            "component_score",
            "group",
            "id",
            "f",
            "f2",
            "mean_trial_difficulty_c",
            "half",
        ]
    ).copy()
    d_model = d_model.rename(columns={"component_score": "score"})

    fit, used_re, fit_errors = fit_mixedlm_with_fallback(
        df_model=d_model,
        formula=FINAL_FORMULA,
        re_formulas=FINAL_RE_FORMULAS,
    )

    if fit is None:
        raise RuntimeError("Final MLM failed.\n" + "\n".join(fit_errors))

    fe = fit.fe_params
    se = fit.bse_fe.reindex(fe.index)
    tvals = fe / se.replace(0, np.nan)
    pvals = fit.pvalues.reindex(fe.index)

    df_final = pd.DataFrame(
        {
            "term": fe.index,
            "beta": fe.values,
            "se": se.values,
            "t": tvals.values,
            "p": pvals.values,
            "random_effects": used_re,
            "n_subjects": d_model["id"].nunique(),
            "n_obs": len(d_model),
            "llf": fit.llf,
            "aic": fit.aic if np.isfinite(fit.aic) else np.nan,
            "bic": fit.bic if np.isfinite(fit.bic) else np.nan,
            "measure": MEASURE,
            "target_effect": target_effect,
            "ridge_alpha": RIDGE_ALPHA,
            "min_retained_filter_mass": MIN_RETAINED_FILTER_MASS,
            "successful_loso_folds": len(score_list),
            "total_subjects": len(subjects),
            "mean_retained_filter_mass": d_final["retained_filter_mass"].mean(skipna=True),
            "median_retained_filter_mass": d_final["retained_filter_mass"].median(skipna=True),
            "mean_available_electrodes": d_final["n_available_electrodes"].mean(skipna=True),
            "median_available_electrodes": d_final["n_available_electrodes"].median(skipna=True),
        }
    )

    if not df_stab_w.empty:
        for col in df_stab_w.columns:
            if col != "kind":
                df_final[f"weight_{col}"] = df_stab_w.iloc[0][col]

    if not df_stab_a.empty:
        for col in df_stab_a.columns:
            if col != "kind":
                df_final[f"haufe_{col}"] = df_stab_a.iloc[0][col]

    df_final.to_csv(combo_dir / f"{MEASURE}_final_mlm_{effect_safe}.csv", index=False)

    with open(combo_dir / f"{MEASURE}_final_mlm_{effect_safe}.txt", "w") as f:
        f.write("Final MLM on LOSO out-of-fold component scores\n")
        f.write(f"Measure: {MEASURE}\n")
        f.write(f"Target effect used to derive filter: {target_effect}\n")
        f.write(f"Ridge alpha: {RIDGE_ALPHA}\n")
        f.write(f"Minimum retained filter mass: {MIN_RETAINED_FILTER_MASS}\n")
        f.write(f"Successful LOSO folds: {len(score_list)} / {len(subjects)}\n")
        f.write(f"Random-effects structure: {used_re}\n\n")
        f.write(str(fit.summary()))

    plot_score_curves_with_counts(
        df_model=d_model,
        fit=fit,
        outcome_name=f"{MEASURE}_component_score",
        out_file=combo_dir / f"{MEASURE}_score_curves_{effect_safe}.png",
        n_bins=N_BINS_PLOT,
    )

    keep_ids = set(df_raw["id"].astype(str).unique())
    psd_meta_use = psd_meta[psd_meta["id"].astype(str).isin(keep_ids)].copy()
    psd_all_use = psd_all[psd_meta_use.index.to_numpy(), :, :]

    if PSD_WEIGHT_SOURCE == "haufe":
        spectral_vec = mean_a.copy()
        vec_label = "haufe"
    elif PSD_WEIGHT_SOURCE == "weights":
        spectral_vec = mean_w.copy()
        vec_label = "weights"
    else:
        raise ValueError("PSD_WEIGHT_SOURCE must be 'haufe' or 'weights'")

    plot_binned_filter_weighted_spectra(
        psd_all=psd_all_use,
        freqs=freqs_psd,
        channels=np.array(electrodes),
        meta=psd_meta_use,
        weight_vec=spectral_vec,
        out_file=combo_dir / f"{MEASURE}_binned_weighted_spectra_{effect_safe}_{vec_label}.png",
        title=f"{MEASURE} | {target_effect} | {vec_label} | weighted spectra by f-bin",
        n_bins=N_BINS_PLOT,
        use_log10=PSD_USE_LOG10,
        spectral_weight_mode=PSD_SPECTRAL_WEIGHT_MODE,
    )

    plot_binned_filter_weighted_spectra_heatmap(
        psd_all=psd_all_use,
        freqs=freqs_psd,
        channels=np.array(electrodes),
        meta=psd_meta_use,
        weight_vec=spectral_vec,
        out_file=combo_dir / f"{MEASURE}_binned_weighted_spectra_heatmap_{effect_safe}_{vec_label}.png",
        title=f"{MEASURE} | {target_effect} | {vec_label} | weighted spectra heatmap",
        n_bins=N_BINS_PLOT,
        use_log10=PSD_USE_LOG10,
        spectral_weight_mode=PSD_SPECTRAL_WEIGHT_MODE,
    )

    all_final_results.append(df_final)

    print(f"Completed target effect: {target_effect}")
    print(f"Random effects: {used_re}")
    print(f"Successful LOSO folds: {len(score_list)} / {len(subjects)}")
    print(f"Mean retained filter mass: {d_final['retained_filter_mass'].mean(skipna=True):.4f}")
    print(f"Mean available electrodes: {d_final['n_available_electrodes'].mean(skipna=True):.2f}")


# -----------------------------------------------------------------------------
# Save master summary
# -----------------------------------------------------------------------------
if len(all_final_results) > 0:
    df_master = pd.concat(all_final_results, ignore_index=True)
    out_master = PATH_OUT / MEASURE / "master_final_results.csv"
    out_master.parent.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(out_master, index=False)
    print("\nSaved master summary:")
    print(out_master)
else:
    print("\nNo successful final results.")