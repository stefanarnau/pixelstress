# -----------------------------------------------------------------------------
# Batch LOSO effect-weighted spatial filtering + final MLM
# - reads CSD-only sequence dataframe
# - applies sequence-level FOOOF QC after loading
# - complete-sequence restriction: keep only sequences with all electrodes present
# - minimum retained complete sequences per subject
# - LOSO effect-weighted spatial filtering
# - final MLM on out-of-fold component scores
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import mne
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_in = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
path_out = Path("/mnt/data_dump/pixelstress/6_batch_loso_effect_weighted/")
path_out.mkdir(parents=True, exist_ok=True)

file_in = path_in / "all_subjects_seq_fooof_rt_channelwise_long_csd.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
measures = [
    # "exponent",
    "theta_flat",
    # "alpha_flat",
    # "beta_flat",
]

target_effects = [
    #"group[T.experimental]",
    #"f_lin",
    #"f_quad",
    #"group[T.experimental]:f_lin",
    "group[T.experimental]:f_quad",
]

formula = """
score ~ group * f_lin + group * f_quad
        + mean_trial_difficulty_c + half
"""

re_formulas = [
    "1 + f_lin + f_quad",
    "1 + f_lin",
    "1",
]

effects_for_topomaps = [
    "group[T.experimental]",
    "f_lin",
    "f_quad",
    "group[T.experimental]:f_lin",
    "group[T.experimental]:f_quad",
    "half[T.second]",
    "mean_trial_difficulty_c",
]

# Electrode-wise model requirements
min_subjects_per_electrode_model = 8
min_obs_per_electrode_model = 50

# Subject retention
min_sequences_per_subject = 8

# Sequence-level FOOOF QC applied AFTER loading
apply_sequence_qc = True
qc_min_r2 = 0.80
qc_max_error = 0.30
qc_min_exponent = 0.50
qc_max_exponent = 3.50

# LOSO
n_jobs_loso = -1
parallel_backend = "loky"
parallel_verbose = 10

# Plotting
n_bins_plot = 9


# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------
BASE_SEQ_COLS = [
    "id",
    "group",
    "block_nr",
    "sequence_nr",
    "half",
    "n_trials",
    "mean_trial_difficulty",
    "mean_trial_difficulty_c",
    "f",
    "f_c",
    "f_lin",
    "f_quad",
    "mean_rt",
    "mean_log_rt",
]

FINAL_MODEL_REQUIRED_COLS = [
    "group",
    "id",
    "f_lin",
    "f_quad",
    "mean_trial_difficulty_c",
    "half",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_name(text: str) -> str:
    return text.replace("[", "").replace("]", "").replace(":", "x")


def get_sequence_key_cols(df_in: pd.DataFrame) -> list[str]:
    return (["window"] if "window" in df_in.columns else []) + BASE_SEQ_COLS


def get_seq_id_cols(df_in: pd.DataFrame) -> list[str]:
    return (["window"] if "window" in df_in.columns else []) + ["id", "block_nr", "sequence_nr"]


def fit_mixedlm_with_fallback(
    df_model: pd.DataFrame,
    formula: str,
    re_formulas: list[str],
    group_col: str = "id",
):
    fit = None
    used_re = None
    fit_error_log = []

    for re_formula in re_formulas:
        try:
            model = smf.mixedlm(
                formula=formula,
                data=df_model,
                groups=df_model[group_col],
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
                fit_error_log.append(f"{re_formula}: converged=False")

        except Exception as exc:
            fit_error_log.append(f"{re_formula}: {exc}")

    return fit, used_re, fit_error_log


def fit_electrode_betas(
    df_long: pd.DataFrame,
    measure: str,
    electrodes: list[str],
    formula: str,
    re_formulas: list[str],
    min_subjects: int = 8,
    min_obs: int = 50,
) -> pd.DataFrame:
    rows = []

    for ch in electrodes:
        dsub = df_long[df_long["ch_name"] == ch].copy()
        dsub = dsub.rename(columns={measure: "score"})

        n_subj = dsub["id"].nunique()
        n_obs = len(dsub)

        if n_subj < min_subjects or n_obs < min_obs:
            continue

        fit, used_re, _ = fit_mixedlm_with_fallback(
            df_model=dsub,
            formula=formula,
            re_formulas=re_formulas,
            group_col="id",
        )

        if fit is None:
            continue

        fe = fit.fe_params
        se = fit.bse_fe.reindex(fe.index)

        for term in fe.index:
            tval = np.nan
            if pd.notna(se[term]) and se[term] != 0:
                tval = float(fe[term] / se[term])

            rows.append(
                {
                    "electrode": ch,
                    "term": term,
                    "beta": float(fe[term]),
                    "se": float(se[term]) if pd.notna(se[term]) else np.nan,
                    "t": tval,
                    "random_effects": used_re,
                    "n_subjects": int(n_subj),
                    "n_obs": int(n_obs),
                }
            )

    return pd.DataFrame(rows)


def build_sequence_wide(
    df_long: pd.DataFrame,
    measure: str,
    electrodes: list[str] | None = None,
):
    key_cols = get_sequence_key_cols(df_long)
    df_use = df_long.dropna(subset=[measure, "ch_name"] + key_cols).copy()

    seq_wide = df_use.pivot_table(
        index=key_cols,
        columns="ch_name",
        values=measure,
        aggfunc="first",
        observed=True,
    )

    if electrodes is not None:
        seq_wide = seq_wide.reindex(columns=electrodes)
    else:
        seq_wide = seq_wide.sort_index(axis=1)

    seq_wide = seq_wide.dropna(axis=0, how="any").copy()

    if seq_wide.shape[0] == 0:
        return None, None, None

    meta = seq_wide.index.to_frame(index=False).reset_index(drop=True)
    X = seq_wide.to_numpy(dtype=float)
    electrode_names = seq_wide.columns.tolist()

    return electrode_names, meta, X


def run_final_score_model(
    df_scores: pd.DataFrame,
    formula: str,
    re_formulas: list[str],
):
    dsub = df_scores.dropna(subset=["EW_score"] + FINAL_MODEL_REQUIRED_COLS).copy()
    dsub = dsub.rename(columns={"EW_score": "score"})

    fit, used_re, fit_error_log = fit_mixedlm_with_fallback(
        df_model=dsub,
        formula=formula,
        re_formulas=re_formulas,
        group_col="id",
    )

    if fit is None:
        raise RuntimeError("Final LOSO MLM did not converge:\n" + "\n".join(fit_error_log))

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
            "n_subjects": dsub["id"].nunique(),
            "n_obs": len(dsub),
            "llf": fit.llf,
            "aic": fit.aic if np.isfinite(fit.aic) else np.nan,
            "bic": fit.bic if np.isfinite(fit.bic) else np.nan,
        }
    )

    return fit, used_re, dsub, df_final


def build_reference_map(
    df_beta_full: pd.DataFrame,
    electrodes: list[str],
    target_effect: str,
    value_col: str = "beta",
):
    if df_beta_full.empty or target_effect not in df_beta_full["term"].unique():
        return None

    w_ref = (
        df_beta_full[df_beta_full["term"] == target_effect]
        .set_index("electrode")
        .reindex(electrodes)[value_col]
        .to_numpy(dtype=float)
    )

    w_ref = np.nan_to_num(w_ref, nan=0.0)

    norm = np.linalg.norm(w_ref)
    if norm == 0:
        return None

    return w_ref / norm


def align_filter_to_reference(w: np.ndarray, w_ref: np.ndarray | None):
    if w_ref is None:
        return w

    valid = np.isfinite(w) & np.isfinite(w_ref)
    if valid.sum() <= 1:
        return w

    r = np.corrcoef(w[valid], w_ref[valid])[0, 1]
    if np.isfinite(r) and r < 0:
        return -w

    return w


def save_topomap(values: np.ndarray, info: mne.Info, title: str, out_file: Path):
    vmax = np.nanmax(np.abs(values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig = plt.figure(figsize=(5, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])
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


def save_full_data_topomap_panel(
    df_beta_full: pd.DataFrame,
    electrodes: list[str],
    info: mne.Info,
    terms_to_plot: list[str],
    title: str,
    out_file: Path,
):
    n_terms = len(terms_to_plot)
    if n_terms == 0:
        return

    fig = plt.figure(figsize=(3.5 * n_terms, 4))
    gs = fig.add_gridspec(1, n_terms + 1, width_ratios=[1] * n_terms + [0.05])

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_terms)]
    cax = fig.add_subplot(gs[0, n_terms])

    for i, term in enumerate(terms_to_plot):
        topo = (
            df_beta_full[df_beta_full["term"] == term]
            .set_index("electrode")
            .reindex(electrodes)["beta"]
            .to_numpy()
        )

        vmax = np.nanmax(np.abs(topo))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        evoked = mne.EvokedArray(topo[:, None], info, tmin=0.0, verbose=False)

        evoked.plot_topomap(
            times=[0],
            axes=axes[i] if i < n_terms - 1 else [axes[i], cax],
            colorbar=(i == n_terms - 1),
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            scalings=1,
            show=False,
            sphere=None,
        )

        axes[i].set_title(term)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_model_curves_with_counts(
    df_model,
    fit,
    outcome_name,
    out_file,
    n_bins=8,
):
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
    f_c_grid = f_grid - f_mean
    f_quad_center = float(np.mean((d["f"] - f_mean) ** 2))

    half_ref = d["half"].mode().iloc[0]
    difficulty_ref = 0.0

    pred_rows = []
    for group_name in group_order:
        for f_val, f_c_val in zip(f_grid, f_c_grid):
            pred_rows.append(
                {
                    "group": group_name,
                    "f": f_val,
                    "f_c": f_c_val,
                    "f_lin": f_c_val,
                    "f_quad": (f_c_val ** 2) - f_quad_center,
                    "mean_trial_difficulty_c": difficulty_ref,
                    "half": half_ref,
                }
            )

    pred = pd.DataFrame(pred_rows)
    pred["pred"] = fit.predict(pred)

    group_colors = {"control": "#1f77b4", "experimental": "#d62728"}

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    ax = axes[0]
    for group_name in group_order:
        dg = agg[agg["group"] == group_name].copy()
        dg_pred = pred[pred["group"] == group_name].copy()
        color = group_colors[group_name]

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

    ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
    ax.set_ylabel(outcome_name)
    ax.set_title(f"{outcome_name}: observed bin means and model curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    width = np.diff(edges).mean() * 0.4

    for group_name in group_order:
        dg = agg[agg["group"] == group_name].copy()
        offset = -width / 2 if group_name == "control" else width / 2
        color = group_colors[group_name]

        ax2.bar(
            dg["f_mid"] + offset,
            dg["n"],
            width=width,
            color=color,
            alpha=0.6,
        )

    ax2.axvline(0, color="k", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Signed feedback (f)")
    ax2.set_ylabel("n")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_base_dataframe(df_raw: pd.DataFrame, measure: str) -> pd.DataFrame:
    df = df_raw.copy()

    needed = [measure, "f", "group", "id", "ch_name", "mean_trial_difficulty", "half"]
    df = df.dropna(subset=needed).copy()

    df["f_c"] = df["f"] - df["f"].mean()
    df["f_lin"] = df["f_c"]
    df["f_quad"] = df["f_c"] ** 2 - np.mean(df["f_c"] ** 2)

    df["mean_trial_difficulty_c"] = (
        df["mean_trial_difficulty"] - df["mean_trial_difficulty"].mean()
    )

    return df


def compute_filter_stability(df_filters_loso: pd.DataFrame, electrodes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_filters_loso.empty:
        return pd.DataFrame(), pd.DataFrame()

    filt_wide = (
        df_filters_loso.pivot(index="held_out_subject", columns="electrode", values="weight")
        .reindex(columns=electrodes)
        .sort_index()
    )

    subjects = filt_wide.index.tolist()
    X = filt_wide.to_numpy(dtype=float)

    pair_rows = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            xi = X[i, :]
            xj = X[j, :]
            valid = np.isfinite(xi) & np.isfinite(xj)

            if valid.sum() <= 1:
                r = np.nan
            else:
                r = np.corrcoef(xi[valid], xj[valid])[0, 1]

            pair_rows.append(
                {
                    "held_out_subject_1": subjects[i],
                    "held_out_subject_2": subjects[j],
                    "r": r,
                }
            )

    df_pairwise = pd.DataFrame(pair_rows)

    w_mean = np.nanmean(X, axis=0)
    mean_norm = np.linalg.norm(np.nan_to_num(w_mean, nan=0.0))
    if mean_norm > 0:
        w_mean = w_mean / mean_norm

    corr_to_mean = []
    for i, subj in enumerate(subjects):
        xi = X[i, :]
        valid = np.isfinite(xi) & np.isfinite(w_mean)
        if valid.sum() <= 1:
            r = np.nan
        else:
            r = np.corrcoef(xi[valid], w_mean[valid])[0, 1]
        corr_to_mean.append({"held_out_subject": subj, "r_to_mean": r})

    df_to_mean = pd.DataFrame(corr_to_mean)

    df_summary = pd.DataFrame(
        {
            "n_successful_folds": [len(subjects)],
            "mean_pairwise_r": [df_pairwise["r"].mean(skipna=True)],
            "median_pairwise_r": [df_pairwise["r"].median(skipna=True)],
            "min_pairwise_r": [df_pairwise["r"].min(skipna=True)],
            "max_pairwise_r": [df_pairwise["r"].max(skipna=True)],
            "sd_pairwise_r": [df_pairwise["r"].std(skipna=True)],
            "mean_r_to_mean_filter": [df_to_mean["r_to_mean"].mean(skipna=True)],
            "median_r_to_mean_filter": [df_to_mean["r_to_mean"].median(skipna=True)],
            "sd_r_to_mean_filter": [df_to_mean["r_to_mean"].std(skipna=True)],
        }
    )

    return df_pairwise, df_summary


def apply_sequence_level_qc(
    df_long: pd.DataFrame,
    min_r2: float,
    max_error: float,
    min_exponent: float,
    max_exponent: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seq_cols = get_seq_id_cols(df_long)
    qc_mask = (
        df_long["r2"].ge(min_r2)
        & df_long["error"].le(max_error)
        & df_long["exponent"].between(min_exponent, max_exponent)
    )

    df_qc = df_long.loc[qc_mask].copy()

    seq_before = (
        df_long.groupby(seq_cols, observed=True)
        .agg(n_rows_before=("ch_name", "size"),
             n_electrodes_before=("ch_name", "nunique"))
        .reset_index()
    )

    seq_after = (
        df_qc.groupby(seq_cols, observed=True)
        .agg(n_rows_after=("ch_name", "size"),
             n_electrodes_after=("ch_name", "nunique"))
        .reset_index()
    )

    seq_summary = seq_before.merge(seq_after, on=seq_cols, how="left")
    seq_summary["n_rows_after"] = seq_summary["n_rows_after"].fillna(0).astype(int)
    seq_summary["n_electrodes_after"] = seq_summary["n_electrodes_after"].fillna(0).astype(int)
    seq_summary["n_rows_dropped_qc"] = seq_summary["n_rows_before"] - seq_summary["n_rows_after"]

    return df_qc, seq_summary


def summarize_and_filter_complete_sequences(
    df_long: pd.DataFrame,
    expected_electrodes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_n_electrodes = len(expected_electrodes)
    seq_cols = get_seq_id_cols(df_long)

    df_seq_counts = (
        df_long.groupby(seq_cols, observed=True)
        .agg(
            group=("group", "first"),
            half=("half", "first"),
            n_electrodes_present=("ch_name", "nunique"),
            n_rows=("ch_name", "size"),
        )
        .reset_index()
    )

    df_seq_counts["is_complete_sequence"] = (
        df_seq_counts["n_electrodes_present"] == expected_n_electrodes
    )

    df_summary = (
        df_seq_counts.groupby("id", observed=True)
        .agg(
            group=("group", "first"),
            n_sequences_total=("is_complete_sequence", "size"),
            n_sequences_complete=("is_complete_sequence", "sum"),
            mean_electrodes_present=("n_electrodes_present", "mean"),
            min_electrodes_present=("n_electrodes_present", "min"),
            max_electrodes_present=("n_electrodes_present", "max"),
        )
        .reset_index()
    )

    df_summary["n_sequences_dropped_incomplete"] = (
        df_summary["n_sequences_total"] - df_summary["n_sequences_complete"]
    )
    df_summary["prop_sequences_complete"] = (
        df_summary["n_sequences_complete"] / df_summary["n_sequences_total"]
    )
    df_summary["expected_n_electrodes"] = expected_n_electrodes

    df_complete_keys = df_seq_counts.loc[df_seq_counts["is_complete_sequence"], seq_cols].copy()
    df_filtered = df_long.merge(df_complete_keys, on=seq_cols, how="inner").copy()

    return df_filtered, df_summary


def summarize_and_filter_subjects_by_sequence_count(
    df_long: pd.DataFrame,
    min_sequences: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seq_cols = get_seq_id_cols(df_long)

    df_seq_counts = (
        df_long[seq_cols]
        .drop_duplicates()
        .groupby("id", observed=True)
        .size()
        .rename("n_complete_sequences")
        .reset_index()
    )

    df_group = (
        df_long.groupby("id", observed=True)
        .agg(group=("group", "first"))
        .reset_index()
    )

    df_summary = df_group.merge(df_seq_counts, on="id", how="left")
    df_summary["n_complete_sequences"] = df_summary["n_complete_sequences"].fillna(0).astype(int)
    df_summary["keep_subject"] = df_summary["n_complete_sequences"] >= min_sequences

    keep_ids = df_summary.loc[df_summary["keep_subject"], "id"]
    df_filtered = df_long[df_long["id"].isin(keep_ids)].copy()

    return df_filtered, df_summary


def run_loso_fold(
    held_out,
    df,
    measure,
    electrodes,
    formula,
    re_formulas,
    min_subjects_per_electrode_model,
    min_obs_per_electrode_model,
    target_effect,
    w_ref,
):
    subject_as_str = str(held_out)
    df_train = df[df["id"].astype(str) != subject_as_str].copy()
    df_test = df[df["id"].astype(str) == subject_as_str].copy()

    df_beta_train = fit_electrode_betas(
        df_long=df_train,
        measure=measure,
        electrodes=electrodes,
        formula=formula,
        re_formulas=re_formulas,
        min_subjects=min_subjects_per_electrode_model,
        min_obs=min_obs_per_electrode_model,
    )

    if df_beta_train.empty or target_effect not in df_beta_train["term"].unique():
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": f"target effect {target_effect} not estimable in training set",
        }

    w = (
        df_beta_train[df_beta_train["term"] == target_effect]
        .set_index("electrode")
        .reindex(electrodes)["beta"]
        .to_numpy(dtype=float)
    )

    if np.all(~np.isfinite(w)) or np.nansum(np.abs(w)) == 0:
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": "invalid filter",
        }

    w = np.nan_to_num(w, nan=0.0)
    norm = np.linalg.norm(w)

    if norm == 0:
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": "zero-norm filter",
        }

    w = w / norm
    w = align_filter_to_reference(w=w, w_ref=w_ref)

    _, _, X_train = build_sequence_wide(
        df_long=df_train,
        measure=measure,
        electrodes=electrodes,
    )

    _, seq_meta_test, X_test = build_sequence_wide(
        df_long=df_test,
        measure=measure,
        electrodes=electrodes,
    )

    if X_train is None:
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": "no complete training sequence rows",
        }

    if X_test is None:
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": "no complete held-out sequence rows",
        }

    train_means = np.nanmean(X_train, axis=0)
    X_test_c = X_test - train_means[None, :]
    score_test = X_test_c @ w

    df_scores_fold = seq_meta_test.copy()
    df_scores_fold["held_out_subject"] = held_out
    df_scores_fold["EW_score"] = score_test

    df_filter_fold = pd.DataFrame(
        {
            "held_out_subject": held_out,
            "electrode": electrodes,
            "weight": w,
        }
    )

    return {
        "held_out": held_out,
        "scores": df_scores_fold,
        "filter": df_filter_fold,
        "status": "ok",
    }


def run_one_measure_effect(
    df_base,
    measure,
    target_effect,
    electrodes,
    info,
    out_root,
):
    print("\n" + "=" * 100)
    print(f"Running measure={measure}, target_effect={target_effect}")

    effect_safe = safe_name(target_effect)
    combo_dir = out_root / measure / effect_safe
    combo_dir.mkdir(parents=True, exist_ok=True)

    df_beta_full = fit_electrode_betas(
        df_long=df_base,
        measure=measure,
        electrodes=electrodes,
        formula=formula,
        re_formulas=re_formulas,
        min_subjects=min_subjects_per_electrode_model,
        min_obs=min_obs_per_electrode_model,
    )

    df_beta_full.to_csv(combo_dir / f"{measure}_electrode_mlm_betas_full_data.csv", index=False)

    terms_to_plot = [t for t in effects_for_topomaps if t in df_beta_full["term"].unique()]
    save_full_data_topomap_panel(
        df_beta_full=df_beta_full,
        electrodes=electrodes,
        info=info,
        terms_to_plot=terms_to_plot,
        title=f"{measure} full-data electrode-wise MLM effect sizes (descriptive)",
        out_file=combo_dir / f"{measure}_full_data_topomaps.png",
    )

    w_ref = build_reference_map(
        df_beta_full=df_beta_full,
        electrodes=electrodes,
        target_effect=target_effect,
        value_col="beta",
    )

    subjects = sorted(df_base["id"].astype(str).unique())

    fold_results = Parallel(
        n_jobs=n_jobs_loso,
        backend=parallel_backend,
        verbose=parallel_verbose,
    )(
        delayed(run_loso_fold)(
            held_out=s,
            df=df_base,
            measure=measure,
            electrodes=electrodes,
            formula=formula,
            re_formulas=re_formulas,
            min_subjects_per_electrode_model=min_subjects_per_electrode_model,
            min_obs_per_electrode_model=min_obs_per_electrode_model,
            target_effect=target_effect,
            w_ref=w_ref,
        )
        for s in subjects
    )

    all_scores = []
    all_filters = []
    failed = []

    for res in fold_results:
        if res["status"] == "ok":
            all_scores.append(res["scores"])
            all_filters.append(res["filter"])
        else:
            failed.append({"held_out_subject": res["held_out"], "status": res["status"]})

    df_failed = pd.DataFrame(failed)
    df_failed.to_csv(combo_dir / f"{measure}_loso_failed_folds_{effect_safe}.csv", index=False)

    if len(all_scores) == 0:
        print(f"No LOSO scores generated for measure={measure}, effect={target_effect}")
        return None

    df_scores_loso = pd.concat(all_scores, ignore_index=True)
    df_filters_loso = pd.concat(all_filters, ignore_index=True)

    df_scores_loso.to_csv(
        combo_dir / f"{measure}_loso_effect_weighted_scores_{effect_safe}.csv",
        index=False,
    )
    df_filters_loso.to_csv(
        combo_dir / f"{measure}_loso_filters_{effect_safe}.csv",
        index=False,
    )

    df_stability_pairwise, df_stability_summary = compute_filter_stability(
        df_filters_loso=df_filters_loso,
        electrodes=electrodes,
    )

    df_stability_pairwise.to_csv(
        combo_dir / f"{measure}_loso_filter_stability_pairwise_{effect_safe}.csv",
        index=False,
    )
    df_stability_summary["measure"] = measure
    df_stability_summary["target_effect"] = target_effect
    df_stability_summary.to_csv(
        combo_dir / f"{measure}_loso_filter_stability_summary_{effect_safe}.csv",
        index=False,
    )

    df_filter_mean = (
        df_filters_loso.groupby("electrode", as_index=False)["weight"]
        .mean()
        .set_index("electrode")
        .reindex(electrodes)
        .reset_index()
    )
    w_mean = df_filter_mean["weight"].to_numpy(dtype=float)

    save_topomap(
        values=w_mean,
        info=info,
        title=f"Mean LOSO filter\n{target_effect}",
        out_file=combo_dir / f"{measure}_mean_loso_filter_{effect_safe}.png",
    )

    df_scores_loso["id"] = df_scores_loso["id"].astype("category")
    df_scores_loso["group"] = df_scores_loso["group"].astype("category")
    df_scores_loso["half"] = df_scores_loso["half"].astype("category")

    if "window" in df_scores_loso.columns:
        df_scores_loso["window"] = df_scores_loso["window"].astype("category")

    df_scores_loso["group"] = df_scores_loso["group"].cat.set_categories(["control", "experimental"])

    for col in [
        "f",
        "f_c",
        "f_lin",
        "f_quad",
        "mean_trial_difficulty",
        "mean_trial_difficulty_c",
        "mean_rt",
        "mean_log_rt",
        "EW_score",
    ]:
        if col in df_scores_loso.columns:
            df_scores_loso[col] = pd.to_numeric(df_scores_loso[col], errors="coerce")

    fit, used_re, dsub, df_final = run_final_score_model(
        df_scores=df_scores_loso,
        formula=formula,
        re_formulas=re_formulas,
    )

    df_final["successful_loso_folds"] = len(all_scores)
    df_final["total_subjects"] = len(subjects)
    df_final["measure"] = measure
    df_final["target_effect"] = target_effect

    if not df_stability_summary.empty:
        for col in df_stability_summary.columns:
            df_final[col] = df_stability_summary.iloc[0][col]

    df_final.to_csv(
        combo_dir / f"{measure}_loso_effect_weighted_final_mlm_{effect_safe}.csv",
        index=False,
    )

    with open(combo_dir / f"{measure}_loso_effect_weighted_final_mlm_{effect_safe}.txt", "w") as f:
        f.write("Final LOSO effect-weighted score model\n")
        f.write(f"Measure: {measure}\n")
        f.write(f"Target effect used to derive filters: {target_effect}\n")
        f.write(f"Random-effects structure: {used_re}\n")
        f.write(f"Successful LOSO folds: {len(all_scores)} / {len(subjects)}\n\n")
        if not df_stability_summary.empty:
            f.write("Filter stability summary\n")
            f.write(df_stability_summary.to_string(index=False))
            f.write("\n\n")
        f.write(str(fit.summary()))

    plot_score_model_curves_with_counts(
        df_model=dsub,
        fit=fit,
        outcome_name=f"{measure}_EW_score",
        out_file=combo_dir / f"{measure}_score_curves_{effect_safe}.png",
        n_bins=n_bins_plot,
    )

    print(f"Completed measure={measure}, target_effect={target_effect}")
    print(f"Random-effects structure: {used_re}")
    print(f"Successful LOSO folds: {len(all_scores)} / {len(subjects)}")
    if not df_stability_summary.empty:
        print(df_stability_summary.to_string(index=False))

    return df_final.copy()


# -----------------------------------------------------------------------------
# Load raw data
# -----------------------------------------------------------------------------
df_raw = pd.read_csv(file_in)

df_raw["id"] = df_raw["id"].astype("category")
df_raw["group"] = df_raw["group"].astype("category")
df_raw["ch_name"] = df_raw["ch_name"].astype("category")
df_raw["half"] = df_raw["half"].astype("category")

if "window" in df_raw.columns:
    df_raw["window"] = df_raw["window"].astype("category")

df_raw["group"] = df_raw["group"].cat.set_categories(["control", "experimental"])

for col in ["f", "mean_trial_difficulty", "mean_rt", "mean_log_rt", "r2", "error", "exponent"] + measures:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

if len(df_raw) == 0:
    raise RuntimeError("No rows found in input file.")


# -----------------------------------------------------------------------------
# Sequence-level QC
# -----------------------------------------------------------------------------
if apply_sequence_qc:
    df_raw_qc, df_sequence_qc_summary = apply_sequence_level_qc(
        df_long=df_raw,
        min_r2=qc_min_r2,
        max_error=qc_max_error,
        min_exponent=qc_min_exponent,
        max_exponent=qc_max_exponent,
    )

    df_sequence_qc_summary.to_csv(
        path_out / "sequence_level_qc_summary.csv",
        index=False,
    )

    n_rows_before_qc = len(df_raw)
    n_rows_after_qc = len(df_raw_qc)

    n_seq_before_qc = df_raw[get_seq_id_cols(df_raw)].drop_duplicates().shape[0]
    n_seq_after_qc = df_raw_qc[get_seq_id_cols(df_raw_qc)].drop_duplicates().shape[0]

    print("\nSequence-level QC summary:")
    print(f"QC thresholds: r2 >= {qc_min_r2}, error <= {qc_max_error}, exponent in [{qc_min_exponent}, {qc_max_exponent}]")
    print(f"Rows before QC: {n_rows_before_qc}")
    print(f"Rows after QC:  {n_rows_after_qc}")
    print(f"Rows dropped by QC: {n_rows_before_qc - n_rows_after_qc}")
    print(f"Sequences before QC: {n_seq_before_qc}")
    print(f"Sequences after QC:  {n_seq_after_qc}")

    df_raw = df_raw_qc.copy()


# -----------------------------------------------------------------------------
# Complete-sequence restriction
# -----------------------------------------------------------------------------
electrodes = sorted(df_raw["ch_name"].dropna().unique())
expected_n_electrodes = len(electrodes)

df_raw_complete, df_sequence_completeness = summarize_and_filter_complete_sequences(
    df_long=df_raw,
    expected_electrodes=electrodes,
)

df_sequence_completeness = df_sequence_completeness.sort_values(
    ["n_sequences_dropped_incomplete", "id"],
    ascending=[False, True],
).reset_index(drop=True)

df_sequence_completeness.to_csv(
    path_out / "sequence_completeness_summary.csv",
    index=False,
)

print("\nSequence completeness summary:")
print(f"Expected electrodes per sequence after QC: {expected_n_electrodes}")
print(f"Rows before complete-sequence filtering: {len(df_raw)}")
print(f"Rows after complete-sequence filtering:  {len(df_raw_complete)}")

seq_cols_print = get_seq_id_cols(df_raw)
n_seq_before = df_raw[seq_cols_print].drop_duplicates().shape[0]
n_seq_after = df_raw_complete[seq_cols_print].drop_duplicates().shape[0]

print(f"Sequences before filtering: {n_seq_before}")
print(f"Sequences after filtering:  {n_seq_after}")
print(f"Sequences dropped:          {n_seq_before - n_seq_after}")


# -----------------------------------------------------------------------------
# Minimum retained complete sequences per subject
# -----------------------------------------------------------------------------
df_raw_filtered, df_subject_sequence_summary = summarize_and_filter_subjects_by_sequence_count(
    df_long=df_raw_complete,
    min_sequences=min_sequences_per_subject,
)

df_subject_sequence_summary = df_subject_sequence_summary.sort_values(
    ["keep_subject", "n_complete_sequences", "id"],
    ascending=[True, True, True],
).reset_index(drop=True)

df_subject_sequence_summary.to_csv(
    path_out / f"subject_sequence_count_summary_min{min_sequences_per_subject}.csv",
    index=False,
)

n_subj_before = df_raw_complete["id"].nunique()
n_subj_after = df_raw_filtered["id"].nunique()

print("\nMinimum complete-sequence criterion:")
print(f"Minimum complete sequences per subject: {min_sequences_per_subject}")
print(f"Subjects before criterion: {n_subj_before}")
print(f"Subjects after criterion:  {n_subj_after}")
print(f"Subjects dropped:          {n_subj_before - n_subj_after}")

print("\nDropped subjects:")
print(
    df_subject_sequence_summary.loc[
        ~df_subject_sequence_summary["keep_subject"],
        ["id", "group", "n_complete_sequences"]
    ].to_string(index=False)
)

df_raw = df_raw_filtered.copy()


# -----------------------------------------------------------------------------
# Electrode list and topomap info
# -----------------------------------------------------------------------------
electrodes = sorted(df_raw["ch_name"].dropna().unique())

info = mne.create_info(
    electrodes,
    sfreq=1000,
    ch_types="eeg",
    verbose=None,
)
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, on_missing="warn", match_case=False)


# -----------------------------------------------------------------------------
# Run all measure/effect combinations
# -----------------------------------------------------------------------------
all_final_results = []

for measure in measures:
    print("\n" + "#" * 100)
    print(f"Preparing base dataframe for measure={measure}")

    df_base = prepare_base_dataframe(df_raw, measure=measure)

    print(f"Rows in df_base: {len(df_base)}")
    print(f"Subjects in df_base: {df_base['id'].nunique()}")

    for target_effect in target_effects:
        try:
            df_final_this = run_one_measure_effect(
                df_base=df_base,
                measure=measure,
                target_effect=target_effect,
                electrodes=electrodes,
                info=info,
                out_root=path_out,
            )

            if df_final_this is not None:
                all_final_results.append(df_final_this)

        except Exception as exc:
            print(f"FAILED: measure={measure}, target_effect={target_effect}")
            print(str(exc))

            fail_row = pd.DataFrame(
                {
                    "measure": [measure],
                    "target_effect": [target_effect],
                    "term": ["RUN_FAILED"],
                    "beta": [np.nan],
                    "se": [np.nan],
                    "t": [np.nan],
                    "p": [np.nan],
                    "random_effects": [np.nan],
                    "n_subjects": [np.nan],
                    "n_obs": [np.nan],
                    "llf": [np.nan],
                    "aic": [np.nan],
                    "bic": [np.nan],
                    "successful_loso_folds": [np.nan],
                    "total_subjects": [np.nan],
                    "n_successful_folds": [np.nan],
                    "mean_pairwise_r": [np.nan],
                    "median_pairwise_r": [np.nan],
                    "min_pairwise_r": [np.nan],
                    "max_pairwise_r": [np.nan],
                    "sd_pairwise_r": [np.nan],
                    "mean_r_to_mean_filter": [np.nan],
                    "median_r_to_mean_filter": [np.nan],
                    "sd_r_to_mean_filter": [np.nan],
                }
            )
            all_final_results.append(fail_row)


# -----------------------------------------------------------------------------
# Save master summary dataframe
# -----------------------------------------------------------------------------
if len(all_final_results) > 0:
    df_master = pd.concat(all_final_results, ignore_index=True)
    df_master.to_csv(
        path_out / "batch_loso_master_results.csv",
        index=False,
    )

    mask_target = df_master["term"] == df_master["target_effect"]
    df_target_only = df_master[mask_target].copy()
    df_target_only.to_csv(
        path_out / "batch_loso_target_term_results.csv",
        index=False,
    )

    print("\nSaved:")
    print(path_out / "batch_loso_master_results.csv")
    print(path_out / "batch_loso_target_term_results.csv")
else:
    print("No results to save.")