# -----------------------------------------------------------------------------
# Batch LOSO effect-weighted spatial filtering + final MLM
# Loops over EEG measures and target effects
# Matches final RT script:
#   - centered difficulty
#   - centered sequence number
#   - orthogonalized feedback terms: f_lin, f_quad
#   - sign alignment of fold filters to full-sample reference map
#   - selects one reference method from input file
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

file_in = path_in / "all_subjects_seq_fooof_rt_channelwise_long_with_reference.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
reference_to_use = "CSD"   # "CAR" or "CSD"

measures = [
    "exponent",
    "theta_flat",
    "alpha_flat",
    "beta_flat",
]

target_effects = [
    "group[T.experimental]",
    "f_lin",
    "f_quad",
    "group[T.experimental]:f_lin",
    "group[T.experimental]:f_quad",
]

#formula = """
#score ~ group * f_lin + group * f_quad
#        + mean_trial_difficulty_c + half + sequence_nr_c
#"""

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
    "sequence_nr_c",
]

min_subjects_per_electrode_model = 8
min_obs_per_electrode_model = 50

n_jobs_loso = -1
parallel_backend = "loky"
parallel_verbose = 10

n_bins_plot = 9


# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------
BASE_SEQ_COLS = [
    "id",
    "group",
    "block_nr",
    "sequence_nr",
    "sequence_nr_c",
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
    "sequence_nr_c",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_name(text: str) -> str:
    return text.replace("[", "").replace("]", "").replace(":", "x")


def get_sequence_key_cols(df_in: pd.DataFrame) -> list[str]:
    return (["window"] if "window" in df_in.columns else []) + BASE_SEQ_COLS


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
    value_col: str = "t",
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


def align_filter_to_reference(
    w: np.ndarray,
    w_ref: np.ndarray | None,
):
    if w_ref is None:
        return w

    valid = np.isfinite(w) & np.isfinite(w_ref)
    if valid.sum() <= 1:
        return w

    r = np.corrcoef(w[valid], w_ref[valid])[0, 1]
    if np.isfinite(r) and r < 0:
        return -w

    return w


def save_topomap(
    values: np.ndarray,
    info: mne.Info,
    title: str,
    out_file: Path,
):
    vmax = np.nanmax(np.abs(values))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig = plt.figure(figsize=(5, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    evoked = mne.EvokedArray(
        values[:, None],
        info,
        tmin=0.0,
        verbose=False,
    )

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

        evoked = mne.EvokedArray(
            topo[:, None],
            info,
            tmin=0.0,
            verbose=False,
        )

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
    sequence_ref = 0.0

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
                    "sequence_nr_c": sequence_ref,
                }
            )

    pred = pd.DataFrame(pred_rows)
    pred["pred"] = fit.predict(pred)

    group_colors = {
        "control": "#1f77b4",
        "experimental": "#d62728",
    }

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

    needed = [measure, "f", "group", "id", "ch_name", "mean_trial_difficulty", "sequence_nr", "half"]
    df = df.dropna(subset=needed).copy()

    df["f_c"] = df["f"] - df["f"].mean()
    df["f_lin"] = df["f_c"]
    df["f_quad"] = df["f_c"] ** 2 - np.mean(df["f_c"] ** 2)

    df["mean_trial_difficulty_c"] = (
        df["mean_trial_difficulty"] - df["mean_trial_difficulty"].mean()
    )
    df["sequence_nr_c"] = df["sequence_nr"] - df["sequence_nr"].mean()

    return df


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
        .reindex(electrodes)["t"]
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

    _, seq_meta_test, X_test = build_sequence_wide(
        df_long=df_test,
        measure=measure,
        electrodes=electrodes,
    )

    if X_test is None:
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": "no complete held-out sequence rows",
        }

    X_test_c = X_test - X_test.mean(axis=0)
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
    reference_to_use,
):
    print("\n" + "=" * 100)
    print(f"Running measure={measure}, target_effect={target_effect}, reference={reference_to_use}")

    effect_safe = safe_name(target_effect)
    combo_dir = out_root / reference_to_use / measure / effect_safe
    combo_dir.mkdir(parents=True, exist_ok=True)

    # full-data electrode MLM
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
        title=f"{measure} full-data electrode-wise MLM effect sizes (descriptive)\nReference: {reference_to_use}",
        out_file=combo_dir / f"{measure}_full_data_topomaps.png",
    )

    # reference map for sign alignment
    w_ref = build_reference_map(
        df_beta_full=df_beta_full,
        electrodes=electrodes,
        target_effect=target_effect,
        value_col="t",
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

    # mean LOSO filter topomap
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
        title=f"Mean LOSO filter\n{target_effect}\nReference: {reference_to_use}",
        out_file=combo_dir / f"{measure}_mean_loso_filter_{effect_safe}.png",
    )

    # final score model
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
        "sequence_nr",
        "sequence_nr_c",
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
    df_final["reference"] = reference_to_use

    df_final.to_csv(
        combo_dir / f"{measure}_loso_effect_weighted_final_mlm_{effect_safe}.csv",
        index=False,
    )

    # save model summary as text
    with open(combo_dir / f"{measure}_loso_effect_weighted_final_mlm_{effect_safe}.txt", "w") as f:
        f.write("Final LOSO effect-weighted score model\n")
        f.write(f"Measure: {measure}\n")
        f.write(f"Reference: {reference_to_use}\n")
        f.write(f"Target effect used to derive filters: {target_effect}\n")
        f.write(f"Random-effects structure: {used_re}\n")
        f.write(f"Successful LOSO folds: {len(all_scores)} / {len(subjects)}\n\n")
        f.write(str(fit.summary()))

    # score curve plot
    plot_score_model_curves_with_counts(
        df_model=dsub,
        fit=fit,
        outcome_name=f"{measure}_EW_score",
        out_file=combo_dir / f"{measure}_score_curves_{effect_safe}.png",
        n_bins=n_bins_plot,
    )

    print(f"Completed measure={measure}, target_effect={target_effect}, reference={reference_to_use}")
    print(f"Random-effects structure: {used_re}")
    print(f"Successful LOSO folds: {len(all_scores)} / {len(subjects)}")

    return df_final.copy()


# -----------------------------------------------------------------------------
# Load raw data once
# -----------------------------------------------------------------------------
df_raw = pd.read_csv(file_in)

df_raw["id"] = df_raw["id"].astype("category")
df_raw["group"] = df_raw["group"].astype("category")
df_raw["ch_name"] = df_raw["ch_name"].astype("category")
df_raw["half"] = df_raw["half"].astype("category")

if "window" in df_raw.columns:
    df_raw["window"] = df_raw["window"].astype("category")

if "reference" not in df_raw.columns:
    raise RuntimeError("Input file does not contain a 'reference' column.")

df_raw["reference"] = df_raw["reference"].astype("category")
df_raw["group"] = df_raw["group"].cat.set_categories(["control", "experimental"])

for col in [
    "f",
    "mean_trial_difficulty",
    "sequence_nr",
    "mean_rt",
    "mean_log_rt",
] + measures:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

df_raw = df_raw[df_raw["reference"] == reference_to_use].copy()

if len(df_raw) == 0:
    raise RuntimeError(f"No rows found for reference = {reference_to_use}")


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
    print(f"Preparing base dataframe for measure={measure}, reference={reference_to_use}")

    df_base = prepare_base_dataframe(df_raw, measure=measure)

    for target_effect in target_effects:
        try:
            df_final_this = run_one_measure_effect(
                df_base=df_base,
                measure=measure,
                target_effect=target_effect,
                electrodes=electrodes,
                info=info,
                out_root=path_out,
                reference_to_use=reference_to_use,
            )

            if df_final_this is not None:
                all_final_results.append(df_final_this)

        except Exception as exc:
            print(f"FAILED: measure={measure}, target_effect={target_effect}, reference={reference_to_use}")
            print(str(exc))

            fail_row = pd.DataFrame(
                {
                    "measure": [measure],
                    "target_effect": [target_effect],
                    "reference": [reference_to_use],
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
                }
            )
            all_final_results.append(fail_row)


# -----------------------------------------------------------------------------
# Save master summary dataframe
# -----------------------------------------------------------------------------
if len(all_final_results) > 0:
    df_master = pd.concat(all_final_results, ignore_index=True)
    df_master.to_csv(
        path_out / f"batch_loso_master_results_{reference_to_use}.csv",
        index=False,
    )

    mask_target = df_master["term"] == df_master["target_effect"]
    df_target_only = df_master[mask_target].copy()
    df_target_only.to_csv(
        path_out / f"batch_loso_target_term_results_{reference_to_use}.csv",
        index=False,
    )

    print("\nSaved:")
    print(path_out / f"batch_loso_master_results_{reference_to_use}.csv")
    print(path_out / f"batch_loso_target_term_results_{reference_to_use}.csv")
else:
    print("No results to save.")