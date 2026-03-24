# -----------------------------------------------------------------------------
# MLM overview for time-resolved CAR data:
#   - RT model once globally
#   - EEG models once per measure, using average across all electrodes
#     at a selected timepoint (window_center nearest TARGET_TIME)
#
# What this script does
# ---------------------
# 1. Loads the time-resolved CAR output file.
# 2. Selects the window_center nearest TARGET_TIME.
# 3. Aggregates EEG measures across all electrodes within:
#       id x block_nr x sequence_nr x selected time window
# 4. Runs the same MLM for:
#       - rt_mean (once globally)
#       - exponent
#       - theta_flat
#       - alpha_flat
#       - beta_flat
# 5. Saves:
#       - model summaries
#       - fixed effects for all terms
#       - terms-of-interest table
#       - wide beta table
#       - wide p-value table
#       - binarized p-value table
#
# Notes
# -----
# - CAR only
# - no subject exclusion
# - RT is deduplicated across electrodes and taken once per sequence
# - EEG is averaged across all electrodes at the selected timepoint
# -----------------------------------------------------------------------------

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_timeresolved_car.csv"

PATH_OUT = PATH_IN / "mlm_car_time_resolved_global"
PATH_OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REFERENCE = "car"
TARGET_TIME = -0.5

MEASURES_EEG = {
    "exponent": "exponent",
    "theta_flat": "theta_flat",
    "alpha_flat": "alpha_flat",
    "beta_flat": "beta_flat",
}

MEASURE_RT = {
    "rt_mean": "mean_rt",
}

FORMULA = """
score ~ group * f + group * f2
        + mean_trial_difficulty + half
"""

RE_FORMULAS = [
    "1 + f + f2",
    "1 + f",
    "1",
]

TERMS_OF_INTEREST = [
    "Intercept",
    "f",
    "f2",
    "group[T.experimental]",
    "group[T.experimental]:f",
    "group[T.experimental]:f2",
    "half[T.second]",
    "mean_trial_difficulty",
]

FIT_METHOD = "lbfgs"
FIT_REML = False
FIT_MAXITER = 4000
MIN_SUBJECTS = 8

P_THRESHOLD = 0.05


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fit_mixedlm_with_fallbacks(d: pd.DataFrame, measure_col: str):
    """
    Fit MixedLM using fallback random-effects structures.

    Returns
    -------
    fit : fitted model or None
    used_re_formula : str or None
    error_message : str or None
    """
    d = d.copy().rename(columns={measure_col: "score"})
    last_error = None

    for re_formula in RE_FORMULAS:
        try:
            model = smf.mixedlm(
                FORMULA,
                d,
                groups=d["id"],
                re_formula=re_formula,
            )

            fit = model.fit(
                method=FIT_METHOD,
                reml=FIT_REML,
                maxiter=FIT_MAXITER,
                disp=False,
            )

            if getattr(fit, "converged", False):
                return fit, re_formula, None

            last_error = f"Fit did not converge for re_formula={re_formula}"

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

    return None, None, last_error


def extract_fixed_effects(fit, analysis_name: str, target_label: str, re_formula: str):
    """
    Extract fixed effects from a fitted MixedLM result.
    """
    rows = []

    fe_params = fit.fe_params
    bse = fit.bse_fe if hasattr(fit, "bse_fe") else pd.Series(index=fe_params.index, dtype=float)
    tvalues = (
        fit.tvalues.reindex(fe_params.index)
        if hasattr(fit, "tvalues")
        else pd.Series(index=fe_params.index, dtype=float)
    )
    pvalues = (
        fit.pvalues.reindex(fe_params.index)
        if hasattr(fit, "pvalues")
        else pd.Series(index=fe_params.index, dtype=float)
    )

    conf = fit.conf_int()
    if isinstance(conf, pd.DataFrame) and conf.shape[1] >= 2:
        conf = conf.iloc[:, :2].copy()
        conf.columns = ["ci_low", "ci_high"]
        conf = conf.reindex(fe_params.index)
    else:
        conf = pd.DataFrame(index=fe_params.index, columns=["ci_low", "ci_high"], dtype=float)

    for term in fe_params.index:
        rows.append(
            {
                "measure": analysis_name,
                "target": target_label,
                "term": term,
                "beta": float(fe_params[term]),
                "se": float(bse[term]) if term in bse.index and pd.notna(bse[term]) else np.nan,
                "t": float(tvalues[term]) if term in tvalues.index and pd.notna(tvalues[term]) else np.nan,
                "p": float(pvalues[term]) if term in pvalues.index and pd.notna(pvalues[term]) else np.nan,
                "ci_low": float(conf.loc[term, "ci_low"]) if term in conf.index and pd.notna(conf.loc[term, "ci_low"]) else np.nan,
                "ci_high": float(conf.loc[term, "ci_high"]) if term in conf.index and pd.notna(conf.loc[term, "ci_high"]) else np.nan,
                "re_formula": re_formula,
                "converged": bool(getattr(fit, "converged", False)),
                "llf": float(fit.llf) if hasattr(fit, "llf") and pd.notna(fit.llf) else np.nan,
                "aic": float(fit.aic) if hasattr(fit, "aic") and pd.notna(fit.aic) else np.nan,
                "bic": float(fit.bic) if hasattr(fit, "bic") and pd.notna(fit.bic) else np.nan,
                "nobs": int(getattr(fit, "nobs", np.nan)) if pd.notna(getattr(fit, "nobs", np.nan)) else np.nan,
            }
        )

    return rows


def choose_nearest_timepoint(df: pd.DataFrame, target_time: float) -> float:
    available = np.sort(df["window_center"].dropna().unique().astype(float))
    if available.size == 0:
        raise RuntimeError("No window_center values available.")
    return float(available[np.argmin(np.abs(available - target_time))])


def aggregate_eeg_global(df_time: pd.DataFrame, measure_col: str) -> pd.DataFrame:
    """
    Average one EEG measure across electrodes for each:
        id x block_nr x sequence_nr x window_center
    """
    needed = [
        "id",
        "group",
        "half",
        "block_nr",
        "sequence_nr",
        "window_center",
        "f",
        "f2",
        "mean_trial_difficulty",
        measure_col,
    ]
    missing = [c for c in needed if c not in df_time.columns]
    if missing:
        raise ValueError(f"Missing required EEG columns: {missing}")

    group_cols = ["id", "group", "half", "block_nr", "sequence_nr", "window_center"]

    out = (
        df_time[group_cols + ["f", "f2", "mean_trial_difficulty", measure_col]]
        .groupby(group_cols, observed=True)
        .agg(
            f=("f", "first"),
            f2=("f2", "first"),
            mean_trial_difficulty=("mean_trial_difficulty", "first"),
            score=(measure_col, "mean"),
        )
        .reset_index()
        .rename(columns={"score": measure_col})
    )

    return out


def aggregate_rt_global(df_time: pd.DataFrame, measure_col: str) -> pd.DataFrame:
    """
    Deduplicate RT across electrodes for each:
        id x block_nr x sequence_nr x window_center
    """
    needed = [
        "id",
        "group",
        "half",
        "block_nr",
        "sequence_nr",
        "window_center",
        "f",
        "f2",
        "mean_trial_difficulty",
        measure_col,
    ]
    missing = [c for c in needed if c not in df_time.columns]
    if missing:
        raise ValueError(f"Missing required RT columns: {missing}")

    group_cols = ["id", "group", "half", "block_nr", "sequence_nr", "window_center"]

    out = (
        df_time[group_cols + ["f", "f2", "mean_trial_difficulty", measure_col]]
        .groupby(group_cols, observed=True)
        .agg(
            f=("f", "first"),
            f2=("f2", "first"),
            mean_trial_difficulty=("mean_trial_difficulty", "first"),
            score=(measure_col, "first"),
        )
        .reset_index()
        .rename(columns={"score": measure_col})
    )

    return out


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(FILE_IN)

required_columns = [
    "id",
    "group",
    "half",
    "reference",
    "ch_name",
    "block_nr",
    "sequence_nr",
    "window_center",
    "f",
    "mean_trial_difficulty",
    "mean_rt",
    "exponent",
    "theta_flat",
    "alpha_flat",
    "beta_flat",
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["id"] = df["id"].astype("category")
df["group"] = df["group"].astype("category")
df["half"] = df["half"].astype("category")
df["reference"] = df["reference"].astype("category")
df["ch_name"] = df["ch_name"].astype("category")

df["group"] = df["group"].cat.set_categories(["control", "experimental"])

for c in [
    "window_center",
    "f",
    "mean_trial_difficulty",
    "mean_rt",
    "exponent",
    "theta_flat",
    "alpha_flat",
    "beta_flat",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["f2"] = df["f"] ** 2

df = df[df["reference"] == REFERENCE].copy()

if df.empty:
    raise RuntimeError(f"No rows left after filtering reference == '{REFERENCE}'.")

chosen_time = choose_nearest_timepoint(df, TARGET_TIME)
df_time = df[np.isclose(df["window_center"], chosen_time)].copy()

if df_time.empty:
    raise RuntimeError(f"No rows found for selected timepoint {chosen_time}.")

print(f"Filtered rows ({REFERENCE} only): {len(df)}")
print(f"Selected target time: {TARGET_TIME}")
print(f"Nearest available window_center: {chosen_time}")
print(f"Rows at selected time: {len(df_time)}")
print(f"Subjects: {df_time['id'].nunique()}")
print(f"Electrodes present: {df_time['ch_name'].nunique()}")


# -----------------------------------------------------------------------------
# Run MLMs
# -----------------------------------------------------------------------------
results_rows = []
model_rows = []

# -------------------------------------------------------------------------
# EEG models: run once per measure on global across-electrode average
# -------------------------------------------------------------------------
for analysis_name, measure_col in MEASURES_EEG.items():
    dsub = aggregate_eeg_global(df_time, measure_col=measure_col)

    needed = [
        measure_col,
        "f",
        "f2",
        "group",
        "id",
        "half",
        "mean_trial_difficulty",
    ]
    dsub = dsub.dropna(subset=needed).copy()

    n_subjects = dsub["id"].nunique()
    n_rows = len(dsub)

    print(
        f"Running {analysis_name} @ global_eeg_avg | "
        f"time={chosen_time}, rows={n_rows}, subjects={n_subjects}"
    )

    if n_subjects < MIN_SUBJECTS:
        model_rows.append(
            {
                "measure": analysis_name,
                "target": "global_eeg_avg",
                "status": "skipped_too_few_subjects",
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "selected_time": chosen_time,
                "re_formula": None,
                "converged": False,
                "error": "Too few subjects for MLM",
            }
        )
        continue

    fit, used_re_formula, err = fit_mixedlm_with_fallbacks(dsub, measure_col)

    if fit is None:
        model_rows.append(
            {
                "measure": analysis_name,
                "target": "global_eeg_avg",
                "status": "fit_failed",
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "selected_time": chosen_time,
                "re_formula": None,
                "converged": False,
                "error": err,
            }
        )
        print(f"  FAILED: {err}")
        continue

    model_rows.append(
        {
            "measure": analysis_name,
            "target": "global_eeg_avg",
            "status": "ok",
            "n_rows": n_rows,
            "n_subjects": n_subjects,
            "selected_time": chosen_time,
            "re_formula": used_re_formula,
            "converged": bool(getattr(fit, "converged", False)),
            "error": None,
            "llf": float(fit.llf) if hasattr(fit, "llf") and pd.notna(fit.llf) else np.nan,
            "aic": float(fit.aic) if hasattr(fit, "aic") and pd.notna(fit.aic) else np.nan,
            "bic": float(fit.bic) if hasattr(fit, "bic") and pd.notna(fit.bic) else np.nan,
        }
    )

    results_rows.extend(
        extract_fixed_effects(
            fit=fit,
            analysis_name=analysis_name,
            target_label="global_eeg_avg",
            re_formula=used_re_formula,
        )
    )

    print(f"  OK | re_formula={used_re_formula}")


# -------------------------------------------------------------------------
# RT model: run once globally at the same selected timepoint
# -------------------------------------------------------------------------
for analysis_name, measure_col in MEASURE_RT.items():
    dsub = aggregate_rt_global(df_time, measure_col=measure_col)

    needed = [
        measure_col,
        "f",
        "f2",
        "group",
        "id",
        "half",
        "mean_trial_difficulty",
    ]
    dsub = dsub.dropna(subset=needed).copy()

    n_subjects = dsub["id"].nunique()
    n_rows = len(dsub)

    print(
        f"Running {analysis_name} @ global_rt | "
        f"time={chosen_time}, rows={n_rows}, subjects={n_subjects}"
    )

    if n_subjects < MIN_SUBJECTS:
        model_rows.append(
            {
                "measure": analysis_name,
                "target": "global_rt",
                "status": "skipped_too_few_subjects",
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "selected_time": chosen_time,
                "re_formula": None,
                "converged": False,
                "error": "Too few subjects for MLM",
            }
        )
        continue

    fit, used_re_formula, err = fit_mixedlm_with_fallbacks(dsub, measure_col)

    if fit is None:
        model_rows.append(
            {
                "measure": analysis_name,
                "target": "global_rt",
                "status": "fit_failed",
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "selected_time": chosen_time,
                "re_formula": None,
                "converged": False,
                "error": err,
            }
        )
        print(f"  FAILED: {err}")
        continue

    model_rows.append(
        {
            "measure": analysis_name,
            "target": "global_rt",
            "status": "ok",
            "n_rows": n_rows,
            "n_subjects": n_subjects,
            "selected_time": chosen_time,
            "re_formula": used_re_formula,
            "converged": bool(getattr(fit, "converged", False)),
            "error": None,
            "llf": float(fit.llf) if hasattr(fit, "llf") and pd.notna(fit.llf) else np.nan,
            "aic": float(fit.aic) if hasattr(fit, "aic") and pd.notna(fit.aic) else np.nan,
            "bic": float(fit.bic) if hasattr(fit, "bic") and pd.notna(fit.bic) else np.nan,
        }
    )

    results_rows.extend(
        extract_fixed_effects(
            fit=fit,
            analysis_name=analysis_name,
            target_label="global_rt",
            re_formula=used_re_formula,
        )
    )

    print(f"  OK | re_formula={used_re_formula}")


# -----------------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------------
df_models = pd.DataFrame(model_rows)
df_results = pd.DataFrame(results_rows)

if df_models.empty:
    raise RuntimeError("No model output produced.")

df_models.to_csv(PATH_OUT / f"mlm_model_summary_time_{chosen_time:+.1f}s.csv", index=False)

if df_results.empty:
    raise RuntimeError("No successful fixed-effect results to save.")

df_results["selected_time"] = chosen_time
df_results.to_csv(PATH_OUT / f"mlm_fixed_effects_all_terms_time_{chosen_time:+.1f}s.csv", index=False)

df_effects = df_results[df_results["term"].isin(TERMS_OF_INTEREST)].copy()
df_effects.to_csv(
    PATH_OUT / f"mlm_fixed_effects_terms_of_interest_time_{chosen_time:+.1f}s.csv",
    index=False,
)

# Wide beta table
df_beta_wide = (
    df_effects
    .pivot_table(
        index=["measure", "target"],
        columns="term",
        values="beta",
        aggfunc="first",
    )
    .reset_index()
)
df_beta_wide.to_csv(PATH_OUT / f"mlm_betas_wide_time_{chosen_time:+.1f}s.csv", index=False)

# Wide p-value table
df_p_wide = (
    df_effects
    .pivot_table(
        index=["measure", "target"],
        columns="term",
        values="p",
        aggfunc="first",
    )
    .reset_index()
)
df_p_wide.to_csv(PATH_OUT / f"mlm_pvalues_wide_time_{chosen_time:+.1f}s.csv", index=False)

# Binarized p-value table
df_p_binary = df_p_wide.copy()
for col in df_p_binary.columns:
    if col not in ["measure", "target"]:
        df_p_binary[col] = (df_p_binary[col] < P_THRESHOLD).astype(int)

df_p_binary.to_csv(
    PATH_OUT / f"mlm_pvalues_binary_{str(P_THRESHOLD).replace('.', 'p')}_time_{chosen_time:+.1f}s.csv",
    index=False,
)

# Save the aggregated data actually used for fitting
df_rt_used = aggregate_rt_global(df_time, measure_col="mean_rt").copy()
df_eeg_used = []

for analysis_name, measure_col in MEASURES_EEG.items():
    dtmp = aggregate_eeg_global(df_time, measure_col=measure_col).copy()
    dtmp["measure"] = analysis_name
    dtmp = dtmp.rename(columns={measure_col: "score"})
    df_eeg_used.append(dtmp)

df_eeg_used = pd.concat(df_eeg_used, ignore_index=True)

df_rt_used.to_csv(PATH_OUT / f"mlm_input_rt_time_{chosen_time:+.1f}s.csv", index=False)
df_eeg_used.to_csv(PATH_OUT / f"mlm_input_eeg_global_time_{chosen_time:+.1f}s.csv", index=False)


# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------
print("\nFinished.")
print("Output directory:", PATH_OUT)
print("Requested target time:", TARGET_TIME)
print("Selected window_center:", chosen_time)

print("\nModel status summary:")
print(
    df_models[
        ["measure", "target", "status", "re_formula", "n_rows", "n_subjects", "selected_time"]
    ].to_string(index=False)
)

print("\nSaved files:")
print(f" - mlm_model_summary_time_{chosen_time:+.1f}s.csv")
print(f" - mlm_fixed_effects_all_terms_time_{chosen_time:+.1f}s.csv")
print(f" - mlm_fixed_effects_terms_of_interest_time_{chosen_time:+.1f}s.csv")
print(f" - mlm_betas_wide_time_{chosen_time:+.1f}s.csv")
print(f" - mlm_pvalues_wide_time_{chosen_time:+.1f}s.csv")
print(f" - mlm_pvalues_binary_{str(P_THRESHOLD).replace('.', 'p')}_time_{chosen_time:+.1f}s.csv")
print(f" - mlm_input_rt_time_{chosen_time:+.1f}s.csv")
print(f" - mlm_input_eeg_global_time_{chosen_time:+.1f}s.csv")