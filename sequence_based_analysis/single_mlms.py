# -----------------------------------------------------------------------------
# Channelwise MLM topographies for sequence-based CAR dual-method data
#
# What this script does
# ---------------------
# 1. Loads channelwise sequence-based dual-method output.
# 2. Fits the same MLM separately for each channel and each EEG measure.
# 3. Extracts fixed-effect betas for terms of interest.
# 4. Plots scalp topographies:
#       rows = measures
#       cols = effects
#
# Notes
# -----
# - CAR only
# - non-time-resolved sequence data
# - same formula as previous MLM script
# - one model per channel x measure
# - topographies show beta coefficients (effect sizes), not p-values
# -----------------------------------------------------------------------------

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car_dualmethod.csv"

PATH_OUT = PATH_IN / "mlm_car_sequence_dualmethod_topographies"
PATH_OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REFERENCE = "car"

MEASURES_BY_METHOD = {
    "avgpsd": [
        "exponent_avgpsd",
        "theta_flat_avgpsd",
        "alpha_flat_avgpsd",
        "beta_flat_avgpsd",
    ],
    "trialavg": [
        "exponent_trialavg",
        "theta_flat_trialavg",
        "alpha_flat_trialavg",
        "beta_flat_trialavg",
    ],
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
    "group[T.experimental]",
    "f",
    "f2",
    "group[T.experimental]:f",
    "group[T.experimental]:f2",
    "half[T.second]",
    "mean_trial_difficulty",
]

FIT_METHOD = "lbfgs"
FIT_REML = False
FIT_MAXITER = 4000
MIN_SUBJECTS = 8

PLOT_ABSOLUTE_VLIM_PER_EFFECT = True
CMAP = "RdBu_r"
FIGSIZE_PER_CELL = (2.6, 2.4)

CHANNEL_LABELS = (
    Path("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt")
    .read_text()
    .splitlines()
)

INFO_ERP = mne.create_info(CHANNEL_LABELS, sfreq=500, ch_types="eeg", verbose=None)
MONTAGE = mne.channels.make_standard_montage("standard_1020")
INFO_ERP.set_montage(MONTAGE, on_missing="warn", match_case=False)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fit_mixedlm_with_fallbacks(d: pd.DataFrame, measure_col: str):
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


def extract_fixed_effects(fit, measure: str, ch_name: str, re_formula: str):
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
                "measure": measure,
                "ch_name": ch_name,
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


def plot_topography_grid(df_effects: pd.DataFrame, measures: list[str], method_name: str):
    """
    Plot rows = measures, cols = terms.
    Values = beta coefficients.
    """
    n_rows = len(measures)
    n_cols = len(TERMS_OF_INTEREST)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIGSIZE_PER_CELL[0] * n_cols, FIGSIZE_PER_CELL[1] * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    fig.suptitle(f"MLM beta topographies | {method_name}", fontsize=14)

    for col_ix, term in enumerate(TERMS_OF_INTEREST):
        # common symmetric scale per term across all rows
        term_vals = df_effects.loc[df_effects["term"] == term, "beta"].to_numpy(dtype=float)
        term_vals = term_vals[np.isfinite(term_vals)]
        if term_vals.size == 0:
            vlim = 1.0
        else:
            if PLOT_ABSOLUTE_VLIM_PER_EFFECT:
                vlim = np.nanmax(np.abs(term_vals))
                if not np.isfinite(vlim) or vlim == 0:
                    vlim = 1.0
            else:
                vlim = None

        for row_ix, measure in enumerate(measures):
            ax = axes[row_ix, col_ix]

            dsub = df_effects[
                (df_effects["measure"] == measure) &
                (df_effects["term"] == term)
            ].copy()

            vals = np.full(len(CHANNEL_LABELS), np.nan, dtype=float)
            for i, ch in enumerate(CHANNEL_LABELS):
                hit = dsub.loc[dsub["ch_name"] == ch, "beta"]
                if len(hit):
                    vals[i] = float(hit.iloc[0])

            mask = np.isfinite(vals)

            mne.viz.plot_topomap(
                vals,
                INFO_ERP,
                axes=ax,
                show=False,
                cmap=CMAP,
                vlim=(-vlim, vlim) if vlim is not None else (None, None),
                mask=mask,
                contours=0,
            )

            if row_ix == 0:
                ax.set_title(term, fontsize=10)
            if col_ix == 0:
                ax.set_ylabel(measure, fontsize=10)

    return fig


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
    "f",
    "mean_trial_difficulty",
    "mean_rt",
    "exponent_avgpsd",
    "theta_flat_avgpsd",
    "alpha_flat_avgpsd",
    "beta_flat_avgpsd",
    "exponent_trialavg",
    "theta_flat_trialavg",
    "alpha_flat_trialavg",
    "beta_flat_trialavg",
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
    "f",
    "mean_trial_difficulty",
    "mean_rt",
    "exponent_avgpsd",
    "theta_flat_avgpsd",
    "alpha_flat_avgpsd",
    "beta_flat_avgpsd",
    "exponent_trialavg",
    "theta_flat_trialavg",
    "alpha_flat_trialavg",
    "beta_flat_trialavg",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["f2"] = df["f"] ** 2
df = df[df["reference"] == REFERENCE].copy()

if df.empty:
    raise RuntimeError(f"No rows left after filtering reference == '{REFERENCE}'.")

print(f"Rows: {len(df)}")
print(f"Subjects: {df['id'].nunique()}")
print(f"Channels: {df['ch_name'].nunique()}")


# -----------------------------------------------------------------------------
# Run channelwise MLMs
# -----------------------------------------------------------------------------
model_rows = []
result_rows = []

all_measures = MEASURES_BY_METHOD["avgpsd"] + MEASURES_BY_METHOD["trialavg"]

for measure_col in all_measures:
    for ch_name in CHANNEL_LABELS:
        dsub = df[df["ch_name"] == ch_name].copy()

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

        print(f"Running {measure_col} | ch={ch_name} | rows={n_rows}, subjects={n_subjects}")

        if n_subjects < MIN_SUBJECTS:
            model_rows.append(
                {
                    "measure": measure_col,
                    "ch_name": ch_name,
                    "status": "skipped_too_few_subjects",
                    "n_rows": n_rows,
                    "n_subjects": n_subjects,
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
                    "measure": measure_col,
                    "ch_name": ch_name,
                    "status": "fit_failed",
                    "n_rows": n_rows,
                    "n_subjects": n_subjects,
                    "re_formula": None,
                    "converged": False,
                    "error": err,
                }
            )
            print(f"  FAILED: {err}")
            continue

        model_rows.append(
            {
                "measure": measure_col,
                "ch_name": ch_name,
                "status": "ok",
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "re_formula": used_re_formula,
                "converged": bool(getattr(fit, "converged", False)),
                "error": None,
                "llf": float(fit.llf) if hasattr(fit, "llf") and pd.notna(fit.llf) else np.nan,
                "aic": float(fit.aic) if hasattr(fit, "aic") and pd.notna(fit.aic) else np.nan,
                "bic": float(fit.bic) if hasattr(fit, "bic") and pd.notna(fit.bic) else np.nan,
            }
        )

        result_rows.extend(
            extract_fixed_effects(
                fit=fit,
                measure=measure_col,
                ch_name=ch_name,
                re_formula=used_re_formula,
            )
        )

        print(f"  OK | re_formula={used_re_formula}")


# -----------------------------------------------------------------------------
# Save tables
# -----------------------------------------------------------------------------
df_models = pd.DataFrame(model_rows)
df_results = pd.DataFrame(result_rows)

df_models.to_csv(PATH_OUT / "mlm_channelwise_model_summary.csv", index=False)
df_results.to_csv(PATH_OUT / "mlm_channelwise_fixed_effects_all_terms.csv", index=False)

df_effects = df_results[df_results["term"].isin(TERMS_OF_INTEREST)].copy()
df_effects.to_csv(PATH_OUT / "mlm_channelwise_fixed_effects_terms_of_interest.csv", index=False)


# -----------------------------------------------------------------------------
# Plot topographies
# -----------------------------------------------------------------------------
for method_name, measures in MEASURES_BY_METHOD.items():
    dplot = df_effects[df_effects["measure"].isin(measures)].copy()

    fig = plot_topography_grid(
        df_effects=dplot,
        measures=measures,
        method_name=method_name,
    )
    fig.savefig(PATH_OUT / f"mlm_topography_grid_{method_name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(PATH_OUT / f"mlm_topography_grid_{method_name}.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------
print("\nFinished.")
print("Output directory:", PATH_OUT)
print("\nSaved files:")
print(" - mlm_channelwise_model_summary.csv")
print(" - mlm_channelwise_fixed_effects_all_terms.csv")
print(" - mlm_channelwise_fixed_effects_terms_of_interest.csv")
print(" - mlm_topography_grid_avgpsd.png / .pdf")
print(" - mlm_topography_grid_trialavg.png / .pdf")