# -----------------------------------------------------------------------------
# Slim ROI mixed model + feedback plots with:
# 1) covariate-adjusted/model feedback curves for multiple RT/FOOOF/ERP measures
# 2) raw PSD by feedback bin
# 3) flattened spectrum by feedback bin
# 4) ERP waveform by feedback bin
#
# Notes:
# - MEASURES is a list of dependent variables for MLMs
# - For each measure:
#     * fit MLM
#     * save coefficient table rows into one combined dataframe
#     * create a feedback/model fit plot
# - PSD / flattened / ERP waveform are made once only,
#   using PLOT_MEASURE only for titles / filenames
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/roi_models/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
MEASURES = [
    #"mean_rt",
    "cnv_mean",
    #"theta_flat",
    #"alpha_flat",
    #"exponent",
]

# Used only for shared plots that are not re-run for every modeled measure
PLOT_MEASURE = "exponent"

ROIS = {
    "frontocentral": ["Fz", "F1", "F2"],
    "central": ["C1", "Cz", "C2"],
    "posterior": ["POz", "PO3", "PO4"],
}

TERMS_EXCLUDE_FROM_SIG = [
    "Intercept",
    "half[T.second]",
    "mean_trial_difficulty_c",
]

N_BINS = 9

RAW_PSD_PLOT_FMIN = 0.0
RAW_PSD_PLOT_FMAX = 20.0

FLAT_PLOT_FMIN = 1.0
FLAT_PLOT_FMAX = 20.0

ERP_PLOT_TMIN = -1.4
ERP_PLOT_TMAX = 0.0
ERP_YLIM = None  # example: (-5, 5), or None for automatic scaling
ERP_INVERT_Y = True  # conventional ERP plotting: negativity upward

FORMULA = """
roi_val ~ group * f + group * f2
          + mean_trial_difficulty_c + half
"""

RE_FORMULAS = [
    "1 + f + f2",
    "1 + f",
    "1",
]

SIGNIFICANT_RESULTS_EXCLUDE_INTERCEPT = True


# -----------------------------------------------------------------------------
# MixedLM helper
# -----------------------------------------------------------------------------
def fit_mixedlm_with_fallback(df_model, formula, re_formulas):
    fit = None
    used_re = None
    fit_error_log = []

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
                fit_error_log.append(f"{re_formula}: converged=False")

        except Exception as exc:
            fit_error_log.append(f"{re_formula}: {exc}")

    return fit, used_re, fit_error_log









# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def make_feedback_bin_edges(df_model, n_bins):
    return np.linspace(df_model["f"].min(), df_model["f"].max(), n_bins + 1)


def add_bottom_colorbar(fig, axes, norm, cmap, label):
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.04])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(label)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    return cbar


# -----------------------------------------------------------------------------
# Plot helper: covariate-adjusted observed/model curves
# -----------------------------------------------------------------------------
def plot_feedback_curves(
    df_model,
    fit,
    outcome_name,
    n_bins=8,
    path_out=None,
    file_tag=None,
):
    """
    Plot model curves and binned observed means on the same covariate scale.

    Model curves are predictions at:
        mean_trial_difficulty_c = 0
        half = modal/reference half

    Binned points are adjusted to that same covariate setting by subtracting
    the row-wise nuisance contribution implied by the fitted model:

        adjusted = observed - (prediction_actual_covariates - prediction_ref_covariates)

    This leaves the group/f/f2 structure intact while removing half and
    difficulty effects from the binned means.
    """
    d = df_model.copy()
    group_order = ["control", "experimental"]

    edges = make_feedback_bin_edges(d, n_bins)
    d["f_bin"] = pd.cut(d["f"], bins=edges, include_lowest=True)

    f_grid = np.linspace(d["f"].min(), d["f"].max(), 300)
    f_mean = float(d["f"].mean())
    f2_grid = (f_grid - f_mean) ** 2 - np.mean((d["f"] - f_mean) ** 2)

    difficulty_ref = 0.0
    half_ref = d["half"].mode().iloc[0]

    # Covariate-adjust observed rows to the same nuisance-covariate setting
    # as the model curves.
    d_ref = d.copy()
    d_ref["mean_trial_difficulty_c"] = difficulty_ref
    d_ref["half"] = half_ref

    pred_actual = fit.predict(d)
    pred_ref = fit.predict(d_ref)
    d["roi_val_adjusted"] = d["roi_val"] - (pred_actual - pred_ref)

    agg = (
        d.groupby(["group", "f_bin"], observed=True)
        .agg(
            mean_score=("roi_val_adjusted", "mean"),
            sem_score=("roi_val_adjusted", "sem"),
        )
        .reset_index()
    )

    agg["f_mid"] = agg["f_bin"].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    agg = agg.sort_values(["group", "f_mid"]).reset_index(drop=True)

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

    group_colors = {
        "control": "#1f77b4",
        "experimental": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

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
            label=f"{group_name} adjusted observed",
        )

        ax.plot(
            dg_pred["f"],
            dg_pred["pred"],
            linewidth=3,
            color=color,
            label=f"{group_name} model",
        )

    ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Signed feedback (f)")
    ax.set_ylabel(outcome_name)
    ax.set_title(f"{outcome_name}: adjusted bin means and model curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if path_out is not None:
        if file_tag is None:
            file_tag = outcome_name
        fig.savefig(
            path_out / f"{file_tag}_feedback_curves_adjusted.png",
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()

# -----------------------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_IN)

df["id"] = df["id"].astype(str)
df["group"] = pd.Categorical(df["group"], categories=["control", "experimental"])
df["half"] = pd.Categorical(df["half"])
df["ch_name"] = df["ch_name"].astype(str)

numeric_candidates = list(
    set(
        MEASURES
        + [PLOT_MEASURE]
        + [
            "f",
            "mean_trial_difficulty",
            "offset",
            "exponent",
        ]
    )
)

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

missing_measures = [m for m in MEASURES if m not in df.columns]
if missing_measures:
    raise ValueError(f"These measures are missing from the dataframe: {missing_measures}")

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
seq_meta["f2"] = seq_meta["f_c"] ** 2 - (seq_meta["f_c"] ** 2).mean()
seq_meta["mean_trial_difficulty_c"] = (
    seq_meta["mean_trial_difficulty"] - seq_meta["mean_trial_difficulty"].mean()
)

print("ROIs:")
for roi_name, roi_channels in ROIS.items():
    print(f"{roi_name}: {roi_channels}")

print("Measures to model:", MEASURES)
print("Subjects:", df["id"].nunique())


# -----------------------------------------------------------------------------
# Fit models for all ROIs and all measures
# -----------------------------------------------------------------------------
all_results = []
model_data_by_roi_measure = {}
fit_by_roi_measure = {}

for roi_name, roi_channels in ROIS.items():

    print(f"\n================ ROI: {roi_name} ================")
    print("Electrodes:", roi_channels)

    for measure in MEASURES:
        print(f"\n--- Fitting ROI={roi_name} | measure={measure} ---")

        df_roi = (
            df[df["ch_name"].isin(roi_channels)]
            .groupby(["id", "block_nr", "sequence_nr"], as_index=False)[measure]
            .mean()
            .rename(columns={measure: "roi_val"})
        )

        d = seq_meta.merge(
            df_roi,
            on=["id", "block_nr", "sequence_nr"],
            how="inner",
        )

        d = d.dropna(
            subset=[
                "roi_val",
                "group",
                "id",
                "half",
                "f",
                "f2",
                "mean_trial_difficulty_c",
            ]
        ).copy()

        print("Rows:", len(d))
        print("Subjects:", d["id"].nunique())

        fit, used_re, fit_error_log = fit_mixedlm_with_fallback(
            df_model=d,
            formula=FORMULA,
            re_formulas=RE_FORMULAS,
        )

        if fit is None:
            print("Model failed:")
            for msg in fit_error_log:
                print(msg)

            fail_row = pd.DataFrame(
                {
                    "term": [np.nan],
                    "beta": [np.nan],
                    "se": [np.nan],
                    "z": [np.nan],
                    "p": [np.nan],
                    "p_fdr": [np.nan],
                    "significant_fdr_05": [False],
                    "random_effects": [np.nan],
                    "n_subjects": [d["id"].nunique()],
                    "n_obs": [len(d)],
                    "llf": [np.nan],
                    "aic": [np.nan],
                    "bic": [np.nan],
                    "measure": [measure],
                    "roi_name": [roi_name],
                    "roi_channels": [",".join(roi_channels)],
                    "model_failed": [True],
                    "fit_error_log": [" | ".join(fit_error_log)],
                }
            )
            all_results.append(fail_row)
            continue

        print(fit.summary())
        print("Random-effects structure:", used_re)

        fe = fit.fe_params
        se = fit.bse_fe.reindex(fe.index)
        zvals = fe / se.replace(0, np.nan)
        pvals = fit.pvalues.reindex(fe.index)

        df_res = pd.DataFrame(
            {
                "term": fe.index,
                "beta": fe.values,
                "se": se.values,
                "z": zvals.values,
                "p": pvals.values,
                "p_fdr": np.nan,
                "significant_fdr_05": False,
                "random_effects": used_re,
                "n_subjects": d["id"].nunique(),
                "n_obs": len(d),
                "llf": fit.llf,
                "aic": fit.aic if np.isfinite(fit.aic) else np.nan,
                "bic": fit.bic if np.isfinite(fit.bic) else np.nan,
                "measure": measure,
                "roi_name": roi_name,
                "roi_channels": ",".join(roi_channels),
                "model_failed": False,
                "fit_error_log": "",
            }
        )

        all_results.append(df_res)

        model_data_by_roi_measure[(roi_name, measure)] = d
        fit_by_roi_measure[(roi_name, measure)] = fit


combined_results = pd.concat(all_results, ignore_index=True)


# -----------------------------------------------------------------------------
# FDR correction
# -----------------------------------------------------------------------------
# FDR is applied only to interpretable task terms, excluding intercept/covariates.
# This avoids half/difficulty dominating the significant-only table.
# -----------------------------------------------------------------------------
fdr_mask = (
    (combined_results["model_failed"] == False)
    & (combined_results["p"].notna())
    & (~combined_results["term"].isin(TERMS_EXCLUDE_FROM_SIG))
)

pvals_for_fdr = combined_results.loc[fdr_mask, "p"].astype(float).values

if len(pvals_for_fdr) > 0:
    reject, p_fdr, _, _ = multipletests(
        pvals_for_fdr,
        alpha=0.05,
        method="fdr_bh",
    )

    combined_results.loc[fdr_mask, "p_fdr"] = p_fdr
    combined_results.loc[fdr_mask, "significant_fdr_05"] = reject


combined_results.to_csv(
    PATH_OUT / "all_rois_all_measures_mixedlm_results_with_fdr.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Significant-only tables
# -----------------------------------------------------------------------------
combined_results_sig_uncorrected = combined_results[
    (combined_results["model_failed"] == False)
    & (combined_results["p"].notna())
    & (combined_results["p"] < 0.05)
    & (~combined_results["term"].isin(TERMS_EXCLUDE_FROM_SIG))
].copy()

combined_results_sig_uncorrected.to_csv(
    PATH_OUT / "all_rois_all_measures_significant_uncorrected_no_covariates.csv",
    index=False,
)

combined_results_sig_fdr = combined_results[
    (combined_results["model_failed"] == False)
    & (combined_results["significant_fdr_05"] == True)
    & (~combined_results["term"].isin(TERMS_EXCLUDE_FROM_SIG))
].copy()

combined_results_sig_fdr.to_csv(
    PATH_OUT / "all_rois_all_measures_significant_fdr05_no_covariates.csv",
    index=False,
)

print("\nSaved:")
print(PATH_OUT / "all_rois_all_measures_mixedlm_results_with_fdr.csv")
print(PATH_OUT / "all_rois_all_measures_significant_uncorrected_no_covariates.csv")
print(PATH_OUT / "all_rois_all_measures_significant_fdr05_no_covariates.csv")


for (roi_name, measure), d in model_data_by_roi_measure.items():
    fit = fit_by_roi_measure[(roi_name, measure)]

    plot_feedback_curves(
        df_model=d,
        fit=fit,
        outcome_name=f"{roi_name}_{measure}",
        n_bins=N_BINS,
        path_out=PATH_OUT,
        file_tag=f"{roi_name}_{measure}",
    )

# -----------------------------------------------------------------------------
# Make feedback/model fit plot for each successfully fitted ROI x measure
# -----------------------------------------------------------------------------
for (roi_name, measure), d in model_data_by_roi_measure.items():
    fit = fit_by_roi_measure[(roi_name, measure)]

    plot_feedback_curves(
        df_model=d,
        fit=fit,
        outcome_name=f"{roi_name}_{measure}",
        n_bins=N_BINS,
        path_out=PATH_OUT,
        file_tag=f"{roi_name}_{measure}",
    )

print("Finished ROI model analysis.")