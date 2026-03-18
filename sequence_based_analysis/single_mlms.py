# -----------------------------------------------------------------------------
# MLM overview across all electrodes (CAR only) + one global RT model
#
# What this script does
# ---------------------
# 1. Runs the same MLM for all EEG electrodes and measures:
#       - exponent
#       - theta_flat
#       - alpha_flat
#       - beta_flat
#
# 2. Runs the RT model only once:
#       - rt_mean (mapped from column "mean_rt")
#
# 3. Saves:
#       - model summaries
#       - fixed effects for all terms
#       - wide beta table
#       - wide p-value table
#       - binarized p-value table (p < .05 -> 1, else 0)
#       - count of significant electrodes per measure/effect
#
# 4. Plots topomaps:
#       - unthresholded p-value topomaps (-log10 p)
#       - binary significance topomaps (p < .05)
#
# Notes
# -----
# - CAR data only
# - no subject exclusion
# - RT is run once globally, not per electrode
# - topomaps are only produced for EEG measures, not RT
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
FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car_csd.csv"

PATH_OUT = PATH_IN / "mlm_car_all_electrodes_overview"
PATH_OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REFERENCE = "car"

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
    Returns:
        fit, used_re_formula, error_message
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


def extract_fixed_effects(fit, analysis_name: str, electrode: str, re_formula: str):
    """
    Extract fixed effects from a fitted MixedLM result.
    """
    rows = []

    fe_params = fit.fe_params
    bse = fit.bse_fe if hasattr(fit, "bse_fe") else pd.Series(index=fe_params.index, dtype=float)
    tvalues = fit.tvalues.reindex(fe_params.index) if hasattr(fit, "tvalues") else pd.Series(index=fe_params.index, dtype=float)
    pvalues = fit.pvalues.reindex(fe_params.index) if hasattr(fit, "pvalues") else pd.Series(index=fe_params.index, dtype=float)

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
                "electrode": electrode,
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


def build_info_from_electrodes(electrodes):
    info = mne.create_info(
        ch_names=electrodes,
        sfreq=1000,
        ch_types="eeg",
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="warn", match_case=False)
    return info


def plot_measure_term_topomaps(
    df_effects: pd.DataFrame,
    electrodes: list,
    measure: str,
    terms: list,
    info,
    path_out: Path,
):
    """
    Create two figures per measure:
    1) -log10(p) topomaps
    2) binary significance topomaps (p < threshold)
    """
    dsub = df_effects[df_effects["measure"] == measure].copy()
    if dsub.empty:
        return

    terms_to_plot = [t for t in terms if t in dsub["term"].unique()]
    if not terms_to_plot:
        return

    # -------------------------------------------------------------------------
    # Figure 1: -log10(p)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(3.8 * len(terms_to_plot), 4.2))
    gs = fig.add_gridspec(1, len(terms_to_plot) + 1, width_ratios=[1] * len(terms_to_plot) + [0.06])

    axes = [fig.add_subplot(gs[0, i]) for i in range(len(terms_to_plot))]
    cax = fig.add_subplot(gs[0, len(terms_to_plot)])

    p_maps = []
    for term in terms_to_plot:
        topo_p = (
            dsub[dsub["term"] == term]
            .set_index("electrode")
            .reindex(electrodes)["p"]
            .to_numpy()
        )
        topo_logp = -np.log10(topo_p)
        topo_logp[~np.isfinite(topo_logp)] = np.nan
        p_maps.append(topo_logp)

    global_vmax_logp = np.nanmax(np.abs(np.concatenate([x[np.isfinite(x)] for x in p_maps if np.any(np.isfinite(x))])))
    if not np.isfinite(global_vmax_logp) or global_vmax_logp == 0:
        global_vmax_logp = 1.0

    for i, term in enumerate(terms_to_plot):
        topo_logp = p_maps[i]

        evoked = mne.EvokedArray(
            topo_logp[:, None],
            info,
            tmin=0.0,
            verbose=False,
        )

        ax = axes[i]
        evoked.plot_topomap(
            times=[0],
            axes=ax if i < len(terms_to_plot) - 1 else [ax, cax],
            colorbar=(i == len(terms_to_plot) - 1),
            cmap="viridis",
            vlim=(0, global_vmax_logp),
            scalings=1,
            show=False,
            sphere=None,
        )
        ax.set_title(term)

    fig.suptitle(f"{measure}: -log10(p) topomaps", fontsize=16)
    plt.tight_layout()
    plt.savefig(path_out / f"{measure}_topomap_log10p.png", dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: binary significance
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(3.8 * len(terms_to_plot), 4.2))
    gs = fig.add_gridspec(1, len(terms_to_plot) + 1, width_ratios=[1] * len(terms_to_plot) + [0.06])

    axes = [fig.add_subplot(gs[0, i]) for i in range(len(terms_to_plot))]
    cax = fig.add_subplot(gs[0, len(terms_to_plot)])

    for i, term in enumerate(terms_to_plot):
        topo_p = (
            dsub[dsub["term"] == term]
            .set_index("electrode")
            .reindex(electrodes)["p"]
            .to_numpy()
        )
        topo_bin = (topo_p < P_THRESHOLD).astype(float)

        evoked = mne.EvokedArray(
            topo_bin[:, None],
            info,
            tmin=0.0,
            verbose=False,
        )

        ax = axes[i]
        evoked.plot_topomap(
            times=[0],
            axes=ax if i < len(terms_to_plot) - 1 else [ax, cax],
            colorbar=(i == len(terms_to_plot) - 1),
            cmap="Reds",
            vlim=(0, 1),
            scalings=1,
            show=False,
            sphere=None,
        )
        ax.set_title(term)

    fig.suptitle(f"{measure}: binary significance topomaps (p < {P_THRESHOLD})", fontsize=16)
    plt.tight_layout()
    plt.savefig(path_out / f"{measure}_topomap_binary_p_lt_{str(P_THRESHOLD).replace('.', 'p')}.png", dpi=200)
    plt.close()


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

ELECTRODES = sorted(df["ch_name"].astype(str).unique().tolist())

print(f"Filtered rows (CAR only): {len(df)}")
print(f"Subjects: {df['id'].nunique()}")
print(f"Electrodes: {len(ELECTRODES)}")


# -----------------------------------------------------------------------------
# Run MLMs
# -----------------------------------------------------------------------------
results_rows = []
model_rows = []

# -------------------------------------------------------------------------
# EEG models: run separately for each measure and electrode
# -------------------------------------------------------------------------
for analysis_name, measure_col in MEASURES_EEG.items():
    for electrode in ELECTRODES:
        dsub = df[df["ch_name"].astype(str) == electrode].copy()

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

        print(f"Running {analysis_name} @ {electrode} | rows={n_rows}, subjects={n_subjects}")

        if n_subjects < MIN_SUBJECTS:
            model_rows.append(
                {
                    "measure": analysis_name,
                    "electrode": electrode,
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
                    "measure": analysis_name,
                    "electrode": electrode,
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
                "measure": analysis_name,
                "electrode": electrode,
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

        results_rows.extend(
            extract_fixed_effects(
                fit=fit,
                analysis_name=analysis_name,
                electrode=electrode,
                re_formula=used_re_formula,
            )
        )

        print(f"  OK | re_formula={used_re_formula}")

# -------------------------------------------------------------------------
# RT model: run once globally
# -------------------------------------------------------------------------
for analysis_name, measure_col in MEASURE_RT.items():
    dsub = df.copy()

    needed = [
        measure_col,
        "f",
        "f2",
        "group",
        "id",
        "half",
        "mean_trial_difficulty",
        "block_nr",
        "sequence_nr",
    ]
    dsub = dsub.dropna(subset=needed).copy()

    # Drop duplicates across electrodes so each sequence enters once
    dsub = dsub.drop_duplicates(subset=["id", "block_nr", "sequence_nr"]).copy()

    n_subjects = dsub["id"].nunique()
    n_rows = len(dsub)

    print(f"Running {analysis_name} (global) | rows={n_rows}, subjects={n_subjects}")

    if n_subjects < MIN_SUBJECTS:
        model_rows.append(
            {
                "measure": analysis_name,
                "electrode": "global",
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
                "measure": analysis_name,
                "electrode": "global",
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
            "measure": analysis_name,
            "electrode": "global",
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

    results_rows.extend(
        extract_fixed_effects(
            fit=fit,
            analysis_name=analysis_name,
            electrode="global",
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

df_models.to_csv(PATH_OUT / "mlm_model_summary.csv", index=False)

if df_results.empty:
    raise RuntimeError("No successful fixed-effect results to save.")

df_results.to_csv(PATH_OUT / "mlm_fixed_effects_all_terms.csv", index=False)

df_effects = df_results[df_results["term"].isin(TERMS_OF_INTEREST)].copy()
df_effects.to_csv(PATH_OUT / "mlm_fixed_effects_terms_of_interest.csv", index=False)

# Wide beta table
df_beta_wide = (
    df_effects
    .pivot_table(
        index=["measure", "electrode"],
        columns="term",
        values="beta",
        aggfunc="first",
    )
    .reset_index()
)
df_beta_wide.to_csv(PATH_OUT / "mlm_betas_wide.csv", index=False)

# Wide p-value table
df_p_wide = (
    df_effects
    .pivot_table(
        index=["measure", "electrode"],
        columns="term",
        values="p",
        aggfunc="first",
    )
    .reset_index()
)
df_p_wide.to_csv(PATH_OUT / "mlm_pvalues_wide.csv", index=False)

# Binarized p-value table
df_p_binary = df_p_wide.copy()
for col in df_p_binary.columns:
    if col not in ["measure", "electrode"]:
        df_p_binary[col] = (df_p_binary[col] < P_THRESHOLD).astype(int)
df_p_binary.to_csv(PATH_OUT / f"mlm_pvalues_binary_{str(P_THRESHOLD).replace('.', 'p')}.csv", index=False)

# Count significant electrodes per measure/effect
df_sig_counts = (
    df_p_binary[df_p_binary["electrode"] != "global"]
    .groupby("measure")
    .sum(numeric_only=True)
    .reset_index()
)
df_sig_counts.to_csv(PATH_OUT / "mlm_significant_counts.csv", index=False)


# -----------------------------------------------------------------------------
# Topomaps
# -----------------------------------------------------------------------------
print("Creating topomaps...")
info = build_info_from_electrodes(ELECTRODES)

for measure in MEASURES_EEG.keys():
    plot_measure_term_topomaps(
        df_effects=df_effects,
        electrodes=ELECTRODES,
        measure=measure,
        terms=TERMS_OF_INTEREST,
        info=info,
        path_out=PATH_OUT,
    )


# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------
print("\nFinished.")
print("Output directory:", PATH_OUT)

print("\nModel status summary:")
print(
    df_models[
        ["measure", "electrode", "status", "re_formula", "n_rows", "n_subjects"]
    ].to_string(index=False)
)

print("\nSaved files:")
print(" - mlm_model_summary.csv")
print(" - mlm_fixed_effects_all_terms.csv")
print(" - mlm_fixed_effects_terms_of_interest.csv")
print(" - mlm_betas_wide.csv")
print(" - mlm_pvalues_wide.csv")
print(f" - mlm_pvalues_binary_{str(P_THRESHOLD).replace('.', 'p')}.csv")
print(" - mlm_significant_counts.csv")
for measure in MEASURES_EEG.keys():
    print(f" - {measure}_topomap_log10p.png")
    print(f" - {measure}_topomap_binary_p_lt_{str(P_THRESHOLD).replace('.', 'p')}.png")