"""
Combined trial-level EEG predictors of reaction time.

Model comparisons use ML and one common complete-case sample. Nakagawa-style
marginal and conditional R² values are calculated for every candidate model.
Likelihood-ratio tests are the primary tests of unique model improvement;
changes in marginal R² are descriptive effect-size estimates.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = Path(
    "/mnt/data_dump/pixelstress/3_trial_data/trial_level_metadata_with_roi_values.csv"
)
OUTPUT_DIR = Path(
    "/mnt/data_dump/pixelstress/3_trial_data/combined_brain_behavior_models"
)

ID_COL = "id"
RT_COL = "rt"
FEEDBACK_COL = "feedback_c"

EEG_COLUMNS = {
    "erp": "erp_roi",             # Mean -0.10 to 0.00 s ROI voltage
    "alpha": "alpha_roi_db",      # Mean -1.00 to 0.00 s ROI power, dB
    "theta": "theta_roi_db",      # Mean 0.15 to 0.35 s ROI power, dB
}

BASE_TERMS = [
    "experimental",
    "feedback_c",
    "feedback2_c",
    "experimental:feedback_c",
    "experimental:feedback2_c",
    "sequence_difficulty_c",
]

OPTIMIZERS = ["lbfgs", "powell"]
MAXITER = 2000


# =============================================================================
# HELPERS
# =============================================================================

def nakagawa_r2(result) -> dict:
    """
    Nakagawa-style marginal and conditional R² for a Gaussian MixedLM.

    Marginal R²:
        variance explained by fixed effects / total variance

    Conditional R²:
        variance explained by fixed + random effects / total variance

    This implementation supports correlated random intercepts and slopes by
    averaging each observation's random-effect variance diag(Z G Z').
    """
    # Fixed-effect fitted values.
    fixed_beta = result.fe_params.to_numpy()
    fixed_design = np.asarray(result.model.exog, dtype=float)
    fixed_prediction = fixed_design @ fixed_beta

    # Population variance of fixed-effect predictions.
    fixed_variance = np.var(fixed_prediction, ddof=0)

    # Random-effect design and estimated covariance matrix.
    random_design = np.asarray(result.model.exog_re, dtype=float)
    random_covariance = np.asarray(result.cov_re, dtype=float)

    if random_design.shape[1] != random_covariance.shape[0]:
        raise ValueError(
            "Random-effect design and covariance dimensions do not match: "
            f"Z={random_design.shape}, G={random_covariance.shape}"
        )

    # Per-observation random-effect variance:
    # diag(Z G Z') without constructing the full n × n matrix.
    random_variance_by_observation = np.einsum(
        "ij,jk,ik->i",
        random_design,
        random_covariance,
        random_design,
    )
    random_variance = random_variance_by_observation.mean()

    # Gaussian residual variance.
    residual_variance = float(result.scale)

    total_variance = (
        fixed_variance
        + random_variance
        + residual_variance
    )

    marginal_r2 = fixed_variance / total_variance
    conditional_r2 = (
        fixed_variance + random_variance
    ) / total_variance

    return {
        "fixed_variance": fixed_variance,
        "random_variance": random_variance,
        "residual_variance": residual_variance,
        "total_variance": total_variance,
        "marginal_r2": marginal_r2,
        "conditional_r2": conditional_r2,
    }

def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"Cannot standardize {series.name}: SD is {sd}.")
    return (series - series.mean()) / sd


def add_eeg_decomposition(
    df: pd.DataFrame,
    participant_col: str,
    eeg_name: str,
    source_col: str,
) -> pd.DataFrame:
    """Add standardized within- and between-participant EEG predictors."""
    participant_mean = df.groupby(participant_col, observed=True)[source_col].transform("mean")
    within_raw = df[source_col] - participant_mean

    participant_means = (
        df.groupby(participant_col, observed=True)[source_col]
        .mean()
        .rename("participant_mean")
    )
    between_z_by_id = zscore(participant_means)

    df[f"{eeg_name}_within"] = zscore(within_raw)
    df[f"{eeg_name}_between"] = df[participant_col].map(between_z_by_id)
    return df


def fit_mixed_model(formula: str, data: pd.DataFrame, reml: bool):
    """Fit a MixedLM, trying a short optimizer sequence."""
    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data[ID_COL],
        re_formula=f"1 + {FEEDBACK_COL}",
    )

    last_error = None
    for method in OPTIMIZERS:
        try:
            result = model.fit(
                reml=reml,
                method=method,
                maxiter=MAXITER,
                disp=False,
            )
            if result.converged:
                return result, method
            last_error = RuntimeError(
                f"Model finished with {method} but did not converge."
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Model failed for formula:\n{formula}\nLast error: {last_error}"
    )


def eeg_terms(measures: tuple[str, ...]) -> list[str]:
    terms = []
    for measure in measures:
        terms.extend([f"{measure}_within", f"{measure}_between"])
    return terms


def make_formula(measures: tuple[str, ...]) -> str:
    terms = BASE_TERMS + eeg_terms(measures)
    return f"{RT_COL} ~ " + " + ".join(terms)


def model_row(name: str, measures: tuple[str, ...], result, optimizer: str) -> dict:
    return {
        "model": name,
        "measures": "+".join(measures) if measures else "none",
        "n_obs": int(result.nobs),
        "n_parameters": int(len(result.params)),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "converged": bool(result.converged),
        "optimizer": optimizer,
    }


def likelihood_ratio_test(smaller, larger) -> tuple[float, int, float]:
    lr = 2.0 * (larger.llf - smaller.llf)
    df_diff = int(len(larger.params) - len(smaller.params))
    p = chi2.sf(lr, df_diff)
    return lr, df_diff, p


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found:\n{DATA_FILE}\n\n"
        "Create a trial-level CSV containing the behavioral and EEG columns, "
        "or change DATA_FILE and EEG_COLUMNS in SETTINGS."
    )

df = pd.read_csv(DATA_FILE)

# The enriched file retains the original metadata columns. Recreate the
# derived predictors here rather than requiring them to have been saved.
source_columns = {
    ID_COL,
    RT_COL,
    "group",
    "feedback",
    "sequence_difficulty",
    *EEG_COLUMNS.values(),
}

missing = sorted(source_columns.difference(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[RT_COL] = pd.to_numeric(df[RT_COL], errors="coerce")
df["feedback"] = pd.to_numeric(df["feedback"], errors="coerce")
df["sequence_difficulty"] = pd.to_numeric(
    df["sequence_difficulty"], errors="coerce"
)

df["experimental"] = (df["group"] == "experimental").astype(int)

# Use one common complete-case sample for every candidate model. This is
# essential for valid AIC/BIC and likelihood-ratio comparisons.
complete_case_columns = sorted(source_columns)
df = df.dropna(subset=complete_case_columns).copy().reset_index(drop=True)

feedback_mean = df["feedback"].mean()
df["feedback_c"] = df["feedback"] - feedback_mean
df["feedback2_c"] = df["feedback_c"] ** 2

difficulty_mean = df["sequence_difficulty"].mean()
df["sequence_difficulty_c"] = (
    df["sequence_difficulty"] - difficulty_mean
)

for eeg_name, source_col in EEG_COLUMNS.items():
    df = add_eeg_decomposition(
        df=df,
        participant_col=ID_COL,
        eeg_name=eeg_name,
        source_col=source_col,
    )

analysis_sample_file = OUTPUT_DIR / "combined_rt_analysis_sample.csv"
df.to_csv(analysis_sample_file, index=False)

print("Loaded analysis data")
print("--------------------")
print(f"Input file:   {DATA_FILE}")
print(f"Participants: {df[ID_COL].nunique()}")
print(f"Trials:       {len(df)}")
print(f"Feedback mean used for centering:   {feedback_mean:.6f}")
print(f"Difficulty mean used for centering: {difficulty_mean:.6f}")
print()


# =============================================================================
# FIT ML MODELS FOR COMPARISON
# =============================================================================

measure_names = tuple(EEG_COLUMNS.keys())
model_specs: list[tuple[str, tuple[str, ...]]] = [("base", tuple())]

for measure in measure_names:
    model_specs.append((measure, (measure,)))

for pair in combinations(measure_names, 2):
    model_specs.append(("+".join(pair), pair))

model_specs.append(("all", measure_names))

ml_results = {}
summary_rows = []

for name, measures in model_specs:
    formula = make_formula(measures)

    print(f"Fitting ML model: {name}")
    result, optimizer = fit_mixed_model(
        formula,
        df,
        reml=False,
    )

    ml_results[name] = result

    r2 = nakagawa_r2(result)

    summary_rows.append({
        "model": name,
        "measures": "+".join(measures) if measures else "none",
        "n_obs": int(result.nobs),
        "n_parameters": int(len(result.params)),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "marginal_r2": r2["marginal_r2"],
        "conditional_r2": r2["conditional_r2"],
        "fixed_variance": r2["fixed_variance"],
        "random_variance": r2["random_variance"],
        "residual_variance": r2["residual_variance"],
        "converged": bool(result.converged),
        "optimizer": optimizer,
    })

model_summary = pd.DataFrame(summary_rows).sort_values("aic")

base_marginal_r2 = model_summary.loc[
    model_summary["model"] == "base",
    "marginal_r2",
].iloc[0]

model_summary["delta_marginal_r2_vs_base"] = (
    model_summary["marginal_r2"] - base_marginal_r2
)

model_summary.to_csv(OUTPUT_DIR / "model_fit_summary_ml.csv", index=False)

r2_by_model = model_summary.set_index("model")["marginal_r2"]

unique_r2 = pd.DataFrame([
    {
        "measure": "erp",
        "reduced_model": "alpha+theta",
        "full_model": "all",
        "unique_delta_marginal_r2": (
            r2_by_model["all"] - r2_by_model["alpha+theta"]
        ),
    },
    {
        "measure": "alpha",
        "reduced_model": "erp+theta",
        "full_model": "all",
        "unique_delta_marginal_r2": (
            r2_by_model["all"] - r2_by_model["erp+theta"]
        ),
    },
    {
        "measure": "theta",
        "reduced_model": "erp+alpha",
        "full_model": "all",
        "unique_delta_marginal_r2": (
            r2_by_model["all"] - r2_by_model["erp+alpha"]
        ),
    },
])

unique_r2.to_csv(
    OUTPUT_DIR / "unique_eeg_delta_marginal_r2.csv",
    index=False,
)

# Descriptive partition of the total EEG-related increase in marginal R².
# This is not a strict commonality analysis because model-specific residual
# and random-effect variance estimates can change when predictors are added.
total_eeg_increment = (
    r2_by_model["all"] - r2_by_model["base"]
)

sum_unique_eeg_increment = unique_r2[
    "unique_delta_marginal_r2"
].sum()

shared_eeg_increment = (
    total_eeg_increment - sum_unique_eeg_increment
)

variance_partition_summary = pd.DataFrame([
    {
        "component": "total_eeg_increment",
        "delta_marginal_r2": total_eeg_increment,
    },
    {
        "component": "sum_unique_eeg_increments",
        "delta_marginal_r2": sum_unique_eeg_increment,
    },
    {
        "component": "shared_or_overlapping_increment",
        "delta_marginal_r2": shared_eeg_increment,
    },
])

variance_partition_summary.to_csv(
    OUTPUT_DIR / "eeg_variance_partition_summary.csv",
    index=False,
)

print()
print("ML model comparison")
print("-------------------")
print(
    model_summary[
        [
            "model",
            "measures",
            "log_likelihood",
            "aic",
            "bic",
            "marginal_r2",
            "conditional_r2",
            "delta_marginal_r2_vs_base",
            "converged",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("Unique EEG contributions")
print("------------------------")
print(
    unique_r2.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print()
print("EEG variance partition summary")
print("------------------------------")
print(
    variance_partition_summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)


# =============================================================================
# INCREMENTAL TESTS
# =============================================================================

lrt_rows = []

for measure in measure_names:
    lr, df_diff, p = likelihood_ratio_test(ml_results["base"], ml_results[measure])
    lrt_rows.append({
        "smaller_model": "base",
        "larger_model": measure,
        "added_measure": measure,
        "lr_chi2": lr,
        "df": df_diff,
        "p": p,
    })

for held_out in measure_names:
    other_two = tuple(m for m in measure_names if m != held_out)
    smaller_name = "+".join(other_two)
    lr, df_diff, p = likelihood_ratio_test(ml_results[smaller_name], ml_results["all"])
    lrt_rows.append({
        "smaller_model": smaller_name,
        "larger_model": "all",
        "added_measure": held_out,
        "lr_chi2": lr,
        "df": df_diff,
        "p": p,
    })

lrt_summary = pd.DataFrame(lrt_rows)
lrt_summary.to_csv(OUTPUT_DIR / "nested_likelihood_ratio_tests.csv", index=False)

print()
print("Likelihood-ratio tests")
print("----------------------")
print(lrt_summary.to_string(index=False))


# =============================================================================
# FINAL FULL MODEL WITH REML
# =============================================================================

full_formula = make_formula(measure_names)

print()
print("Refitting full model with REML")
print("------------------------------")

full_reml, optimizer = fit_mixed_model(full_formula, df, reml=True)
print(full_reml.summary())

full_reml_r2 = nakagawa_r2(full_reml)
pd.DataFrame([full_reml_r2]).to_csv(
    OUTPUT_DIR / "full_model_reml_nakagawa_r2.csv",
    index=False,
)

print()
print("Full REML model Nakagawa R²")
print("---------------------------")
for name, value in full_reml_r2.items():
    print(f"{name:20s}: {value:.6f}")

with open(OUTPUT_DIR / "full_model_reml_summary.txt", "w", encoding="utf-8") as file:
    file.write(f"Optimizer: {optimizer}\n\n")
    file.write(str(full_reml.summary()))

ci = full_reml.conf_int().loc[full_reml.fe_params.index]
fixed_effects = pd.DataFrame({
    "beta": full_reml.fe_params,
    "se": full_reml.bse_fe,
    "ci_low": ci[0],
    "ci_high": ci[1],
    "p": full_reml.pvalues.loc[full_reml.fe_params.index],
})
fixed_effects.to_csv(OUTPUT_DIR / "full_model_reml_fixed_effects.csv")

print()
print(f"Saved analysis sample to:\n{analysis_sample_file}")
print(f"Saved outputs to:\n{OUTPUT_DIR}")
