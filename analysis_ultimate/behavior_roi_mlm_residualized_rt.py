# -----------------------------------------------------------------------------
# PixelStress: trial-level behavioral ROI-style analysis
#
# Mirrors the final/converged CNV analysis where applicable:
#   - continuous mean-centered feedback
#   - linear + quadratic feedback terms
#   - group interactions with both terms
#   - sequence difficulty as centered covariate
#   - participant RANDOM INTERCEPT ONLY for RT (same structure as converged CNV)
#   - binomial GEE with participant clustering for accuracy
#   - common 5-bin descriptive feedback visualization using the CNV color scheme
#   - continuous fixed-effect predictions with 95% CIs
#   - group-specific linear/quadratic feedback contrasts
#
# IMPORTANT:
# Feedback bins are descriptive only. All inference uses continuous feedback.
# -----------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.cm import get_cmap
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_trial_data/")
PATH_OUT = PATH_IN / "behavior_mlm_results"
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_METADATA = PATH_IN / "trial_level_metadata.csv"

OPTIMIZER = "bfgs"
MAXITER = 5000

N_BINS = 5
BIN_LABELS = None

PREDICTION_POINTS = 200
CI_LEVEL = 0.95

FIGURE_DPI = 600
SAVE_FIGURE_PDF = True
SHOW_PARTICIPANT_BIN_VALUES = True

GROUP_ORDER = ["control", "experimental"]
GROUP_TITLES = {
    "control": "Control",
    "experimental": "Experimental",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normal_critical_value(ci_level):
    lookup = {
        0.90: 1.644854,
        0.95: 1.959964,
        0.99: 2.575829,
    }
    if ci_level not in lookup:
        raise ValueError("CI level must be one of 0.90, 0.95, or 0.99.")
    return lookup[ci_level]


def make_feedback_bins(feedback):
    """Create common equal-width feedback bins across both groups."""
    feedback = np.asarray(feedback, dtype=float)
    edges = np.linspace(
        np.nanmin(feedback),
        np.nanmax(feedback),
        N_BINS + 1,
    )
    edges[0] -= np.finfo(float).eps
    edges[-1] += np.finfo(float).eps

    if BIN_LABELS is None:
        labels = [
            f"{edges[index]:.2f} to {edges[index + 1]:.2f}"
            for index in range(N_BINS)
        ]
    else:
        if len(BIN_LABELS) != N_BINS:
            raise ValueError("BIN_LABELS must contain exactly N_BINS labels.")
        labels = BIN_LABELS

    bins = pd.cut(
        feedback,
        bins=edges,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    return bins, edges, labels


def save_figure(fig, stem):
    png_file = PATH_OUT / f"{stem}.png"
    fig.savefig(png_file, dpi=FIGURE_DPI, bbox_inches="tight")
    saved = [png_file]

    if SAVE_FIGURE_PDF:
        pdf_file = PATH_OUT / f"{stem}.pdf"
        fig.savefig(pdf_file, bbox_inches="tight")
        saved.append(pdf_file)

    return saved


def fixed_effect_prediction_rt(result, feedback_values, experimental, feedback_mean):
    """RT predictions and normal-theory CIs from fixed effects only."""
    fixed_names = list(result.fe_params.index)
    beta = result.fe_params.loc[fixed_names].to_numpy()
    covariance = result.cov_params().loc[fixed_names, fixed_names].to_numpy()

    rows = []
    feedback_centered = feedback_values - feedback_mean

    for feedback_c in feedback_centered:
        values = {
            "Intercept": 1.0,
            "experimental": float(experimental),
            "feedback_c": feedback_c,
            "feedback2_c": feedback_c ** 2,
            "experimental:feedback_c": float(experimental) * feedback_c,
            "experimental:feedback2_c": float(experimental) * feedback_c ** 2,
            "sequence_difficulty_c": 0.0,
        }
        rows.append([values.get(name, 0.0) for name in fixed_names])

    design = np.asarray(rows)
    prediction = design @ beta
    standard_error = np.sqrt(
        np.einsum("ij,jk,ik->i", design, covariance, design)
    )
    critical = normal_critical_value(CI_LEVEL)

    return (
        prediction,
        prediction - critical * standard_error,
        prediction + critical * standard_error,
    )


def fixed_effect_prediction_accuracy(result, feedback_values, experimental, feedback_mean):
    """
    Accuracy predictions from the GEE on the response-probability scale.

    CIs are first calculated on the linear-predictor (logit) scale and then
    transformed with the inverse-logit.
    """
    fixed_names = list(result.params.index)
    beta = result.params.loc[fixed_names].to_numpy()
    covariance = result.cov_params().loc[fixed_names, fixed_names].to_numpy()

    rows = []
    feedback_centered = feedback_values - feedback_mean

    for feedback_c in feedback_centered:
        values = {
            "Intercept": 1.0,
            "experimental": float(experimental),
            "feedback_c": feedback_c,
            "feedback2_c": feedback_c ** 2,
            "experimental:feedback_c": float(experimental) * feedback_c,
            "experimental:feedback2_c": float(experimental) * feedback_c ** 2,
            "sequence_difficulty_c": 0.0,
        }
        rows.append([values.get(name, 0.0) for name in fixed_names])

    design = np.asarray(rows)
    eta = design @ beta
    eta_se = np.sqrt(
        np.einsum("ij,jk,ik->i", design, covariance, design)
    )
    critical = normal_critical_value(CI_LEVEL)

    inv_logit = lambda x: 1.0 / (1.0 + np.exp(-x))

    prediction = inv_logit(eta)
    lower = inv_logit(eta - critical * eta_se)
    upper = inv_logit(eta + critical * eta_se)

    return prediction, lower, upper


def linear_combination_test(params, covariance, weights, label):
    """Wald test for a linear combination of model coefficients."""
    names = list(params.index)
    contrast = np.asarray([weights.get(name, 0.0) for name in names])

    beta = params.loc[names].to_numpy()
    cov = covariance.loc[names, names].to_numpy()

    estimate = contrast @ beta
    variance = contrast @ cov @ contrast
    se = np.sqrt(variance)
    z_value = estimate / se
    p_value = 2 * norm.sf(abs(z_value))
    critical = normal_critical_value(CI_LEVEL)

    return {
        "effect": label,
        "estimate": estimate,
        "se": se,
        "z": z_value,
        "p": p_value,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
    }


def group_specific_feedback_effects(params, covariance):
    """Linear and quadratic feedback coefficients within each group."""
    tests = [
        linear_combination_test(
            params,
            covariance,
            {"feedback_c": 1.0},
            "Control: linear feedback",
        ),
        linear_combination_test(
            params,
            covariance,
            {"feedback2_c": 1.0},
            "Control: quadratic feedback",
        ),
        linear_combination_test(
            params,
            covariance,
            {
                "feedback_c": 1.0,
                "experimental:feedback_c": 1.0,
            },
            "Experimental: linear feedback",
        ),
        linear_combination_test(
            params,
            covariance,
            {
                "feedback2_c": 1.0,
                "experimental:feedback2_c": 1.0,
            },
            "Experimental: quadratic feedback",
        ),
    ]
    return pd.DataFrame(tests)


def participant_weighted_bins(data, outcome):
    """
    Calculate descriptive bin values in two stages:
      1. mean within participant x group x feedback bin
      2. mean/SEM across participants

    Thus participants, rather than trials, receive equal weight in the figure.
    """
    participant_values = (
        data.groupby(
            ["id", "group", "feedback_bin"],
            observed=True,
        )[outcome]
        .mean()
        .reset_index()
    )

    summary = (
        participant_values.groupby(
            ["group", "feedback_bin"],
            observed=True,
        )[outcome]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    return participant_values, summary


# -----------------------------------------------------------------------------
# Load and prepare metadata
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_METADATA)

df["experimental"] = (df["group"] == "experimental").astype(int)

df["feedback"] = pd.to_numeric(df["feedback"], errors="coerce")
df["sequence_difficulty"] = pd.to_numeric(
    df["sequence_difficulty"],
    errors="coerce",
)

# Use ONE common feedback centering constant for RT and accuracy so that the
# parameterization and prediction curves are directly comparable.
feedback_mean = df["feedback"].mean()
sequence_difficulty_mean = df["sequence_difficulty"].mean()

df["feedback_c"] = df["feedback"] - feedback_mean
df["feedback2_c"] = df["feedback_c"] ** 2
df["sequence_difficulty_c"] = (
    df["sequence_difficulty"] - sequence_difficulty_mean
)

# Use one common set of bin edges for both behavioral outcomes.
df["feedback_bin"], feedback_bin_edges, feedback_bin_labels = (
    make_feedback_bins(df["feedback"])
)


# -----------------------------------------------------------------------------
# RT analysis: correct trials with valid positive RT
# -----------------------------------------------------------------------------
rt_df = df.loc[
    df["rt"].notna()
    & np.isfinite(df["rt"])
    & (df["rt"] > 0)
].copy()

if "accuracy" in rt_df.columns:
    rt_df = rt_df.loc[rt_df["accuracy"] == 1].copy()

rt_columns = [
    "rt",
    "id",
    "group",
    "experimental",
    "feedback",
    "feedback_c",
    "feedback2_c",
    "sequence_difficulty_c",
    "feedback_bin",
]
rt_df = rt_df.dropna(subset=rt_columns).copy()

rt_formula = (
    "rt ~ "
    "experimental "
    "+ feedback_c "
    "+ feedback2_c "
    "+ experimental:feedback_c "
    "+ experimental:feedback2_c "
    "+ sequence_difficulty_c"
)

# IMPORTANT: random-intercept-only structure, matching the converged CNV model.
rt_model = smf.mixedlm(
    formula=rt_formula,
    data=rt_df,
    groups=rt_df["id"],
    re_formula="1",
)

rt_result = rt_model.fit(
    reml=False,
    method=OPTIMIZER,
    maxiter=MAXITER,
)

rt_fixed_names = list(rt_result.fe_params.index)
rt_ci = rt_result.conf_int().loc[rt_fixed_names]

rt_coefficients = pd.DataFrame({
    "term": rt_fixed_names,
    "estimate": rt_result.fe_params.loc[rt_fixed_names].to_numpy(),
    "standard_error": rt_result.bse_fe.loc[rt_fixed_names].to_numpy(),
    "z_value": rt_result.tvalues.loc[rt_fixed_names].to_numpy(),
    "p_value": rt_result.pvalues.loc[rt_fixed_names].to_numpy(),
    "ci_low": rt_ci[0].to_numpy(),
    "ci_high": rt_ci[1].to_numpy(),
})

rt_simple_effects = group_specific_feedback_effects(
    rt_result.fe_params,
    rt_result.cov_params().loc[rt_fixed_names, rt_fixed_names],
)


# -----------------------------------------------------------------------------
# Accuracy analysis: binomial GEE
# -----------------------------------------------------------------------------
accuracy_df = df.loc[df["accuracy"].notna()].copy()

accuracy_columns = [
    "accuracy",
    "id",
    "group",
    "experimental",
    "feedback",
    "feedback_c",
    "feedback2_c",
    "sequence_difficulty_c",
    "feedback_bin",
]
accuracy_df = accuracy_df.dropna(subset=accuracy_columns).copy()

accuracy_formula = (
    "accuracy ~ "
    "experimental "
    "+ feedback_c "
    "+ feedback2_c "
    "+ experimental:feedback_c "
    "+ experimental:feedback2_c "
    "+ sequence_difficulty_c"
)

accuracy_model = smf.gee(
    formula=accuracy_formula,
    groups="id",
    data=accuracy_df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Exchangeable(),
)

accuracy_result = accuracy_model.fit()

accuracy_names = list(accuracy_result.params.index)
accuracy_ci = accuracy_result.conf_int().loc[accuracy_names]

accuracy_coefficients = pd.DataFrame({
    "term": accuracy_names,
    "estimate": accuracy_result.params.loc[accuracy_names].to_numpy(),
    "standard_error": accuracy_result.bse.loc[accuracy_names].to_numpy(),
    "z_value": accuracy_result.tvalues.loc[accuracy_names].to_numpy(),
    "p_value": accuracy_result.pvalues.loc[accuracy_names].to_numpy(),
    "ci_low": accuracy_ci[0].to_numpy(),
    "ci_high": accuracy_ci[1].to_numpy(),
    "odds_ratio": np.exp(accuracy_result.params.loc[accuracy_names].to_numpy()),
    "or_ci_low": np.exp(accuracy_ci[0].to_numpy()),
    "or_ci_high": np.exp(accuracy_ci[1].to_numpy()),
})

accuracy_simple_effects = group_specific_feedback_effects(
    accuracy_result.params,
    accuracy_result.cov_params().loc[accuracy_names, accuracy_names],
)


# -----------------------------------------------------------------------------
# Console output
# -----------------------------------------------------------------------------
print("\nRT analysis definition")
print("----------------------")
print(f"Participants:      {rt_df['id'].nunique()}")
print(f"Trials:            {len(rt_df)}")
print(f"Formula:           {rt_formula}")
print("Random effects:    participant intercept")
print(f"Feedback mean:     {feedback_mean:.6f}")
print(f"Feedback bins:     {N_BINS}, edges={np.round(feedback_bin_edges, 4).tolist()}")
print()
print(rt_result.summary())

print("\nRT group-specific feedback effects")
print("----------------------------------")
print(rt_simple_effects.to_string(index=False))

print("\nAccuracy analysis definition")
print("----------------------------")
print(f"Participants:      {accuracy_df['id'].nunique()}")
print(f"Trials:            {len(accuracy_df)}")
print(f"Formula:           {accuracy_formula}")
print("Correlation:       participant-clustered exchangeable GEE")
print(f"Feedback mean:     {feedback_mean:.6f}")
print(f"Feedback bins:     {N_BINS}, edges={np.round(feedback_bin_edges, 4).tolist()}")
print()
print(accuracy_result.summary())

print("\nAccuracy group-specific feedback effects")
print("----------------------------------------")
print(accuracy_simple_effects.to_string(index=False))


# -----------------------------------------------------------------------------
# Save numerical outputs
# -----------------------------------------------------------------------------
saved_files = []

rt_summary_file = PATH_OUT / "behavior_rt_model_summary.txt"
rt_coefficient_file = PATH_OUT / "behavior_rt_coefficients.csv"
rt_simple_file = PATH_OUT / "behavior_rt_group_specific_feedback_effects.csv"

rt_definition = f"""RT analysis definition
----------------------
Participants:      {rt_df["id"].nunique()}
Trials:            {len(rt_df)}
Formula:           {rt_formula}
Random effects:    participant intercept
Estimation:        ML
Feedback mean:     {feedback_mean:.6f}
Feedback bins:     {N_BINS}, edges={np.round(feedback_bin_edges, 4).tolist()}

"""

rt_summary_file.write_text(
    rt_definition
    + rt_result.summary().as_text()
    + "\n\nGroup-specific feedback effects\n"
    + "--------------------------------\n"
    + rt_simple_effects.to_string(index=False),
    encoding="utf-8",
)
rt_coefficients.to_csv(rt_coefficient_file, index=False)
rt_simple_effects.to_csv(rt_simple_file, index=False)

saved_files.extend([
    rt_summary_file,
    rt_coefficient_file,
    rt_simple_file,
])

accuracy_summary_file = PATH_OUT / "behavior_accuracy_model_summary.txt"
accuracy_coefficient_file = PATH_OUT / "behavior_accuracy_coefficients.csv"
accuracy_simple_file = (
    PATH_OUT / "behavior_accuracy_group_specific_feedback_effects.csv"
)

accuracy_definition = f"""Accuracy analysis definition
----------------------------
Participants:      {accuracy_df["id"].nunique()}
Trials:            {len(accuracy_df)}
Formula:           {accuracy_formula}
Model:             binomial GEE
Correlation:       participant-clustered exchangeable
Feedback mean:     {feedback_mean:.6f}
Feedback bins:     {N_BINS}, edges={np.round(feedback_bin_edges, 4).tolist()}

"""

accuracy_summary_file.write_text(
    accuracy_definition
    + accuracy_result.summary().as_text()
    + "\n\nGroup-specific feedback effects\n"
    + "--------------------------------\n"
    + accuracy_simple_effects.to_string(index=False),
    encoding="utf-8",
)
accuracy_coefficients.to_csv(accuracy_coefficient_file, index=False)
accuracy_simple_effects.to_csv(accuracy_simple_file, index=False)

saved_files.extend([
    accuracy_summary_file,
    accuracy_coefficient_file,
    accuracy_simple_file,
])


# -----------------------------------------------------------------------------
# Difficulty-only nuisance model for descriptive RT adjustment
# -----------------------------------------------------------------------------
# This model is used ONLY to remove the known sequence-difficulty contribution
# from RT before plotting descriptive feedback bins. It is separate from the
# inferential feedback model above.
rt_nuisance_formula = "rt ~ sequence_difficulty_c"

rt_nuisance_model = smf.mixedlm(
    formula=rt_nuisance_formula,
    data=rt_df,
    groups=rt_df["id"],
    re_formula="1",
)

rt_nuisance_result = rt_nuisance_model.fit(
    reml=False,
    method=OPTIMIZER,
    maxiter=MAXITER,
)

difficulty_beta_for_adjustment = float(
    rt_nuisance_result.fe_params["sequence_difficulty_c"]
)

# Remove only the estimated difficulty contribution. This preserves the RT
# scale in milliseconds, including the overall mean and all non-difficulty
# variation.
rt_df["rt_difficulty_adjusted"] = (
    rt_df["rt"]
    - difficulty_beta_for_adjustment * rt_df["sequence_difficulty_c"]
)

print("\nRT descriptive nuisance adjustment")
print("----------------------------------")
print(f"Formula:           {rt_nuisance_formula}")
print("Random effects:    participant intercept")
print(f"Difficulty beta:   {difficulty_beta_for_adjustment:.6f} ms/unit")
print(f"Converged:         {rt_nuisance_result.converged}")


# -----------------------------------------------------------------------------
# Descriptive binned values
# -----------------------------------------------------------------------------
rt_participant_bins, rt_bin_summary = participant_weighted_bins(
    rt_df,
    "rt_difficulty_adjusted",
)
accuracy_participant_bins, accuracy_bin_summary = participant_weighted_bins(
    accuracy_df,
    "accuracy",
)

rt_nuisance_summary_file = PATH_OUT / "behavior_rt_difficulty_nuisance_model.txt"
rt_nuisance_summary_file.write_text(
    rt_nuisance_result.summary().as_text(),
    encoding="utf-8",
)

rt_bin_file = PATH_OUT / "behavior_rt_difficulty_adjusted_feedback_bins.csv"
accuracy_bin_file = PATH_OUT / "behavior_accuracy_raw_feedback_bins.csv"

rt_bin_summary.to_csv(rt_bin_file, index=False)
accuracy_bin_summary.to_csv(accuracy_bin_file, index=False)

saved_files.extend([
    rt_nuisance_summary_file,
    rt_bin_file,
    accuracy_bin_file,
])


# -----------------------------------------------------------------------------
# Combined figure
# -----------------------------------------------------------------------------
cmap = get_cmap("plasma", N_BINS)
bin_colors = [cmap(index) for index in range(N_BINS)]
critical = normal_critical_value(CI_LEVEL)

group_line_styles = {
    "control": "--",
    "experimental": "-",
}
group_colors = {
    "control": "C0",
    "experimental": "C1",
}

figure, axes = plt.subplots(2, 2, figsize=(13, 9.5))
rt_bin_axis = axes[0, 0]
accuracy_bin_axis = axes[0, 1]
rt_prediction_axis = axes[1, 0]
accuracy_prediction_axis = axes[1, 1]

group_positions = np.arange(len(GROUP_ORDER), dtype=float)
group_labels = [GROUP_TITLES[group] for group in GROUP_ORDER]

# Slight dodge of the five bin estimates around each group position.
DODGE_WIDTH = 0.12
bin_offsets = np.linspace(-DODGE_WIDTH, DODGE_WIDTH, N_BINS)

# -----------------------------------------------------------------------------
# Descriptive RT interaction plot: difficulty-adjusted RT
# -----------------------------------------------------------------------------
for bin_index, bin_label in enumerate(feedback_bin_labels):
    means, cis = [], []
    for group in GROUP_ORDER:
        row = rt_bin_summary.loc[
            (rt_bin_summary["group"] == group)
            & (rt_bin_summary["feedback_bin"] == bin_label)
        ]
        if len(row) != 1:
            means.append(np.nan)
            cis.append(np.nan)
        else:
            means.append(float(row["mean"].iloc[0]))
            cis.append(critical * float(row["sem"].iloc[0]))

    x_positions = group_positions + bin_offsets[bin_index]

    rt_bin_axis.errorbar(
        x_positions,
        np.asarray(means),
        yerr=np.asarray(cis),
        marker="o",
        markersize=7,
        linewidth=1.8,
        capsize=3,
        color=bin_colors[bin_index],
        label=bin_label,
    )

rt_bin_axis.set_xticks(group_positions)
rt_bin_axis.set_xticklabels(group_labels)
rt_bin_axis.set_xlabel("Group")
rt_bin_axis.set_ylabel("Difficulty-adjusted RT (ms)")
rt_bin_axis.set_title("RT: descriptive feedback bins")

# -----------------------------------------------------------------------------
# Descriptive accuracy interaction plot: raw accuracy
# -----------------------------------------------------------------------------
accuracy_plot_lows = []
accuracy_plot_highs = []

for bin_index, bin_label in enumerate(feedback_bin_labels):
    means, cis = [], []
    for group in GROUP_ORDER:
        row = accuracy_bin_summary.loc[
            (accuracy_bin_summary["group"] == group)
            & (accuracy_bin_summary["feedback_bin"] == bin_label)
        ]
        if len(row) != 1:
            means.append(np.nan)
            cis.append(np.nan)
        else:
            means.append(float(row["mean"].iloc[0]))
            cis.append(critical * float(row["sem"].iloc[0]))

    means = np.asarray(means, dtype=float)
    cis = np.asarray(cis, dtype=float)
    x_positions = group_positions + bin_offsets[bin_index]

    accuracy_plot_lows.extend((means - cis)[np.isfinite(means - cis)])
    accuracy_plot_highs.extend((means + cis)[np.isfinite(means + cis)])

    accuracy_bin_axis.errorbar(
        x_positions,
        means,
        yerr=cis,
        marker="o",
        markersize=7,
        linewidth=1.8,
        capsize=3,
        color=bin_colors[bin_index],
        label=bin_label,
    )

accuracy_bin_axis.set_xticks(group_positions)
accuracy_bin_axis.set_xticklabels(group_labels)
accuracy_bin_axis.set_xlabel("Group")
accuracy_bin_axis.set_ylabel("Proportion correct")
accuracy_bin_axis.set_title("Accuracy: descriptive feedback bins")

# Shared feedback-bin legend
handles, labels = rt_bin_axis.get_legend_handles_labels()
figure.legend(
    handles=handles,
    labels=labels,
    title="Feedback bin",
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=N_BINS,
    frameon=False,
)

# -----------------------------------------------------------------------------
# Continuous model predictions -- inferential models unchanged
# -----------------------------------------------------------------------------
feedback_grid = np.linspace(
    df["feedback"].min(),
    df["feedback"].max(),
    PREDICTION_POINTS,
)

accuracy_prediction_lows = []
accuracy_prediction_highs = []

for group in GROUP_ORDER:
    experimental_value = int(group == "experimental")

    rt_prediction, rt_lower, rt_upper = fixed_effect_prediction_rt(
        rt_result,
        feedback_grid,
        experimental_value,
        feedback_mean,
    )
    rt_prediction_axis.plot(
        feedback_grid,
        rt_prediction,
        color=group_colors[group],
        linestyle=group_line_styles[group],
        linewidth=2.2,
        label=GROUP_TITLES[group],
        zorder=3,
    )
    rt_prediction_axis.fill_between(
        feedback_grid,
        rt_lower,
        rt_upper,
        color=group_colors[group],
        alpha=0.13,
        linewidth=0,
        zorder=2,
    )

    acc_prediction, acc_lower, acc_upper = fixed_effect_prediction_accuracy(
        accuracy_result,
        feedback_grid,
        experimental_value,
        feedback_mean,
    )
    accuracy_prediction_lows.extend(acc_lower[np.isfinite(acc_lower)])
    accuracy_prediction_highs.extend(acc_upper[np.isfinite(acc_upper)])

    accuracy_prediction_axis.plot(
        feedback_grid,
        acc_prediction,
        color=group_colors[group],
        linestyle=group_line_styles[group],
        linewidth=2.2,
        label=GROUP_TITLES[group],
        zorder=3,
    )
    accuracy_prediction_axis.fill_between(
        feedback_grid,
        acc_lower,
        acc_upper,
        color=group_colors[group],
        alpha=0.13,
        linewidth=0,
        zorder=2,
    )

rt_prediction_axis.set_xlabel("Feedback")
rt_prediction_axis.set_ylabel("Model-predicted RT (ms)")
rt_prediction_axis.set_title("RT: continuous model prediction")
rt_prediction_axis.legend(frameon=False)

accuracy_prediction_axis.set_xlabel("Feedback")
accuracy_prediction_axis.set_ylabel("Model-predicted probability correct")
accuracy_prediction_axis.set_title("Accuracy: continuous model prediction")
accuracy_prediction_axis.legend(frameon=False)

# -----------------------------------------------------------------------------
# Tight, COMMON y-axis limits for both accuracy panels.
# Include all descriptive 95% CIs and model-prediction 95% CIs, then add a
# small margin. Clamp to the valid probability range.
# -----------------------------------------------------------------------------
all_accuracy_lows = np.asarray(
    accuracy_plot_lows + accuracy_prediction_lows,
    dtype=float,
)
all_accuracy_highs = np.asarray(
    accuracy_plot_highs + accuracy_prediction_highs,
    dtype=float,
)

accuracy_min = float(np.nanmin(all_accuracy_lows))
accuracy_max = float(np.nanmax(all_accuracy_highs))
accuracy_span = max(accuracy_max - accuracy_min, 0.02)
accuracy_margin = 0.08 * accuracy_span

accuracy_ylim = (
    max(0.0, accuracy_min - accuracy_margin),
    min(1.0, accuracy_max + accuracy_margin),
)

accuracy_bin_axis.set_ylim(accuracy_ylim)
accuracy_prediction_axis.set_ylim(accuracy_ylim)

figure.suptitle("Behavioral feedback effects", y=0.985)
figure.subplots_adjust(top=0.84, hspace=0.38, wspace=0.28)

figure_file_stem = "behavior_feedback_combined_figure"
saved_files.extend(save_figure(figure, figure_file_stem))
plt.close(figure)


# -----------------------------------------------------------------------------
# Finish
# -----------------------------------------------------------------------------
print("\nSaved:")
for saved_file in saved_files:
    print(saved_file)
