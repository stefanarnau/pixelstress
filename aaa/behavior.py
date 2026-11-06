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
#   - participant-weighted descriptive feedback points without error bars
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
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_trial_data/")
PATH_OUT = PATH_IN / "behavior_mlm_results"
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_METADATA = PATH_IN / "trial_level_metadata.csv"

OPTIMIZER = "bfgs"
MAXITER = 5000

N_BINS = 40
DENSITY_BINS = 90
DENSITY_SMOOTH_SIGMA = 2.0  # Gaussian smoothing in histogram-bin units
BIN_LABELS = None

PREDICTION_POINTS = 200
CI_LEVEL = 0.95

FIGURE_DPI = 600
SAVE_FIGURE_PDF = True

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
    """Equal-participant-weighted bin means, SEM, and observed feedback positions."""
    participant_values = (
        data.groupby(["id", "group", "feedback_bin"], observed=True)
        .agg(value=(outcome, "mean"), feedback_x=("feedback", "mean"), n=(outcome, "size"))
        .reset_index()
    )
    summary = (
        participant_values.groupby(["group", "feedback_bin"], observed=True)
        .agg(mean=("value", "mean"), std=("value", "std"),
             count=("value", "size"), feedback_x=("feedback_x", "mean"),
             trials=("n", "sum"))
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    return summary


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
# Descriptive RT adjustment: use difficulty coefficient from the INFERENTIAL model
# (the inferential models themselves are unchanged).
# -----------------------------------------------------------------------------
rt_df["rt_difficulty_adjusted"] = (
    rt_df["rt"]
    - float(rt_result.fe_params["sequence_difficulty_c"])
    * rt_df["sequence_difficulty_c"]
)
rt_bin_summary = participant_weighted_bins(rt_df, "rt_difficulty_adjusted")
accuracy_bin_summary = participant_weighted_bins(accuracy_df, "accuracy")

rt_bin_summary.to_csv(PATH_OUT / "behavior_rt_difficulty_adjusted_feedback_bins.csv", index=False)
accuracy_bin_summary.to_csv(PATH_OUT / "behavior_accuracy_raw_feedback_bins.csv", index=False)


# -----------------------------------------------------------------------------
# Publication figure: model curves, descriptive means, compact feedback densities
# -----------------------------------------------------------------------------
COLORS = {
    "control": "#526B80",
    "experimental": "#BD587A"
}
LINESTYLES = {"control": "--", "experimental": "-"}

fig = plt.figure(figsize=(11.8, 4.9), layout="constrained")
gs = fig.add_gridspec(
    2, 2,
    height_ratios=[6, 0.85],
    hspace=0.035,
    wspace=0.17
)

main_axes = [fig.add_subplot(gs[0, j]) for j in range(2)]
density_axes = [
    fig.add_subplot(gs[1, j], sharex=main_axes[j])
    for j in range(2)
]

feedback_grid = np.linspace(
    df["feedback"].min(),
    df["feedback"].max(),
    PREDICTION_POINTS
)

density_edges = np.linspace(-1, 1, DENSITY_BINS + 1)
density_centers = (density_edges[:-1] + density_edges[1:]) / 2

# Collect model confidence limits for automatic y-axis scaling
model_limits = {"RT": [], "Accuracy": []}

for j, (main_ax, density_ax, dataset, summary, outcome) in enumerate(zip(
    main_axes,
    density_axes,
    [rt_df, accuracy_df],
    [rt_bin_summary, accuracy_bin_summary],
    ["RT", "Accuracy"]
)):

    for group in GROUP_ORDER:
        experimental_value = int(group == "experimental")

        # Model predictions and 95% confidence intervals
        if outcome == "RT":
            prediction, lower, upper = fixed_effect_prediction_rt(
                rt_result,
                feedback_grid,
                experimental_value,
                feedback_mean
            )
        else:
            prediction, lower, upper = fixed_effect_prediction_accuracy(
                accuracy_result,
                feedback_grid,
                experimental_value,
                feedback_mean
            )

        # Store confidence limits for axis scaling
        model_limits[outcome].extend([lower, upper])

        color = COLORS[group]

        # Confidence band
        main_ax.fill_between(
            feedback_grid,
            lower,
            upper,
            color=color,
            alpha=0.13,
            linewidth=0
        )

        # Continuous model prediction
        main_ax.plot(
            feedback_grid,
            prediction,
            color=color,
            linestyle=LINESTYLES[group],
            linewidth=2.2,
            label=GROUP_TITLES[group]
        )

        # Participant-weighted descriptive bin means
        # No confidence intervals or connecting lines
        sub = (
            summary.loc[summary["group"] == group]
            .sort_values("feedback_x")
        )

        main_ax.plot(
            sub["feedback_x"],
            sub["mean"],
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.3,
            zorder=5
        )

        # Smoothed trial-count histogram
        values = dataset.loc[
            dataset["group"] == group, "feedback"
        ].dropna()

        counts, _ = np.histogram(
            values,
            bins=density_edges
        )

        smooth_counts = gaussian_filter1d(
            counts.astype(float),
            sigma=DENSITY_SMOOTH_SIGMA,
            mode="constant"
        )

        density_ax.plot(
            density_centers,
            smooth_counts,
            color=color,
            linestyle=LINESTYLES[group],
            linewidth=1.5
        )

    # Shared axis formatting
    for ax in (main_ax, density_ax):
        ax.axvline(
            0,
            color="0.65",
            linestyle=":",
            linewidth=0.9
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-1, 1)

    main_ax.set_title(
        "A  Reaction time" if j == 0 else "B  Accuracy",
        loc="left",
        fontweight="bold"
    )

    main_ax.set_ylabel(
        "Difficulty-adjusted RT (ms)"
        if j == 0 else "Probability correct"
    )

    main_ax.tick_params(
        axis="x",
        labelbottom=False,
        bottom=False
    )

    main_ax.legend(
        frameon=False,
        loc="upper left" if j == 0 else "lower left"
    )

    # Compact feedback distribution
    density_ax.set_xlabel("Feedback relative to target")
    density_ax.set_ylabel("Trials", fontsize=9)
    density_ax.tick_params(axis="both", labelsize=9)
    density_ax.set_ylim(bottom=0)
    density_ax.spines["left"].set_visible(False)
    density_ax.set_yticks([])


# -----------------------------------------------------------------------------
# Automatic y-axis limits from model confidence bands
# -----------------------------------------------------------------------------
Y_PADDING = 0.05

for ax, outcome in zip(main_axes, ["RT", "Accuracy"]):

    values = np.concatenate(model_limits[outcome])

    ymin = np.nanmin(values)
    ymax = np.nanmax(values)

    yrange = ymax - ymin
    padding = Y_PADDING * yrange

    ax.set_ylim(
        ymin - padding,
        ymax + padding
    )

# Keep accuracy axis within valid probability range
accuracy_ylim = main_axes[1].get_ylim()
main_axes[1].set_ylim(
    max(0, accuracy_ylim[0]),
    min(1, accuracy_ylim[1])
)

plt.show()


# No global title or embedded figure caption; those belong in the manuscript.
saved_files.extend(save_figure(fig, "behavior_feedback_final"))
plt.close(fig)

print("\nSaved:")
for saved_file in saved_files:
    print(saved_file)
