# -----------------------------------------------------------------------------
# Mixed models for sequence-level RT + model prediction plots with binned data
# Includes sequence_nr_c as a fixed-effect covariate
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_in = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
path_out = Path("/mnt/data_dump/pixelstress/7_rt_models/")
path_out.mkdir(parents=True, exist_ok=True)

file_in = path_in / "all_subjects_seq_fooof_rt_channelwise_long.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
primary_outcome = "mean_log_rt"   # primary RT outcome
secondary_outcome = "mean_rt"     # robustness check
run_secondary_model = True

n_bins = 9

trim_extremes = False
trim_quantile = 0.95

formula = """
score ~ group * f_lin + group * f_quad
        + mean_trial_difficulty_c + half + sequence_nr_c
"""

re_formulas = [
    "1 + f_lin + f_quad",
    "1 + f_lin",
    "1",
]


# -----------------------------------------------------------------------------
# Plot helper: model curves + binned observed means + counts
# -----------------------------------------------------------------------------
def plot_feedback_curves_with_counts(
    df_model,
    fit,
    outcome_name,
    n_bins=8,
    path_out=None,
):
    d = df_model.copy()
    group_order = ["control", "experimental"]

    # fixed-width bins across the observed feedback range
    edges = np.linspace(d["f"].min(), d["f"].max(), n_bins + 1)
    d["f_bin"] = pd.cut(d["f"], bins=edges, include_lowest=True)

    # observed means by group x bin
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

    # smooth model predictions
    f_grid = np.linspace(d["f"].min(), d["f"].max(), 300)
    difficulty_ref = 0.0      # centered predictor
    sequence_ref = 0.0        # centered predictor
    half_ref = d["half"].mode().iloc[0]

    pred_rows = []
    for group_name in group_order:
        for f_val in f_grid:
            pred_rows.append(
                {
                    "group": group_name,
                    "f_lin": f_val,
                    "f_quad": f_val**2 - np.mean(d["f"]**2),
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

    # -------------------------
    # Top panel: curves + bins
    # -------------------------
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
            dg_pred["f_lin"],
            dg_pred["pred"],
            linewidth=3,
            color=color,
            label=f"{group_name} model",
        )

    ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
    ax.set_ylabel("RT (ms)" if outcome_name == "mean_rt" else outcome_name)
    ax.set_title(f"{outcome_name}: observed bin means and model curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -------------------------
    # Bottom panel: counts
    # -------------------------
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
            label=group_name,
        )

    ax2.axvline(0, color="k", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Signed feedback (f)")
    ax2.set_ylabel("n")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if path_out is not None:
        fig.savefig(
            path_out / f"{outcome_name}_feedback_curves_counts.png",
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(file_in)

df["id"] = df["id"].astype("category")
df["group"] = df["group"].astype("category")
df["ch_name"] = df["ch_name"].astype("category")
df["half"] = df["half"].astype("category")

if "window" in df.columns:
    df["window"] = df["window"].astype("category")

df["group"] = df["group"].cat.set_categories(["control", "experimental"])

for col in ["f", "f2", "mean_trial_difficulty", "mean_rt", "mean_log_rt", "sequence_nr"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# -----------------------------------------------------------------------------
# Build sequence-level dataframe
# -----------------------------------------------------------------------------
key_cols = [
    "id",
    "group",
    "block_nr",
    "sequence_nr",
    "half",
    "n_trials",
    "mean_trial_difficulty",
    "f",
    "f2",
    "mean_rt",
    "mean_log_rt",
]

if "window" in df.columns:
    key_cols = ["window"] + key_cols

df_seq = df[key_cols].drop_duplicates().reset_index(drop=True)

df_seq = df_seq.dropna(
    subset=[
        "id",
        "group",
        "sequence_nr",
        "half",
        "f",
        "f2",
        "mean_trial_difficulty",
        "mean_rt",
        "mean_log_rt",
    ]
).copy()

df_seq = df_seq[df_seq["mean_rt"] > 0].copy()
df_seq = df_seq[np.isfinite(df_seq["mean_log_rt"])].copy()

# centered predictors
df_seq["sequence_nr_c"] = df_seq["sequence_nr"] - df_seq["sequence_nr"].mean()
df_seq["mean_trial_difficulty_c"] = (
    df_seq["mean_trial_difficulty"] - df_seq["mean_trial_difficulty"].mean()
)

# orthogonal polynomial representation of feedback
f_center = df_seq["f"] - df_seq["f"].mean()

df_seq["f_lin"] = f_center
df_seq["f_quad"] = f_center**2 - np.mean(f_center**2)

# optional trimming of extreme |f| values
if trim_extremes:
    df_seq["abs_f"] = df_seq["f"].abs()
    cut = df_seq["abs_f"].quantile(trim_quantile)
    df_seq = df_seq[df_seq["abs_f"] <= cut].copy()
    df_seq = df_seq.drop(columns="abs_f")

print("Sequence-level rows:", len(df_seq))
print("Subjects:", df_seq["id"].nunique())


# -----------------------------------------------------------------------------
# Model fitting helper
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
# Run one model
# -----------------------------------------------------------------------------
def run_rt_model(df_seq, outcome_name, formula, re_formulas, path_out):
    d = df_seq.copy()
    d = d.dropna(
        subset=[
            outcome_name,
            "group",
            "id",
            "f",
            "f2",
            "mean_trial_difficulty_c",
            "half",
            "sequence_nr_c",
        ]
    ).copy()

    d = d.rename(columns={outcome_name: "score"})

    fit, used_re, fit_error_log = fit_mixedlm_with_fallback(d, formula, re_formulas)

    if fit is None:
        raise RuntimeError(
            f"Model for {outcome_name} failed:\n" + "\n".join(fit_error_log)
        )

    print("\n" + "=" * 80)
    print(f"Outcome: {outcome_name}")
    print(f"Random-effects structure: {used_re}")
    print(fit.summary())

    fe = fit.fe_params
    se = fit.bse_fe.reindex(fe.index)
    tvals = fe / se.replace(0, np.nan)
    pvals = fit.pvalues.reindex(fe.index)

    df_res = pd.DataFrame(
        {
            "outcome": outcome_name,
            "term": fe.index,
            "beta": fe.values,
            "se": se.values,
            "t": tvals.values,
            "p": pvals.values,
            "random_effects": used_re,
            "converged": bool(getattr(fit, "converged", True)),
            "n_subjects": d["id"].nunique(),
            "n_obs": len(d),
            "llf": fit.llf,
            "aic": fit.aic if np.isfinite(fit.aic) else np.nan,
            "bic": fit.bic if np.isfinite(fit.bic) else np.nan,
        }
    )

    df_res.to_csv(path_out / f"{outcome_name}_mixedlm_results.csv", index=False)

    return fit, d, df_res


# -----------------------------------------------------------------------------
# Run models
# -----------------------------------------------------------------------------
fit_log, d_log, res_log = run_rt_model(
    df_seq=df_seq,
    outcome_name=primary_outcome,
    formula=formula,
    re_formulas=re_formulas,
    path_out=path_out,
)

if run_secondary_model:
    fit_rt, d_rt, res_rt = run_rt_model(
        df_seq=df_seq,
        outcome_name=secondary_outcome,
        formula=formula,
        re_formulas=re_formulas,
        path_out=path_out,
    )


# -----------------------------------------------------------------------------
# Make plots
# -----------------------------------------------------------------------------
plot_feedback_curves_with_counts(
    df_model=d_log,
    fit=fit_log,
    outcome_name=primary_outcome,
    n_bins=n_bins,
    path_out=path_out,
)

if run_secondary_model:
    plot_feedback_curves_with_counts(
        df_model=d_rt,
        fit=fit_rt,
        outcome_name=secondary_outcome,
        n_bins=n_bins,
        path_out=path_out,
    )