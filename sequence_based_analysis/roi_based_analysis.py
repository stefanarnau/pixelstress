# -----------------------------------------------------------------------------
# Slim ROI mixed model + RT-style feedback plot with binned observed data
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data2/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/roi_models/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
#exponent strongest at CPz, Pz (Cz)
MEASURE = "exponent"   # "theta_flat", "alpha_flat", "beta_flat", "exponent"
ROI = ["Cz"]
ROI_NAME = "central_exponent"

N_BINS = 9

FORMULA = """
roi_val ~ group * f + group * f2
          + mean_trial_difficulty_c + half
"""

RE_FORMULAS = [
    "1 + f + f2",
    "1 + f",
    "1",
]


# -----------------------------------------------------------------------------
# Plot helper: same logic as RT script
# -----------------------------------------------------------------------------
def plot_feedback_curves_with_counts(
    df_model,
    fit,
    outcome_name,
    n_bins=8,
    path_out=None,
    file_tag=None,
):
    d = df_model.copy()
    group_order = ["control", "experimental"]

    edges = np.linspace(d["f"].min(), d["f"].max(), n_bins + 1)
    d["f_bin"] = pd.cut(d["f"], bins=edges, include_lowest=True)

    agg = (
        d.groupby(["group", "f_bin"], observed=True)
        .agg(
            mean_score=("roi_val", "mean"),
            sem_score=("roi_val", "sem"),
            n=("roi_val", "size"),
        )
        .reset_index()
    )

    agg["f_mid"] = agg["f_bin"].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    agg = agg.sort_values(["group", "f_mid"]).reset_index(drop=True)

    f_grid = np.linspace(d["f"].min(), d["f"].max(), 300)
    f_mean = float(d["f"].mean())
    f2_grid = (f_grid - f_mean) ** 2 - np.mean((d["f"] - f_mean) ** 2)

    difficulty_ref = 0.0
    half_ref = d["half"].mode().iloc[0]

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

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    # top panel
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

    # bottom panel
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
        if file_tag is None:
            file_tag = outcome_name
        fig.savefig(
            path_out / f"{file_tag}_feedback_curves_counts.png",
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


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
# Load and prepare data
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_IN)

df["id"] = df["id"].astype(str)
df["group"] = pd.Categorical(df["group"], categories=["control", "experimental"])
df["half"] = pd.Categorical(df["half"])
df["ch_name"] = df["ch_name"].astype(str)

for col in [MEASURE, "f", "mean_trial_difficulty"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# unique sequence metadata
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

# ROI average
df_roi = (
    df[df["ch_name"].isin(ROI)]
    .groupby(["id", "block_nr", "sequence_nr"], as_index=False)[MEASURE]
    .mean()
    .rename(columns={MEASURE: "roi_val"})
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

print("ROI:", ROI_NAME)
print("Measure:", MEASURE)
print("Electrodes:", ROI)
print("Rows:", len(d))
print("Subjects:", d["id"].nunique())


# -----------------------------------------------------------------------------
# Fit model
# -----------------------------------------------------------------------------
fit, used_re, fit_error_log = fit_mixedlm_with_fallback(
    df_model=d,
    formula=FORMULA,
    re_formulas=RE_FORMULAS,
)

if fit is None:
    raise RuntimeError("Model failed:\n" + "\n".join(fit_error_log))

print(fit.summary())
print("Random-effects structure:", used_re)

# save coefficient table
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
        "random_effects": used_re,
        "n_subjects": d["id"].nunique(),
        "n_obs": len(d),
        "llf": fit.llf,
        "aic": fit.aic if np.isfinite(fit.aic) else np.nan,
        "bic": fit.bic if np.isfinite(fit.bic) else np.nan,
        "measure": MEASURE,
        "roi_name": ROI_NAME,
    }
)

df_res.to_csv(PATH_OUT / f"{ROI_NAME}_{MEASURE}_mixedlm_results.csv", index=False)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plot_feedback_curves_with_counts(
    df_model=d,
    fit=fit,
    outcome_name=f"{ROI_NAME}_{MEASURE}",
    n_bins=N_BINS,
    path_out=PATH_OUT,
    file_tag=f"{ROI_NAME}_{MEASURE}",
)