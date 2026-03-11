# -----------------------------------------------------------------------------
# Mixed models for sequence-level RT + model prediction plots with binned data
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
# Visualization: binned RT heatmap by group + model prediction curve
# -----------------------------------------------------------------------------



def plot_feedback_heatmap_with_model(
    df_model,
    fit,
    outcome_name="mean_rt",
    n_bins=12,
    path_out=None,
):
    d = df_model.copy()

    # fixed group order
    group_order = ["control", "experimental"]

    # -------------------------------------------------------------------------
    # Bin signed feedback
    # -------------------------------------------------------------------------
    edges = np.linspace(d["f"].min(), d["f"].max(), n_bins + 1)
    d["f_bin"] = pd.cut(d["f"], bins=edges, include_lowest=True)

    # midpoint for plotting
    mids = np.array([(iv.left + iv.right) / 2 for iv in d["f_bin"].cat.categories])

    # -------------------------------------------------------------------------
    # Aggregate observed data per group x feedback bin
    # -------------------------------------------------------------------------
    agg = (
        d.groupby(["group", "f_bin"], observed=True)
        .agg(
            mean_score=("score", "mean"),
            sem_score=("score", "sem"),
            n=("score", "size"),
        )
        .reset_index()
    )

    agg["f_mid"] = agg["f_bin"].map(
        {cat: (cat.left + cat.right) / 2 for cat in d["f_bin"].cat.categories}
    )

    # -------------------------------------------------------------------------
    # Build model predictions across a smooth f grid
    # -------------------------------------------------------------------------
    f_grid = np.linspace(d["f"].min(), d["f"].max(), 300)
    difficulty_ref = float(d["mean_trial_difficulty"].mean())
    half_ref = d["half"].mode().iloc[0]

    pred_rows = []
    for g in group_order:
        for f_val in f_grid:
            pred_rows.append(
                {
                    "group": g,
                    "f": f_val,
                    "f2": f_val ** 2,
                    "mean_trial_difficulty": difficulty_ref,
                    "half": half_ref,
                }
            )

    pred = pd.DataFrame(pred_rows)
    pred["pred"] = fit.predict(pred)

    # -------------------------------------------------------------------------
    # Common color scale for heatmaps
    # -------------------------------------------------------------------------
    vmin = agg["mean_score"].min()
    vmax = agg["mean_score"].max()

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7),
        gridspec_kw={"height_ratios": [5, 1]},
        sharex="col", sharey="row"
    )

    for col_idx, g in enumerate(group_order):
        # Top row: heatmap-like binned means + model curve
        ax = axes[0, col_idx]
        dg = agg[agg["group"] == g].copy().sort_values("f_mid")
        dg_pred = pred[pred["group"] == g].copy().sort_values("f")

        # draw tiles manually
        for _, row in dg.iterrows():
            iv = row["f_bin"]
            rect = plt.Rectangle(
                (iv.left, 0),
                iv.right - iv.left,
                1,
                transform=ax.get_xaxis_transform(),
                alpha=0.85,
                color=plt.cm.viridis((row["mean_score"] - vmin) / (vmax - vmin + 1e-12)),
                zorder=0,
            )
            ax.add_patch(rect)

        # observed bin means
        ax.errorbar(
            dg["f_mid"],
            dg["mean_score"],
            yerr=dg["sem_score"],
            fmt="o",
            capsize=3,
            linestyle="none",
            alpha=0.95,
            label="Binned observed mean",
            zorder=3,
        )

        # model prediction
        ax.plot(
            dg_pred["f"],
            dg_pred["pred"],
            linewidth=3,
            label="Model prediction",
            zorder=4,
        )

        ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
        ax.set_title(g)
        ax.set_ylabel("RT (ms)")
        ax.grid(True, alpha=0.25)

        # Bottom row: counts per bin
        ax_n = axes[1, col_idx]
        ax_n.bar(
            dg["f_mid"],
            dg["n"],
            width=np.diff(edges).mean() * 0.9,
            align="center",
            alpha=0.85,
        )
        ax_n.axvline(0, color="k", linestyle="--", linewidth=1.2)
        ax_n.set_xlabel("Signed feedback (f)")
        ax_n.set_ylabel("n")
        ax_n.grid(True, alpha=0.25)

    # colorbar for tile values
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0, :], shrink=0.9, pad=0.02)
    cbar.set_label("Mean binned RT (ms)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)

    fig.suptitle(
        f"{outcome_name}: feedback-binned RT with model prediction",
        fontsize=15
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if path_out is not None:
        fig.savefig(
            path_out / f"{outcome_name}_feedback_heatmap_model.png",
            dpi=150,
            bbox_inches="tight"
        )

    plt.show()
    
    
# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
primary_outcome = "mean_log_rt"   # recommended primary outcome
secondary_outcome = "mean_rt"     # optional robustness check

run_secondary_model = True

n_bins = 9

formula = """
score ~ group * f + group * f2
        + mean_trial_difficulty + half
"""

re_formulas = [
    "1 + f + f2",
    "1 + f",
    "1",
]


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(file_in)

# typing
df["id"] = df["id"].astype("category")
df["group"] = df["group"].astype("category")
df["ch_name"] = df["ch_name"].astype("category")
df["half"] = df["half"].astype("category")

if "window" in df.columns:
    df["window"] = df["window"].astype("category")

df["group"] = df["group"].cat.set_categories(["control", "experimental"])

for c in ["f", "f2", "mean_trial_difficulty", "mean_rt", "mean_log_rt"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# -----------------------------------------------------------------------------
# Build sequence-level dataframe
# -----------------------------------------------------------------------------
# one row per sequence
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

df_seq = (
    df[key_cols]
    .drop_duplicates()
    .reset_index(drop=True)
)

# basic cleanup
df_seq = df_seq.dropna(
    subset=["id", "group", "half", "f", "f2", "mean_trial_difficulty", "mean_rt", "mean_log_rt"]
).copy()

# keep only sensible RTs
df_seq = df_seq[df_seq["mean_rt"] > 0].copy()
df_seq = df_seq[np.isfinite(df_seq["mean_log_rt"])].copy()


# Sanity check. remove later
d = df_seq.copy()
d["abs_f"] = d["f"].abs()
cut = d["abs_f"].quantile(0.95)
df_seq = d[d["abs_f"] <= cut].copy()

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
                formula,
                df_model,
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

        except Exception as e:
            fit_error_log.append(f"{re_formula}: {str(e)}")

    return fit, used_re, fit_error_log


# -----------------------------------------------------------------------------
# Run one model
# -----------------------------------------------------------------------------
def run_rt_model(df_seq, outcome_name, formula, re_formulas, path_out):
    d = df_seq.copy()
    d = d.dropna(subset=[outcome_name, "group", "id", "f", "f2", "mean_trial_difficulty", "half"]).copy()
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

    df_res = pd.DataFrame({
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
    })

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



    
plot_feedback_heatmap_with_model(
    df_model=d_rt,
    fit=fit_rt,
    outcome_name="mean_rt",
    n_bins=12,
    path_out=path_out,
)