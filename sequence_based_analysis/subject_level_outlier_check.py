# -----------------------------------------------------------------------------
# Subject-level influence screening for electrode-wise sequence MLM data
#
# Goal
# ----
# Assess how much each subject influences fixed effects in MLMs, using
# leave-one-subject-out (LOSO) refits.
#
# Scope
# -----
# - reference: CAR only
# - electrodes: Fz, CPz, POz
# - measures: exponent, theta_flat, alpha_flat, beta_flat
# - effects of interest:
#       f
#       f2
#       group[T.experimental]
#       group[T.experimental]:f
#       group[T.experimental]:f2
#
# Output
# ------
# Saves:
#   1. full-model fixed effects
#   2. LOSO fixed effects and deltas
#   3. subject-level influence summaries
#   4. basic heatmaps / barplots
# -----------------------------------------------------------------------------

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car_csd.csv"

PATH_OUT = PATH_IN / "subject_influence_qc_car_fz_cpz_poz"
PATH_OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REFERENCE = "car"
ELECTRODES = ["Fz", "CPz", "POz"]

MEASURES = [
    "exponent",
    "theta_flat",
    "alpha_flat",
    "beta_flat",
]

FORMULA = """
score ~ group * f + group * f2
        + mean_trial_difficulty + half
"""

# Use a simple random-intercept model for stable and fast screening
RE_FORMULA = "1"

TERMS_OF_INTEREST = [
    "f",
    "f2",
    "group[T.experimental]",
    "group[T.experimental]:f",
    "group[T.experimental]:f2",
]

MIN_SUBJECTS_TOTAL = 8
MIN_SUBJECTS_AFTER_DROP = 6

FIT_METHOD = "lbfgs"
FIT_REML = False
FIT_MAXITER = 2000

REL_EPS = 1e-8

# Plotting
sns.set_context("talk")
sns.set_style("whitegrid")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def safe_mixedlm_fit(d: pd.DataFrame, measure: str):
    """
    Fit a simple mixed model and return the fitted result.
    Returns None if fitting fails.
    """
    d = d.copy().rename(columns={measure: "score"})

    try:
        model = smf.mixedlm(
            FORMULA,
            d,
            groups=d["id"],
            re_formula=RE_FORMULA,
        )

        fit = model.fit(
            method=FIT_METHOD,
            reml=FIT_REML,
            maxiter=FIT_MAXITER,
            disp=False,
        )
        return fit

    except Exception:
        return None


def make_subject_metadata(d: pd.DataFrame) -> pd.DataFrame:
    """
    Basic subject-level metadata for later interpretation.
    """
    rows = []

    for sid, ds in d.groupby("id"):
        rows.append(
            {
                "id": sid,
                "group": ds["group"].mode(dropna=True).iloc[0] if not ds["group"].mode(dropna=True).empty else np.nan,
                "n_rows": len(ds),
                "n_electrodes": ds["ch_name"].nunique(),
                "n_sequences": ds[["block_nr", "sequence_nr"]].drop_duplicates().shape[0],
                "mean_f": ds["f"].mean(),
                "sd_f": ds["f"].std(),
                "mean_difficulty": ds["mean_trial_difficulty"].mean(),
            }
        )

    return pd.DataFrame(rows)


def robust_flag_series(x: pd.Series, thresh: float = 3.5) -> pd.Series:
    """
    Median/MAD-based robust z-score flagging.
    """
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=x.index)

    rz = (x - med) / (1.4826 * mad)
    return np.abs(rz) > thresh


# -----------------------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(FILE_IN)

required_cols = [
    "id", "group", "half", "reference", "ch_name",
    "f", "mean_trial_difficulty",
    "block_nr", "sequence_nr",
] + MEASURES

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["id"] = df["id"].astype(str)
df["group"] = df["group"].astype("category")
df["half"] = df["half"].astype("category")
df["reference"] = df["reference"].astype("category")
df["ch_name"] = df["ch_name"].astype("category")

df["group"] = df["group"].cat.set_categories(["control", "experimental"])

for c in ["f", "mean_trial_difficulty"] + MEASURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["f2"] = df["f"] ** 2

df = df[
    (df["reference"] == REFERENCE) &
    (df["ch_name"].isin(ELECTRODES))
].copy()

df = df.dropna(
    subset=["id", "group", "half", "f", "f2", "mean_trial_difficulty"] + MEASURES
).copy()

if df["id"].nunique() < MIN_SUBJECTS_TOTAL:
    raise RuntimeError(
        f"Too few subjects after filtering: {df['id'].nunique()} "
        f"(need at least {MIN_SUBJECTS_TOTAL})."
    )

print(f"Rows after filtering: {len(df)}")
print(f"Subjects after filtering: {df['id'].nunique()}")
print(f"Electrodes: {sorted(df['ch_name'].astype(str).unique().tolist())}")

subject_meta = make_subject_metadata(df)
subject_meta.to_csv(PATH_OUT / "subject_metadata.csv", index=False)


# -----------------------------------------------------------------------------
# Full-model fits
# -----------------------------------------------------------------------------
print("Fitting full models...")

full_rows = []
full_fit_objects = {}

for measure in MEASURES:
    dsub = df.dropna(subset=[measure]).copy()

    if dsub["id"].nunique() < MIN_SUBJECTS_TOTAL:
        print(f"Skipping {measure}: too few subjects.")
        continue

    fit = safe_mixedlm_fit(dsub, measure)

    if fit is None:
        print(f"Full model failed for {measure}.")
        continue

    full_fit_objects[measure] = fit

    for term, beta in fit.fe_params.items():
        full_rows.append(
            {
                "measure": measure,
                "term": term,
                "beta_full": float(beta),
                "converged": bool(getattr(fit, "converged", False)),
                "n_rows": len(dsub),
                "n_subjects": dsub["id"].nunique(),
            }
        )

df_full = pd.DataFrame(full_rows)

if df_full.empty:
    raise RuntimeError("No full-model fits succeeded.")

df_full.to_csv(PATH_OUT / "full_model_fixed_effects.csv", index=False)


# -----------------------------------------------------------------------------
# LOSO influence analysis
# -----------------------------------------------------------------------------
print("Running leave-one-subject-out refits...")

loso_rows = []
subjects = sorted(df["id"].unique().tolist())

for measure in MEASURES:
    if measure not in full_fit_objects:
        continue

    beta_full = full_fit_objects[measure].fe_params.copy()
    d_measure = df.dropna(subset=[measure]).copy()

    for sid in subjects:
        d_leave = d_measure[d_measure["id"] != sid].copy()

        n_subj_left = d_leave["id"].nunique()
        if n_subj_left < MIN_SUBJECTS_AFTER_DROP:
            continue

        fit_lo = safe_mixedlm_fit(d_leave, measure)
        if fit_lo is None:
            continue

        beta_lo = fit_lo.fe_params

        for term in beta_full.index:
            if term not in beta_lo.index:
                continue

            beta_f = float(beta_full[term])
            beta_l = float(beta_lo[term])
            delta = beta_f - beta_l

            loso_rows.append(
                {
                    "measure": measure,
                    "left_out_subject": sid,
                    "term": term,
                    "beta_full": beta_f,
                    "beta_loso": beta_l,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "rel_delta": abs(delta) / (abs(beta_f) + REL_EPS),
                    "loso_converged": bool(getattr(fit_lo, "converged", False)),
                    "n_rows_loso": len(d_leave),
                    "n_subjects_loso": n_subj_left,
                }
            )

df_loso = pd.DataFrame(loso_rows)

if df_loso.empty:
    raise RuntimeError("No LOSO models succeeded.")

df_loso.to_csv(PATH_OUT / "loso_influence_all_terms.csv", index=False)


# -----------------------------------------------------------------------------
# Restrict to terms of interest
# -----------------------------------------------------------------------------
df_loso_terms = df_loso[df_loso["term"].isin(TERMS_OF_INTEREST)].copy()
df_loso_terms.to_csv(PATH_OUT / "loso_influence_terms_of_interest.csv", index=False)


# -----------------------------------------------------------------------------
# Subject-level summaries
# -----------------------------------------------------------------------------
print("Summarizing influence...")

# Per subject x measure: max and mean influence across target terms
summary_subject_measure = (
    df_loso_terms
    .groupby(["left_out_subject", "measure"], as_index=False)
    .agg(
        max_abs_delta=("abs_delta", "max"),
        mean_abs_delta=("abs_delta", "mean"),
        median_abs_delta=("abs_delta", "median"),
        max_rel_delta=("rel_delta", "max"),
        mean_rel_delta=("rel_delta", "mean"),
        n_terms=("term", "nunique"),
    )
    .rename(columns={"left_out_subject": "id"})
)

# Per subject x measure x term
summary_subject_measure_term = (
    df_loso_terms
    .groupby(["left_out_subject", "measure", "term"], as_index=False)
    .agg(
        abs_delta=("abs_delta", "mean"),
        rel_delta=("rel_delta", "mean"),
        beta_full=("beta_full", "mean"),
    )
    .rename(columns={"left_out_subject": "id"})
)

# Overall subject influence across all measures and target terms
summary_subject_overall = (
    df_loso_terms
    .groupby("left_out_subject", as_index=False)
    .agg(
        max_abs_delta=("abs_delta", "max"),
        mean_abs_delta=("abs_delta", "mean"),
        median_abs_delta=("abs_delta", "median"),
        max_rel_delta=("rel_delta", "max"),
        mean_rel_delta=("rel_delta", "mean"),
        n_measure_term_pairs=("term", "count"),
    )
    .rename(columns={"left_out_subject": "id"})
)

# Merge metadata
summary_subject_measure = summary_subject_measure.merge(subject_meta, on="id", how="left")
summary_subject_measure_term = summary_subject_measure_term.merge(subject_meta, on="id", how="left")
summary_subject_overall = summary_subject_overall.merge(subject_meta, on="id", how="left")

# Robust flags
summary_subject_overall["flag_max_abs_delta"] = robust_flag_series(
    summary_subject_overall["max_abs_delta"]
)
summary_subject_overall["flag_mean_abs_delta"] = robust_flag_series(
    summary_subject_overall["mean_abs_delta"]
)
summary_subject_overall["flag_max_rel_delta"] = robust_flag_series(
    summary_subject_overall["max_rel_delta"]
)

summary_subject_overall["flag_any"] = (
    summary_subject_overall["flag_max_abs_delta"] |
    summary_subject_overall["flag_mean_abs_delta"] |
    summary_subject_overall["flag_max_rel_delta"]
)

summary_subject_measure.to_csv(PATH_OUT / "subject_influence_summary_by_measure.csv", index=False)
summary_subject_measure_term.to_csv(PATH_OUT / "subject_influence_summary_by_measure_term.csv", index=False)
summary_subject_overall.to_csv(PATH_OUT / "subject_influence_summary_overall.csv", index=False)

flagged = summary_subject_overall[summary_subject_overall["flag_any"]].copy()
flagged.to_csv(PATH_OUT / "flagged_subjects.csv", index=False)


# -----------------------------------------------------------------------------
# Save a compact rank table
# -----------------------------------------------------------------------------
rank_table = summary_subject_overall.sort_values(
    ["flag_any", "max_abs_delta", "mean_abs_delta"],
    ascending=[False, False, False],
).copy()

rank_table.to_csv(PATH_OUT / "subject_influence_ranked.csv", index=False)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
print("Making plots...")

# 1) Overall max influence per subject
plt.figure(figsize=(12, 6))
plot_df = rank_table.copy()
plot_df["label"] = plot_df["id"].astype(str)

ax = sns.barplot(
    data=plot_df,
    x="label",
    y="max_abs_delta",
    hue="flag_any",
    dodge=False,
)
ax.set_title("Subject influence screening: max |Δβ| across measures and target effects")
ax.set_xlabel("Subject")
ax.set_ylabel("max |Δβ|")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(PATH_OUT / "subject_influence_max_abs_delta_overall.png", dpi=200)
plt.close()


# 2) Heatmap per measure x subject using max abs influence across target terms
heat_df = (
    summary_subject_measure
    .pivot(index="id", columns="measure", values="max_abs_delta")
)

plt.figure(figsize=(10, max(6, 0.3 * len(heat_df))))
sns.heatmap(heat_df, cmap="viridis")
plt.title("Subject influence heatmap: max |Δβ| by measure")
plt.xlabel("Measure")
plt.ylabel("Subject")
plt.tight_layout()
plt.savefig(PATH_OUT / "subject_influence_heatmap_measure.png", dpi=200)
plt.close()


# 3) Heatmap per subject x term for each measure
for measure in MEASURES:
    sub = summary_subject_measure_term[summary_subject_measure_term["measure"] == measure].copy()
    if sub.empty:
        continue

    piv = sub.pivot(index="id", columns="term", values="abs_delta")
    cols = [c for c in TERMS_OF_INTEREST if c in piv.columns]
    piv = piv.reindex(columns=cols)

    plt.figure(figsize=(10, max(6, 0.3 * len(piv))))
    sns.heatmap(piv, cmap="magma")
    plt.title(f"Subject influence heatmap: {measure} (|Δβ|)")
    plt.xlabel("Effect")
    plt.ylabel("Subject")
    plt.tight_layout()
    plt.savefig(PATH_OUT / f"subject_influence_heatmap_{measure}.png", dpi=200)
    plt.close()


# 4) Per-measure ranked barplots
for measure in MEASURES:
    sub = summary_subject_measure[summary_subject_measure["measure"] == measure].copy()
    if sub.empty:
        continue

    sub = sub.sort_values("max_abs_delta", ascending=False)

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        data=sub,
        x="id",
        y="max_abs_delta",
        color="steelblue",
    )
    ax.set_title(f"Subject influence ranking for {measure}: max |Δβ|")
    ax.set_xlabel("Subject")
    ax.set_ylabel("max |Δβ| across target effects")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(PATH_OUT / f"subject_influence_ranking_{measure}.png", dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Console summary
# -----------------------------------------------------------------------------
print("\nDone.")
print(f"Output directory: {PATH_OUT}")
print(f"Subjects screened: {df['id'].nunique()}")
print(f"Measures analyzed: {[m for m in MEASURES if m in df_full['measure'].unique()]}")
print(f"Flagged subjects: {len(flagged)}")

if not flagged.empty:
    print("\nFlagged subjects:")
    print(
        flagged[
            ["id", "group", "max_abs_delta", "mean_abs_delta", "max_rel_delta"]
        ]
        .sort_values("max_abs_delta", ascending=False)
        .to_string(index=False)
    )
else:
    print("\nNo subjects flagged by robust influence thresholds.")