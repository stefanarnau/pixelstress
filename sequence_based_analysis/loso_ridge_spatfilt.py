# -----------------------------------------------------------------------------
# LOSO MLM-derived spatial filter
#
# For each held-out subject:
#   1) Fit electrode-wise MLMs on all other subjects.
#   2) Extract channel-wise TERM statistics as spatial weights.
#   3) Normalize weights.
#   4) Project held-out subject's sequence-level scalp data onto weights.
#
# Then:
#   Fit one final MLM on LOSO filter activations.
# -----------------------------------------------------------------------------

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/spatial_filter_loso/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"

MEASURE = "exponent"
TERM = "group[T.experimental]:f2"

# "z" recommended; "beta" also possible
WEIGHT_SOURCE = "z"

GROUP_ORDER = ["control", "experimental"]
MONTAGE_NAME = "standard_1020"

FORMULA = """
roi_val ~ group * f + group * f2
          + mean_trial_difficulty_c + half
"""

RE_FORMULAS = ["1 + f + f2", "1 + f", "1"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fit_mixedlm_with_fallback(df_model, formula, re_formulas):
    for re_formula in re_formulas:
        try:
            model = smf.mixedlm(
                formula=formula,
                data=df_model,
                groups=df_model["id"],
                re_formula=re_formula,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(
                    method="lbfgs",
                    reml=False,
                    maxiter=4000,
                    disp=False,
                )

            if getattr(fit, "converged", False):
                return fit, re_formula

        except Exception:
            pass

    return None, None


def make_info(ch_names):
    info = mne.create_info(ch_names=list(ch_names), sfreq=500, ch_types="eeg")
    montage = mne.channels.make_standard_montage(MONTAGE_NAME)
    info.set_montage(montage, on_missing="ignore", match_case=False)
    return info


def compute_channelwise_weights(df, seq_meta, ch_names, train_ids):
    """
    Fit electrode-wise MLMs on training subjects only.
    Return normalized spatial weights.
    """
    rows = []

    train_ids = set(train_ids)

    for ch in ch_names:
        df_ch = (
            df[
                (df["ch_name"] == ch)
                & (df["id"].isin(train_ids))
            ]
            .groupby(["id", "block_nr", "sequence_nr"], as_index=False)[MEASURE]
            .mean()
            .rename(columns={MEASURE: "roi_val"})
        )

        dch = seq_meta[seq_meta["id"].isin(train_ids)].merge(
            df_ch,
            on=["id", "block_nr", "sequence_nr"],
            how="inner",
        )

        dch = dch.dropna(
            subset=[
                "roi_val", "group", "id", "half",
                "f", "f2", "mean_trial_difficulty_c",
            ]
        ).copy()

        fit, used_re = fit_mixedlm_with_fallback(
            dch,
            FORMULA,
            RE_FORMULAS,
        )

        if fit is None:
            beta = np.nan
            se = np.nan
            z = np.nan
            p = np.nan
        else:
            beta = fit.fe_params.get(TERM, np.nan)
            se = fit.bse_fe.get(TERM, np.nan)
            z = beta / se if np.isfinite(se) and se > 0 else np.nan
            p = fit.pvalues.get(TERM, np.nan)

        rows.append(
            {
                "ch_name": ch,
                "beta": beta,
                "se": se,
                "z": z,
                "p": p,
                "used_re": used_re,
            }
        )

    beta_map = pd.DataFrame(rows)

    vals = (
        beta_map.set_index("ch_name")
        .reindex(ch_names)[WEIGHT_SOURCE]
        .to_numpy(dtype=float)
    )

    vals = np.nan_to_num(vals, nan=0.0)

    norm = np.sqrt(np.sum(vals ** 2))
    if norm == 0:
        vals[:] = np.nan
    else:
        vals = vals / norm

    beta_map["weight"] = vals

    return vals, beta_map


# -----------------------------------------------------------------------------
# Load / prepare
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_IN)

df["id"] = df["id"].astype(str)
df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER)
df["half"] = pd.Categorical(df["half"])
df["ch_name"] = df["ch_name"].astype(str)

for col in [MEASURE, "f", "mean_trial_difficulty"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

seq_meta = (
    df[
        [
            "id", "group", "block_nr", "sequence_nr",
            "half", "mean_trial_difficulty", "f"
        ]
    ]
    .drop_duplicates()
    .copy()
)

seq_meta["f_c"] = seq_meta["f"] - seq_meta["f"].mean()
seq_meta["f2"] = seq_meta["f_c"] ** 2 - np.mean(seq_meta["f_c"] ** 2)
seq_meta["mean_trial_difficulty_c"] = (
    seq_meta["mean_trial_difficulty"]
    - seq_meta["mean_trial_difficulty"].mean()
)

ch_names = sorted(df["ch_name"].dropna().unique().tolist())
subjects = sorted(df["id"].dropna().unique().tolist())

print("Measure:", MEASURE)
print("Term:", TERM)
print("Weight source:", WEIGHT_SOURCE)
print("Subjects:", len(subjects))
print("Channels:", len(ch_names))


# -----------------------------------------------------------------------------
# Wide sequence-level EEG matrix
# -----------------------------------------------------------------------------
wide = (
    df.pivot_table(
        index=["id", "group", "block_nr", "sequence_nr"],
        columns="ch_name",
        values=MEASURE,
        aggfunc="mean",
    )
    .reset_index()
)

for ch in ch_names:
    if ch not in wide.columns:
        wide[ch] = np.nan

X_all = wide[ch_names].to_numpy(dtype=float)

# Fill rare missing values with channel means.
col_means = np.nanmean(X_all, axis=0)
inds = np.where(np.isnan(X_all))
X_all[inds] = np.take(col_means, inds[1])

wide[ch_names] = X_all


# -----------------------------------------------------------------------------
# LOSO projection
# -----------------------------------------------------------------------------
loso_rows = []
weight_rows = []

for k, heldout_id in enumerate(subjects, start=1):
    print(f"\nLOSO {k}/{len(subjects)} | held out subject {heldout_id}")

    train_ids = [s for s in subjects if s != heldout_id]

    weights, beta_map = compute_channelwise_weights(
        df=df,
        seq_meta=seq_meta,
        ch_names=ch_names,
        train_ids=train_ids,
    )

    beta_map["heldout_id"] = heldout_id
    weight_rows.append(beta_map)

    held = wide[wide["id"] == heldout_id].copy()
    X_held = held[ch_names].to_numpy(dtype=float)

    held["roi_val"] = X_held @ weights
    held["heldout_id"] = heldout_id

    loso_rows.append(
        held[["id", "group", "block_nr", "sequence_nr", "roi_val", "heldout_id"]]
    )

loso_activation = pd.concat(loso_rows, ignore_index=True)
loso_weights = pd.concat(weight_rows, ignore_index=True)

loso_weights.to_csv(
    PATH_OUT / f"{MEASURE}_{TERM}_loso_spatial_filter_weights.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Merge LOSO activation with metadata
# -----------------------------------------------------------------------------
filter_data = seq_meta.merge(
    loso_activation[["id", "block_nr", "sequence_nr", "roi_val"]],
    on=["id", "block_nr", "sequence_nr"],
    how="inner",
)

filter_data = filter_data.dropna(
    subset=[
        "roi_val", "group", "id", "half",
        "f", "f2", "mean_trial_difficulty_c",
    ]
).copy()

filter_data.to_csv(
    PATH_OUT / f"{MEASURE}_{TERM}_loso_filter_activation_sequence_data.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Final MLM on LOSO filter activation
# -----------------------------------------------------------------------------
fit_filter, used_re_filter = fit_mixedlm_with_fallback(
    filter_data,
    FORMULA,
    RE_FORMULAS,
)

if fit_filter is None:
    print("LOSO filter activation MLM failed.")
else:
    print("\nLOSO filter activation MLM:")
    print(fit_filter.summary())
    print("Random effects:", used_re_filter)

    fe = fit_filter.fe_params
    se = fit_filter.bse_fe.reindex(fe.index)
    zvals = fe / se.replace(0, np.nan)
    pvals = fit_filter.pvalues.reindex(fe.index)

    filter_res = pd.DataFrame(
        {
            "term": fe.index,
            "beta": fe.values,
            "se": se.values,
            "z": zvals.values,
            "p": pvals.values,
            "used_re": used_re_filter,
        }
    )

    filter_res.to_csv(
        PATH_OUT / f"{MEASURE}_{TERM}_loso_filter_activation_mlm_results.csv",
        index=False,
    )


# -----------------------------------------------------------------------------
# Plot average LOSO spatial filter
# -----------------------------------------------------------------------------
mean_weights = (
    loso_weights
    .groupby("ch_name", as_index=False)["weight"]
    .mean()
)

vals = (
    mean_weights.set_index("ch_name")
    .reindex(ch_names)["weight"]
    .to_numpy(dtype=float)
)

info = make_info(ch_names)

vmax = np.nanmax(np.abs(vals))
vlim = (-vmax, vmax)

fig, ax = plt.subplots(figsize=(5, 4))
im, _ = mne.viz.plot_topomap(
    vals,
    info,
    axes=ax,
    show=False,
    cmap="RdBu_r",
    sensors=True,
    contours=6,
    vlim=vlim,
)
ax.set_title(
    f"Mean LOSO spatial filter weights\n{MEASURE}: {TERM}"
)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label(f"mean normalized {WEIGHT_SOURCE} weight")

fig.savefig(
    PATH_OUT / f"{MEASURE}_{TERM}_mean_loso_spatial_filter_topomap.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()

print("\nFinished.")
print("Saved to:", PATH_OUT)