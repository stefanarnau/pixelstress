# -----------------------------------------------------------------------------
# Leave-one-subject-out effect-weighted spatial filtering + final MLM
# PARALLELIZED ACROSS HELD-OUT SUBJECTS
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import mne
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_in = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
path_out = Path("/mnt/data_dump/pixelstress/5_effect_weighted_loso/")
path_out.mkdir(parents=True, exist_ok=True)

file_in = path_in / "all_subjects_seq_fooof_rt_channelwise_long.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
measure = "alpha_flat"
target_effect = "group[T.experimental]:f2"

formula = """
score ~ group * f + group * f2
        + mean_trial_difficulty + half
"""

re_formulas = [
    "1 + f + f2",
    "1 + f",
    "1",
]

effects = [
    "f",
    "f2",
    "group[T.experimental]",
    "group[T.experimental]:f",
    "group[T.experimental]:f2",
    "half[T.second]",
    "mean_trial_difficulty",
]

min_subjects_per_electrode_model = 8
min_obs_per_electrode_model = 50

# Parallelization
n_jobs_loso = -1
parallel_backend = "loky"
parallel_verbose = 10


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

for c in [measure, "f", "f2", "mean_trial_difficulty", "mean_rt", "mean_log_rt"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[measure, "f", "f2", "group", "id", "ch_name"]).copy()


# -----------------------------------------------------------------------------
# Electrode list and topomap info
# -----------------------------------------------------------------------------
electrodes = sorted(df["ch_name"].unique())

info = mne.create_info(
    electrodes,
    sfreq=1000,
    ch_types="eeg",
    verbose=None
)
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, on_missing="warn", match_case=False)


# -----------------------------------------------------------------------------
# Helper: fit electrode-wise MLM betas on a dataframe
# -----------------------------------------------------------------------------
def fit_electrode_betas(df_long, measure, electrodes, formula, re_formulas,
                        min_subjects=8, min_obs=50):
    rows = []

    for ch in electrodes:
        dsub = df_long[df_long["ch_name"] == ch].copy()
        dsub = dsub.rename(columns={measure: "score"})

        n_subj = dsub["id"].nunique()
        n_obs = len(dsub)

        if n_subj < min_subjects or n_obs < min_obs:
            continue

        fit = None
        used_re = None

        for re_formula in re_formulas:
            try:
                model = smf.mixedlm(
                    formula,
                    dsub,
                    groups=dsub["id"],
                    re_formula=re_formula
                )

                fit_try = model.fit(
                    method="lbfgs",
                    reml=False,
                    maxiter=4000,
                    disp=False
                )

                if bool(getattr(fit_try, "converged", False)):
                    fit = fit_try
                    used_re = re_formula
                    break

            except Exception:
                pass

        if fit is None:
            continue

        fe = fit.fe_params
        se = fit.bse_fe.reindex(fe.index)

        for term in fe.index:
            rows.append({
                "electrode": ch,
                "term": term,
                "beta": float(fe[term]),
                "se": float(se[term]) if pd.notna(se[term]) else np.nan,
                "t": float(fe[term] / se[term]) if pd.notna(se[term]) and se[term] != 0 else np.nan,
                "random_effects": used_re,
                "n_subjects": int(n_subj),
                "n_obs": int(n_obs),
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Helper: build sequence x electrode matrix for one dataframe
# -----------------------------------------------------------------------------
def make_sequence_matrix(df_long, measure):
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

    if "window" in df_long.columns:
        key_cols = ["window"] + key_cols

    df_use = df_long.dropna(subset=[measure, "ch_name"] + key_cols).copy()

    seq_wide = (
        df_use.pivot_table(
            index=key_cols,
            columns="ch_name",
            values=measure,
            aggfunc="first",
            observed=True,
        )
        .sort_index(axis=1)
    )

    seq_wide = seq_wide.dropna(axis=0, how="any").copy()

    if seq_wide.shape[0] == 0:
        return None, None, None

    meta = seq_wide.index.to_frame(index=False).reset_index(drop=True)
    X = seq_wide.to_numpy(dtype=float)
    return seq_wide.columns.tolist(), meta, X


# -----------------------------------------------------------------------------
# Optional: full-data descriptive beta maps (for inspection only)
# -----------------------------------------------------------------------------
df_beta_full = fit_electrode_betas(
    df,
    measure=measure,
    electrodes=electrodes,
    formula=formula,
    re_formulas=re_formulas,
    min_subjects=min_subjects_per_electrode_model,
    min_obs=min_obs_per_electrode_model,
)

df_beta_full.to_csv(path_out / f"{measure}_electrode_mlm_betas_full_data.csv", index=False)

terms_to_plot = [t for t in effects if t in df_beta_full["term"].unique()]
n_terms = len(terms_to_plot)

if n_terms > 0:
    fig = plt.figure(figsize=(3.5 * n_terms, 4))
    gs = fig.add_gridspec(1, n_terms + 1, width_ratios=[1] * n_terms + [0.05])

    axes = [fig.add_subplot(gs[0, i]) for i in range(n_terms)]
    cax = fig.add_subplot(gs[0, n_terms])

    for i, term in enumerate(terms_to_plot):
        topo = (
            df_beta_full[df_beta_full["term"] == term]
            .set_index("electrode")
            .reindex(electrodes)["beta"]
            .to_numpy()
        )

        vmax = np.nanmax(np.abs(topo))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        evoked = mne.EvokedArray(
            topo[:, None],
            info,
            tmin=0.0,
            verbose=False
        )

        evoked.plot_topomap(
            times=[0],
            axes=axes[i] if i < n_terms - 1 else [axes[i], cax],
            colorbar=(i == n_terms - 1),
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            scalings=1,
            show=False,
            sphere=None
        )

        axes[i].set_title(term)

    fig.suptitle(f"{measure} full-data electrode-wise MLM effect sizes (descriptive)", fontsize=16)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# One LOSO fold
# -----------------------------------------------------------------------------
def run_loso_fold(held_out, df, measure, electrodes, formula, re_formulas,
                  min_subjects_per_electrode_model, min_obs_per_electrode_model,
                  target_effect):

    print(f"LOSO fold: hold out subject {held_out}")

    df_train = df[df["id"].astype(int) != held_out].copy()
    df_test = df[df["id"].astype(int) == held_out].copy()

    # 1) training-set electrode MLM
    df_beta_train = fit_electrode_betas(
        df_train,
        measure=measure,
        electrodes=electrodes,
        formula=formula,
        re_formulas=re_formulas,
        min_subjects=min_subjects_per_electrode_model,
        min_obs=min_obs_per_electrode_model,
    )

    if target_effect not in df_beta_train["term"].unique():
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": f"target effect {target_effect} not estimable in training set"
        }

    # 2) build training-derived filter
    w = (
        df_beta_train[df_beta_train["term"] == target_effect]
        .set_index("electrode")
        .reindex(electrodes)["t"]
        .to_numpy(dtype=float)
    )

    if np.all(~np.isfinite(w)) or np.nansum(np.abs(w)) == 0:
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": "invalid filter"
        }

    w = np.nan_to_num(w, nan=0.0)

    norm = np.linalg.norm(w)
    if norm == 0:
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": "zero-norm filter"
        }

    w = w / norm

    if np.nansum(w) < 0:
        w = -w

    # 3) build held-out sequence matrix
    electrode_names_test, seq_meta_test, X_test = make_sequence_matrix(df_test, measure)

    if X_test is None:
        return {
            "held_out": held_out,
            "scores": None,
            "filter": None,
            "status": "no complete held-out sequence rows"
        }

    if electrode_names_test != electrodes:
        # rebuild explicitly in training order
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
        if "window" in df_test.columns:
            key_cols = ["window"] + key_cols

        seq_wide_test = (
            df_test.pivot_table(
                index=key_cols,
                columns="ch_name",
                values=measure,
                aggfunc="first",
                observed=True,
            )
            .reindex(columns=electrodes)
        )

        seq_wide_test = seq_wide_test.dropna(axis=0, how="any")

        if seq_wide_test.shape[0] == 0:
            return {
                "held_out": held_out,
                "scores": None,
                "filter": None,
                "status": "no complete held-out rows after reordering"
            }

        seq_meta_test = seq_wide_test.index.to_frame(index=False).reset_index(drop=True)
        X_test = seq_wide_test.to_numpy(dtype=float)

    # 4) center held-out data and project
    X_test_c = X_test - X_test.mean(axis=0)
    score_test = X_test_c @ w

    df_scores_fold = seq_meta_test.copy()
    df_scores_fold["held_out_subject"] = held_out
    df_scores_fold["EW_score"] = score_test

    df_filter_fold = pd.DataFrame({
        "held_out_subject": held_out,
        "electrode": electrodes,
        "weight": w,
    })

    return {
        "held_out": held_out,
        "scores": df_scores_fold,
        "filter": df_filter_fold,
        "status": "ok"
    }


# -----------------------------------------------------------------------------
# Parallel LOSO
# -----------------------------------------------------------------------------
subjects = sorted(df["id"].astype(int).unique())

fold_results = Parallel(
    n_jobs=n_jobs_loso,
    backend=parallel_backend,
    verbose=parallel_verbose,
)(
    delayed(run_loso_fold)(
        held_out=s,
        df=df,
        measure=measure,
        electrodes=electrodes,
        formula=formula,
        re_formulas=re_formulas,
        min_subjects_per_electrode_model=min_subjects_per_electrode_model,
        min_obs_per_electrode_model=min_obs_per_electrode_model,
        target_effect=target_effect,
    )
    for s in subjects
)

# collect successful folds
all_scores = []
all_filters = []
failed = []

for res in fold_results:
    if res["status"] == "ok":
        all_scores.append(res["scores"])
        all_filters.append(res["filter"])
    else:
        failed.append({"held_out_subject": res["held_out"], "status": res["status"]})

df_failed = pd.DataFrame(failed)
df_failed.to_csv(
    path_out / f"{measure}_loso_failed_folds_{target_effect.replace('[','').replace(']','').replace(':','x')}.csv",
    index=False
)

if len(all_scores) == 0:
    raise RuntimeError("No LOSO scores were generated.")

df_scores_loso = pd.concat(all_scores, ignore_index=True)
df_filters_loso = pd.concat(all_filters, ignore_index=True)

df_scores_loso.to_csv(
    path_out / f"{measure}_loso_effect_weighted_scores_{target_effect.replace('[','').replace(']','').replace(':','x')}.csv",
    index=False
)
df_filters_loso.to_csv(
    path_out / f"{measure}_loso_filters_{target_effect.replace('[','').replace(']','').replace(':','x')}.csv",
    index=False
)


# -----------------------------------------------------------------------------
# Plot mean LOSO filter (descriptive)
# -----------------------------------------------------------------------------
df_filter_mean = (
    df_filters_loso.groupby("electrode", as_index=False)["weight"]
    .mean()
    .set_index("electrode")
    .reindex(electrodes)
    .reset_index()
)

w_mean = df_filter_mean["weight"].to_numpy(dtype=float)
vmax = np.nanmax(np.abs(w_mean))
if not np.isfinite(vmax) or vmax == 0:
    vmax = 1.0

fig = plt.figure(figsize=(5, 4))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05])
ax = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])

evoked = mne.EvokedArray(
    w_mean[:, None],
    info,
    tmin=0.0,
    verbose=False
)

evoked.plot_topomap(
    times=[0],
    axes=[ax, cax],
    colorbar=True,
    time_format="",
    cmap="RdBu_r",
    vlim=(-vmax, vmax),
    scalings=1,
    show=False,
    sphere=None
)

ax.set_title(f"Mean LOSO filter\n{target_effect}")
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Final MLM on cross-validated scores
# -----------------------------------------------------------------------------
df_scores_loso["id"] = df_scores_loso["id"].astype("category")
df_scores_loso["group"] = df_scores_loso["group"].astype("category")
df_scores_loso["half"] = df_scores_loso["half"].astype("category")

if "window" in df_scores_loso.columns:
    df_scores_loso["window"] = df_scores_loso["window"].astype("category")

df_scores_loso["group"] = df_scores_loso["group"].cat.set_categories(["control", "experimental"])

for c in ["f", "f2", "mean_trial_difficulty", "mean_rt", "mean_log_rt", "EW_score"]:
    if c in df_scores_loso.columns:
        df_scores_loso[c] = pd.to_numeric(df_scores_loso[c], errors="coerce")

dsub = df_scores_loso.dropna(
    subset=["EW_score", "group", "id", "f", "f2", "mean_trial_difficulty", "half"]
).copy()
dsub = dsub.rename(columns={"EW_score": "score"})

fit = None
used_re = None
fit_error_log = []

for re_formula in re_formulas:
    try:
        model = smf.mixedlm(
            formula,
            dsub,
            groups=dsub["id"],
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

if fit is None:
    raise RuntimeError("Final LOSO MLM did not converge:\n" + "\n".join(fit_error_log))

print("\nFinal LOSO effect-weighted score model")
print(f"Measure: {measure}")
print(f"Target effect used to derive filters: {target_effect}")
print(f"Random-effects structure: {used_re}")
print(f"Successful LOSO folds: {len(all_scores)} / {len(subjects)}")
print(fit.summary())

# save tidy results
fe = fit.fe_params
se = fit.bse_fe.reindex(fe.index)
tvals = fe / se.replace(0, np.nan)
pvals = fit.pvalues.reindex(fe.index)

df_final = pd.DataFrame({
    "term": fe.index,
    "beta": fe.values,
    "se": se.values,
    "t": tvals.values,
    "p": pvals.values,
    "random_effects": used_re,
    "n_subjects": dsub["id"].nunique(),
    "n_obs": len(dsub),
    "successful_loso_folds": len(all_scores),
    "total_subjects": len(subjects),
})
df_final.to_csv(
    path_out / f"{measure}_loso_effect_weighted_final_mlm_{target_effect.replace('[','').replace(']','').replace(':','x')}.csv",
    index=False
)


# -----------------------------------------------------------------------------
# Optional model-implied curve for the LOSO score
# -----------------------------------------------------------------------------
group_order = ["control", "experimental"]
f_grid = np.linspace(dsub["f"].min(), dsub["f"].max(), 300)

difficulty_ref = float(dsub["mean_trial_difficulty"].mean())
half_ref = dsub["half"].mode().iloc[0]

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

fig, ax = plt.subplots(figsize=(8, 5))
for g in group_order:
    dg = pred[pred["group"] == g].sort_values("f")
    ax.plot(dg["f"], dg["pred"], linewidth=3, label=f"{g} model")

ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
ax.set_xlabel("Feedback relative to target (0 = target)")
ax.set_ylabel(f"{measure} LOSO effect-weighted score")
ax.set_title(f"LOSO effect-weighted model-implied curve\nfilter from {target_effect}")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()