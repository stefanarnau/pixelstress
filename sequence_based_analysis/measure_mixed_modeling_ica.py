# -----------------------------------------------------------------------------
# Group ICA + mixed models for sequence-based EEG measures
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_in = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
path_out = Path("/mnt/data_dump/pixelstress/4_group_ica/")
path_out.mkdir(parents=True, exist_ok=True)

file_in = path_in / "all_subjects_seq_fooof_rt_channelwise_long.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
measure = "exponent"   # change to "theta_flat", "beta_flat", "alpha_flat", or "exponent"

# used only to determine how many ICs to extract
variance_threshold = 0.80

# optional hard cap on IC count
max_n_components = 20

min_subjects_per_component_model = 8
min_obs_per_component_model = 50

random_state = 42

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

df["id"] = df["id"].astype("category")
df["group"] = df["group"].astype("category")
df["ch_name"] = df["ch_name"].astype("category")
df["half"] = df["half"].astype("category")

if "window" in df.columns:
    df["window"] = df["window"].astype("category")

if "control" in df["group"].astype(str).unique() and "experimental" in df["group"].astype(str).unique():
    df["group"] = df["group"].cat.set_categories(["control", "experimental"])

for c in [measure, "f", "f2", "mean_trial_difficulty", "mean_rt", "mean_log_rt"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# -----------------------------------------------------------------------------
# Build sequence-level electrode matrix
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

df_use = df.dropna(subset=[measure, "ch_name"] + key_cols).copy()

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
    raise RuntimeError("No complete sequence rows available for ICA.")

electrode_names = seq_wide.columns.tolist()
seq_meta = seq_wide.index.to_frame(index=False).reset_index(drop=True)
X = seq_wide.to_numpy(dtype=float)


# -----------------------------------------------------------------------------
# Center only (no variance scaling)
# -----------------------------------------------------------------------------
Xc = X - X.mean(axis=0)


# -----------------------------------------------------------------------------
# Determine ICA dimensionality using PCA
# -----------------------------------------------------------------------------
pca_full = PCA()
pca_full.fit(Xc)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
n_components = min(n_components, max_n_components, Xc.shape[1], Xc.shape[0])

print(
    f"Using {n_components} ICA components "
    f"(PCA prepass reached {variance_threshold:.0%} variance at this dimensionality)."
)

df_dim = pd.DataFrame({
    "component": [f"IC{i+1}" for i in range(n_components)],
    "pca_explained_variance_ratio_reference": pca_full.explained_variance_ratio_[:n_components],
    "pca_cumulative_explained_variance_reference": np.cumsum(pca_full.explained_variance_ratio_[:n_components]),
})
df_dim.to_csv(path_out / f"{measure}_ica_dimension_reference.csv", index=False)


# -----------------------------------------------------------------------------
# Group ICA
# -----------------------------------------------------------------------------
ica = FastICA(
    n_components=n_components,
    algorithm="parallel",
    whiten="unit-variance",
    fun="logcosh",
    max_iter=2000,
    tol=1e-4,
    random_state=random_state,
)

# scores: sequence x component
scores = ica.fit_transform(Xc)

# loadings / mixing matrix: electrode x component
loadings = ica.mixing_

component_names = [f"IC{i+1}" for i in range(n_components)]

# sign flip for consistency
for i in range(n_components):
    if np.nansum(loadings[:, i]) < 0:
        loadings[:, i] *= -1
        scores[:, i] *= -1


# -----------------------------------------------------------------------------
# Save ICA outputs
# -----------------------------------------------------------------------------
df_loadings = pd.DataFrame(
    loadings,
    index=electrode_names,
    columns=component_names,
).reset_index().rename(columns={"index": "electrode"})
df_loadings.to_csv(path_out / f"{measure}_ica_loadings.csv", index=False)

df_scores = seq_meta.copy()
for i, name in enumerate(component_names):
    df_scores[name] = scores[:, i]

df_scores.to_csv(path_out / f"{measure}_sequence_ica_scores.csv", index=False)


# -----------------------------------------------------------------------------
# Plot ICA topographies
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import mne

info = mne.create_info(
    electrode_names,
    sfreq=1000,
    ch_types="eeg",
)

montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, on_missing="warn", match_case=False)

n_ic = loadings.shape[1]

fig = plt.figure(figsize=(3.2 * n_ic, 3.5))
gs = fig.add_gridspec(1, n_ic + 1, width_ratios=[1] * n_ic + [0.05])

axes = [fig.add_subplot(gs[0, i]) for i in range(n_ic)]
cax = fig.add_subplot(gs[0, n_ic])

vmax = np.max(np.abs(loadings))
vlim = (-vmax, vmax)

for i in range(n_ic):
    topo = loadings[:, i]

    evoked = mne.EvokedArray(
        topo[:, None],
        info,
        tmin=0.0,
        verbose=False,
    )

    evoked.plot_topomap(
        times=[0],
        axes=axes[i] if i < n_ic - 1 else [axes[i], cax],
        colorbar=(i == n_ic - 1),
        time_format="",
        cmap="RdBu_r",
        vlim=vlim,
        scalings=1,
        show=False,
        sphere=None,
    )

    axes[i].set_title(f"IC{i+1}")

fig.suptitle(f"{measure} ICA loadings (topographies)", fontsize=16)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Mixed models on ICA scores
# -----------------------------------------------------------------------------
results = []
errors = []
fitted_models = {}

df_scores["id"] = df_scores["id"].astype("category")
df_scores["group"] = df_scores["group"].astype("category")
df_scores["half"] = df_scores["half"].astype("category")

if "window" in df_scores.columns:
    df_scores["window"] = df_scores["window"].astype("category")

if "control" in df_scores["group"].astype(str).unique() and "experimental" in df_scores["group"].astype(str).unique():
    df_scores["group"] = df_scores["group"].cat.set_categories(["control", "experimental"])

for c in ["f", "f2", "mean_trial_difficulty", "mean_rt", "mean_log_rt"]:
    if c in df_scores.columns:
        df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce")

for ic in component_names:
    dsub = df_scores.dropna(subset=[ic, "group", "id", "f", "f2", "mean_trial_difficulty", "half"]).copy()
    dsub = dsub.rename(columns={ic: "score"})

    n_subj = dsub["id"].nunique()
    n_obs = len(dsub)

    if n_subj < min_subjects_per_component_model or n_obs < min_obs_per_component_model:
        errors.append({
            "component": ic,
            "n_subjects": int(n_subj),
            "n_obs": int(n_obs),
            "error": "Too few observations",
        })
        continue

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
        errors.append({
            "component": ic,
            "n_subjects": int(n_subj),
            "n_obs": int(n_obs),
            "error": " | ".join(fit_error_log),
        })
        continue

    fitted_models[ic] = fit

    fe = fit.fe_params
    se = fit.bse_fe
    tvals = fe / se.replace(0, np.nan)
    pvals = fit.pvalues.reindex(fe.index)

    for term in fe.index:
        results.append({
            "measure": measure,
            "component": ic,
            "term": term,
            "beta": float(fe[term]),
            "se": float(se[term]),
            "t": float(tvals[term]),
            "p": float(pvals[term]) if pd.notna(pvals[term]) else np.nan,
            "n_subjects": int(n_subj),
            "n_obs": int(n_obs),
            "random_effects": used_re,
            "converged": bool(getattr(fit, "converged", True)),
            "llf": float(fit.llf),
            "aic": float(fit.aic) if np.isfinite(fit.aic) else np.nan,
            "bic": float(fit.bic) if np.isfinite(fit.bic) else np.nan,
        })

df_mlm = pd.DataFrame(results)
df_err = pd.DataFrame(errors)

df_mlm.to_csv(path_out / f"{measure}_ica_mixedlm_results.csv", index=False)
df_err.to_csv(path_out / f"{measure}_ica_mixedlm_errors.csv", index=False)

print("Done.")
print(f"Measure: {measure}")
print(f"Sequence rows used for ICA: {X.shape[0]}")
print(f"Electrodes used for ICA: {X.shape[1]}")
print(f"Components retained: {n_components}")
print(f"Successful model rows: {len(df_mlm)}")
print(f"Failed component fits: {len(df_err)}")

if not df_mlm.empty:
    print("\nEffects of interest:")
    effects_of_interest = [
        "f",
        "f2",
        "group[T.experimental]",
        "group[T.experimental]:f",
        "group[T.experimental]:f2",
    ]
    view = (
        df_mlm[df_mlm["term"].isin(effects_of_interest)]
        .sort_values(["component", "term"])
        .reset_index(drop=True)
    )
    print(view.head(50))