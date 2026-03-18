# -----------------------------------------------------------------------------
# Electrode-wise MLM effect maps for both reference schemes
# -----------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_in = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
file_in = path_in / "all_subjects_seq_fooof_rt_channelwise_long_car_csd.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
measure = "theta_flat"

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

reference_order = ["car", "csd"]


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(file_in)

df["id"] = df["id"].astype("category")
df["group"] = df["group"].astype("category")
df["ch_name"] = df["ch_name"].astype("category")
df["half"] = df["half"].astype("category")
df["reference"] = df["reference"].astype("category")

df["group"] = df["group"].cat.set_categories(["control", "experimental"])

for c in [measure, "f", "mean_trial_difficulty"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# compute quadratic feedback term
df["f2"] = df["f"] ** 2

df = df.dropna(
    subset=[measure, "f", "f2", "group", "id", "reference"]
).copy()


# -----------------------------------------------------------------------------
# Electrode list
# -----------------------------------------------------------------------------
electrodes = sorted(df["ch_name"].unique())


# -----------------------------------------------------------------------------
# MLM per electrode and reference
# -----------------------------------------------------------------------------
rows = []

available_references = [r for r in reference_order if r in df["reference"].unique()]

for reference in available_references:
    df_ref = df[df["reference"] == reference].copy()

    for ch in electrodes:
        dsub = df_ref[df_ref["ch_name"] == ch].copy()
        dsub = dsub.rename(columns={measure: "score"})

        if dsub["id"].nunique() < 8:
            continue

        fit = None
        used_re_formula = None

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

                if fit_try.converged:
                    fit = fit_try
                    used_re_formula = re_formula
                    break

            except Exception:
                pass

        if fit is None:
            continue

        fe = fit.fe_params

        for term in fe.index:
            rows.append(
                {
                    "reference": reference,
                    "electrode": ch,
                    "term": term,
                    "beta": float(fe[term]),
                    "re_formula": used_re_formula,
                }
            )

df_beta = pd.DataFrame(rows)

if df_beta.empty:
    raise RuntimeError("No converged MLM results available for plotting.")


# -----------------------------------------------------------------------------
# Prepare topomap plotting
# -----------------------------------------------------------------------------
info = mne.create_info(
    electrodes,
    sfreq=1000,
    ch_types="eeg",
)

montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, on_missing="warn", match_case=False)


# -----------------------------------------------------------------------------
# Terms and references to plot
# -----------------------------------------------------------------------------
terms_to_plot = [t for t in effects if t in df_beta["term"].unique()]
refs_to_plot = [r for r in reference_order if r in df_beta["reference"].unique()]

if len(terms_to_plot) == 0:
    raise RuntimeError("None of the requested effects were found in the fitted models.")

n_rows = len(refs_to_plot)
n_terms = len(terms_to_plot)


# -----------------------------------------------------------------------------
# Plot topographies
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(3.5 * n_terms, 4.0 * n_rows))
gs = fig.add_gridspec(
    n_rows,
    n_terms + 1,
    width_ratios=[1] * n_terms + [0.05],
)

for r, reference in enumerate(refs_to_plot):
    row_axes = [fig.add_subplot(gs[r, i]) for i in range(n_terms)]
    cax = fig.add_subplot(gs[r, n_terms])

    for i, term in enumerate(terms_to_plot):
        topo = (
            df_beta[
                (df_beta["reference"] == reference) &
                (df_beta["term"] == term)
            ]
            .set_index("electrode")
            .reindex(electrodes)["beta"]
            .to_numpy()
        )

        vmax = np.nanmax(np.abs(topo))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        vlim = (-vmax, vmax)

        evoked = mne.EvokedArray(
            topo[:, None],
            info,
            tmin=0.0,
            verbose=False,
        )

        ax = row_axes[i]

        evoked.plot_topomap(
            times=[0],
            axes=ax if i < n_terms - 1 else [ax, cax],
            colorbar=(i == n_terms - 1),
            cmap="RdBu_r",
            vlim=vlim,
            scalings=1,
            show=False,
            sphere=None,
        )

        if r == 0:
            ax.set_title(term)

        if i == 0:
            ax.set_ylabel(reference.upper(), fontsize=12)


fig.suptitle(
    f"{measure} electrode-wise MLM effect sizes by reference",
    fontsize=16,
)

plt.tight_layout()
plt.show()