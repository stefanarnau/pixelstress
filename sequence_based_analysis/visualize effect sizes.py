# -----------------------------------------------------------------------------
# Electrode-wise MLM effect maps
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import mne
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_in = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
file_in = path_in / "all_subjects_seq_fooof_rt_channelwise_long.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
measure = "exponent"

formula = """
score ~ group * f + group * f2
        + mean_trial_difficulty + half
"""

re_formulas = [
    "1 + f + f2",
    "1 + f",
    "1"
]

effects = [
    "f",
    "f2",
    "group[T.experimental]",
    "group[T.experimental]:f",
    "group[T.experimental]:f2",
    "half[T.second]",
    "mean_trial_difficulty"
]


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(file_in)

df["id"] = df["id"].astype("category")
df["group"] = df["group"].astype("category")
df["ch_name"] = df["ch_name"].astype("category")
df["half"] = df["half"].astype("category")

df["group"] = df["group"].cat.set_categories(["control","experimental"])

for c in [measure,"f","f2","mean_trial_difficulty"]:
    df[c] = pd.to_numeric(df[c],errors="coerce")

df = df.dropna(subset=[measure,"f","f2","group","id"]).copy()


# -----------------------------------------------------------------------------
# Electrode list
# -----------------------------------------------------------------------------
electrodes = sorted(df["ch_name"].unique())


# -----------------------------------------------------------------------------
# MLM per electrode
# -----------------------------------------------------------------------------
rows = []

for ch in electrodes:

    dsub = df[df["ch_name"]==ch].copy()
    dsub = dsub.rename(columns={measure:"score"})

    if dsub["id"].nunique() < 8:
        continue

    fit = None

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

            if fit_try.converged:
                fit = fit_try
                break

        except:
            pass

    if fit is None:
        continue

    fe = fit.fe_params

    for term in fe.index:

        rows.append({
            "electrode": ch,
            "term": term,
            "beta": float(fe[term])
        })


df_beta = pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Prepare topomap plotting
# -----------------------------------------------------------------------------
info = mne.create_info(
    electrodes,
    sfreq=1000,
    ch_types="eeg"
)

montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage, on_missing="warn", match_case=False)


# -----------------------------------------------------------------------------
# Plot topographies
# -----------------------------------------------------------------------------
terms_to_plot = [t for t in effects if t in df_beta["term"].unique()]

n_terms = len(terms_to_plot)

fig = plt.figure(figsize=(3.5*n_terms,4))
gs = fig.add_gridspec(1,n_terms+1,width_ratios=[1]*n_terms+[0.05])

axes = [fig.add_subplot(gs[0,i]) for i in range(n_terms)]
cax = fig.add_subplot(gs[0,n_terms])

for i, term in enumerate(terms_to_plot):

    topo = (
        df_beta[df_beta["term"] == term]
        .set_index("electrode")
        .reindex(electrodes)["beta"]
        .to_numpy()
    )

    # scale per effect
    vmax = np.nanmax(np.abs(topo))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    vlim = (-vmax, vmax)

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
        vlim=vlim,
        scalings=1,
        show=False,
        sphere=None
    )

    axes[i].set_title(term)

fig.suptitle(
    f"{measure} electrode-wise MLM effect sizes",
    fontsize=16
)

plt.tight_layout()
plt.show()

