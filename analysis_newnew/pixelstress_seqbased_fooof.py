# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io
from fooof import FOOOFGroup
from mne.time_frequency import psd_array_multitaper
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_out = Path("/mnt/data_dump/pixelstress/3_sequence_data_fooof/")
path_out.mkdir(parents=True, exist_ok=True)

path_in = "/mnt/data_dump/pixelstress/2_autocleaned_45/"
datasets = glob.glob(f"{path_in}/*erp.set")


# -----------------------------------------------------------------------------
# Exclusion list
# -----------------------------------------------------------------------------
ids_to_drop = {1, 2, 3, 4, 5, 6, 13, 17, 18, 25, 40, 49, 83}


# -----------------------------------------------------------------------------
# Channel labels
# -----------------------------------------------------------------------------
channel_labels = (
    open("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt", "r")
    .read()
    .split("\n")[:-1]
)

info_tf = mne.create_info(channel_labels, sfreq=200, ch_types="eeg", verbose=None)

montage = mne.channels.make_standard_montage("standard_1020")
info_tf.set_montage(montage, on_missing="warn", match_case=False)


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------
windows = {
    "pre_cue": (-2.4, -1.2),
    "cue_target": (-1.2, 0.0),
    "task": (0.0, 1.2),
}

fmin_fit, fmax_fit = 3.0, 35.0
mt_bandwidth = 3.0

min_trials_per_sequence = 5


# -----------------------------------------------------------------------------
# Subject processing function
# -----------------------------------------------------------------------------
def process_subject(dataset):

    base = dataset.split("_cleaned")[0]

    df_tf = pd.read_csv(base + "_tf_trialinfo.csv")

    subj_id = int(df_tf["id"].iloc[0])

    if subj_id in ids_to_drop:
        return None


    # -------------------------------------------------------------------------
    # Load EEG data
    # -------------------------------------------------------------------------
    mat_path = dataset.split("_erp.set")[0] + "_tf.set"

    mat = scipy.io.loadmat(mat_path)

    tf_data = np.transpose(mat["data"], [2, 0, 1])
    tf_times = mat["times"].ravel().astype(float)

    tf_times_sec = tf_times / 1000.0 if np.nanmax(np.abs(tf_times)) > 20 else tf_times


    # -------------------------------------------------------------------------
    # Precompute window indices
    # -------------------------------------------------------------------------
    window_indices = {
        w: np.where((tf_times_sec >= t0) & (tf_times_sec < t1))[0]
        for w, (t0, t1) in windows.items()
    }


    # -------------------------------------------------------------------------
    # Trial preprocessing
    # -------------------------------------------------------------------------
    df_tf["accuracy"] = (df_tf["accuracy"] == 1).astype(int)

    df_tf = df_tf.rename(columns={"session_condition": "group"})
    df_tf["group"] = df_tf["group"].replace({1: "experimental", 2: "control"})


    mask = df_tf["sequence_nr"] > 1
    df_tf = df_tf.loc[mask].reset_index(drop=True)
    tf_data = tf_data[mask.to_numpy(), :, :]


    mask = df_tf["accuracy"] == 1
    df_tf = df_tf.loc[mask].reset_index(drop=True)
    tf_data = tf_data[mask.to_numpy(), :, :]


    g = df_tf.groupby(["block_nr", "sequence_nr"], sort=True)


    seq_rows = []
    seq_psd = []
    seq_index = []


    # -------------------------------------------------------------------------
    # Sequence loop
    # -------------------------------------------------------------------------
    for (block_nr, seq_nr), idx in g.indices.items():

        idx = np.asarray(idx)

        n_trials = len(idx)

        if n_trials < min_trials_per_sequence:
            continue


        df_sub = df_tf.loc[idx]

        f = float(df_sub["last_feedback_scaled"].iloc[0])
        mean_difficulty = float(df_sub["trial_difficulty"].mean())
        half = "first" if int(block_nr) <= 4 else "second"
        group = df_sub["group"].iloc[0]


        for wname, tidx in window_indices.items():

            if tidx.size == 0:
                continue


            x = tf_data[idx][:, :, tidx]


            # -----------------------------------------------------------------
            # PSD computation
            # -----------------------------------------------------------------
            psd, freqs = psd_array_multitaper(
                x,
                sfreq=info_tf["sfreq"],
                fmin=fmin_fit,
                fmax=fmax_fit,
                bandwidth=mt_bandwidth,
                normalization="full",
                verbose=False,
                n_jobs=1
            )


            psd_seq = psd.mean(axis=0)


            seq_psd.append(psd_seq)

            seq_index.append(
                {
                    "id": subj_id,
                    "group": group,
                    "block_nr": int(block_nr),
                    "sequence_nr": int(seq_nr),
                    "window": wname,
                    "half": half,
                    "n_trials": int(n_trials),
                    "mean_trial_difficulty": mean_difficulty,
                    "f": f,
                    "f2": f**2,
                }
            )


            # -----------------------------------------------------------------
            # Batch FOOOF fitting
            # -----------------------------------------------------------------
            fg = FOOOFGroup(
                aperiodic_mode="fixed",
                peak_width_limits=(2, 12),
                max_n_peaks=6,
                min_peak_height=0.1,
                verbose=False
            )

            fg.fit(freqs, psd_seq, [fmin_fit, fmax_fit])


            aperiodic = fg.get_params("aperiodic_params")
            r2 = fg.get_params("r_squared")
            err = fg.get_params("error")


            offsets = aperiodic[:, 0]
            exponents = aperiodic[:, 1]


            for ci in range(psd_seq.shape[0]):

                seq_rows.append(
                    {
                        "id": subj_id,
                        "group": group,
                        "block_nr": int(block_nr),
                        "sequence_nr": int(seq_nr),
                        "half": half,
                        "n_trials": int(n_trials),
                        "mean_trial_difficulty": mean_difficulty,
                        "f": f,
                        "f2": f**2,
                        "window": wname,
                        "ch_ix": int(ci),
                        "ch_name": channel_labels[ci],
                        "offset": offsets[ci],
                        "exponent": exponents[ci],
                        "r2": r2[ci],
                        "error": err[ci],
                    }
                )


    df_seq_fooof = pd.DataFrame(seq_rows)

    if len(df_seq_fooof) == 0:
        return None


    df_seq_fooof = df_seq_fooof[
        (df_seq_fooof["r2"] >= 0.8) &
        (df_seq_fooof["error"] <= 0.2)
    ].copy()


    # -------------------------------------------------------------------------
    # Save subject results
    # -------------------------------------------------------------------------
    out_csv = path_out / f"sub-{subj_id:03d}_seq_fooof_channelwise.csv"

    df_seq_fooof.to_csv(out_csv, index=False)


    if len(seq_psd) > 0:

        psd_arr = np.stack(seq_psd, axis=0)

        np.savez_compressed(
            path_out / f"sub-{subj_id:03d}_seq_psd_channelwise.npz",
            psd=psd_arr,
            freqs=freqs,
            channels=np.array(channel_labels)
        )

        pd.DataFrame(seq_index).to_csv(
            path_out / f"sub-{subj_id:03d}_seq_psd_channelwise_index.csv",
            index=False
        )


    print(f"Saved subject {subj_id:03d}")

    return df_seq_fooof


# -----------------------------------------------------------------------------
# Run subjects in parallel
# -----------------------------------------------------------------------------
seq_data = Parallel(
    n_jobs=-1,
    backend="loky",
    verbose=10
)(
    delayed(process_subject)(dataset) for dataset in datasets
)


seq_data = [d for d in seq_data if d is not None and len(d) > 0]

df = pd.concat(seq_data, ignore_index=True)


# -----------------------------------------------------------------------------
# Save combined dataframe
# -----------------------------------------------------------------------------
df.to_csv(
    path_out / "all_subjects_seq_fooof_channelwise_long.csv",
    index=False
)

print("Finished.")



# ------------------------------------------------------------
# Starting point: final combined dataframe
# ------------------------------------------------------------
# df = pd.read_csv("all_subjects_seq_fooof_channelwise_long.csv")

# Ensure correct datatypes
df["id"] = df["id"].astype("category")
df["group"] = df["group"].astype("category")
df["ch_name"] = df["ch_name"].astype("category")
df["window"] = df["window"].astype("category")
df["half"] = df["half"].astype("category")

df["group"] = df["group"].cat.reorder_categories(
    ["control", "experimental"], ordered=False
)

for c in ["f", "f2", "mean_trial_difficulty", "exponent"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[
    "exponent",
    "f",
    "f2",
    "mean_trial_difficulty",
    "half",
    "group",
    "id",
    "ch_name",
    "window",
]).copy()

# ------------------------------------------------------------
# Model formula
# ------------------------------------------------------------
formula = """
exponent ~ group * f + group * f2
           + mean_trial_difficulty + half
"""

# Ordered fallback random-effects structures
re_formulas = [
    "1 + f + f2",
    "1 + f",
    "1",
]

results = []
errors = []
fitted_models = {}

# ------------------------------------------------------------
# Loop over windows and electrodes
# ------------------------------------------------------------
for w in df["window"].cat.categories:
    for ch in df["ch_name"].cat.categories:

        dsub = df[(df["window"] == w) & (df["ch_name"] == ch)].copy()

        n_subj = dsub["id"].nunique()
        n_obs = len(dsub)

        if n_subj < 8 or n_obs < 50:
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

                # Accept only if statsmodels reports convergence
                if bool(getattr(fit_try, "converged", False)):
                    fit = fit_try
                    used_re = re_formula
                    break
                else:
                    fit_error_log.append(
                        f"re_formula={re_formula}: fit returned converged=False"
                    )

            except Exception as e:
                fit_error_log.append(f"re_formula={re_formula}: {str(e)}")

        if fit is None:
            errors.append({
                "window": str(w),
                "electrode": str(ch),
                "n_subjects": int(n_subj),
                "n_obs": int(n_obs),
                "error": " | ".join(fit_error_log),
            })
            continue

        fitted_models[(str(w), str(ch))] = fit

        fe = fit.fe_params
        se = fit.bse_fe
        tvals = fe / se.replace(0, np.nan)
        pvals = fit.pvalues.reindex(fe.index)

        for term in fe.index:
            results.append({
                "window": str(w),
                "electrode": str(ch),
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

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
df_mlm.to_csv("mixedlm_exponent_by_window_electrode.csv", index=False)
df_err.to_csv("mixedlm_exponent_model_errors.csv", index=False)

print("Models completed.")
print("Successful coefficient rows:", len(df_mlm))
print("Failed fits:", len(df_err))

# Quick summary: how often each random-effects structure was needed
if not df_mlm.empty:
    print("\nRandom-effects structures used:")
    print(df_mlm[["window", "electrode", "random_effects"]]
          .drop_duplicates()["random_effects"]
          .value_counts())

# Quick view of effects of interest
effects_of_interest = [
    "f",
    "f2",
    "group[T.experimental]",
    "group[T.experimental]:f",
    "group[T.experimental]:f2",
]

if "term" in df_mlm.columns and not df_mlm.empty:
    view = (
        df_mlm.loc[df_mlm["term"].isin(effects_of_interest)]
              .sort_values(["window", "electrode", "term"])
              .reset_index(drop=True)
    )
    print(view.head(50))
else:
    print("No successful model results to display.")
    
    
    
df["exponent"].describe()

df.groupby(["window"])["exponent"].describe()




terms = {
    "f": "Feedback (f)",
    "f2": "Feedback² (f²)",
    "group[T.experimental]": "Group",
    "group[T.experimental]:f": "Group × f",
    "group[T.experimental]:f2": "Group × f²",
}

windows = ["pre_cue", "cue_target", "task"]

for term, title in terms.items():

    vmax = np.nanmax(np.abs(df_mlm.loc[df_mlm["term"] == term, "beta"]))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    for i, win in enumerate(windows):

        d = df_mlm[
            (df_mlm["term"] == term) &
            (df_mlm["window"] == win)
        ].copy()

        d = d.set_index("electrode").reindex(channel_labels)

        data = d["beta"].to_numpy(dtype=float)

        # uncorrected significance mask
        if "p" in d.columns:
            mask = (d["p"].to_numpy(dtype=float) < 0.05)[:, np.newaxis]
        else:
            mask = None

        evoked = mne.EvokedArray(
            data[:, None],
            info_tf,
            tmin=0.0,
            verbose=False
        )

        evoked.plot_topomap(
            times=[0],
            axes=axes[i] if i < 2 else [axes[i], cax],
            colorbar=(i == 2),
            time_format="",
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            scalings=1,
            show=False,
            sphere=None,
            mask=mask,
            mask_params=dict(
                marker="o",
                markerfacecolor="none",
                markeredgecolor="k",
                linewidth=1.2,
                markersize=6,
            ),
        )

        axes[i].set_title(win)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()






# ---------------------------------------------------------------------
# Assumes these already exist:
# df            : combined long dataframe with exponent/r2/error/ch_name etc.
# df_mlm        : coefficient dataframe from the electrode-wise mixed models
# info_tf       : MNE Info with montage
# channel_labels: channel order used throughout
# ---------------------------------------------------------------------

# -----------------------------
# Settings
# -----------------------------
term_to_inspect = "f2"         # e.g. "f", "f2", "group[T.experimental]:f"
window_to_inspect = "task"     # e.g. "pre_cue", "cue_target", "task"
n_show = 3                     # how many strongest electrodes to inspect

# FOOOF settings used in your pipeline
fmin_fit, fmax_fit = 3.0, 35.0

# -----------------------------
# Helper: plot one topomap from channel-wise values
# -----------------------------
def plot_single_topomap(values_by_ch, title, info, channel_order, cmap="RdBu_r", symmetric=False):
    vals = values_by_ch.reindex(channel_order).to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if symmetric else np.nanmax(vals)
    vmin = -vmax if symmetric else np.nanmin(vals)

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    if not np.isfinite(vmin):
        vmin = -1.0 if symmetric else 0.0

    fig = plt.figure(figsize=(4.5, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.06])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    evoked = mne.EvokedArray(vals[:, None], info, tmin=0.0, verbose=False)
    evoked.plot_topomap(
        times=[0],
        axes=[ax, cax],
        colorbar=True,
        time_format="",
        cmap=cmap,
        vlim=(vmin, vmax),
        scalings=1,
        show=False,
        sphere=None,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 1) Mean exponent topomap
# -----------------------------
mean_exp = df.groupby("ch_name")["exponent"].mean()
plot_single_topomap(
    mean_exp,
    title="Mean aperiodic exponent",
    info=info_tf,
    channel_order=channel_labels,
    cmap="RdBu_r",
    symmetric=False,
)

# -----------------------------
# 2) Mean r² topomap
# -----------------------------
mean_r2 = df.groupby("ch_name")["r2"].mean()
plot_single_topomap(
    mean_r2,
    title="Mean FOOOF r²",
    info=info_tf,
    channel_order=channel_labels,
    cmap="viridis",
    symmetric=False,
)

# -----------------------------
# 3) Mean error topomap
# -----------------------------
mean_err = df.groupby("ch_name")["error"].mean()
plot_single_topomap(
    mean_err,
    title="Mean FOOOF error",
    info=info_tf,
    channel_order=channel_labels,
    cmap="magma_r",
    symmetric=False,
)

# -----------------------------
# 4) Pick suspicious electrodes from mixed-model coefficients
# -----------------------------
coef_sub = df_mlm[
    (df_mlm["term"] == term_to_inspect) &
    (df_mlm["window"] == window_to_inspect)
].copy()

coef_sub = coef_sub.sort_values("beta", ascending=False)

top_pos = coef_sub["electrode"].head(n_show).tolist()
top_neg = coef_sub["electrode"].tail(n_show).tolist()

# Add a few reference electrodes if they exist
reference_ch = [ch for ch in ["Fz", "Cz", "Pz", "Oz"] if ch in channel_labels]

electrodes_to_check = []
for ch in top_pos + top_neg + reference_ch:
    if ch not in electrodes_to_check:
        electrodes_to_check.append(ch)

print("Electrodes selected for PSD/FOOOF inspection:")
print(electrodes_to_check)

# -----------------------------
# 5) Load PSD index + PSD arrays from saved subject files
#    and compute average PSD per selected electrode for the chosen window
# -----------------------------
# We use the saved subject-level PSD files:
# sub-XXX_seq_psd_channelwise.npz
# sub-XXX_seq_psd_channelwise_index.csv

all_psd_rows = []
freqs_ref = None

for subj_id in sorted(df["id"].astype(int).unique()):
    npz_file = Path(path_out) / f"sub-{subj_id:03d}_seq_psd_channelwise.npz"
    idx_file = Path(path_out) / f"sub-{subj_id:03d}_seq_psd_channelwise_index.csv"

    if not npz_file.exists() or not idx_file.exists():
        continue

    arr = np.load(npz_file, allow_pickle=True)
    psd_arr = arr["psd"]              # shape: (n_seqwin, n_ch, n_freqs)
    freqs = arr["freqs"]

    if freqs_ref is None:
        freqs_ref = freqs

    df_idx = pd.read_csv(idx_file)

    keep = df_idx["window"] == window_to_inspect
    if keep.sum() == 0:
        continue

    psd_sel = psd_arr[keep.to_numpy(), :, :]   # (n_seq_in_window, n_ch, n_freqs)

    # average across sequences for this subject
    psd_mean_subj = np.nanmean(psd_sel, axis=0)  # (n_ch, n_freqs)
    all_psd_rows.append(psd_mean_subj)

if len(all_psd_rows) == 0:
    raise RuntimeError("No saved PSD files found or no PSD rows matched the selected window.")

# average across subjects
psd_grand = np.nanmean(np.stack(all_psd_rows, axis=0), axis=0)  # (n_ch, n_freqs)

# -----------------------------
# 6) Plot average PSD + FOOOF fit for selected electrodes
# -----------------------------
n_cols = 3
n_rows = int(np.ceil(len(electrodes_to_check) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

for ax, ch in zip(axes.ravel(), electrodes_to_check):
    ch_ix = channel_labels.index(ch)
    spectrum = psd_grand[ch_ix]

    fm = FOOOF(
        aperiodic_mode="fixed",
        peak_width_limits=(2, 12),
        max_n_peaks=6,
        min_peak_height=0.1,
        verbose=False,
    )
    fm.fit(freqs_ref, spectrum, [fmin_fit, fmax_fit])

    # plot spectrum and fit
    fm.plot(ax=ax, plot_peaks="shade")
    ax.set_title(
        f"{ch}\nexp={fm.aperiodic_params_[1]:.3f}, r²={fm.r_squared_:.3f}, err={fm.error_:.3f}"
    )

# hide unused axes
for ax in axes.ravel()[len(electrodes_to_check):]:
    ax.axis("off")

fig.suptitle(f"Average PSD + FOOOF fits ({window_to_inspect}, selected electrodes)", fontsize=14)
plt.tight_layout()
plt.show()

# -----------------------------
# 7) Print quick diagnostic summaries for selected electrodes
# -----------------------------
diag = (
    df[df["window"] == window_to_inspect]
    .groupby("ch_name")
    .agg(
        mean_exp=("exponent", "mean"),
        std_exp=("exponent", "std"),
        mean_r2=("r2", "mean"),
        mean_err=("error", "mean"),
        n=("exponent", "count"),
    )
    .reindex(electrodes_to_check)
)

coef_diag = (
    coef_sub.set_index("electrode")[["beta", "p", "n_subjects", "n_obs"]]
    .reindex(electrodes_to_check)
)

print("\nElectrode diagnostics:")
print(pd.concat([diag, coef_diag], axis=1))






# ---------------------------------------------------------------------
# Assumes df contains:
# id, group, ch_name, window, sequence_nr, f, exponent
# ---------------------------------------------------------------------

# -----------------------------
# ROI definition
# -----------------------------
roi = {
    "frontal": ["Fz","F1","F2"],
    "central": ["Cz","C1","C2"],
    "parietal": ["Pz","P1","P2"],
    "occipital": ["Oz","O1","O2"],
}

roi_name = "occipital"
roi_ch = roi[roi_name]

# -----------------------------
# Settings
# -----------------------------
windows = ["pre_cue", "cue_target", "task"]
n_bins = 9

# -----------------------------
# Prepare dataframe
# -----------------------------
d = df.copy()

d = d[d["ch_name"].isin(roi_ch)].copy()

d["id"] = d["id"].astype("category")
d["group"] = d["group"].astype("category")

d["window"] = pd.Categorical(
    d["window"],
    categories=windows,
    ordered=True
)

for c in ["f","exponent"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

d = d.dropna(subset=["id","group","window","sequence_nr","f","exponent"])

# compute quadratic predictor
d["f2"] = d["f"]**2

# -----------------------------
# ROI aggregation
# average across channels
# -----------------------------
d_roi = (
    d.groupby(
        ["id","group","window","sequence_nr","f","f2"],
        as_index=False,
        observed=True
    )
    .agg(exponent=("exponent","mean"))
)

# -----------------------------
# Create bins
# -----------------------------
d_roi["f_bin"] = pd.qcut(
    d_roi["f"],
    q=n_bins,
    labels=False,
    duplicates="drop"
)

d_roi["f2_bin"] = pd.qcut(
    d_roi["f2"],
    q=n_bins,
    labels=False,
    duplicates="drop"
)

# -----------------------------
# Within-subject bin averages
# -----------------------------
roi_bin_f = (
    d_roi.groupby(
        ["id","group","window","f_bin"],
        as_index=False,
        observed=True
    )
    .agg(
        exponent=("exponent","mean"),
        f=("f","mean")
    )
)

roi_bin_f2 = (
    d_roi.groupby(
        ["id","group","window","f2_bin"],
        as_index=False,
        observed=True
    )
    .agg(
        exponent=("exponent","mean"),
        f2=("f2","mean")
    )
)

# -----------------------------
# Group-level averages
# -----------------------------
roi_plot_f = (
    roi_bin_f.groupby(
        ["group","window","f_bin"],
        as_index=False,
        observed=True
    )
    .agg(
        exponent=("exponent","mean"),
        sem=("exponent","sem"),
        f=("f","mean")
    )
)

roi_plot_f2 = (
    roi_bin_f2.groupby(
        ["group","window","f2_bin"],
        as_index=False,
        observed=True
    )
    .agg(
        exponent=("exponent","mean"),
        sem=("exponent","sem"),
        f2=("f2","mean")
    )
)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2,3, figsize=(14,8), sharey=True)

group_order = ["control","experimental"]

# ---- row 1 : f ----
for i, win in enumerate(windows):

    ax = axes[0,i]

    dw = roi_plot_f[roi_plot_f["window"] == win]

    for g in group_order:

        dg = dw[dw["group"] == g].sort_values("f")

        ax.errorbar(
            dg["f"],
            dg["exponent"],
            yerr=dg["sem"],
            marker="o",
            capsize=3,
            label=g
        )

    ax.set_title(win)
    ax.set_xlabel("Feedback (f)")
    ax.grid(True, alpha=0.3)

# ---- row 2 : f² ----
for i, win in enumerate(windows):

    ax = axes[1,i]

    dw = roi_plot_f2[roi_plot_f2["window"] == win]

    for g in group_order:

        dg = dw[dw["group"] == g].sort_values("f2")

        ax.errorbar(
            dg["f2"],
            dg["exponent"],
            yerr=dg["sem"],
            marker="o",
            capsize=3,
            label=g
        )

    ax.set_xlabel("Feedback² (f²)")
    ax.grid(True, alpha=0.3)

axes[0,0].set_ylabel("Aperiodic exponent")
axes[1,0].set_ylabel("Aperiodic exponent")

axes[0,0].legend(title="Group")

fig.suptitle(
    f"{roi_name.capitalize()} ROI — binned feedback curves",
    fontsize=16
)

plt.tight_layout()
plt.show()