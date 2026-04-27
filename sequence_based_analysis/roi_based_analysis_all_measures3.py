# -----------------------------------------------------------------------------
# Slim ROI mixed model + feedback plots with:
# 1) observed/model feedback curves for multiple RT/FOOOF measures
# 2) raw PSD by feedback bin
# 3) flattened spectrum by feedback bin
# 4) exponent violin plot by feedback bin
#
# Notes:
# - MEASURES is a list of dependent variables for MLMs
# - For each measure:
#     * fit MLM
#     * save coefficient table rows into one combined dataframe
#     * create a feedback/model fit plot
# - PSD / flattened / exponent violin are made once only,
#   using PLOT_MEASURE only for titles / filenames
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
MEASURES = [
    "mean_rt",
    "theta_flat",
    "alpha_flat",
    "beta_flat",
    "offset",
    "exponent",
]

# Used only for shared plots that are not re-run for every modeled measure
PLOT_MEASURE = "exponent"

ROI = ["Cz"]
ROI_NAME = "central_roi"

N_BINS = 9

RAW_PSD_PLOT_FMIN = 0.0
RAW_PSD_PLOT_FMAX = 20.0

FLAT_PLOT_FMIN = 1.0
FLAT_PLOT_FMAX = 20.0

FORMULA = """
roi_val ~ group * f + group * f2
          + mean_trial_difficulty_c + half
"""

RE_FORMULAS = [
    "1 + f + f2",
    "1 + f",
    "1",
]

CORR_MEASURES = [
    "f",
    "f2",
    "mean_trial_difficulty",
    "mean_rt",
    "offset",
    "exponent",
    "theta_flat",
    "alpha_flat",
    "beta_flat",
]

SIGNIFICANT_RESULTS_EXCLUDE_INTERCEPT = True


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
# PSD loader
# -----------------------------------------------------------------------------
def load_sequence_psd_long(path_in, roi):
    psd_rows = []
    index_files = sorted(path_in.glob("sub-*_seq_psd_channelwise_index_car.csv"))

    if not index_files:
        return None, None

    freqs_ref = None

    for idx_file in index_files:
        subj_tag = idx_file.name.replace("_seq_psd_channelwise_index_car.csv", "")
        npz_file = path_in / f"{subj_tag}_seq_psd_channelwise_car.npz"

        if not npz_file.exists():
            continue

        idx_df = pd.read_csv(idx_file)
        npz = np.load(npz_file, allow_pickle=True)

        psd = npz["psd"]
        freqs = npz["freqs"]
        channels = npz["channels"].astype(str)

        if freqs_ref is None:
            freqs_ref = freqs.copy()
        elif not np.allclose(freqs_ref, freqs):
            raise ValueError(f"Frequency mismatch in {subj_tag}")

        roi_idx = [i for i, ch in enumerate(channels) if ch in roi]
        if len(roi_idx) == 0:
            continue

        if psd.shape[0] != len(idx_df):
            raise ValueError(
                f"Mismatch for {subj_tag}: PSD sequences={psd.shape[0]}, "
                f"index rows={len(idx_df)}"
            )

        psd_roi = psd[:, roi_idx, :].mean(axis=1)

        tmp = idx_df.copy()
        tmp["id"] = tmp["id"].astype(str)
        tmp["block_nr"] = tmp["block_nr"].astype(int)
        tmp["sequence_nr"] = tmp["sequence_nr"].astype(int)
        tmp["group"] = pd.Categorical(
            tmp["group"],
            categories=["control", "experimental"],
        )
        tmp["psd_roi"] = list(psd_roi)

        psd_rows.append(
            tmp[["id", "group", "block_nr", "sequence_nr", "f", "psd_roi"]]
        )

    if not psd_rows:
        return None, None

    return pd.concat(psd_rows, ignore_index=True), freqs_ref


def build_roi_aperiodic_table(df_long, roi):
    dfa = df_long[df_long["ch_name"].isin(roi)].copy()

    out = (
        dfa.groupby(
            ["id", "group", "block_nr", "sequence_nr", "f"],
            as_index=False,
        )[["offset", "exponent"]]
        .mean()
    )

    out["id"] = out["id"].astype(str)
    out["block_nr"] = out["block_nr"].astype(int)
    out["sequence_nr"] = out["sequence_nr"].astype(int)
    out["group"] = pd.Categorical(
        out["group"],
        categories=["control", "experimental"],
    )

    return out


# -----------------------------------------------------------------------------
# Correlation helper
# -----------------------------------------------------------------------------
def plot_correlation_matrix(
    df_in,
    measures,
    title,
    method="pearson",
    path_out=None,
    file_tag=None,
):
    cols = [c for c in measures if c in df_in.columns]
    if len(cols) < 2:
        print(f"Not enough measures available for correlation: {cols}")
        return None

    d = df_in[cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(how="all")

    if d.empty:
        print("No usable data for correlation matrix.")
        return None

    corr = d.corr(method=method)

    fig, ax = plt.subplots(figsize=(0.8 * len(cols) + 3, 0.8 * len(cols) + 3))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{method} r")

    ax.set_title(title)
    plt.tight_layout()

    if path_out is not None and file_tag is not None:
        fig.savefig(
            path_out / f"{file_tag}.png",
            dpi=150,
            bbox_inches="tight",
        )
        corr.to_csv(path_out / f"{file_tag}.csv")

    plt.show()

    return corr


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def make_feedback_bin_edges(df_model, n_bins):
    return np.linspace(df_model["f"].min(), df_model["f"].max(), n_bins + 1)


def add_bottom_colorbar(fig, axes, norm, cmap, label):
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.04])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(label)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    return cbar


# -----------------------------------------------------------------------------
# Plot helper: observed/model curves
# -----------------------------------------------------------------------------
def plot_feedback_curves(
    df_model,
    fit,
    outcome_name,
    n_bins=8,
    path_out=None,
    file_tag=None,
):
    d = df_model.copy()
    group_order = ["control", "experimental"]

    edges = make_feedback_bin_edges(d, n_bins)
    d["f_bin"] = pd.cut(d["f"], bins=edges, include_lowest=True)

    agg = (
        d.groupby(["group", "f_bin"], observed=True)
        .agg(
            mean_score=("roi_val", "mean"),
            sem_score=("roi_val", "sem"),
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

    fig, ax = plt.subplots(figsize=(8, 6))

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
    ax.set_xlabel("Signed feedback (f)")
    ax.set_ylabel(outcome_name)
    ax.set_title(f"{outcome_name}: observed bin means and model curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if path_out is not None:
        if file_tag is None:
            file_tag = outcome_name
        fig.savefig(
            path_out / f"{file_tag}_feedback_curves.png",
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# -----------------------------------------------------------------------------
# Plot helper: raw PSD by feedback bin
# -----------------------------------------------------------------------------
def plot_psd_by_feedback_bins(
    df_model,
    df_psd,
    freqs,
    roi_name,
    measure,
    n_bins=8,
    fmin_plot=0.0,
    fmax_plot=20.0,
    path_out=None,
    file_tag=None,
):
    group_order = ["control", "experimental"]
    edges = make_feedback_bin_edges(df_model, n_bins)

    p = df_psd.copy()
    p["f_bin"] = pd.cut(p["f"], bins=edges, include_lowest=True)

    psd_bin_rows = []
    for (group_name, f_bin), dg in p.groupby(["group", "f_bin"], observed=True):
        if len(dg) == 0:
            continue

        psd_stack = np.stack(dg["psd_roi"].values, axis=0)
        psd_mean = psd_stack.mean(axis=0)
        f_mid = (f_bin.left + f_bin.right) / 2

        psd_bin_rows.append(
            {
                "group": group_name,
                "f_mid": float(f_mid),
                "psd_mean": psd_mean,
            }
        )

    psd_bins = pd.DataFrame(psd_bin_rows)
    psd_bins = psd_bins.sort_values(["group", "f_mid"]).reset_index(drop=True)

    freq_mask = (freqs >= fmin_plot) & (freqs <= fmax_plot)
    freqs_plot = freqs[freq_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    norm = mcolors.Normalize(
        vmin=psd_bins["f_mid"].min(),
        vmax=psd_bins["f_mid"].max(),
    )
    cmap = cm.viridis

    for ax, group_name in zip(axes, group_order):
        dg = psd_bins[psd_bins["group"] == group_name]

        for _, row in dg.iterrows():
            color = cmap(norm(row["f_mid"]))
            psd_plot = np.asarray(row["psd_mean"])[freq_mask]
            ax.plot(freqs_plot, psd_plot, color=color, linewidth=2)

        ax.set_title(group_name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(fmin_plot, fmax_plot)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("PSD")

    add_bottom_colorbar(
        fig=fig,
        axes=axes,
        norm=norm,
        cmap=cmap,
        label="Signed feedback (f)",
    )

    fig.suptitle(f"{roi_name}_{measure}: ROI PSD by feedback bin", y=0.95)
    plt.subplots_adjust(bottom=0.18, top=0.85, wspace=0.25)

    if path_out is not None:
        if file_tag is None:
            file_tag = f"{roi_name}_{measure}"
        fig.savefig(
            path_out / f"{file_tag}_psd_by_feedback_0to20Hz.png",
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# -----------------------------------------------------------------------------
# Plot helper: flattened spectrum by feedback bin
# -----------------------------------------------------------------------------
def plot_flattened_psd_by_feedback_bins(
    df_model,
    df_psd,
    df_aperiodic,
    freqs,
    roi_name,
    measure,
    n_bins=8,
    fmin_plot=1.0,
    fmax_plot=20.0,
    path_out=None,
    file_tag=None,
):
    group_order = ["control", "experimental"]

    p = df_psd.merge(
        df_aperiodic,
        on=["id", "group", "block_nr", "sequence_nr", "f"],
        how="inner",
    ).copy()

    edges = make_feedback_bin_edges(df_model, n_bins)
    p["f_bin"] = pd.cut(p["f"], bins=edges, include_lowest=True)

    freq_mask = (freqs >= fmin_plot) & (freqs <= fmax_plot)
    freqs_plot = freqs[freq_mask]

    eps = np.finfo(float).tiny
    flat_specs = []

    for _, row in p.iterrows():
        psd_1d = np.asarray(row["psd_roi"], dtype=float)
        log_psd = np.log10(np.maximum(psd_1d, eps))
        aperiodic_fit = row["offset"] - row["exponent"] * np.log10(freqs)
        flat_specs.append(log_psd - aperiodic_fit)

    p["flat_roi"] = flat_specs

    flat_bin_rows = []
    for (group_name, f_bin), dg in p.groupby(["group", "f_bin"], observed=True):
        if len(dg) == 0:
            continue

        flat_stack = np.stack(dg["flat_roi"].values, axis=0)
        flat_mean = flat_stack.mean(axis=0)
        f_mid = (f_bin.left + f_bin.right) / 2

        flat_bin_rows.append(
            {
                "group": group_name,
                "f_mid": float(f_mid),
                "flat_mean": flat_mean,
            }
        )

    flat_bins = pd.DataFrame(flat_bin_rows)
    flat_bins = flat_bins.sort_values(["group", "f_mid"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    norm = mcolors.Normalize(
        vmin=flat_bins["f_mid"].min(),
        vmax=flat_bins["f_mid"].max(),
    )
    cmap = cm.viridis

    for ax, group_name in zip(axes, group_order):
        dg = flat_bins[flat_bins["group"] == group_name]

        for _, row in dg.iterrows():
            color = cmap(norm(row["f_mid"]))
            flat_plot = np.asarray(row["flat_mean"])[freq_mask]
            ax.plot(freqs_plot, flat_plot, color=color, linewidth=2)

        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(group_name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(fmin_plot, fmax_plot)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Flattened log10 power")

    add_bottom_colorbar(
        fig=fig,
        axes=axes,
        norm=norm,
        cmap=cmap,
        label="Signed feedback (f)",
    )

    fig.suptitle(f"{roi_name}_{measure}: ROI flattened spectrum by feedback bin", y=0.95)
    plt.subplots_adjust(bottom=0.18, top=0.85, wspace=0.25)

    if path_out is not None:
        if file_tag is None:
            file_tag = f"{roi_name}_{measure}"
        fig.savefig(
            path_out / f"{file_tag}_flattened_psd_by_feedback_1to20Hz.png",
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# -----------------------------------------------------------------------------
# Plot helper: exponent violin by feedback bin
# -----------------------------------------------------------------------------
def plot_exponent_violin_by_feedback_bins(
    df_model,
    df_aperiodic,
    roi_name,
    measure,
    n_bins=8,
    path_out=None,
    file_tag=None,
):
    group_order = ["control", "experimental"]
    group_colors = {
        "control": "#1f77b4",
        "experimental": "#d62728",
    }

    p = df_aperiodic.copy()
    edges = make_feedback_bin_edges(df_model, n_bins)
    p["f_bin"] = pd.cut(p["f"], bins=edges, include_lowest=True)

    bin_intervals = list(
        pd.IntervalIndex(
            pd.cut(df_model["f"], bins=edges, include_lowest=True).cat.categories
        )
    )
    bin_to_x = {iv: i for i, iv in enumerate(bin_intervals)}

    agg = (
        p.groupby(["group", "f_bin"], observed=True)
        .agg(
            exponent_mean=("exponent", "mean"),
            exponent_sem=("exponent", "sem"),
            n=("exponent", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(max(8, 1.15 * n_bins), 6))

    offsets = {
        "control": -0.18,
        "experimental": 0.18,
    }
    violin_width = 0.30

    for group_name in group_order:
        color = group_colors[group_name]
        dg_points = p[p["group"] == group_name].copy()

        for iv, d_bin in dg_points.groupby("f_bin", observed=True):
            if iv not in bin_to_x or len(d_bin) == 0:
                continue

            x_center = bin_to_x[iv] + offsets[group_name]
            y = d_bin["exponent"].dropna().to_numpy()

            if len(y) < 2:
                ax.scatter(
                    [x_center],
                    y,
                    s=20,
                    color=color,
                    alpha=0.6,
                    edgecolors="none",
                )
                continue

            parts = ax.violinplot(
                dataset=[y],
                positions=[x_center],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.35)

        dg_agg = agg[agg["group"] == group_name].copy()
        x_means = [bin_to_x[iv] + offsets[group_name] for iv in dg_agg["f_bin"]]

        ax.errorbar(
            x_means,
            dg_agg["exponent_mean"],
            yerr=dg_agg["exponent_sem"],
            fmt="o",
            color=color,
            capsize=3,
            linewidth=1.5,
            markersize=6,
            label=group_name,
        )

    tick_labels = [f"{(iv.left + iv.right) / 2:.2f}" for iv in bin_intervals]
    ax.set_xticks(range(len(bin_intervals)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Signed feedback bin midpoint")
    ax.set_ylabel("Exponent")
    ax.set_title(f"{roi_name}_{measure}: exponent by feedback bin")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if path_out is not None:
        if file_tag is None:
            file_tag = f"{roi_name}_{measure}"
        fig.savefig(
            path_out / f"{file_tag}_exponent_violin_by_feedback_bin.png",
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()


# -----------------------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_IN)

df["id"] = df["id"].astype(str)
df["group"] = pd.Categorical(df["group"], categories=["control", "experimental"])
df["half"] = pd.Categorical(df["half"])
df["ch_name"] = df["ch_name"].astype(str)

numeric_candidates = list(
    set(
        MEASURES
        + [PLOT_MEASURE]
        + CORR_MEASURES
        + [
            "f",
            "mean_trial_difficulty",
            "offset",
            "exponent",
        ]
    )
)

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

missing_measures = [m for m in MEASURES if m not in df.columns]
if missing_measures:
    raise ValueError(f"These measures are missing from the dataframe: {missing_measures}")

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

print("ROI:", ROI_NAME)
print("Electrodes:", ROI)
print("Measures to model:", MEASURES)
print("Subjects:", df["id"].nunique())


# -----------------------------------------------------------------------------
# Fit models for all measures
# -----------------------------------------------------------------------------
all_results = []
model_data_by_measure = {}
fit_by_measure = {}

for measure in MEASURES:
    print(f"\n--- Fitting measure: {measure} ---")

    df_roi = (
        df[df["ch_name"].isin(ROI)]
        .groupby(["id", "block_nr", "sequence_nr"], as_index=False)[measure]
        .mean()
        .rename(columns={measure: "roi_val"})
    )

    d = seq_meta.merge(df_roi, on=["id", "block_nr", "sequence_nr"], how="inner")

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

    print("Rows:", len(d))
    print("Subjects:", d["id"].nunique())

    fit, used_re, fit_error_log = fit_mixedlm_with_fallback(
        df_model=d,
        formula=FORMULA,
        re_formulas=RE_FORMULAS,
    )

    if fit is None:
        print("Model failed:")
        for msg in fit_error_log:
            print(msg)

        fail_row = pd.DataFrame(
            {
                "term": [np.nan],
                "beta": [np.nan],
                "se": [np.nan],
                "z": [np.nan],
                "p": [np.nan],
                "random_effects": [np.nan],
                "n_subjects": [d["id"].nunique()],
                "n_obs": [len(d)],
                "llf": [np.nan],
                "aic": [np.nan],
                "bic": [np.nan],
                "measure": [measure],
                "roi_name": [ROI_NAME],
                "model_failed": [True],
                "fit_error_log": [" | ".join(fit_error_log)],
            }
        )
        all_results.append(fail_row)
        continue

    print(fit.summary())
    print("Random-effects structure:", used_re)

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
            "measure": measure,
            "roi_name": ROI_NAME,
            "model_failed": False,
            "fit_error_log": "",
        }
    )
    all_results.append(df_res)

    model_data_by_measure[measure] = d
    fit_by_measure[measure] = fit

combined_results = pd.concat(all_results, ignore_index=True)
combined_results.to_csv(
    PATH_OUT / f"{ROI_NAME}_all_measures_mixedlm_results.csv",
    index=False,
)

combined_results_sig = combined_results[
    (combined_results["model_failed"] == False)
    & (combined_results["p"].notna())
    & (combined_results["p"] < 0.05)
    & (combined_results["term"] != "Intercept")
].copy()

combined_results_sig.to_csv(
    PATH_OUT / f"{ROI_NAME}_all_measures_mixedlm_results_significant_only.csv",
    index=False,
)


# -----------------------------------------------------------------------------
# Make feedback/model fit plot for each successfully fitted measure
# -----------------------------------------------------------------------------
for measure in MEASURES:
    if measure not in model_data_by_measure or measure not in fit_by_measure:
        continue

    plot_feedback_curves(
        df_model=model_data_by_measure[measure],
        fit=fit_by_measure[measure],
        outcome_name=f"{ROI_NAME}_{measure}",
        n_bins=N_BINS,
        path_out=PATH_OUT,
        file_tag=f"{ROI_NAME}_{measure}",
    )


# -----------------------------------------------------------------------------
# Reference model data for shared plots
# -----------------------------------------------------------------------------
if PLOT_MEASURE in model_data_by_measure:
    d_plot = model_data_by_measure[PLOT_MEASURE]
else:
    df_roi_plot = (
        df[df["ch_name"].isin(ROI)]
        .groupby(["id", "block_nr", "sequence_nr"], as_index=False)[PLOT_MEASURE]
        .mean()
        .rename(columns={PLOT_MEASURE: "roi_val"})
    )

    d_plot = seq_meta.merge(
        df_roi_plot,
        on=["id", "block_nr", "sequence_nr"],
        how="inner",
    )

    d_plot = d_plot.dropna(
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


# -----------------------------------------------------------------------------
# Load PSD / aperiodic data once
# -----------------------------------------------------------------------------
keep_cols = ["id", "block_nr", "sequence_nr", "group", "f"]
keep_df = d_plot[keep_cols].drop_duplicates()

df_psd, freqs = load_sequence_psd_long(PATH_IN, ROI)

if df_psd is not None:
    df_psd = df_psd.merge(
        keep_df,
        on=["id", "block_nr", "sequence_nr", "group", "f"],
        how="inner",
    )
    print("PSD rows after merge:", len(df_psd))
else:
    print("No PSD files found.")

if {"offset", "exponent"}.issubset(df.columns):
    df_aperiodic = build_roi_aperiodic_table(df, ROI)
    df_aperiodic = df_aperiodic.merge(
        keep_df,
        on=["id", "block_nr", "sequence_nr", "group", "f"],
        how="inner",
    )
    print("Aperiodic rows after merge:", len(df_aperiodic))
else:
    df_aperiodic = None
    print("No aperiodic columns found.")


# -----------------------------------------------------------------------------
# Shared plots, made once using PLOT_MEASURE as label
# -----------------------------------------------------------------------------
if df_psd is not None and freqs is not None:
    plot_psd_by_feedback_bins(
        df_model=d_plot,
        df_psd=df_psd,
        freqs=freqs,
        roi_name=ROI_NAME,
        measure=PLOT_MEASURE,
        n_bins=N_BINS,
        fmin_plot=RAW_PSD_PLOT_FMIN,
        fmax_plot=RAW_PSD_PLOT_FMAX,
        path_out=PATH_OUT,
        file_tag=f"{ROI_NAME}_{PLOT_MEASURE}",
    )

if df_psd is not None and freqs is not None and df_aperiodic is not None:
    plot_flattened_psd_by_feedback_bins(
        df_model=d_plot,
        df_psd=df_psd,
        df_aperiodic=df_aperiodic,
        freqs=freqs,
        roi_name=ROI_NAME,
        measure=PLOT_MEASURE,
        n_bins=N_BINS,
        fmin_plot=FLAT_PLOT_FMIN,
        fmax_plot=FLAT_PLOT_FMAX,
        path_out=PATH_OUT,
        file_tag=f"{ROI_NAME}_{PLOT_MEASURE}",
    )

if df_aperiodic is not None:
    plot_exponent_violin_by_feedback_bins(
        df_model=d_plot,
        df_aperiodic=df_aperiodic,
        roi_name=ROI_NAME,
        measure=PLOT_MEASURE,
        n_bins=N_BINS,
        path_out=PATH_OUT,
        file_tag=f"{ROI_NAME}_{PLOT_MEASURE}",
    )


# -----------------------------------------------------------------------------
# Correlation matrix
# -----------------------------------------------------------------------------
corr_seq = plot_correlation_matrix(
    df_in=df,
    measures=CORR_MEASURES,
    title="Sequence-level correlation matrix",
    method="pearson",
    path_out=PATH_OUT,
    file_tag="correlation_matrix_sequence_level",
)