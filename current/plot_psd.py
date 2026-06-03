# -----------------------------------------------------------------------------
# Plot residualized raw PSD, flattened PSD, and FOOOF fits by feedback bin
# Residualized for mean_trial_difficulty_c and half.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import statsmodels.formula.api as smf


PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/psd_bin_plots_residualized/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
ROI_NAME = "posterior"
#ROI =  ["Fz", "FCz", "FC1", "FC2", "F1", "F2"]
#ROI = ["Cz", "CPz", "FCz", "C1", "C2"]
ROI = ["POz", "PO3", "PO4", "O1", "Oz", "O2"]
N_BINS = 7

FMIN = 1.0
FMAX = 40.0

GROUP_ORDER = ["control", "experimental"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_sequence_psd_long(path_in, roi):
    psd_rows = []
    index_files = sorted(path_in.glob("sub-*_seq_psd_channelwise_index_car.csv"))

    if not index_files:
        raise FileNotFoundError("No PSD index files found.")

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
        tmp["group"] = pd.Categorical(tmp["group"], categories=GROUP_ORDER)
        tmp["psd_roi"] = list(psd_roi)

        psd_rows.append(
            tmp[["id", "group", "block_nr", "sequence_nr", "f", "psd_roi"]]
        )

    if not psd_rows:
        raise RuntimeError("No PSD rows loaded.")

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
    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER)

    return out


def add_feedback_bins(df, edges):
    out = df.copy()
    out["f_bin"] = pd.cut(out["f"], bins=edges, include_lowest=True)
    out["f_mid"] = out["f_bin"].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
    return out


def residualize_vector(y, covars):
    """
    Residualize y against covariates and add grand mean back.
    """
    tmp = covars.copy()
    tmp["y"] = y

    fit = smf.ols("y ~ mean_trial_difficulty_c + half", data=tmp).fit()

    y_resid = fit.resid + tmp["y"].mean()

    return y_resid.to_numpy()


def residualize_psd_matrix(log_psd_matrix, covars):
    """
    Residualize log10 PSD frequency-by-frequency.
    Shape: rows x freqs.
    """
    out = np.empty_like(log_psd_matrix)

    for fi in range(log_psd_matrix.shape[1]):
        out[:, fi] = residualize_vector(log_psd_matrix[:, fi], covars)

    return out


def add_bottom_colorbar(fig, norm, cmap, f_min, f_max):
    cbar_ax = fig.add_axes([0.22, 0.04, 0.56, 0.035])

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Signed feedback bin midpoint")

    cbar.set_ticks([f_min, 0.0, f_max])
    cbar.set_ticklabels([f"{f_min:.2f}", "0", f"{f_max:.2f}"])

    return cbar


# -----------------------------------------------------------------------------
# Load metadata
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_IN)

df["id"] = df["id"].astype(str)
df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER)
df["half"] = pd.Categorical(df["half"])
df["ch_name"] = df["ch_name"].astype(str)

for col in ["f", "offset", "exponent", "mean_trial_difficulty"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

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

seq_meta["id"] = seq_meta["id"].astype(str)
seq_meta["block_nr"] = seq_meta["block_nr"].astype(int)
seq_meta["sequence_nr"] = seq_meta["sequence_nr"].astype(int)
seq_meta["group"] = pd.Categorical(seq_meta["group"], categories=GROUP_ORDER)
seq_meta["half"] = pd.Categorical(seq_meta["half"])

seq_meta["mean_trial_difficulty_c"] = (
    seq_meta["mean_trial_difficulty"] - seq_meta["mean_trial_difficulty"].mean()
)

edges = np.linspace(seq_meta["f"].min(), seq_meta["f"].max(), N_BINS + 1)

print("ROI:", ROI_NAME)
print("Channels:", ROI)
print("Feedback bin edges:", np.round(edges, 3))


# -----------------------------------------------------------------------------
# Load PSD and aperiodic data
# -----------------------------------------------------------------------------
df_psd, freqs = load_sequence_psd_long(PATH_IN, ROI)
df_aperiodic = build_roi_aperiodic_table(df, ROI)

keep_cols = ["id", "group", "block_nr", "sequence_nr", "f"]

df_psd = df_psd.merge(
    seq_meta[
        [
            "id",
            "group",
            "block_nr",
            "sequence_nr",
            "f",
            "half",
            "mean_trial_difficulty_c",
        ]
    ],
    on=keep_cols,
    how="inner",
)

df_aperiodic = df_aperiodic.merge(
    seq_meta[
        [
            "id",
            "group",
            "block_nr",
            "sequence_nr",
            "f",
            "half",
            "mean_trial_difficulty_c",
        ]
    ],
    on=keep_cols,
    how="inner",
)

df_psd = add_feedback_bins(df_psd, edges)
df_aperiodic = add_feedback_bins(df_aperiodic, edges)

p = df_psd.merge(
    df_aperiodic,
    on=[
        "id",
        "group",
        "block_nr",
        "sequence_nr",
        "f",
        "half",
        "mean_trial_difficulty_c",
        "f_bin",
        "f_mid",
    ],
    how="inner",
)

print("Merged rows:", len(p))


# -----------------------------------------------------------------------------
# Residualize PSD and aperiodic parameters
# -----------------------------------------------------------------------------
eps = np.finfo(float).tiny

psd_stack = np.stack(p["psd_roi"].values, axis=0)
log_psd_stack = np.log10(np.maximum(psd_stack, eps))

covars = p[["mean_trial_difficulty_c", "half"]].copy()

print("Residualizing log10 PSD frequency-by-frequency...")
log_psd_resid = residualize_psd_matrix(log_psd_stack, covars)

p["log_psd_resid"] = list(log_psd_resid)

p["offset_resid"] = residualize_vector(
    p["offset"].to_numpy(dtype=float),
    covars,
)

p["exponent_resid"] = residualize_vector(
    p["exponent"].to_numpy(dtype=float),
    covars,
)


# -----------------------------------------------------------------------------
# Compute per-bin spectra
# -----------------------------------------------------------------------------
freq_mask = (freqs >= FMIN) & (freqs <= FMAX)
freqs_plot = freqs[freq_mask]

rows = []

for (group_name, f_bin), dg in p.groupby(["group", "f_bin"], observed=True):
    if len(dg) == 0:
        continue

    f_mid = float((f_bin.left + f_bin.right) / 2)

    log_psd_resid_stack = np.stack(dg["log_psd_resid"].values, axis=0)
    log_psd_resid_mean = log_psd_resid_stack.mean(axis=0)

    offset_mean = dg["offset_resid"].mean()
    exponent_mean = dg["exponent_resid"].mean()

    fooof_fit_resid = offset_mean - exponent_mean * np.log10(freqs)
    flattened_resid = log_psd_resid_mean - fooof_fit_resid

    rows.append(
        {
            "group": group_name,
            "f_mid": f_mid,
            "n": len(dg),
            "log_psd_resid_mean": log_psd_resid_mean,
            "fooof_fit_resid": fooof_fit_resid,
            "flattened_resid": flattened_resid,
            "offset_resid_mean": offset_mean,
            "exponent_resid_mean": exponent_mean,
        }
    )

bin_df = pd.DataFrame(rows).sort_values(["group", "f_mid"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Shared plotting style
# -----------------------------------------------------------------------------
cmap = cm.Spectral

f_min = float(bin_df["f_mid"].min())
f_max = float(bin_df["f_mid"].max())

norm = mcolors.TwoSlopeNorm(
    vmin=f_min,
    vcenter=0.0,
    vmax=f_max,
)


# -----------------------------------------------------------------------------
# 1) Residualized log10 PSD by feedback bin
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, group_name in zip(axes, GROUP_ORDER):
    dg = bin_df[bin_df["group"] == group_name]

    for _, row in dg.iterrows():
        color = cmap(norm(row["f_mid"]))
        y = np.asarray(row["log_psd_resid_mean"])[freq_mask]

        ax.plot(freqs_plot, y, color=color, linewidth=2)

    ax.set_title(group_name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(FMIN, FMAX)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Residualized log10 PSD")

fig.suptitle(f"{ROI_NAME}: residualized log10 PSD by feedback bin", y=0.96)
add_bottom_colorbar(fig, norm, cmap, f_min, f_max)
plt.subplots_adjust(bottom=0.18, top=0.85, wspace=0.25)
fig.savefig(PATH_OUT / f"{ROI_NAME}_residualized_logpsd_by_feedback_bins.png", dpi=200, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------
# 2) Residualized flattened spectrum by feedback bin
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, group_name in zip(axes, GROUP_ORDER):
    dg = bin_df[bin_df["group"] == group_name]

    for _, row in dg.iterrows():
        color = cmap(norm(row["f_mid"]))
        y = np.asarray(row["flattened_resid"])[freq_mask]

        ax.plot(freqs_plot, y, color=color, linewidth=2)

    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title(group_name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(FMIN, FMAX)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Residualized flattened log10 power")

fig.suptitle(f"{ROI_NAME}: residualized flattened spectrum by feedback bin", y=0.96)
add_bottom_colorbar(fig, norm, cmap, f_min, f_max)
plt.subplots_adjust(bottom=0.18, top=0.85, wspace=0.25)
fig.savefig(PATH_OUT / f"{ROI_NAME}_residualized_flattened_spectrum_by_feedback_bins.png", dpi=200, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------
# 3) Residualized FOOOF / aperiodic fits by feedback bin
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, group_name in zip(axes, GROUP_ORDER):
    dg = bin_df[bin_df["group"] == group_name]

    for _, row in dg.iterrows():
        color = cmap(norm(row["f_mid"]))
        y = np.asarray(row["fooof_fit_resid"])[freq_mask]

        ax.plot(freqs_plot, y, color=color, linewidth=2)

    ax.set_title(group_name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(FMIN, FMAX)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Residualized aperiodic fit: log10 power")

fig.suptitle(f"{ROI_NAME}: residualized FOOOF fits by feedback bin", y=0.96)
add_bottom_colorbar(fig, norm, cmap, f_min, f_max)
plt.subplots_adjust(bottom=0.18, top=0.85, wspace=0.25)
fig.savefig(PATH_OUT / f"{ROI_NAME}_residualized_fooof_fits_by_feedback_bins.png", dpi=200, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------
# Save binned summary
# -----------------------------------------------------------------------------
summary = bin_df[
    [
        "group",
        "f_mid",
        "n",
        "offset_resid_mean",
        "exponent_resid_mean",
    ]
].copy()

summary.to_csv(
    PATH_OUT / f"{ROI_NAME}_residualized_feedback_bin_fooof_summary.csv",
    index=False,
)

print("Finished.")
print("Saved to:", PATH_OUT)