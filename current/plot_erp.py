# -----------------------------------------------------------------------------
# Plot ROI ERP / slow wave by feedback bin
# Uses same f-binning logic as topo / PSD scripts.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/erp_bin_plots/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
ROI_NAME = "central"
ROI = ["C1", "Cz", "C2", "FC1", "FC2", "FCz", "CPz", "Fz", "CP1", "CP3", "FC3"]

N_BINS = 9

TMIN = -1.4
TMAX = 0.0

Y_LIM = None          # example: (-5, 5), or None
INVERT_Y = True      # conventional ERP plotting: negativity upward

GROUP_ORDER = ["control", "experimental"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_sequence_erp_long(path_in, roi):
    erp_rows = []
    index_files = sorted(path_in.glob("sub-*_seq_erp_channelwise_index_car.csv"))

    if not index_files:
        raise FileNotFoundError("No ERP index files found.")

    times_ref = None

    for idx_file in index_files:
        subj_tag = idx_file.name.replace("_seq_erp_channelwise_index_car.csv", "")
        npz_file = path_in / f"{subj_tag}_seq_erp_channelwise_car.npz"

        if not npz_file.exists():
            continue

        idx_df = pd.read_csv(idx_file)
        npz = np.load(npz_file, allow_pickle=True)

        erp = npz["erp"]
        times = npz["times"]
        channels = npz["channels"].astype(str)

        if times_ref is None:
            times_ref = times.copy()
        elif not np.allclose(times_ref, times):
            raise ValueError(f"ERP time-vector mismatch in {subj_tag}")

        roi_idx = [i for i, ch in enumerate(channels) if ch in roi]

        if len(roi_idx) == 0:
            continue

        if erp.shape[0] != len(idx_df):
            raise ValueError(
                f"Mismatch for {subj_tag}: ERP sequences={erp.shape[0]}, "
                f"index rows={len(idx_df)}"
            )

        erp_roi = erp[:, roi_idx, :].mean(axis=1)

        tmp = idx_df.copy()
        tmp["id"] = tmp["id"].astype(str)
        tmp["block_nr"] = tmp["block_nr"].astype(int)
        tmp["sequence_nr"] = tmp["sequence_nr"].astype(int)
        tmp["group"] = pd.Categorical(tmp["group"], categories=GROUP_ORDER)
        tmp["erp_roi"] = list(erp_roi)

        erp_rows.append(
            tmp[["id", "group", "block_nr", "sequence_nr", "f", "erp_roi"]]
        )

    if not erp_rows:
        raise RuntimeError("No ERP rows loaded.")

    return pd.concat(erp_rows, ignore_index=True), times_ref


def add_feedback_bins(df, edges):
    out = df.copy()
    out["f_bin"] = pd.cut(out["f"], bins=edges, include_lowest=True)
    out["f_mid"] = out["f_bin"].apply(lambda iv: (iv.left + iv.right) / 2).astype(float)
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
df["ch_name"] = df["ch_name"].astype(str)
df["f"] = pd.to_numeric(df["f"], errors="coerce")

seq_meta = (
    df[
        [
            "id",
            "group",
            "block_nr",
            "sequence_nr",
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

edges = np.linspace(seq_meta["f"].min(), seq_meta["f"].max(), N_BINS + 1)

print("ROI:", ROI_NAME)
print("Channels:", ROI)
print("Feedback bin edges:", np.round(edges, 3))


# -----------------------------------------------------------------------------
# Load ERP data
# -----------------------------------------------------------------------------
df_erp, times = load_sequence_erp_long(PATH_IN, ROI)

keep_cols = ["id", "group", "block_nr", "sequence_nr", "f"]

df_erp = df_erp.merge(
    seq_meta[keep_cols],
    on=keep_cols,
    how="inner",
)

df_erp = add_feedback_bins(df_erp, edges)

print("ERP rows:", len(df_erp))


# -----------------------------------------------------------------------------
# Compute per-bin ERP waveforms
# -----------------------------------------------------------------------------
time_mask = (times >= TMIN) & (times <= TMAX)
times_plot = times[time_mask]

rows = []

for (group_name, f_bin), dg in df_erp.groupby(["group", "f_bin"], observed=True):
    if len(dg) == 0:
        continue

    f_mid = float((f_bin.left + f_bin.right) / 2)

    erp_stack = np.stack(dg["erp_roi"].values, axis=0)
    erp_mean = erp_stack.mean(axis=0)
    erp_sem = erp_stack.std(axis=0, ddof=1) / np.sqrt(erp_stack.shape[0])

    rows.append(
        {
            "group": group_name,
            "f_mid": f_mid,
            "n": len(dg),
            "erp_mean": erp_mean,
            "erp_sem": erp_sem,
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
# Plot ERP / slow wave by feedback bin
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, group_name in zip(axes, GROUP_ORDER):
    dg = bin_df[bin_df["group"] == group_name]

    for _, row in dg.iterrows():
        color = cmap(norm(row["f_mid"]))
        y = np.asarray(row["erp_mean"])[time_mask]

        ax.plot(
            times_plot,
            y,
            color=color,
            linewidth=2,
            label=f"f={row['f_mid']:.2f}",
        )

    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)

    # Optional shaded late slow-wave / pre-feedback window
    ax.axvspan(-0.3, 0.0, color="grey", alpha=0.12, linewidth=0)

    ax.set_title(group_name)
    ax.set_xlabel("Time relative to feedback / sequence end (s)")
    ax.set_xlim(TMIN, TMAX)

    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)

    if INVERT_Y:
        ax.invert_yaxis()

    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("ERP / slow-wave amplitude")

fig.suptitle(f"{ROI_NAME}: ROI ERP / slow wave by feedback bin", y=0.96)
add_bottom_colorbar(fig, norm, cmap, f_min, f_max)
plt.subplots_adjust(bottom=0.18, top=0.85, wspace=0.25)

fig.savefig(
    PATH_OUT / f"{ROI_NAME}_erp_slowwave_by_feedback_bins.png",
    dpi=200,
    bbox_inches="tight",
)

plt.show()


# -----------------------------------------------------------------------------
# Save binned summary
# -----------------------------------------------------------------------------
summary = bin_df[["group", "f_mid", "n"]].copy()

summary.to_csv(
    PATH_OUT / f"{ROI_NAME}_feedback_bin_erp_summary.csv",
    index=False,
)

print("Finished.")
print("Saved to:", PATH_OUT)