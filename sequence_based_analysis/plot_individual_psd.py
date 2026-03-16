from pathlib import Path

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fooof import FOOOF

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
PATH_OUT = PATH_IN / "fooof_sequence_inspection"
PATH_OUT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
SUBJECT_ID = 32
BLOCK_NR = 4
SEQUENCE_NR = 6

FMIN_FIT = 1.0
FMAX_FIT = 30.0

FOOOF_KWARGS = dict(
    aperiodic_mode="fixed",
    peak_width_limits=(2, 12),
    max_n_peaks=8,
    min_peak_height=0.05,
    verbose=False,
)

# Layout
N_COLS = 8
FIGSIZE_PER_COL = 3.0
FIGSIZE_PER_ROW = 2.4

# Plot options
PLOT_LOG10_POWER = True
SHOW_R2_ERROR = True
LINEWIDTH_PSD = 1.2
LINEWIDTH_FIT = 1.2


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_subject_sequence_psd(path_in: Path, subject_id: int):
    subj_tag = f"sub-{subject_id:03d}"

    psd_file = path_in / f"{subj_tag}_seq_psd_channelwise_csd.npz"
    index_file = path_in / f"{subj_tag}_seq_psd_channelwise_index_csd.csv"

    if not psd_file.exists():
        raise FileNotFoundError(f"PSD file not found: {psd_file}")
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    npz = np.load(psd_file, allow_pickle=True)
    psd = npz["psd"]               # shape: sequences x channels x freqs
    freqs = npz["freqs"]
    channels = npz["channels"]

    df_index = pd.read_csv(index_file)

    return psd, freqs, channels, df_index


def find_sequence_row(df_index: pd.DataFrame, block_nr: int, sequence_nr: int):
    mask = (df_index["block_nr"] == block_nr) & (df_index["sequence_nr"] == sequence_nr)
    matches = df_index.index[mask].tolist()

    if len(matches) == 0:
        raise ValueError(
            f"No sequence found for block_nr={block_nr}, sequence_nr={sequence_nr}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple matches found for block_nr={block_nr}, sequence_nr={sequence_nr}."
        )

    return matches[0]


def fit_fooof_channel(freqs: np.ndarray, psd_1d: np.ndarray):
    fm = FOOOF(**FOOOF_KWARGS)
    fm.fit(freqs, psd_1d, [FMIN_FIT, FMAX_FIT])

    ap_params = fm.get_params("aperiodic_params")
    offset = float(ap_params[0])
    exponent = float(ap_params[1])
    r2 = float(fm.get_params("r_squared"))
    error = float(fm.get_params("error"))

    # Reconstruct aperiodic fit in log10 space
    fit_mask = (freqs >= FMIN_FIT) & (freqs <= FMAX_FIT)
    freqs_fit = freqs[fit_mask]
    log_psd_fit = np.log10(psd_1d[fit_mask])

    ap_fit = offset - exponent * np.log10(freqs_fit)
    full_fit = fm._ap_fit + fm._peak_fit

    return {
        "fm": fm,
        "freqs_fit": freqs_fit,
        "log_psd_fit": log_psd_fit,
        "ap_fit": ap_fit,
        "full_fit": full_fit,
        "offset": offset,
        "exponent": exponent,
        "r2": r2,
        "error": error,
    }


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
psd_all, freqs, channels, df_index = load_subject_sequence_psd(PATH_IN, SUBJECT_ID)

seq_row = find_sequence_row(df_index, BLOCK_NR, SEQUENCE_NR)
psd_seq = psd_all[seq_row]  # shape: channels x freqs

meta = df_index.loc[seq_row].to_dict()

n_channels = psd_seq.shape[0]
n_cols = N_COLS
n_rows = math.ceil(n_channels / n_cols)

fig_w = n_cols * FIGSIZE_PER_COL
fig_h = n_rows * FIGSIZE_PER_ROW

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
axes = axes.ravel()

# -----------------------------------------------------------------------------
# Plot all electrodes
# -----------------------------------------------------------------------------
for ch_ix in range(n_channels):
    ax = axes[ch_ix]

    ch_name = str(channels[ch_ix])
    psd_1d = psd_seq[ch_ix]

    # Skip invalid PSD
    if np.any(~np.isfinite(psd_1d)) or np.any(psd_1d <= 0):
        ax.text(0.5, 0.5, f"{ch_name}\ninvalid PSD", ha="center", va="center")
        ax.axis("off")
        continue

    try:
        res = fit_fooof_channel(freqs, psd_1d)
    except Exception as exc:
        ax.text(0.5, 0.5, f"{ch_name}\nFOOOF failed\n{exc}", ha="center", va="center")
        ax.axis("off")
        continue

    freqs_fit = res["freqs_fit"]

    if PLOT_LOG10_POWER:
        y_psd = res["log_psd_fit"]
        y_ap = res["ap_fit"]
        y_fit = res["full_fit"]
        ylab = "log10 PSD"
    else:
        # Convert fits back to linear space
        y_psd = 10 ** res["log_psd_fit"]
        y_ap = 10 ** res["ap_fit"]
        y_fit = 10 ** res["full_fit"]
        ylab = "PSD"

    ax.plot(freqs_fit, y_psd, linewidth=LINEWIDTH_PSD, label="PSD")
    ax.plot(freqs_fit, y_ap, "--", linewidth=LINEWIDTH_FIT, label="aperiodic")
    ax.plot(freqs_fit, y_fit, linewidth=LINEWIDTH_FIT, label="FOOOF fit")

    title = ch_name
    if SHOW_R2_ERROR:
        title += f"\nexp={res['exponent']:.2f}, r²={res['r2']:.2f}, err={res['error']:.2f}"
    ax.set_title(title, fontsize=8)

    ax.set_xlim(FMIN_FIT, FMAX_FIT)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.25)

    if ch_ix % n_cols == 0:
        ax.set_ylabel(ylab, fontsize=8)
    if ch_ix >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Frequency (Hz)", fontsize=8)

# Hide unused axes
for k in range(n_channels, len(axes)):
    axes[k].axis("off")

# Add one shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=9)

fig.suptitle(
    f"Subject {SUBJECT_ID:03d} | block {BLOCK_NR} | sequence {SEQUENCE_NR}\n"
    f"group={meta.get('group')} | half={meta.get('half')} | "
    f"n_trials={meta.get('n_trials')} | f={meta.get('f')}",
    fontsize=14,
    y=0.995,
)

plt.tight_layout(rect=[0, 0, 0.98, 0.97])

out_file = PATH_OUT / f"sub-{SUBJECT_ID:03d}_block-{BLOCK_NR}_seq-{SEQUENCE_NR}_fooof_all_electrodes.png"
fig.savefig(out_file, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved:", out_file)