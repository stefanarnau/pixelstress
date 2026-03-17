from pathlib import Path

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fooof import FOOOF


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
PATH_OUT = PATH_IN / "fooof_random_sequence_per_subject"
PATH_OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
FMIN_FIT = 1.0
FMAX_FIT = 40.0

FOOOF_KWARGS = dict(
    aperiodic_mode="fixed",
    peak_width_limits=(2, 12),
    max_n_peaks=8,
    min_peak_height=0.05,
    verbose=False,
)

# Reproducible random selection
RANDOM_SEED = 42

# Channel to display for each participant
CHANNEL_TO_PLOT = "FCz"   # e.g. "Pz", "Cz", "Oz", None for first channel

# Layout
N_COLS = 6
FIGSIZE_PER_COL = 4.2
FIGSIZE_PER_ROW = 3.2

# Plot options
PLOT_LOG10_POWER = True
SHOW_R2_ERROR = True
LINEWIDTH_PSD = 1.2
LINEWIDTH_FIT = 1.2


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_subject_files(path_in: Path):
    psd_files = sorted(path_in.glob("sub-*_seq_psd_channelwise_csd.npz"))
    subjects = []

    for psd_file in psd_files:
        subj_tag = psd_file.name.split("_seq_psd_channelwise_csd.npz")[0]
        index_file = path_in / f"{subj_tag}_seq_psd_channelwise_index_csd.csv"

        if not index_file.exists():
            print(f"Skipping {subj_tag}: missing index file")
            continue

        subjects.append(
            {
                "subj_tag": subj_tag,
                "psd_file": psd_file,
                "index_file": index_file,
            }
        )

    if len(subjects) == 0:
        raise RuntimeError("No subject PSD/index file pairs found.")

    return subjects


def load_subject_sequence_psd(psd_file: Path, index_file: Path):
    npz = np.load(psd_file, allow_pickle=True)
    psd = npz["psd"]               # shape: sequences x channels x freqs
    freqs = npz["freqs"]
    channels = npz["channels"]

    df_index = pd.read_csv(index_file)

    return psd, freqs, channels, df_index


def pick_random_sequence_row(df_index: pd.DataFrame, rng: random.Random):
    if len(df_index) == 0:
        raise ValueError("Index dataframe is empty.")
    return rng.randrange(len(df_index))


def get_channel_index(channels, channel_to_plot=None):
    channel_names = [str(ch) for ch in channels]

    if channel_to_plot is None:
        return 0

    if channel_to_plot not in channel_names:
        raise ValueError(
            f"Requested channel '{channel_to_plot}' not found. "
            f"Available example channels: {channel_names[:10]}"
        )

    return channel_names.index(channel_to_plot)


def fit_fooof_channel(freqs: np.ndarray, psd_1d: np.ndarray):
    fm = FOOOF(**FOOOF_KWARGS)
    fm.fit(freqs, psd_1d, [FMIN_FIT, FMAX_FIT])

    fit_mask = (freqs >= FMIN_FIT) & (freqs <= FMAX_FIT)
    freqs_fit = freqs[fit_mask]
    psd_fit = psd_1d[fit_mask]

    # Match your extraction logic exactly
    ap_params = fm.get_params("aperiodic_params")
    offset = float(ap_params[0])
    exponent = float(ap_params[1])
    r2 = float(fm.get_params("r_squared"))
    error = float(fm.get_params("error"))

    log_psd_fit = np.log10(psd_fit)
    aperiodic_fit = offset - exponent * np.log10(freqs_fit)

    # Full fitted model in the same log10 space used by your pipeline
    full_fit = fm.fooofed_spectrum_[fit_mask]

    return {
        "fm": fm,
        "freqs_fit": freqs_fit,
        "log_psd_fit": log_psd_fit,
        "ap_fit": aperiodic_fit,
        "full_fit": full_fit,
        "offset": offset,
        "exponent": exponent,
        "r2": r2,
        "error": error,
    }


# -----------------------------------------------------------------------------
# Load subject list
# -----------------------------------------------------------------------------
subjects = load_subject_files(PATH_IN)
rng = random.Random(RANDOM_SEED)

n_subjects = len(subjects)
n_cols = N_COLS
n_rows = math.ceil(n_subjects / n_cols)

fig_w = n_cols * FIGSIZE_PER_COL
fig_h = n_rows * FIGSIZE_PER_ROW

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
axes = axes.ravel()

selection_rows = []

legend_handles = None
legend_labels = None


# -----------------------------------------------------------------------------
# Plot one random sequence for each participant
# -----------------------------------------------------------------------------
for i, subj in enumerate(subjects):
    ax = axes[i]

    subj_tag = subj["subj_tag"]
    psd_file = subj["psd_file"]
    index_file = subj["index_file"]

    try:
        psd_all, freqs, channels, df_index = load_subject_sequence_psd(psd_file, index_file)

        if psd_all.shape[0] != len(df_index):
            raise ValueError(
                f"Sequence count mismatch: PSD has {psd_all.shape[0]} rows, "
                f"index has {len(df_index)} rows."
            )

        seq_row = pick_random_sequence_row(df_index, rng)
        meta = df_index.loc[seq_row].to_dict()

        ch_ix = get_channel_index(channels, CHANNEL_TO_PLOT)
        ch_name = str(channels[ch_ix])

        psd_1d = psd_all[seq_row, ch_ix, :]

        if np.any(~np.isfinite(psd_1d)) or np.any(psd_1d <= 0):
            raise ValueError("Invalid PSD values in selected sequence/channel.")

        res = fit_fooof_channel(freqs, psd_1d)
        freqs_fit = res["freqs_fit"]

        if PLOT_LOG10_POWER:
            y_psd = res["log_psd_fit"]
            y_ap = res["ap_fit"]
            y_fit = res["full_fit"]
            ylab = "log10 PSD"
        else:
            y_psd = 10 ** res["log_psd_fit"]
            y_ap = 10 ** res["ap_fit"]
            y_fit = 10 ** res["full_fit"]
            ylab = "PSD"

        ax.plot(freqs_fit, y_psd, linewidth=LINEWIDTH_PSD, label="PSD")
        ax.plot(freqs_fit, y_ap, "--", linewidth=LINEWIDTH_FIT, label="aperiodic")
        ax.plot(freqs_fit, y_fit, linewidth=LINEWIDTH_FIT, label="FOOOF fit")

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        title = (
            f"{subj_tag} | {ch_name}\n"
            f"b{int(meta.get('block_nr'))} s{int(meta.get('sequence_nr'))} | "
            f"n={int(meta.get('n_trials'))}"
        )
        if SHOW_R2_ERROR:
            title += (
                f"\nexp={res['exponent']:.2f}, "
                f"r²={res['r2']:.2f}, err={res['error']:.2f}"
            )

        ax.set_title(title, fontsize=8)
        ax.set_xlim(FMIN_FIT, FMAX_FIT)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.25)

        if i % n_cols == 0:
            ax.set_ylabel(ylab, fontsize=8)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Frequency (Hz)", fontsize=8)

        selection_rows.append(
            {
                "subj_tag": subj_tag,
                "subject_id": int(subj_tag.split("-")[1]),
                "block_nr": int(meta.get("block_nr")),
                "sequence_nr": int(meta.get("sequence_nr")),
                "group": meta.get("group"),
                "half": meta.get("half"),
                "n_trials": int(meta.get("n_trials")),
                "f": float(meta.get("f")),
                "channel": ch_name,
                "exponent": res["exponent"],
                "r2": res["r2"],
                "error": res["error"],
            }
        )

    except Exception as exc:
        ax.text(
            0.5,
            0.5,
            f"{subj_tag}\nFAILED\n{exc}",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.axis("off")


# Hide unused axes
for k in range(n_subjects, len(axes)):
    axes[k].axis("off")


# Shared legend
if legend_handles is not None:
    fig.legend(legend_handles, legend_labels, loc="upper right", fontsize=9)

fig.suptitle(
    f"One random sequence PSD + FOOOF fit per participant | channel={CHANNEL_TO_PLOT or 'first'}\n"
    f"fit range = {FMIN_FIT:.1f}-{FMAX_FIT:.1f} Hz | random_seed = {RANDOM_SEED}",
    fontsize=14,
    y=0.995,
)

plt.tight_layout(rect=[0, 0, 0.985, 0.965])

out_file = PATH_OUT / (
    f"all_subjects_random_sequence_fooof_"
    f"channel-{CHANNEL_TO_PLOT or 'first'}_"
    f"f{int(FMIN_FIT)}-{int(FMAX_FIT)}.png"
)
fig.savefig(out_file, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved figure:", out_file)


# -----------------------------------------------------------------------------
# Save selected sequences table
# -----------------------------------------------------------------------------
df_selected = pd.DataFrame(selection_rows).sort_values("subject_id").reset_index(drop=True)

csv_file = PATH_OUT / (
    f"all_subjects_random_sequence_selection_"
    f"channel-{CHANNEL_TO_PLOT or 'first'}_"
    f"f{int(FMIN_FIT)}-{int(FMAX_FIT)}.csv"
)
df_selected.to_csv(csv_file, index=False)

print("Saved selection table:", csv_file)
print("Subjects plotted:", len(df_selected))