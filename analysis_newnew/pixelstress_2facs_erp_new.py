# step2_erp_plots.py

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths / IO
# -----------------------------
path_out = Path("/mnt/data_dump/pixelstress/3_sequence_data_plus_fitted/")
index = pd.read_csv(path_out / "subject_index.csv")

# If you want to exclude subjects here too, do it by filtering index.

# -----------------------------
# Helpers
# -----------------------------
NAMES = ["Intercept", "f", "f2", "difficulty", "half"]

def load_betas_evoked(fif_path: str, names=NAMES):
    """Return dict name->Evoked from a FIF written with mne.write_evokeds."""
    evs = mne.read_evokeds(fif_path, condition=None, verbose=False)
    if len(evs) != len(names):
        raise RuntimeError(f"{fif_path}: expected {len(names)} evokeds, got {len(evs)}")
    return dict(zip(names, evs))

def stack_evokeds(evokeds):
    """Return (X, times, ch_names, info) where X is (n_subj, n_ch, n_time)."""
    data = np.stack([e.data for e in evokeds], axis=0)
    return data, evokeds[0].times, evokeds[0].ch_names, evokeds[0].info

def plot_beta_heatmap(beta_ch_time, times, ch_names, title, vlim=None):
    """
    beta_ch_time: (n_ch, n_time)
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(
        beta_ch_time,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, len(ch_names)],
        vmin=None if vlim is None else -vlim,
        vmax=None if vlim is None else vlim,
    )
    plt.colorbar(label="Beta amplitude")
    plt.yticks(np.arange(len(ch_names)) + 0.5, ch_names, fontsize=7)
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Load subject betas
# -----------------------------
betas_all = []  # list of dicts: {"id", "group", "betas": {name: Evoked}}
for row in index.itertuples(index=False):
    betas = load_betas_evoked(row.erp_betas_path)
    betas_all.append(dict(id=row.id, group=row.group, betas=betas))

# Split by group
betas_exp = [d for d in betas_all if d["group"] == "experimental"]
betas_ctl = [d for d in betas_all if d["group"] == "control"]

print(f"Loaded ERP betas: total n={len(betas_all)} | exp n={len(betas_exp)} | ctl n={len(betas_ctl)}")

# -----------------------------
# Grand-average maps (all subjects)
# -----------------------------
for pred in ["f", "f2"]:
    evs = [d["betas"][pred] for d in betas_all]
    X, times, ch_names, info = stack_evokeds(evs)          # (n_subj, n_ch, n_time)
    mean_map = X.mean(axis=0)                              # (n_ch, n_time)

    # symmetric color scaling for comparability
    vmax = np.nanpercentile(np.abs(mean_map), 99)

    plot_beta_heatmap(
        mean_map,
        times,
        ch_names,
        title=f"ERP β_{pred}: mean across subjects (n={len(evs)})",
        vlim=vmax,
    )

# -----------------------------
# Group main effect: Intercept difference (EXP - CTL)
# -----------------------------
if len(betas_exp) >= 2 and len(betas_ctl) >= 2:
    Xexp, times, ch_names, _ = stack_evokeds([d["betas"]["Intercept"] for d in betas_exp])
    Xctl, _, _, _ = stack_evokeds([d["betas"]["Intercept"] for d in betas_ctl])

    diff_map = Xexp.mean(axis=0) - Xctl.mean(axis=0)
    vmax = np.nanpercentile(np.abs(diff_map), 99)

    plot_beta_heatmap(
        diff_map,
        times,
        ch_names,
        title=f"ERP Intercept group difference (EXP − CTL)",
        vlim=vmax,
    )
else:
    print("Not enough subjects per group for intercept difference plot.")

# -----------------------------
# f2 × group interaction (difference of f2 betas: EXP - CTL)
# -----------------------------
if len(betas_exp) >= 2 and len(betas_ctl) >= 2:
    Xexp, times, ch_names, _ = stack_evokeds([d["betas"]["f2"] for d in betas_exp])
    Xctl, _, _, _ = stack_evokeds([d["betas"]["f2"] for d in betas_ctl])

    int_map = Xexp.mean(axis=0) - Xctl.mean(axis=0)
    vmax = np.nanpercentile(np.abs(int_map), 99)

    plot_beta_heatmap(
        int_map,
        times,
        ch_names,
        title=f"ERP f² × group interaction proxy (β_f² EXP − β_f² CTL)",
        vlim=vmax,
    )
else:
    print("Not enough subjects per group for f2×group plot.")
