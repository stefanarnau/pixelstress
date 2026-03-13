# -----------------------------------------------------------------------------
# Diagnose sequence loss from sequence-level FOOOF QC
# - loads the same CSD long file as the EEG analysis
# - evaluates row-level and sequence-level loss
# - identifies which criterion drives sequence loss
# - runs single-criterion sweeps and optional joint grid
# -----------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
path_in = Path("/mnt/data_dump/pixelstress/3_sequence_data/")
path_out = Path("/mnt/data_dump/pixelstress/diagnose_qc_loss/")
path_out.mkdir(parents=True, exist_ok=True)

file_in = path_in / "all_subjects_seq_fooof_rt_channelwise_long_csd.csv"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Baseline QC currently used in EEG analysis
baseline_qc = {
    "min_r2": 0.80,
    "max_error": 0.30,
    "min_exponent": 0.50,
    "max_exponent": 3.50,
}

# Sweeps: vary one threshold at a time while keeping others fixed at baseline
r2_values = np.round(np.arange(0.60, 0.96, 0.02), 2)
error_values = np.round(np.arange(0.15, 0.51, 0.02), 2)
exp_min_values = np.round(np.arange(0.00, 1.01, 0.10), 2)
exp_max_values = np.round(np.arange(2.50, 4.51, 0.10), 2)

# Optional smaller joint grid around your current values
run_joint_grid = False
joint_r2_values = [0.70, 0.75, 0.80, 0.85]
joint_error_values = [0.20, 0.25, 0.30, 0.35]
joint_exp_min_values = [0.30, 0.50, 0.70]
joint_exp_max_values = [3.0, 3.5, 4.0]

# Sequence identifier columns
seq_id_cols = ["id", "block_nr", "sequence_nr"]

# Plotting
dpi = 150


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def apply_qc_mask(df, min_r2, max_error, min_exponent, max_exponent):
    return (
        df["r2"].ge(min_r2)
        & df["error"].le(max_error)
        & df["exponent"].between(min_exponent, max_exponent)
    )


def summarize_qc(df, min_r2, max_error, min_exponent, max_exponent):
    """
    Returns one summary row for a given QC setting.
    """
    mask_r2 = df["r2"].ge(min_r2)
    mask_err = df["error"].le(max_error)
    mask_exp = df["exponent"].between(min_exponent, max_exponent)

    qc_mask = mask_r2 & mask_err & mask_exp
    df_qc = df.loc[qc_mask].copy()

    expected_n_electrodes = df["ch_name"].nunique()

    # Row-level
    n_rows_total = len(df)
    n_rows_kept = len(df_qc)
    n_rows_dropped = n_rows_total - n_rows_kept
    prop_rows_kept = n_rows_kept / n_rows_total if n_rows_total > 0 else np.nan

    # Sequence-level before/after QC
    seq_before = (
        df.groupby(seq_id_cols, observed=True)
        .agg(
            n_electrodes_before=("ch_name", "nunique"),
            n_rows_before=("ch_name", "size"),
        )
        .reset_index()
    )

    seq_after = (
        df_qc.groupby(seq_id_cols, observed=True)
        .agg(
            n_electrodes_after=("ch_name", "nunique"),
            n_rows_after=("ch_name", "size"),
        )
        .reset_index()
    )

    seq_merged = seq_before.merge(seq_after, on=seq_id_cols, how="left")
    seq_merged["n_electrodes_after"] = seq_merged["n_electrodes_after"].fillna(0).astype(int)
    seq_merged["n_rows_after"] = seq_merged["n_rows_after"].fillna(0).astype(int)

    seq_merged["is_complete_after_qc"] = seq_merged["n_electrodes_after"] == expected_n_electrodes
    seq_merged["is_lost_after_qc"] = ~seq_merged["is_complete_after_qc"]

    n_seq_total = len(seq_merged)
    n_seq_complete = int(seq_merged["is_complete_after_qc"].sum())
    n_seq_lost = n_seq_total - n_seq_complete
    prop_seq_complete = n_seq_complete / n_seq_total if n_seq_total > 0 else np.nan

    # Criterion-specific row failures
    n_fail_r2 = int((~mask_r2).sum())
    n_fail_err = int((~mask_err).sum())
    n_fail_exp = int((~mask_exp).sum())

    # Unique contribution at row level
    n_fail_only_r2 = int(((~mask_r2) & mask_err & mask_exp).sum())
    n_fail_only_err = int((mask_r2 & (~mask_err) & mask_exp).sum())
    n_fail_only_exp = int((mask_r2 & mask_err & (~mask_exp)).sum())

    # For each sequence, count whether at least one electrode failed each criterion
    seq_fail_r2 = (
        (~mask_r2)
        .groupby([df[c] for c in seq_id_cols], observed=True)
        .any()
        .reset_index(name="seq_any_fail_r2")
    )
    seq_fail_err = (
        (~mask_err)
        .groupby([df[c] for c in seq_id_cols], observed=True)
        .any()
        .reset_index(name="seq_any_fail_err")
    )
    seq_fail_exp = (
        (~mask_exp)
        .groupby([df[c] for c in seq_id_cols], observed=True)
        .any()
        .reset_index(name="seq_any_fail_exp")
    )

    seq_fail = seq_before.merge(seq_fail_r2, on=seq_id_cols, how="left")
    seq_fail = seq_fail.merge(seq_fail_err, on=seq_id_cols, how="left")
    seq_fail = seq_fail.merge(seq_fail_exp, on=seq_id_cols, how="left")

    for col in ["seq_any_fail_r2", "seq_any_fail_err", "seq_any_fail_exp"]:
        seq_fail[col] = seq_fail[col].fillna(False)

    n_seq_any_fail_r2 = int(seq_fail["seq_any_fail_r2"].sum())
    n_seq_any_fail_err = int(seq_fail["seq_any_fail_err"].sum())
    n_seq_any_fail_exp = int(seq_fail["seq_any_fail_exp"].sum())

    n_seq_fail_only_r2 = int(
        (seq_fail["seq_any_fail_r2"] & ~seq_fail["seq_any_fail_err"] & ~seq_fail["seq_any_fail_exp"]).sum()
    )
    n_seq_fail_only_err = int(
        (~seq_fail["seq_any_fail_r2"] & seq_fail["seq_any_fail_err"] & ~seq_fail["seq_any_fail_exp"]).sum()
    )
    n_seq_fail_only_exp = int(
        (~seq_fail["seq_any_fail_r2"] & ~seq_fail["seq_any_fail_err"] & seq_fail["seq_any_fail_exp"]).sum()
    )

    # How incomplete are lost sequences?
    mean_electrodes_after = float(seq_merged.loc[seq_merged["is_lost_after_qc"], "n_electrodes_after"].mean()) \
        if n_seq_lost > 0 else np.nan
    median_electrodes_after = float(seq_merged.loc[seq_merged["is_lost_after_qc"], "n_electrodes_after"].median()) \
        if n_seq_lost > 0 else np.nan

    return {
        "min_r2": min_r2,
        "max_error": max_error,
        "min_exponent": min_exponent,
        "max_exponent": max_exponent,
        "expected_n_electrodes": expected_n_electrodes,
        "n_rows_total": n_rows_total,
        "n_rows_kept": n_rows_kept,
        "n_rows_dropped": n_rows_dropped,
        "prop_rows_kept": prop_rows_kept,
        "n_seq_total": n_seq_total,
        "n_seq_complete": n_seq_complete,
        "n_seq_lost": n_seq_lost,
        "prop_seq_complete": prop_seq_complete,
        "n_fail_r2_rows": n_fail_r2,
        "n_fail_error_rows": n_fail_err,
        "n_fail_exponent_rows": n_fail_exp,
        "n_fail_only_r2_rows": n_fail_only_r2,
        "n_fail_only_error_rows": n_fail_only_err,
        "n_fail_only_exponent_rows": n_fail_only_exp,
        "n_seq_any_fail_r2": n_seq_any_fail_r2,
        "n_seq_any_fail_error": n_seq_any_fail_err,
        "n_seq_any_fail_exponent": n_seq_any_fail_exp,
        "n_seq_fail_only_r2": n_seq_fail_only_r2,
        "n_seq_fail_only_error": n_seq_fail_only_err,
        "n_seq_fail_only_exponent": n_seq_fail_only_exp,
        "mean_electrodes_after_in_lost_seq": mean_electrodes_after,
        "median_electrodes_after_in_lost_seq": median_electrodes_after,
    }


def run_single_sweep(df, vary_name, values, baseline):
    rows = []

    for val in values:
        pars = baseline.copy()
        pars[vary_name] = float(val)
        rows.append(summarize_qc(df=df, **pars))

    return pd.DataFrame(rows)


def plot_single_sweep(df_sweep, x_col, title, out_png):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(df_sweep[x_col], df_sweep["prop_rows_kept"], label="rows kept")
    ax1.plot(df_sweep[x_col], df_sweep["prop_seq_complete"], label="complete sequences")
    ax1.set_xlabel(x_col)
    ax1.set_ylabel("proportion kept")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_baseline_reason_table(df, baseline):
    """
    More detailed baseline diagnostics:
    for the baseline QC, identify for each lost sequence whether failure is due to r2, error, exponent, or overlaps.
    """
    min_r2 = baseline["min_r2"]
    max_error = baseline["max_error"]
    min_exponent = baseline["min_exponent"]
    max_exponent = baseline["max_exponent"]

    mask_r2 = df["r2"].ge(min_r2)
    mask_err = df["error"].le(max_error)
    mask_exp = df["exponent"].between(min_exponent, max_exponent)

    seq_before = (
        df.groupby(seq_id_cols, observed=True)
        .agg(
            group=("group", "first"),
            n_electrodes_before=("ch_name", "nunique"),
        )
        .reset_index()
    )

    seq_fail_r2 = (
        (~mask_r2)
        .groupby([df[c] for c in seq_id_cols], observed=True)
        .any()
        .reset_index(name="fail_r2")
    )
    seq_fail_err = (
        (~mask_err)
        .groupby([df[c] for c in seq_id_cols], observed=True)
        .any()
        .reset_index(name="fail_error")
    )
    seq_fail_exp = (
        (~mask_exp)
        .groupby([df[c] for c in seq_id_cols], observed=True)
        .any()
        .reset_index(name="fail_exponent")
    )

    df_qc = df.loc[mask_r2 & mask_err & mask_exp].copy()
    expected_n_electrodes = df["ch_name"].nunique()

    seq_after = (
        df_qc.groupby(seq_id_cols, observed=True)
        .agg(n_electrodes_after=("ch_name", "nunique"))
        .reset_index()
    )

    out = seq_before.merge(seq_fail_r2, on=seq_id_cols, how="left")
    out = out.merge(seq_fail_err, on=seq_id_cols, how="left")
    out = out.merge(seq_fail_exp, on=seq_id_cols, how="left")
    out = out.merge(seq_after, on=seq_id_cols, how="left")

    for col in ["fail_r2", "fail_error", "fail_exponent"]:
        out[col] = out[col].fillna(False)

    out["n_electrodes_after"] = out["n_electrodes_after"].fillna(0).astype(int)
    out["lost_sequence"] = out["n_electrodes_after"] < expected_n_electrodes

    def classify_reason(row):
        flags = []
        if row["fail_r2"]:
            flags.append("r2")
        if row["fail_error"]:
            flags.append("error")
        if row["fail_exponent"]:
            flags.append("exponent")
        return "+".join(flags) if flags else "none"

    out["failure_pattern"] = out.apply(classify_reason, axis=1)

    return out


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
    if "window" not in seq_id_cols:
        seq_id_cols = ["id", "window", "block_nr", "sequence_nr"]

df["group"] = df["group"].cat.set_categories(["control", "experimental"])

for col in ["r2", "error", "exponent", "f", "mean_rt", "mean_log_rt", "mean_trial_difficulty"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("Loaded rows:", len(df))
print("Subjects:", df["id"].nunique())
print("Sequences:", df[seq_id_cols].drop_duplicates().shape[0])
print("Electrodes:", df["ch_name"].nunique())


# -----------------------------------------------------------------------------
# Baseline summary
# -----------------------------------------------------------------------------
baseline_summary = pd.DataFrame([summarize_qc(df=df, **baseline_qc)])
baseline_summary.to_csv(path_out / "baseline_qc_summary.csv", index=False)

print("\nBaseline QC summary")
print(baseline_summary.to_string(index=False))

baseline_reason_table = make_baseline_reason_table(df, baseline_qc)
baseline_reason_table.to_csv(path_out / "baseline_qc_sequence_reason_table.csv", index=False)

baseline_reason_counts = (
    baseline_reason_table.loc[baseline_reason_table["lost_sequence"]]
    .groupby("failure_pattern", observed=True)
    .size()
    .rename("n_lost_sequences")
    .reset_index()
    .sort_values("n_lost_sequences", ascending=False)
)
baseline_reason_counts.to_csv(path_out / "baseline_qc_sequence_reason_counts.csv", index=False)

print("\nLost-sequence failure patterns at baseline QC")
print(baseline_reason_counts.to_string(index=False))


# -----------------------------------------------------------------------------
# Single-criterion sweeps
# -----------------------------------------------------------------------------
df_r2 = run_single_sweep(df, "min_r2", r2_values, baseline_qc)
df_r2.to_csv(path_out / "qc_sweep_r2.csv", index=False)

df_error = run_single_sweep(df, "max_error", error_values, baseline_qc)
df_error.to_csv(path_out / "qc_sweep_error.csv", index=False)

df_exp_min = run_single_sweep(df, "min_exponent", exp_min_values, baseline_qc)
df_exp_min.to_csv(path_out / "qc_sweep_exponent_min.csv", index=False)

df_exp_max = run_single_sweep(df, "max_exponent", exp_max_values, baseline_qc)
df_exp_max.to_csv(path_out / "qc_sweep_exponent_max.csv", index=False)

plot_single_sweep(
    df_r2,
    x_col="min_r2",
    title="QC sweep: minimum r2",
    out_png=path_out / "qc_sweep_r2.png",
)

plot_single_sweep(
    df_error,
    x_col="max_error",
    title="QC sweep: maximum error",
    out_png=path_out / "qc_sweep_error.png",
)

plot_single_sweep(
    df_exp_min,
    x_col="min_exponent",
    title="QC sweep: minimum exponent",
    out_png=path_out / "qc_sweep_exponent_min.png",
)

plot_single_sweep(
    df_exp_max,
    x_col="max_exponent",
    title="QC sweep: maximum exponent",
    out_png=path_out / "qc_sweep_exponent_max.png",
)


# -----------------------------------------------------------------------------
# Optional joint grid
# -----------------------------------------------------------------------------
if run_joint_grid:
    grid_rows = []

    for min_r2 in joint_r2_values:
        for max_error in joint_error_values:
            for min_exponent in joint_exp_min_values:
                for max_exponent in joint_exp_max_values:
                    grid_rows.append(
                        summarize_qc(
                            df=df,
                            min_r2=min_r2,
                            max_error=max_error,
                            min_exponent=min_exponent,
                            max_exponent=max_exponent,
                        )
                    )

    df_grid = pd.DataFrame(grid_rows)
    df_grid.to_csv(path_out / "qc_joint_grid_summary.csv", index=False)

    # show the least damaging settings first
    df_grid_sorted = df_grid.sort_values(
        ["prop_seq_complete", "prop_rows_kept"],
        ascending=[False, False],
    ).reset_index(drop=True)

    print("\nTop 20 least damaging joint QC settings")
    print(
        df_grid_sorted[
            [
                "min_r2",
                "max_error",
                "min_exponent",
                "max_exponent",
                "prop_rows_kept",
                "prop_seq_complete",
                "n_seq_lost",
            ]
        ].head(20).to_string(index=False)
    )


print("\nSaved outputs to:", path_out)