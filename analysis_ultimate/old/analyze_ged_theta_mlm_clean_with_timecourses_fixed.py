from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# Settings
PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned2/")
PATH_COMPONENTS = Path("/mnt/data_dump/pixelstress/3_trial_data/ged_theta/components")
PATH_OUT = Path("/mnt/data_dump/pixelstress/3_trial_data/ged_theta/mlm_results")
PATH_OUT.mkdir(parents=True, exist_ok=True)

IDS_TO_DROP = {1, 2, 3, 4, 5, 6, 13, 17, 25, 40, 49, 83}

TMIN, TMAX = 0.20, 0.30
POWER_MODE = "baseline_db"  # "absolute_log" or "baseline_db"
BASELINE_TMIN, BASELINE_TMAX = -2.000, -1.400
CORRECT_ONLY = False
INCLUDE_HALF = False

N_FEEDBACK_BINS = 9
N_TIMECOURSE_BINS = 9
SHOW_FIGURE = True
FIGURE_DPI = 180


# Load metadata and GED-theta values
frames = []
theta_power_all = []

for dataset in sorted(PATH_IN.glob("*erp.set")):
    trialinfo_file = Path(str(dataset).split("_cleaned")[0] + "_erp_trialinfo.csv")
    subject_df = pd.read_csv(trialinfo_file).reset_index(drop=True)
    subject_id = int(subject_df["id"].iloc[0])

    if subject_id in IDS_TO_DROP:
        continue

    component_file = PATH_COMPONENTS / f"sub-{subject_id:03d}_theta_component.npz"
    if not component_file.exists():
        component_file = PATH_COMPONENTS / f"sub-{subject_id}_theta_component.npz"

    with np.load(component_file) as component:
        theta_power = component["theta_power"]
        times = component["times"]

    analysis_mask = (times >= TMIN) & (times <= TMAX)
    baseline_mask = (times >= BASELINE_TMIN) & (times <= BASELINE_TMAX)
    
    log_power = 10 * np.log10(theta_power)
    
    analysis_log_power = np.nanmean(
        log_power[:, analysis_mask],
        axis=1,
    )
    
    if POWER_MODE == "absolute_log":
        eeg_value = analysis_log_power
    
    elif POWER_MODE == "baseline_db":
        baseline_log_power = np.nanmean(
            log_power[:, baseline_mask],
            axis=1,
        )
        eeg_value = analysis_log_power - baseline_log_power
    
    else:
        raise ValueError("POWER_MODE must be 'absolute_log' or 'baseline_db'")

    subject_df["eeg_value"] = eeg_value

    subject_df = subject_df.rename(columns={
        "session_condition": "group",
        "trial_nr": "trial_nr_sequence",
        "trial_nr_total": "trial_nr_global",
        "last_feedback_scaled": "feedback",
    })

    subject_df["group"] = subject_df["group"].replace({
        1: "experimental",
        2: "control",
    })
    subject_df["experimental"] = (
        subject_df["group"] == "experimental"
    ).astype(int)
    subject_df["accuracy"] = (
        pd.to_numeric(subject_df["accuracy"], errors="coerce") == 1
    ).astype(int)
    subject_df["feedback"] = pd.to_numeric(
        subject_df["feedback"],
        errors="coerce",
    )
    subject_df["sequence_difficulty"] = pd.to_numeric(
        subject_df["sequence_difficulty"],
        errors="coerce",
    )
    subject_df["half"] = (
        pd.to_numeric(subject_df["block_nr"], errors="coerce") > 4
    ).astype(int)
    subject_df["sequence_uid"] = (
        subject_df["id"].astype(str)
        + "_"
        + subject_df["block_nr"].astype(str)
        + "_"
        + subject_df["sequence_nr"].astype(str)
    )

    keep_subject = subject_df["sequence_nr"].to_numpy() > 1
    frames.append(subject_df.loc[keep_subject])
    theta_power_all.append(theta_power[keep_subject])


df = pd.concat(frames, ignore_index=True)
theta_power_all = np.concatenate(theta_power_all, axis=0)

df["feedback_c"] = df["feedback"] - df["feedback"].mean()
df["feedback2_c"] = df["feedback_c"] ** 2
df["sequence_difficulty_c"] = (
    df["sequence_difficulty"] - df["sequence_difficulty"].mean()
)

if INCLUDE_HALF:
    df["half_c"] = df["half"] - df["half"].mean()

keep = np.ones(len(df), dtype=bool)

if CORRECT_ONLY:
    keep &= df["accuracy"].to_numpy() == 1

model_columns = [
    "eeg_value",
    "id",
    "experimental",
    "feedback_c",
    "feedback2_c",
    "sequence_difficulty_c",
]
if INCLUDE_HALF:
    model_columns.append("half_c")

keep &= df[model_columns].notna().all(axis=1).to_numpy()

df = df.loc[keep].reset_index(drop=True)
theta_power_all = theta_power_all[keep]


# Mixed model
fixed_terms = [
    "experimental",
    "feedback_c",
    "feedback2_c",
    "experimental:feedback_c",
    "experimental:feedback2_c",
    "sequence_difficulty_c",
]
if INCLUDE_HALF:
    fixed_terms.append("half_c")

formula = "eeg_value ~ " + " + ".join(fixed_terms)

model = smf.mixedlm(
    formula=formula,
    data=df,
    groups=df["id"],
    re_formula="1 + feedback_c",
)

result = model.fit(
    reml=False,
    method="bfgs",
    maxiter=2000,
)

print("\nLoaded data")
print("-----------")
print(f"Participants: {df['id'].nunique()}")
print(f"Trials:       {len(df)}")
print(df["group"].value_counts())
print(f"\nFormula: {formula}")
print(result.summary())


# Save model outputs
window_label = (
    f"{TMIN:+.3f}_{TMAX:+.3f}"
    .replace("+", "p")
    .replace("-", "m")
    .replace(".", "p")
)
trial_label = "correct" if CORRECT_ONLY else "all"
stem = (
    f"theta_ged_{window_label}_{trial_label}_{POWER_MODE}_"
    "participant_feedback_slope_RE"
)

summary_file = PATH_OUT / f"{stem}_model_summary.txt"
coefficients_file = PATH_OUT / f"{stem}_coefficients.csv"
trial_values_file = PATH_OUT / f"{stem}_trial_values.csv"
participant_bins_file = PATH_OUT / f"{stem}_participant_bin_means.csv"
group_bins_file = PATH_OUT / f"{stem}_group_bin_summary.csv"
plot_file = PATH_OUT / f"{stem}_feedback_bins.png"
timecourse_file = PATH_OUT / f"{stem}_feedback_bin_timecourses.png"
timecourse_summary_file = PATH_OUT / f"{stem}_feedback_bin_timecourses.csv"

analysis_definition = f"""Analysis definition
-------------------
Measure:           GED theta power
Time window:       {TMIN:.3f} to {TMAX:.3f} s
Power mode:        {POWER_MODE}
Correct only:      {CORRECT_ONLY}
Trials:            {len(df)}
Participants:      {df['id'].nunique()}
Sequences:         {df['sequence_uid'].nunique()}
Formula:           {formula}
Random effects:    participant intercept + feedback slope

"""

summary_file.write_text(
    analysis_definition + result.summary().as_text(),
    encoding="utf-8",
)

fixed_effect_names = result.fe_params.index
ci = result.conf_int().loc[fixed_effect_names]

coefficients = pd.DataFrame({
    "term": fixed_effect_names,
    "estimate": result.fe_params,
    "standard_error": result.bse_fe,
    "z_value": result.tvalues.loc[fixed_effect_names],
    "p_value": result.pvalues.loc[fixed_effect_names],
    "ci_low": ci[0],
    "ci_high": ci[1],
})
coefficients.to_csv(coefficients_file, index=False)

trial_columns = [
    "id",
    "group",
    "experimental",
    "block_nr",
    "sequence_nr",
    "sequence_uid",
    "trial_nr_sequence",
    "trial_nr_global",
    "feedback",
    "feedback_c",
    "feedback2_c",
    "sequence_difficulty",
    "sequence_difficulty_c",
    "accuracy",
    "eeg_value",
]
if INCLUDE_HALF:
    trial_columns.extend(["half", "half_c"])

df[trial_columns].to_csv(trial_values_file, index=False)


# Feedback-bin plot
edges = np.linspace(
    df["feedback"].min(),
    df["feedback"].max(),
    N_FEEDBACK_BINS + 1,
)
df["feedback_bin"] = pd.cut(
    df["feedback"],
    bins=edges,
    include_lowest=True,
)

participant_bins = (
    df.groupby(
        ["id", "experimental", "feedback_bin"],
        observed=True,
    )
    .agg(
        eeg_value=("eeg_value", "mean"),
        mean_feedback=("feedback", "mean"),
        n_trials=("eeg_value", "size"),
    )
    .reset_index()
)

group_bins = (
    participant_bins.groupby(
        ["experimental", "feedback_bin"],
        observed=True,
    )
    .agg(
        mean=("eeg_value", "mean"),
        sd=("eeg_value", "std"),
        n_participants=("eeg_value", "count"),
        mean_feedback=("mean_feedback", "mean"),
    )
    .reset_index()
)
group_bins["sem"] = (
    group_bins["sd"] / np.sqrt(group_bins["n_participants"])
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, group_code, group_label in zip(
    axes,
    [0, 1],
    ["Control", "Experimental"],
):
    plot_data = group_bins.loc[
        group_bins["experimental"] == group_code
    ]
    x = np.arange(len(plot_data))

    ax.errorbar(
        x,
        plot_data["mean"],
        yerr=plot_data["sem"],
        marker="o",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{interval.left:.2f}\nto\n{interval.right:.2f}"
        for interval in plot_data["feedback_bin"]
    ])
    ax.set_title(group_label)
    ax.set_xlabel("Linear feedback bin")
    ax.axhline(0, linewidth=0.8)

ylabel = (
    "GED theta power (dB, absolute log)"
    if POWER_MODE == "absolute_log"
    else "GED theta power (dB relative to baseline)"
)
axes[0].set_ylabel(ylabel)

fig.suptitle(
    f"GED theta power by linear feedback bin\n"
    f"{TMIN:.3f} to {TMAX:.3f} s"
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(plot_file, dpi=FIGURE_DPI, bbox_inches="tight")

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)


participant_bins.to_csv(participant_bins_file, index=False)
group_bins.to_csv(group_bins_file, index=False)


# Feedback-bin time courses
log_power_all = 10 * np.log10(theta_power_all)

if POWER_MODE == "absolute_log":
    theta_timecourses = log_power_all

else:
    baseline_mask = (times >= BASELINE_TMIN) & (times <= BASELINE_TMAX)
    baseline_log_power = np.nanmean(
        log_power_all[:, baseline_mask],
        axis=1,
    )
    theta_timecourses = (
        log_power_all
        - baseline_log_power[:, None]
    )
    
timecourse_edges = np.linspace(
    df["feedback"].min(),
    df["feedback"].max(),
    N_TIMECOURSE_BINS + 1,
)
timecourse_bins = pd.cut(
    df["feedback"],
    bins=timecourse_edges,
    include_lowest=True,
)

participant_timecourses = []

for subject_id in sorted(df["id"].unique()):
    subject_mask = df["id"].to_numpy() == subject_id
    subject_group = int(df.loc[subject_mask, "experimental"].iloc[0])

    for feedback_bin in timecourse_bins.cat.categories:
        bin_mask = (
            subject_mask
            & (timecourse_bins == feedback_bin).to_numpy()
        )

        if not np.any(bin_mask):
            continue

        mean_timecourse = np.nanmean(
            theta_timecourses[bin_mask],
            axis=0,
        )

        participant_timecourses.append(pd.DataFrame({
            "id": subject_id,
            "experimental": subject_group,
            "feedback_bin": str(feedback_bin),
            "feedback_midpoint": feedback_bin.mid,
            "time": times,
            "theta_power": mean_timecourse,
        }))

participant_timecourses = pd.concat(
    participant_timecourses,
    ignore_index=True,
)

timecourse_summary = (
    participant_timecourses.groupby(
        [
            "experimental",
            "feedback_bin",
            "feedback_midpoint",
            "time",
        ],
        observed=True,
    )
    .agg(
        mean=("theta_power", "mean"),
        sd=("theta_power", "std"),
        n_participants=("theta_power", "count"),
    )
    .reset_index()
)
timecourse_summary["sem"] = (
    timecourse_summary["sd"]
    / np.sqrt(timecourse_summary["n_participants"])
)
timecourse_summary.to_csv(timecourse_summary_file, index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

for ax, group_code, group_label in zip(
    axes,
    [0, 1],
    ["Control", "Experimental"],
):
    group_data = timecourse_summary.loc[
        timecourse_summary["experimental"] == group_code
    ]

    for feedback_bin in sorted(
        group_data["feedback_bin"].unique(),
        key=lambda label: float(
            label.strip("()[]").split(",")[0]
        ),
    ):
        line_data = group_data.loc[
            group_data["feedback_bin"] == feedback_bin
        ]

        ax.plot(
            line_data["time"],
            line_data["mean"],
            label=feedback_bin,
        )

    ax.axvline(0, linewidth=0.8)
    ax.axvspan(TMIN, TMAX, alpha=0.12)
    ax.axhline(0, linewidth=0.8)
    ax.set_title(group_label)
    ax.set_xlabel("Time relative to target (s)")

axes[0].set_ylabel(ylabel)
axes[1].legend(
    title="Feedback bin",
    fontsize=8,
    title_fontsize=8,
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

fig.suptitle("GED theta time courses by feedback bin")
fig.tight_layout(rect=(0, 0, 0.86, 0.94))
fig.savefig(timecourse_file, dpi=FIGURE_DPI, bbox_inches="tight")

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)

print("\nSaved:")
for file in [
    summary_file,
    coefficients_file,
    trial_values_file,
    participant_bins_file,
    group_bins_file,
    plot_file,
    timecourse_summary_file,
    timecourse_file,
]:
    print(file)
