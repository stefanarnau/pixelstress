from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# Settings
PATH_IN = Path("/mnt/data_dump/pixelstress/3_trial_data/")
PATH_OUT = PATH_IN / "roi_mlm_results"
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_EEG = PATH_IN / "trial_level_eeg.h5"
FILE_METADATA = PATH_IN / "trial_level_metadata.csv"

MEASURE = "alpha"  # "erp", "theta", "alpha", or "beta"
CHANNELS = ["POz", "Pz", "PO7", "PO8"]
TIME_TMIN, TIME_TMAX = -0.8, 0

CORRECT_ONLY = False

APPLY_POWER_BASELINE = False
BASELINE_TMIN, BASELINE_TMAX = -1.800, -1.400

COVARIATES_TO_CENTER = [
    "sequence_difficulty",
    # "half",
]

INCLUDE_SEQUENCE_RANDOM_INTERCEPT = False

OPTIMIZER = "bfgs"
MAXITER = 2000
SAVE_TRIAL_VALUES = True


# Load metadata and EEG information
df = pd.read_csv(FILE_METADATA)

with h5py.File(FILE_EEG, "r") as h5:
    times = h5["times"][:]
    channels = np.asarray(h5["channels"].asstr()[:])

channel_indices = np.array([
    np.where(channels == channel)[0][0]
    for channel in CHANNELS
])

time_indices = np.where(
    (times >= TIME_TMIN) & (times <= TIME_TMAX)
)[0]

baseline_indices = np.where(
    (times >= BASELINE_TMIN) & (times < BASELINE_TMAX)
)[0]


# Select trials
selection_mask = np.ones(len(df), dtype=bool)

if CORRECT_ONLY:
    selection_mask &= df["accuracy"].to_numpy() == 1

selected_indices = np.where(selection_mask)[0]
model_df = df.loc[selection_mask].copy().reset_index(drop=True)


# Extract one ROI value per trial
with h5py.File(FILE_EEG, "r") as h5:
    if MEASURE == "erp":
        roi_data = h5[MEASURE][
            selected_indices,
            :,
            :,
        ][:, channel_indices, :][:, :, time_indices]

        eeg_value = roi_data.mean(axis=(1, 2))

    else:
        power_data = h5[MEASURE][
            selected_indices,
            :,
            :,
        ][:, channel_indices, :].astype(float)

        log_power = 10 * np.log10(power_data)

        if APPLY_POWER_BASELINE:
            participant_ids = model_df["id"].to_numpy()
            transformed_power = np.empty_like(log_power)

            for participant in np.sort(model_df["id"].unique()):
                participant_mask = participant_ids == participant

                baseline_reference = log_power[
                    participant_mask,
                    :,
                    :,
                ][:, :, baseline_indices].mean(axis=(0, 2))

                transformed_power[participant_mask] = (
                    log_power[participant_mask]
                    - baseline_reference[None, :, None]
                )
        else:
            transformed_power = log_power

        eeg_value = transformed_power[
            :,
            :,
            time_indices,
        ].mean(axis=(1, 2))

model_df["eeg_value"] = eeg_value


# Prepare predictors
model_df["experimental"] = (
    model_df["group"] == "experimental"
).astype(int)

model_df["feedback"] = pd.to_numeric(
    model_df["feedback"],
    errors="coerce",
)
model_df["feedback_c"] = (
    model_df["feedback"] - model_df["feedback"].mean()
)
model_df["feedback2_c"] = model_df["feedback_c"] ** 2

model_df["sequence_uid"] = (
    model_df["id"].astype(str)
    + "_b"
    + model_df["block_nr"].astype(str)
    + "_s"
    + model_df["sequence_nr"].astype(str)
)

centered_covariates = []

for covariate in COVARIATES_TO_CENTER:
    centered_name = f"{covariate}_c"
    numeric_covariate = pd.to_numeric(
        model_df[covariate],
        errors="coerce",
    )
    model_df[centered_name] = (
        numeric_covariate - numeric_covariate.mean()
    )
    centered_covariates.append(centered_name)

model_columns = [
    "eeg_value",
    "id",
    "experimental",
    "feedback_c",
    "feedback2_c",
    "sequence_uid",
] + centered_covariates

model_df = (
    model_df
    .dropna(subset=model_columns)
    .reset_index(drop=True)
)


# Mixed-effects model
fixed_terms = [
    "experimental",
    "feedback_c",
    "feedback2_c",
    "experimental:feedback_c",
    "experimental:feedback2_c",
] + centered_covariates

formula = "eeg_value ~ " + " + ".join(fixed_terms)

vc_formula = None
if INCLUDE_SEQUENCE_RANDOM_INTERCEPT:
    vc_formula = {
        "sequence": "0 + C(sequence_uid)"
    }

model = smf.mixedlm(
    formula=formula,
    data=model_df,
    groups=model_df["id"],
    re_formula="1 + feedback_c",
    vc_formula=vc_formula,
)

result = model.fit(
    reml=False,
    method=OPTIMIZER,
    maxiter=MAXITER,
)


# Output
print("\nAnalysis definition")
print("-------------------")
print(f"Measure:           {MEASURE}")
print(f"Channels:          {CHANNELS}")
print(f"Time window:       {TIME_TMIN:.3f} to {TIME_TMAX:.3f} s")
print(f"Correct only:      {CORRECT_ONLY}")

if MEASURE != "erp":
    print(f"Power baseline:    {APPLY_POWER_BASELINE}")
    if APPLY_POWER_BASELINE:
        print(
            f"Baseline window:   "
            f"{BASELINE_TMIN:.3f} to {BASELINE_TMAX:.3f} s"
        )

print(f"Trials:            {len(model_df)}")
print(f"Participants:      {model_df['id'].nunique()}")
print(f"Sequences:         {model_df['sequence_uid'].nunique()}")
print(f"Formula:           {formula}")
print(
    f"Random effects:    participant intercept + feedback slope"
    + (
        " + sequence intercept"
        if INCLUDE_SEQUENCE_RANDOM_INTERCEPT
        else ""
    )
)
print()
print(result.summary())


# Save outputs
channel_label = "-".join(CHANNELS)
trial_label = "correct" if CORRECT_ONLY else "all"

if MEASURE == "erp":
    baseline_label = ""
elif APPLY_POWER_BASELINE:
    baseline_label = "_grandbaseline_db"
else:
    baseline_label = "_absolute_log"

random_label = (
    "participant_feedback_slope_sequence_RE"
    if INCLUDE_SEQUENCE_RANDOM_INTERCEPT
    else "participant_feedback_slope_RE"
)

analysis_stem = (
    f"{MEASURE}"
    f"_{channel_label}"
    f"_{TIME_TMIN:+.3f}_{TIME_TMAX:+.3f}"
    f"_{trial_label}"
    f"{baseline_label}"
    f"_{random_label}"
).replace(".", "p")

summary_file = PATH_OUT / f"{analysis_stem}_model_summary.txt"
coefficient_file = PATH_OUT / f"{analysis_stem}_coefficients.csv"

analysis_definition = f"""Analysis definition
-------------------
Measure:           {MEASURE}
Channels:          {CHANNELS}
Time window:       {TIME_TMIN:.3f} to {TIME_TMAX:.3f} s
Correct only:      {CORRECT_ONLY}
Power baseline:    {APPLY_POWER_BASELINE if MEASURE != "erp" else "n/a"}
Baseline window:   {f"{BASELINE_TMIN:.3f} to {BASELINE_TMAX:.3f} s" if MEASURE != "erp" and APPLY_POWER_BASELINE else "n/a"}
Trials:            {len(model_df)}
Participants:      {model_df["id"].nunique()}
Sequences:         {model_df["sequence_uid"].nunique()}
Formula:           {formula}
Random effects:    participant intercept + feedback slope{" + sequence intercept" if INCLUDE_SEQUENCE_RANDOM_INTERCEPT else ""}

"""

summary_file.write_text(
    analysis_definition + result.summary().as_text(),
    encoding="utf-8",
)

fixed_effect_names = result.fe_params.index
confidence_intervals = result.conf_int().loc[fixed_effect_names]

coefficient_table = pd.DataFrame({
    "term": fixed_effect_names,
    "estimate": result.fe_params,
    "standard_error": result.bse_fe,
    "z_value": result.tvalues.loc[fixed_effect_names],
    "p_value": result.pvalues.loc[fixed_effect_names],
    "ci_low": confidence_intervals[0],
    "ci_high": confidence_intervals[1],
})
coefficient_table.to_csv(coefficient_file, index=False)

saved_files = [
    summary_file,
    coefficient_file,
]

if SAVE_TRIAL_VALUES:
    trial_file = PATH_OUT / f"{analysis_stem}_trial_values.csv"
    model_df.to_csv(trial_file, index=False)
    saved_files.append(trial_file)

print("\nSaved:")
for saved_file in saved_files:
    print(saved_file)
