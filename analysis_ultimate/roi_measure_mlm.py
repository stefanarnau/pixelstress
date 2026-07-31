# -----------------------------------------------------------------------------
# Trial-level ROI extraction and mixed-effects model
#
# Select:
# - measure: ERP, theta, alpha, or beta
# - channels
# - time window
# - correct-only versus all trials
# - baseline-corrected versus absolute log power
#
# The script extracts one scalar EEG value per trial and fits one hierarchical
# mixed-effects model with trials nested in sequences and participants.
# -----------------------------------------------------------------------------

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_trial_data/")
PATH_OUT = PATH_IN / "roi_mlm_results"

FILE_EEG = PATH_IN / "trial_level_eeg.h5"
FILE_METADATA = PATH_IN / "trial_level_metadata.csv"

PATH_OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------

# Stored HDF5 measure: "erp", "theta", "alpha", or "beta"
MEASURE = "alpha"

# Channels used for the spatial average.
CHANNELS = ["Pz"]

# Time window in seconds relative to target.
TIME_TMIN = -0.5
TIME_TMAX = 0

# Trial selection.
CORRECT_ONLY = False

# Power transformation.
#
# Used only for theta, alpha, and beta:
# True  -> dB relative to the participant-specific grand pre-cue baseline
# False -> absolute log power in dB
#
# ERP data are kept in their stored voltage units.
APPLY_POWER_BASELINE = True
BASELINE_TMIN = -1.800
BASELINE_TMAX = -1.400

# Predictors.
#
# Feedback remains target-centered:
# feedback = 0 denotes performance at the target.
#
# Continuous nuisance covariates listed here are grand-mean centered before
# model fitting. Remove an item if it should not enter the model.
COVARIATES_TO_CENTER = [
    "sequence_difficulty",
    #"half",
]

# Random-effects structure.
#
# Participant random intercept is always included.
# When True, a sequence random intercept is additionally fitted as a variance
# component. Sequences are uniquely identified within participant and block.
INCLUDE_SEQUENCE_RANDOM_INTERCEPT = True

# Estimation settings.
REML = False
OPTIMIZER = "lbfgs"
MAXITER = 2000

# Save the extracted trial-level ROI values.
SAVE_TRIAL_VALUES = True


# -----------------------------------------------------------------------------
# Validate settings
# -----------------------------------------------------------------------------
VALID_MEASURES = {"erp", "theta", "alpha", "beta"}

if MEASURE not in VALID_MEASURES:
    raise ValueError(
        f"MEASURE must be one of {sorted(VALID_MEASURES)}, got {MEASURE!r}."
    )

if not CHANNELS:
    raise ValueError("CHANNELS must contain at least one channel.")

if TIME_TMIN >= TIME_TMAX:
    raise ValueError("TIME_TMIN must be smaller than TIME_TMAX.")

if BASELINE_TMIN >= BASELINE_TMAX:
    raise ValueError("BASELINE_TMIN must be smaller than BASELINE_TMAX.")


# -----------------------------------------------------------------------------
# Load metadata and EEG information
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_METADATA)

with h5py.File(FILE_EEG, mode="r") as h5_file:
    all_times = h5_file["times"][:]
    all_channels = np.asarray(
        h5_file["channels"].asstr()[:]
    )

available_channels = set(all_channels.tolist())
missing_channels = [
    channel
    for channel in CHANNELS
    if channel not in available_channels
]

if missing_channels:
    raise ValueError(
        f"Requested channels not found in the HDF5 file: {missing_channels}"
    )

channel_indices = np.asarray(
    [
        np.where(all_channels == channel)[0][0]
        for channel in CHANNELS
    ],
    dtype=int,
)

# Inclusive ROI window.
time_mask = (
    (all_times >= TIME_TMIN)
    & (all_times <= TIME_TMAX)
)
time_indices = np.where(time_mask)[0]

if len(time_indices) == 0:
    raise ValueError(
        "No EEG samples found in the requested analysis window."
    )

# Baseline upper bound is exclusive.
baseline_mask = (
    (all_times >= BASELINE_TMIN)
    & (all_times < BASELINE_TMAX)
)
baseline_indices = np.where(baseline_mask)[0]

if (
    MEASURE != "erp"
    and APPLY_POWER_BASELINE
    and len(baseline_indices) == 0
):
    raise ValueError(
        "No EEG samples found in the requested baseline window."
    )


# -----------------------------------------------------------------------------
# Trial selection
# -----------------------------------------------------------------------------
selection_mask = np.ones(
    len(df),
    dtype=bool,
)

if CORRECT_ONLY:
    selection_mask &= (
        df["accuracy"].to_numpy() == 1
    )

selected_indices = np.where(selection_mask)[0]
model_df = df.loc[selection_mask].copy().reset_index(drop=True)

if len(model_df) == 0:
    raise ValueError("No trials remain after trial selection.")


# -----------------------------------------------------------------------------
# Load and reduce EEG data
# -----------------------------------------------------------------------------
#
# One scalar value is extracted per trial by averaging across the selected
# channels and selected time points.
#
# Power:
# - positivity is checked before log transformation;
# - baseline correction uses a participant-specific, channel-specific grand
#   baseline averaged across all selected trials of that participant;
# - the transformed data are then averaged across the requested ROI.
#
# ERP:
# - stored voltage values are averaged directly.
# -----------------------------------------------------------------------------
with h5py.File(FILE_EEG, mode="r") as h5_file:

    if MEASURE == "erp":

        roi_data = h5_file[MEASURE][
            selected_indices,
            :,
            :,
        ][
            :,
            channel_indices,
            :,
        ][
            :,
            :,
            time_indices,
        ].astype(np.float64)

        if not np.all(np.isfinite(roi_data)):
            raise ValueError(
                "ERP ROI contains non-finite values."
            )

        eeg_value = roi_data.mean(axis=(1, 2))

    else:

        # Load only the selected channels, but retain the full time axis so
        # the baseline and analysis windows can both be extracted.
        power_data = h5_file[MEASURE][
            selected_indices,
            :,
            :,
        ][
            :,
            channel_indices,
            :,
        ].astype(np.float64)

        if (
            not np.all(np.isfinite(power_data))
            or np.any(power_data <= 0)
        ):
            raise ValueError(
                f"{MEASURE.capitalize()} power contains non-finite or "
                "non-positive values."
            )

        if APPLY_POWER_BASELINE:

            transformed_power = np.empty_like(
                power_data,
                dtype=np.float64,
            )

            participant_ids = model_df["id"].to_numpy()

            for participant in np.sort(
                model_df["id"].unique()
            ):

                participant_mask = (
                    participant_ids == participant
                )

                # One grand baseline value per selected channel.
                baseline_reference = power_data[
                    participant_mask,
                    :,
                    :,
                ][
                    :,
                    :,
                    baseline_indices,
                ].mean(axis=(0, 2))

                if (
                    not np.all(np.isfinite(baseline_reference))
                    or np.any(baseline_reference <= 0)
                ):
                    raise ValueError(
                        f"Invalid {MEASURE} baseline reference for "
                        f"participant {participant}."
                    )

                transformed_power[
                    participant_mask,
                    :,
                    :,
                ] = 10.0 * np.log10(
                    power_data[
                        participant_mask,
                        :,
                        :,
                    ]
                    / baseline_reference[
                        np.newaxis,
                        :,
                        np.newaxis,
                    ]
                )

        else:

            transformed_power = 10.0 * np.log10(
                power_data
            )

        eeg_value = transformed_power[
            :,
            :,
            time_indices,
        ].mean(axis=(1, 2))


model_df["eeg_value"] = eeg_value


# -----------------------------------------------------------------------------
# Prepare predictors
# -----------------------------------------------------------------------------
required_columns = [
    "id",
    "group",
    "feedback",
    "feedback2",
    "block_nr",
    "sequence_nr",
]

missing_columns = [
    column
    for column in required_columns
    if column not in model_df.columns
]

if missing_columns:
    raise ValueError(
        f"Required metadata columns are missing: {missing_columns}"
    )

model_df["experimental"] = (
    model_df["group"] == "experimental"
).astype(int)

# Unique sequence identifier. sequence_nr repeats across blocks.
model_df["sequence_uid"] = (
    model_df["id"].astype(str)
    + "_b"
    + model_df["block_nr"].astype(str)
    + "_s"
    + model_df["sequence_nr"].astype(str)
)

centered_covariates = []

for covariate in COVARIATES_TO_CENTER:

    if covariate not in model_df.columns:
        raise ValueError(
            f"Requested covariate {covariate!r} is absent from metadata."
        )

    centered_name = f"{covariate}_c"

    model_df[centered_name] = (
        model_df[covariate]
        - model_df[covariate].mean()
    )

    centered_covariates.append(
        centered_name
    )


# -----------------------------------------------------------------------------
# Remove incomplete model rows
# -----------------------------------------------------------------------------
model_columns = [
    "eeg_value",
    "id",
    "experimental",
    "feedback",
    "feedback2",
    "sequence_uid",
] + centered_covariates

complete_mask = np.ones(
    len(model_df),
    dtype=bool,
)

for column in model_columns:

    if pd.api.types.is_numeric_dtype(
        model_df[column]
    ):
        complete_mask &= np.isfinite(
            model_df[column].to_numpy(dtype=float)
        )
    else:
        complete_mask &= model_df[column].notna().to_numpy()

removed_rows = int(
    np.sum(~complete_mask)
)

if removed_rows:
    print(
        f"Removing {removed_rows} rows with incomplete model data."
    )

model_df = model_df.loc[
    complete_mask
].copy().reset_index(drop=True)

if len(model_df) == 0:
    raise ValueError(
        "No complete rows remain for model fitting."
    )


# -----------------------------------------------------------------------------
# Mixed-effects model
# -----------------------------------------------------------------------------
fixed_terms = [
    "experimental",
    "feedback",
    "feedback2",
    "experimental:feedback",
    "experimental:feedback2",
] + centered_covariates

formula = (
    "eeg_value ~ "
    + " + ".join(fixed_terms)
)

vc_formula = None

if INCLUDE_SEQUENCE_RANDOM_INTERCEPT:
    vc_formula = {
        "sequence": "0 + C(sequence_uid)"
    }

print()
print("Analysis definition")
print("-------------------")
print(f"Measure:           {MEASURE}")
print(f"Channels:          {CHANNELS}")
print(
    f"Time window:       {TIME_TMIN:.3f} to {TIME_TMAX:.3f} s"
)
print(f"Correct only:      {CORRECT_ONLY}")

if MEASURE != "erp":
    print(
        f"Power baseline:    {APPLY_POWER_BASELINE}"
    )
    if APPLY_POWER_BASELINE:
        print(
            f"Baseline window:   "
            f"{BASELINE_TMIN:.3f} to {BASELINE_TMAX:.3f} s"
        )

print(f"Trials:            {len(model_df)}")
print(
    f"Participants:      {model_df['id'].nunique()}"
)
print(
    f"Sequences:         {model_df['sequence_uid'].nunique()}"
)
print(f"Formula:           {formula}")
print(
    f"Sequence random:   {INCLUDE_SEQUENCE_RANDOM_INTERCEPT}"
)
print()

model = smf.mixedlm(
    formula=formula,
    data=model_df,
    groups=model_df["id"],
    re_formula="1",
    vc_formula=vc_formula,
)

result = model.fit(
    reml=REML,
    method=OPTIMIZER,
    maxiter=MAXITER,
    full_output=True,
    disp=True,
)

print()
print(result.summary())


# -----------------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------------
channel_label = "-".join(CHANNELS)
trial_label = (
    "correct"
    if CORRECT_ONLY
    else "all"
)
baseline_label = ""

if MEASURE != "erp":
    baseline_label = (
        "_grandbaseline"
        if APPLY_POWER_BASELINE
        else "_absolute_log"
    )

analysis_stem = (
    f"{MEASURE}"
    f"_{channel_label}"
    f"_{TIME_TMIN:+.3f}_{TIME_TMAX:+.3f}"
    f"_{trial_label}"
    f"{baseline_label}"
).replace(".", "p")

summary_file = (
    PATH_OUT
    / f"{analysis_stem}_model_summary.txt"
)

with open(
    summary_file,
    mode="w",
    encoding="utf-8",
) as file:
    file.write(result.summary().as_text())

coefficient_table = pd.DataFrame(
    {
        "term": result.params.index,
        "estimate": result.params.to_numpy(),
        "standard_error": result.bse.to_numpy(),
        "z_value": result.tvalues.to_numpy(),
        "p_value": result.pvalues.to_numpy(),
    }
)

coefficient_file = (
    PATH_OUT
    / f"{analysis_stem}_coefficients.csv"
)

coefficient_table.to_csv(
    coefficient_file,
    index=False,
)

if SAVE_TRIAL_VALUES:

    trial_file = (
        PATH_OUT
        / f"{analysis_stem}_trial_values.csv"
    )

    model_df.to_csv(
        trial_file,
        index=False,
    )

print()
print("Saved:")
print(summary_file)
print(coefficient_file)

if SAVE_TRIAL_VALUES:
    print(trial_file)
