from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned2/")
IDS_TO_DROP = {1, 2, 3, 4, 5, 6, 13, 17, 25, 40, 49, 83}


# Load the same participant-specific trial metadata as the GED analysis
frames = []

for dataset in sorted(PATH_IN.glob("*erp.set")):
    trialinfo_file = Path(str(dataset).split("_cleaned")[0] + "_erp_trialinfo.csv")
    subject_df = pd.read_csv(trialinfo_file).reset_index(drop=True)
    subject_id = int(subject_df["id"].iloc[0])

    if subject_id in IDS_TO_DROP:
        continue

    subject_df = subject_df.rename(columns={
        "session_condition": "group",
        "trial_nr": "trial_nr_sequence",
        "trial_nr_total": "trial_nr_global",
        "last_feedback_scaled": "feedback",
    })

    subject_df["group"] = subject_df["group"].replace({1: "experimental", 2: "control"})
    subject_df["experimental"] = (subject_df["group"] == "experimental").astype(int)
    subject_df["accuracy"] = (pd.to_numeric(subject_df["accuracy"], errors="coerce") == 1).astype(int)
    subject_df["feedback"] = pd.to_numeric(subject_df["feedback"], errors="coerce")
    subject_df["sequence_difficulty"] = pd.to_numeric(
        subject_df["sequence_difficulty"], errors="coerce"
    )

    # Same sequence exclusion as the GED analysis
    subject_df = subject_df.loc[subject_df["sequence_nr"] > 1]
    frames.append(subject_df)


df = pd.concat(frames, ignore_index=True)
df["feedback_c"] = df["feedback"] - df["feedback"].mean()
df["feedback2_c"] = df["feedback_c"] ** 2
df["sequence_difficulty_c"] = (
    df["sequence_difficulty"] - df["sequence_difficulty"].mean()
)

print("\nLoaded data")
print("-----------")
print(f"Participants: {df['id'].nunique()}")
print(f"Trials:       {len(df)}")
print(df["group"].value_counts())


# Reaction time: valid, correct trials only
rt_df = df.loc[
    df["rt"].notna()
    & np.isfinite(df["rt"])
    & (df["rt"] > 0)
    & (df["accuracy"] == 1)
].copy()

rt_formula = (
    "rt ~ experimental + feedback_c + feedback2_c "
    "+ experimental:feedback_c + experimental:feedback2_c "
    "+ sequence_difficulty_c"
)

rt_model = smf.mixedlm(
    formula=rt_formula,
    data=rt_df,
    groups=rt_df["id"],
    re_formula="1 + feedback_c",
)

rt_result = rt_model.fit(
    reml=False,
    method="bfgs",
    maxiter=5000,
)

print("\n" + "=" * 70)
print("Trial-level RT model")
print("=" * 70)
print(f"Participants: {rt_df['id'].nunique()}")
print(f"Trials:       {len(rt_df)}")
print(f"Formula:      {rt_formula}\n")
print(rt_result.summary())

rt_ci = rt_result.conf_int().loc[rt_result.fe_params.index]
rt_fixed = pd.DataFrame({
    "beta": rt_result.fe_params,
    "se": rt_result.bse_fe,
    "ci_low": rt_ci[0],
    "ci_high": rt_ci[1],
    "p": rt_result.pvalues.loc[rt_result.fe_params.index],
})
print("\nFixed effects")
print(rt_fixed.round(4))


# Accuracy: all retained trials
accuracy_df = df.dropna(subset=[
    "accuracy",
    "id",
    "experimental",
    "feedback_c",
    "feedback2_c",
    "sequence_difficulty_c",
]).copy()

accuracy_formula = (
    "accuracy ~ experimental + feedback_c + feedback2_c "
    "+ experimental:feedback_c + experimental:feedback2_c "
    "+ sequence_difficulty_c"
)

accuracy_model = smf.gee(
    formula=accuracy_formula,
    groups="id",
    data=accuracy_df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Exchangeable(),
)
accuracy_result = accuracy_model.fit()

print("\n" + "=" * 70)
print("Trial-level accuracy model (GEE)")
print("=" * 70)
print(f"Participants: {accuracy_df['id'].nunique()}")
print(f"Trials:       {len(accuracy_df)}")
print(f"Formula:      {accuracy_formula}\n")
print(accuracy_result.summary())

accuracy_ci = accuracy_result.conf_int()
accuracy_fixed = pd.DataFrame({
    "beta": accuracy_result.params,
    "se": accuracy_result.bse,
    "ci_low": accuracy_ci[0],
    "ci_high": accuracy_ci[1],
    "p": accuracy_result.pvalues,
    "odds_ratio": np.exp(accuracy_result.params),
    "or_ci_low": np.exp(accuracy_ci[0]),
    "or_ci_high": np.exp(accuracy_ci[1]),
})
print("\nFixed effects")
print(accuracy_fixed.round(4))
