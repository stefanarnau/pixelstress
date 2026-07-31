# -----------------------------------------------------------------------------
# Trial-level behavioral model
# -----------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/3_trial_data/")

FILE_METADATA = PATH_IN / "trial_level_metadata.csv"


# -----------------------------------------------------------------------------
# Load trial-level metadata
# -----------------------------------------------------------------------------
df = pd.read_csv(FILE_METADATA)


# -----------------------------------------------------------------------------
# Prepare variables
# -----------------------------------------------------------------------------
df["experimental"] = (
    df["group"] == "experimental"
).astype(int)

# Create squared feedback term if it is not already present
if "feedback2" not in df.columns:
    df["feedback2"] = df["feedback"] ** 2


# -----------------------------------------------------------------------------
# Select valid RT trials
# -----------------------------------------------------------------------------
analysis_df = df.loc[
    df["rt"].notna()
    & np.isfinite(df["rt"])
    & (df["rt"] > 0)
].copy()

# Include only correct trials when an accuracy variable is available
if "accuracy" in analysis_df.columns:
    analysis_df = analysis_df.loc[
        analysis_df["accuracy"] == 1
    ].copy()


# -----------------------------------------------------------------------------
# Trial-level mixed-effects model
#
# Fixed effects:
# group
# feedback
# feedback²
# group × feedback
# group × feedback²
# trial difficulty
# experimental half
#
# Random effects:
# participant intercept
# -----------------------------------------------------------------------------
formula = (
    "rt ~ "
    "experimental "
    "+ feedback "
    "+ feedback2 "
    "+ experimental:feedback "
    "+ experimental:feedback2 "
    #"+ trial_difficulty "
    "+ sequence_difficulty "
    #"+ half"
)

model = smf.mixedlm(
    formula=formula,
    data=analysis_df,
    groups=analysis_df["id"],
    re_formula="1",
)

result = model.fit(
    reml=True,
    method="bfgs",
    maxiter=5000,
)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
print()
print(f"Participants: {analysis_df['id'].nunique()}")
print(f"Trials:       {len(analysis_df)}")
print()

print(result.summary())


# -----------------------------------------------------------------------------
# Fixed-effect table
# -----------------------------------------------------------------------------
confidence_intervals = result.conf_int()

fixed_effects = pd.DataFrame(
    {
        "beta": result.fe_params,
        "se": result.bse_fe,
        "ci_low": confidence_intervals.loc[
            result.fe_params.index,
            0,
        ],
        "ci_high": confidence_intervals.loc[
            result.fe_params.index,
            1,
        ],
        "p": result.pvalues.loc[
            result.fe_params.index
        ],
    }
)

print()
print("Fixed effects")
print(fixed_effects.round(4))


# -----------------------------------------------------------------------------
# Trial-level accuracy model (GEE)
# -----------------------------------------------------------------------------
import statsmodels.api as sm


analysis_df = df.loc[
    df["accuracy"].notna()
].copy()

formula = (
    "accuracy ~ "
    "experimental "
    "+ feedback "
    "+ feedback2 "
    "+ experimental:feedback "
    "+ experimental:feedback2 "
    #"+ trial_difficulty "
    "+ sequence_difficulty "
    #"+ half"
)

gee = smf.gee(
    formula=formula,
    groups="id",
    data=analysis_df,
    family=sm.families.Binomial(),
    cov_struct=sm.cov_struct.Exchangeable(),
)

result = gee.fit()

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
print()
print("=" * 70)
print("Trial-level accuracy model (GEE)")
print("=" * 70)
print()

print(f"Participants: {analysis_df['id'].nunique()}")
print(f"Trials:       {len(analysis_df)}")
print()

print(result.summary())

# -----------------------------------------------------------------------------
# Fixed-effect table
# -----------------------------------------------------------------------------
confidence_intervals = result.conf_int()

fixed_effects = pd.DataFrame(
    {
        "beta": result.params,
        "se": result.bse,
        "ci_low": confidence_intervals[0],
        "ci_high": confidence_intervals[1],
        "p": result.pvalues,
        "odds_ratio": np.exp(result.params),
        "or_ci_low": np.exp(confidence_intervals[0]),
        "or_ci_high": np.exp(confidence_intervals[1]),
    }
)

print()
print("Fixed effects")
print(fixed_effects.round(4))



