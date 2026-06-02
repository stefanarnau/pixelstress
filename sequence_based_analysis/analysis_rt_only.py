# -----------------------------------------------------------------------------
# RT-only analysis from full sequence EEG dataframe
# Input file includes channel-wise EEG rows, but RT is sequence-level.
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PATH_IN = Path("/mnt/data_dump/pixelstress/3_sequence_data3/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/rt_analysis/")
PATH_OUT.mkdir(parents=True, exist_ok=True)

FILE_IN = PATH_IN / "all_subjects_seq_fooof_rt_channelwise_long_car.csv"

df = pd.read_csv(FILE_IN)

# One row per subject x sequence; EEG rows are channel-wise duplicates for RT
seq = (
    df[
        [
            "id", "group", "block_nr", "sequence_nr",
            "half", "f", "mean_trial_difficulty", "mean_rt"
        ]
    ]
    .drop_duplicates()
    .copy()
)

seq["id"] = seq["id"].astype(str)
seq["group"] = pd.Categorical(seq["group"], categories=["control", "experimental"])
seq["half"] = pd.Categorical(seq["half"])

seq["f"] = pd.to_numeric(seq["f"], errors="coerce")
seq["mean_rt"] = pd.to_numeric(seq["mean_rt"], errors="coerce")
seq["mean_trial_difficulty"] = pd.to_numeric(seq["mean_trial_difficulty"], errors="coerce")

seq["f_c"] = seq["f"] - seq["f"].mean()
seq["f2"] = seq["f_c"] ** 2 - np.mean(seq["f_c"] ** 2)

seq["mean_trial_difficulty_c"] = (
    seq["mean_trial_difficulty"] - seq["mean_trial_difficulty"].mean()
)

seq = seq.dropna(
    subset=["mean_rt", "group", "id", "half", "f", "f2", "mean_trial_difficulty_c"]
)

print("Rows:", len(seq))
print("Subjects:", seq["id"].nunique())

FORMULA = """
mean_rt ~ group * f + group * f2
          + mean_trial_difficulty_c + half
"""

RE_FORMULAS = ["1 + f + f2", "1 + f", "1"]

def fit_mixedlm_with_fallback(df_model):
    logs = []
    for re_formula in RE_FORMULAS:
        try:
            model = smf.mixedlm(
                FORMULA,
                data=df_model,
                groups=df_model["id"],
                re_formula=re_formula,
            )
            fit = model.fit(method="lbfgs", reml=False, maxiter=4000, disp=False)

            if bool(getattr(fit, "converged", False)):
                return fit, re_formula, logs

            logs.append(f"{re_formula}: converged=False")
        except Exception as exc:
            logs.append(f"{re_formula}: {exc}")

    return None, None, logs

fit, used_re, logs = fit_mixedlm_with_fallback(seq)

if fit is None:
    raise RuntimeError("RT model failed: " + " | ".join(logs))

print(fit.summary())
print("Random-effects structure:", used_re)

fe = fit.fe_params
se = fit.bse_fe.reindex(fe.index)
zvals = fe / se.replace(0, np.nan)
pvals = fit.pvalues.reindex(fe.index)

results = pd.DataFrame({
    "term": fe.index,
    "beta": fe.values,
    "se": se.values,
    "z": zvals.values,
    "p": pvals.values,
    "random_effects": used_re,
    "n_subjects": seq["id"].nunique(),
    "n_obs": len(seq),
    "llf": fit.llf,
    "aic": fit.aic if np.isfinite(fit.aic) else np.nan,
    "bic": fit.bic if np.isfinite(fit.bic) else np.nan,
})

results.to_csv(PATH_OUT / "rt_mixedlm_results.csv", index=False)
seq.to_csv(PATH_OUT / "rt_sequence_model_data.csv", index=False)

print("Saved:", PATH_OUT / "rt_mixedlm_results.csv")