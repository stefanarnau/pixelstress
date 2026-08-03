# -----------------------------------------------------------------------------
# Analyze saved GED-theta component power with a trial-level mixed model
# and create feedback-bin plots for control and experimental groups.
# -----------------------------------------------------------------------------

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_DATA = Path("/mnt/data_dump/pixelstress/3_trial_data/")
PATH_COMPONENTS = PATH_DATA / "ged_theta" / "components"
FILE_METADATA = PATH_DATA / "trial_level_metadata.csv"
PATH_OUT = PATH_DATA / "ged_theta" / "mlm_results"

PATH_OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------
TMIN = 0.100
TMAX = 0.500

# "absolute_log":
#   10 * log10(mean linear theta power in the analysis window)
#
# "baseline_db":
#   10 * log10(
#       mean theta power in analysis window
#       /
#       mean theta power in baseline window
#   )
POWER_MODE = "absolute_log"

BASELINE_TMIN = -1.800
BASELINE_TMAX = -1.400

CORRECT_ONLY = False

ID_COLUMN = "id"
GROUP_COLUMN = "group"
ACCURACY_COLUMN = "accuracy"

# Change this if needed.
FEEDBACK_COLUMN = "last_feedback_scaled"

DIFFICULTY_COLUMN = "sequence_difficulty"
SEQUENCE_COLUMN = "sequence_uid"

INCLUDE_SEQUENCE_RANDOM_INTERCEPT = True
INCLUDE_HALF = False

OPTIMIZER = "lbfgs"
MAXITER = 2000

# Plot settings
N_FEEDBACK_BINS = 3
SHOW_FIGURE = True
FIGURE_DPI = 180


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def get_component_file(subject_id):
    """Locate one participant's saved theta-component file."""
    candidates = [
        PATH_COMPONENTS / f"sub-{int(subject_id):03d}_theta_component.npz",
        PATH_COMPONENTS / f"sub-{int(subject_id)}_theta_component.npz",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No theta component file found for participant {subject_id} "
        f"in {PATH_COMPONENTS}."
    )


def extract_window_value(theta_power, times):
    """
    Reduce trial x time linear theta power to one scalar per trial.
    """
    theta_power = np.asarray(theta_power, dtype=float)
    times = np.asarray(times, dtype=float)

    analysis_mask = (
        (times >= TMIN)
        & (times <= TMAX)
    )

    if not np.any(analysis_mask):
        raise ValueError(
            f"No samples in analysis window {TMIN:.3f} to {TMAX:.3f} s."
        )

    analysis_power = np.nanmean(
        theta_power[:, analysis_mask],
        axis=1,
    )

    tiny = np.finfo(float).tiny

    if POWER_MODE == "absolute_log":
        return 10.0 * np.log10(
            np.maximum(analysis_power, tiny)
        )

    if POWER_MODE == "baseline_db":
        baseline_mask = (
            (times >= BASELINE_TMIN)
            & (times <= BASELINE_TMAX)
        )

        if not np.any(baseline_mask):
            raise ValueError(
                f"No samples in baseline window "
                f"{BASELINE_TMIN:.3f} to {BASELINE_TMAX:.3f} s."
            )

        baseline_power = np.nanmean(
            theta_power[:, baseline_mask],
            axis=1,
        )

        return 10.0 * np.log10(
            np.maximum(analysis_power, tiny)
            / np.maximum(baseline_power, tiny)
        )

    raise ValueError(
        "POWER_MODE must be 'absolute_log' or 'baseline_db'."
    )


def code_group(df):
    """Create experimental coding: control=0, experimental=1."""
    df = df.copy()

    if "experimental" in df.columns:
        numeric = pd.to_numeric(
            df["experimental"],
            errors="coerce",
        )

        if numeric.notna().all():
            df["experimental"] = numeric.astype(int)
            return df

    if GROUP_COLUMN not in df.columns:
        raise KeyError(
            f"Neither 'experimental' nor '{GROUP_COLUMN}' exists."
        )

    labels = (
        df[GROUP_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "control": 0,
        "ctrl": 0,
        "0": 0,
        "experimental": 1,
        "experiment": 1,
        "exp": 1,
        "1": 1,
    }

    df["experimental"] = labels.map(mapping)

    if df["experimental"].isna().any():
        unknown = sorted(
            labels[df["experimental"].isna()].unique()
        )
        raise ValueError(
            f"Unrecognized group labels: {unknown}"
        )

    df["experimental"] = df["experimental"].astype(int)

    return df


def prepare_predictors(df):
    """Create fixed-effect predictors and sequence identifiers."""
    df = code_group(df)

    if FEEDBACK_COLUMN not in df.columns:
        raise KeyError(
            f"Feedback column '{FEEDBACK_COLUMN}' not found.\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    df["feedback"] = pd.to_numeric(
        df[FEEDBACK_COLUMN],
        errors="coerce",
    )
    df["feedback2"] = df["feedback"] ** 2

    if DIFFICULTY_COLUMN not in df.columns:
        raise KeyError(
            f"Difficulty column '{DIFFICULTY_COLUMN}' not found."
        )

    difficulty = pd.to_numeric(
        df[DIFFICULTY_COLUMN],
        errors="coerce",
    )

    df["sequence_difficulty_c"] = (
        difficulty - difficulty.mean()
    )

    if INCLUDE_HALF:
        if "half" in df.columns:
            half = pd.to_numeric(
                df["half"],
                errors="coerce",
            )
        elif "block_nr" in df.columns:
            half = (
                pd.to_numeric(
                    df["block_nr"],
                    errors="coerce",
                ) > 4
            ).astype(float)
        else:
            raise KeyError(
                "INCLUDE_HALF=True, but no 'half' or 'block_nr' exists."
            )

        df["half_c"] = half - half.mean()

    if SEQUENCE_COLUMN not in df.columns:
        required = {
            ID_COLUMN,
            "block_nr",
            "sequence_nr",
        }

        if not required.issubset(df.columns):
            raise KeyError(
                f"'{SEQUENCE_COLUMN}' absent and cannot be reconstructed."
            )

        df[SEQUENCE_COLUMN] = (
            df[ID_COLUMN].astype(str)
            + "_b"
            + df["block_nr"].astype(str)
            + "_s"
            + df["sequence_nr"].astype(str)
        )

    return df


# -----------------------------------------------------------------------------
# Load trial values
# -----------------------------------------------------------------------------
def load_trial_dataframe():
    """
    Load metadata and attach the saved GED-theta power value to each trial.

    Assumption:
    Trial order within each participant is identical in the component file
    and trial_level_metadata.csv.
    """
    metadata = pd.read_csv(FILE_METADATA)

    if ID_COLUMN not in metadata.columns:
        raise KeyError(
            f"Metadata does not contain '{ID_COLUMN}'."
        )

    participant_frames = []

    for subject_id, subject_df in metadata.groupby(
        ID_COLUMN,
        sort=False,
    ):
        subject_df = subject_df.copy().reset_index(drop=True)

        component_file = get_component_file(subject_id)

        with np.load(
            component_file,
            allow_pickle=False,
        ) as component:
            if "theta_power" not in component:
                raise KeyError(
                    f"{component_file} lacks 'theta_power'."
                )

            if "times" not in component:
                raise KeyError(
                    f"{component_file} lacks 'times'."
                )

            theta_power = component["theta_power"]
            times = component["times"]

        if theta_power.shape[0] != len(subject_df):
            raise ValueError(
                f"Participant {subject_id}: "
                f"{theta_power.shape[0]} component trials versus "
                f"{len(subject_df)} metadata rows."
            )

        subject_df["eeg_value"] = extract_window_value(
            theta_power=theta_power,
            times=times,
        )

        participant_frames.append(subject_df)

    df = pd.concat(
        participant_frames,
        ignore_index=True,
    )

    df = prepare_predictors(df)

    if CORRECT_ONLY:
        if ACCURACY_COLUMN not in df.columns:
            raise KeyError(
                f"Correct-only requested, but '{ACCURACY_COLUMN}' is absent."
            )

        df = df.loc[
            pd.to_numeric(
                df[ACCURACY_COLUMN],
                errors="coerce",
            ) == 1
        ].copy()

    required_model_columns = [
        "eeg_value",
        ID_COLUMN,
        SEQUENCE_COLUMN,
        "experimental",
        "feedback",
        "feedback2",
        "sequence_difficulty_c",
    ]

    if INCLUDE_HALF:
        required_model_columns.append("half_c")

    df = (
        df.dropna(
            subset=required_model_columns
        )
        .reset_index(drop=True)
    )

    return df


# -----------------------------------------------------------------------------
# Mixed-effects model
# -----------------------------------------------------------------------------
def fit_model(df):
    fixed_terms = [
        "experimental",
        "feedback",
        "feedback2",
        "experimental:feedback",
        "experimental:feedback2",
        "sequence_difficulty_c",
    ]

    if INCLUDE_HALF:
        fixed_terms.append("half_c")

    formula = (
        "eeg_value ~ "
        + " + ".join(fixed_terms)
    )

    model_kwargs = {
        "formula": formula,
        "data": df,
        "groups": df[ID_COLUMN],
        "re_formula": "1",
    }

    if INCLUDE_SEQUENCE_RANDOM_INTERCEPT:
        model_kwargs["vc_formula"] = {
            "sequence": f"0 + C({SEQUENCE_COLUMN})"
        }

    model = smf.mixedlm(
        **model_kwargs
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        result = model.fit(
            reml=False,
            method=OPTIMIZER,
            maxiter=MAXITER,
            full_output=True,
            disp=True,
        )

    return formula, result


# -----------------------------------------------------------------------------
# Binned feedback plot
# -----------------------------------------------------------------------------
def make_feedback_bins(df):
    """Create fixed-width bins across the observed feedback range."""
    lower = float(df["feedback"].min())
    upper = float(df["feedback"].max())

    if np.isclose(lower, upper):
        raise ValueError(
            "Feedback contains no variation."
        )

    edges = np.linspace(
        lower,
        upper,
        N_FEEDBACK_BINS + 1,
    )

    edges[0] -= np.finfo(float).eps
    edges[-1] += np.finfo(float).eps

    df = df.copy()

    df["feedback_bin"] = pd.cut(
        df["feedback"],
        bins=edges,
        include_lowest=True,
        ordered=True,
    )

    return df


def calculate_bin_summaries(df):
    """
    Average trials within participant x group x feedback bin,
    then summarize across participants.
    """
    df = make_feedback_bins(df)

    participant_means = (
        df.groupby(
            [
                ID_COLUMN,
                "experimental",
                "feedback_bin",
            ],
            observed=True,
        )
        .agg(
            eeg_value=("eeg_value", "mean"),
            mean_feedback=("feedback", "mean"),
            n_trials=("eeg_value", "size"),
        )
        .reset_index()
    )

    group_summary = (
        participant_means.groupby(
            [
                "experimental",
                "feedback_bin",
            ],
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

    group_summary["sem"] = (
        group_summary["sd"]
        / np.sqrt(
            group_summary["n_participants"]
        )
    )

    return participant_means, group_summary


def plot_feedback_bins(df, output_file):
    """
    Plot participant-averaged theta values by feedback bin,
    separately for control and experimental groups.
    """
    participant_means, group_summary = (
        calculate_bin_summaries(df)
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharey=True,
    )

    group_definitions = [
        (0, "Control"),
        (1, "Experimental"),
    ]

    for ax, (
        group_code,
        group_label,
    ) in zip(
        axes,
        group_definitions,
    ):
        group_data = group_summary.loc[
            group_summary["experimental"]
            == group_code
        ].copy()

        x = np.arange(
            len(group_data)
        )

        ax.errorbar(
            x,
            group_data["mean"],
            yerr=group_data["sem"],
            marker="o",
            linewidth=1.8,
            capsize=4,
        )

        labels = [
            f"{interval.left:.2f}\nto\n{interval.right:.2f}"
            for interval in group_data["feedback_bin"]
        ]

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(group_label)
        ax.set_xlabel("Linear feedback bin")
        ax.axhline(
            0,
            linewidth=0.8,
        )

    if POWER_MODE == "absolute_log":
        ylabel = "GED theta power (dB, absolute log)"
    else:
        ylabel = "GED theta power (dB relative to baseline)"

    axes[0].set_ylabel(ylabel)

    fig.suptitle(
        f"GED theta power by linear feedback bin\n"
        f"{TMIN:.3f} to {TMAX:.3f} s"
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.92)
    )

    fig.savefig(
        output_file,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)

    return participant_means, group_summary


# -----------------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------------
def save_outputs(
    df,
    formula,
    result,
):
    time_label = (
        f"{TMIN:+.3f}_{TMAX:+.3f}"
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )

    trial_label = (
        "correct"
        if CORRECT_ONLY
        else "all"
    )

    random_label = (
        "participant_sequence_RE"
        if INCLUDE_SEQUENCE_RANDOM_INTERCEPT
        else "participant_RE"
    )

    stem = (
        f"theta_ged_{time_label}_"
        f"{trial_label}_{POWER_MODE}_"
        f"{random_label}"
    )

    model_summary_file = (
        PATH_OUT
        / f"{stem}_model_summary.txt"
    )

    coefficients_file = (
        PATH_OUT
        / f"{stem}_coefficients.csv"
    )

    trial_values_file = (
        PATH_OUT
        / f"{stem}_trial_values.csv"
    )

    participant_bins_file = (
        PATH_OUT
        / f"{stem}_participant_bin_means.csv"
    )

    group_bins_file = (
        PATH_OUT
        / f"{stem}_group_bin_summary.csv"
    )

    plot_file = (
        PATH_OUT
        / f"{stem}_feedback_bins.png"
    )

    definition = f"""Analysis definition
-------------------
Measure:           GED theta power
Time window:       {TMIN:.3f} to {TMAX:.3f} s
Power mode:        {POWER_MODE}
Correct only:      {CORRECT_ONLY}
Trials:            {len(df)}
Participants:      {df[ID_COLUMN].nunique()}
Sequences:         {df[SEQUENCE_COLUMN].nunique()}
Formula:           {formula}
Sequence random:   {INCLUDE_SEQUENCE_RANDOM_INTERCEPT}

"""

    model_summary_file.write_text(
        definition
        + result.summary().as_text(),
        encoding="utf-8",
    )

    confidence_intervals = (
        result.conf_int()
    )

    coefficients = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "standard_error": (
            result.bse
            .reindex(result.params.index)
            .values
        ),
        "z_value": (
            result.tvalues
            .reindex(result.params.index)
            .values
        ),
        "p_value": (
            result.pvalues
            .reindex(result.params.index)
            .values
        ),
        "ci_low": (
            confidence_intervals
            .reindex(result.params.index)[0]
            .values
        ),
        "ci_high": (
            confidence_intervals
            .reindex(result.params.index)[1]
            .values
        ),
    })

    coefficients.to_csv(
        coefficients_file,
        index=False,
    )

    columns_to_save = [
        column
        for column in [
            ID_COLUMN,
            GROUP_COLUMN,
            "experimental",
            "block_nr",
            "sequence_nr",
            SEQUENCE_COLUMN,
            "trial_nr",
            FEEDBACK_COLUMN,
            "feedback",
            "feedback2",
            DIFFICULTY_COLUMN,
            "sequence_difficulty_c",
            ACCURACY_COLUMN,
            "eeg_value",
        ]
        if column in df.columns
    ]

    df[columns_to_save].to_csv(
        trial_values_file,
        index=False,
    )

    participant_bins, group_bins = (
        plot_feedback_bins(
            df=df,
            output_file=plot_file,
        )
    )

    participant_bins.to_csv(
        participant_bins_file,
        index=False,
    )

    group_bins.to_csv(
        group_bins_file,
        index=False,
    )

    return {
        "model_summary": model_summary_file,
        "coefficients": coefficients_file,
        "trial_values": trial_values_file,
        "participant_bin_means": participant_bins_file,
        "group_bin_summary": group_bins_file,
        "plot": plot_file,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    df = load_trial_dataframe()

    formula, result = fit_model(
        df
    )

    print()
    print("Analysis definition")
    print("-------------------")
    print("Measure:           GED theta power")
    print(
        f"Time window:       "
        f"{TMIN:.3f} to {TMAX:.3f} s"
    )
    print(f"Power mode:        {POWER_MODE}")
    print(f"Correct only:      {CORRECT_ONLY}")
    print(f"Trials:            {len(df)}")
    print(
        f"Participants:      "
        f"{df[ID_COLUMN].nunique()}"
    )
    print(
        f"Sequences:         "
        f"{df[SEQUENCE_COLUMN].nunique()}"
    )
    print(f"Formula:           {formula}")
    print(
        f"Sequence random:   "
        f"{INCLUDE_SEQUENCE_RANDOM_INTERCEPT}"
    )
    print()

    print(
        result.summary()
    )

    saved_files = save_outputs(
        df=df,
        formula=formula,
        result=result,
    )

    print()
    print("Saved:")

    for saved_file in saved_files.values():
        print(saved_file)


if __name__ == "__main__":
    main()
