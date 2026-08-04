# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import scipy.signal

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PATH_IN = Path("/mnt/data_dump/pixelstress/2_autocleaned2/")
PATH_OUT = Path("/mnt/data_dump/pixelstress/3_trial_data/")

DATASETS = sorted(PATH_IN.glob("*erp.set"))

# -----------------------------------------------------------------------------
# Exclusions
# -----------------------------------------------------------------------------
IDS_TO_DROP = {1, 2, 3, 4, 5, 6, 13, 17, 25, 40, 49, 83}


# -----------------------------------------------------------------------------
# EEG parameters
# -----------------------------------------------------------------------------
SFREQ = 500


# -----------------------------------------------------------------------------
# Channel information
# -----------------------------------------------------------------------------
CHANNEL_LABELS = (
    Path("/home/plkn/repos/pixelstress/chanlabels_pixelstress.txt")
    .read_text()
    .splitlines()
)

INFO_ERP = mne.create_info(
    CHANNEL_LABELS,
    sfreq=SFREQ,
    ch_types="eeg",
    verbose=None,
)

MONTAGE = mne.channels.make_standard_montage("standard_1020")

INFO_ERP.set_montage(
    MONTAGE,
    on_missing="warn",
    match_case=False,
)

# -----------------------------------------------------------------------------
# GED parameters: theta
# -----------------------------------------------------------------------------
THETA_BAND = (4.0, 8.0)
BROAD_BAND = (1.0, 30.0)

GED_TMIN = 0.100
GED_TMAX = 0.500

# Shrinkage applied to the reference covariance.
# A small fixed value is usually enough with 65 channels and many trials.
GED_REGULARIZATION = 0.01

# Number of highest-eigenvalue components among which the template match
# is evaluated. Avoid allowing a low-eigenvalue component to win solely
# because of a chance topographic correlation.
N_COMPONENTS_FOR_TEMPLATE = 10

# -----------------------------------------------------------------------------
# FCz-centered spatial template
# -----------------------------------------------------------------------------
def make_fcz_template(info, sigma=0.075):
    """
    Smooth positive template centered on FCz.

    Parameters
    ----------
    info : mne.Info
        EEG channel information with montage coordinates.
    sigma : float
        Spatial width in metres.

    Returns
    -------
    template : ndarray, shape (n_channels,)
        Unit-norm spatial template.
    """
    ch_pos = info.get_montage().get_positions()["ch_pos"]

    labels = info.ch_names
    xyz = np.array([ch_pos[ch] for ch in labels])

    fcz_idx = labels.index("FCz")
    fcz_xyz = xyz[fcz_idx]

    distances = np.linalg.norm(xyz - fcz_xyz, axis=1)

    template = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    template -= template.mean()
    template /= np.linalg.norm(template)

    return template


FCZ_TEMPLATE = make_fcz_template(INFO_ERP)

# -----------------------------------------------------------------------------
# Signal processing functions
# -----------------------------------------------------------------------------
def bandpass_epochs(data, sfreq, l_freq, h_freq):
    """
    Band-pass filter epoched data along the final dimension.

    Parameters
    ----------
    data : ndarray, shape (trials, channels, times)
    sfreq : float
    l_freq, h_freq : float

    Returns
    -------
    filtered : ndarray
    """
    return mne.filter.filter_data(
        data.astype(np.float64, copy=False),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params={
            "order": 4,
            "ftype": "butter",
        },
        phase="zero",
        verbose=False,
    )


def mean_trial_covariance(data):
    """
    Compute a trial-balanced channel covariance matrix.

    Each trial contributes one covariance matrix, and these matrices
    are averaged. Thus, every trial receives equal weight.

    Parameters
    ----------
    data : ndarray, shape (trials, channels, times)

    Returns
    -------
    covariance : ndarray, shape (channels, channels)
    """
    n_trials, n_channels, _ = data.shape

    covariance = np.zeros(
        (n_channels, n_channels),
        dtype=np.float64,
    )

    for trial in range(n_trials):

        x = data[trial]

        # Remove each channel's temporal mean within this trial.
        x = x - x.mean(axis=1, keepdims=True)

        trial_cov = x @ x.T
        trial_cov /= x.shape[1] - 1

        # Trace normalization prevents high-amplitude trials from
        # dominating the participant-level covariance estimate.
        trace = np.trace(trial_cov)

        if np.isfinite(trace) and trace > 0:
            covariance += trial_cov / trace

    covariance /= n_trials

    return covariance


def regularize_covariance(covariance, gamma):
    """
    Shrink covariance toward a scaled identity matrix.
    """
    n_channels = covariance.shape[0]
    mean_eigenvalue = np.trace(covariance) / n_channels

    return (
        (1.0 - gamma) * covariance
        + gamma * mean_eigenvalue * np.eye(n_channels)
    )

def extract_theta_component_power(
    erp_data,
    spatial_filter,
    sfreq,
    theta_band=(4.0, 8.0),
):
    """
    Project epoched EEG through one spatial filter and return the
    raw component signal, theta-filtered signal, and theta power.

    Parameters
    ----------
    erp_data : ndarray, shape (trials, channels, times)
        Original epoched EEG.
    spatial_filter : ndarray, shape (channels,)
        Selected GED spatial filter.
    sfreq : float
        Sampling frequency.
    theta_band : tuple of float
        Theta passband.

    Returns
    -------
    component_signal : ndarray, shape (trials, times)
        Broadband spatially filtered component signal.
    component_theta : ndarray, shape (trials, times)
        Theta-band component signal.
    component_theta_power : ndarray, shape (trials, times)
        Linear analytic theta power.
    """
    spatial_filter = np.asarray(
        spatial_filter,
        dtype=np.float64,
    ).ravel()

    if erp_data.shape[1] != spatial_filter.size:
        raise ValueError(
            "Spatial-filter length does not match the number of channels."
        )

    # trials x time
    component_signal = np.einsum(
        "c,tcs->ts",
        spatial_filter,
        erp_data,
        optimize=True,
    )

    # Filter the full epoch, not only the 100–500 ms interval.
    component_theta = mne.filter.filter_data(
        component_signal.astype(np.float64, copy=False),
        sfreq=sfreq,
        l_freq=theta_band[0],
        h_freq=theta_band[1],
        method="iir",
        iir_params={
            "order": 4,
            "ftype": "butter",
        },
        phase="zero",
        verbose=False,
    )

    analytic_signal = scipy.signal.hilbert(
        component_theta,
        axis=-1,
    )

    component_theta_power = (
        np.abs(analytic_signal) ** 2
    )

    return {
        "component_signal": component_signal,
        "component_theta": component_theta,
        "component_theta_power": component_theta_power,
    }

# -----------------------------------------------------------------------------
# Save component
# -----------------------------------------------------------------------------

def save_theta_component(
    component_result,
    ged_result,
    times,
    trial_metadata,
    subj_id,
    output_dir,
    analysis_tmin,
    analysis_tmax,
):
    """
    Save the selected theta component and trial-level summary.

    Power is stored in linear units so baseline correction and log
    transformation can be chosen later.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = np.asarray(times, dtype=float)

    analysis_mask = (
        (times >= analysis_tmin)
        & (times <= analysis_tmax)
    )

    if not np.any(analysis_mask):
        raise ValueError(
            "Theta analysis window does not overlap the epoch."
        )

    theta_power = component_result["component_theta_power"]

    trial_mean_theta_power = np.nanmean(
        theta_power[:, analysis_mask],
        axis=1,
    )

    trial_mean_theta_log_power = 10.0 * np.log10(
        np.maximum(
            trial_mean_theta_power,
            np.finfo(float).tiny,
        )
    )

    save_path = (
        output_dir
        / f"sub-{int(subj_id):03d}_theta_component.npz"
    )
    
    trial_key_columns = [
        "id",
        "block_nr",
        "sequence_nr",
        "trial_nr",
    ]
    
    missing_columns = [
        column
        for column in trial_key_columns
        if column not in trial_metadata.columns
    ]
    
    if missing_columns:
        raise KeyError(
            f"Missing trial-key columns: {missing_columns}"
        )
    
    if len(trial_metadata) != theta_power.shape[0]:
        raise ValueError(
            f"Metadata contains {len(trial_metadata)} rows, "
            f"but theta power contains {theta_power.shape[0]} trials."
        )
    
    if trial_metadata.duplicated(trial_key_columns).any():
        raise ValueError(
            "Trial-key columns do not uniquely identify trials."
        )
        
    trial_metadata = trial_metadata.reset_index(drop=True).copy()

    np.savez_compressed(
        save_path,

        # Full time series
        theta_power=theta_power.astype(np.float32),
        times=times.astype(np.float32),

        # Useful trial-level summaries
        trial_mean_theta_power=trial_mean_theta_power.astype(
            np.float32
        ),
        trial_mean_theta_log_power=trial_mean_theta_log_power.astype(
            np.float32
        ),

        # Component definition
        spatial_filter=ged_result["spatial_filter"].astype(
            np.float32
        ),
        forward_model=ged_result["forward_model"].astype(
            np.float32
        ),
        component_index=np.int32(
            ged_result["component_index"]
        ),
        eigenvalue=np.float32(
            ged_result["eigenvalue"]
        ),
        template_correlation=np.float32(
            ged_result["template_correlation"]
        ),

        # Analysis definition
        theta_band=np.asarray(
            THETA_BAND,
            dtype=np.float32,
        ),
        analysis_window=np.asarray(
            [analysis_tmin, analysis_tmax],
            dtype=np.float32,
        ),
        channel_labels=np.asarray(CHANNEL_LABELS),
        
        trial_id=trial_metadata["id"].to_numpy(),
        trial_block_nr=trial_metadata["block_nr"].to_numpy(),
        trial_sequence_nr=trial_metadata["sequence_nr"].to_numpy(),
        trial_trial_nr=trial_metadata["trial_nr"].to_numpy(),
    )

    return save_path

# -----------------------------------------------------------------------------
# GED
# -----------------------------------------------------------------------------
def run_theta_ged(
    erp_data,
    erp_times_sec,
    sfreq,
    template,
):
    """
    Compute a participant-specific theta-versus-broadband GED.

    Returns
    -------
    spatial_filter : ndarray, shape (channels,)
    forward_model : ndarray, shape (channels,)
    eigenvalue : float
    component_index : int
    template_correlation : float
    """
    time_mask = (
        (erp_times_sec >= GED_TMIN)
        & (erp_times_sec <= GED_TMAX)
    )

    if not np.any(time_mask):
        raise ValueError("GED time window does not overlap the epoch.")

    # Filter the full epoch to reduce edge artefacts in the GED interval.
    theta_data = bandpass_epochs(
        erp_data,
        sfreq,
        THETA_BAND[0],
        THETA_BAND[1],
    )

    broadband_data = bandpass_epochs(
        erp_data,
        sfreq,
        BROAD_BAND[0],
        BROAD_BAND[1],
    )

    # Restrict to the predefined post-target covariance window.
    theta_window = theta_data[:, :, time_mask]
    broadband_window = broadband_data[:, :, time_mask]

    signal_cov = mean_trial_covariance(theta_window)
    reference_cov = mean_trial_covariance(broadband_window)

    reference_cov = regularize_covariance(
        reference_cov,
        GED_REGULARIZATION,
    )

    # Generalized eigendecomposition.
    # scipy.linalg.eigh returns eigenvalues in ascending order.
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        signal_cov,
        reference_cov,
    )

    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Forward models / activation patterns.
    #
    # For component interpretation, use covariance @ filter rather
    # than inspecting the filter coefficients directly.
    forward_models = signal_cov @ eigenvectors

    # Normalize each map before template correlation.
    forward_models -= forward_models.mean(axis=0, keepdims=True)

    map_norms = np.linalg.norm(
        forward_models,
        axis=0,
        keepdims=True,
    )

    forward_models /= np.maximum(map_norms, np.finfo(float).eps)

    n_candidates = min(
        N_COMPONENTS_FOR_TEMPLATE,
        forward_models.shape[1],
    )

    correlations = np.array([
        np.corrcoef(
            forward_models[:, component],
            template,
        )[0, 1]
        for component in range(n_candidates)
    ])

    # Polarity is arbitrary, so identify by absolute correlation.
    component_index = int(np.nanargmax(np.abs(correlations)))

    spatial_filter = eigenvectors[:, component_index].copy()
    forward_model = forward_models[:, component_index].copy()
    template_correlation = correlations[component_index]

    # Orient consistently toward the positive FCz template.
    if template_correlation < 0:
        spatial_filter *= -1
        forward_model *= -1
        template_correlation *= -1

    # Normalize filter scale. This does not affect relative trial power,
    # but avoids arbitrary numerical scaling across participants.
    spatial_filter /= np.linalg.norm(spatial_filter)

    return {
        "spatial_filter": spatial_filter,
        "forward_model": forward_model,
        "component_index": component_index,
        "eigenvalue": eigenvalues[component_index],
        "template_correlation": template_correlation,
        "all_filters": eigenvectors,
        "all_forward_models": forward_models,
        "all_eigenvalues": eigenvalues,
    }

def plot_ged_topographies(
    ged_result,
    info,
    template,
    subj_id=None,
    n_components=20,
    n_cols=5,
    title=None,
    save_dir=None,
    filename=None,
    dpi=150,
    show=False,
):
    """
    Plot GED forward-model topographies, highlight the selected component,
    and optionally save the figure.

    Parameters
    ----------
    ged_result : dict
        Output from run_theta_ged(). Must contain:
        - all_forward_models
        - all_eigenvalues
        - component_index
    info : mne.Info
        Channel information with montage coordinates.
    template : ndarray, shape (n_channels,)
        Spatial template used for component selection.
    subj_id : int or str, optional
        Participant identifier used in the title and default filename.
    n_components : int
        Number of leading GED components to plot.
    n_cols : int
        Number of columns in the figure.
    title : str, optional
        Figure title. Generated automatically when omitted.
    save_dir : str or pathlib.Path, optional
        Output directory. Figure is not saved when omitted.
    filename : str, optional
        Output filename. Generated automatically when omitted.
    dpi : int
        Resolution for saved figure.
    show : bool
        Whether to display the figure interactively.

    Returns
    -------
    fig : matplotlib.figure.Figure
    template_correlations : ndarray
        Signed template correlations for all GED components.
    save_path : pathlib.Path or None
        Saved file path, or None if the figure was not saved.
    """
    maps = np.asarray(ged_result["all_forward_models"])
    eigenvalues = np.asarray(ged_result["all_eigenvalues"])
    selected_idx = int(ged_result["component_index"])

    if maps.ndim != 2:
        raise ValueError(
            "all_forward_models must have shape "
            "(n_channels, n_components)."
        )

    if maps.shape[0] != len(info.ch_names):
        raise ValueError(
            "Number of channels in all_forward_models does not match info."
        )

    template = np.asarray(template, dtype=float).ravel()

    if template.size != maps.shape[0]:
        raise ValueError(
            "Template length does not match the number of channels."
        )

    n_available = maps.shape[1]
    n_components = min(n_components, n_available)

    # -------------------------------------------------------------------------
    # Correlate every forward model with the spatial template
    # -------------------------------------------------------------------------
    template_correlations = np.full(n_available, np.nan)

    template_centered = template - np.nanmean(template)
    template_norm = np.linalg.norm(template_centered)

    if template_norm == 0:
        raise ValueError("Template has zero variance.")

    for component_idx in range(n_available):

        component_map = maps[:, component_idx]
        component_centered = (
            component_map - np.nanmean(component_map)
        )
        component_norm = np.linalg.norm(component_centered)

        if component_norm > 0:
            template_correlations[component_idx] = (
                component_centered @ template_centered
            ) / (
                component_norm * template_norm
            )

    # -------------------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------------------
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )

    for component_idx, ax in enumerate(axes.ravel()):

        if component_idx >= n_components:
            ax.axis("off")
            continue

        component_map = maps[:, component_idx]

        # Each map gets its own symmetric scale because the stored maps
        # are normalized for morphology rather than component magnitude.
        vmax = np.nanmax(np.abs(component_map))

        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        mne.viz.plot_topomap(
            component_map,
            info,
            axes=ax,
            show=False,
            contours=0,
            vlim=(-vmax, vmax),
        )

        is_selected = component_idx == selected_idx
        correlation = template_correlations[component_idx]

        component_title = (
            f"GED {component_idx + 1}\n"
            f"λ = {eigenvalues[component_idx]:.3f}, "
            f"r = {correlation:.2f}"
        )

        if is_selected:
            component_title += " | SELECTED"

        ax.set_title(
            component_title,
            fontweight="bold" if is_selected else "normal",
        )

        if is_selected:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)

    # -------------------------------------------------------------------------
    # Main title
    # -------------------------------------------------------------------------
    if title is None:
        if subj_id is None:
            title = "Theta GED forward models"
        else:
            title = f"Subject {subj_id}: theta GED"

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------
    save_path = None

    if save_dir is not None:

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            if subj_id is None:
                filename = "theta_ged_topographies.png"
            else:
                filename = (
                    f"sub-{int(subj_id):03d}_"
                    "theta_ged_topographies.png"
                )

        save_path = save_dir / filename

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, template_correlations, save_path

# -----------------------------------------------------------------------------
# Subject loop
# -----------------------------------------------------------------------------
for dataset in DATASETS:

    # -------------------------------------------------------------------------
    # Load trial metadata
    # -------------------------------------------------------------------------
    base = str(dataset).split("_cleaned")[0]

    df_trials = pd.read_csv(base + "_erp_trialinfo.csv")

    subj_id = int(df_trials["id"].iloc[0])

    if subj_id in IDS_TO_DROP:
        continue

    print(f"Processing subject {subj_id}")

    # -------------------------------------------------------------------------
    # Load EEG data
    # -------------------------------------------------------------------------
    mat = scipy.io.loadmat(dataset)

    # MATLAB:
    # channels x time x trials
    #
    # Python:
    # trials x channels x time
    erp_data = np.transpose(
        mat["data"],
        [2, 0, 1],
    )

    erp_times = mat["times"].ravel().astype(float)

    erp_times_sec = (
        erp_times / 1000
        if np.nanmax(np.abs(erp_times)) > 20
        else erp_times
    )
    
    # -------------------------------------------------------------------------
    # Theta GED
    # -------------------------------------------------------------------------
    ged_result = run_theta_ged(
        erp_data=erp_data,
        erp_times_sec=erp_times_sec,
        sfreq=SFREQ,
        template=FCZ_TEMPLATE,
    )
    
    plot_ged_topographies(
        ged_result=ged_result,
        info=INFO_ERP,
        template=FCZ_TEMPLATE,
        subj_id=subj_id,
        n_components=20,
        n_cols=5,
        save_dir=PATH_OUT / "ged_theta" / "topographies",
        dpi=150,
        show=False,
    )
    
    # -------------------------------------------------------------------------
    # Extract selected theta-component power
    # -------------------------------------------------------------------------
    theta_component = extract_theta_component_power(
        erp_data=erp_data,
        spatial_filter=ged_result["spatial_filter"],
        sfreq=SFREQ,
        theta_band=THETA_BAND,
    )
    
    theta_component_path = save_theta_component(
        component_result=theta_component,
        ged_result=ged_result,
        times=erp_times_sec,
        trial_metadata=df_trials,
        subj_id=subj_id,
        output_dir=PATH_OUT / "ged_theta" / "components",
        analysis_tmin=GED_TMIN,
        analysis_tmax=GED_TMAX,
    )
        
    print(f"  saved theta component: {theta_component_path}")