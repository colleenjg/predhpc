import numpy as np

from predhpc.util import ext_util, signal_util

WIDTH = 0.5  # symmetical width (m) around PC peak to use for pre/post weight ratio


def get_PC_info(Pyrs, PC_name="PCs", smoothed=False, effective=False, max_recurrence=8):
    """
    get_PC_info(Pyrs)

    Get place cell information from the Pyr. layer.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - PC_name (str, optional): Name of the place cell layer. Default is "PCs".
    - smoothed (bool, optional): Whether to return smoothed weights, if effective is
        False. Default is False.
    - effective (bool, optional): Whether to return effective weights corresponding to
        the output of the Pyramidal layer based on the input weights. Default is False.
    - max_recurrence (int, optional): Maximum number of time get_state() recursively calls
        recurrent inputs (prevents infinite recursion error). Appears to stabilize
        after about 7. Default is 5.

    Returns:
    - PC_weights (np.ndarray): Sorted place cell weights.
    - PC_centres (np.ndarray): Sorted place cell centres.
    - peak_idx (int or None): Index of the place cell weight peak. If all weights are
        equal, returns None.
    """

    if Pyrs.Agent.Environment.D != 1:
        raise NotImplementedError(
            "Place field analysis tools only implemented for 1D environments."
        )

    if hasattr(Pyrs, "SomaCompartment"):
        Pyrs = Pyrs.SomaCompartment

    if PC_name not in Pyrs.inputs.keys():
        raise KeyError(f"PC name '{PC_name}' not found in Pyrs inputs.")

    PCs = Pyrs.inputs["PCs"]["layer"]

    PC_centres = PCs.place_cell_centres[:, 0]
    sorter = np.argsort(PC_centres)
    PC_centres = PC_centres[sorter]

    if effective:
        if smoothed:
            raise ValueError("smoothed and effective cannot both be True.")
        PC_weights = Pyrs.get_state(
            evaluate_at="pos",
            pos=PC_centres.reshape(-1, 1),
            max_recurrence=max_recurrence,
        )[0]

    else:
        PC_weights = Pyrs.inputs["PCs"]["w"][0][sorter]

        if smoothed:
            sigma_in_steps = float(PCs.widths) / np.absolute(np.diff(PC_centres)).mean()
            PC_weights = Pyrs.inputs["PCs"]["w"][0][sorter]
            PC_weights = signal_util.smooth_kernel(
                PC_weights, sigma_in_steps
            )  # change to smooth circularly!

    if not np.isfinite(PC_weights).all():
        peak_idx = None
    elif np.max(PC_weights) == np.min(PC_weights):
        peak_idx = None
    else:
        peak_idx = np.argmax(PC_weights)

    return PC_weights, PC_centres, peak_idx


def get_PC_weight_peak_relative_position(Pyrs, target_position=None):
    """
    get_PC_weight_peak_relative_position(Pyrs, target_position=None)

    Compute the position of the place cell weight peak relative to a target position.
    If the place cell weights are flat, np.nan is returned.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - target_position (float, optional): Target position to compute relative position
        from. If None, the target position is taken from Pyrs.Agent.target_position[0].
        Default is None.

    Returns:
    - peak_rel_pos (float): Relative position of the place cell weight peak.
        If all place cell weights are equal, returns np.nan.
    """

    _, PC_centres, peak_idx = get_PC_info(Pyrs)

    if peak_idx is None:
        peak_rel_pos = np.nan
    else:
        if target_position is None:
            target_position = Pyrs.Agent.target_position[0]
        peak_rel_pos = PC_centres[peak_idx] - target_position
        scale = Pyrs.Agent.Environment.scale
        if peak_rel_pos < -scale / 2:
            peak_rel_pos += scale
        elif peak_rel_pos > scale / 2:
            peak_rel_pos -= scale

    return peak_rel_pos


def get_PC_weight_ratio(Pyrs, width=WIDTH):
    """
    get_PC_weight_ratio(Pyrs)

    Compute the ratio of place cell weights at a symmetrical distance before and after
    the peak weight. If the place cell weights are flat, np.nan is returned.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - width (float, optional): Width (m) around PC peak to use for pre/post weight
        ratio. Default is WIDTH.

    Returns:
    - pre_post_ratio (float): Ratio of place cell weights before and after the peak
        weight (pre/post). If place cell weights are flat, returns np.nan.
    """

    PC_weights, PC_centres, peak_idx = get_PC_info(Pyrs)

    if peak_idx is None:
        pre_post_ratio = np.nan
    else:
        pre_centre = PC_centres[peak_idx] - width / 2
        post_centre = PC_centres[peak_idx] + width / 2

        pre_idx = np.argmin(np.abs(PC_centres - pre_centre))
        post_idx = np.argmin(np.abs(PC_centres - post_centre))

        pre_post_ratio = PC_weights[pre_idx] / PC_weights[post_idx]

    return pre_post_ratio


def compute_PC_FWHM(Pyrs, k=1):
    """
    compute_PC_FWHM(Pyrs)

    Compute the full width at half maximum for a place field. If the place cell
    weights are flat, np.nan is returned.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - k (int, optional): Kernel size for circular smoothing. Default is 1.

    Returns:
    - pre_post_ratio (float): Ratio of place cell weights before and after the peak
        weight (pre/post). If place cell weights are flat, returns np.nan.
    """

    PC_weights, PC_centres, peak_idx = get_PC_info(Pyrs)

    if peak_idx is None:
        FWHM = 0
    else:
        scale = Pyrs.Agent.Environment.scale
        FWHM = signal_util.compute_FWHM(PC_weights, PC_centres, k=k, max_x=scale)

    return FWHM


def compute_BTSP_metrics(Pyrs, t_start=0, bins=21, width=WIDTH, k=1):
    """
    compute_BTSP_metrics(Pyrs)

    Compute BTSP metrics.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - start_time (int, optional): Time from which to gather metrics. Default is 0.
    - bins (int, optional): Number of bins to use for binning positions on linear
        track and counting number of positions a BTSP event occurred in. Default is 21.
    - width (float, optional): Width (m) around PC peak to use for pre/post weight
        ratio. Default is WIDTH.
    - k (int, optional): Kernel size for circular smoothing when computing FWHM.
        Default is 1.

    Returns:
    - BTSP_metrics (dict): Dictionary of BTSP metrics, with keys and values:
        - BTSP_metric: BTSP metric value
    """

    if Pyrs.Agent.Environment.D != 1:
        raise NotImplementedError("BTSP metrics only implemented for 1D environments.")

    num_BTSP_events = Pyrs.SomaCompartment.get_BTSP_counts(t_start=t_start)[0]

    if num_BTSP_events:
        first_BTSP_info = Pyrs.SomaCompartment.get_BTSP_info(
            t_start=t_start, neuron_idx=0
        )
        first_BTSP_time = first_BTSP_info["time"]
        first_BTSP_relative_position = (
            first_BTSP_info["position"][0] - Pyrs.Agent.target_position[0]
        )
        num_BTSP_positions = Pyrs.SomaCompartment.get_nbr_BTSP_position_bins(
            t_start=t_start, bins=bins
        )[0]
    else:
        first_BTSP_time = np.nan
        first_BTSP_relative_position = np.nan
        num_BTSP_positions = 0

    BTSP_ramp_max = Pyrs.SomaCompartment.get_BTSP_ramp_peaks(t_start=t_start)[0]

    PC_weight_ratio_pre_post = get_PC_weight_ratio(Pyrs, width=width)
    PC_weight_peak_relative_position = get_PC_weight_peak_relative_position(Pyrs)

    PC_weight_FWHM = compute_PC_FWHM(Pyrs, k=k)

    BTSP_metrics = {
        "metric/num_BTSP_events": num_BTSP_events,
        "metric/num_BTSP_positions": num_BTSP_positions,
        "metric/first_BTSP_time": first_BTSP_time,
        "metric/first_BTSP_relative_position": first_BTSP_relative_position,
        "metric/max_BTSP_ramp": BTSP_ramp_max,
        "metric/PC_weight_peak_relative_position": PC_weight_peak_relative_position,
        "metric/PC_weight_FWHM": PC_weight_FWHM,
        "metric/PC_weight_ratio_pre_post": PC_weight_ratio_pre_post,
    }

    return BTSP_metrics
