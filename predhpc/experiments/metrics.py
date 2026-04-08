import numpy as np

from predhpc.util import signal_util, gen_util, ext_util

WIDTH = 0.5  # symmetical width (m) around PC peak to use for pre/post weight ratio

SMOOTH_K = 5  # smoothing factor for PF width computation from firingrate history


def get_smoothed_1D_weights(all_weights, PF_centers, PC_widths):
    """
    get_smoothed_1D_weights(all_weights, PF_centers, PC_widths)

    Apply a Gaussian smoothing kernel to 1D weights to take into account the spatial
    structure of the place fields.

    Args:
    - all_weights (2D np.ndarray): Weights from place cells to pyramidal neurons
        (weights x place cells).
    - PF_centers (1D or 2D np.ndarray): Place field centers (place cells x 1).
    - PC_widths (float): Width of the place cells (same unit as PF_centers).

    Returns:
    - sm_weights (2D np.ndarray): Smoothed weights (neurons x place cells).
    - PF_centers (1D np.ndarray): Place field centers (place cells).
    """

    if len(PF_centers.shape) == 2:
        if PF_centers.shape[1] != 1:
            raise NotImplementedError(
                "Expected 1D PF_centers or second dimension to have length 1."
            )
        PF_centers = PF_centers[:, 0]

    sorter = np.argsort(PF_centers)
    PF_centers = PF_centers[sorter]
    all_weights = all_weights[..., sorter]

    dist = np.absolute(np.diff(PF_centers)).mean()

    sigma_in_steps = float(PC_widths) / dist
    sm_weights = list()
    for weights in all_weights.reshape((-1, PF_centers.shape[0])):
        sm_weights.append(
            signal_util.gaussian_smooth_kernel(weights, sigma_in_steps, circular=True)
        )
    sm_weights = np.asarray(sm_weights).reshape(all_weights.shape)

    return sm_weights, PF_centers


def evaluate_PFs(
    Pyrs,
    PC_name="PCs",
    method="weights",
    t_start=None,
    t_end=None,
    chosen_neurons="all",
):
    """
    evaluate_PFs(Pyrs)

    Evaluate the place fields of a pyramidal layer.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - PC_name (str, optional): Name of the place cell layer. Default is "PCs".
    - method (str, optional): Method to use for evaluating place fields.
        "weights" returns the input weights from the place cell layer.
        "applied_weights" calculates Pyramidal neuron activity considering only place
            cell input.
        "smoothed_weights" applies a Gaussian smoothing kernel to the weights to take
            into account the spatial structure of the place fields.
        "history" uses firingrate history to compute place field.
        Default is "weights".
    - t_start (float, optional): Start time for history evaluation. Default is None.
    - t_end (float, optional): End time for history evaluation. Default is None.
    - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".

    Returns:
    - PFs (2D np.ndarray): Place fields (neurons x PF centers).
    - PF_centers (1D or 2D np.ndarray): Center coordinates for each place field
        (place cells x dimensions).
    """

    if gen_util.attribute_type_checker(Pyrs, "TwoCompLayer"):
        Pyrs = Pyrs.ProximalCompartment

    if PC_name not in Pyrs.inputs.keys():
        raise KeyError(f"PC name '{PC_name}' not found in Pyrs inputs.")

    PCs = Pyrs.inputs["PCs"]["layer"]

    PF_centers = PCs.place_cell_centers
    if Pyrs.Environment.D == 1:
        PF_centers = PF_centers[:, 0]
        sorter = np.argsort(PF_centers)
        PF_centers = PF_centers[sorter]
    elif Pyrs.Environment.D == 2:
        if len(PF_centers.shape) != 2 or PF_centers.shape[1] != 2:
            raise ValueError(
                "PF_centers must be a 2D array with shape (num_PF_centers, 2)."
            )
    else:
        raise NotImplementedError(
            f"Expected environment to be 1 or 2D, but found {Pyrs.Environment.D}."
        )

    chosen_neurons = Pyrs.get_chosen_neurons(chosen_neurons=chosen_neurons)

    if method in ["weights", "applied_weights", "smoothed_weights"]:
        PFs = Pyrs.inputs["PCs"]["w"][chosen_neurons]
        if method == "smoothed_weights":
            if Pyrs.Environment.D == 1:
                PFs = PFs[:, sorter]
            else:
                raise NotImplementedError(
                    "Smoothed weights only implemented for 1D environments."
                )
            if PCs.description != "gaussian" or not isinstance(
                PCs.widths, (int, float)
            ):
                raise ValueError(
                    "PCs must have a Gaussian description and a single numeric width "
                    "to use 'smoothed_weights' method."
                )
            PFs, PF_centers = get_smoothed_1D_weights(PFs, PF_centers, PCs.widths)

        if method == "applied_weights":
            PC_inputs = PCs.get_state(evaluate_at="pos", pos=PF_centers.reshape(-1, 1))
            V = np.matmul(PFs, PC_inputs)
            if Pyrs.biases.shape != V.shape:
                Pyrs.biases = Pyrs.biases.reshape((-1, 1))
            V += Pyrs.biases
            PFs = Pyrs.activation_function(V, deriv=False)

    elif method == "history":
        if Pyrs.Environment.D == 1:
            dist = np.absolute(np.diff(PF_centers)).mean()
        elif Pyrs.Environment.D == 2:
            dist = np.inf
            for i in range(2):
                dist = min(
                    dist,
                    np.absolute(np.diff(np.sort(np.unique(PF_centers[:, i])))).min(),
                )
        bin_size = dist / 2
        PFs, PF_centers = Pyrs.get_history_ratemap(
            t_start=t_start,
            t_end=t_end,
            bin_size=bin_size,
            chosen_neurons=chosen_neurons,
        )

    else:
        raise ValueError(f"Unknown method: {method}.")

    return PFs, PF_centers


def get_PF_info(Pyrs, PC_name="PCs", **kwargs):
    """
    get_PF_info(Pyrs)

    Get place field information from the Pyr. layer.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - PC_name (str, optional): Name of the place cell layer. Default is "PCs".

    Keyword args:
    - **kwargs: Keyword arguments passed to evaluate_PFs().

    Returns:
    - PFs (2D np.ndarray): Place field.
    - PF_centers (1D or 2D np.ndarray): Center coordinates for each place field
        (place cells x dimensions).
    - peak_idx (int or None): Index of the place cell weight peak. If all weights are
        equal, returns None.
    """

    PFs, PF_centers = evaluate_PFs(Pyrs, PC_name=PC_name, **kwargs)

    if len(PFs) > 1:
        raise NotImplementedError(
            "Place field analysis for multiple place fields is not implemented."
        )

    PFs = PFs[0]

    if not np.isfinite(PFs).any():
        peak_idx = None
    elif np.nanmax(PFs) == np.nanmin(PFs):
        peak_idx = None
    else:
        peak_idx = np.nanargmax(PFs)

    return PFs, PF_centers, peak_idx


def get_PF_weight_peak_relative_position(Pyrs, target_position=None, **kwargs):
    """
    get_PF_weight_peak_relative_position(Pyrs)

    Compute the position of the place field peak relative to a target position.
    If the place field is flat, np.nan is returned.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - target_position (float, optional): Target position to compute relative position
        from. If None, the target position is taken from Pyrs.Agent.target_position[0].
        Default is None.

    Keyword args:
    - **kwargs: Keyword arguments passed to get_PF_info().

    Returns:
    - peak_rel_pos (float): Relative position of the place field peak.
        If all place field values are equal, returns np.nan.
    """

    _, PF_centers, peak_idx = get_PF_info(Pyrs, **kwargs)

    if peak_idx is None:
        peak_rel_pos = np.nan
    else:
        if target_position is None:
            target_position = Pyrs.Agent.target_position[0]
        peak_rel_pos = PF_centers[peak_idx] - target_position
        scale = Pyrs.Environment.scale
        if peak_rel_pos < -scale / 2:
            peak_rel_pos += scale
        elif peak_rel_pos > scale / 2:
            peak_rel_pos -= scale

    return peak_rel_pos


def get_PF_ratio(Pyrs, width=WIDTH, **kwargs):
    """
    get_PF_ratio(Pyrs)

    Compute the ratio of place field values at a symmetrical distance before and after
    the peak weight. If the place field is flat, np.nan is returned.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - width (float, optional): Width (m) around PC peak to use for pre/post weight
        ratio. Default is WIDTH.

    Keyword args:
    - **kwargs: Keyword arguments passed to get_PF_info().

    Returns:
    - pre_post_ratio (float): Ratio of place field values before and after the peak
        weight (pre/post). If place field values are flat, returns np.nan.
    """

    PF, PF_centers, peak_idx = get_PF_info(Pyrs, **kwargs)

    if peak_idx is None:
        pre_post_ratio = np.nan
    else:
        pre_center = PF_centers[peak_idx] - width / 2
        post_center = PF_centers[peak_idx] + width / 2

        pre_idx = np.argmin(np.abs(PF_centers - pre_center))
        post_idx = np.argmin(np.abs(PF_centers - post_center))

        pre_post_ratio = PF[pre_idx] / PF[post_idx]

    return pre_post_ratio


def compute_PF_width(Pyrs, k=1, prop_peak=signal_util.DFT_PROP_PEAK, **kwargs):
    """
    compute_PF_width(Pyrs)

    Compute the full width at half maximum for a place field. If the place field
    is flat, np.nan is returned.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - k (int, optional): Kernel size for circular smoothing. Default is 1.
    - prop_peak (float, optional): Proportion of peak to use for width calculation.
        Default is signal_util.DFT_PROP_PEAK.

    Keyword args:
    - **kwargs: Keyword arguments passed to get_PF_info().

    Returns:
    - width (float): Width of the place field.
    """

    if Pyrs.Environment.D != 1:
        raise NotImplementedError("PF width only implemented for 1D environments.")

    PF, PF_centers, peak_idx = get_PF_info(Pyrs, **kwargs)

    if peak_idx is None:
        width = 0
    else:
        scale = Pyrs.Environment.scale
        width = signal_util.compute_signal_width(
            PF, PF_centers, prop_peak=prop_peak, k=k, max_x=scale
        )

    return width


def compute_BTSP_metrics(Pyrs, t_start=0, bins=31, width=WIDTH, k=1, **kwargs):
    """
    compute_BTSP_metrics(Pyrs)

    Compute BTSP metrics.

    Args:
    - Pyrs (two_comp_neurons.TwoCompLayer): Pyr. layer.
    - t_start (int, optional): Time from which to gather metrics. Default is 0.
    - bins (int, optional): Number of bins to use for binning positions on linear
        track and counting number of positions a BTSP event occurred in. Default is 31.
    - width (float, optional): Width (m) around PC peak to use for pre/post weight
        ratio. Default is WIDTH.
    - k (int, optional): Kernel size for circular smoothing when computing width.
        Default is 1.

    Keyword args:
    - **kwargs: Keyword arguments passed to get_PF_ratio(),
        get_PF_weight_peak_relative_position() and compute_PF_width().

    Returns:
    - BTSP_metrics (dict): Dictionary of BTSP metrics, with keys and values:
        - BTSP_metric: BTSP metric value
    """

    if Pyrs.Environment.D != 1:
        raise NotImplementedError("BTSP metrics only implemented for 1D environments.")

    num_BTSP_events = Pyrs.ProximalCompartment.get_BTSP_counts(t_start=t_start)[0]

    if num_BTSP_events:
        first_BTSP_info = Pyrs.ProximalCompartment.get_BTSP_info(
            t_start=t_start, neuron_idx=0
        )
        first_BTSP_time = first_BTSP_info["time"]
        first_BTSP_relative_position = (
            first_BTSP_info["position"][0] - Pyrs.Agent.target_position[0]
        )
        num_BTSP_positions = Pyrs.ProximalCompartment.get_num_BTSP_position_bins(
            t_start=t_start, bins=bins
        )[0]
    else:
        first_BTSP_time = np.nan
        first_BTSP_relative_position = np.nan
        num_BTSP_positions = 0

    BTSP_ramp_max = Pyrs.ProximalCompartment.get_BTSP_ramp_peaks(t_start=t_start)[0]

    PC_weight_ratio_pre_post = get_PF_ratio(Pyrs, width=width, **kwargs)
    PC_weight_peak_relative_position = get_PF_weight_peak_relative_position(
        Pyrs, **kwargs
    )

    PC_weight_width = compute_PF_width(Pyrs, k=k, **kwargs)

    norm_values = Pyrs.ProximalCompartment.get_normalization_values(
        "PCs", t_start=t_start
    )[-1]
    max_norm = np.nanmax(norm_values) if len(norm_values) > 0 else np.nan

    activity_98p = Pyrs.ProximalCompartment.get_percentile_firingrates(
        t_start=t_start, percentiles=98
    )[0]

    BTSP_metrics = {
        "metric/num_BTSP_events": num_BTSP_events,
        "metric/num_BTSP_positions": num_BTSP_positions,
        "metric/first_BTSP_time": first_BTSP_time,
        "metric/first_BTSP_relative_position": first_BTSP_relative_position,
        "metric/max_BTSP_ramp": BTSP_ramp_max,
        "metric/PC_weight_peak_relative_position": PC_weight_peak_relative_position,
        "metric/PC_weight_width": PC_weight_width,
        "metric/PC_weight_ratio_pre_post": PC_weight_ratio_pre_post,
        "metric/max_normalization": max_norm,
        "metric/98th_percentile_firingrate": activity_98p,
    }

    return BTSP_metrics


def add_traj_idxs_from_times(data_dict, learner):
    """
    add_traj_idxs_from_times(data_dict, learner)

    Adds trajectory indices to the data dictionary based on time values.

    Args:
    - data_dict (dict): Dictionary containing time values with keys containing
        "_times".
    - learner (Learner): The learner object to get trajectory information from.

    Modifies:
    - data_dict: Adds new keys with trajectory indices corresponding to the time
        values.
    """

    data_dict = data_dict.copy()

    time_keys = [key for key in data_dict.keys() if "_time" in key]
    for key in time_keys:
        traj_key = key.replace("_time", "_traj_idx")
        if traj_key in data_dict.keys():
            continue
        traj_idxs = np.full(np.asarray(data_dict[key]).ravel().shape, np.nan)
        for i, time in enumerate(np.asarray(data_dict[key]).ravel()):
            if np.isnan(time):
                continue
            traj_idxs[i] = learner.Pyrs.Agent.get_trajectory_idx(time=time)
        if isinstance(data_dict[key], np.ndarray):
            data_dict[traj_key] = traj_idxs.reshape(data_dict[key].shape)
        else:
            data_dict[traj_key] = traj_idxs[0]  # single value case

    return data_dict


def gather_PF_info(learner, k=SMOOTH_K, position_name=None, min_total=None):
    """
    gather_PF_info(learner)

    Gathers information about place fields (PFs) from the given learner object in a 1D
    environment using various metrics ("weights", "smoothed_weights", "history").

    Args:
    - learner (Learner): The learner object to gather information from.
    - k (int): The smoothing factor for place field width computation from firingrate
        history. Default is SMOOTH_K.
    - position_name (str, optional): Name of the position to gather visit times for.
        If None, visit times are not gathered. Default is None.
    - min_total (int, optional): Minimum total amount of time (s) for computing PFs
        from history. If None, defaults are used based on environment dimensionality:
        60 sec for 1D and 5 min for 2D. Default is None.

    Returns:
    - PF_info (dict): A dictionary containing gathered PF information:
        - "BTSP_times": Times of applied BTSP events.
        - "BTSP_traj_idxs": Trajectory indices of applied BTSP events.
        - "num_BTSP": Number of recorded BTSP events.
        - "BTSP_applied_times": Times of applied BTSP events.
        - "BTSP_applied_traj_idxs": Trajectory indices of applied BTSP events.
        - "num_BTSP_applied": Number of applied BTSP events.
        - "PC_place_centers": Place cell centers.
        - "PC_weights": Place cell input weights.
        - "PFs": Place fields computed from history.
        - "PF_centers": Place field centers.
        - "PF_times": Times used to compute each place field.
        - "PF_traj_idxs": Trajectory indices corresponding to times used to compute
            each place field.

        if position_name is not None:
        - "visit_times": Times of position visits.
        - "visit_traj_idxs": Trajectory indices of position visits.
        - "num_visits": Number of position visits.

        if 1D environment:
        - "PC_weight_widths": Last place cell input weight widths.
        - "PC_smoothed_weights": Smoothed place cell input weights.
        - "PC_smoothed_weight_widths": Last smoothed place cell input weight widths.
        - "PF_widths": Last place field widths.
    """

    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(learner.Pyrs)

    PF_info = dict()

    # BTSP steps
    for apply in [False, True]:
        apply_str = "_applied" if apply else ""
        BTSP_steps = learner.Pyrs.ProximalCompartment.get_BTSP_steps(
            applied_only=apply, apply_step=apply
        )
        PF_info[f"BTSP{apply_str}_times"] = BTSP_steps * learner.Pyrs.Agent.dt
        PF_info[f"num_BTSP{apply_str}"] = len(BTSP_steps)

    # visit steps
    if position_name is not None:
        visit_steps = learner.Pyrs.Agent.get_position_visits(
            position_name=position_name
        )
        PF_info["visit_times"] = visit_steps * learner.Pyrs.Agent.dt
        PF_info["num_visits"] = len(visit_steps)

    # from input weights
    PF_info["PC_place_centers"] = PCs.place_cell_centers
    PC_weights = learner.get_recorded_weights()["weights"]
    if learner.Pyrs.n == 1:
        PC_weights = PC_weights[:, 0]
    PF_info["PC_weights"] = PC_weights

    # for 1D environments
    if learner.Pyrs.Environment.D == 1:
        next_trajectory = True
        sorter = np.argsort(PF_info["PC_place_centers"][:, 0])
        PF_info["PC_place_centers"] = PF_info["PC_place_centers"][sorter, 0]
        PF_info["PC_weights"] = PF_info["PC_weights"][..., sorter]
        PF_info["PC_weight_widths"] = compute_PF_width(learner.Pyrs, k=1)

        PF_info["PC_smoothed_weights"], _ = get_smoothed_1D_weights(
            PF_info["PC_weights"], PF_info["PC_place_centers"], PCs.widths
        )
        PF_info["PC_smoothed_weight_widths"] = compute_PF_width(
            learner.Pyrs, k=1, method="smoothed_weights"
        )
        min_total = min_total or 60
    else:
        next_trajectory = False
        min_total = min_total or 60 * 5

    # from history
    history_PFs = list()
    PF_shape = None
    PF_times = ext_util.get_times_for_each_BTSP_event(
        learner.Pyrs.ProximalCompartment,
        next_trajectory=next_trajectory,
        min_total=min_total,
        use_nans=True,
    )

    last_idx = None
    for i, (t_start, t_end) in enumerate(PF_times):
        if np.isnan(t_start):
            if PF_shape is None:
                history_PFs.append(None)
            else:
                history_PFs.append(np.full(PF_shape, np.nan))
        else:
            last_idx = i
            PFs, PF_centers = evaluate_PFs(
                learner.Pyrs, method="history", t_start=t_start, t_end=t_end
            )
            if learner.Pyrs.n == 1:
                PFs = PFs[0]
            history_PFs.append(PFs)
            PF_shape = PFs.shape

    if last_idx is None:
        raise RuntimeError("No valid PFs obtained from history evaluation.")

    for i, PF in enumerate(history_PFs):
        if PF is None:
            history_PFs[i] = np.full(PF_shape, np.nan)

    PF_info["PFs"] = np.asarray(history_PFs)
    PF_info["PF_centers"] = PF_centers
    PF_info["PF_times"] = np.asarray(PF_times)

    if learner.Pyrs.Environment.D == 1:
        t_start, t_end = PF_times[last_idx]
        PF_info["PF_widths"] = compute_PF_width(
            learner.Pyrs, k=k, method="history", t_start=t_start, t_end=t_end
        )

    PF_info = add_traj_idxs_from_times(PF_info, learner)

    return PF_info
