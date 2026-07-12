#!/usr/bin/env python3

import argparse
import copy
import time
from pathlib import Path
import warnings

import itertools
from joblib import Parallel, delayed
from matplotlib import animation as mpl_animation
import numpy as np
import ratinabox
from tqdm.auto import tqdm

from predhpc import run_manager, paper_plot_fcts
from predhpc.util import gen_util, params_util, ext_util, plot_util, hyper_util
from predhpc.experiments import metrics

PAPER_SEED = 18

SPEED_MEANS = gen_util.get_rounded_linspace(0.05, 0.4, 29)
SPEED_EXAMPLES = [0.15, 0.25, 0.35]

LANDMARK_SHIFTS = gen_util.get_rounded_linspace(-3.6, 2.4, 61)
SHIFT_EXAMPLES = [1.0, 0, -0.4, -3.0]

COMPARISON_KWARGS = {
    "proximal_BTSP_lr": [0.10, 0.30, 5],
    "inhibitory_input_filter_tau": [0.2, 0.4, 5],
    "inhibitory_weight": [0.5, 2.0, 4],
}


NUM_TRAJ_SPEED = 15
EX_TRAJ_IDX = 8

OPENFIELD_TIME_IN_MIN = 12
OPENFIELD_TIME_AFTER_BTSP_IN_MIN = 10

OPENFIELD_TELEPORT_REPEAT_TIME_IN_MIN = 30

OPENFIELD_MULTITARGET_TIME_IN_MIN = 90
OPENFIELD_MULTITARGET_TIME_AFTER_BTSP_IN_MIN = 15


def suppress_warnings():
    """
    suppress_warnings()

    Suppress expected warnings.
    """

    warnings.filterwarnings("ignore", message="solid 1D boundary", category=UserWarning)
    warnings.filterwarnings(
        "ignore", message="invalid value encountered", category=RuntimeWarning
    )


def get_fig_directory():
    """
    get_fig_directory()

    Get the figure directory path.
    """

    direc = ratinabox.figure_directory

    if direc == "undefined":
        raise OSError(
            "Figure directory is not defined. Please set the figure directory "
            "using ratinabox.set_figure_directory()."
        )

    direc = Path(direc)

    if not direc.is_dir():
        raise OSError(f"Figure directory does not exist: {direc}.")

    return direc


def initialize_paper_parameters(**kwargs):
    """
    initialize_paper_parameters()

    Initializes parameters for paper.

    Keywords args:
    - **kwargs: Parameters passed to paper_plot_fcts.initialize_paper_parameters().
    """

    suppress_warnings()
    paper_plot_fcts.initialize_paper_parameters(**kwargs)


def gather_learner_data(
    learner,
    seed=False,
    k=metrics.SMOOTH_K,
    position_name=None,
    teleport=False,
    **kwargs,
):
    """
    gather_learner_data(learner)

    Gathers data from a learner object into a dictionary.

    Args:
    - learner (Learner): Learner object.
    - seed (bool or int): Whether to record the seed used for the experiment and if
        so, the seed to record. Default is False.
    - k (int): Smoothing factor for measuring place field width from firingrate history,
        used only for 1D environments. Default is metrics.SMOOTH_K.
    - position_name (str, optional): Name of the position to gather data for.
        Default is None.
    - teleport (bool): Whether to gather teleportation data. Default is False.

    Keyword Args:
    - **kwargs: Additional key-value pairs to include in the data dictionary.

    Returns:
    - data_dict (dict): Dictionary containing the gathered data.
    """

    norm_values = learner.Pyrs.ProximalCompartment.get_normalization_values("PCs")[1]

    data_dict = metrics.gather_PF_info(learner, k=k, position_name=position_name)
    data_dict["norm_values"] = norm_values[..., 0]
    data_dict["end_time"] = learner.Agent.t

    for key, value in kwargs.items():
        data_dict[key] = value

    if seed:
        data_dict["seed"] = int(seed)

    if teleport:
        if not hasattr(learner.Agent, "teleportation_df"):
            raise ValueError("Agent does not have teleportation data.")
        data_dict["num_teleportations"] = len(learner.Agent.teleportation_df)
        data_dict["teleportation_times"] = learner.Agent.teleportation_df["time"].values
        data_dict["init_teleport_pairs"] = np.asarray(
            learner.Environment.init_teleport_pairs
        )
        data_dict["horizontal_in_from_left"] = (
            learner.Environment.horizontal_in_from_left
        )

    data_dict = metrics.add_traj_idxs_from_times(data_dict, learner)

    return data_dict


def aggregate_from_data_dicts(data_dicts):
    """
    aggregate_from_data_dicts(data_dicts)

    Aggregates a list of PF data dictionaries into a single data dictionary with
    appropriately padded arrays.

    Args:
    - data_dicts (list of dict): List of data dictionaries to compile with the
        following keys: "PC_place_centers", "PC_weights", "PFs", "PF_centers",
        "PF_times", "PF_traj_idxs", "BTSP_times", "BTSP_traj_idxs", "num_BTSP", and
        optionally "visit_times", "visit_traj_idxs", "num_visits", "norm_values",
        "seeds", "end_time_initial", "teleportation_times", "teleportation_traj_idxs",
        "num_teleportations".
    - seeds (list of int, optional): List of seeds corresponding to each data
        dictionary. Default is None.

    Returns:
    - data_dict (dict): Compiled data dictionary.
    """

    max_num_BTSP = max(data_dict["num_BTSP"] for data_dict in data_dicts)
    BTSP_shape = (len(data_dicts), max_num_BTSP)

    max_num_BTSP_applied = max(
        data_dict["num_BTSP_applied"] for data_dict in data_dicts
    )
    BTSP_applied_shape = (len(data_dicts), max_num_BTSP_applied)
    weights_shape = (len(data_dicts), max_num_BTSP_applied + 1)

    if "visit_times" in data_dicts[0].keys():
        max_num_visits = max(data_dict["num_visits"] for data_dict in data_dicts)
        visit_shape = (len(data_dicts), max_num_visits)

    if "teleportation_times" in data_dicts[0].keys():
        max_num_teleportations = max(
            len(data_dict["teleportation_times"]) for data_dict in data_dicts
        )
        teleportation_shape = (len(data_dicts), max_num_teleportations)

    keys_to_reshape = ["PC_weights", "PC_smoothed_weights", "PFs", "norm_values"]
    for key_start in ["PF", "BTSP", "BTSP_applied", "visit", "teleportation"]:
        for key_end in ["times", "traj_idxs"]:
            keys_to_reshape.append(f"{key_start}_{key_end}")

    data_dict = dict()
    for key in data_dicts[0].keys():
        if key in ["PC_place_centers", "PF_centers"]:
            data_dict[key] = data_dicts[0][key]
        elif key in keys_to_reshape:
            if key in ["visit_times", "visit_traj_idxs"]:
                shape = visit_shape
            elif key in ["teleportation_times", "teleportation_traj_idxs"]:
                shape = teleportation_shape
            elif key in ["BTSP_times", "BTSP_traj_idxs"]:
                shape = BTSP_shape
            elif key in ["BTSP_applied_times", "BTSP_applied_traj_idxs"]:
                shape = BTSP_applied_shape
            else:
                shape = weights_shape

            data_dict[key] = np.full(shape + data_dicts[0][key].shape[1:], np.nan)
            for j, sub_data_dict in enumerate(data_dicts):
                data = sub_data_dict[key]
                data_dict[key][j, : data.shape[0]] = data
        else:
            data_dict[key] = np.asarray(
                [sub_data_dict[key] for sub_data_dict in data_dicts]
            )

    if "seed" in data_dict.keys():
        data_dict["seeds"] = data_dict.pop("seed")

    for key in ["end_time", "end_time_initial"]:
        if key in data_dict.keys():
            data_dict[key.replace("time", "times")] = data_dict.pop(key)

    for key in ["end_traj_idx", "end_traj_idx_initial"]:
        if key in data_dict.keys():
            data_dict[key.replace("idx", "idxs")] = data_dict.pop(key)

    if "landmark_shift" in data_dict.keys():
        data_dict["landmark_shifts"] = data_dict.pop("landmark_shift")

    if "speed_mean" in data_dict.keys():
        data_dict["speed_means"] = data_dict.pop("speed_mean")

    return data_dict


def get_last_PFs_from_data_dict(data_dict, PF_type="history"):
    """
    get_last_PFs_from_data_dict(data_dict)

    Args:
    - data_dict (dict): Data dictionary containing place field information.
    - PF_type (str): Type of place field to retrieve ("history" or "weights").

    Returns:
    - PFs (np.ndarray): Last place fields for the specified type.
    - PF_centers (np.ndarray): Centers of the last place fields.
    """

    if PF_type == "history":
        PF_key = "PFs"
        PF_center_key = "PF_centers"
    elif PF_type == "weights":
        PF_key = "PC_weights"
        PF_center_key = "PC_place_centers"
    else:
        raise ValueError(f"PF type not recognized: {PF_type}.")

    PFs = list()
    for data in data_dict[PF_key]:
        idx = np.where(np.isfinite(data).any(axis=1))[0][-1]
        PFs.append(data[idx])
    PFs = np.asarray(PFs)
    PF_centers = np.asarray(data_dict[PF_center_key])

    return PFs, PF_centers


def log_max_normalization_value(norm_values):
    """
    log_max_normalization_value(norm_values)

    Logs the maximum weight normalization value recorded.

    Args:
    - norm_values (np.ndarray): Array of normalization values to log.
    """

    if isinstance(norm_values, list):
        norm_values = np.concatenate(norm_values)

    finite = np.isfinite(norm_values)
    if finite.any():
        max_norm = np.nanmax(norm_values)
        if max_norm > 1:
            log_str = f"Max. weight normalization value applied: {max_norm:.4f}"
            n = (norm_values[finite] > 1).sum()
            if finite.sum() > 1:
                log_str = f"{log_str} ({n}/{finite.sum()} values > 1)."
            else:
                log_str = f"{log_str}."
        else:
            log_str = (
                "No weight normalization applied "
                f"(max value of {max_norm:.4f} <= 1)."
            )
    else:
        log_str = "No weight normalization values found."
    print(log_str)


def log_num_BTSP_if_above(num_BTSP, above=1, traj_idxs=None, num_traj_total=None):
    """
    log_num_BTSP_if_above(num_BTSP)

    Logs if any number of BTSP events are above a certain threshold.

    Args:
    - num_BTSP (1D np.ndarray): Number of BTSP events recorded.
    - above (int): Threshold value. Default is 1.
    - traj_idxs (list): List of trajectory indices to log. Default is None.
    - num_traj_total (int): Total number of trajectories. Default is None.
    """

    if np.any(num_BTSP > above):
        n_strs = list()
        for n in np.sort(np.unique(num_BTSP)):
            if n > above:
                n_strs.append(f"{n} in {np.sum(num_BTSP == n)}/{len(num_BTSP)}")
        event_str = "event" if above == 1 else "events"
        log_str = f"More than {above} BTSP {event_str}: {', '.join(n_strs)}"
        if traj_idxs is None:
            traj_str = "."
        else:
            traj_idx_min = int(np.nanmin(traj_idxs))
            traj_idx_max = int(np.nanmax(traj_idxs))
            if traj_idx_min != traj_idx_max:
                traj_str = f"btw traj. {traj_idx_min + 1} and {traj_idx_max + 1}"
            else:
                traj_str = f"on traj. {traj_idx_min + 1})."

            if num_traj_total is not None:
                num_traj_min = int(np.nanmin(num_traj_total))
                num_traj_max = int(np.nanmax(num_traj_total))
                if num_traj_min != num_traj_max:
                    traj_total_str = f" of {num_traj_min}-{num_traj_max} total"
                else:
                    traj_total_str = f" of {num_traj_min} total"

            traj_str = f" ({traj_str}{traj_total_str})."

        log_str = f"{log_str}{traj_str}"
        print(log_str)


def log_num_teleportations(num_teleportations):
    """
    log_num_teleportations(num_teleportations)

    Logs the number of teleportation events recorded.

    Args:
    - num_teleportations (1D np.ndarray): Number of teleportation events recorded.
    """

    num_teleports, counts = np.unique(num_teleportations, return_counts=True)
    order = np.argsort(num_teleports)

    n_strs = list()
    for i in order:
        n_str = f"{num_teleports[i]} in {counts[i]}/{len(num_teleportations)}"
        n_strs.append(n_str)

    log_str = f"Teleportation events: {', '.join(n_strs)}."
    print(log_str)


def estimate_steps_per_linear_track_trajectory(Ag):
    """
    estimate_steps_per_linear_track_trajectory(Ag)

    Estimates the number of steps per trajectory in a linear track environment based on
    the agent's speed and the environment's scale.

    Args:
    - Ag (Agent): Agent object for which to estimate steps per trajectory.

    Returns:
    - avg_steps_per_traj (int): Estimated average number of steps per trajectory.
    """

    if Ag.speed_mean == 0:
        raise ValueError(
            "Agent speed mean must be non negative to estimate steps per trajectory."
        )

    avg_steps_per_traj = int(
        Ag.Environment.scale / (Ag.speed_mean * Ag.dt) + Ag.wait_after_trajectory
    )

    return avg_steps_per_traj


def get_linear_track_Pyrs(
    scale=params_util.SCALE_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD_LINEAR,
    wait_after_trajectory=0,
    log_BTSP=True,
    seed=True,
    **Pyr_kwargs,
):
    """
    get_linear_track_Pyrs()

    Initializes Pyr parameters for linear environment.

    Args:
    - scale (float): Scale of the environment. Default is params_util.SCALE_LINEAR.
    - speed_mean (float): Mean speed of the agent. Default is
        params_util.SPEED_MEAN_LINEAR.
    - speed_std (float): Standard deviation of the agent's speed. Default is
        params_util.SPEED_STD_LINEAR.
    - wait_after_trajectory (int): Number of steps to wait after completing a
        trajectory. Default is 0.
    - log_BTSP (bool): Whether to log BTSP events. Default is True.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword Args:
    - **Pyr_kwargs: Additional keyword arguments passed to
        params_util.get_Pyr_params().

    Returns:
    - Pyrs (Pyr): Pyr object initialized with the specified parameters.
    """

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    env_params = params_util.get_env_params(
        environment="linear",
        scale=scale,
        init_env_object_prop=params_util.REL_ENV_OBJECT_POS,
    )

    agent_params = params_util.get_agent_params(
        environment="linear",
        scale=scale,
        speed_mean=speed_mean,
        speed_std=speed_std,
        wait_after_trajectory=wait_after_trajectory,
    )

    PC_params = params_util.get_PC_params(
        environment="linear",
    )

    Pyr_params = params_util.get_Pyr_params(
        environment="linear", log_BTSP=log_BTSP, **Pyr_kwargs
    )

    Obj_params = params_util.get_Obj_params(
        environment="linear",
    )

    Pyrs = run_manager.init_env_objects(
        env_params=env_params,
        agent_params=agent_params,
        PC_params=PC_params,
        Pyr_params=Pyr_params,
        Obj_params=Obj_params,
        environment="linear",
        plot=False,
    )

    return Pyrs


def run_linear_track(
    Pyrs_or_learner=None,
    time_in_min_can_stop=None,
    num_target_reaches_can_stop=None,
    num_traj_can_stop=4,
    BTSP_on=None,
    min_traj_after_BTSP=2,
    seed=True,
    inhibition="balanced",
    factor=2.0,
    no_logs=False,
    **kwargs,
):
    """
    run_linear_track()

    Runs a linear environment with the specified Pyr parameters.

    Args:
    - Pyrs_or_learner (Pyr or Learner, optional): Pyr or Learner object with
        initialized parameters. If None, a new Pyr object is created with default
        parameters.
    - num_traj_can_stop (int or None, optional): Number of trajectories after which
        learning can stop. If specified, it overrides time_in_min_can_stop. Note that
        additional criteria (minimum number of BTSP events, etc.) may prolong learning.
        Default is 4.
    - num_target_reaches_can_stop (int or None, optional): Number of target reaches
        after which early stopping can occur. Default is None.
    - BTSP_on (int): Trajectory on which to turn on BTSP. 1 for first trajectory.
        Default is None.
    - min_traj_after_BTSP (int): Minimum number of trajectories to complete after
        turning on BTSP before stopping. Ignored if inhibition is "insufficient".
        Default is 2.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - inhibition (str): Type of inhibition to apply. Options are "balanced",
        "excessive", or "insufficient". Default is "balanced".
    - factor (float): Factor by which to adjust inhibitory weight for "excessive" or
        "insufficient" inhibition. Default is 2.0.
    - no_logs (bool): Whether to disable logging. Default is False.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    """

    start_time = time.perf_counter()

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    if Pyrs_or_learner is None:
        if inhibition in ["balanced", "excessive", "insufficient"]:
            inhibitory_weight = params_util.get_Pyr_params()["inhibitory_weight"]
            if inhibition == "excessive":
                inhibitory_weight *= factor
            elif inhibition == "insufficient":
                inhibitory_weight /= factor
                min_traj_after_BTSP = 0
            if inhibition != "balanced":
                print(
                    f"Using {inhibition} inhibition (weight: {inhibitory_weight:.2f})."
                )
        else:
            raise ValueError(f"Unknown inhibition type: {inhibition}.")

        Pyrs_or_learner = get_linear_track_Pyrs(
            seed=False,
            wait_after_trajectory=params_util.WAIT_LINEAR,
            inhibitory_weight=inhibitory_weight,
        )
    if gen_util.attribute_type_checker(Pyrs_or_learner, "Learner"):
        Pyrs = Pyrs_or_learner.Pyrs
    else:
        Pyrs = Pyrs_or_learner

    num_steps_can_stop = None
    if time_in_min_can_stop is not None:
        num_steps_can_stop = int(time_in_min_can_stop * 60 / Pyrs.Agent.dt)

    min_steps_after_BTSP = 0
    if min_traj_after_BTSP:
        avg_steps_per_traj = estimate_steps_per_linear_track_trajectory(Pyrs.Agent)
        min_steps_after_BTSP = int(min_traj_after_BTSP * avg_steps_per_traj)

    learner = run_manager.learn_1D_BTSP(
        Pyrs_or_learner,
        BTSP_on=BTSP_on,
        num_steps_can_stop=num_steps_can_stop,
        num_traj_can_stop=num_traj_can_stop,
        num_target_reaches_can_stop=num_target_reaches_can_stop,
        min_steps_after_BTSP=min_steps_after_BTSP,
        plot=False,
        no_logs=no_logs,
        **kwargs,
    )

    if not no_logs:
        gen_util.get_duration_str(start_time, log=True)

    return learner


def plot_linear_track(
    learner=None,
    num_traj_can_stop=4,
    inhibition="balanced",
    factor=1.8,
    plot_type="summary",
    **kwargs,
):
    """
    plot_linear_track()

    Produces plots for a linear experiment.

    Args:
    - learner (Learner): Learner object.
    - num_traj_can_stop (int or None, optional): Number of trajectories after which
        learning can stop. Default is 4.
    - inhibition (str): Type of inhibition to apply. Options are "balanced",
        "excessive", or "insufficient". Default is "balanced".
    - factor (float): Factor by which to adjust inhibitory weight for "excessive" or
        "insufficient" inhibition. Default is 1.8.
    - plot_type (str): Type of plot to produce. Options are "summary",
        "neural_activity", "place_fields", or "binned_rates". Default is "summary".

    Keywords args:
    - **kwargs: Additional keyword arguments passed to the plotting functions.

    Returns:
    - ax (plt.Axes or 1D np.ndarray of plt.Axes): Subplots with linear data plotted.
    """

    if plot_type in ["environment", "BTSP_kernel"]:
        if learner is None:
            Pyrs = get_linear_track_Pyrs(
                wait_after_trajectory=params_util.WAIT_LINEAR,
            )
        else:
            Pyrs = learner.Pyrs

    elif learner is None:
        learner = run_linear_track(
            num_traj_can_stop=num_traj_can_stop,
            inhibition=inhibition,
            factor=factor,
        )

    if plot_type == "environment":
        ax = paper_plot_fcts.plot_linear_track_environment(
            Pyrs.Agent.Environment, **kwargs
        )
    elif plot_type == "BTSP_kernel":
        ax = paper_plot_fcts.plot_BTSP_kernel(Pyrs, **kwargs)
    elif plot_type == "summary":
        ax = paper_plot_fcts.plot_linear_track_summary(learner, **kwargs)
    elif plot_type == "neural_activity":
        ax = paper_plot_fcts.plot_linear_track_neural_activity(learner, **kwargs)
    elif plot_type == "place_fields":
        ax = paper_plot_fcts.plot_linear_track_place_fields(learner, **kwargs)
    elif plot_type == "binned_rates":
        ax = paper_plot_fcts.plot_linear_track_binned_rates(learner, **kwargs)
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def run_linear_track_speed(
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD_LINEAR,
    test_speed_mean=None,
    test_speed_std=None,
    num_traj_can_stop=NUM_TRAJ_SPEED,
    wait_after_trajectory=0,
    min_traj_after_BTSP=10,
    k=metrics.SMOOTH_K,
    no_logs=True,
    return_data_dict=True,
    seed=True,
    **Pyrs_kwargs,
):
    """
    run_linear_track_speed()

    Runs and collects data for a single linear speed experiment.

    Args:
    - speed_mean (float): Mean speed for the experiment.
        Default is params_util.SPEED_MEAN_LINEAR.
    - speed_std (float): Standard deviation of speed for the experiment.
        Default is params_util.SPEED_STD_LINEAR.
    - test_speed_mean (float, optional): Mean speed to switch to after initial learning
        phase. If None, speed is not changed. Default is None.
    - test_speed_std (float, optional): Standard deviation of speed to switch to after
        initial learning phase. If None, speed is not changed. Default is None.
    - num_traj_can_stop (int): Number of trajectories after which learning may stop.
        Default is NUM_TRAJ_SPEED.
    - wait_after_trajectory (int): Number of steps to wait after completing a
        trajectory. Default is 0.
    - min_traj_after_BTSP (int or None): Minimum number of trajectories to complete
        after a BTSP event. Default is 10.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is metrics.SMOOTH_K.
    - no_logs (bool): Whether to disable logging. Default is True.
    - return_data_dict (bool): Whether to return a data dictionary containing the
        results of the experiment. Default is True.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword Args:
    - **Pyrs_kwargs: Additional keyword arguments passed to get_linear_track_Pyrs().

    Returns:
    - learner (Learner): The learner object after running the experiment.
    if return_data_dict:
    - data_dict (dict): Dictionary containing the results of the experiment under keys:
        - "speed_mean": Mean speed for the experiment.
        - "PC_place_centers": Place cell centers.
        - "PC_weights": Place cell input weights.
        - "PC_weight_widths": Last place cell input weight widths.
        - "PC_smoothed_weights": Smoothed place cell input weights.
        - "PC_smoothed_weight_widths": Last smoothed place cell input weight widths.
        - "PFs": Place fields computed from history.
        - "PF_times": Times used to compute each place field.
        - "PF_traj_idxs": Trajectory indices corresponding to times used to compute
            each place field.
        - "PF_centers": Place field centers.
        - "PF_widths": Last place field widths.
        - "BTSP_times": Times of BTSP events.
        - "BTSP_traj_idxs": Trajectory indices of BTSP events.
        - "num_BTSP": Number of BTSP events.
        - "BTSP_applied_traj_idxs": Trajectory indices of applied BTSP events.
        - "num_BTSP_applied": Number of applied BTSP events.
        - "end_time_initial": End time after initial learning phase.
        - "end_traj_idx_initial": End trajectory index after initial learning phase.
        - "end_time": End time of the experiment.
        - "end_traj_idx": End trajectory index of the experiment.
        - "norm_values": Weight normalization values used.
        if seed:
        - "seed": Seed for the experiment.
    """

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    Pyrs = get_linear_track_Pyrs(
        speed_mean=speed_mean,
        speed_std=speed_std,
        log_BTSP=False,
        wait_after_trajectory=wait_after_trajectory,
        seed=False,
        **Pyrs_kwargs,
    )

    initial_min_traj_after_BTSP = min_traj_after_BTSP
    if test_speed_mean is not None or test_speed_std is not None:
        initial_min_traj_after_BTSP = 0

    learner = run_linear_track(
        Pyrs,
        num_traj_can_stop=num_traj_can_stop,
        num_target_reaches_can_stop=None,
        min_traj_after_BTSP=initial_min_traj_after_BTSP,
        no_logs=no_logs,
        seed=False,
    )

    num_BTSP_applied = len(
        Pyrs.ProximalCompartment.get_BTSP_steps(applied_only=True, apply_step=True)
    )

    if num_BTSP_applied == 0:
        raise RuntimeError("No BTSP occurred.")

    kwargs = dict()
    if test_speed_mean is not None or test_speed_std is not None:
        kwargs["end_time_initial"] = learner.Agent.t
        learner.Agent.set_speed(mean=test_speed_mean, std=test_speed_std)

        num_prev_traj_compl = len(learner.Agent.get_completed_trajectory_df())
        learner = run_linear_track(
            learner,
            num_traj_can_stop=num_prev_traj_compl + num_traj_can_stop,
            num_target_reaches_can_stop=None,
            min_traj_after_BTSP=min_traj_after_BTSP,
            no_logs=no_logs,
            seed=False,
        )

    if return_data_dict:
        data_dict = gather_learner_data(
            learner,
            seed=seed,
            k=k,
            speed_mean=speed_mean,
            **kwargs,
        )

        return learner, data_dict

    else:
        return learner


def run_linear_track_speeds(
    seed=True,
    num_traj_can_stop=NUM_TRAJ_SPEED,
    min_traj_after_BTSP=10,
    num_repeats=1,
    k=metrics.SMOOTH_K,
    num_jobs=1,
):
    """
    run_linear_track_speeds()

    Runs a linear environment with varying speeds and collects data on place field
    widths and weights.

    Args:
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, a randomly selected seed is used for each
        repeat. Default is True.
    - num_traj_can_stop (int or None, optional): Number of trajectories after which
        learning can stop. Default is NUM_TRAJ_SPEED.
    - min_traj_after_BTSP (int or None): Minimum number of trajectories to complete
        after a BTSP event. Default is 10.
    - num_repeats (int): Number of repeats for the experiment. Default is 1.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is metrics.SMOOTH_K.
    - num_jobs (int): Number of parallel jobs to run. Default is 1.

    Returns:
    - speed_data (dict): Dictionary containing:
        - "speed_means" (1D np.ndarray): Array of speed means used in the experiment.
        - "PC_place_centers" (1D np.ndarray): Array of place cell centers.
        - "PC_weights" (3D np.ndarray): Array of place cell input weights with shape
            (speeds, weights, centers).
        - "PC_weight_widths" (1D np.ndarray): Array of last place cell input weight
            widths.
        - "PC_smoothed_weights" (3D np.ndarray): Array of smoothed place cell input
            weights with shape (speeds, weights, centers).
        - "PC_smoothed_weight_widths" (1D np.ndarray): Array of smoothed last place
            cell input weight widths.
        - "PF_centers" (1D np.ndarray): Array of place field centers.
        - "PFs" (3D np.ndarray): Array of place fields computed from history with shape
            (speeds, fields, centers).
        - "PF_widths" (1D np.ndarray): Array of last place field widths.
        - "PF_times" (3D np.ndarray): Array of times used to compute each place field
            with shape (speeds, fields, 2).
        - "PF_traj_idxs" (3D np.ndarray): Array of trajectory indices corresponding to
            times used to compute each place field with shape (speeds, fields, 2).
        - "BTSP_times" (2D np.ndarray): Array of BTSP event times with shape
            (speeds, events).
        - "BTSP_traj_idxs" (2D np.ndarray): Array of trajectory indices of BTSP events
            with shape (speeds, events).
        - "num_BTSP" (1D np.ndarray): Number of BTSP events recorded for each speed.
        - "BTSP_applied_times" (2D np.ndarray): Array of BTSP event application times
            with shape (speeds, events).
        - "BTSP_applied_traj_idxs" (2D np.ndarray): Array of trajectory indices of
            applied BTSP events with shape (speeds, events).
        - "num_BTSP_applied" (1D np.ndarray): Number of BTSP events applied for each
            speed.
        - "norm_values" (2D np.ndarray): Weight normalization values used for each speed.
        - "end_times_initial" (1D np.ndarray): End time after initial learning phase
            for each speed.
        - "end_traj_idxs_initial" (1D np.ndarray): End trajectory index after initial
            learning phase for each speed.
        - "end_times" (1D np.ndarray): End time of the experiment for each speed.
        - "end_traj_idxs" (1D np.ndarray): End trajectory index of the experiment for
            each speed.
        - "seeds" (1D np.ndarray): Array of seeds for each speed.
    """

    speed_means = SPEED_MEANS

    # product of means and seeds
    total = num_repeats * len(speed_means)
    num_jobs = gen_util.get_num_jobs(num_jobs, num_tasks=total)

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        seeds = np.arange(seed, seed + num_repeats)
    else:
        seeds = np.sort(np.random.choice(10000, size=num_repeats, replace=False))

    iterations = itertools.product(speed_means, seeds)

    kwargs = {
        "speed_std": 0,
        "num_traj_can_stop": num_traj_can_stop,
        "min_traj_after_BTSP": min_traj_after_BTSP,
        "test_speed_mean": params_util.SPEED_MEAN_LINEAR,
        "test_speed_std": params_util.SPEED_MEAN_LINEAR,
        "k": k,
        "no_logs": True,
    }

    if num_jobs > 1:
        outputs = Parallel(n_jobs=num_jobs)(
            delayed(run_linear_track_speed)(speed_mean=speed_mean, seed=seed, **kwargs)
            for speed_mean, seed in tqdm(iterations, total=total)
        )
        _, speed_dicts = zip(*outputs)
    else:
        speed_dicts = list()
        for speed_mean, seed in tqdm(iterations, total=total):
            _, speed_dict = run_linear_track_speed(
                speed_mean=speed_mean, seed=seed, **kwargs
            )
            speed_dicts.append(speed_dict)

    speed_data = aggregate_from_data_dicts(speed_dicts)

    return speed_data


def plot_linear_track_speed_PFs(
    speed_data=None,
    examples=SPEED_EXAMPLES,
    PF_type="history",
    plot_type="all",
    seed=True,
    **kwargs,
):
    """
    plot_linear_track_speed_PFs()

    Plots place fields for different speeds on the linear track.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see run_linear_track_speeds()). If not provided, data is loaded or experiment
        is run from scratch. Default is None.
    - examples (list): List of example speed means. Default is SPEED_EXAMPLES.
    - PF_type (str): PF type to plot. Default is "history".
    - plot_type (str): Type of plot to produce. Options are "examples" or "all".
        Default is "all".
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keywords args:
    - **kwargs: Additional keyword arguments passed to the plotting functions.

    Returns:
    - ax (plt.Axes or np.ndarray of plt.Axes): Subplots with place fields data plotted.
    """

    if speed_data is None:
        speed_data = run_linear_track_fct("speeds", overwrite=False, seed=seed)

    if plot_type == "examples":
        keep_seed = speed_data["seeds"].min()
        for key, vals in [("seeds", [keep_seed]), ("speed_means", examples)]:
            speed_data = gen_util.get_filtered_np_data_dict(
                speed_data,
                key,
                values=vals,
                skip_keys=["PF_centers", "PC_place_centers"],
            )

        Pyrs = get_linear_track_Pyrs()
        _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)  # to add plot markers

        k = metrics.SMOOTH_K if PF_type == "history" else 1

        ax = paper_plot_fcts.plot_linear_track_speed_PF_examples(
            speed_data, Ag=Ag, PF_type=PF_type, k=k, **kwargs
        )
    elif plot_type == "all":
        ax = paper_plot_fcts.plot_linear_track_speed_PF_widths(
            speed_data, mark_examples=examples, PF_type=PF_type, **kwargs
        )
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def run_linear_track_shift(
    learner=None,
    landmark_shift=0,
    i=0,
    speed_std=0,
    num_traj_can_stop=6,
    wait_after_trajectory=0,
    min_traj_after_BTSP=6,
    k=metrics.SMOOTH_K,
    no_logs=True,
    return_data_dict=True,
    seed=True,
):
    """
    run_linear_track_shift()

    Runs and collects data for a single linear speed experiment.

    Args:
    - learner (Learner, optional): Learner object. If None, new learner object is
        created and run before shift is evaluated. Default is None.
    - landmark_shift (float): Landmark object shift for the experiment. Default is 0.
    - i (int): Index for the experiment run. Default is 0.
    - speed_std (float): Standard deviation of speed for the experiment.
        Default is 0.
    - num_traj_can_stop (int): Number of trajectories after which learning may stop,
        before and after the shift. Default is 6.
    - wait_after_trajectory (int): Number of steps to wait after completing a
        trajectory. Default is 0.
    - min_traj_after_BTSP (int): Minimum number of trajectories to complete after
        a BTSP event occurs. Default is 6.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is metrics.SMOOTH_K.
    - no_logs (bool): Whether to disable logging. Default is True.
    - return_data_dict (bool): Whether to return a data dictionary containing the
        results of the experiment. Default is True.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Returns:
    - learner (Learner): The learner object after running the experiment.
    if return_data_dict:
    - data_dict (dict): Dictionary containing the results of the experiment under keys:
        - "landmark_shift": Landmark object shift for the experiment.
        - "PC_place_centers": Place cell centers.
        - "PC_weights": Place cell input weights.
        - "PC_weight_widths": Last place cell input weight widths.
        - "PC_smoothed_weights": Smoothed place cell input weights.
        - "PC_smoothed_weight_widths": Last smoothed place cell input weight widths.
        - "PFs": Place fields computed from history.
        - "PF_centers": Place field centers.
        - "PF_widths": Last place field widths.
        - "num_BTSP_applied": Number of BTSP events that were applied in total for the
            neuron layer.
        - "norm_values": Weight normalization values used.
        - "end_time_initial": End time after initial learning phase.
        - "end_time": End time of the experiment.
        if seed:
        - "seed": Seed for the experiment.
    """

    if seed:
        seed = PAPER_SEED + i
        gen_util.seed_all(seed)

    if learner is None:
        learner = run_linear_track_speed(
            speed_mean=params_util.SPEED_MEAN_LINEAR,
            speed_std=speed_std,
            num_traj_can_stop=num_traj_can_stop,
            wait_after_trajectory=wait_after_trajectory,
            min_traj_after_BTSP=min_traj_after_BTSP,
            k=k,
            no_logs=no_logs,
            return_data_dict=False,
            seed=False,
        )

    num_BTSP_applied = len(
        learner.Pyrs.ProximalCompartment.get_BTSP_steps(
            applied_only=True, apply_step=True
        )
    )
    if num_BTSP_applied != 1:
        raise RuntimeError("Learner does not have exactly one BTSP event.")

    end_time_initial = learner.Agent.t

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Target position"
        )
        learner.Agent.shift_target_position(landmark_shift)

    num_prev_traj_compl = learner.Agent.get_num_completed_trajectories()
    run_linear_track(
        learner,
        num_traj_can_stop=num_prev_traj_compl + num_traj_can_stop,
        min_traj_after_BTSP=min_traj_after_BTSP,
        no_logs=no_logs,
        seed=False,
    )

    num_BTSP_applied = len(
        learner.Pyrs.ProximalCompartment.get_BTSP_steps(
            applied_only=True, apply_step=True
        )
    )

    if num_BTSP_applied not in [1, 2]:
        raise RuntimeError(
            "Expected exactly one or two BTSP events to occur, "
            f"but found {num_BTSP_applied}."
        )

    if return_data_dict:
        data_dict = gather_learner_data(
            learner,
            seed=seed,
            k=k,
            landmark_shift=landmark_shift,
            end_time_initial=end_time_initial,
        )

        return learner, data_dict

    else:
        return learner


def run_linear_track_shifts(
    seed=True, num_traj_can_stop=6, num_repeats=1, k=metrics.SMOOTH_K, num_jobs=1
):
    """
    run_linear_track_shifts()

    Runs a linear environment with varying landmark position shifts and collects data
    on place field widths and weights.

    Args:
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - num_traj_can_stop (int, optional): Number of trajectories after which learning can
        stop. If specified, it overrides time_in_min_can_stop. Note that additional
        criteria (minimum number of BTSP events, etc.) may
        prolong learning. Default is 6.
    - num_repeats (int): Number of repeats for the experiment. Default is 1.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is metrics.SMOOTH_K.
    - num_jobs (int): Number of parallel jobs to run. Default is 1.

    Returns:
    - shift_data (dict): Dictionary containing:
        - "landmark_shifts" (1D np.ndarray): Array of landmark position shifts used in the
            experiment.
        - "PC_place_centers" (1D np.ndarray): Array of place cell centers.
        - "PC_weights" (3D np.ndarray): Array of place cell input weights with shape
            (shifts, weights, centers).
        - "PC_weight_widths" (1D np.ndarray): Array of last place cell input weight
            widths.
        - "PC_smoothed_weights" (3D np.ndarray): Array of smoothed place cell input
            weights with shape (shifts, weights, centers).
        - "PC_smoothed_weight_widths" (1D np.ndarray): Array of smoothed last place
            cell input weight widths.
        - "PF_centers" (1D np.ndarray): Array of place field centers.
        - "PFs" (3D np.ndarray): Array of place fields computed from history with shape
            (shifts, fields, centers).
        - "PF_widths" (1D np.ndarray): Array of last place field widths.
        - "PF_times" (3D np.ndarray): Array of start and end times for place fields
            computed from history with shape (shifts, fields, 2).
        - "PF_traj_idxs" (3D np.ndarray): Array of trajectory indices corresponding to
            times used to compute each place field with shape (shifts, fields, 2).
        - "BTSP_times" (2D np.ndarray): Array of BTSP event times with shape
            (shifts, events).
        - "BTSP_traj_idxs" (2D np.ndarray): Array of trajectory indices of BTSP events
            with shape (shifts, events).
        - "num_BTSP" (1D np.ndarray): Number of BTSP events recorded for each shift.
        - "BTSP_applied_times" (2D np.ndarray): Array of BTSP event application times
            with shape (shifts, events).
        - "BTSP_applied_traj_idxs" (2D np.ndarray): Array of trajectory indices of
            applied BTSP events with shape (shifts, events).
        - "num_BTSP_applied" (1D np.ndarray): Number of BTSP events applied for each
            shift.
        - "end_times_initial" (1D np.ndarray): End time after initial learning phase
            for each speed.
        - "end_traj_idxs_initial" (1D np.ndarray): End trajectory index after initial
            learning phase for each speed.
        - "end_times" (1D np.ndarray): End time of the experiment for each shift.
        - "end_traj_idxs" (1D np.ndarray): End trajectory index of the experiment for
            each shift.
        - "norm_values" (2D np.ndarray): Weight normalization values used for each speed.
        - "seeds" (1D np.ndarray): Array of seeds for each shift.
    """

    landmark_shifts = LANDMARK_SHIFTS

    # product of means and seeds
    total = num_repeats * len(landmark_shifts)
    num_jobs = gen_util.get_num_jobs(num_jobs, num_tasks=total)
    iterations = itertools.product(landmark_shifts, range(num_repeats))

    kwargs = {
        "num_traj_can_stop": num_traj_can_stop,
        "k": k,
        "no_logs": True,
    }

    learner, initial_shift_dict = run_linear_track_speed(
        speed_mean=params_util.SPEED_MEAN_LINEAR, speed_std=0, seed=seed, **kwargs
    )
    kwargs["seed"] = False

    if initial_shift_dict["num_BTSP_applied"] != 1:
        raise RuntimeError("Initial run did not produce exactly one BTSP event.")

    if num_jobs > 1:
        outputs = Parallel(n_jobs=num_jobs)(
            delayed(run_linear_track_shift)(
                landmark_shift=landmark_shift, i=i, learner=learner, **kwargs
            )
            for landmark_shift, i in tqdm(iterations, total=total)
        )
        _, shift_dicts = zip(*outputs)
    else:
        shift_dicts = list()
        for landmark_shift, i in tqdm(iterations, total=total):
            _, shift_dict = run_linear_track_shift(
                landmark_shift=landmark_shift,
                i=i,
                learner=copy.deepcopy(learner),
                **kwargs,
            )
            shift_dicts.append(shift_dict)

    shift_dict = aggregate_from_data_dicts(shift_dicts)

    return shift_dict


def plot_linear_track_shift_PFs(
    shift_data=None, examples=SHIFT_EXAMPLES, plot_cmap=False, plot_type="all", **kwargs
):
    """
    plot_linear_track_shift_PFs()

    Plots place fields for different landmark shifts on the linear track.

    Args:
    - shift_data (dict): Dictionary containing landmark shift-related data
        (see run_linear_track_shifts()). If not provided, data is loaded or experiment
        is run from scratch. Default is None.
    - examples (list): List of example landmark shifts. Default is SHIFT_EXAMPLES.
    - plot_cmap (bool): Whether to plot the place field colormap instead of a peak shift
        plot. Default is False.
    - plot_type (str): Type of plot to produce. Options are "all" or "examples".
        Default is "all".

    Keyword args:
    - **kwargs: Keyword arguments passed to plotting functions.

    Returns:
    - ax (plt.Axes or np.ndarray of plt.Axes): Subplots with linear shift PF data
        plotted.
    """

    if shift_data is None:
        shift_data = run_linear_track_fct("shifts", overwrite=False)

    Pyrs = get_linear_track_Pyrs()
    _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)  # to add plot markers

    if plot_type == "examples":
        start, end = gen_util.get_value_index_range(
            shift_data["num_BTSP_applied"], 1, single_range_only=True
        )

        one_BTSP_pos_range = [
            shift_data["landmark_shifts"][start] + Ag.target_position[0],
            shift_data["landmark_shifts"][end - 1] + Ag.target_position[0],
        ]

        shift_data = gen_util.get_filtered_np_data_dict(
            shift_data,
            "landmark_shifts",
            values=examples,
            skip_keys=["PF_centers", "PC_place_centers"],
        )

        ax = paper_plot_fcts.plot_linear_track_shift_PF_examples(
            shift_data,
            Ag=Ag,
            plot_cmap=plot_cmap,
            mark_pos_range=one_BTSP_pos_range,
            **kwargs,
        )

    elif plot_type == "all":
        ax = paper_plot_fcts.plot_landmark_shift_PFs(
            shift_data, Ag=Ag, mark_examples=examples, plot_cmap=plot_cmap, **kwargs
        )

    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def animate_linear_track_shift(
    landmark_shift=SHIFT_EXAMPLES[2],
    speed_std=params_util.SPEED_STD_LINEAR,
    wait_after_trajectory=params_util.WAIT_LINEAR,
    fps=8,
    speed_up=5,
    seed=True,
    **kwargs,
):
    """
    animate_linear_track_shift()

    Creates an animation of a linear track shift experiment.

    Args:
    - landmark_shift (float): Landmark object shift for the experiment.
        Default is SHIFT_EXAMPLES[2].
    - speed_std (float): Standard deviation of speed for the experiment.
        Default is params_util.SPEED_STD_LINEAR.
    - wait_after_trajectory (int): Number of steps to wait after completing a
    trajectory. Default is params_util.WAIT_LINEAR.
    - fps (int): Frames per second for the animation. Default is 8.
    - speed_up (int): Factor by which to speed up the animation. Default is 3.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to the animation function.

    Returns:
    - anim (matplotlib.animation.FuncAnimation): Animation object for the linear
        shift experiment.
    """

    learner, data_dict = run_linear_track_shift(
        landmark_shift=landmark_shift,
        i=0,
        speed_std=speed_std,
        num_traj_can_stop=3,
        wait_after_trajectory=wait_after_trajectory,
        min_traj_after_BTSP=0,
        k=1,
        no_logs=True,
        seed=seed,
    )

    landmark_shift = data_dict["landmark_shift"]
    shift_time = data_dict["end_time_initial"] / 60
    addendum = f"{landmark_shift:.2f}m shift at {shift_time:.2f} min"

    anim = paper_plot_fcts.animate_linear(
        learner,
        fps=fps,
        speed_up=speed_up,
        addendum=addendum,
        plot_positions=["landmark"],
        reached_only=False,  # plot visits too
        **kwargs,
    )

    return anim


def run_linear_track_fct(fct_name="speeds", overwrite=False, seed=True, num_jobs=1):
    """
    run_linear_track_fct()

    Runs a specified linear function (either 'speeds' or 'shifts'),
    loading an existing data dictionary if it exists or rerunning the experiment.

    Args:
    - fct_name (str): Name of the function to run. Options are 'speeds' or
        'shifts'. Default is 'speeds'.
    - overwrite (bool): Whether to overwrite existing data. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, seed run_linear_track_speeds() or
        run_linear_track_shifts() for details. Default is True.

    Returns:
    - data_dict (dict): Dictionary containing the results of the experiment.
    """

    seed_str = ""
    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        seed_str = f"_{seed}"

    if fct_name == "speeds":
        fct = run_linear_track_speeds
        data_name = "speed_data"
        above = 1
    elif fct_name == "shifts":
        fct = run_linear_track_shifts
        data_name = "shift_data"
        above = 2
    else:
        raise ValueError(f"fct_name '{fct_name}' not recognized.")

    save_path = Path(get_fig_directory(), f"{data_name}{seed_str}.npz")
    if overwrite:
        gen_util.delete_file(save_path)
    data_dict = gen_util.load_np_dict(save_path)

    if data_dict is None:
        print("Running...")
        start_time = time.perf_counter()
        data_dict = fct(seed=seed, num_jobs=num_jobs)
        gen_util.save_np_dict(save_path, data_dict)
        gen_util.get_duration_str(start_time, log=True)

    traj_idxs = data_dict["BTSP_traj_idxs"][:, 1:] - (
        data_dict["end_traj_idxs_initial"].reshape(-1, 1) + 1
    )
    num_traj_total = data_dict["end_traj_idxs"] - data_dict["end_traj_idxs_initial"]
    log_num_BTSP_if_above(
        data_dict["num_BTSP"],
        above=above,
        traj_idxs=traj_idxs,
        num_traj_total=num_traj_total,
    )
    if "norm_values" in data_dict.keys():
        log_max_normalization_value(data_dict["norm_values"])

    return data_dict


def run_linear_track_hyperparameter_comparison(
    num_repeats=4, num_jobs=1, overwrite=False, seed=True, **kwargs
):
    """
    run_linear_track_hyperparameter_comparison()

    Runs a hyperparameter comparison for the linear environment, collecting BTSP-related
    metrics for each run.

    Args:
    - num_repeats (int): Number of repeats for each hyperparameter configuration.
        Default is 4.
    - num_jobs (int): Number of parallel jobs to run. Default is 1.
    - overwrite (bool): Whether to overwrite existing data. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to the run_linear_track_speed
        function for each run.

    Returns:
    - data_df (pd.DataFrame): Dataframe containing the results of the hyperparameter
        comparison.
    """

    def objective(config):
        """
        objective(config)

        Objective function for a hyperparameter comparison run.

        Args:
        - config (dict): Configuration dictionary, specifying parameters for a specific
            run.

        Returns:
        - output_dict (dict): Output dictionary, with metrics for the run.
        """

        kwargs_use = kwargs.copy()

        seed = False
        if "seed" in config.keys():
            seed = config.pop("seed")

        kwargs_use.update(config)

        Pyrs = get_linear_track_Pyrs(
            speed_mean=params_util.SPEED_MEAN_LINEAR,
            speed_std=params_util.SPEED_MEAN_LINEAR,
            log_BTSP=False,
            wait_after_trajectory=0,
            seed=seed,
            **kwargs_use,
        )

        run_linear_track(
            Pyrs, num_traj_can_stop=10, min_traj_after_BTSP=0, no_logs=True, seed=False
        )

        output_dict = metrics.compute_BTSP_metrics(Pyrs, k=metrics.SMOOTH_K, bins=31)
        output_dict["num_traj_total"] = Pyrs.Agent.get_num_completed_trajectories()
        output_dict["time_total"] = Pyrs.Agent.t

        return output_dict

    data_name = "hyperparameters"
    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        data_name = f"{data_name}_{seed}"

    save_path = Path(get_fig_directory(), f"{data_name}.csv")
    if overwrite:
        gen_util.delete_file(save_path)
    data_df = gen_util.load_df(save_path)

    if data_df is None:

        print("Running...")
        start_time = time.perf_counter()

        if seed:
            seeds = np.arange(seed, seed + num_repeats)
        else:
            seeds = np.sort(np.random.choice(10000, size=num_repeats, replace=False))

        comparison_space = hyper_util.get_search_space(**COMPARISON_KWARGS, seed=seeds)

        hyper_util.run_hyperparameter_search(
            objective,
            comparison_space,
            direc=get_fig_directory(),
            save_name=data_name,
            num_jobs=num_jobs,
            num_repeats=1,
            use_date_time=False,
            plot=False,
        )

        data_df = gen_util.load_df(save_path)

        gen_util.get_duration_str(start_time, log=True)

    return data_df


def plot_linear_track_hyperparameter_comparison(data_df=None, **kwargs):
    """
    plot_linear_track_hyperparameter_comparison()

    Plots the results of a hyperparameter comparison.

    Args:
    - data_df (pd.DataFrame, optional): Dataframe containing the results of the
        hyperparameter comparison. If None, data is loaded from file. Default is None.
    - overwrite (bool): Whether to overwrite existing data. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Returns:
    - ax1D (1D array of matplotlib axes): Axes with hyperparameter comparison results
        plotted.
    """

    if data_df is None:
        data_df = run_linear_track_hyperparameter_comparison(**kwargs)

    Pyrs = get_linear_track_Pyrs()

    mark = (
        Pyrs.ProximalCompartment.BTSP_lr,
        Pyrs.inhibitory_input_filter_tau,
        Pyrs.inhibitory_weight,
    )

    ax1D = paper_plot_fcts.plot_linear_track_hyperparameter_comparison(
        data_df, mark=mark
    )

    return ax1D


def get_openfield_Pyrs(
    corridor=False,
    n=None,
    log_BTSP=True,
    always_log_teleportation=True,
    init_teleport_pairs=None,
    horizontal_in_from_left=True,
    proximal_noise_std=0,
    proximal_w_init_scale=0,
    seed=True,
):
    """
    get_openfield_Pyrs()

    Initializes Pyr parameters for openfield environment.

    Args:
    - corridor (bool): Whether to use the corridor environment. Default is False.
    - n (int, optional): Number of landmark objects to initialize in the openfield
        environment. If None, defaults are used. Default is None.
    - log_BTSP (bool): Whether to log BTSP events. Default is True.
    - always_log_teleportation (bool): Whether to always log teleportation events.
        Default is True.
    - init_teleport_pairs (3D np.ndarray, optional): Teleport pairs to initialized
        with shape (pair, port, coord). Default is None.
    - horizontal_in_from_left (bool): Whether to make teleport entry on the left
        instead of the right. Default is True.
    - proximal_noise_std (float): Standard deviation of noise added to proximal
        compartment neural activity. Default is 0.
    - proximal_w_init_scale (float): Standard deviation of initial weights to proximal
        compartments. Default is 0.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Returns:
    - Pyrs (Pyr): Pyr object initialized with the specified parameters.
    """

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    environment = "openfield_corridor" if corridor else "openfield"
    if n is None:
        n = 1 if corridor else 40

    env_params = {
        "horizontal_in_from_left": horizontal_in_from_left,
    }
    if init_teleport_pairs is not None:
        env_params["init_teleport_pairs"] = init_teleport_pairs
    if not corridor:
        env_params["init_random_walls"] = 4
        env_params["init_random_objects"] = {"landmark": n}
        env_params["init_random_teleport_pairs"] = 0
        env_params["min_dist"] = 0.15

    env_params = params_util.get_env_params(environment=environment, **env_params)

    agent_params = params_util.get_agent_params(
        environment=environment, always_log_teleportation=always_log_teleportation
    )

    Pyr_params = params_util.get_Pyr_params(
        n=n,
        environment=environment,
        log_BTSP=log_BTSP,
        proximal_noise_std=proximal_noise_std,
        proximal_w_init_scale=proximal_w_init_scale,
    )

    Pyrs = run_manager.init_env_objects(
        env_params=env_params,
        agent_params=agent_params,
        Pyr_params=Pyr_params,
        environment=environment,
        plot=False,
    )

    return Pyrs


def run_openfield_corridor(
    Pyrs=None,
    num_steps_can_stop=None,
    time_in_min_can_stop=OPENFIELD_TIME_IN_MIN,
    teleportation_enabled=False,
    min_time_after_BTSP=8 * 60,
    no_logs=False,
    seed=True,
    teleport_kwargs=dict(),
    **kwargs,
):
    """
    run_openfield_corridor()

    Runs a corridor openfield environment with the specified Pyr parameters.

    Args:
    - Pyrs (Pyr, optional): Pyr object with initialized parameters. If None,
        a new Pyr object is created with default parameters.
    - num_steps_can_stop (int or None, optional): Number of steps after which early
        stopping can occur. May prevent the learner object's other stopping conditions
        (number of target reaches or trajectories) from being reached. Pass None to
        avoid constraining these by number of steps, and early stopping will only be
        triggered when one (either) of those conditions is reached, if provided.
        Default is None.
    - time_in_min_can_stop (float, optional): Time in minutes after which learning can
        stop. If specified, it overrides num_steps_can_stop. Note that additional
        criteria (trajectory completion, minimum number of BTSP events, etc.) may
        prolong learning. Default is OPENFIELD_TIME_IN_MIN.
    - teleportation_enabled (bool, optional): Whether to enable teleportation. Default
        is False.
    - min_time_after_BTSP (float): Minimum time in seconds since last
        BTSP event was applied to end the experiment.
        Default is 360 seconds (6 minutes).
    - no_logs (bool): Whether to disable logging. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - teleport_kwargs (dict): Keyword arguments to pass to get_openfield_Pyrs() for
        teleportation initialization. Default is empty dict.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_openfield_BTSP().

    Returns:
    - learner (Learner): The learner object after training.
    """

    start_time = time.perf_counter()

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    if Pyrs is None:
        Pyrs = get_openfield_Pyrs(
            corridor=True,
            seed=False,
            log_BTSP=not (no_logs),
            always_log_teleportation=not (no_logs),
            **teleport_kwargs,
        )

    Pyrs.Agent.update_target_probability_factor_dict("landmark", 1)
    Pyrs.Agent.update_target_probability_factor_dict("no_target", 2)

    min_steps_after_BTSP = int(np.ceil(min_time_after_BTSP / Pyrs.Agent.dt))

    learner = run_manager.learn_openfield_BTSP(
        Pyrs_or_learner=Pyrs,
        num_steps_can_stop=num_steps_can_stop,
        time_in_min_can_stop=time_in_min_can_stop,
        corridor=True,
        teleportation_enabled=teleportation_enabled,
        min_steps_after_BTSP=min_steps_after_BTSP,
        no_logs=no_logs,
        **kwargs,
    )

    if not no_logs:
        gen_util.get_duration_str(start_time, log=True)

    return learner


def plot_openfield_corridor(
    Pyrs=None, plot_type="components", kernel_time=None, **kwargs
):
    """
    plot_openfield_corridor()

    Plots data for an openfield corridor experiment.

    Args:
    - Pyrs (Pyr): Pyr object for openfield corridor.
        If None, a new Pyr object is created.
    - plot_type (str): Type of plot to produce. Options are "components", "last_PF",
        "BTSP_trajectory", "timeseries", or "BTSP_kernel_timeseries".
        Default is "components".
    - kernel_time (float, optional): Time in seconds for BTSP kernel timeseries plot.
        If None, uses default time range for the plot. Default is None.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to plotting functions.

    Returns:
    - ax (plt.Axes or np.ndarray of plt.Axes): Array of subplots with openfield
        corridor data plotted.
    """

    if Pyrs is None:
        learner = run_openfield_corridor(
            time_in_min_can_stop=OPENFIELD_TIME_IN_MIN,
            teleportation_enabled=False,
        )
        Pyrs = learner.Pyrs

    if plot_type == "components":
        ax = paper_plot_fcts.plot_openfield_components(
            Pyrs, traj_idx=EX_TRAJ_IDX, **kwargs
        )
    elif plot_type == "last_PF":
        ax = paper_plot_fcts.plot_last_openfield_PF(Pyrs, **kwargs)
    elif plot_type == "BTSP_trajectory":
        ax = paper_plot_fcts.plot_openfield_corridor_BTSP_trajectory(
            Pyrs, obj_base_s=20, clabel_length=12, **kwargs
        )
    elif plot_type == "timeseries":
        ax = paper_plot_fcts.plot_single_neuron_rate_timeseries(
            Pyrs.ProximalCompartment,
            mark_traj_idxs=[EX_TRAJ_IDX],
            BTSP_kernel_lw=0.02,
            **kwargs,
        )
    elif plot_type == "BTSP_kernel_timeseries":
        if kernel_time is None:
            kernel_time = (18.5, 41.5)  # manually identified for paper example
        t_start, t_end = kernel_time

        ax = paper_plot_fcts.plot_single_neuron_rate_timeseries(
            Pyrs.ProximalCompartment,
            t_start=t_start,
            t_end=t_end,
            in_min=False,
            num_ticks=13,
            BTSP_kernel_lw=2.5,
            fig_size=(9.5, 1.7),
            **kwargs,
        )
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def get_openfield_corridor_repeat_run_params(
    i=0, seed=True, max_runs=100, time_in_min_can_stop=None, teleport=False
):
    """
    get_openfield_corridor_repeat_run_params()

    Obtains parameters for the openfield corridor experiments based on the run index.

    Args:
    - i (int): Run index. Default is 0.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - max_runs (int): Maximum number of runs for generating seeds. Default is 100.
    - time_in_min_can_stop (float, optional): Time in minutes after which early stopping
        can occur. If None, uses OPENFIELD_TIME_IN_MIN or
        OPENFIELD_TELEPORT_REPEAT_TIME_IN_MIN based on teleport argument.
        Default is None.
    - teleport (bool): Whether to retrieve teleportation kwargs.

    Returns:
    - seed (int): Seed to use for the experiment.
    - run_kwargs (dict): Dictionary of keyword arguments to use for the run and pass to
        run_openfield_corridor_teleport() if teleport or
        run_openfield_corridor() otherwise.
    """

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        randst = np.random.RandomState(seed)
    else:
        randst = np.random.RandomState()

    if i > 0:
        if i >= max_runs:
            raise ValueError(f"i must be less than max_runs ({max_runs}).")
        seed = np.sort(randst.choice(10000, size=max_runs, replace=False))[i]

    seed = int(seed)

    run_kwargs = dict()
    if teleport:
        run_kwargs["min_num_teleports"] = 6
        run_kwargs["time_in_min_can_stop"] = (
            time_in_min_can_stop or OPENFIELD_TELEPORT_REPEAT_TIME_IN_MIN
        )

        in_x, in_y = params_util.TELEPORT_IN
        out_x, out_y = params_util.TELEPORT_OUT

        horizontal_in_from_lefts = [True, True, False, False]
        teleport_in_xs = np.asarray([in_x, in_x, in_x - 0.1, in_x - 0.1])
        teleport_out_xs = np.asarray([out_x, out_x - 0.03, out_x + 0.1, out_x + 0.13])

        h_idx = i % len(horizontal_in_from_lefts)

        run_kwargs["teleport_kwargs"] = {
            "horizontal_in_from_left": horizontal_in_from_lefts[h_idx],
            "init_teleport_pairs": [
                (
                    np.array(
                        [[teleport_in_xs[h_idx], in_y], [teleport_out_xs[h_idx], out_y]]
                    )
                    * params_util.SCALE
                ),
            ],
        }

    else:
        run_kwargs["teleportation_enabled"] = False
        run_kwargs["time_in_min_can_stop"] = (
            time_in_min_can_stop or OPENFIELD_TIME_IN_MIN
        )

    return seed, run_kwargs


def run_openfield_corridors(seed=True, num_repeats=10):
    """
    run_openfield_corridors()

    Runs multiple repeats of the openfield corridor experiment and collects data on
    place field widths and weights.

    Args:
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, randomly selected seeds are used for each repeat.
        Default is True.
    - num_repeats (int): Number of repeats for the experiment. Default is 10.

    Returns:
    - data_dict (dict): Dictionary containing:
        - "PC_place_centers" (2D np.ndarray): Array of place cell centers with shape
            (centers, coords).
        - "PC_weights" (3D np.ndarray): Array of place cell input weights with shape
            (repeats, weights, centers).
        - "PF_centers" (2D np.ndarray): Array of place field centers with shape
            (centers, coords).
        - "PFs" (3D np.ndarray): Array of place fields computed from history with shape
            (repeats, fields, centers).
        - "PF_times" (3D np.ndarray): Array of start and end times for place fields
            computed from history with shape (speeds, fields, 2).
        - "PF_traj_idxs" (3D np.ndarray): Array of trajectory indices corresponding to
            times used to compute each place field with shape (repeats, fields, 2).
        - "BTSP_times" (2D np.ndarray): Array of times at which BTSP events were
            applied for each repeat.
        - "BTSP_traj_idxs" (2D np.ndarray): Array of trajectory indices of BTSP events
            with shape (repeats, events).
        - "num_BTSP" (1D np.ndarray): Number of BTSP events that were recorded for each
            run.
        - "BTSP_applied_times" (2D np.ndarray): Array of times at which BTSP events
            were applied for each repeat.
        - "BTSP_applied_traj_idxs" (2D np.ndarray): Array of trajectory indices of
            applied BTSP events with shape (repeats, events).
        - "num_BTSP_applied" (1D np.ndarray): Number of BTSP events that were applied
            for each run.
        - "visit_times" (1D np.ndarray): Array of times at which the agent visited the
            landmark location.
        - "visit_traj_idxs" (1D np.ndarray): Array of trajectory indices at which the
            agent visited the landmark location.
        - "num_visits" (int): Number of visits to the landmark location for each run.
        - "norm_values" (1D np.ndarray): Normalization values used for each run.
        - "end_times" (1D np.ndarray): End time of the experiment for each run.
        - "end_traj_idxs" (1D np.ndarray): End trajectory index of the experiment for
            each run.
        - "seeds": Array of random seeds used for each run.
    """

    data_dicts = list()
    for i in tqdm(range(num_repeats)):
        run_seed, run_kwargs = get_openfield_corridor_repeat_run_params(i, seed=seed)
        learner = run_openfield_corridor(seed=run_seed, no_logs=True, **run_kwargs)
        data_dict = gather_learner_data(
            learner, seed=run_seed, position_name="landmark"
        )
        data_dicts.append(data_dict)

    data_dict = aggregate_from_data_dicts(data_dicts)

    return data_dict


def plot_openfield_corridors(
    corridor_data=None, plot_type="timelines", PF_type="history", **kwargs
):
    """
    plot_openfield_corridors()

    Plots data for openfield corridor experiments.

    Args:
    - corridor_data (dict): Dictionary containing openfield corridor data
        (see run_openfield_corridors()). If not provided, data is loaded or
        experiment is run from scratch. Default is None.
    - plot_type (str): Type of plot to produce. Options are "timelines" or "PFs".
        Default is "timelines".
    - PF_type (str): PF type to plot. Options are "history" or "weights".
        Default is "history".

    Keyword args:
    - **kwargs: Keyword arguments passed to plotting functions.

    Returns:
    - ax (plt.Axes or np.ndarray of plt.Axes): Array of subplots with openfield
        corridor experiment elements plotted.
    """

    if corridor_data is None:
        corridor_data = run_openfield_fct("corridors", overwrite=False)

    if plot_type == "timelines":
        ax = paper_plot_fcts.plot_openfield_corridor_timelines(
            corridor_data["BTSP_times"],
            corridor_data["visit_times"],
            corridor_data["PF_times"],
            end_times=corridor_data["end_times"],
            num_ticks=13,
            **kwargs,
        )
    elif plot_type == "PFs":
        Pyrs = get_openfield_Pyrs(corridor=True)
        PFs, PF_centers = get_last_PFs_from_data_dict(corridor_data, PF_type=PF_type)
        ax = paper_plot_fcts.plot_openfield_corridor_PFs(
            Pyrs, PFs, PF_centers, PF_type=PF_type, num_BTSP=corridor_data["num_BTSP"]
        )
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def run_openfield_corridor_teleport(
    Pyrs=None,
    seed=True,
    num_steps_can_stop=None,
    time_in_min_can_stop=OPENFIELD_TIME_IN_MIN,
    min_num_teleports=6,
    disable_teleportation_between=True,
    min_time_after_BTSP=OPENFIELD_TIME_AFTER_BTSP_IN_MIN * 60,
    no_logs=False,
    teleport_kwargs=dict(),
):
    """
    run_openfield_corridor_teleport()

    Runs an openfield corridor experiment with teleportation enabled until a minimum
    number of teleportation events have occurred.

    Args:
    - Pyrs (Pyr, optional): Pyr object with initialized parameters. If None,
        a new Pyr object is created with default parameters. Default is None.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - num_steps_can_stop (int or None, optional): Number of steps after which early
        stopping can occur. May prevent the learner object's other stopping conditions
        (number of target reaches or trajectories) from being reached. Pass None to
        avoid constraining these by number of steps, and early stopping will only be
        triggered when one (either) of those conditions is reached, if provided.
        Default is None.
    - time_in_min_can_stop (float, optional): Time in minutes after which learning can
        stop. If specified, it overrides num_steps_can_stop. Note that additional
        criteria (trajectory completion, minimum number of BTSP events, etc.) may
        prolong learning. Default is OPENFIELD_TIME_IN_MIN.
    - min_num_teleports (int): Minimum number of teleportation events to occur
        before stopping the experiment. Default is 6.
    - disable_teleportation_between (int): If True, teleportation is disabled for
        6 minutes after a BTSP event (increases probability that PFs can be
        calculated for each BTSP event if teleportation events induce BTSP events).
        Default is True.
    - min_time_after_BTSP (float): Minimum time in seconds since last
        BTSP event was applied to end the experiment. Default is
        OPENFIELD_TIME_AFTER_BTSP_IN_MIN * 60.
    - no_logs (bool): Whether to suppress logging. Default is False.
    - teleport_kwargs (dict): Additional keyword arguments passed to
        run_openfield_corridor() for teleportation initialization.

    Returns:
    - learner (Learner): The learner object after training.
    """

    start_time = time.perf_counter()

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    learner = run_openfield_corridor(
        seed=False, Pyrs=Pyrs, no_logs=no_logs, teleport_kwargs=teleport_kwargs
    )

    if not no_logs:
        print("\nTeleportation enabled.")

    learner.Agent.enable_teleportation(True)

    learner.Agent.update_target_probability_factor_dict("landmark", 2)
    learner.Agent.update_target_probability_factor_dict("no_target", 5)
    learner.Agent.update_target_probability_factor_dict("teleport", 3)

    disable_teleportation = 0
    if disable_teleportation_between:
        disable_teleportation = int(360 / learner.Agent.dt)

    updater = run_manager.TeleportObjectUpdater(
        learner.Agent,
        Pyrs=learner.Pyrs_for_weights,
        disable_teleportation=disable_teleportation,
    )

    min_steps_after_BTSP = int(np.ceil(min_time_after_BTSP / learner.Agent.dt))

    run_manager.learn_openfield_BTSP(
        Pyrs_or_learner=learner,
        use_Hebbian=False,
        num_steps_can_stop=num_steps_can_stop,
        time_in_min_can_stop=time_in_min_can_stop,
        min_steps_after_BTSP=min_steps_after_BTSP,
        min_num_teleports=min_num_teleports,
        corridor=True,
        updater=updater,
        no_logs=no_logs,
    )

    learner.Agent.enable_teleportation(True)

    if not no_logs:
        gen_util.get_duration_str(start_time, log=True)

    return learner


def plot_openfield_teleportation(learner=None, plot_type="summary", **kwargs):
    """
    plot_openfield_teleportation()

    Plots a summary of the openfield corridor teleportation experiment.

    Args:
    - learner (Learner): Learner object after training.
        If None, a new learner object is created. Default is None.
    - plot_type (str): Type of plot to produce. Default is 'summary'.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to plotting functions.

    Returns:
    - ax (plt.Axes): Axes with openfield teleportation summary plotted.
    """

    if learner is None:
        learner = run_openfield_corridor_teleport(
            seed=True,
            time_in_min_can_stop=OPENFIELD_TIME_IN_MIN,
            min_num_teleports=4,
            disable_teleportation_between=True,
        )

    if plot_type == "summary":
        axes = paper_plot_fcts.plot_openfield_teleportation_summary(learner, **kwargs)
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return axes


def animate_openfield_teleportation(
    learner=None,
    fps=8,
    speed_up=10,
    seed=True,
    **kwargs,
):
    """
    animate_openfield_teleportation()

    Creates an animation of an openfield teleportation experiment.

    Args:
    - learner (Learner, optional): Learner object. If None, a new learner object is
        run first. Default is None.
    - fps (int): Frames per second for the animation. Default is 8.
    - speed_up (int): Factor by which to speed up the animation. Default is 10.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to the animation function.

    Returns:
    - anim (matplotlib.animation.FuncAnimation): Animation object for the linear
        shift experiment.
    """

    if learner is None:
        learner = run_openfield_corridor_teleport(
            seed=seed,
            time_in_min_can_stop=OPENFIELD_TIME_IN_MIN,
            min_num_teleports=4,
            disable_teleportation_between=True,
        )

    addendum = None
    if len(learner.Agent.teleportation_disabled):
        first_teleportation_disabled = learner.Agent.teleportation_disabled[0]
        if np.isclose(first_teleportation_disabled[0], 0):
            enabled_time = first_teleportation_disabled[1] * learner.Agent.dt / 60
            addendum = f"teleportation enabled at {enabled_time:.2f} min"

    anim = paper_plot_fcts.animate_openfield(
        learner,
        fps=fps,
        speed_up=speed_up,
        addendum=addendum,
        plot_positions=["landmark"],
        reached_only=False,  # plot both target reaches and visits to landmark
        **kwargs,
    )

    return anim


def run_openfield_corridor_teleports(num_repeats=2, seed=True):
    """
    run_openfield_corridor_teleports()

    Runs an openfield corridor experiment with teleportation enabled until a minimum
    number of teleportation events have occurred and collects data on place field widths
    and weights.

    Args:
    - num_repeats (int): Number of repeats for the experiment. Default is 2.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, randomly selected seeds are used for each repeat.
        Default is True.

    Returns:
    - data_dict (dict): Dictionary containing:
        - "PC_place_centers": Array of place cell centers.
        - "PC_weights": Array of place cell input weights.
        - "PFs": Array of place fields computed from history.
        - "PF_centers": Array of place field centers.
        - "PF_times": Array of place field history collection times.
        - "PF_traj_idxs": Array of trajectory indices corresponding to times used to
            compute each place field.
        - "BTSP_times": Array of times at which BTSP events were applied.
        - "BTSP_traj_idxs": Array of trajectory indices of BTSP events.
        - "num_BTSP": Number of BTSP events that were recorded in total.
        - "BTSP_applied_times": Array of times at which BTSP events were applied.
        - "BTSP_applied_traj_idxs": Array of trajectory indices of applied BTSP events.
        - "num_BTSP_applied": Number of BTSP events that were applied.
        - "visit_times": Array of times at which the agent visited the landmark location.
        - "visit_traj_idxs": Array of trajectory indices at which the agent visited the
            target location.
        - "num_visits": Number of visits to the target location.
        - "teleportation_times": Array of times at which teleportation events occurred.
        - "num_teleportations": Number of teleportation events that occurred.
        - "init_teleport_pairs": Array of teleportation pairs coordinates initialized.
        - "horizontal_in_from_left": Array of teleportation in port directions.
        - "norm_values": Normalization values used for each run.
        - "end_times": End time of the experiment for each run.
        - "end_traj_idxs": End trajectory index of the experiment for each run.
        - "seeds": Array of random seeds used for each run.
    """

    data_dicts = list()
    for i in tqdm(range(num_repeats * 4)):  # 4 teleportation parameter combinations
        run_seed, run_kwargs = get_openfield_corridor_repeat_run_params(
            i, seed=seed, teleport=True
        )
        learner = run_openfield_corridor_teleport(
            seed=run_seed, no_logs=True, **run_kwargs
        )

        data_dict = gather_learner_data(
            learner, seed=run_seed, position_name="landmark", teleport=True
        )
        data_dicts.append(data_dict)

    data_dict = aggregate_from_data_dicts(data_dicts)

    return data_dict


def plot_openfield_teleportations(
    teleport_data=None, plot_type="timelines", PF_type="history", **kwargs
):
    """
    plot_openfield_teleportations()

    Plots data for openfield corridor teleportation experiments.

    Args:
    - teleport_data (dict): Dictionary containing openfield corridor teleportation data
        (see run_openfield_corridor_teleports()). If not provided, data is loaded or
        experiment is run from scratch. Default is None.
    - plot_type (str): Type of plot to produce. Options are "timelines" or "PFs".
        Default is "timelines".
    - PF_type (str): PF type to plot. Options are "history" or "weights".
        Default is "history".

    Keyword args:
    - **kwargs: Keyword arguments passed to plotting functions.

    Returns:
    - ax (plt.Axes or np.ndarray of plt.Axes): Array of subplots with openfield
        teleportation experiment elements plotted.

    """

    if teleport_data is None:
        teleport_data = run_openfield_fct("teleports", overwrite=False)

    if plot_type == "timelines":
        ax = paper_plot_fcts.plot_openfield_corridor_timelines(
            teleport_data["BTSP_times"],
            teleport_data["visit_times"],
            teleport_data["PF_times"],
            end_times=teleport_data["end_times"],
            teleportation_times=teleport_data["teleportation_times"],
            num_teleportation_pairs=teleport_data["init_teleport_pairs"].shape[1],
            **kwargs,
        )
    elif plot_type == "PFs":
        Pyrs = list()
        for i, init_teleport_pairs in enumerate(teleport_data["init_teleport_pairs"]):
            Pyrs.append(
                get_openfield_Pyrs(
                    corridor=True,
                    init_teleport_pairs=init_teleport_pairs,
                    horizontal_in_from_left=teleport_data["horizontal_in_from_left"][i],
                )
            )
        PFs, PF_centers = get_last_PFs_from_data_dict(teleport_data, PF_type=PF_type)
        num_cols = int(np.ceil(len(PFs) / 2))
        ax = paper_plot_fcts.plot_openfield_corridor_PFs(
            Pyrs,
            PFs,
            PF_centers,
            PF_type=PF_type,
            num_BTSP=teleport_data["num_BTSP"],
            num_teleportations=teleport_data["num_teleportations"],
            num_cols=num_cols,
            obj_base_s=5,
            no_teleport=False,
            **kwargs,
        )
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def run_openfield_multitarget(
    Pyrs=None,
    num_steps_can_stop=None,
    time_in_min_can_stop=OPENFIELD_MULTITARGET_TIME_IN_MIN,
    min_time_after_BTSP=OPENFIELD_MULTITARGET_TIME_AFTER_BTSP_IN_MIN * 60,
    proximal_noise_std=0.1,
    proximal_w_init_scale=0.1,
    no_logs=False,
    seed=True,
    **kwargs,
):
    """
    run_openfield_corridor()

    Runs a corridor openfield environment with the specified Pyr parameters.

    Args:
    - Pyrs (Pyr, optional): Pyr object with initialized parameters. If None,
        a new Pyr object is created with default parameters.
    - num_steps_can_stop (int or None, optional): Number of steps after which early
        stopping can occur. May prevent the learner object's other stopping conditions
        (number of target reaches or trajectories) from being reached. Pass None to
        avoid constraining these by number of steps, and early stopping will only be
        triggered when one (either) of those conditions is reached, if provided.
        Default is None.
    - time_in_min_can_stop (float, optional): Time in minutes after which learning can
        stop. If specified, it overrides num_steps_can_stop. Note that additional
        criteria (trajectory completion, minimum number of BTSP events, etc.) may
        prolong learning. Default is OPENFIELD_MULTITARGET_TIME_IN_MIN.
    - min_time_after_BTSP (float): Minimum time in seconds since last
        BTSP event was applied to end the experiment.
        Default is OPENFIELD_MULTITARGET_TIME_AFTER_BTSP_IN_MIN * 60.
    - proximal_noise_std (float): Standard deviation of noise to add to proximal
        activity. Default is 0.1.
    - proximal_w_init_scale (float): Standard deviation of initial weights to proximal
        compartments. Default is 0.1.
    - no_logs (bool): Whether to disable logging. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    Returns:
    - learner (Learner): The learner object after training.
    """

    start_time = time.perf_counter()

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    if Pyrs is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="add_walls()")
            Pyrs = get_openfield_Pyrs(
                corridor=False,
                log_BTSP=False,
                seed=False,
                proximal_noise_std=proximal_noise_std,
                proximal_w_init_scale=proximal_w_init_scale,
            )

    min_steps_after_BTSP = int(np.ceil(min_time_after_BTSP / Pyrs.Agent.dt))

    learner = run_manager.learn_openfield_BTSP(
        Pyrs_or_learner=Pyrs,
        num_steps_can_stop=num_steps_can_stop,
        time_in_min_can_stop=time_in_min_can_stop,
        min_steps_after_BTSP=min_steps_after_BTSP,
        no_logs=no_logs,
        **kwargs,
    )

    if not no_logs:
        gen_util.get_duration_str(start_time, log=True)

    return learner


def plot_openfield_multitarget(learner=None, plot_type="summary", **kwargs):
    """
    plot_openfield_multitarget()

    Plots data for an openfield multitarget experiment.

    Args:
    - learner (Learner): Learner object for openfield multitarget.
        If None, a new Learner object is created.
    - plot_type (str): Type of plot to produce. Options are "summary", "PFs",
        "counts", "normalization", or "BTSP_responses". Default is "summary".

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to plotting functions.

    Returns:
    - ax (plt.Axes or np.ndarray of plt.Axes): Array of subplots with openfield
        multitarget data plotted.
    """

    if learner is None:
        learner = run_openfield_multitarget(
            time_in_min_can_stop=OPENFIELD_TIME_IN_MIN,
        )

    if plot_type == "summary":
        ax = paper_plot_fcts.plot_openfield_multitarget_summary(learner, **kwargs)
    elif plot_type == "PFs":
        ax = paper_plot_fcts.plot_openfield_multitarget_PFs(learner.Pyrs, **kwargs)
    elif plot_type == "counts":
        ax = paper_plot_fcts.plot_BTSP_counts_vs_target_visits(learner.Pyrs, **kwargs)
    elif plot_type == "normalization":
        ax = paper_plot_fcts.plot_normalization_values(learner.Pyrs, **kwargs)
    elif plot_type == "BTSP_responses":
        ax = paper_plot_fcts.plot_BTSP_responses(
            learner.Pyrs, align_to_plateau_onset=True, **kwargs
        )

    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def animate_openfield_multitarget(
    learner=None,
    fps=8,
    speed_up=[10, 30, 100, 300],
    seed=True,
    **kwargs,
):
    """
    animate_openfield_multitarget()

    Creates an animation of a multitarget openfield experiment.

    Args:
    - learner (Learner, optional): Learner object. If None, a new learner object is
        run first. Default is None.
    - fps (int): Frames per second for the animation. Default is 8.
    - speed_up (int or list): Factor or list of factors by which to speed up the
        animation. Default is [10, 30, 100, 300].
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to the animation function.

    Returns:
    - anim (matplotlib.animation.FuncAnimation): Animation object for the linear
        shift experiment.
    """

    if learner is None:
        learner = run_openfield_multitarget(
            time_in_min_can_stop=OPENFIELD_TIME_IN_MIN,
            seed=seed,
        )

    addendum = f"{learner.Pyrs.n} target objects"

    anim = paper_plot_fcts.animate_openfield(
        learner,
        fps=fps,
        speed_up=speed_up,
        addendum=addendum,
        plot_positions=["target"],
        reached_only=True,  # plot only when objects were visited as targets
        **kwargs,
    )

    return anim


def run_openfield_multitarget_remapping(
    Pyrs=None,
    num_steps_can_stop=None,
    time_in_min_can_stop=OPENFIELD_MULTITARGET_TIME_IN_MIN,
    min_time_after_BTSP=OPENFIELD_MULTITARGET_TIME_AFTER_BTSP_IN_MIN * 60,
    no_logs=False,
    seed=True,
    **kwargs,
):
    """
    run_openfield_corridor()

    Runs a corridor openfield environment with the specified Pyr parameters.

    Args:
    - Pyrs (Pyr, optional): Pyr object with initialized parameters. If None,
        a new Pyr object is created with default parameters.
    - num_steps_can_stop (int or None, optional): Number of steps after which early
        stopping can occur. May prevent the learner object's other stopping conditions
        (number of target reaches or trajectories) from being reached. Pass None to
        avoid constraining these by number of steps, and early stopping will only be
        triggered when one (either) of those conditions is reached, if provided.
        Default is None.
    - time_in_min_can_stop (float, optional): Time in minutes after which learning can
        stop. If specified, it overrides num_steps_can_stop. Note that additional
        criteria (trajectory completion, minimum number of BTSP events, etc.) may
        prolong learning. Default is OPENFIELD_MULTITARGET_TIME_IN_MIN.
    - min_time_after_BTSP (float): Minimum time in seconds since last
        BTSP event was applied to end the experiment.
        Default is OPENFIELD_MULTITARGET_TIME_AFTER_BTSP_IN_MIN * 60.
    - no_logs (bool): Whether to disable logging. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    Returns:
    - learner (Learner): The learner object after training.
    """

    start_time = time.perf_counter()

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    kwargs["num_steps_can_stop"] = num_steps_can_stop
    kwargs["time_in_min_can_stop"] = time_in_min_can_stop
    kwargs["no_logs"] = no_logs

    learner = run_openfield_multitarget(
        Pyrs=Pyrs,
        min_time_after_BTSP=min_time_after_BTSP,
        seed=False,
        **kwargs,
    )

    # shuffle PCs (remap)
    randst = seed if seed else None
    learner.remap_PC_weights(randst=randst, no_logs=no_logs)

    min_steps_after_BTSP = int(np.ceil(min_time_after_BTSP / learner.Agent.dt))
    run_manager.learn_openfield_BTSP(
        Pyrs_or_learner=learner,
        min_steps_after_BTSP=min_steps_after_BTSP,
        **kwargs,
    )

    if not no_logs:
        gen_util.get_duration_str(start_time, log=True)

    return learner


def plot_openfield_multitarget_remapping(learner=None, plot_type="summary", **kwargs):
    """
    plot_openfield_multitarget_remapping()

    Plots data for an openfield multitarget experiment.

    Args:
    - learner (Learner): Learner object for openfield multitarget.
        If None, a new Learner object is created.
    - plot_type (str): Type of plot to produce. Options are "summary",
        "pre_post_weights", "pre_post_BTSP", "correlations" or see
        plot_openfield_multitarget(). Default is "summary".

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to plotting functions.

    Returns:
    - ax (plt.Axes or np.ndarray of plt.Axes): Array of subplots with openfield
        multitarget data plotted.
    """

    if learner is None:
        learner = run_openfield_multitarget_remapping(
            time_in_min_can_stop=OPENFIELD_TIME_IN_MIN,
        )

    remap_time = (
        paper_plot_fcts.get_learner_remap_step(learner, idx=0, num_total=1)
        * learner.Agent.dt
    )

    if plot_type == "pre_post_weights":
        ax = paper_plot_fcts.plot_openfield_remapping_pre_post_weights(
            learner, **kwargs
        )
    elif plot_type == "correlations":
        ax = paper_plot_fcts.plot_remapping_correlation_matrices(learner, **kwargs)
    elif plot_type == "pre_post_BTSP":
        ax = paper_plot_fcts.plot_remapping_pre_post_BTSP(learner, **kwargs)
    else:
        if plot_type == "summary":
            kwargs["after_remap"] = True
        elif plot_type == "PFs":
            kwargs["split_time"] = remap_time
        elif plot_type == "counts":
            kwargs["t_start"] = remap_time
        elif plot_type == "normalization":
            kwargs["shift_time"] = remap_time
        ax = plot_openfield_multitarget(learner, plot_type=plot_type, **kwargs)

    return ax


def run_openfield_fct(
    fct_name="corridors", num_repeats=None, overwrite=False, seed=True
):
    """
    run_openfield_fct()

    Runs a specified openfield function, loading an existing data dictionary if it
    exists or rerunning the experiment.

    Args:
    - fct_name (str): Name of the function to run. Options are 'corridors' or
        'teleports'. Default is 'corridors'.
    - num_repeats (int): Number of repeats for the experiment. If None, defaults for
        each function type are used. Default is None.
    - overwrite (bool): Whether to overwrite existing data. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, seed run_openfield_corridors() for details.
        Default is True.

    Returns:
    - data_dict (dict): Dictionary containing the results of the experiment.
    """

    seed_str = ""
    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        seed_str = f"_{seed}"

    if fct_name == "corridors":
        fct = run_openfield_corridors
        data_name = "corridor_data"
        num_repeats = num_repeats or 10
    elif fct_name == "teleports":
        fct = run_openfield_corridor_teleports
        data_name = "teleport_data"
        num_repeats = num_repeats or 2
    else:
        raise ValueError(f"fct_name '{fct_name}' not recognized.")

    save_path = Path(get_fig_directory(), f"{data_name}{seed_str}.npz")
    if overwrite:
        gen_util.delete_file(save_path)
    data_dict = gen_util.load_np_dict(save_path)

    if data_dict is None:
        print("Running...")
        start_time = time.perf_counter()
        data_dict = fct(seed=seed, num_repeats=num_repeats)
        gen_util.save_np_dict(save_path, data_dict)
        gen_util.get_duration_str(start_time, log=True)

    log_num_BTSP_if_above(data_dict["num_BTSP"], above=1)
    if fct_name == "teleports":
        log_num_teleportations(data_dict["num_teleportations"])
    if "norm_values" in data_dict.keys():
        log_max_normalization_value(data_dict["norm_values"])

    return data_dict


def get_fig_dict():
    """
    get_fig_dict()

    Returns a dictionary mapping figure numbers and panels to their corresponding
    plotting functions and parameters.

    Returns:
    - fig_dict (dict): Dictionary mapping figure numbers and panels to plotting
        functions and parameters.
    """

    fig_dict = {
        1: {
            "A": {"fct": "schematic"},
            "B": {"fct": plot_linear_track, "plot_type": "environment"},
            "C": {"fct": "schematic"},
            "D": {"fct": plot_linear_track, "plot_type": "BTSP_kernel"},
        },
        2: {
            "A": {"fct": plot_linear_track, "plot_type": "summary"},
            "B": {"fct": plot_linear_track, "plot_type": "place_fields"},
            "C": {"fct": plot_linear_track, "plot_type": "binned_rates"},
        },
        "1S": {
            "A-E": {
                "fct": plot_linear_track_hyperparameter_comparison,
                "plot_type": "hyperparameters",
            }
        },
        "2S": {
            "A": {
                "fct": plot_linear_track,
                "plot_type": "neural_activity",
                "inhibition": "balanced",
            },
            "B": {
                "fct": plot_linear_track,
                "plot_type": "neural_activity",
                "inhibition": "insufficient",
            },
            "C": {
                "fct": plot_linear_track,
                "plot_type": "neural_activity",
                "inhibition": "excessive",
            },
        },
        3: {
            "A": {"fct": plot_linear_track_speed_PFs, "plot_type": "examples"},
            "B": {"fct": plot_linear_track_speed_PFs, "plot_type": "all"},
        },
        "3S": {
            "A": {
                "fct": plot_linear_track_speed_PFs,
                "plot_type": "examples",
                "PF_type": "weights",
            },
            "B": {
                "fct": plot_linear_track_speed_PFs,
                "plot_type": "all",
                "PF_type": "weights",
            },
        },
        4: {
            "A-B": {"fct": plot_linear_track_shift_PFs, "plot_type": "examples"},
        },
        "4S": {
            "A-B": {"fct": plot_linear_track_shift_PFs, "plot_type": "all"},
        },
        "4V": {
            "A": {"fct": animate_linear_track_shift},
        },
        5: {
            "A-D": {"fct": plot_openfield_corridor, "plot_type": "components"},
            "E": {"fct": plot_openfield_corridor, "plot_type": "last_PF"},
            "F": {"fct": plot_openfield_corridor, "plot_type": "BTSP_trajectory"},
            "G": {"fct": plot_openfield_corridor, "plot_type": "timeseries"},
        },
        "5S": {
            "A": {
                "fct": plot_openfield_corridor,
                "plot_type": "BTSP_kernel_timeseries",
            },
            "B": {
                "fct": plot_openfield_corridor,
                "plot_type": "last_PF",
                "PF_type": "weights",
                "fig_side": 2.6,
            },
            "C": {"fct": plot_openfield_corridors, "plot_type": "timelines"},
            "D": {"fct": plot_openfield_corridors, "plot_type": "PFs"},
        },
        6: {
            "A-D": {"fct": plot_openfield_teleportation, "plot_type": "summary"},
        },
        "6S": {
            "A": {
                "fct": plot_openfield_teleportations,
                "plot_type": "timelines",
                "fig_width": 10,
            },
            "B": {
                "fct": plot_openfield_teleportations,
                "plot_type": "PFs",
                "fig_side": 1.8,
            },
        },
        "6V": {
            "A": {"fct": animate_openfield_teleportation},
        },
        7: {
            "A-C": {"fct": plot_openfield_multitarget, "plot_type": "summary"},
            "D": {"fct": plot_openfield_multitarget, "plot_type": "counts"},
            "E": {
                "fct": plot_openfield_multitarget,
                "plot_type": "PFs",
                "n": 9,
            },
        },
        "7S": {
            "A": {
                "fct": plot_openfield_multitarget,
                "plot_type": "PFs",
                "n": "all",
                "width_per": 0.76,
            },
            "B": {"fct": plot_openfield_multitarget, "plot_type": "BTSP_responses"},
            "C": {"fct": plot_openfield_multitarget, "plot_type": "normalization"},
        },
        "7V": {
            "A": {"fct": animate_openfield_multitarget},
        },
    }

    return fig_dict


def get_fig_panel_list(fig=1):
    """
    get_fig_panel_list()

    Retrieves a list of figure-panel combinations to plot.

    Args:
    - fig (int or str): Figure number or "all" to get all figure-panel combinations.

    Returns:
    - fig_panel_list (list of tuples): List of (figure, panel) tuples to plot.
    """

    fig_dict = get_fig_dict()

    if str(fig) == "all":
        fig_panel_list = list()
        for fig_key in fig_dict.keys():
            for panel in fig_dict[fig_key].keys():
                fig_panel_list.append((fig_key, panel))

    else:
        if str(fig).isnumeric():
            fig = int(fig)
        if fig not in fig_dict.keys():
            raise KeyError(f"Unknown figure: {fig}.")
        fig_panel_list = [(fig, panel) for panel in fig_dict[fig].keys()]

    return fig_panel_list


def get_fct_kwargs(fig=1, panel="A"):
    """
    get_fct_kwargs()

    Retrieves the plotting function and its keyword arguments for a specific figure
    and panel.

    Args:
    - fig (int or str): Figure number.
    - panel (str): Panel letter (A, B, C, etc.).

    Returns:
    - fct (callable or str): The plotting function or "schematic" for schematic plots.
    - fct_kwargs (dict): Dictionary of keyword arguments for the plotting function.
    """

    if str(fig).isnumeric():
        fig = int(fig)

    panel = panel.upper()

    fig_dict = get_fig_dict()

    if fig not in fig_dict.keys():
        raise KeyError(f"Unknown figure: {fig}.")

    if panel not in fig_dict[fig].keys():
        raise ValueError(f"Unknown panel for figure {fig}: {panel}.")

    fct_kwargs = fig_dict[fig][panel]
    fct = fct_kwargs.pop("fct")

    return fct, fct_kwargs


def plot_figure_panel(*args, fig=1, panel="A", save=True, **kwargs):
    """
    plot_figure_panel()

    Plots a specific panel of a figure.

    Args:
    - *args: Positional arguments passed to the plotting function.
    - fig (int or str): Figure number.
    - panel (str): Panel letter (A, B, C, etc.).
    - save (bool): Whether to save the figure.
    - **kwargs: Keyword arguments passed to the plotting function.

    Returns:
    - ax: The axes object for the plotted panel.
    """

    fct, fct_kwargs = get_fct_kwargs(fig=fig, panel=panel)

    if fct == "schematic":
        ax = None
        print("Schematic plot.")
    else:
        ax = fct(*args, **fct_kwargs, **kwargs)

    if save and ax is not None:
        key = f"{fig}{panel}"
        if isinstance(ax, mpl_animation.FuncAnimation):
            fig = ax
            dpi = 300
            fig_save_types = "mp4"
        else:
            fig = ax.ravel()[0].figure if isinstance(ax, np.ndarray) else ax.figure
            dpi = 600
            fig_save_types = ["png", "svg"]

        plot_util.save_figure(
            fig, key, no_timestamp=True, dpi=dpi, fig_save_types=fig_save_types
        )

    return ax


def get_args():
    """
    get_args()

    Parses command-line arguments for figure plotting or data dictionary generation.

    Returns:
    - args: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--fig", type=str, default="all", help="Figure number.")
    parser.add_argument("--panel", type=str, default="all", help="Panel letter.")
    parser.add_argument(
        "--data_dict", type=str, default=None, help="Generate data dictionary."
    )
    parser.add_argument("--gen_dir", type=str, default=".", help="Figure directory.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite data dictionary."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--num_jobs", type=int, default=1, help="Number of jobs.")

    args = parser.parse_args()

    return args


def main():
    """
    main()

    Main function to run figure panel plotting or generate data dictionaries.
    """

    args = get_args()

    initialize_paper_parameters(gen_dir=args.gen_dir)

    if args.data_dict is None:
        if args.panel == "all":
            fig_panel_list = get_fig_panel_list(fig=args.fig)
            print(f"Plotting {len(fig_panel_list)} panels.")
        else:
            fig_panel_list = [(args.fig, args.panel)]
        for fig, panel in fig_panel_list:
            print(f"\nPlotting figure {fig} panel {panel}.")
            seed_kwarg = dict() if args.seed is None else {"seed": args.seed}
            plot_figure_panel(fig=fig, panel=panel, **seed_kwarg)

    else:
        kwargs = {"overwrite": args.overwrite}
        if args.seed is not None:
            kwargs["seed"] = args.seed
        if args.data_dict in ["speeds", "shifts"]:
            run_linear_track_fct(args.data_dict, num_jobs=args.num_jobs, **kwargs)
        elif args.data_dict in ["corridors", "teleports"]:
            run_openfield_fct(args.data_dict, **kwargs)
        else:
            raise ValueError(f"data_dict '{args.data_dict}' not recognized.")


if __name__ == "__main__":

    main()
