#!/usr/bin/env python3

import copy
import time
from pathlib import Path
import warnings

import itertools
from joblib import Parallel, delayed
import numpy as np
import ratinabox
from tqdm import tqdm

from predhpc import run_manager, paper_plot_fcts
from predhpc.util import gen_util, params_util, ext_util, plot_util
from predhpc.experiments import metrics

PAPER_SEED = 18

SPEED_MEANS = gen_util.get_rounded_linspace(0.05, 0.4, 29)  # (0.05, 0.55, 41)
SPEED_EXAMPLES = [0.15, 0.25, 0.35]

TARGET_SHIFTS = gen_util.get_rounded_linspace(-3.6, 2.4, 61)
SHIFT_EXAMPLES = [1.0, 0, -0.4, -3.0]

NUM_TRAJ_SPEED = 20
OPENFIELD_MAX_STEPS = 20000
OPENFIELD_TELEPORT_REPEAT_STEPS = 60000
EX_TRAJ_IDX = 8


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

    norm_values = learner.Pyrs.SomaticCompartment.get_normalization_values("PCs")[1]

    data_dict = metrics.gather_PF_info(learner, k=k, position_name=position_name)
    data_dict["norm_values"] = norm_values[..., 0]
    data_dict["end_time"] = learner.Pyrs.Agent.t

    for key, value in kwargs.items():
        data_dict[key] = value

    if seed:
        data_dict["seed"] = int(seed)

    if teleport:
        if not hasattr(learner.Agent, "teleportation_df"):
            raise ValueError("Learner does not have teleportation data.")
        data_dict["num_teleportations"] = len(learner.Agent.teleportation_df)
        data_dict["teleportation_times"] = learner.Agent.teleportation_df["time"].values
        data_dict["init_teleport_pairs"] = np.asarray(learner.Env.init_teleport_pairs)
        data_dict["horizontal_in_from_left"] = learner.Env.horizontal_in_from_left

    return data_dict


def aggregate_from_data_dicts(data_dicts):
    """
    aggregate_from_data_dicts(data_dicts)

    Aggregates a list of PF data dictionaries into a single data dictionary with
    appropriately padded arrays.

    Args:
    - data_dicts (list of dict): List of data dictionaries to compile with the
        following keys: "PC_place_centers", "PC_weights", "PFs", "PF_centers",
        "PF_times", "BTSP_times", "num_BTSP", and optionally
        "visit_times", "num_visits", "norm_values", "seed", and
        optionally "teleportation_times", "num_teleportations".
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

    data_dict = dict()
    for key in data_dicts[0].keys():
        if key in ["PC_place_centers", "PF_centers"]:
            data_dict[key] = data_dicts[0][key]
        elif key in [
            "PC_weights",
            "PC_smoothed_weights",
            "PFs",
            "PF_times",
            "norm_values",
            "BTSP_applied_times",
            "BTSP_times",
            "visit_times",
            "teleportation_times",
        ]:
            if key == "visit_times":
                shape = visit_shape
            elif key == "teleportation_times":
                shape = teleportation_shape
            elif key == "BTSP_times":
                shape = BTSP_shape
            elif key == "BTSP_applied_times":
                shape = BTSP_applied_shape
            else:
                shape = weights_shape

            data_dict[key] = np.full(shape + data_dicts[0][key].shape[1:], np.nan)
            for j, sub_data_dict in enumerate(data_dicts):
                data = sub_data_dict[key]
                data_dict[key][j, : data.shape[0]] = data
        else:
            data_dict[key] = np.asarray([data_dict[key] for data_dict in data_dicts])

    if "seed" in data_dict.keys():
        data_dict["seeds"] = data_dict.pop("seed")

    if "end_time" in data_dict.keys():
        data_dict["end_times"] = data_dict.pop("end_time")

    if "target_shift" in data_dict.keys():
        data_dict["target_shifts"] = data_dict.pop("target_shift")

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


def log_num_BTSP_if_above(num_BTSP, above=1):
    """
    log_num_BTSP_if_above(num_BTSP, above=1)

    Logs if any number of BTSP events are above a certain threshold.

    Args:
    - num_BTSP (1D np.ndarray): Number of BTSP events recorded.
    - above (int): Threshold value. Default is 1.
    """

    if np.any(num_BTSP > above):
        n_strs = list()
        for n in np.sort(np.unique(num_BTSP)):
            if n > above:
                n_strs.append(f"{n} in {np.sum(num_BTSP == n)}/{len(num_BTSP)}")
        event_str = "event" if above == 1 else "events"
        log_str = f"More than {above} BTSP {event_str}: {', '.join(n_strs)}."
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


def get_linear_Pyrs(
    scale=params_util.SCALE_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD,
    wait_after_trajectory=0,
    log_BTSP=True,
    seed=True,
    **Pyr_kwargs,
):
    """
    get_linear_Pyrs()

    Initializes Pyr parameters for linear environment.

    Args:
    - scale (float): Scale of the environment. Default is params_util.SCALE_LINEAR.
    - speed_mean (float): Mean speed of the agent. Default is
        params_util.SPEED_MEAN_LINEAR.
    - speed_std (float): Standard deviation of the agent's speed. Default is
        params_util.SPEED_STD.
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

    target_position = params_util.get_target_position()

    env_params = params_util.get_env_params(
        environment="linear",
        scale=scale,
    )

    agent_params = params_util.get_agent_params(
        environment="linear",
        scale=scale,
        target_position=target_position,
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


def run_linear(
    Pyrs=None,
    max_num_steps=3800,
    max_time_min=None,
    BTSP_on=None,
    seed=True,
    inhibition="balanced",
    factor=2.0,
    **kwargs,
):
    """
    run_linear()

    Runs a linear environment with the specified Pyr parameters.

    Args:
    - Pyrs (Pyr, optional): Pyr object with initialized parameters. If None,
        a new Pyr object is created with default parameters.
    - max_num_steps (int): Maximum number of steps to run the environment. Note that
        if learner is set to complete final trajectory, max_num_steps will be exceeded
        to complete any incomplete trajectories. Default is 3800.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step. Note
        that if learner is set to complete final trajectory, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is None.
    - BTSP_on (int): Trajectory on which to turn on BTSP. 1 for first trajectory.
        Default is None.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - inhibition (str): Type of inhibition to apply. Options are "balanced",
        "excessive", or "insufficient". Default is "balanced".
    - factor (float): Factor by which to adjust inhibitory weight for "excessive" or
        "insufficient" inhibition. Default is 2.0.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    """

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    if Pyrs is None:
        if inhibition in ["balanced", "excessive", "insufficient"]:
            inhibitory_weight = params_util.get_Pyr_params()["inhibitory_weight"]
            if inhibition == "excessive":
                inhibitory_weight *= factor
            elif inhibition == "insufficient":
                inhibitory_weight /= factor
            if inhibition != "balanced":
                print(
                    f"Using {inhibition} inhibition (weight: {inhibitory_weight:.2f})."
                )
        else:
            raise ValueError(f"Unknown inhibition type: {inhibition}.")

        Pyrs = get_linear_Pyrs(
            seed=False,
            wait_after_trajectory=params_util.WAIT_LINEAR,
            inhibitory_weight=inhibitory_weight,
        )

    if max_time_min is not None:
        max_num_steps = int(max_time_min * 60 / Pyrs.Agent.dt)

    learner = run_manager.learn_1D_BTSP(
        Pyrs, BTSP_on=BTSP_on, max_num_steps=max_num_steps, plot=False, **kwargs
    )

    return learner


def plot_linear(
    learner=None,
    max_time_min=2.0,
    inhibition="balanced",
    factor=1.8,
    plot_type="summary",
    **kwargs,
):
    """
    plot_linear()

    Produces plots for a linear experiment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step. Note
        that if learner is set to complete final trajectory, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is 2.0.
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
            Pyrs = get_linear_Pyrs(
                wait_after_trajectory=params_util.WAIT_LINEAR,
            )
        else:
            Pyrs = learner.Pyrs

    elif learner is None:
        learner = run_linear(
            max_time_min=max_time_min, inhibition=inhibition, factor=factor
        )

    if plot_type == "environment":
        ax = paper_plot_fcts.plot_linear_environment(Ag=Pyrs.Agent, **kwargs)
    elif plot_type == "BTSP_kernel":
        ax = paper_plot_fcts.plot_BTSP_kernel(Pyrs, **kwargs)
    elif plot_type == "summary":
        ax = paper_plot_fcts.plot_linear_summary(learner, **kwargs)
    elif plot_type == "neural_activity":
        ax = paper_plot_fcts.plot_linear_neural_activity(learner, **kwargs)
    elif plot_type == "place_fields":
        ax = paper_plot_fcts.plot_linear_place_fields(learner, **kwargs)
    elif plot_type == "binned_rates":
        ax = paper_plot_fcts.plot_linear_binned_rates(learner, **kwargs)
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def run_linear_speed(
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD,
    test_speed_mean=None,
    test_speed_std=None,
    max_time_min=NUM_TRAJ_SPEED,
    max_num_traj=NUM_TRAJ_SPEED,
    k=metrics.SMOOTH_K,
    no_logs=True,
    seed=True,
):
    """
    run_linear_speed()

    Runs and collects data for a single linear speed experiment.

    Args:
    - speed_mean (float): Mean speed for the experiment.
        Default is params_util.SPEED_MEAN_LINEAR.
    - max_time_min (float, optional): Maximum time in minutes to run the environment
        for assessing place field. Note that if learner is set to complete all
        trajectories, max_time_min will be exceeded to complete any incomplete
        trajectories. Default is NUM_TRAJ_SPEED.
    - max_num_traj (int): Maximum number of trajectories to run for assessing place
        field. Default is NUM_TRAJ_SPEED.
    - BTSP_on (int): Trajectory on which to enable. Later trajectories allow more
        time Default is 5.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is metrics.SMOOTH_K.
    - no_logs (bool): Whether to disable logging. Default is True.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    Returns:
    - learner (Learner): The learner object after running the experiment.
    - data_dict (dict): Dictionary containing the results of the experiment under keys:
        - "speed_mean": Mean speed for the experiment.
        - "PC_place_centers": Place cell centers.
        - "PC_weights": Place cell input weights.
        - "PC_weight_widths": Last place cell input weight widths.
        - "PC_smoothed_weights": Smoothed place cell input weights.
        - "PC_smoothed_weight_widths": Last smoothed place cell input weight widths.
        - "PFs": Place fields computed from history.
        - "PF_centers": Place field centers.
        - "PF_widths": Last place field widths.
        - "BTSP_times": Times of BTSP events.
        - "num_BTSP": Number of BTSP events.
        - "BTSP_applied_times": Times when BTSP events were applied.
        - "num_BTSP_applied": Number of applied BTSP events.
        - "end_time": End time of the experiment.
        - "norm_values": Weight normalization values used.
        if seed:
        - "seed": Seed for the experiment.
    """

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    Pyrs = get_linear_Pyrs(
        speed_mean=speed_mean,
        speed_std=speed_std,
        log_BTSP=False,
        wait_after_trajectory=0,
        seed=False,
    )

    for _ in range(5):
        learner = run_linear(
            Pyrs,
            max_time_min=max_time_min,
            max_num_traj=2,
            max_num_target_reaches=2,
            no_logs=no_logs,
            seed=False,
        )

        num_BTSP_applied = len(
            Pyrs.SomaticCompartment.get_BTSP_steps(applied_only=True, apply_step=True)
        )
        if num_BTSP_applied:
            break

    if num_BTSP_applied == 0:
        raise RuntimeError("No BTSP occurred.")

    # t_start = Pyrs.Agent.t
    Pyrs.Agent.set_speed(mean=test_speed_mean, std=test_speed_std)
    run_linear(
        Pyrs,
        max_time_min=max_time_min,
        max_num_traj=max_num_traj,
        max_num_target_reaches=max_num_traj,
        no_logs=no_logs,
        seed=False,
    )

    data_dict = gather_learner_data(learner, seed=seed, k=k, speed_mean=speed_mean)

    return learner, data_dict


def run_linear_speeds(
    seed=True,
    max_time_min=NUM_TRAJ_SPEED,
    num_repeats=1,
    k=metrics.SMOOTH_K,
    num_jobs=1,
):
    """
    run_linear_speeds()

    Runs a linear environment with varying speeds and collects data on place field
    widths and weights.

    Args:
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, a randomly selected seed is used for each
        repeat. Default is True.
    - max_time_min (float): Maximum time in minutes to run the environment for. Note
        that if learner is set to complete all trajectories, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is NUM_TRAJ_SPEED.
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
        - "PF_times" (3D np.ndarray): Array of start and end times for place fields
            computed from history with shape (speeds, fields, 2).
        - "BTSP_times" (2D np.ndarray): Array of BTSP event times with shape
            (speeds, events).
        - "num_BTSP" (1D np.ndarray): Number of BTSP events recorded for each speed.
        - "BTSP_applied_times" (2D np.ndarray): Array of BTSP event application times
            with shape (speeds, events).
        - "num_BTSP_applied" (1D np.ndarray): Number of BTSP events applied for each
            speed.
        - "norm_values" (2D np.ndarray): Weight normalization values used for each speed.
        - "end_times" (1D np.ndarray): End time of the experiment for each speed.
        - "seeds" (1D np.ndarray): Array of seeds for each speed.
    """

    speed_means = SPEED_MEANS

    # product of means and seeds
    total = num_repeats * len(speed_means)
    n_jobs = min(num_jobs, total)

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        seeds = np.arange(seed, seed + num_repeats)
    else:
        seeds = np.sort(np.random.choice(10000, size=num_repeats, replace=False))

    iterations = itertools.product(speed_means, seeds)

    kwargs = {
        "speed_std": 0,
        "max_time_min": max_time_min,
        "test_speed_mean": params_util.SPEED_MEAN_LINEAR,
        "test_speed_std": params_util.SPEED_MEAN_LINEAR,
        "k": k,
        "no_logs": True,
    }

    if num_jobs > 1:
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(run_linear_speed)(speed_mean=speed_mean, seed=seed, **kwargs)
            for speed_mean, seed in tqdm(iterations, total=total)
        )
        _, speed_dicts = zip(*outputs)
    else:
        speed_dicts = list()
        for speed_mean, seed in tqdm(iterations, total=total):
            _, speed_dict = run_linear_speed(speed_mean=speed_mean, seed=seed, **kwargs)
            speed_dicts.append(speed_dict)

    speed_data = aggregate_from_data_dicts(speed_dicts)

    return speed_data


def plot_linear_speed_PFs(
    speed_data=None,
    examples=SPEED_EXAMPLES,
    PF_type="history",
    plot_type="all",
    seed=True,
    **kwargs,
):
    """
    plot_linear_speed_PFs()

    Plots place fields for different speeds on the linear track.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see run_linear_speeds()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
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
        speed_data = run_linear_fct("speeds", overwrite=False, seed=seed)

    if plot_type == "examples":
        keep_seed = speed_data["seeds"].min()
        for key, vals in [("seeds", [keep_seed]), ("speed_means", examples)]:
            speed_data = gen_util.get_filtered_np_data_dict(
                speed_data,
                key,
                values=vals,
                skip_keys=["PF_centers", "PC_place_centers"],
            )

        Pyrs = get_linear_Pyrs()
        _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)  # to add plot markers

        k = metrics.SMOOTH_K if PF_type == "history" else 1

        ax = paper_plot_fcts.plot_linear_speed_PF_examples(
            speed_data, Ag=Ag, PF_type=PF_type, k=k, **kwargs
        )
    elif plot_type == "all":
        ax = paper_plot_fcts.plot_linear_speed_PF_widths(
            speed_data, mark_examples=examples, PF_type=PF_type, **kwargs
        )
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def run_linear_shift(
    learner=None,
    target_shift=0,
    i=0,
    speed_std=0,
    max_time_min=5,
    max_num_traj=5,
    k=metrics.SMOOTH_K,
    no_logs=True,
    seed=True,
):
    """
    run_linear_shift()

    Runs and collects data for a single linear speed experiment.

    Args:
    - learner (Learner, optional): Learner object. If None, new learner object is
        created and run before shift is evaluated. Default is None.
    - target_shift (float): Target shift for the experiment.
        Default is 0.
    - i (int): Index for the experiment run. Default is 0.
    - speed_std (float): Standard deviation of speed for the experiment.
        Default is 0.
    - max_time_min (float): Maximum time in minutes to run the environment for. Note
        that if learner is set to complete all trajectories, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is 5.
    - max_num_traj (int): Maximum number of trajectories to run for assessing place
        field. Default is 5.
    - BTSP_on (int): Trajectory on which to enable. Later trajectories allow more
        time Default is 5.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is metrics.SMOOTH_K.
    - no_logs (bool): Whether to disable logging. Default is True.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    Returns:
    - data_dict (dict): Dictionary containing the results of the experiment under keys:
        - "target_shift": Target shift for the experiment.
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
        - "end_time": End time of the experiment.
        if seed:
        - "seed": Seed for the experiment.
    """

    if seed:
        seed = PAPER_SEED + i
        gen_util.seed_all(seed)

    # initial_shift_dict = None
    if learner is None:
        learner, _ = run_linear_speed(
            speed_mean=params_util.SPEED_MEAN_LINEAR,
            i=0,
            speed_std=speed_std,
            max_time_min=max_time_min,
            k=k,
            no_logs=no_logs,
            seed=False,
        )

    num_BTSP_applied = len(
        learner.Pyrs.SomaticCompartment.get_BTSP_steps(
            applied_only=True, apply_step=True
        )
    )
    if num_BTSP_applied != 1:
        raise RuntimeError("Learner does not have exactly one BTSP event.")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Target position"
        )
        learner.Pyrs.Agent.shift_target_position(target_shift)

    for i in range(5):
        learner = run_linear(
            learner.Pyrs,
            max_time_min=max_time_min,
            max_num_traj=2,
            max_num_target_reaches=2,
            no_logs=no_logs,
            seed=False,
        )

        num_BTSP_applied = len(
            learner.Pyrs.SomaticCompartment.get_BTSP_steps(
                applied_only=True, apply_step=True
            )
        )
        if num_BTSP_applied == 2:
            break

    if num_BTSP_applied not in [1, 2]:
        raise RuntimeError(
            "Expected exactly one or two BTSP events to occur, "
            f"but found {num_BTSP_applied}."
        )

    # t_start = learner.Pyrs.Agent.t
    run_linear(
        learner.Pyrs,
        max_time_min=max_time_min,
        max_num_traj=max_num_traj,
        max_num_target_reaches=max_num_traj,
        no_logs=no_logs,
        seed=False,
    )

    num_BTSP_applied_after = len(
        learner.Pyrs.SomaticCompartment.get_BTSP_steps(
            applied_only=True, apply_step=True
        )
    )

    additional = num_BTSP_applied_after - num_BTSP_applied
    if additional > 0:
        raise RuntimeError(f"Expected no new BTSP events, but {additional} occurred.")

    data_dict = gather_learner_data(learner, seed=seed, k=k, target_shift=target_shift)

    return learner, data_dict


def run_linear_shifts(
    seed=True, max_time_min=5, num_repeats=1, k=metrics.SMOOTH_K, num_jobs=1
):
    """
    run_linear_shifts()

    Runs a linear environment with varying target position shifts and collects data
    on place field widths and weights.

    Args:
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - max_time_min (float): Maximum time in minutes to run the environment for. Note
        that if learner is set to complete all trajectories, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is 5.
    - num_repeats (int): Number of repeats for the experiment. Default is 1.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is metrics.SMOOTH_K.
    - num_jobs (int): Number of parallel jobs to run. Default is 1.

    Returns:
    - shift_data (dict): Dictionary containing:
        - "target_shifts" (1D np.ndarray): Array of target position shifts used in the
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
        - "BTSP_times" (2D np.ndarray): Array of BTSP event times with shape
            (shifts, events).
        - "num_BTSP" (1D np.ndarray): Number of BTSP events recorded for each shift.
        - "BTSP_applied_times" (2D np.ndarray): Array of BTSP event application times
            with shape (shifts, events).
        - "num_BTSP_applied" (1D np.ndarray): Number of BTSP events applied for each
            shift.
        - "end_times" (1D np.ndarray): End time of the experiment for each shift.
        - "norm_values" (2D np.ndarray): Weight normalization values used for each speed.
        - "seeds" (1D np.ndarray): Array of seeds for each shift.
    """

    target_shifts = TARGET_SHIFTS

    # product of means and seeds
    total = num_repeats * len(target_shifts)
    n_jobs = min(num_jobs, total)
    iterations = itertools.product(target_shifts, range(num_repeats))

    kwargs = {
        "max_time_min": max_time_min,
        "k": k,
        "no_logs": True,
    }

    learner, initial_shift_dict = run_linear_speed(
        speed_mean=params_util.SPEED_MEAN_LINEAR, speed_std=0, seed=seed, **kwargs
    )
    kwargs["seed"] = False

    if initial_shift_dict["num_BTSP_applied"] != 1:
        raise RuntimeError("Initial run did not produce exactly one BTSP event.")

    if num_jobs > 1:
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(run_linear_shift)(
                target_shift=target_shift, i=i, learner=learner, **kwargs
            )
            for target_shift, i in tqdm(iterations, total=total)
        )
        _, shift_dicts = zip(*outputs)
    else:
        shift_dicts = list()
        for target_shift, i in tqdm(iterations, total=total):
            _, shift_dict = run_linear_shift(
                target_shift=target_shift, i=i, learner=copy.deepcopy(learner), **kwargs
            )
            shift_dicts.append(shift_dict)

    shift_dict = aggregate_from_data_dicts(shift_dicts)

    return shift_dict


def plot_linear_shift_PFs(
    shift_data=None, examples=SHIFT_EXAMPLES, plot_cmap=False, plot_type="all", **kwargs
):
    """
    plot_linear_shift_PFs()

    Plots place fields for different target shifts on the linear track.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see run_linear_shifts()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - examples (list): List of example target shifts. Default is SHIFT_EXAMPLES.
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
        shift_data = run_linear_fct("shifts", overwrite=False)

    Pyrs = get_linear_Pyrs()
    _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)  # to add plot markers

    if plot_type == "examples":
        start, end = gen_util.get_value_index_range(
            shift_data["num_BTSP_applied"], 1, single_range_only=True
        )

        one_BTSP_pos_range = [
            shift_data["target_shifts"][start] + Ag.target_position[0],
            shift_data["target_shifts"][end - 1] + Ag.target_position[0],
        ]

        shift_data = gen_util.get_filtered_np_data_dict(
            shift_data,
            "target_shifts",
            values=examples,
            skip_keys=["PF_centers", "PC_place_centers"],
        )

        ax = paper_plot_fcts.plot_linear_shift_PF_examples(
            shift_data,
            Ag=Ag,
            plot_cmap=plot_cmap,
            mark_pos_range=one_BTSP_pos_range,
            **kwargs,
        )

    elif plot_type == "all":
        ax = paper_plot_fcts.plot_target_shift_PFs(
            shift_data, Ag=Ag, mark_examples=examples, plot_cmap=plot_cmap, **kwargs
        )

    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def run_linear_fct(fct_name="speeds", overwrite=False, seed=True, num_jobs=1):
    """
    run_linear_fct()

    Runs a specified linear function (either 'speeds' or 'shifts'),
    loading an existing data dictionary if it exists or rerunning the experiment.

    Args:
    - fct_name (str): Name of the function to run. Options are 'speeds' or
        'shifts'. Default is 'speeds'.
    - overwrite (bool): Whether to overwrite existing data. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, seed run_linear_speeds() or run_linear_shifts()
        for details. Default is True.

    Returns:
    - data_dict (dict): Dictionary containing the results of the experiment.
    """

    seed_str = ""
    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        seed_str = f"_{seed}"

    if fct_name == "speeds":
        fct = run_linear_speeds
        data_name = "speed_data"
        above = 1
    elif fct_name == "shifts":
        fct = run_linear_shifts
        data_name = "shift_data"
        above = 2
    else:
        raise ValueError(f"fct_name '{fct_name}' not recognized.")

    save_path = Path(get_fig_directory(), f"{data_name}{seed_str}.npz")
    if overwrite:
        gen_util.delete_np_dict(save_path)
    data_dict = gen_util.load_np_dict(save_path)

    if data_dict is None:
        print("Running...")
        start_time = time.perf_counter()
        data_dict = fct(seed=seed, num_jobs=num_jobs)
        gen_util.save_np_dict(save_path, data_dict)
        gen_util.get_duration_str(start_time, log=True)

    log_num_BTSP_if_above(data_dict["num_BTSP"], above=above)
    if "norm_values" in data_dict.keys():
        log_max_normalization_value(data_dict["norm_values"])

    return data_dict


def get_openfield_Pyrs(
    corridor=False,
    n=None,
    log_BTSP=True,
    always_log_teleportation=True,
    init_reward_only=False,
    init_teleport_pairs=None,
    horizontal_in_from_left=True,
    seed=True,
):
    """
    get_openfield_Pyrs()

    Initializes Pyr parameters for openfield environment.

    Args:
    - corridor (bool): Whether to use the corridor environment. Default is False.
    - n (int, optional): Number of reward objects to initialize in the openfield environment.
        If None, defaults are used. Default is None.
    - log_BTSP (bool): Whether to log BTSP events. Default is True.
    - always_log_teleportation (bool): Whether to always log teleportation events.
        Default is True.
    - init_reward_only (bool): Whether to initialize the agent with only reward
        inputs. Only implemented for corridor environment. Default is False.
    - init_teleport_pairs (3D np.ndarray, optional): Teleport pairs to initialized
        with shape (pair, port, coord). Default is None.
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
        "init_random_reward_obj": n,
        "horizontal_in_from_left": horizontal_in_from_left,
    }
    if init_teleport_pairs is not None:
        env_params["init_teleport_pairs"] = init_teleport_pairs
    if not corridor:
        env_params["init_random_walls"] = 4
        env_params["init_random_novel_obj"] = 0
        env_params["init_random_teleport_pairs"] = 0
        env_params["min_dist"] = 0.15

    env_params = params_util.get_env_params(environment=environment, **env_params)

    if init_reward_only:
        if not corridor:
            raise NotImplementedError(
                "'init_reward_only' is only implemented for corridor environment."
            )
        agent_params = params_util.get_agent_params(
            environment=environment,
            reward_factor=1,
            no_target_factor=0,
            always_log_teleportation=always_log_teleportation,
        )
    else:
        agent_params = params_util.get_agent_params(
            environment=environment, always_log_teleportation=always_log_teleportation
        )

    Pyr_params = params_util.get_Pyr_params(
        n=n,
        environment=environment,
        log_BTSP=log_BTSP,
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
    max_num_steps=OPENFIELD_MAX_STEPS,
    max_time_min=None,
    teleportation_enabled=False,
    min_time_since_last_BTSP_applied=60 * 6,
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
    - max_num_steps (int): Maximum number of steps to run the environment. Note
        that if learner is set to complete final trajectory, max_num_steps will be
        exceeded to complete any incomplete trajectories. Default is OPENFIELD_MAX_STEPS.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step. Note
        that if learner is set to complete final trajectory, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is None.
    - teleportation_enabled (bool, optional): Whether to enable teleportation. Default
        is False.
    - min_time_since_last_BTSP_applied (float): Minimum time in seconds since last
        BTSP event was applied to end the experiment.
        Default is 360 seconds (6 minutes).
    - no_logs (bool): Whether to disable logging. Default is False.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - teleport_kwargs (dict): Keyword arguments to pass to get_openfield_Pyrs() for
        teleportation initialization. Default is empty dict.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    Returns:
    - learner (Learner): The learner object after training.
    """

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    if Pyrs is None:
        Pyrs = get_openfield_Pyrs(
            corridor=True,
            init_reward_only=True,
            seed=False,
            log_BTSP=not (no_logs),
            always_log_teleportation=not (no_logs),
            **teleport_kwargs,
        )

    Pyrs.Agent.set_no_target_factor(2)

    if max_time_min is not None:
        max_num_steps = int(max_time_min * 60 / Pyrs.Agent.dt)

    min_steps_after_BTSP = int(
        np.ceil(min_time_since_last_BTSP_applied / Pyrs.Agent.dt)
    )

    learner = run_manager.learn_openfield_BTSP(
        Pyrs_or_learner=Pyrs,
        max_num_steps=max_num_steps,
        corridor=True,
        teleportation_enabled=teleportation_enabled,
        min_steps_after_BTSP=min_steps_after_BTSP,
        no_logs=no_logs,
        **kwargs,
    )

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
            max_num_steps=OPENFIELD_MAX_STEPS,
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
            Pyrs, obj_s=30, **kwargs
        )
    elif plot_type == "timeseries":
        ax = paper_plot_fcts.plot_single_neuron_rate_timeseries(
            Pyrs.SomaticCompartment,
            mark_traj_idxs=[EX_TRAJ_IDX],
            BTSP_kernel_lw=0.02,
            **kwargs,
        )
    elif plot_type == "BTSP_kernel_timeseries":
        if kernel_time is None:
            kernel_time = (18.5, 41.5)  # manually identified for paper example
        t_start, t_end = kernel_time

        ax = paper_plot_fcts.plot_single_neuron_rate_timeseries(
            Pyrs.SomaticCompartment,
            t_start=t_start,
            t_end=t_end,
            in_min=False,
            num_ticks=13,
            BTSP_kernel_lw=2.5,
            **kwargs,
        )
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return ax


def get_openfield_corridor_repeat_run_params(
    i=0, seed=True, max_runs=100, max_num_steps=None, teleport=False
):
    """
    get_openfield_corridor_repeat_run_params()

    Obtains parameters for the openfield corridor experiments based on the run index.

    Args:
    - i (int): Run index. Default is 0.
    - seed (bool or int): Whether to seed the random number generator with the paper
        seed or seed to use. If False, experiment is not seeded. Default is True.
    - max_runs (int): Maximum number of runs for generating seeds. Default is 100.
    - max_num_steps (int, optional): Maximum number of steps to run the environment.
        If None, uses OPENFIELD_MAX_STEPS or OPENFIELD_TELEPORT_REPEAT_STEPS based on
        teleport argument. Default is None.
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
        run_kwargs["min_time_since_last_BTSP_applied"] = 10 * 60  # better coverage
        run_kwargs["max_num_steps"] = max_num_steps or OPENFIELD_TELEPORT_REPEAT_STEPS

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
        run_kwargs["max_num_steps"] = max_num_steps or OPENFIELD_MAX_STEPS

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
        - "BTSP_times" (2D np.ndarray): Array of times at which BTSP events were
            applied for each repeat.
        - "num_BTSP" (1D np.ndarray): Number of BTSP events that were recorded for each
            run.
        - "BTSP_applied_times" (2D np.ndarray): Array of times at which BTSP events
            were applied for each repeat.
        - "num_BTSP_applied" (1D np.ndarray): Number of BTSP events that were applied
            for each run.
        - "visit_times" (1D np.ndarray): Array of times at which the agent visited the
            reward location.
        - "num_visits" (int): Number of visits to the reward location for each run.
        - "norm_values" (1D np.ndarray): Normalization values used for each run.
        - "end_times" (1D np.ndarray): End time of the experiment for each run.
        - "seeds": Array of random seeds used for each run.
    """

    data_dicts = list()
    for i in tqdm(range(num_repeats)):
        run_seed, run_kwargs = get_openfield_corridor_repeat_run_params(i, seed=seed)
        learner = run_openfield_corridor(seed=run_seed, no_logs=True, **run_kwargs)
        data_dict = gather_learner_data(learner, seed=run_seed, position_name="reward")
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
    max_num_steps=OPENFIELD_MAX_STEPS,
    min_num_teleports=4,
    disable_teleportation_between=True,
    min_time_since_last_BTSP_applied=60 * 6,
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
    - max_num_steps (int): Maximum number of steps to run the environment per learning
        call until the target number of teleportation events is met. Note that if
        learner is set to complete final trajectory, max_num_steps will be exceeded to
        complete any incomplete trajectories. Default is OPENFIELD_MAX_STEPS.
    - min_num_teleports (int): Minimum number of teleportation events to occur
        before stopping the experiment. Default is 4.
    - disable_teleportation_between (int): If True, teleportation is disabled for
        6 minutes after a BTSP event (increases probability that PFs can be
        calculated for each BTSP event if teleportation events induce BTSP events).
        Default is True.
    - min_time_since_last_BTSP_applied (float): Minimum time in seconds since last
        BTSP event was applied to end the experiment. Default is 60 * 6.
    - no_logs (bool): Whether to suppress logging. Default is False.
    - teleport_kwargs (dict): Additional keyword arguments passed to
        run_openfield_corridor() for teleportation initialization.

    Returns:
    - learner (Learner): The learner object after training.
    """

    if seed:
        seed = PAPER_SEED if isinstance(seed, bool) else seed
        gen_util.seed_all(seed)

    learner = run_openfield_corridor(
        seed=False, Pyrs=Pyrs, no_logs=no_logs, teleport_kwargs=teleport_kwargs
    )

    if not no_logs:
        print("\nTeleportation enabled.")

    learner.Agent.set_reward_factor(0.5)
    learner.Agent.set_no_target_factor(0.5)
    learner.Agent.allow_teleportation(True)

    disable_teleportation = 0
    if disable_teleportation_between:
        disable_teleportation = int(360 / learner.Agent.dt)

    updater = run_manager.TeleportRewardUpdater(
        learner.Agent,
        Pyrs=learner.Pyrs_for_weights,
        disable_teleportation=disable_teleportation,
    )

    num_teleports = len(learner.Agent.teleportation_df)

    while True:
        run_manager.learn_openfield_BTSP(
            Pyrs_or_learner=learner,
            use_Hebbian=False,
            max_num_steps=max_num_steps,
            corridor=True,
            updater=updater,
            no_logs=no_logs,
        )

        num_additional_teleports = len(learner.Agent.teleportation_df) - num_teleports
        if num_additional_teleports >= min_num_teleports:
            break
        elif not no_logs:
            print(
                f"\nContinuing to reach at least {min_num_teleports} "
                f"teleportation events ({num_additional_teleports} so far)."
            )

    min_steps_after_BTSP = int(
        np.ceil(min_time_since_last_BTSP_applied / learner.Agent.dt)
    )
    if min_steps_after_BTSP:
        run_manager.continue_learn_to_min_steps_after_BTSP_applied(
            learner,
            min_steps=min_steps_after_BTSP,
            updater=updater,
            no_logs=no_logs,
            max_num_steps=min_steps_after_BTSP * 10,
        )

    learner.Agent.allow_teleportation(True)

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
            max_num_steps=OPENFIELD_MAX_STEPS,
            min_num_teleports=4,
            disable_teleportation_between=True,
        )

    if plot_type == "summary":
        axes = paper_plot_fcts.plot_openfield_teleportation_summary(learner, **kwargs)
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

    return axes


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
        - "BTSP_times": Array of times at which BTSP events were applied.
        - "num_BTSP": Number of BTSP events that were applied in total.
        - "visit_times": Array of times at which the agent visited the reward location.
        - "num_visits": Number of visits to the reward location.
        - "teleportation_times": Array of times at which teleportation events occurred.
        - "num_teleportations": Number of teleportation events that occurred.
        - "init_teleport_pairs": Array of teleportation pairs coordinates initialized.
        - "horizontal_in_from_left": Array of teleportation in port directions.
        - "norm_values": Normalization values used for each run.
        - "end_times": End time of the experiment for each run.
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
            learner, seed=run_seed, position_name="reward", teleport=True
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
            obj_s=6,
            no_teleport=False,
        )
    else:
        raise ValueError(f"Plot type not recognized: {plot_type}.")

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
        gen_util.delete_np_dict(save_path)
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

    if str(fig).isnumeric():
        fig = int(fig)

    panel = panel.upper()

    fig_dict = {
        1: {
            "A": {"fct": "schematic"},
            "B": {"fct": plot_linear, "plot_type": "environment"},
            "C": {"fct": "schematic"},
            "D": {"fct": plot_linear, "plot_type": "BTSP_kernel"},
        },
        2: {
            "A": {"fct": plot_linear, "plot_type": "summary"},
            "B": {"fct": plot_linear, "plot_type": "place_fields"},
            "C": {"fct": plot_linear, "plot_type": "binned_rates"},
        },
        "2S": {
            "A": {
                "fct": plot_linear,
                "plot_type": "neural_activity",
                "inhibition": "balanced",
            },
            "B": {
                "fct": plot_linear,
                "plot_type": "neural_activity",
                "inhibition": "insufficient",
            },
            "C": {
                "fct": plot_linear,
                "plot_type": "neural_activity",
                "inhibition": "excessive",
            },
        },
        3: {
            "A": {"fct": plot_linear_speed_PFs, "plot_type": "examples"},
            "B": {"fct": plot_linear_speed_PFs, "plot_type": "all"},
        },
        "3S": {
            "A": {
                "fct": plot_linear_speed_PFs,
                "plot_type": "examples",
                "PF_type": "weights",
            },
            "B": {
                "fct": plot_linear_speed_PFs,
                "plot_type": "all",
                "PF_type": "weights",
            },
        },
        4: {
            "A": {"fct": plot_linear_shift_PFs, "plot_type": "examples"},
        },
        "4S": {
            "A": {"fct": plot_linear_shift_PFs, "plot_type": "all"},
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
                "fig_side": 2.5,
            },
            "C": {"fct": plot_openfield_corridors, "plot_type": "timelines"},
            "D": {"fct": plot_openfield_corridors, "plot_type": "PFs"},
        },
        6: {
            "A": {"fct": plot_openfield_teleportation, "plot_type": "summary"},
        },
        "6S": {
            "A": {
                "fct": plot_openfield_teleportations,
                "plot_type": "timelines",
                "fig_width": 8,
            },
            "B": {"fct": plot_openfield_teleportations, "plot_type": "PFs"},
        },
    }

    if fig not in fig_dict.keys():
        raise KeyError(f"Unknown figure: {fig}")

    if panel not in fig_dict[fig].keys():
        raise ValueError(f"Unknown panel for figure {fig}: {panel}")

    fct_kwargs = fig_dict[fig][panel]
    fct = fct_kwargs.pop("fct")

    if fct == "schematic":
        ax = None
        print("Schematic plot.")
    else:
        ax = fct(*args, **fct_kwargs, **kwargs)

    if save and ax is not None:
        key = f"{fig}{panel}"
        fig = ax.ravel()[0].figure if isinstance(ax, np.ndarray) else ax.figure
        plot_util.save_figure(fig, key, no_timestamp=True, dpi=600)

    return ax
