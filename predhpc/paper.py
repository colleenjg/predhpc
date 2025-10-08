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
gen_util.seed_all(PAPER_SEED)

SPEED_MEANS = gen_util.get_rounded_linspace(0.05, 0.4, 29)  # (0.05, 0.55, 41)
SPEED_EXAMPLES = [0.15, 0.25, 0.35]

TARGET_SHIFTS = gen_util.get_rounded_linspace(-3.6, 2.4, 61)
SHIFT_EXAMPLES = [1.0, 0, -0.4, -3.0]

SMOOTH_K = 5  # across 120 samples

NUM_TRAJ_SPEED = 20
OPENFIELD_MAX_STEPS = 20000
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


def initialize_paper_parameters(**kwargs):
    """
    initialize_paper_parameters()

    Initializes parameters for paper.

    Keywords args:
    - **kwargs: Parameters passed to paper_plot_fcts.initialize_paper_parameters().
    """

    suppress_warnings()
    paper_plot_fcts.initialize_paper_parameters(**kwargs)


def gather_PF_info(learner, k=SMOOTH_K, t_start=None):
    """
    gather_PF_info(learner)

    Gathers information about place fields (PFs) from the given learner object using
    various metrics ("weights", "smoothed_weights", "history").

    Args:
    - learner (Learner): The learner object to gather information from.
    - k (int): The smoothing factor for place field width computation from firingrate
        history. Default is SMOOTH_K.
    - t_start (float): The start time for history evaluation. Default is None.

    Returns:
    - PF_info (dict): A dictionary containing gathered PF information:
        - "PC_place_centers": Place cell centers.
        - "PC_weights": Place cell input weights.
        - "PC_weight_widths": Last place cell input weight widths.
        - "PC_smoothed_weights": Smoothed place cell input weights.
        - "PC_smoothed_weight_widths": Last smoothed place cell input weight widths.
        - "PFs": Place fields computed from history.
        - "PF_centers": Place field centers.
        - "PF_widths": Last place field widths.
    """

    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(learner.Pyrs)

    PF_info = dict()

    # from input weights
    PC_place_centers = PCs.place_cell_centers[:, 0]
    sorter = np.argsort(PC_place_centers)
    PF_info["PC_place_centers"] = PC_place_centers[sorter]

    PF_info["PC_weights"] = learner.get_recorded_weights()["weights"][:, 0, sorter]
    PF_info["PC_weight_widths"] = metrics.compute_PF_width(learner.Pyrs, k=1)

    # from input weights, smoothed
    PF_info["PC_smoothed_weights"], _ = metrics.get_smoothed_weights(
        PF_info["PC_weights"], PF_info["PC_place_centers"], PCs.widths
    )
    PF_info["PC_smoothed_weight_widths"] = metrics.compute_PF_width(
        learner.Pyrs, k=1, method="smoothed_weights"
    )

    # from history
    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(learner.Pyrs)
    PFs, PF_centers = metrics.evaluate_PFs(
        learner.Pyrs, method="history", t_start=t_start
    )
    PF_info["PFs"] = PFs
    PF_info["PF_centers"] = PF_centers
    PF_info["PF_widths"] = metrics.compute_PF_width(
        learner.Pyrs, k=k, method="history", t_start=t_start
    )

    return PF_info


def get_linear_Pyrs(
    scale=params_util.SCALE_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD,
    wait_after_trajectory=0,
    log_BTSP=True,
    seed=True,
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
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - Pyrs (Pyr): Pyr object initialized with the specified parameters.
    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

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
        environment="linear",
        log_BTSP=log_BTSP,
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


def plot_linear_environment(Ag=None, **kwargs):
    """
    plot_linear_environment()

    Plots the environment for the linear experiment.

    Args:
    - Ag (Agent, optional): Agent object to plot. If None, a new Agent object is created.
        Default is None.

    Keywords args:
    - **kwargs: Additional keyword arguments passed to
        paper_plot_fcts.plot_linear_environment().

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted environment.
    """

    if Ag is None:
        Pyrs = get_linear_Pyrs()
        _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    sub_ax = paper_plot_fcts.plot_linear_environment(Ag, **kwargs)

    return sub_ax


def plot_BTSP_kernel(Pyrs=None, **kwargs):
    """
    plot_BTSP_kernel()

    Plots the BTSP kernel for the given Pyrs object.

    Args:
    - Pyrs (Pyr): Pyr object containing the agent and place cells.

    Keywords args:
    - **kwargs: Additional keyword arguments passed to
        paper_plot_fcts.plot_BTSP_kernel().

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted BTSP kernel.
    """

    if Pyrs is None:
        Pyrs = get_linear_Pyrs()

    sub_ax = paper_plot_fcts.plot_BTSP_kernel(Pyrs, **kwargs)

    return sub_ax


def run_linear(
    Pyrs=None, max_num_steps=3800, max_time_min=None, BTSP_on=None, seed=True, **kwargs
):
    """
    run_linear()

    Runs a linear environment with the specified Pyr parameters.

    Args:
    - Pyrs (Pyr, optional): Pyr object with initialized parameters. If None,
        a new Pyr object is created with default parameters.
    - max_num_steps (int): Maximum number of steps to run the environment. Note that
        if learner is set to finish final trajectory, max_num_steps will be exceeded
        to complete any incomplete trajectories. Default is 3800.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step. Note
        that if learner is set to finish final trajectory, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is None.
    - BTSP_on (int): Trajectory on which to turn on BTSP. 1 for first trajectory.
        Default is None.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

    if Pyrs is None:
        Pyrs = get_linear_Pyrs(
            seed=False, wait_after_trajectory=params_util.WAIT_LINEAR
        )

    if max_time_min is not None:
        max_num_steps = int(max_time_min * 60 / Pyrs.Agent.dt)

    learner = run_manager.learn_1D_BTSP(
        Pyrs, BTSP_on=BTSP_on, max_num_steps=max_num_steps, plot=False, **kwargs
    )

    return learner


def plot_linear_summary(learner=None, max_time_min=2.0, **kwargs):
    """
    plot_linear_summary()

    Plots summary of linear experiment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step. Note
        that if learner is set to finish final trajectory, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is 2.0.

    Keywords args:
    - **kwargs: Additional keyword arguments passed to
        paper_plot_fcts.plot_linear_summary().

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with linear data plotted.
    """

    if learner is None:
        learner = run_linear(max_time_min=max_time_min)

    ax1D = paper_plot_fcts.plot_linear_summary(learner, **kwargs)

    return ax1D


def plot_linear_place_fields(learner, max_time_min=2.0, **kwargs):
    """
    plot_linear_place_fields(learner)

    Plots place weights and place field for a linear environment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step. Note
        that if learner is set to finish final trajectory, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is 2.0.

    Keywords args:
    - **kwargs: Additional keyword arguments passed to
        paper_plot_fcts.plot_linear_place_fields().

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with the linear place fields plotted.
    """

    if learner is None:
        learner = run_linear(max_time_min=max_time_min)

    ax1D = paper_plot_fcts.plot_linear_place_fields(learner, **kwargs)

    return ax1D


def plot_linear_binned_rates(learner, max_time_min=2.0, **kwargs):
    """
    plot_linear_binned_rates(learner)

    Plots binned rates for linear experiment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step. Note
        that if learner is set to finish final trajectory, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is 2.0.

    Keywords args:
    - **kwargs: Additional keyword arguments passed to
        paper_plot_fcts.plot_linear_binned_rates().

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with linear binned rates plotted.
    """

    if learner is None:
        learner = run_linear(max_time_min=max_time_min)

    ax1D = paper_plot_fcts.plot_linear_binned_rates(learner, **kwargs)

    return ax1D


def run_linear_speed(
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    i=0,
    speed_std=params_util.SPEED_STD,
    test_speed_mean=None,
    test_speed_std=None,
    max_time_min=NUM_TRAJ_SPEED,
    max_num_traj=NUM_TRAJ_SPEED,
    k=SMOOTH_K,
    no_logs=True,
    seed=True,
):
    """
    run_linear_speed()

    Runs and collects data for a single linear speed experiment.

    Args:
    - speed_mean (float): Mean speed for the experiment.
        Default is params_util.SPEED_MEAN_LINEAR.
    - i (int): Index for the experiment run. Default is 0.
    - max_time_min (float, optional): Maximum time in minutes to run the environment
        for assessing place field. Note that if learner is set to complete all
        trajectories, max_time_min will be exceeded to complete any incomplete
        trajectories. Default is NUM_TRAJ_SPEED.
    - max_num_traj (int): Maximum number of trajectories to run for assessing place
        field. Default is NUM_TRAJ_SPEED.
    - BTSP_on (int): Trajectory on which to enable. Later trajectories allow more
        time Default is 5.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is SMOOTH_K.
    - no_logs (bool): Whether to disable logging. Default is True.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

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
        - "num_BTSP_applied": Number of BTSP events that were applied in total for
            each target shift.
        - "max_norm_value": Maximum weight normalization value used.
        if seed:
        - "seed": Seed for the experiment.
    """

    if seed:
        seed_value = PAPER_SEED + i
        gen_util.seed_all(seed_value)

    Pyrs = get_linear_Pyrs(
        speed_mean=speed_mean,
        speed_std=speed_std,
        log_BTSP=False,
        wait_after_trajectory=0,
        seed=False,
    )

    for i in range(5):
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

    t_start = Pyrs.Agent.t
    Pyrs.Agent.set_speed(mean=test_speed_mean, std=test_speed_std)
    run_linear(
        Pyrs,
        max_time_min=max_time_min,
        max_num_traj=max_num_traj,
        max_num_target_reaches=max_num_traj,
        no_logs=no_logs,
        seed=False,
    )

    data_dict = gather_PF_info(learner, k=k, t_start=t_start)

    num_BTSP_applied = len(
        Pyrs.SomaticCompartment.get_BTSP_steps(applied_only=True, apply_step=True)
    )
    norm_values = learner.Pyrs.SomaticCompartment.get_normalization_values("PCs")[1]
    max_norm_value = norm_values.max() if len(norm_values) > 0 else np.nan

    data_dict["speed_mean"] = speed_mean
    data_dict["num_BTSP_applied"] = num_BTSP_applied
    data_dict["max_norm_value"] = max_norm_value

    if seed:
        data_dict["seed"] = seed_value

    return learner, data_dict


def run_linear_speeds(
    seed=True, max_time_min=NUM_TRAJ_SPEED, num_repeats=1, k=SMOOTH_K, num_jobs=1
):
    """
    run_linear_speeds()

    Runs a linear environment with varying speeds and collects data on place field
    widths and weights.

    Args:
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.
    - max_time_min (float): Maximum time in minutes to run the environment for. Note
        that if learner is set to complete all trajectories, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is NUM_TRAJ_SPEED.
    - num_repeats (int): Number of repeats for the experiment. Default is 1.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is SMOOTH_K.
    - num_jobs (int): Number of parallel jobs to run. Default is 1.

    Returns:
    - speed_data (dict): Dictionary containing:
        - "speed_means": Array of speed means used in the experiment.
        - "PC_place_centers": Array of place cell centers.
        - "PC_weights": Array of place cell input weights.
        - "PC_weight_widths": Array of place cell input weight widths.
        - "PC_smoothed_weights": Array of smoothed place cell input weights.
        - "PC_smoothed_weight_widths": Array of smoothed place cell input weight widths.
        - "PFs": Array of place fields computed from history.
        - "PF_centers": Array of place field centers.
        - "PF_widths": Array of place field widths.
        - "num_BTSP_applied": Number of BTSP events applied for each speed.
        - "max_norm_values": Maximum weight normalization value used for each speed.
        if seed:
        - "seeds": Array of seeds for each run.
    """

    speed_means = SPEED_MEANS

    # product of means and seeds
    total = num_repeats * len(speed_means)
    n_jobs = min(num_jobs, total)
    iterations = itertools.product(speed_means, range(num_repeats))

    kwargs = {
        "speed_std": 0,
        "max_time_min": max_time_min,
        "test_speed_mean": params_util.SPEED_MEAN_LINEAR,
        "test_speed_std": params_util.SPEED_MEAN_LINEAR,
        "k": k,
        "no_logs": True,
        "seed": seed,
    }

    if num_jobs > 1:
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(run_linear_speed)(speed_mean=speed_mean, i=i, **kwargs)
            for speed_mean, i in tqdm(iterations, total=total)
        )
        _, speed_dicts = zip(*outputs)
    else:
        speed_dicts = list()
        for speed_mean, i in tqdm(iterations, total=total):
            _, speed_dict = run_linear_speed(speed_mean=speed_mean, i=i, **kwargs)
            speed_dicts.append(speed_dict)

    speed_data = dict()
    for key in speed_dicts[0].keys():
        if key in ["PF_centers", "PC_place_centers"]:
            speed_data[key] = speed_dicts[0][key]
        else:
            speed_data[key] = np.asarray(
                [speed_dict[key] for speed_dict in speed_dicts]
            )
    speed_data["speed_means"] = speed_data.pop("speed_mean")
    speed_data["max_norm_values"] = speed_data.pop("max_norm_value")

    if "seed" in speed_data.keys():
        speed_data["seeds"] = speed_data.pop("seed")

    return speed_data


def plot_linear_speed_PF_examples(
    speed_data=None, to_plot=SPEED_EXAMPLES, PF_type="history", **kwargs
):
    """
    plot_linear_speed_PF_examples()

    Plots examples of place fields for different speeds on the linear track.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see run_linear_speeds()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - to_plot (list): List of speed means to plot. Default is SPEED_EXAMPLES.
    - PF_type (str): PF type to plot. Default is "history".

    Keywords args:
    - **kwargs: Additional keyword arguments passed to
        paper_plot_fcts.plot_linear_speed_PF_examples().

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with example place fields plotted.
    """

    if speed_data is None:
        speed_data = run_linear_fct("linear_speeds", overwrite=False)

    for key, vals in [("seeds", [PAPER_SEED]), ("speed_means", to_plot)]:
        speed_data = gen_util.get_filtered_np_data_dict(
            speed_data,
            key,
            values=vals,
            skip_keys=["PF_centers", "PC_place_centers"],
        )

    Pyrs = get_linear_Pyrs()
    _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)  # to add plot markers

    k = SMOOTH_K if PF_type == "history" else 1

    ax1D = paper_plot_fcts.plot_linear_speed_PF_examples(
        speed_data, Ag=Ag, PF_type=PF_type, k=k, **kwargs
    )

    return ax1D


def plot_linear_speed_PF_widths(
    speed_data=None, mark_examples=SPEED_EXAMPLES, PF_type="history", **kwargs
):
    """
    plot_linear_speed_PF_widths()

    Plots the relationship between speed means and place weight widths for the linear
    experiment.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see run_linear_speeds()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - mark (list): List of speed means to mark. Default is list().
    - PF_type (str): PF type to plot. Default is "history".

    Keywords args:
    - **kwargs: Additional keyword arguments passed to
        paper_plot_fcts.plot_linear_speed_PF_widths().

    Returns:
    - sub_ax (plt.Axes): The subplot with the speed means and place weight widths
        plotted.
    """

    if speed_data is None:
        speed_data = run_linear_fct("linear_speeds", overwrite=False)

    sub_ax = paper_plot_fcts.plot_linear_speed_PF_widths(
        speed_data, mark_examples=mark_examples, PF_type=PF_type, **kwargs
    )

    return sub_ax


def run_linear_shift(
    learner=None,
    target_shift=0,
    i=0,
    speed_std=0,
    max_time_min=5,
    max_num_traj=5,
    k=SMOOTH_K,
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
        Default is SMOOTH_K.
    - no_logs (bool): Whether to disable logging. Default is True.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

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
        - "max_norm_value": Maximum weight normalization value used.
        if seed:
        - "seed": Seed for the experiment.
    """

    if seed:
        seed_value = PAPER_SEED + i
        gen_util.seed_all(seed_value)

    initial_shift_dict = None
    if learner is None:
        learner, initial_shift_dict = run_linear_speed(
            speed_mean=params_util.SPEED_MEAN_LINEAR,
            i=0,
            speed_std=speed_std,
            max_time_min=max_time_min,
            k=k,
            no_logs=no_logs,
            seed=seed,
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

    t_start = learner.Pyrs.Agent.t
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

    data_dict = gather_PF_info(learner, k=k, t_start=t_start)

    # pad data if num of BTSP events is below 2
    PF_keys = ["PC_weights", "PC_smoothed_weights", "PFs"]
    if num_BTSP_applied == 1:
        for PF_key in PF_keys:
            data = data_dict[PF_key]
            if PF_key == "PFs":
                data_dict[PF_key] = np.full_like(data, np.nan)
            else:
                data_dict[PF_key] = np.concatenate(
                    [data, np.full((1, data.shape[1]), np.nan)], axis=0
                )

    # add initial results, if available
    if initial_shift_dict is not None:
        for PF_key in PF_keys:
            data_dict[PF_key] = np.concatenate(
                [initial_shift_dict[PF_key][:1], data_dict[PF_key]], axis=0
            )

    norm_values = learner.Pyrs.SomaticCompartment.get_normalization_values("PCs")[1]
    max_norm_value = norm_values.max() if len(norm_values) > 0 else np.nan

    data_dict["target_shift"] = target_shift
    data_dict["num_BTSP_applied"] = num_BTSP_applied
    data_dict["max_norm_value"] = max_norm_value

    if seed:
        data_dict["seed"] = seed_value

    return learner, data_dict


def run_linear_shifts(seed=True, max_time_min=5, num_repeats=1, k=SMOOTH_K, num_jobs=1):
    """
    run_linear_shifts()

    Runs a linear environment with varying target position shifts and collects data
    on place field widths and weights.

    Args:
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.
    - max_time_min (float): Maximum time in minutes to run the environment for. Note
        that if learner is set to complete all trajectories, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is 5.
    - num_repeats (int): Number of repeats for the experiment. Default is 1.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is SMOOTH_K.
    - num_jobs (int): Number of parallel jobs to run. Default is 1.

    Returns:
    - shift_data (dict): Dictionary containing:
        - "target_shifts": Array of target position shifts used in the experiment.
        - "PC_place_centers": Array of place cell centers.
        - "PC_weights": Array of place cell input weights.
        - "PC_weight_widths": Array of place cell input weight widths.
        - "PC_smoothed_weights": Array of smoothed place cell input weights.
        - "PC_smoothed_weight_widths": Array of smoothed place cell input weight widths.
        - "PFs": Array of place fields computed from history.
        - "PF_centers": Array of place field centers.
        - "PF_widths": Array of place field widths.
        - "num_BTSP_applied": Number of BTSP events that were applied in total for
            each target shift.
        - "max_norm_value": Maximum weight normalization value used for each target shift.
        if seed:
        - "seeds": Array of seeds for each run.
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
        speed_mean=params_util.SPEED_MEAN_LINEAR, i=0, speed_std=0, seed=seed, **kwargs
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

    shift_dict = dict()
    for key in shift_dicts[0].keys():
        if key in ["PF_centers", "PC_place_centers"]:
            shift_dict[key] = shift_dicts[0][key]
        else:
            shift_dict[key] = np.asarray(
                [shift_dict[key] for shift_dict in shift_dicts]
            )

    # concatenate initial data to front of arrays
    for key in ["PC_weights", "PC_smoothed_weights", "PFs"]:
        initial_data = np.tile(initial_shift_dict[key][:1], (len(target_shifts), 1, 1))
        shift_dict[key] = np.concatenate([initial_data, shift_dict[key]], axis=1)

    shift_dict["target_shifts"] = shift_dict.pop("target_shift")
    shift_dict["max_norm_values"] = shift_dict.pop("max_norm_value")

    if seed:
        shift_dict["seeds"] = np.full(len(target_shifts), PAPER_SEED)

    return shift_dict


def plot_linear_shift_PF_examples(
    shift_data=None, to_plot=SHIFT_EXAMPLES, plot_cmap=False, **kwargs
):
    """
    plot_linear_shift_PF_examples()

    Plots examples of place fields for different target shifts on the linear track.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see run_linear_shifts()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - to_plot (list): List of target shifts to plot. Default is SHIFT_EXAMPLES.
    - plot_cmap (bool): Whether to plot the place field colormap instead of a peak shift
        plot. Default is False.

    Keyword args:
    - **kwargs: Keyword arguments passed to
        paper_plot_fcts.plot_linear_shift_PF_examples().

    Returns:
    - axes (2D np.ndarray of plt.Axes): Subplots with example place fields plotted.
    """

    if shift_data is None:
        shift_data = run_linear_fct("linear_shifts", overwrite=False)

    shift_data = gen_util.get_filtered_np_data_dict(
        shift_data,
        "target_shifts",
        values=to_plot,
        skip_keys=["PF_centers", "PC_place_centers"],
    )

    Pyrs = get_linear_Pyrs()
    _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)  # to add plot markers

    axes = paper_plot_fcts.plot_linear_shift_PF_examples(
        shift_data, Ag=Ag, plot_cmap=plot_cmap, **kwargs
    )

    return axes


def plot_target_shift_PFs(
    shift_data=None, mark_examples=SHIFT_EXAMPLES, plot_cmap=False, **kwargs
):
    """
    plot_target_shift_PFs()

    Plots the relationship between target shifts and place field weights for the linear
    experiment.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see run_linear_shifts()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - mark_examples (list): List of target shifts to plot arrows for.
        Default is SHIFT_EXAMPLES.
    - plot_cmap (bool): Whether to plot the place field colormap instead of a peak shift
        plot. Default is False.

    Keyword args:
    - **kwargs: Keyword arguments passed to paper_plot_fcts.plot_target_shift_PFs().

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with target shifts and place field
        weights plotted.
    """

    if shift_data is None:
        shift_data = run_linear_fct("linear_shifts", overwrite=False)

    Pyrs = get_linear_Pyrs()
    _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)  # to add plot markers

    ax1D = paper_plot_fcts.plot_target_shift_PFs(
        shift_data, Ag=Ag, mark_examples=mark_examples, plot_cmap=plot_cmap, **kwargs
    )

    return ax1D


def log_max_normalization_value(norm_values):
    """
    log_max_normalization_value(norm_values)

    Logs the maximum weight normalization value recorded.

    Args:
    - norm_values (np.ndarray): Array of normalization values to log.
    """

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


def run_linear_fct(fct_name="linear_speeds", overwrite=False, seed=True, num_jobs=1):
    """
    run_linear_fct()

    Runs a specified linear function (either 'linear_speeds' or 'linear_shifts'),
    loading an existing data dictionary if it exists or rerunning the experiment.

    Args:
    - fct_name (str): Name of the function to run. Options are 'linear_speeds' or
        'linear_shifts'. Default is 'linear_speeds'.
    - overwrite (bool): Whether to overwrite existing data. Default is False.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - data_dict (dict): Dictionary containing the results of the experiment.
    """

    if fct_name == "linear_speeds":
        fct = run_linear_speeds
        data_name = "speed_data"
    elif fct_name == "linear_shifts":
        fct = run_linear_shifts
        data_name = "shift_data"
    else:
        raise ValueError(f"fct_name '{fct_name}' not recognized.")

    save_path = Path(ratinabox.figure_directory, f"{data_name}_{PAPER_SEED}.npz")
    if overwrite:
        gen_util.delete_np_dict(save_path)
    data_dict = gen_util.load_np_dict(save_path)

    if data_dict is None:
        print("Running...")
        start_time = time.perf_counter()
        data_dict = fct(seed=seed, num_jobs=num_jobs)
        gen_util.save_np_dict(save_path, data_dict)
        gen_util.get_duration_str(start_time, log=True)

    if "max_norm_values" in data_dict.keys():
        log_max_normalization_value(data_dict["max_norm_values"])

    return data_dict


def get_openfield_Pyrs(
    corridor=False,
    log_BTSP=True,
    init_reward_only=False,
    seed=True,
):
    """
    get_openfield_Pyrs()

    Initializes Pyr parameters for openfield environment.

    Args:
    - environment (str): The environment to initialize. Default is "openfield".
    - log_BTSP (bool): Whether to log BTSP events. Default is True.
    - init_reward_only (bool): Whether to initialize the agent with only reward
        inputs. Only implemented for corridor environment. Default is False.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - Pyrs (Pyr): Pyr object initialized with the specified parameters.
    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

    environment = "openfield_corridor" if corridor else "openfield"

    agent_params = None
    if init_reward_only:
        if not corridor:
            raise NotImplementedError(
                "'init_reward_only' is only implemented for corridor environment."
            )
        agent_params = params_util.get_agent_params(
            environment=environment,
            reward_factor=1,
            no_target_factor=0,
        )

    Pyr_params = params_util.get_Pyr_params(
        environment=environment,
        log_BTSP=log_BTSP,
    )

    Pyrs = run_manager.init_env_objects(
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
    weight_recording_freq=1000,
    teleportation_enabled=False,
    seed=True,
    **kwargs,
):
    """
    run_openfield_corridor()

    Runs a corridor openfield environment with the specified Pyr parameters.

    Args:
    - Pyrs (Pyr, optional): Pyr object with initialized parameters. If None,
        a new Pyr object is created with default parameters.
    - max_num_steps (int): Maximum number of steps to run the environment. Note
        that if learner is set to finish final trajectory, max_num_steps will be
        exceeded to complete any incomplete trajectories. Default is OPENFIELD_MAX_STEPS.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step. Note
        that if learner is set to finish final trajectory, max_time_min will be
        exceeded to complete any incomplete trajectories. Default is None.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    Returns:
    - learner (Learner): The learner object after training.
    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

    if Pyrs is None:
        Pyrs = get_openfield_Pyrs(corridor=True, init_reward_only=True, seed=False)

    Pyrs.Agent.set_no_target_factor(2)

    if max_time_min is not None:
        max_num_steps = int(max_time_min * 60 / Pyrs.Agent.dt)

    learner = run_manager.learn_openfield_BTSP(
        Pyrs_or_learner=Pyrs,
        corridor=True,
        max_num_steps=max_num_steps,
        weight_recording_freq=weight_recording_freq,
        teleportation_enabled=teleportation_enabled,
        **kwargs,
    )

    return learner


def plot_openfield_corridor_components(Pyrs=None, **kwargs):
    """
    plot_openfield_corridor_components()

    Plots the components of an openfield corridor experiment.

    Args:
    - Pyrs (Pyr): Pyr object for openfield corridor.
        If None, a new Pyr object is created.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to
        run_manager.plot_openfield_corridor_components().

    Returns:
    - axes (np.ndarray of plt.Axes): Array of subplots with openfield components plotted.
    """

    if Pyrs is None:
        learner = run_manager.learn_openfield_BTSP(
            Pyrs_or_learner=Pyrs,
            corridor=True,
            max_num_steps=OPENFIELD_MAX_STEPS,
            weight_recording_freq=1000,
            teleportation_enabled=False,
        )
        Pyrs = learner.Pyrs

    axes = paper_plot_fcts.plot_openfield_components(
        Pyrs, traj_idx=EX_TRAJ_IDX, **kwargs
    )

    return axes


def plot_openfield_corridor_PFs(Pyrs=None, **kwargs):
    """
    plot_openfield_corridor_PFs()

    Plots the weights of the Pyr neuron in the openfield corridor.

    Args:
    - Pyrs (Pyr): Pyr object for openfield corridor.
        If None, a new Pyr object is created.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to
        run_manager.plot_openfield_corridor_weights().

    Returns:
    - sub_ax (plt.Axes): Subplot with openfield corridor weights plotted.
    """

    if Pyrs is None:
        learner = run_manager.learn_openfield_BTSP(
            Pyrs_or_learner=Pyrs,
            corridor=True,
            max_num_steps=OPENFIELD_MAX_STEPS,
            weight_recording_freq=1000,
            teleportation_enabled=False,
        )
        Pyrs = learner.Pyrs

    sub_ax = paper_plot_fcts.plot_openfield_PFs(Pyrs, **kwargs)

    return sub_ax


def plot_openfield_corridor_BTSP_trajectory(Pyrs=None, **kwargs):
    """
    plot_openfield_corridor_BTSP_trajectory()

    Plots the trajectory of the agent in the openfield corridor around the first BTSP
    event.

    Args:
    - Pyrs (Pyr): Pyr object for openfield corridor.
        If None, a new Pyr object is created.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to
        run_manager.plot_openfield_corridor_trajectory().

    Returns:
    - sub_ax (plt.Axes): Subplot with openfield corridor trajectory plotted.
    """

    if Pyrs is None:
        learner = run_manager.learn_openfield_BTSP(
            Pyrs_or_learner=Pyrs,
            corridor=True,
            max_num_steps=OPENFIELD_MAX_STEPS,
            weight_recording_freq=1000,
            teleportation_enabled=False,
        )
        Pyrs = learner.Pyrs

    sub_ax = paper_plot_fcts.plot_openfield_corridor_BTSP_trajectory(Pyrs, **kwargs)

    return sub_ax


def plot_openfield_corridor_timeseries(Pyrs=None, **kwargs):
    """
    plot_openfield_corridor_timeseries()

    Plots the rate timeseries of the Pyr neuron in the openfield corridor.

    Args:
    - Pyrs (Pyr): Pyr object for openfield corridor.
        If None, a new Pyr object is created.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to
        run_manager.plot_single_neuron_rate_timeseries().

    Returns:
    - sub_ax (plt.Axes): Subplot with openfield corridor weights plotted.
    """

    if Pyrs is None:
        learner = run_manager.learn_openfield_BTSP(
            Pyrs_or_learner=Pyrs,
            corridor=True,
            max_num_steps=OPENFIELD_MAX_STEPS,
            weight_recording_freq=1000,
            teleportation_enabled=False,
        )
        Pyrs = learner.Pyrs

    sub_ax = paper_plot_fcts.plot_single_neuron_rate_timeseries(
        Pyrs.SomaticCompartment, mark_traj_idxs=[EX_TRAJ_IDX], **kwargs
    )

    return sub_ax


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
            "A": ("schematic", dict()),
            "B": (plot_linear_environment, dict()),
            "C": ("schematic", dict()),
            "D": (plot_BTSP_kernel, dict()),
        },
        2: {
            "A": (plot_linear_summary, dict()),
            "B": (plot_linear_place_fields, dict()),
            "C": (plot_linear_binned_rates, dict()),
        },
        3: {
            "A": (plot_linear_speed_PF_examples, dict()),
            "B": (plot_linear_speed_PF_widths, dict()),
        },
        "3S": {
            "A": (plot_linear_speed_PF_examples, {"PF_type": "weights"}),
            "B": (plot_linear_speed_PF_widths, {"PF_type": "weights"}),
        },
        4: {
            "A": (plot_linear_shift_PF_examples, dict()),
            "B": (plot_target_shift_PFs, dict()),
        },
        5: {
            "A-D": (plot_openfield_corridor_components, dict()),
            "E": (plot_openfield_corridor_PFs, dict()),
            "F": (plot_openfield_corridor_BTSP_trajectory, dict()),
            "G": (plot_openfield_corridor_timeseries, dict()),
        },
        "5S": {
            "A": (plot_openfield_corridor_PFs, {"PF_type": "weights"}),
        },
    }

    if fig not in fig_dict.keys():
        raise KeyError(f"Unknown figure: {fig}")

    if panel not in fig_dict[fig].keys():
        raise ValueError(f"Unknown panel for figure {fig}: {panel}")

    fct, fct_kwargs = fig_dict[fig][panel]
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
