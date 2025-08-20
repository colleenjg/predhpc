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

SPEED_EXAMPLES = [0.15, 0.25, 0.35]
SHIFT_EXAMPLES = [1.0, -0.4, -3.0]

SMOOTH_K = 1


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


def get_linear_Pyrs(
    scale=params_util.SCALE_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD,
    wait_at_end=0,
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
    - wait_at_end (int): Number of steps to wait at the end of the environment.
        Default is 0.
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
        wait_at_end=wait_at_end,
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
    - max_num_steps (int): Maximum number of steps to run the environment.
        Default is 3800.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is None.
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
        Pyrs = get_linear_Pyrs(seed=False, wait_at_end=int(15 / params_util.DT))

    if max_time_min is not None:
        max_num_steps = int(max_time_min * 60 / Pyrs.Agent.dt)

    learner = run_manager.learn_1D_BTSP(
        Pyrs, BTSP_on=BTSP_on, max_num_steps=max_num_steps, plot=False, **kwargs
    )

    return learner


def plot_linear_summary(learner=None, max_time_min=1.8, **kwargs):
    """
    plot_linear_summary()

    Plots summary of linear experiment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is 1.8.

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


def plot_linear_place_fields(learner, max_time_min=1.8, **kwargs):
    """
    plot_linear_place_fields(learner)

    Plots place weights and place field for a linear environment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is 1.8.

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


def plot_linear_binned_rates(learner, max_time_min=1.8, **kwargs):
    """
    plot_linear_binned_rates(learner)

    Plots binned rates for linear experiment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is 1.8.

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
    max_time_min=20,
    max_num_traj=20,
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
    - max_time_min (float): Maximum time in minutes to run the environment.
        Default is 20.
    - max_num_traj (int): Maximum number of trajectories to run. Default is 20.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is SMOOTH_K.
    - no_logs (bool): Whether to disable logging. Default is True.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    Returns:
    - data_dict (dict): Dictionary containing the results of the experiment under keys:
        - "speed_mean": Mean speed for the experiment.
        - "PFs": Place fields computed from history.
        - "PF_widths": Place field widths.
        - "PF_centers": Place field centers.
        - "PC_weights": Place cell input weights.
        - "PC_weight_widths": Place cell input weight widths.
        - "PC_place_centers": Place cell centers.
        if seed:
        - "seed": Seed for the experiment.
    """

    if seed:
        gen_util.seed_all(PAPER_SEED + i)

    Pyrs = get_linear_Pyrs(
        speed_mean=speed_mean,
        speed_std=0,
        log_BTSP=False,
        wait_at_end=0,
        seed=False,
    )
    learner = run_linear(
        Pyrs,
        max_time_min=max_time_min,
        max_num_traj=max_num_traj,
        max_num_target_reaches=max_num_traj,
        no_logs=no_logs,
        BTSP_on=5,  # to ensure sufficient time before BTSP with very high running speeds
        seed=False,
    )

    # compute place fields from history
    history_kwargs = {
        "method": "history",
        "t_start": ext_util.choose_t_start_after_BTSP(
            Pyrs.SomaCompartment, next_trajectory=True
        ),
    }

    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)
    PFs, PF_centers = metrics.evaluate_PFs(Pyrs, **history_kwargs)

    data_dict = {
        "speed_mean": speed_mean,
        "PFs": PFs,
        "PF_widths": metrics.compute_PF_width(Pyrs, k=k, **history_kwargs),
        "PF_centers": PF_centers,
        "PC_weight_widths": metrics.compute_PF_width(Pyrs),
        "PC_weights": learner.get_recorded_weights()["weights"][:, 0],
        "PC_place_centers": PCs.place_cell_centers[:, 0],
        "PF_smoothing": k,
    }

    if seed:
        data_dict["seed"] = PAPER_SEED + i

    return data_dict


def run_linear_speeds(
    seed=True, max_time_min=20, num_repeats=1, k=SMOOTH_K, num_jobs=1
):
    """
    run_linear_speeds()

    Runs a linear environment with varying speeds and collects data on place field
    widths and weights.

    Args:
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.
    - max_time_min (float): Maximum time in minutes to run the environment.
        Default is 20.
    - num_repeats (int): Number of repeats for the experiment. Default is 1.
    - k (int): Smoothing factor for measuring place field width from firingrate history.
        Default is SMOOTH_K.
    - num_jobs (int): Number of parallel jobs to run. Default is 1.

    Returns:
    - speed_data (dict): Dictionary containing:
        - "speed_means": Array of speed means used in the experiment.
        - "PFs": List of place fields computed from history for each speed mean.
        - "PF_widths": List of place field widths for each speed mean.
        - "PF_centers": List of place field centers.
        - "PC_weights": List of place cell input weights for each speed mean.
        - "PC_weight_widths": List of place cell input weight widths for each speed mean.
        - "PC_place_centers": Array of place cell centers.
        - "PF_smoothing": Smoothing factor for place fields.
        if seed:
        - "seeds": Array of seeds for each run.
    """

    speed_means = gen_util.get_rounded_linspace(0.05, 0.55, 41)

    # product of means and seeds
    total = num_repeats * len(speed_means)
    n_jobs = min(num_jobs, total)
    iterations = itertools.product(speed_means, range(num_repeats))

    kwargs = {
        "max_time_min": max_time_min,
        "k": k,
        "no_logs": True,
        "seed": seed,
    }

    if num_jobs > 1:
        speed_dicts = Parallel(n_jobs=n_jobs)(
            delayed(run_linear_speed)(speed_mean=speed_mean, i=i, **kwargs)
            for speed_mean, i in tqdm(iterations, total=total)
        )
    else:
        speed_dicts = list()
        for speed_mean, i in tqdm(iterations, total=total):
            speed_dict = run_linear_speed(speed_mean=speed_mean, i=i, **kwargs)
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

    for key, vals in [("speed_means", to_plot), ("seeds", [PAPER_SEED])]:
        speed_data = gen_util.get_filtered_np_data_dict(
            speed_data,
            key,
            values=vals,
            skip_keys=["PF_centers", "PC_place_centers"],
        )

    Pyrs = get_linear_Pyrs()
    _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)  # to add plot markers

    ax1D = paper_plot_fcts.plot_linear_speed_PF_examples(
        speed_data, Ag=Ag, PF_type=PF_type, **kwargs
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


def run_linear_shifts(seed=True):
    """
    run_linear_shifts()

    Runs a linear environment with varying target position shifts and collects data
    on place field widths and weights.

    Args:
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - shift_data (dict): Dictionary containing:
        - "target_shifts": Array of target position shifts used in the experiment.
        - "PF_widths": List of place field widths for each target shift.
        - "PF_weights": List of place field weights for each target shift.
        - "PC_place_centers": Array of place cell centers for each target shift.
    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

    shift_data = {
        "target_shifts": gen_util.get_rounded_linspace(-3.6, 2.4, 61),
        "PF_widths": list(),
        "PF_weights": list(),
    }

    Pyrs = get_linear_Pyrs(speed_std=0, log_BTSP=False, wait_at_end=0)
    orig_learner = run_linear(Pyrs, max_time_min=1.2, no_logs=True)
    for shift in tqdm(shift_data["target_shifts"]):
        if seed:
            gen_util.seed_all(PAPER_SEED)
        learner = copy.deepcopy(orig_learner)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="Target position"
            )
            learner.Pyrs.Agent.move_target_position(shift)
        learner = run_linear(learner, max_time_min=1.2, no_logs=True)
        shift_data["PF_widths"].append(metrics.compute_PF_width(Pyrs))
        shift_data["PF_weights"].append(learner.get_recorded_weights()["weights"][:, 0])

    num = max([len(wei) for wei in shift_data["PF_weights"]])
    shift_data["PF_weights"] = [
        np.pad(wei, ((0, num - len(wei)), (0, 0)), "constant", constant_values=np.nan)
        for wei in shift_data["PF_weights"]
    ]

    shift_data = {key: np.asarray(val) for key, val in shift_data.items()}

    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)
    shift_data["PC_place_centers"] = PCs.place_cell_centers[:, 0]

    return shift_data


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

    return data_dict


def get_openfield_Pyrs(
    corridor=False,
    log_BTSP=True,
    seed=True,
):
    """
    get_openfield_Pyrs()

    Initializes Pyr parameters for openfield environment.

    Args:
    - environment (str): The environment to initialize. Default is "openfield".
    - log_BTSP (bool): Whether to log BTSP events. Default is True.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - Pyrs (Pyr): Pyr object initialized with the specified parameters.
    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

    environment = "openfield_corridor" if corridor else "openfield"

    Pyr_params = params_util.get_Pyr_params(
        environment=environment,
        log_BTSP=log_BTSP,
    )

    Pyrs = run_manager.init_env_objects(
        Pyr_params=Pyr_params,
        environment=environment,
        plot=False,
    )

    return Pyrs


def run_openfield_corridor(
    Pyrs=None,
    max_num_steps=15000,
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
    - max_num_steps (int): Maximum number of steps to run the environment.
        Default is 3800.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is None.
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
        Pyrs = get_openfield_Pyrs(corridor=True, seed=False)

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
            max_num_steps=15000,
            weight_recording_freq=1000,
            teleportation_enabled=False,
        )
        Pyrs = learner.Pyrs

    axes = paper_plot_fcts.plot_openfield_components(Pyrs, **kwargs)

    return axes


def plot_openfield_corridor_weights(Pyrs=None, **kwargs):
    """
    plot_openfield_corridor_weights()

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
            max_num_steps=15000,
            weight_recording_freq=1000,
            teleportation_enabled=False,
        )
        Pyrs = learner.Pyrs

    sub_ax = paper_plot_fcts.plot_openfield_weights(Pyrs, **kwargs)

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
            max_num_steps=15000,
            weight_recording_freq=1000,
            teleportation_enabled=False,
        )
        Pyrs = learner.Pyrs

    sub_ax = paper_plot_fcts.plot_single_neuron_rate_timeseries(
        Pyrs.SomaCompartment, **kwargs
    )

    return sub_ax


def plot_figure_panel(*args, fig=1, panel="A", save=True, **kwargs):
    """
    plot_figure_panel()

    Plots a specific panel of a figure.

    Args:
    - *args: Positional arguments passed to the plotting function.
    - fig (int): Figure number.
    - panel (str): Panel letter (A, B, C, etc.).
    - save (bool): Whether to save the figure.
    - **kwargs: Keyword arguments passed to the plotting function.

    Returns:
    - ax: The axes object for the plotted panel.
    """

    fig = int(fig)
    panel = panel.upper()

    fig_dict = {
        1: {
            "A": "schematic",
            "B": plot_linear_environment,
            "C": "schematic",
            "D": plot_BTSP_kernel,
        },
        2: {
            "A": plot_linear_summary,
            "B": plot_linear_place_fields,
            "C": plot_linear_binned_rates,
        },
        3: {
            "A": plot_linear_speed_PF_examples,
            "B": plot_linear_speed_PF_widths,
        },
        4: {
            "A": plot_linear_shift_PF_examples,
            "B": plot_target_shift_PFs,
        },
        5: {
            "A-D": plot_openfield_corridor_components,
            "E": plot_openfield_corridor_weights,
            "F": plot_openfield_corridor_timeseries,
        },
    }

    if fig not in fig_dict.keys():
        raise KeyError(f"Unknown figure: {fig}")

    if panel not in fig_dict[fig].keys():
        raise ValueError(f"Unknown panel for figure {fig}: {panel}")

    fct = fig_dict[fig][panel]
    if fct == "schematic":
        ax = None
        print("Schematic plot.")
    else:
        ax = fig_dict[fig][panel](*args, **kwargs)

    if save and ax is not None:
        key = f"{fig}{panel}"
        fig = ax.ravel()[0].figure if isinstance(ax, np.ndarray) else ax.figure
        plot_util.save_figure(fig, key, no_timestamp=True)

    return ax
