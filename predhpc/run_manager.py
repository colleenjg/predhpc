#!/usr/bin/env python3

import copy
from typing import Any, Sequence
import warnings

from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import markers as mpl_markers
import numpy as np
from tqdm import tqdm  # type: ignore[import]

from predhpc import agent, env, plot_fcts
from predhpc.neurons import (
    riab_neurons,
    learning_neurons,
    two_comp_neurons,
    object_neurons,
)
from predhpc.util import gen_util, plot_util, params_util


### 2D FUNCTIONS ###


def extract_objects_from_Pyrs(Pyrs):
    """
    extract_objects_from_Pyrs(Pyrs)

    Extract objects from a Pyrs object.

    Args:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer

    Returns:
    - Env (env.Environment): Environment
    - Ag (agent.Agent): Agent
    - PCs (riab_neurons.PlaceCells): Place cells
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer
    - Objs (object_neurons.ObjectCells or object_neurons.ObjectInstanceCells):
        Object cells
    """

    Env = Pyrs.Agent.Environment

    Ag = Pyrs.Agent

    if isinstance(Pyrs, two_comp_neurons.TwoCompLayer):
        Obj_key = list(Pyrs.DendriteCompartment.inputs.keys())[-1]
        Objs = Pyrs.DendriteCompartment.inputs[Obj_key]["layer"]

        PC_key = list(Pyrs.SomaCompartment.inputs.keys())[0]
        PCs = Pyrs.SomaCompartment.inputs[PC_key]["layer"]
    else:
        Objs = None
        PC_key = list(Pyrs.inputs.keys())[0]
        PCs = Pyrs.inputs[PC_key]["layer"]

    return Env, Ag, PCs, Pyrs, Objs


def plot_2D_initial_conditions(Pyrs, num_samples=10, autosave: bool | None = None):
    """
    plot_2D_initial_conditions(Pyrs)

    Plot initial conditions for a 2D environment experiment.

    Args:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer
    - num_samples (int, optional): Number of samples to plot. Default is 10.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - fields_axes (2D np.ndarray): Array of subplots with place fields plotted, with
        shape (num_layers, num_samples).
    - aggreg_ax1D (1D np.ndarray): Array of subplots with environment and aggregated
        fields plotted, with shape (3,).
    """

    Env, _, PCs, Pyrs, Objs = extract_objects_from_Pyrs(Pyrs)

    # Plot fields
    if Objs is None:
        num_cols = min(PCs.n, num_samples)
        neurons = [PCs]
    else:
        num_cols = min(max(PCs.n, Objs.n), num_samples)
        neurons = [Objs, PCs]

    fields_fig, fields_axes = plt.subplots(
        len(neurons), num_cols, figsize=(num_cols * 2, len(neurons) * 2), squeeze=False
    )
    title_i = max(0, num_cols // 2 - 1)
    for i, NeuronLayer in enumerate(neurons):
        if num_cols >= NeuronLayer.n:
            chosen_neurons = np.arange(NeuronLayer.n)
        else:
            chosen_neurons = np.sort(np.random.choice(NeuronLayer.n, num_cols))
        ax1D = fields_axes[i, : len(chosen_neurons)]
        NeuronLayer.plot_rate_map(
            chosen_neurons=chosen_neurons, ax=ax1D, no_legend=True, autosave=False
        )
        name = NeuronLayer.name.replace("_", " ")
        ax1D[title_i].set_title(f"{name} rate maps", fontsize="x-large")  # type: ignore[attr-defined]

    # Plot aggregated fields
    aggreg_fields, aggreg_ax1D = plt.subplots(1, 3, figsize=(9, 3))

    for sub_ax in aggreg_ax1D[:2]:
        Env.plot_environment(sub_ax=sub_ax, no_legend=True, autosave=False)
    aggreg_ax1D[0].set_title("Environment")

    plot_fcts.plot_overlayed_rate_maps(
        PCs, method="max", colorbar=False, sub_ax=aggreg_ax1D[1], autosave=False
    )
    aggreg_ax1D[1].set_title("Overlayed place fields")

    PCs.plot_place_cell_locations(sub_ax=aggreg_ax1D[2], autosave=False)
    aggreg_ax1D[2].set_title("Place cell centers")

    if autosave:
        plot_util.save_figure(fields_fig, "openfield_init_fields", save=autosave)
        plot_util.save_figure(aggreg_fields, "openfield_init_aggreg", save=autosave)

    return fields_axes, aggreg_ax1D


def init_2D_env_objects(
    env_params: dict[str, Any] | None = None,
    agent_params: dict[str, Any] | None = None,
    PC_params: dict[str, Any] | None = None,
    Pyr_params: dict[str, Any] | None = None,
    Obj_params: dict[str, Any] | None = None,
    environment="openfield",
    two_compartment: bool = True,
    autosave: bool | None = None,
    plot: bool = True,
):
    """
    init_2D_env_objects()

    Initialize objects for a 2D environment experiment, and obtain Pyrs.

    Args:
    - env_params (dict, optional): Parameters for the environment. Default is None.
    - agent_params (dict, optional): Parameters for the agent. Default is None.
    - PC_params (dict, optional): Parameters for the place cells. Default is
        None.
    - Pyr_params (dict, optional): Parameters for the Pyr. neurons. Default is None.
    - Obj_params (dict, optional): Parameters for the object neurons. Default is None.
    - two_compartment (bool, optional): Whether to use two-compartment model.
        Default is True.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.
    - plot (bool, optional): Whether to plot the environment and neurons. Default is
        True.

    Returns:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer
    """
    env_params = env_params or params_util.get_env_params(environment=environment)
    agent_params = agent_params or params_util.get_agent_params(environment=environment)

    if environment == "tmaze":
        Env = env.TEnv(params=env_params)
        Ag = agent.TAgent(Env, params=agent_params)
    elif environment == "openfield":
        Env = env.OpenField(params=env_params)
        Ag = agent.OpenFieldAgent(Env, params=agent_params)
    else:
        raise ValueError(f"Invalid environment: {environment}")

    PC_params = PC_params or params_util.get_PC_params(environment=environment)
    PCs = riab_neurons.PlaceCells(Ag, params=PC_params)

    if two_compartment:
        Obj_params = Obj_params or params_util.get_Obj_params(environment=environment)
        if environment == "tmaze":
            Objs = object_neurons.ObjectCells(Ag, params=Obj_params)
        else:
            Objs = object_neurons.ObjectInstanceCells(Ag, params=Obj_params)
    else:
        if Obj_params is not None:
            warnings.warn("Obj_params will be ignored if two_compartment is False.")
        Objs = None

    if Pyr_params is None:
        n_kwargs = {"n": Objs.n} if two_compartment else dict()
        Pyr_params = params_util.get_Pyr_params(
            environment=environment,
            BTSP=True,
            NMDA=two_compartment,
            two_compartment=two_compartment,
            **n_kwargs,
        )

    if two_compartment:
        Pyr_params["soma_input_layers"] = [PCs]  # type: ignore[assignment]
        if Pyr_params["n"] is None:
            Pyr_params["n"] = Objs.n
        elif Pyr_params["n"] != Objs.n:
            raise ValueError(
                f"If provided, Pyr_params['n'] ({Pyr_params['n']}) must be equal to "
                f"Objs.n ({Objs.n})."
            )
    else:
        Pyr_params["input_layers"] = [PCs]  # type: ignore[assignment]

    if two_compartment:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No input layers")
            Pyrs = two_comp_neurons.TwoCompLayer(Ag, params=Pyr_params)

        Obj_to_Pyr_w = gen_util.get_weights(Objs.n, Pyrs.n)
        Pyrs.DendriteCompartment.add_input(Objs, w=Obj_to_Pyr_w)
        Pyrs.set_BTSP_learn(soma=True, dend=False)
    else:
        Pyrs = learning_neurons.BTSPLayer(Ag, params=Pyr_params)
        Pyrs.set_BTSP_learn()

    if plot:
        plot_2D_initial_conditions(Pyrs, autosave=autosave)

    return Pyrs


### 2D (OPENFIELD) FUNCTIONS ###


def learn_openfield_BTSP(
    Pyrs: learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer | None = None,
    max_num_steps: int = 10000,
    record_weights_at_BTSP: bool = True,
    use_Hebbian: bool = False,
    num_end_without_BTSP: int = 0,
    two_compartment: bool = True,
    autosave: bool | None = None,
    **init_kwargs,
) -> tuple[
    env.Environment,
    agent.ResetableAgent,
    object_neurons.ObjectCells | None,
    riab_neurons.PlaceCells,
    learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer,
]:
    """
    learn_openfield_BTSP()

    Run an openfield learning experiment with BTSP learning.

    Args:
    - max_steps (int, optional): Maximum number of steps to run. Default is 10000.
    - record_weights_at_BTSP (bool, optional): Whether to record weights at BTSP events.
        Default is True.
    - use_Hebbian (bool, optional): Whether to use Hebbian learning. Default is False.
    - num_end_without_BTSP (int, optional): Number of final steps to run without BTSP
        learning. Default is 0.
    - two_compartment (bool, optional): Whether to use two-compartment model. Default
        is True.
    - autosave (bool, optional): Whether to autosave. Default is None.

    Keyword Args:
    - **init_kwargs: Keyword arguments for init_2D_env_objects().

    Returns:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer
    - weights_at_BTSP (dict): Dictionary with keys "weights" and "steps" in which
        input weights from place cells and steps are recorded. None if
        record_weights_at_BTSP is False.
    """

    if Pyrs is None:
        Pyrs = init_2D_env_objects(
            two_compartment=two_compartment,
            environment="openfield",
            autosave=autosave,
            plot=False,
            **init_kwargs,
        )

    _, Ag, PCs, Pyrs, Objs = extract_objects_from_Pyrs(Pyrs)

    if two_compartment:
        Pyrs.set_BTSP_learn(soma=True, dend=False)
        Pyrs_for_weights = Pyrs.SomaCompartment
        Pyrs.set_learn(soma=use_Hebbian, dend=False, inhibit=False)
    else:
        Pyrs.set_BTSP_learn()
        Pyrs_for_weights = Pyrs
        Pyrs.set_learn(use_Hebbian)

    # run learning
    start_step = len(Pyrs_for_weights.history["t"])
    stop_BTSP = max(0, start_step + max_num_steps - num_end_without_BTSP)

    BTSP_stopped = False
    start_num_BTSP = len(Pyrs_for_weights.history["BTSP_events"])

    BTSP_steps, weights, steps = list(), list(), list()
    for i in tqdm(range(max_num_steps)):
        Ag.update(speed_fact=3, drift_to_random_strength_ratio=1)
        Objs.update()
        PCs.update()
        Pyrs.update()
        if record_weights_at_BTSP:
            num_BTSP = len(Pyrs_for_weights.history["BTSP_events"])
            if num_BTSP > len(BTSP_steps) + start_num_BTSP:
                BTSP_steps.append(len(Pyrs_for_weights.history["t"]))

        if not BTSP_stopped and i + start_step >= stop_BTSP:
            if two_compartment:
                Pyrs.set_BTSP_learn(soma=False, dend=False)
            else:
                Pyrs.set_BTSP_learn()
            print(f"BTSP blocked from step {i + start_step}.")
            BTSP_stopped = True

        if Pyrs_for_weights.BTSP_applied.any():
            weights.append(copy.deepcopy(Pyrs_for_weights.inputs["PCs"]["w"]))
            steps.append(len(Pyrs_for_weights.history["t"]))

    BTSP_steps = np.asarray(BTSP_steps)
    print(
        f"{len(BTSP_steps)} BTSP events recorded (allowed from steps "
        f"{start_step} to {stop_BTSP}): occurred between steps "
        f"{BTSP_steps.min()} and {BTSP_steps.max()}."
    )

    weights_at_BTSP = None
    if record_weights_at_BTSP:
        weights_at_BTSP = {"weights": np.asarray(weights), "steps": steps}

    return Pyrs, weights_at_BTSP


### 2D (T-MAZE) FUNCTIONS ###


def plot_T_maze(
    Ag: agent.TAgent,
    PCs: riab_neurons.PlaceCells,
    Pyrs_or_Objs: learning_neurons.BTSPLayer | object_neurons.ObjectCells,
    method: str = "groundtruth",
    autosave: bool | None = None,
):
    """
    plot_T_maze(Ag, PCs, Pyrs_or_Objs)

    Plot the T-maze environment:
        (1) Agent trajectories,
        (2) Place cell locations, and
        (3) Pyr. or Obj. overlayed rate maps.

    Args:
    - Ag (agent.Agent): Agent.
    - PCs (riab_neurons.PlaceCells): Place cells.
    - Pyrs_or_Objs (learning_neurons.BTSPLayer or object_neurons.ObjectCells):
        Pyrs layer.
    - method (str, optional): Method to use for plotting the Pyr. rate map. Default
        is "groundtruth".
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - axes (2D np.ndarray): Array of subplots with T maze information plotted, with
        shape (3, 1). See description for details.
    """

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), squeeze=False)
    ax1D = np.asarray(axes).ravel()

    # Plot trajectories on T-maze
    Ag.plot_trajectories(scale_cmap_per=False, s_2D=5, alpha=0.3, sub_ax=ax1D[0])
    ax1D[0].set_title("Trajectories")

    # Plot place cell locations on T-maze
    plot_fcts.plot_overlayed_rate_maps(
        PCs, sub_ax=ax1D[1], method="max", colorbar=False
    )
    PCs.plot_place_cell_locations(sub_ax=ax1D[1])
    ax1D[1].scatter(
        *Ag.target_position,
        marker=".",
        color="blue",
        s=18,
        zorder=5,
    )
    ax1D[1].set_title("Place cell rate maps")

    # Plot Pyr. rate map on T-maze
    if isinstance(Pyrs_or_Objs, learning_neurons.BTSPLayer):
        Pyrs_or_Objs.plot_rate_map(ax=ax1D[2], method=method)
        title = f"{Pyrs_or_Objs.name.replace('_', ' ')} rate map"  # type: ignore[attr-defined]
    else:
        plot_fcts.plot_overlayed_rate_maps(
            Pyrs_or_Objs,
            sub_ax=ax1D[2],
            method="max",
            colorbar=False,
            replot_env=True,
        )
        title = "Obj. rate map"
    ax1D[2].scatter(
        *Ag.target_position,
        marker=".",
        color="blue",
        s=18,
        zorder=5,
    )
    ax1D[2].set_title(title)

    plot_util.save_figure(fig, "T_maze", save=autosave)

    return axes


def learn_T_maze_BTSP(
    Pyrs: learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer | None = None,
    num_rewards: int = 200,
    max_num_steps: int = 10000,
    weight_recording_freq: int = 100,
    use_Hebbian: bool = False,
    BTSP_after_num_target_reaches: int = 2,
    two_compartment: bool = True,
    autosave: bool | None = None,
    **init_kwargs,
) -> tuple[
    env.Environment,
    agent.ResetableAgent,
    object_neurons.ObjectCells | None,
    riab_neurons.PlaceCells,
    learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer,
]:
    """
    learn_T_maze_BTSP()

    Run a T-maze learning experiment with BTSP learning.

    Args:
    - env_params (dict, optional): Parameters for the environment. Default is None.
    - agent_params (dict, optional): Parameters for the agent. Default is None.
    - PC_params (dict, optional): Parameters for the place cells. Default is
        None.
    - Pyr_params (dict, optional): Parameters for the Pyr. neurons. Default is None.
    - num_rewards (int, optional): Target number of rewards to reach. Default is 200.
    - max_steps (int, optional): Maximum number of steps to run. Default is 10000.
    - weight_recording_freq (int, optional): Frequency at which to record weights.
        Default is 100.
    - use_Hebbian (bool, optional): Whether to use Hebbian learning. Default is False.
    - BTSP_after_num_target_reaches (int, optional): Number of times to reach target
        before enabling BTSP learning. Default is 2.
    - autosave (bool, optional): Whether to autosave. Default is None.

    Keyword Args:
    - **init_kwargs: Keyword arguments for init_2D_env_objects().

    Returns:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer
    """

    if Pyrs is None:
        Pyrs = init_2D_env_objects(
            two_compartment=two_compartment,
            environment="tmaze",
            autosave=autosave,
            plot=False,
            **init_kwargs,
        )

    _, Ag, PCs, Pyrs, Objs = extract_objects_from_Pyrs(Pyrs)

    if two_compartment:
        Pyrs.set_BTSP_learn(soma=True, dend=False)
        Pyrs_for_weights = Pyrs.SomaCompartment
        Pyrs.set_learn(soma=use_Hebbian, dend=False, inhibit=False)
    else:
        Pyrs.set_BTSP_learn()
        Pyrs_for_weights = Pyrs
        Pyrs.set_learn(use_Hebbian)

    # run learning
    restarted = False
    Pyr_weights = [Pyrs_for_weights.inputs[PCs.name]["w"].copy()]  # type: ignore[attr-defined]
    break_in_n = -1
    for i in tqdm(range(max_num_steps)):
        Ag.update(speed_fact=3, drift_to_random_strength_ratio=1)

        if Objs is not None:
            Objs.update()

        PCs.update()

        # check whether a restart BTSP signal should go out
        if not two_compartment:
            BTSP_targets = []
            if restarted and Pyrs.n > 1:  # type: ignore[attr-defined]
                BTSP_targets = [Pyrs.n - 1]  # type: ignore[attr-defined]

            # check whether a target BTSP signal should go out
            if (
                Ag.reached_target
                and len(Ag.target_df) == BTSP_after_num_target_reaches + 1
            ):
                BTSP_targets = [0]

        # check for restart
        restarted = Ag.reached_end

        # run update
        if two_compartment:
            Pyrs.update()
        else:
            Pyrs.update(BTSP_targets=BTSP_targets)
        if not i % weight_recording_freq:
            Pyr_weights.append(Pyrs_for_weights.inputs[PCs.name]["w"].copy())  # type: ignore[attr-defined]

        if break_in_n < 0:
            if len(Ag.target_df) > num_rewards:
                break_in_n = 20
        else:
            if break_in_n == 0:
                break
            break_in_n -= 1

    if len(Ag.target_df) <= num_rewards:
        print(
            f"Only reached the reward {len(Ag.target_df) - 1} "
            f"times (target: {num_rewards})."
        )

    Ag.log_trajectory_stats_to_date()
    Ag.log_trajectory_stats_to_date(log_as_time=False)

    if two_compartment:
        plot_T_maze(Ag, PCs, Objs, autosave=autosave, method="history")  # type: ignore[arg-type]
    else:
        plot_T_maze(Ag, PCs, Pyrs, autosave=autosave, method="groundtruth")  # type: ignore[arg-type]

    Pyrs.plot_rate_maps_across_learning()  # type: ignore[attr-defined]

    plot_fcts.plot_time_series_with_BTSP_events(Pyrs)  # type: ignore[arg-type]

    return Pyrs


### 1D (LINEAR TRACK) FUNCTIONS ###


def plot_1D_spatial_info(
    Ag: agent.ResetableAgent,
    PCs: riab_neurons.PlaceCells,
    Pyrs: learning_neurons.BTSPLayer,
    Pyr_weights: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] | None = None,
    Pyrs_norm_by: str | float | None = None,
    autosave: bool | None = None,
) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
    """
    plot_1D_spatial_info(Ag, PCs, Pyrs)

    Plot spatial info for a 1D environment experiment:
        (1) Environment,
        (2) Place cell locations,
        (3) Pyr. overlayed rate map,
        (4, optional) Pyr. input weights (if provided),
        (5-7) Pyr. rate map across learning
        (8) Environment (again).

    Args:
    - Ag (agent.ResetableAgent): Agent.
    - PCs (riab_neurons.PlaceCells): Place cells.
    - Pyrs (learning_neurons.BTSPLayer): Pyr. neurons.
    - Pyr_weights (list): List of Pyr. weights with shape (num_epochs, num_cells, num_PCs).
        Default is None.
    - Pyrs_norm_by (str, optional): Normalization method for rate maps. If None,
        default is used. Default is None.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - Axes (2D np.ndarray): Array of subplots with 1D environment experiment info
        plotted, with shape (7 or 8, 1). See description for details.
    """

    # 7 or 8 plots
    height_ratios = [1, 1.2, 1.5, 1, 1, 1, 1]
    if Pyr_weights is not None:
        height_ratios.insert(3, 2)  # add height ratio for weights
    gridspec_kw = {"height_ratios": height_ratios}
    figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
    fig, axes = plt.subplots(
        nrows=len(height_ratios),
        figsize=figsize,
        sharex=True,
        gridspec_kw=gridspec_kw,
        squeeze=False,
    )
    ax1D = np.asarray(axes).ravel()

    # Plot environment
    plot_fcts.plot_1D_reset_environment(Ag, sub_ax=ax1D[0], autosave=False)

    # Plot place cell locations
    PCs.plot_place_cell_locations(sub_ax=ax1D[1], autosave=False, plot_objects=False)
    plot_fcts.plot_overlayed_rate_maps(
        PCs, sub_ax=ax1D[1], method="max", autosave=False
    )
    ymin, ymax = ax1D[1].get_ylim()
    ymin = min(ymin, 0)
    ax1D[1].set_ylim((ymin - 0.05 * (ymax - ymin)), ymax)
    ax1D[1].set_title("Place cell locations")

    # Plot place cell rate map
    PCs.plot_rate_map(chosen_neurons="all", ax=ax1D[2], autosave=False)
    ax1D[2].set_title("Place cell rate map")

    # Plot Pyr. weights
    i = 3
    if Pyr_weights is not None:
        plot_fcts.plot_1D_input_place_cell_weights(
            np.asarray(Pyr_weights),
            PCs,
            sub_ax=ax1D[i],
            autosave=False,
        )
        i += 1

    # Plot Pyr. rate maps across learning
    plot_fcts.plot_1D_rate_map_across_learning(
        Ag, Pyrs, axes=ax1D[i : i + 3], norm_by=Pyrs_norm_by, autosave=False  # type: ignore[arg-type]
    )

    # Plot environment
    plot_fcts.plot_1D_reset_environment(Ag, sub_ax=ax1D[i + 3], autosave=False)

    for a, sub_ax in enumerate(ax1D[:-1]):
        sub_ax.set_xlabel("")
        if a > 1:
            sub_ax.spines["bottom"].set_visible(False)
            sub_ax.xaxis.set_visible(False)

    plot_util.save_figure(fig, "1D_env_info", save=autosave)

    return axes


def plot_1D_time_info(
    Ag: agent.ResetableAgent,
    PCs: riab_neurons.PlaceCells,
    Pyrs: learning_neurons.BTSPLayer,
    Pyrs_norm_by: str | float | None = None,
    autosave: bool | None = None,
) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
    """
    plot_1D_time_info(Ag, PCs, Pyrs)

    Plot time info for a 1D experiment:
        (1) Trajectories,
        (2) Place cell rate timeseries,
        (3) Pyr. rate timeseries

    Args:
    - Ag (agent.ResetableAgent): Agent.
    - PCs (riab_neurons.PlaceCells): Place cells.
    - Pyrs (learning_neurons.BTSPLayer): Pyr. neurons.
    - Pyrs_norm_by (str, optional): Normalization method for rate maps. If None,
        default is used. Default is None.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - Axes (2D np.ndarray): Array of subplots with 1D time info plotted,
        with shape (3, 1). See description for details.
    """

    # 3 plots
    height_ratios = [1.5, 1, 1.1**Pyrs.n]
    gridspec_kw = {"height_ratios": height_ratios}
    figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
    fig, axes = plt.subplots(
        nrows=len(height_ratios),
        figsize=figsize,
        sharex=True,
        gridspec_kw=gridspec_kw,
        squeeze=False,
    )
    ax1D = np.asarray(axes).ravel()

    # Plot trajectories
    Ag.plot_trajectories_across_time(
        framerate=1 / Ag.dt, sub_ax=ax1D[0], autosave=False
    )
    ax1D[0].set_title("Trajectories")

    # Plot place cell rate timeseries
    PCs.plot_rate_timeseries(
        chosen_neurons="all", spikes=False, sub_ax=ax1D[1], autosave=False
    )
    ax1D[1].set_title("Place cell rate timeseries")

    # Plot Pyr. rate timeseries
    kwargs = dict()
    if Pyrs_norm_by is not None:
        kwargs["norm_by"] = Pyrs_norm_by
    Pyrs.plot_rate_timeseries(
        chosen_neurons="all",
        spikes=True,
        sub_ax=ax1D[2],
        shift=-10,
        overlap=1,
        autosave=False,
        **kwargs,
    )
    ax1D[2].set_title("Pyr. rate timeseries")

    plot_fcts.mark_target_and_reset_points(Ag, Pyrs, sub_ax=ax1D[2])

    for sub_ax in ax1D[:-1]:
        sub_ax.set_xlabel("")

    plot_util.save_figure(fig, "time_info", save=autosave)

    return axes


def learn_1D_BTSP(
    env_params: dict[str, Any] | None = None,
    agent_params: dict[str, Any] | None = None,
    PC_params: dict[str, Any] | None = None,
    Pyr_params: dict[str, Any] | None = None,
    num_rewards: int = 10,
    max_num_steps: int = 5000,
    weight_recording_freq: int = 100,
    use_Hebbian: bool = False,
    BTSP_after_num_target_reaches: int = 5,
    two_compartment: bool = False,
    autosave: bool | None = None,
) -> tuple[
    env.Environment,
    agent.ResetableAgent,
    riab_neurons.PlaceCells,
    learning_neurons.BTSPLayer,
]:
    """
    learn_1D_BTSP()

    Run a 1D learning experiment with BTSP learning. Plot spatial and time information.

    Args:
    - env_params (dict, optional): Parameters for the environment. Default is None.
    - agent_params (dict, optional): Parameters for the agent. Default is None.
    - PC_params (dict, optional): Parameters for the place cells.
        Default is None.
    - Pyr_params (dict, optional): Parameters for the Pyr. neurons. Default is None.
    - num_rewards (int, optional): Target number of rewards to reach.
        Default is 200.
    - max_num_steps (int, optional): Maximum number of steps to run.
        Default is 5000.
    - weight_recording_freq (int, optional): Frequency at which to record weights.
        Default is 100.
    - use_Hebbian (bool, optional): Whether to use Hebbian learning.
        Default is False.
    - BTSP_after_num_target_reaches (int, optional): Number of target reaches at which to
        apply BTSP event. Default is 5.
    - two_compartment (bool, optional): Whether to use two-compartment model.
        Default is False.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer
    - spatial_axes (2D np.ndarray): Array of subplots with 1D environment experiment
        info plotted, with shape (8, 1). See run_manager.plot_1D_spatial_info() for
        details.
    - time_axes (2D np.ndarray): Array of subplots with 1D time info plotted, with
        shape (3, 1). See run_manager.plot_1D_time_info() for details.
    """

    env_params = env_params or params_util.get_env_params(environment="linear")
    Env = env.Environment(params=env_params)

    agent_params = agent_params or params_util.get_agent_params(environment="linear")
    Ag = agent.ResetableAgent(Env, params=agent_params)

    PC_params = PC_params or params_util.get_PC_params(environment="linear")
    PCs = riab_neurons.PlaceCells(Ag, params=PC_params)

    if Pyr_params is None:
        Pyr_params = params_util.get_Pyr_params(
            environment="linear",
            BTSP=True,
            NMDA=two_compartment,
            two_compartment=two_compartment,
        )

    Pyr_params["input_layers"] = [PCs]
    if two_compartment:
        Pyrs = two_comp_neurons.TwoCompLayer(Ag, params=Pyr_params)
    else:
        Pyrs = learning_neurons.BTSPLayer(Ag, params=Pyr_params)
    Pyrs.set_learn(use_Hebbian)
    Pyrs.set_BTSP_learn()

    # run learning
    restarted = False
    PCs_name = PCs.name  # type: ignore[attr-defined]
    Pyrs_n = Pyrs.n  # type: ignore[attr-defined]
    Pyr_weights = [Pyrs.inputs[PCs_name]["w"].copy()]
    break_in_n = -1
    for i in tqdm(range(max_num_steps)):
        Ag.update()
        PCs.update()

        # check whether a restart BTSP signal should go out
        if not two_compartment:
            BTSP_targets = list()
            if len(Ag.target_df) == BTSP_after_num_target_reaches + 1:
                if restarted and Pyrs_n > 1:
                    BTSP_targets = [Pyrs_n - 1]

                # check whether a target BTSP signal should go out
                if Ag.reached_target:
                    BTSP_targets = [0]

        # check for restart
        restarted = Ag.reached_end

        # run update
        if two_compartment:
            Pyrs.update()
        else:
            Pyrs.update(BTSP_targets=BTSP_targets)
        if not i % weight_recording_freq:
            Pyr_weights.append(Pyrs.inputs[PCs_name]["w"].copy())

        if break_in_n < 0:
            if len(Ag.target_df) > num_rewards:
                break_in_n = 20
        else:
            if break_in_n == 0:
                break
            break_in_n -= 1

    if len(Ag.target_df) <= num_rewards:
        print(
            f"Only reached the reward {len(Ag.target_df) - 1} times "
            f"(target: {num_rewards})."
        )

    Ag.log_trajectory_stats_to_date()
    Ag.log_trajectory_stats_to_date(log_as_time=False)

    spatial_axes = plot_1D_spatial_info(Ag, PCs, Pyrs, Pyr_weights, autosave=autosave)

    time_axes = plot_1D_time_info(Ag, PCs, Pyrs, autosave=autosave)

    return Pyrs, spatial_axes, time_axes


def plot_interleaved_openfield_rate_maps(Pyrs, Objs, num_cols=10):
    """
    plot_interleaved_openfield_rate_maps(Pyrs, Objs)

    Plot interleaved open field rate maps for Pyr. neuron somata and the object neurons
    thattarget their dendrites.

    Rate maps are computed theoretically based on place cell inputs.

    Args:
    - Pyrs (learning_neurons.BTSPLayer): Pyr. neurons.
    - Objs (object_neurons.ObjectCells): Object neurons.
    """

    if Pyrs.n != Objs.n:
        raise ValueError("Pyrs and Objs should have the same number of neurons.")

    num_cols = min(10, Objs.n)
    num_rows = int(np.ceil(Objs.n / num_cols)) * 2

    _, axes = plt.subplots(
        num_rows, num_cols, figsize=(1.5 * num_cols, 1.5 * num_rows), squeeze=False
    )

    Objs.plot_rate_map(ax=axes[::2].ravel()[: Objs.n], no_legend=True)
    plot_fcts.plot_2D_input_place_cell_weights(
        Pyrs.SomaCompartment,
        ax=axes[1::2].ravel()[: Objs.n],
        alpha=0.5,
        plot_BTSP_events=True,
        no_legend=True,
    )

    return axes


if __name__ == "__main__":
    Pyrs, spatial_axes, time_axes = learn_1D_BTSP()

    breakpoint()
