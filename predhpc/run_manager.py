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


def extract_objects_from_CA1s(CA1s):
    """
    extract_objects_from_CA1s(CA1s)

    Extract objects from a CA1s object.

    Args:
    - CA1s (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        CA1 neuron layer

    Returns:
    - Env (env.Environment): Environment
    - Ag (agent.Agent): Agent
    - CA3_PCs (riab_neurons.PlaceCells): CA3 place cells
    - CA1s (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        CA1 neuron layer
    - ECs (object_neurons.ObjectCells or object_neurons.ObjectInstanceCells):
        Object cells
    """

    Env = CA1s.Agent.Environment

    Ag = CA1s.Agent

    if isinstance(CA1s, two_comp_neurons.TwoCompLayer):
        EC_key = list(CA1s.DendriteCompartment.inputs.keys())[-1]
        ECs = CA1s.DendriteCompartment.inputs[EC_key]["layer"]

        CA3_key = list(CA1s.SomaCompartment.inputs.keys())[0]
        CA3_PCs = CA1s.SomaCompartment.inputs[CA3_key]["layer"]
    else:
        ECs = None
        CA3_key = list(CA1s.inputs.keys())[0]
        CA3_PCs = CA1s.inputs[CA3_key]["layer"]

    return Env, Ag, CA3_PCs, CA1s, ECs


def plot_2D_initial_conditions(CA1s, num_samples=10, autosave: bool | None = None):
    """
    plot_2D_initial_conditions(CA1s)

    Plot initial conditions for a 2D environment experiment.

    Args:
    - CA1s (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        CA1 neuron layer
    - num_samples (int, optional): Number of samples to plot. Default is 10.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - fields_axes (2D np.ndarray): Array of subplots with place fields plotted, with
        shape (num_layers, num_samples).
    - aggreg_ax1D (1D np.ndarray): Array of subplots with environment and aggregated
        fields plotted, with shape (3,).
    """

    Env, _, CA3_PCs, CA1s, ECs = extract_objects_from_CA1s(CA1s)

    # Plot fields
    if ECs is None:
        num_cols = min(CA3_PCs.n, num_samples)
        neurons = [CA3_PCs]
    else:
        num_cols = min(max(CA3_PCs.n, ECs.n), num_samples)
        neurons = [ECs, CA3_PCs]

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
        CA3_PCs, method="max", colorbar=False, sub_ax=aggreg_ax1D[1], autosave=False
    )
    aggreg_ax1D[1].set_title("CA3 overlayed place fields")

    CA3_PCs.plot_place_cell_locations(sub_ax=aggreg_ax1D[2], autosave=False)
    aggreg_ax1D[2].set_title("CA3 place cell centers")

    if autosave:
        plot_util.save_figure(fields_fig, "openfield_init_fields", save=autosave)
        plot_util.save_figure(aggreg_fields, "openfield_init_aggreg", save=autosave)

    return fields_axes, aggreg_ax1D


def init_2D_env_objects(
    env_params: dict[str, Any] | None = None,
    agent_params: dict[str, Any] | None = None,
    CA3_PC_params: dict[str, Any] | None = None,
    CA1_params: dict[str, Any] | None = None,
    EC_params: dict[str, Any] | None = None,
    environment="openfield",
    two_compartment: bool = True,
    autosave: bool | None = None,
    plot: bool = True,
):
    """
    init_2D_env_objects()

    Initialize objects for a 2D environment experiment, and obtain CA1s.

    Args:
    - env_params (dict, optional): Parameters for the environment. Default is None.
    - agent_params (dict, optional): Parameters for the agent. Default is None.
    - CA3_PC_params (dict, optional): Parameters for the CA3 place cells. Default is
        None.
    - CA1_params (dict, optional): Parameters for the CA1 neurons. Default is None.
    - EC_params (dict, optional): Parameters for the EC neurons. Default is None.
    - two_compartment (bool, optional): Whether to use two-compartment model.
        Default is True.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.
    - plot (bool, optional): Whether to plot the environment and neurons. Default is
        True.

    Returns:
    - CA1s (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        CA1 neuron layer
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

    CA3_PC_params = CA3_PC_params or params_util.get_CA3_PC_params(
        environment=environment
    )
    CA3_PCs = riab_neurons.PlaceCells(Ag, params=CA3_PC_params)

    if two_compartment:
        EC_params = EC_params or params_util.get_EC_params(environment=environment)
        if environment == "tmaze":
            ECs = object_neurons.ObjectCells(Ag, params=EC_params)
        else:
            ECs = object_neurons.ObjectInstanceCells(Ag, params=EC_params)
    else:
        if EC_params is not None:
            warnings.warn("EC_params will be ignored if two_compartment is False.")
        ECs = None

    if CA1_params is None:
        n_kwargs = {"n": ECs.n} if two_compartment else dict()
        CA1_params = params_util.get_CA1_params(
            environment=environment,
            BTSP=True,
            NMDA=two_compartment,
            two_compartment=two_compartment,
            **n_kwargs,
        )

    if two_compartment:
        CA1_params["soma_input_layers"] = [CA3_PCs]  # type: ignore[assignment]
        if CA1_params["n"] is None:
            CA1_params["n"] = ECs.n
        elif CA1_params["n"] != ECs.n:
            raise ValueError(
                f"If provided, CA1_params['n'] ({CA1_params['n']}) must be equal to "
                f"ECs.n ({ECs.n})."
            )
    else:
        CA1_params["input_layers"] = [CA3_PCs]  # type: ignore[assignment]

    if two_compartment:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No input layers")
            CA1s = two_comp_neurons.TwoCompLayer(Ag, params=CA1_params)

        EC_to_CA1_w = gen_util.get_weights(ECs.n, CA1s.n)
        CA1s.DendriteCompartment.add_input(ECs, w=EC_to_CA1_w)
        CA1s.set_BTSP_learn(soma=True, dend=False)
    else:
        CA1s = learning_neurons.BTSPLayer(Ag, params=CA1_params)
        CA1s.set_BTSP_learn()

    if plot:
        plot_2D_initial_conditions(CA1s, autosave=autosave)

    return CA1s


### 2D (OPENFIELD) FUNCTIONS ###


def learn_openfield_BTSP(
    CA1s: learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer | None = None,
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
    - CA1s (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        CA1 neuron layer
    - weights_at_BTSP (dict): Dictionary with keys "weights" and "steps" in which
        input weights from CA3 and steps are recorded. None if record_weights_at_BTSP
        is False.
    """

    if CA1s is None:
        CA1s = init_2D_env_objects(
            two_compartment=two_compartment,
            environment="openfield",
            autosave=autosave,
            plot=False,
            **init_kwargs,
        )

    _, Ag, CA3_PCs, CA1s, ECs = extract_objects_from_CA1s(CA1s)

    if two_compartment:
        CA1s.set_BTSP_learn(soma=True, dend=False)
        CA1s_for_weights = CA1s.SomaCompartment
        CA1s.set_learn(soma=use_Hebbian, dend=False, inhibit=False)
    else:
        CA1s.set_BTSP_learn()
        CA1s_for_weights = CA1s
        CA1s.set_learn(use_Hebbian)

    # run learning
    start_step = len(CA1s_for_weights.history["t"])
    stop_BTSP = max(0, start_step + max_num_steps - num_end_without_BTSP)

    BTSP_stopped = False
    start_num_BTSP = len(CA1s_for_weights.history["BTSP_events"])

    BTSP_steps, weights, steps = list(), list(), list()
    for i in tqdm(range(max_num_steps)):
        Ag.update(speed_fact=3, drift_to_random_strength_ratio=1)
        ECs.update()
        CA3_PCs.update()
        CA1s.update()
        if record_weights_at_BTSP:
            num_BTSP = len(CA1s_for_weights.history["BTSP_events"])
            if num_BTSP > len(BTSP_steps) + start_num_BTSP:
                BTSP_steps.append(len(CA1s_for_weights.history["t"]))

        if not BTSP_stopped and i + start_step >= stop_BTSP:
            if two_compartment:
                CA1s.set_BTSP_learn(soma=False, dend=False)
            else:
                CA1s.set_BTSP_learn()
            print(f"BTSP blocked from step {i + start_step}.")
            BTSP_stopped = True

        if CA1s_for_weights.BTSP_applied.any():
            weights.append(copy.deepcopy(CA1s_for_weights.inputs["CA3_PCs"]["w"]))
            steps.append(len(CA1s_for_weights.history["t"]))

    BTSP_steps = np.asarray(BTSP_steps)
    print(
        f"{len(BTSP_steps)} BTSP events recorded (allowed from steps "
        f"{start_step} to {stop_BTSP}): occurred between steps "
        f"{BTSP_steps.min()} and {BTSP_steps.max()}."
    )

    weights_at_BTSP = None
    if record_weights_at_BTSP:
        weights_at_BTSP = {"weights": np.asarray(weights), "steps": steps}

    return CA1s, weights_at_BTSP


### 2D (T-MAZE) FUNCTIONS ###


def plot_T_maze(
    Ag: agent.TAgent,
    CA3_PCs: riab_neurons.PlaceCells,
    CA1s_or_ECs: learning_neurons.BTSPLayer | object_neurons.ObjectCells,
    method: str = "groundtruth",
    autosave: bool | None = None,
):
    """
    plot_T_maze(Ag, CA3_PCs, CA1s_or_ECs)

    Plot the T-maze environment:
        (1) Agent trajectories,
        (2) CA3 place cell locations, and
        (3) CA1 or EC overlayed rate maps.

    Args:
    - Ag (agent.Agent): Agent.
    - CA3_PCs (riab_neurons.PlaceCells): CA3 place cells.
    - CA1s_or_ECs (learning_neurons.BTSPLayer or object_neurons.ObjectCells):
        CA1s layer.
    - method (str, optional): Method to use for plotting the CA1 rate map. Default
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

    # Plot CA3 place cell locations on T-maze
    plot_fcts.plot_overlayed_rate_maps(
        CA3_PCs, sub_ax=ax1D[1], method="max", colorbar=False
    )
    CA3_PCs.plot_place_cell_locations(sub_ax=ax1D[1])
    ax1D[1].scatter(
        *Ag.target_position,
        marker=".",
        color="blue",
        s=18,
        zorder=5,
    )
    ax1D[1].set_title("CA3 rate maps")

    # Plot CA1 rate map on T-maze
    if isinstance(CA1s_or_ECs, learning_neurons.BTSPLayer):
        CA1s_or_ECs.plot_rate_map(ax=ax1D[2], method=method)
        title = f"{CA1s_or_ECs.name.replace('_', ' ')} rate map"  # type: ignore[attr-defined]
    else:
        plot_fcts.plot_overlayed_rate_maps(
            CA1s_or_ECs,
            sub_ax=ax1D[2],
            method="max",
            colorbar=False,
            replot_env=True,
        )
        title = "EC rate map"
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
    CA1s: learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer | None = None,
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
    - CA3_PC_params (dict, optional): Parameters for the CA3 place cells. Default is
        None.
    - CA1_params (dict, optional): Parameters for the CA1 neurons. Default is None.
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
    - CA1s (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        CA1 neuron layer
    """

    if CA1s is None:
        CA1s = init_2D_env_objects(
            two_compartment=two_compartment,
            environment="tmaze",
            autosave=autosave,
            plot=False,
            **init_kwargs,
        )

    _, Ag, CA3_PCs, CA1s, ECs = extract_objects_from_CA1s(CA1s)

    if two_compartment:
        CA1s.set_BTSP_learn(soma=True, dend=False)
        CA1s_for_weights = CA1s.SomaCompartment
        CA1s.set_learn(soma=use_Hebbian, dend=False, inhibit=False)
    else:
        CA1s.set_BTSP_learn()
        CA1s_for_weights = CA1s
        CA1s.set_learn(use_Hebbian)

    # run learning
    restarted = False
    CA1_weights = [CA1s_for_weights.inputs[CA3_PCs.name]["w"].copy()]  # type: ignore[attr-defined]
    break_in_n = -1
    for i in tqdm(range(max_num_steps)):
        Ag.update(speed_fact=3, drift_to_random_strength_ratio=1)

        if ECs is not None:
            ECs.update()

        CA3_PCs.update()

        # check whether a restart BTSP signal should go out
        if not two_compartment:
            BTSP_targets = []
            if restarted and CA1s.n > 1:  # type: ignore[attr-defined]
                BTSP_targets = [CA1s.n - 1]  # type: ignore[attr-defined]

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
            CA1s.update()
        else:
            CA1s.update(BTSP_targets=BTSP_targets)
        if not i % weight_recording_freq:
            CA1_weights.append(CA1s_for_weights.inputs[CA3_PCs.name]["w"].copy())  # type: ignore[attr-defined]

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
        plot_T_maze(Ag, CA3_PCs, ECs, autosave=autosave, method="history")  # type: ignore[arg-type]
    else:
        plot_T_maze(Ag, CA3_PCs, CA1s, autosave=autosave, method="groundtruth")  # type: ignore[arg-type]

    CA1s.plot_rate_maps_across_learning()  # type: ignore[attr-defined]

    plot_fcts.plot_time_series_with_BTSP_events(CA1s)  # type: ignore[arg-type]

    return CA1s


### 1D (LINEAR TRACK) FUNCTIONS ###


def plot_1D_spatial_info(
    Ag: agent.ResetableAgent,
    CA3_PCs: riab_neurons.PlaceCells,
    CA1s: learning_neurons.BTSPLayer,
    CA1_weights: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] | None = None,
    autosave: bool | None = None,
) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
    """
    plot_1D_spatial_info(Ag, CA3_PCs, CA1s)

    Plot spatial info for a 1D environment experiment:
        (1) Environment,
        (2) CA3 place cell locations,
        (3) CA1 overlayed rate map,
        (4, optional) CA1 input weights (if provided),
        (5-7) CA1 rate map across learning
        (8) Environment (again).

    Args:
    - Ag (agent.ResetableAgent): Agent.
    - CA3_PCs (riab_neurons.PlaceCells): CA3 place cells.
    - CA1s (learning_neurons.BTSPLayer): CA1 neurons.
    - CA1_weights (list): List of CA1 weights with shape (num_epochs, num_cells, num_PCs).
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - Axes (2D np.ndarray): Array of subplots with 1D environment experiment info
        plotted, with shape (7 or 8, 1). See description for details.
    """

    # 7 or 8 plots
    height_ratios = [1, 1.2, 1.5, 1, 1, 1, 1]
    if CA1_weights is not None:
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

    # Plot CA3 place cell locations
    CA3_PCs.plot_place_cell_locations(
        sub_ax=ax1D[1], autosave=False, plot_objects=False
    )
    plot_fcts.plot_overlayed_rate_maps(
        CA3_PCs, sub_ax=ax1D[1], method="max", autosave=False
    )
    ymin, ymax = ax1D[1].get_ylim()
    ymin = min(ymin, 0)
    ax1D[1].set_ylim((ymin - 0.05 * (ymax - ymin)), ymax)
    ax1D[1].set_title("CA3 place cell locations")

    # Plot CA3 rate map
    CA3_PCs.plot_rate_map(chosen_neurons="all", ax=ax1D[2], autosave=False)
    ax1D[2].set_title("CA3 rate map")

    # Plot CA1 weights
    i = 3
    if CA1_weights is not None:
        plot_fcts.plot_1D_input_place_cell_weights(
            np.asarray(CA1_weights),
            CA3_PCs,
            sub_ax=ax1D[i],
            autosave=False,
        )
        i += 1

    # Plot CA1 rate maps across learning
    plot_fcts.plot_1D_rate_map_across_learning(
        Ag, CA1s, axes=ax1D[i : i + 3], autosave=False  # type: ignore[arg-type]
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
    CA3_PCs: riab_neurons.PlaceCells,
    CA1s: learning_neurons.BTSPLayer,
    autosave: bool | None = None,
) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
    """
    plot_1D_time_info(Ag, CA3_PCs, CA1s)

    Plot time info for a 1D experiment:
        (1) Trajectories,
        (2) CA3 rate timeseries,
        (3) CA1 rate timeseries

    Args:
    - Ag (agent.ResetableAgent): Agent.
    - CA3_PCs (riab_neurons.PlaceCells): CA3 place cells.
    - CA1s (learning_neurons.BTSPLayer): CA1 neurons.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - Axes (2D np.ndarray): Array of subplots with 1D time info plotted,
        with shape (3, 1). See description for details.
    """

    # 3 plots
    height_ratios = [1.5, 1, 1.1**CA1s.n]
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

    # Plot CA3 rate timeseries
    CA3_PCs.plot_rate_timeseries(
        chosen_neurons="all", spikes=False, sub_ax=ax1D[1], autosave=False
    )
    ax1D[1].set_title("CA3 rate timeseries")

    # Plot CA1 rate timeseries
    CA1s.plot_rate_timeseries(
        chosen_neurons="all",
        spikes=True,
        sub_ax=ax1D[2],
        shift=-10,
        overlap=1,
        autosave=False,
    )
    ax1D[2].set_title("CA1 rate timeseries")

    plot_fcts.mark_target_and_reset_points(Ag, CA1s, sub_ax=ax1D[2])

    for sub_ax in ax1D[:-1]:
        sub_ax.set_xlabel("")

    plot_util.save_figure(fig, "time_info", save=autosave)

    return axes


def learn_1D_BTSP(
    env_params: dict[str, Any] | None = None,
    agent_params: dict[str, Any] | None = None,
    CA3_PC_params: dict[str, Any] | None = None,
    CA1_params: dict[str, Any] | None = None,
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
    - CA3_PC_params (dict, optional): Parameters for the CA3 place cells.
        Default is None.
    - CA1_params (dict, optional): Parameters for the CA1 neurons. Default is None.
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
    - CA1s (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        CA1 neuron layer
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

    CA3_PC_params = CA3_PC_params or params_util.get_CA3_PC_params(environment="linear")
    CA3_PCs = riab_neurons.PlaceCells(Ag, params=CA3_PC_params)

    if CA1_params is None:
        CA1_params = params_util.get_CA1_params(
            environment="linear",
            BTSP=True,
            NMDA=two_compartment,
            two_compartment=two_compartment,
        )

    CA1_params["input_layers"] = [CA3_PCs]
    if two_compartment:
        CA1s = two_comp_neurons.TwoCompLayer(Ag, params=CA1_params)
    else:
        CA1s = learning_neurons.BTSPLayer(Ag, params=CA1_params)
    CA1s.set_learn(use_Hebbian)
    CA1s.set_BTSP_learn()

    # run learning
    restarted = False
    CA3_PCs_name = CA3_PCs.name  # type: ignore[attr-defined]
    CA1s_n = CA1s.n  # type: ignore[attr-defined]
    CA1_weights = [CA1s.inputs[CA3_PCs_name]["w"].copy()]
    break_in_n = -1
    for i in tqdm(range(max_num_steps)):
        Ag.update()
        CA3_PCs.update()

        # check whether a restart BTSP signal should go out
        if not two_compartment:
            BTSP_targets = list()
            if len(Ag.target_df) == BTSP_after_num_target_reaches + 1:
                if restarted and CA1s_n > 1:
                    BTSP_targets = [CA1s_n - 1]

                # check whether a target BTSP signal should go out
                if Ag.reached_target:
                    BTSP_targets = [0]

        # check for restart
        restarted = Ag.reached_end

        # run update
        if two_compartment:
            CA1s.update()
        else:
            CA1s.update(BTSP_targets=BTSP_targets)
        if not i % weight_recording_freq:
            CA1_weights.append(CA1s.inputs[CA3_PCs_name]["w"].copy())

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

    spatial_axes = plot_1D_spatial_info(
        Ag, CA3_PCs, CA1s, CA1_weights, autosave=autosave
    )

    time_axes = plot_1D_time_info(Ag, CA3_PCs, CA1s, autosave=autosave)

    return CA1s, spatial_axes, time_axes


def plot_interleaved_openfield_rate_maps(CA1s, ECs, num_cols=10):
    """
    plot_interleaved_openfield_rate_maps(CA1s, ECs)

    Plot interleaved open field rate maps for CA1 neuron somata and the EC neurons that
    target their dendrites.

    Rate maps are computed theoretically based on place cell inputs.

    Args:
    - CA1s (learning_neurons.BTSPLayer): CA1 neurons.
    - ECs (object_neurons.ObjectCells): EC neurons.
    """

    if CA1s.n != ECs.n:
        raise ValueError("CA1s and ECs should have the same number of neurons.")

    num_cols = min(10, ECs.n)
    num_rows = int(np.ceil(ECs.n / num_cols)) * 2

    _, axes = plt.subplots(
        num_rows, num_cols, figsize=(1.5 * num_cols, 1.5 * num_rows), squeeze=False
    )

    ECs.plot_rate_map(ax=axes[::2].ravel()[: ECs.n], no_legend=True)
    plot_fcts.plot_2D_input_place_cell_weights(
        CA1s.SomaCompartment,
        ax=axes[1::2].ravel()[: ECs.n],
        alpha=0.5,
        plot_BTSP_events=True,
        no_legend=True,
    )

    return axes


if __name__ == "__main__":
    CA1s, spatial_axes, time_axes = learn_1D_BTSP()

    breakpoint()
