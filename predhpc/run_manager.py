#!/usr/bin/env python3

from typing import Any, Sequence
import copy
import warnings

from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import markers
from matplotlib import figure as mpl_figure
import numpy as np
from tqdm import tqdm  # type: ignore[import]
from ratinabox import Environment, PlaceCells, ObjectVectorCells  # type: ignore[import]

from predhpc import agent, env, plot_util, util
from predhpc.neurons import learning_neurons, two_comp_neurons


SCALE = 1.0
DT = 0.02


### T-MAZE PARAMETERS ###

ENV_PARAMS_T_MAZE = {
    "prop_env": 0.1,
}

AGENT_PARAMS_T_MAZE = {
    "dt": DT,
    "head_direction_smoothing_timescale": DT * 2,
    "thigmotaxis": 0.5,
    "speed_mean": 0.16,
    "speed_std": 0.16,
    "reset_reached_within_tolerance_prop_to_dt": 1,
    "target_reached_within_tolerance_prop_to_dt": 3,
    "left_arm_prop": 0.5,
}

EC_PARAMS_T_MAZE = {
    "name": "EC_grid",
    "n": 10,
    "pref_object_dist": DT * 2,
    "angle_spread_degrees": 30,
    "max_fr": 10,
}

CA3_PC_PARAMS_T_MAZE = {
    "name": "CA3_PCs",
    "n": 40,
    "description": "gaussian_threshold",
    "place_cell_centres": "uniform",
    "min_fr": 0,
    "max_fr": 10,
    "color": "C5",
    "widths": DT * 8,
    "wall_geometry": "line_of_sight",  # due to environment shape
}

CA1_PARAMS_T_MAZE = {
    "name": "CA1_BTSP",
    "color": "C2",
    "biases": None,
    "init_weights_zero": True,
    "w_init_scale": 0.1,  # set fairly small
    "lr": 5e-5,
    "apply_Ojas_rule": True,
    "btsp_tau": DT * 8,
    "btsp_fr": 20,
    "n": 1,
}

CA1_TWO_COMP_PARAMS = {
    "n": 1,
    "name": "CA1_TwoComp",
    "biases": None,
    "lr": 1e-4,
    "apply_Ojas_rule": True,
    "dend_init_weights_zero": False,
    "dend_w_init_loc": 0.03,
    "dend_w_init_scale": 0.01,  # fairly narrow distribution
    "soma_init_weights_zero": True,
    "soma_to_dend_weight": 0.3,
    "dend_to_soma_weight": 0.2,
    "inhibit_weight": 0.5,
    "soma_btsp_tau": DT * 15,
    "soma_btsp_fr": 80,
    "soma_color": "C2",
    "dend_color": "C3",
    "inhibit_color": "C3",
}

### 1D (LINEAR TRACK) PARAMETERS ###

ENV_PARAMS_1D = {
    "dimensionality": "1D",
    "scale": SCALE,
}

AGENT_PARAMS_1D = {
    "dt": DT,
    "head_direction_smoothing_timescale": DT * 2,
    "reset_reached_within_tolerance_prop_to_dt": 0.8,
    "target_reached_within_tolerance_prop_to_dt": 3,
    "speed_mean": 1,  # sets directionality
    "speed_std": 0.5,
    "start_position": 0 + DT,
    "reset_position": SCALE - DT,
    "target_position": SCALE - DT * 8,
    "fixed_direction": True,
    "wait_between_targets": 30,
}

CA3_PC_PARAMS_1D = {
    "name": "CA3_PCs",
    "n": 16,
    "description": "gaussian_threshold",
    "place_cell_centres": "uniform",
    "min_fr": 0,
    "max_fr": 10,
    "color": "C5",
    "widths": DT * 5,
}

CA1_PARAMS_1D = {
    "name": "CA1_BTSP",
    "color": "C2",
    "biases": None,
    "init_weights_zero": False,
    "w_init_scale": 0.1,  # set fairly small
    "lr": 1e-4,
    "btsp_tau": DT * 8,
    "btsp_fr": 10,
}


### T-MAZE FUNCTIONS ###


def plot_T_maze(
    Ag: agent.TAgent,
    CA3_PCs: PlaceCells,
    CA1s_or_ECs: learning_neurons.BTSPLayer | ObjectVectorCells,
    method: str = "groundtruth",
    autosave: bool | None = None,
):
    """Plot the T-maze environment, agent trajectory, CA3 place cell locations and CA1 rate map.

    Args:
        Ag (agent.Agent): Agent.
        CA3_PCs (neurons.PlaceCells): CA3 place cells.
        CA1s (neurons.BTSPLayer): CA1s layer.

    Returns:
        fig, axes: Figure and axes.
    """

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3))
    axes_flat = np.asarray(axes).reshape(-1)

    # Plot trajectories on T-maze
    Ag.plot_trajectory(
        scale_cmap_per=False, ms_2D=5, alpha=0.3, fig=fig, ax=axes_flat[0]
    )
    axes_flat[0].set_title("Trajectories")

    # Plot CA3 place cell locations on T-maze
    plot_util.plot_overlayed_rate_maps(
        CA3_PCs, fig=fig, ax=axes_flat[1], method="max", colorbar=False
    )
    CA3_PCs.plot_place_cell_locations(fig=fig, ax=axes_flat[1])
    axes_flat[1].scatter(
        *Ag.target_position,
        marker="d",
        color="gold",
        s=20,
        zorder=5,
        edgecolors="darkgoldenrod",
        linewidth=0.5,
    )
    axes_flat[1].set_title("CA3 rate maps")

    # Plot CA1 rate map on T-maze
    if isinstance(CA1s_or_ECs, learning_neurons.BTSPLayer):
        CA1s_or_ECs.plot_rate_map(fig=fig, ax=axes_flat[2], method=method)
        title = f"{CA1s_or_ECs.name.replace('_', ' ')} rate map"  # type: ignore[attr-defined]
    else:
        plot_util.plot_overlayed_rate_maps(
            CA1s_or_ECs,
            fig=fig,
            ax=axes_flat[2],
            method="max",
            colorbar=False,
            replot_env=True,
        )
        title = "EC rate map"
    axes_flat[2].scatter(
        *Ag.target_position,
        marker="d",
        color="gold",
        s=20,
        zorder=5,
        edgecolors="darkgoldenrod",
        linewidth=0.5,
    )
    axes_flat[2].set_title(title)

    util.save_figure(fig, "T_maze", save=autosave)

    return fig, axes


def plot_time_series_with_btsp_events(
    CA1s: learning_neurons.BTSPLayer,
    fig: mpl_figure.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[mpl_figure.Figure, plt.Axes]:
    """Plot the time series of the CA1s layer with BTSP events marked.

    Args:
        CA1s (neurons.BTSPLayer): CA1s layer.
        fig (mpl_figure.Figure, optional): Figure to plot on. Defaults to None.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.

    Returns:
        fig, ax: Figure and axes.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5, 1))

    CA1s.plot_rate_timeseries(chosen_neurons="all", spikes=True, fig=fig, ax=ax)
    lo, hi = ax.get_ylim()
    for t in CA1s.history["btsp_events"]:
        y_hei = lo + (hi - lo) * 0.95
        ax.scatter(
            CA1s.history["t"][t] / 60,
            y_hei,
            marker=markers.MarkerStyle("x"),
            s=8,
            color="k",
            alpha=0.7,
        )

    target_reached_step = CA1s.Agent.target_df["reached_step"].to_numpy()  # type: ignore[attr-defined]
    if np.isnan(target_reached_step[-1]):
        target_reached_step = target_reached_step[:-1]
    target_reached_step = target_reached_step.astype(int)

    for t in target_reached_step:
        y_hei = lo + (hi - lo) * 0.82
        ax.scatter(
            CA1s.Agent.history["t"][t] / 60,
            y_hei,
            marker=markers.MarkerStyle("o"),
            s=6,
            color="k",
            alpha=0.7,
        )

    # add distance from target below
    all_positions = np.asarray(CA1s.Agent.history["pos"])
    time_in_min = np.asarray(CA1s.Agent.history["t"]) / 60
    distances = np.linalg.norm(
        CA1s.Agent.target_position - all_positions, ord=2, axis=1  # type: ignore[attr-defined]
    )
    norm_dist = distances / distances.max()
    ax.plot(time_in_min, -norm_dist, color="black", alpha=0.6, lw=1)
    ax.set_ylim(-norm_dist.max() * 1.2, ax.get_ylim()[1])

    ax.set_title("CA1 time series with BTSP events (with proximity to target)", y=1.1)

    return fig, ax


def learn_T_maze_btsp(
    env_params: dict[str, Any] = ENV_PARAMS_T_MAZE,
    agent_params: dict[str, Any] = AGENT_PARAMS_T_MAZE,
    CA3_PC_params: dict[str, Any] = CA3_PC_PARAMS_T_MAZE,
    CA1_params: dict[str, Any] | None = None,
    EC_params: dict[str, Any] | None = None,
    num_rewards: int = 200,
    max_num_steps: int = 10000,
    weight_recording_freq: int = 100,
    use_Hebbian: bool = False,
    btsp_after_num_target_reaches: int = 2,
    two_compartment: bool = True,
    autosave: bool | None = None,
) -> tuple[
    Environment,
    agent.ResetableAgent,
    ObjectVectorCells | None,
    PlaceCells,
    learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer,
]:
    """Run a T-maze learning experiment with BTSP learning.

    Args:
        env_params (dict): Parameters for the environment. Defaults to ENV_PARAMS_T_MAZE.
        agent_params (dict): Parameters for the agent. Defaults to AGENT_PARAMS_T_MAZE.
        CA3_PC_params (dict): Parameters for the CA3 place cells. Defaults to
            CA3_PC_PARAMS_T_MAZE.
        CA1_params (dict): Parameters for the CA1 neurons. Defaults to CA1_PARAMS_T_MAZE.
        num_rwd (int, optional): Target number of rewards to reach. Defaults to 200.
        max_steps (int, optional): Maximum number of steps to run. Defaults to 10000.
        weight_recording_freq (int, optional): Frequency at which to record weights.
            Defaults to 100.
        use_Hebbian (bool, optional): Whether to use Hebbian learning. Defaults to False.
        btsp_after_num_target_reaches (int, optional): Number of times to reach target before
            enabling BTSP learning. Defaults to 2.
        autosave (bool, optional): Whether to autosave. Defaults to None.

    Returns:
        Environment, Agent, EC cells, CA3 place cells, CA1 neurons
    """

    Env = env.TEnv(params=env_params)

    Ag = agent.TAgent(Env, params=agent_params)

    CA3_PCs = PlaceCells(Ag, params=CA3_PC_params)

    if CA1_params is None:
        if two_compartment:
            CA1_params = CA1_TWO_COMP_PARAMS
        else:
            CA1_params = CA1_PARAMS_T_MAZE

    CA1_params = copy.copy(CA1_params)

    if two_compartment:
        if EC_params is None:
            EC_params = EC_PARAMS_T_MAZE
        ECs = ObjectVectorCells(Ag, params=EC_params)
        CA1_params["dend_input_layers"] = [ECs]  # type: ignore[assignment]
        CA1_params["soma_input_layers"] = [CA3_PCs]  # type: ignore[assignment]
    else:
        if EC_params is not None:
            warnings.warn("EC_params will be ignored if two_compartment is False.")
        ECs = None
        CA1_params["input_layers"] = [CA3_PCs]  # type: ignore[assignment]

    if two_compartment:
        CA1s = two_comp_neurons.TwoCompLayer(Ag, params=CA1_params)
        CA1s.set_btsp_learn(soma=True, dend=False)
        CA1s.set_btsp_freeze(soma=False, dend=True)
        CA1s.set_freeze(inhibit=True)
        CA1s_for_weights = CA1s.SomaCompartment
    else:
        CA1s = learning_neurons.BTSPLayer(Ag, params=CA1_params)
        CA1s.set_btsp_learn()
        CA1s_for_weights = CA1s

    if not use_Hebbian:
        CA1s.set_freeze()

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
        btsp_targets = []
        if restarted and CA1s.n > 1:  # type: ignore[attr-defined]
            btsp_targets = [CA1s.n - 1]  # type: ignore[attr-defined]

        # check whether a target BTSP signal should go out
        if Ag.reached_target and len(Ag.target_df) == btsp_after_num_target_reaches + 1:
            btsp_targets = [0]

        # check for restart
        restarted = Ag.reached_end

        # run update
        CA1s.update(btsp_targets=btsp_targets)
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

    if two_compartment:
        fig, _ = CA1s.DendriteCompartment.plot_rate_maps_across_learning()  # type: ignore[attr-defined]
        fig.suptitle("Rate maps across learning (dendrites)", y=0.90)
        fig, _ = CA1s.SomaCompartment.plot_rate_maps_across_learning()  # type: ignore[attr-defined]
        fig.suptitle("Rate maps across learning (somata)", y=0.90)
    else:
        CA1s.plot_rate_maps_across_learning()  # type: ignore[attr-defined]

    if two_compartment:
        _, ax = plot_time_series_with_btsp_events(CA1s.DendriteCompartment)  # type: ignore[attr-defined]
        ax.set_title("Time series with BTSP events and proximity to target (dendrites)")
        _, ax = plot_time_series_with_btsp_events(CA1s.SomaCompartment)  # type: ignore[attr-defined]
        ax.set_title("Time series with BTSP events and proximity to target (somata)")
    else:
        plot_time_series_with_btsp_events(CA1s)  # type: ignore[arg-type]

    return Env, Ag, ECs, CA3_PCs, CA1s


### 1D (LINEAR TRACK) FUNCTIONS ###


def plot_1D_env_info(
    Ag: agent.ResetableAgent,
    CA3_PCs: PlaceCells,
    CA1s: learning_neurons.BTSPLayer,
    CA1_weights: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    autosave: bool | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]]:
    """Plot environment info for a 1D experiment:
        environment, place cell locations, rate map, CA1 weights, CA1 rate map

    Args:
        Ag (agent.ResetableAgent): Agent.
        CA3_PCs (PlaceCells): CA3 place cells.
        CA1s (neurons.BTSPLayer): CA1 neurons.
        CA1_weights (list): List of CA1 weights (num_epochs x num_cells x num_PCs).
        autosave (bool, optional): Whether to autosave the figure. Defaults to None.

    Returns:
        fig, axes: Figure and axes.
    """

    # 8 plots
    height_ratios = [1, 1.2, 1.5, 2, 1, 1, 1, 1]
    gridspec_kw = {"height_ratios": height_ratios}
    figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
    fig, axes = plt.subplots(
        nrows=len(height_ratios), figsize=figsize, sharex=True, gridspec_kw=gridspec_kw
    )
    axes_flat = np.asarray(axes).reshape(-1)

    # Plot environment
    plot_util.plot_1D_reset_environment(Ag, fig=fig, ax=axes_flat[0], autosave=False)

    # Plot CA3 place cell locations
    CA3_PCs.plot_place_cell_locations(fig=fig, ax=axes_flat[1], autosave=False)
    plot_util.plot_overlayed_rate_maps(
        CA3_PCs, fig=fig, ax=axes_flat[1], method="max", autosave=False
    )
    ymin, ymax = axes_flat[1].get_ylim()
    ymin = min(ymin, 0)
    axes_flat[1].set_ylim((ymin - 0.05 * (ymax - ymin)), ymax)
    axes_flat[1].set_title("CA3 place cell locations")

    # Plot CA3 rate map
    CA3_PCs.plot_rate_map(
        chosen_neurons="all", fig=fig, ax=axes_flat[2], autosave=False
    )
    axes_flat[2].set_title("CA3 rate map")

    # Plot CA1 weights
    plot_util.plot_1D_input_place_cell_weights(
        np.asarray(CA1_weights), CA3_PCs, fig=fig, ax=axes_flat[3], autosave=False
    )

    # Plot CA1 rate maps across learning
    plot_util.plot_1D_rate_map_across_learning(
        Ag, CA1s, fig=fig, axes=axes_flat[4:7], autosave=False  # type: ignore[arg-type]
    )

    # Plot environment
    plot_util.plot_1D_reset_environment(Ag, fig=fig, ax=axes_flat[7], autosave=False)

    for a, ax_ in enumerate(axes_flat[:-1]):
        ax_.set_xlabel("")
        if a > 1:
            ax_.spines["bottom"].set_visible(False)
            ax_.xaxis.set_visible(False)

    util.save_figure(fig, "1D_env_info", save=autosave)

    return fig, axes


def plot_1D_time_info(
    Ag: agent.ResetableAgent,
    CA3_PCs: PlaceCells,
    CA1s: learning_neurons.BTSPLayer,
    autosave: bool | None = None,
) -> tuple[mpl_figure.Figure, np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]]:
    """Plot time info for a 1D experiment:
        trajectories, CA1 rate timeseries, CA3 rate timeseries

    Args:
        Ag (agent.ResetableAgent): Agent.
        CA3_PCs (PlaceCells): CA3 place cells.
        CA1s (neurons.BTSPLayer): CA1 neurons.
        autosave (bool, optional): Whether to autosave the figure. Defaults to None.

    Returns:
        fig, axes: Figure and axes.
    """

    # 3 plots
    height_ratios = [1.5, 1, 1]
    gridspec_kw = {"height_ratios": height_ratios}
    figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
    fig, axes = plt.subplots(
        nrows=len(height_ratios), figsize=figsize, sharex=True, gridspec_kw=gridspec_kw
    )
    axes_flat = np.asarray(axes).reshape(-1)

    # Plot trajectories
    Ag.plot_trajectory_resets(
        framerate=1 / Ag.dt, fig=fig, ax=axes_flat[0], autosave=False
    )
    axes_flat[0].set_title("Trajectories")

    # Plot CA3 rate timeseries
    CA3_PCs.plot_rate_timeseries(
        chosen_neurons="all", spikes=True, fig=fig, ax=axes_flat[1], autosave=False
    )
    axes_flat[1].set_title("CA3 rate timeseries")

    # Plot CA1 rate timeseries
    CA1s.plot_rate_timeseries(
        chosen_neurons="all",
        spikes=True,
        fig=fig,
        ax=axes_flat[2],
        shift=-10,
        overlap=1,
        autosave=False,
    )
    axes_flat[2].set_title("CA1 rate timeseries")
    lo, hi = axes_flat[2].get_ylim()
    for t in CA1s.history["btsp_events"]:
        y_hei = lo + (hi - lo) * 0.95
        axes_flat[2].scatter(
            CA1s.history["t"][t] / 60,
            y_hei,
            marker=markers.MarkerStyle("x"),
            s=8,
            color="k",
            alpha=0.7,
        )
    for end_point, ls in [
        ("reset", "dashed"),
        ("target", "dotted"),
    ]:
        if end_point == "reset":
            positions = Ag.trajectory_df["stop_step"].to_numpy()
        elif end_point == "target":
            positions = Ag.target_df["reached_step"].to_numpy()
        else:
            raise ValueError(f"Unknown end point: {end_point}")
        if np.isnan(positions[-1]):
            positions = positions[:-1]
        positions = positions.astype(int)

        for t in positions:
            axes_flat[2].axvline(
                CA1s.Agent.history["t"][t] / 60,
                alpha=0.7,
                zorder=-1,
                lw=1,
                ls=ls,
                color="k",
            )

    for ax in axes_flat[:-1]:
        ax.set_xlabel("")

    util.save_figure(fig, "time_info", save=autosave)

    return fig, axes


def learn_1D_btsp(
    env_params: dict[str, Any] = ENV_PARAMS_1D,
    agent_params: dict[str, Any] = AGENT_PARAMS_1D,
    CA3_PC_params: dict[str, Any] = CA3_PC_PARAMS_1D,
    CA1_params: dict[str, Any] = CA1_PARAMS_1D,
    num_rewards: int = 10,
    max_num_steps: int = 5000,
    weight_recording_freq: int = 100,
    use_Hebbian: bool = False,
    btsp_after_num_target_reaches: int = 5,
    autosave: bool | None = None,
) -> tuple[Environment, agent.ResetableAgent, PlaceCells, learning_neurons.BTSPLayer]:
    """Run a 1D learning experiment with BTSP learning.

    Args:
        env_params (dict): Parameters for the environment. Defaults to ENV_PARAMS_1D.
        agent_params (dict): Parameters for the agent. Defaults to AGENT_PARAMS_1D.
        CA3_PC_params (dict): Parameters for the CA3 place cells. Defaults to
            CA3_PC_PARAMS_1D.
        CA1_params (dict): Parameters for the CA1 neurons. Defaults to CA1_PARAMS_1D.
        num_rewards (int, optional): Target number of rewards to reach.
            Defaults to 200.
        max_num_steps (int, optional): Maximum number of steps to run.
            Defaults to 5000.
        weight_recording_freq (int, optional): Frequency at which to record weights.
            Defaults to 100.
        use_Hebbian (bool, optional): Whether to use Hebbian learning.
            Defaults to False.
        btsp_after_num_target_reaches (int, optional): Number of target reaches at which to
            apply BTSP event. Defaults to 5.
        autosave (bool, optional): Whether to autosave the figure. Defaults to None.

    Returns:
        Environment, Agent, CA3 place cells, CA1 neurons
    """

    Env = Environment(params=env_params)

    Ag = agent.ResetableAgent(Env, params=agent_params)
    CA3_PCs = PlaceCells(Ag, params=CA3_PC_params)

    CA1_params = copy.copy(CA1_params)
    CA1_params["input_layers"] = [CA3_PCs]
    CA1s = learning_neurons.BTSPLayer(Ag, params=CA1_params)
    if not use_Hebbian:
        CA1s.set_freeze()
    CA1s.set_btsp_learn()

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
        btsp_targets = list()
        if len(Ag.target_df) == btsp_after_num_target_reaches + 1:
            if restarted and CA1s_n > 1:
                btsp_targets = [CA1s_n - 1]

            # check whether a target BTSP signal should go out
            if Ag.reached_target:
                btsp_targets = [0]

        # check for restart
        restarted = Ag.reached_end

        # run update
        CA1s.update(btsp_targets=btsp_targets)
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

    plot_1D_env_info(Ag, CA3_PCs, CA1s, CA1_weights, autosave=autosave)

    plot_1D_time_info(Ag, CA3_PCs, CA1s, autosave=autosave)

    return Env, Ag, CA3_PCs, CA1s


if __name__ == "__main__":
    Env, Ag, CA3_PCs, CA1s = learn_1D_btsp(
        ENV_PARAMS_1D, AGENT_PARAMS_1D, CA3_PC_PARAMS_1D, CA1_PARAMS_1D
    )

    breakpoint()
