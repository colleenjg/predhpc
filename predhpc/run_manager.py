#!/usr/bin/env python3

from typing import Any

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from ratinabox import Environment, PlaceCells

from predhpc import agent, neurons, plot_util


ENV_PARAMS = {
    "dimensionality": "1D",
    "scale": 1,
}

DT = 0.02
AGENT_PARAMS = {
    "reset_tolerance_prop": 0.8,
    "target_tolerance_prop": 3,
    "dt": DT,
    "speed_mean": 1,  # sets directionality
    "speed_std": 0.5,
    "start_pos": 0 + DT,
    "reset_pos": ENV_PARAMS["scale"] - DT,
    "target_pos": ENV_PARAMS["scale"] - DT * 8,
    "fixed_direction": True,
    "target_wait": 100,
}

CA3_PC_PARAMS = {
    "name": "CA3_PCs",
    "n": 16,
    "description": "gaussian_threshold",
    "place_cell_centres": "uniform",
    "min_fr": 0,
    "max_fr": 10,
    "color": "C5",
    "widths": DT * 5,
}

CA1_PARAMS = {
    "name": "CA1_BTSP",
    "color": "C2",
    "biases": None,
    "init_weights_zero": False,
    "w_init_scale": 0.1,  # set fairly small
    "lr": 1e-4,
    "btsp_tau": DT * 8,
    "btsp_fr": 10,
}


def plot_1D_env_info(
    Ag: agent.ResetableAgent,
    CA3_PCs: PlaceCells,
    CA1s: neurons.BTSPLayer,
    CA1_weights: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
):
    """Plot environment info for a 1D experiment:
        environment, place cell locations, rate map, CA1 weights, CA1 rate map

    Args:
        Ag (agent.ResetableAgent): Agent.
        CA3_PCs (PlaceCells): CA3 place cells.
        CA1s (neurons.BTSPLayer): CA1 neurons.
        CA1_weights (list): List of CA1 weights (num_epochs x num_cells x num_PCs).
    """

    # 8 plots
    height_ratios = [1, 1.2, 1.5, 2, 1, 1, 1, 1]
    gridspec_kw = {"height_ratios": height_ratios}
    figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
    fig, ax = plt.subplots(
        nrows=len(height_ratios), figsize=figsize, sharex=True, gridspec_kw=gridspec_kw
    )

    plot_util.plot_1D_reset_environment(Ag, fig=fig, ax=ax[0])

    CA3_PCs.plot_place_cell_locations(fig=fig, ax=ax[1])
    plot_util.plot_overlayed_rate_maps(CA3_PCs, fig=fig, ax=ax[1], method="max")
    ymin, ymax = ax[1].get_ylim()
    ymin = min(ymin, 0)
    ax[1].set_ylim((ymin - 0.05 * (ymax - ymin)), ymax)
    ax[1].set_title("CA3 place cell locations")

    CA3_PCs.plot_rate_map(chosen_neurons="all", fig=fig, ax=ax[2])
    ax[2].set_title("CA3 rate map")

    plot_util.plot_1D_input_place_cell_weights(
        np.asarray(CA1_weights), CA3_PCs, fig=fig, ax=ax[3]
    )
    plot_util.plot_1D_rate_map_across_learning(Ag, CA1s, fig=fig, ax=ax[4:7])

    plot_util.plot_1D_reset_environment(Ag, fig=fig, ax=ax[7])

    for a, ax_ in enumerate(ax.ravel()[:-1]):
        ax_.set_xlabel("")
        if a > 1:
            ax_.spines["bottom"].set_visible(False)
            ax_.xaxis.set_visible(False)


def plot_time_info(
    Ag: agent.ResetableAgent, CA3_PCs: PlaceCells, CA1s: neurons.BTSPLayer
):
    """Plot time info for a 1D experiment:
        trajectories, CA1 rate timeseries, CA3 rate timeseries

    Args:
        Ag (agent.ResetableAgent): Agent.
        CA3_PCs (PlaceCells): CA3 place cells.
        CA1s (neurons.BTSPLayer): CA1 neurons.
    """

    # 3 plots
    height_ratios = [1.5, 1, 1]
    gridspec_kw = {"height_ratios": height_ratios}
    figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
    fig, ax = plt.subplots(
        nrows=len(height_ratios), figsize=figsize, sharex=True, gridspec_kw=gridspec_kw
    )

    Ag.plot_trajectory_resets(framerate=1 / Ag.dt, fig=fig, ax=ax[0])
    ax[0].set_title("Trajectories")

    CA3_PCs.plot_rate_timeseries(chosen_neurons="all", spikes=True, fig=fig, ax=ax[2])
    ax[2].set_title("CA3 rate timeseries")

    CA1s.plot_rate_timeseries(
        chosen_neurons="all", spikes=True, fig=fig, ax=ax[1], shift=-10, overlap=1
    )
    ax[1].set_title("CA1 rate timeseries")
    lo, hi = ax[1].get_ylim()
    for t in CA1s.history["btsp_events"]:
        y_hei = lo + (hi - lo) * 0.95
        ax[1].scatter(
            CA1s.history["t"][t] / 60, y_hei, marker="x", s=8, color="k", alpha=0.7
        )
    for positions, ls in [
        (Ag.reached_reset_pos, "dashed"),
        (Ag.reached_target_pos, "dotted"),
    ]:
        for t in positions:
            ax[1].axvline(
                CA1s.Agent.history["t"][t] / 60,
                alpha=0.7,
                zorder=-1,
                lw=1,
                ls=ls,
                color="k",
            )

    for ax_ in ax.ravel()[:-1]:
        ax_.set_xlabel("")


def learn_1D_btsp(
    env_params: dict[str, Any],
    agent_params: dict[str, Any],
    CA3_PC_params: dict[str, Any],
    CA1_params: dict[str, Any],
    num_rewards: int = 10,
    max_num_steps: int = 5000,
    weight_recording_freq: int = 100,
    use_Hebbian: bool = False,
    num_target_reaches: int = 5,
) -> tuple[Environment, agent.ResetableAgent, PlaceCells, neurons.BTSPLayer]:
    """Run a 1D learning experiment with BTSP learning.

    Args:
        env_params (dict): Parameters for the environment.
        agent_params (dict): Parameters for the agent.
        CA3_PC_params (dict): Parameters for the CA3 place cells.
        CA1_params (dict): Parameters for the CA1 neurons.
        num_rewards (int, optional): Target number of rewards to reach.
            Defaults to 200.
        max_num_steps (int, optional): Maximum number of steps to run.
            Defaults to 5000.
        weight_recording_freq (int, optional): Frequency at which to record weights.
            Defaults to 100.
        use_Hebbian (bool, optional): Whether to use Hebbian learning.
            Defaults to False.
        num_target_reaches (int, optional): Number of targets to use. Defaults to 5.

    Returns:
        Environment, Agent, CA3 place cells, CA1 neurons
    """

    Env = Environment(params=env_params)

    Ag = agent.ResetableAgent(Env, params=agent_params)
    CA3_PCs = PlaceCells(Ag, params=CA3_PC_params)

    CA1_params["input_layers"] = [CA3_PCs]
    CA1s = neurons.BTSPLayer(Ag, params=CA1_params)
    if not use_Hebbian:
        CA1s.set_freeze()
    CA1s.set_btsp_learn()

    # run learning
    restarted = False
    CA3_PCs_name = CA3_PCs.name  # type: ignore[reportGeneralTypeIssues]
    CA3_PCs_n = CA3_PCs.n  # type: ignore[reportGeneralTypeIssues]
    CA1_weights = [CA1s.inputs[CA3_PCs_name]["w"].copy()]
    break_in_n = -1
    for i in tqdm(range(max_num_steps)):
        Ag.update()
        CA3_PCs.update()

        # check whether a restart BTSP signal should go out
        btsp_targs = []
        if len(Ag.reached_target_pos) == num_target_reaches:
            if restarted and CA3_PCs_n > 1:
                btsp_targs = [CA3_PCs_n - 1]

            # check whether a target BTSP signal should go out
            if Ag.reached_target:
                btsp_targs = [0]

        # check for restart
        restarted = Ag.reached_end

        # run update
        CA1s.update(btsp_targs=btsp_targs)
        if not i % weight_recording_freq:
            CA1_weights.append(CA1s.inputs[CA3_PCs_name]["w"].copy())

        if break_in_n < 0:
            if len(Ag.reached_target_pos) >= num_rewards:
                break_in_n = 20
        else:
            if break_in_n == 0:
                break
            break_in_n -= 1

    if len(Ag.reached_target_pos) < num_rewards:
        print(
            f"Only reached the reward {len(Ag.reached_target_pos)} times "
            f"(target: {num_rewards})."
        )

    Ag.log_trajectory_stats_to_date()
    Ag.log_trajectory_stats_to_date(time=False)

    plot_1D_env_info(Ag, CA3_PCs, CA1s, CA1_weights)

    plot_time_info(Ag, CA3_PCs, CA1s)

    return Env, Ag, CA3_PCs, CA1s


if __name__ == "__main__":
    Env, Ag, CA3_PCs, CA1s = learn_1D_btsp(
        ENV_PARAMS, AGENT_PARAMS, CA3_PC_PARAMS, CA1_PARAMS
    )

    breakpoint()
