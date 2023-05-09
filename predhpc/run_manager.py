#!/usr/bin/env python3

from matplotlib import pyplot as plt
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
    "dt": DT,
    "speed_mean": 1, # sets directionality
    "speed_std": 0.5, 
    "start_pos": 0 + DT,
    "reset_pos": ENV_PARAMS["scale"] - DT,
    "fixed_direction": True,
}

CA3_PC_PARAMS = {
    "name": "CA3_PCs",
    "n": 8,
    "description": "gaussian_threshold",
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
    "w_init_scale": 0.1, # set fairly small
    "lr": 1e-4,
    "btsp_tau": DT * 8,
    "btsp_fr": 10,
}



def plot_1D_env_info(Ag, CA3_PCs, CA1s, CA1_weights):
    """Plot environment info for a 1D experiment: 
        environment, place cell locations, rate map, CA1 weights, CA1 rate map

    Args:
        Ag (agent.ResetAgent): Agent.
        CA3_PCs (PlaceCells): CA3 place cells.
        CA1s (neurons.BTSPLayer): CA1 neurons.
        CA1_weights (list): List of CA1 weights.
    """

    # 7 plots
    hei_ratios = [1, 1, 1.5, 2, 1, 1, 1]
    gridspec_kw = {"height_ratios": hei_ratios}
    figsize = plot_util.get_figsize(sum(hei_ratios), squat=True)
    fig, ax = plt.subplots(nrows=7, figsize=figsize, sharex=True, gridspec_kw=gridspec_kw)

    plot_util.plot_1D_reset_environment(Ag, fig=fig, ax=ax[0])
    
    CA3_PCs.plot_place_cell_locations(fig, ax=ax[1])
    ax[1].set_title("CA3 place cell locations")

    CA3_PCs.plot_rate_map(chosen_neurons="all", fig=fig, ax=ax[2])
    ax[2].set_title("CA3 rate map")

    plot_util.plot_1D_input_place_cell_weights(CA1_weights, CA3_PCs, fig=fig, ax=ax[3])
    plot_util.plot_1D_rate_map_across_learning(Ag, CA1s, fig=fig, ax=ax[4:])

    for a, ax_ in enumerate(ax.ravel()[:-1]):
        ax_.set_xlabel("")
        if a > 1:
            ax_.spines["bottom"].set_visible(False)
            ax_.xaxis.set_visible(False)    


def plot_time_info(Ag, CA3_PCs, CA1s):
    """Plot time info for a 1D experiment:
        trajectories, CA1 rate timeseries, CA3 rate timeseries

    Args:
        Ag (agent.ResetAgent): Agent.
        CA3_PCs (PlaceCells): CA3 place cells.
        CA1s (neurons.BTSPLayer): CA1 neurons.
    """

    # 3 plots
    hei_ratios = [1.5, 1, 1]
    gridspec_kw = {"height_ratios": hei_ratios}
    figsize = plot_util.get_figsize(sum(hei_ratios), squat=True)
    fig, ax = plt.subplots(nrows=3, figsize=figsize, sharex=True, gridspec_kw=gridspec_kw)

    Ag.plot_trajectory_resets(framerate=1/Ag.dt, fig=fig, ax=ax[0])
    ax[0].set_title("Trajectories")

    CA3_PCs.plot_rate_timeseries(chosen_neurons="all", spikes=True, fig=fig, ax=ax[2])
    ax[2].set_title("CA3 rate timeseries")

    CA1s.plot_rate_timeseries(chosen_neurons="all", spikes=True, fig=fig, ax=ax[1], shift=-10, overlap=1)
    ax[1].set_title("CA1 rate timeseries")
    
    for ax_ in ax.ravel()[:-1]:
        ax_.set_xlabel("")


def learn_1D_btsp(env_params, agent_params, CA3_PC_params, CA1_params, num_rwd=200, max_steps=5000, wei_freq=100, hebbian=False):
    """Run a 1D learning experiment with BTSP learning.

    Args:
        env_params (dict): Parameters for the environment.
        agent_params (dict): Parameters for the agent.
        CA3_PC_params (dict): Parameters for the CA3 place cells.
        CA1_params (dict): Parameters for the CA1 neurons.
        num_rwd (int, optional): Target number of rewards to reach. Defaults to 200.
        max_steps (int, optional): Maximum number of steps to run. Defaults to 5000.
        wei_freq (int, optional): Frequency at which to record weights. Defaults to 100.
        hebbian (bool, optional): Whether to use Hebbian learning. Defaults to False.
    
    Returns:
        Environment, Agent, CA1 neurons, CA3 place cells
    """
    

    Env = Environment(params=env_params)

    Ag = agent.ResetAgent(Env, params=agent_params)
    CA3_PCs = PlaceCells(Ag, params=CA3_PC_params)

    CA1_params["input_layers"] = [CA3_PCs]
    CA1s = neurons.BTSPLayer(Ag, params=CA1_params)
    if not hebbian:
        CA1s.set_freeze()
    CA1s.set_btsp_learn()
    
    # run learning
    reached_last = False
    CA1_weights = [CA1s.inputs[CA3_PCs.name]["w"].copy()]
    for i in tqdm(range(max_steps)):
        Ag.update()
        CA3_PCs.update()

        btsp_targs = []
        if reached_last and CA1s.n > 1:
            btsp_targs = [CA1s.n - 1]

        reached_last = False
        if Ag.check_reset_pos():
            btsp_targs = [0]
            reached_last = True        

        CA1s.update(btsp_targs=btsp_targs)
        if not i % wei_freq:
            CA1_weights.append(CA1s.inputs[CA3_PCs.name]["w"].copy())

        if len(Ag.reached_reset_pos) >= num_rwd:
            break
    
    if len(Ag.reached_reset_pos) < num_rwd:
        print(f"Only reached the reward {len(Ag.reached_reset_pos)} times (target: {num_rwd}).")
    
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

