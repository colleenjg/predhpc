import warnings

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as mpl_animation
import numpy as np

from ratinabox import utils as rutils  # type: ignore[import]

from predhpc import plot_fcts
from predhpc.util import plot_util


class TemporarilyMoveTarget:
    """
    TemporarilyMoveTarget()

    Context manager to temporarily move the target position of an agent.
    """

    def __init__(self, Ag, temp_target_position):
        """
        TemporarilyMoveTarget(Ag, temp_target_position)

        Initialises the context manager.

        Attributes:
        - Ag (Resetable.Agent): The agent whose target position should be temporarily
            moved.
        - original_target_position (float or 1D np.ndarray):
            The original target position. Single value for a 1D environment.
            [x, y] for a 2D environment.
        - temp_target_position (float or 1D np.ndarray):
            The temporary new target position. Single value for a 1D environment.
            [x, y] for a 2D environment.

        Args:
        - Ag (Resetable.Agent): The agent whose target position should be temporarily
            moved.
        - temp_target_position (float or 1D np.ndarray):
            The temporary new target position. Single value for a 1D environment.
            [x, y] for a 2D environment.
        """

        self.Ag = Ag
        self.original_target_position = Ag.target_position
        self.temp_target_position = temp_target_position

    def __enter__(self):
        """
        Temporarily moves the target position of the agent.
        """

        self.Ag.set_target_position(self.temp_target_position)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Restores the target position of the agent to the original target position.
        """

        self.Ag.set_target_position(self.original_target_position)


def init_linear_track_fig():
    """
    init_linear_track_fig()

    Initialises a figure with subplots for a linear track environment.

    Returns:
    - fig (mpl_figure.Figure): Figure.
    - axes (2D np.ndarray): Array of axes, with shape (6, 1).
    """

    gridspec_kw = {
        "height_ratios": [0.5, 1, 0.3, 1, 1, 1],
        "hspace": 0.2,
    }
    fig, axes = plt.subplots(
        6, 1, figsize=[6, 6], gridspec_kw=gridspec_kw, squeeze=False
    )

    return fig, axes


def init_openfield_fig():
    """
    init_openfield_fig()

    Initialises a figure with subplots for an open field environment.

    Returns:
    - fig (mpl_figure.Figure): Figure.
    - axes (2D np.ndarray): Array of subplots obtained from a mosaic, but sorted from
        top left to bottom right, and with shape (5, 1).
    """

    mosaic = [
        ["left", "right"],
        ["left", "right"],
        ["left", "right"],
        ["top", "top"],
        ["middle", "middle"],
        ["bottom", "bottom"],
    ]
    fig, axd = plt.subplot_mosaic(mosaic, layout="constrained", figsize=[7, 6])
    axes = np.asarray(
        [[axd["left"], axd["right"], axd["top"], axd["middle"], axd["bottom"]]]
    ).T  # shape (5, 1)

    return fig, axes


def get_previous_target_positions(
    previous_target_positions, target_move_times, t_end=0
):
    """
    get_previous_target_positions(previous_target_positions, target_move_times)

    Returns the current target position and the previous target positions.

    Args:
    - previous_target_positions (list): Previous target positions, before the final
        position.
    - target_move_times (list): Timepoints target moved. Should have the same length as
        previous_target_positions.
    - t_end (float, optional): End timepoint for the plot. Default is 0.

    Returns:
    - current_target_position (float): Current target position.
    - previous_target_positions (list): Previous target positions.
    """

    if len(previous_target_positions) != len(target_move_times):
        raise ValueError(
            "Length of 'previous_target_positions' "
            f"({len(previous_target_positions)}) and 'target_move_times' "
            f"({len(target_move_times)}) must be the same."
        )
    target_move_times = np.asarray(target_move_times)
    if np.argsort(target_move_times) != np.arange(len(target_move_times)):
        raise ValueError("'target_move_times' must be in ascending order.")
    past = np.where(t_end > target_move_times)[0]
    if len(past) == len(target_move_times):
        current_target_position = None
        previous_target_positions = previous_target_positions
    else:
        current_target_position = previous_target_positions[len(past)]
        previous_target_positions = previous_target_positions[: len(past)]

    return current_target_position, previous_target_positions


def fix_xlims_and_ticks(
    axes, t_start, t_end, actual_t_end=None, dt=0.03, convert_to_min=False
):
    """
    fix_xaxes(axes, t_start, t_end)

    Fixes the x-axis of the subplots.

    Args:
    - axes (2D np.ndarray): Array of subplots.
    - t_start (float): Start timepoint for the plot.
    - t_end (float): End timepoint for the plot.
    - actual_t_end (float, optional): Actual end timepoint to use if it is close enough
        to t_end. Default is None.
    - dt (float, optional): Time step to use to measure how close actual_t_end is to
        t_end. Default is 0.03.
    - convert_to_min (bool, optional): Whether to convert the x-axis to minutes.
        Default is False.
    """

    if actual_t_end is not None:
        if np.absolute(t_end - actual_t_end) < dt / 2:
            t_end = actual_t_end

    factor = 1
    if convert_to_min:
        factor = 1 / 60

    for sub_ax in axes.ravel():
        xticks = [t_start * factor, t_end * factor]
        xticklabels = [f"{float(str(x)):.2f}" for x in xticks]
        if xticks[0] == 0:
            xticklabels[0] = "0.0"
        sub_ax.set_xlim(xticks)
        sub_ax.set_xticks(xticks)
        sub_ax.set_xticklabels(xticklabels)


def plot_linear_track(
    Ag,
    CA1s,
    CA1s_weights,
    CA1s_weights_t,
    target_CA1_idx=0,
    axes=None,
    t_start=0,
    t_end=100,
    addendum=None,
    previous_target_positions=list(),
    target_move_times=list(),
    **kwargs,
):
    """
    plot_linear_track(Ag, CA1s, CA1s_weights, CA1s_weights_t)

    Plots linear track experiment for a specific timepoint. The plot consists of
    the following subplots:
        (1) 1D environment with agent's trajectory and target position.
        (2) CA3 input weights to first CA1 soma.
        (3) (Blank subplot.)
        (3) Time series of CA1 soma.
        (4) Time series of CA1 apical dendrite.
        (5) Time series of CA1 interneuron.


    Args:
    - Ag (agent.ResetableAgent): Agent.
    - CA1s (two_comp_neurons.TwoComp): CA1 neurons.
    - CA1s_weights (list): List of input weights from CA3 place cells to CA1 somata,
        across time, where input weights have shape (n_CA1, n_CA3).
    - CA1s_weights_t (list): List of timepoints for the input weights from CA3 place
        cells to CA1 somata.
    - target_CA1_idx (int, optional): Index of the target CA1 neuron for which to plot
        input CA3 weights. Default is 0.
    - axes (2D np.ndarray, optional): Array of 6 subplots. Default is None.
    - t_start (float, optional): Start timepoint for the plot. Default is 0.
    - t_end (float, optional): End timepoint for the plot. Default is 100.
    - addendum (str, optional): Addendum for the title. Default is None.
    - previous_target_positions (list, optional): Previous target positions, before the
        final position. Default is None.
    - target_move_times (list, optional): Timepoints target moved. Should have the same
    length as previous_target_positions. Default is None.

    Keyword args:
    - **kwargs: Additional keyword arguments for plotting, passed to
        two_comp_neurons.TwoComp.plot_rate_timeseries().

    Returns:
    - axes (2D np.ndarray): Array of 6 subplots. If input axes is None, shape is
        (6, 1).
    """

    if axes is None:
        fig, axes = init_linear_track_fig()
    else:
        fig = np.asarray(axes).ravel()[0].figure

    ax1D = np.asarray(axes).ravel()

    suptitle = "BTSP on a linear track"
    if addendum is not None and len(addendum):
        suptitle = f"{suptitle}\n({addendum})"

    fig.suptitle(suptitle, fontweight="bold", y=0.95)

    CA3_PCs = CA1s.SomaCompartment.inputs["CA3_PCs"]["layer"]

    current_target_position, previous_target_positions = get_previous_target_positions(
        previous_target_positions, target_move_times, t_end=t_end
    )
    if current_target_position is None:
        current_target_position = Ag.target_position

    ncol = 4
    with TemporarilyMoveTarget(Ag, current_target_position):
        plot_fcts.plot_1D_reset_environment(Ag, sub_ax=ax1D[0], autosave=False)
        _, t_idx_end = plot_util.get_plotting_times(Ag.history["t"], t_start, t_end)

        # plot previous target positions
        for p, plot_position in enumerate(previous_target_positions):
            label = None
            if p == 0:
                plural = "" if len(previous_target_positions) == 1 else "s"
                label = f"prev. target{plural}"
            target_kwargs = plot_util.get_plot_marker_kwargs("target")
            ax1D[0].scatter(
                plot_position[0], 0, zorder=10, alpha=0.2, label=label, **target_kwargs
            )
            ncol += 1

        agent_kwargs = plot_util.get_plot_marker_kwargs("agent")
        agent_kwargs["color"] = CA1s.SomaCompartment.color
        ax1D[0].scatter(
            Ag.history["pos"][t_idx_end],
            0,
            zorder=10,
            alpha=0.7,
            label="agent",
            **agent_kwargs,
        )
        ax1D[0].set_title("")
        ax1D[0].set_xticks([])
        ax1D[0].set_xlabel("")
        ax1D[0].set_ylim([-0.5, 2.0])
        ax1D[0].legend(ncol=ncol, frameon=False, loc="upper right")

        # plot CA3 input weights to CA1
        plot_fcts.plot_previous_1D_input_place_cell_weights(
            np.asarray(CA1s_weights)[:, target_CA1_idx],
            CA1s_weights_t,
            input_centres=CA3_PCs.place_cell_centres,
            color=CA3_PCs.color,
            sub_ax=ax1D[1],
            t_start=t_start,
            t_end=t_end,
        )

        target_CA1_idx_str = ""
        if target_CA1_idx >= CA1s.n:
            raise ValueError(
                f"weight_target_idx ({target_CA1_idx}) must be less than "
                f"CA1s.n ({CA1s.n})"
            )
        elif CA1s.n > 1:
            target_CA1_idx_str = f" (#{target_CA1_idx})"

        ax1D[1].axvline(current_target_position[0], ls="dashed", color="k")
        ax1D[1].spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax1D[1].set_xticks([])
        ax1D[1].set_yticks([])
        ax1D[1].set_ylabel(f"Input weights\nCA3 to CA1 soma{target_CA1_idx_str}")
        plot_util.pad_axis(ax1D[1], axis="y", pad_prop=0.2)

        # turn off buffer axis
        ax1D[2].axis("off")

        # plot time series
        CA1s.plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            ax=ax1D[3:],
            adjust_xlim=True,
            autosave=False,
            separate_axes=True,
            **kwargs,
        )

        t = CA1s.SomaCompartment.get_plotting_times(t_start, t_end)[0]
        actual_t_end = t[-1] if len(t) else None
        fix_xlims_and_ticks(
            ax1D[3:],
            t_start,
            t_end,
            actual_t_end=actual_t_end,
            dt=CA1s.Agent.dt,
            convert_to_min=True,
        )

        ax1D[3].set_ylabel("Soma")
        ax1D[4].set_ylabel("Apical dendrite")
        ax1D[5].set_ylabel("Interneuron")

    return axes


def plot_openfield(
    Ag,
    CA1s,
    CA1s_weights,
    CA1s_weights_t,
    target_CA1_idx=0,
    axes=None,
    t_start=0,
    t_end=100,
    addendum=None,
    traj_kwargs=dict(),
    CA1_kwargs=dict(),
):
    """
    plot_openfield(Ag, CA1s, CA1s_weights, CA1s_weights_t)

    Plots open field experiment for a specific timepoint. The plot consists of
    the following subplots:
        (1) Agent's trajectory (top left).
        (2) CA3 input weights to CA1 soma (top right).
        (3) Time series of CA1 soma (next top, full width).
        (4) Time series of CA1 apical dendrite (mid, full width).
        (5) Time series of CA1 interneuron (bottom, full width).

    Args:
    - Ag (agent.ResetableAgent): Agent.
    - CA1s (Resetable.Neuron): CA1 neuron.
    - CA1s_weights (list): List of input weights from CA3 place cells to CA1 somata,
        across time, where input weights have shape (n_CA1, n_CA3).
    - CA1s_weights_t (list): List of timepoints for the input weights from CA3 place
        cells to CA1 somata.
    - target_CA1_idx (int, optional): Index of the target CA1 neuron for which to plot
        input CA3 weights. Default is 0.
    - axes (2D np.ndarray, optional): Array of 5 subplots. Default is None.
    - t_start (float, optional): Start timepoint for the plot. Default is 0.
    - t_end (float, optional): End timepoint for the plot. Default is 100.
    - addendum (str, optional): Addendum for the title. Default is None.
    - traj_kwargs (dict, optional): Keyword arguments passed to
        agent.ResetableAgent.plot_trajectories(). Default is dict().
    - CA1_kwargs (dict, optional): Keyword arguments passed to
        two_comp_neurons.TwoComp.plot_rate_timeseries(). Default is dict().

    Returns:
    - axes (2D np.ndarray): Array of 5 subplots. If input axes is None,
        shape is (5, 1).
    """

    if axes is None:
        fig, axes = init_openfield_fig()
    else:
        fig = np.asarray(axes).ravel()[0].figure

    suptitle = "BTSP in an open field"
    if addendum is not None and len(addendum):
        suptitle = f"{suptitle}\n({addendum})"

    fig.suptitle(suptitle, fontweight="bold", y=1.02)

    ax1D = np.asarray(axes).ravel()
    Ag.plot_trajectories(
        t_start=t_start,
        t_end=t_end,
        sub_ax=ax1D[0],
        decay_point_size=True,
        decay_point_timescale=1,
        plot_head_direction=True,
        alpha=0.3,
        autosave=False,
        **traj_kwargs,
    )

    ax1D[0].set_title("Trajectories")

    # plot CA3 input weights to CA1
    target_CA1_idx_str = ""
    if target_CA1_idx >= CA1s.n:
        raise ValueError(
            f"weight_target_idx ({target_CA1_idx}) must be less than "
            f"CA1s.n ({CA1s.n})"
        )
    elif CA1s.n > 1:
        target_CA1_idx_str = f" (#{target_CA1_idx})"

    weight_idx = np.where(np.asarray(CA1s_weights_t) < t_end)[0][-1]
    plot_fcts.plot_2D_input_place_cell_weights(
        CA1s.SomaCompartment,
        PCs_input_name="CA3_PCs",
        place_weights=CA1s_weights[weight_idx][target_CA1_idx : target_CA1_idx + 1],
        alpha=0.3,
        t_end=t_end,
        plot_BTSP_events=True,
        no_legend=True,
        ax=ax1D[1],
    )
    ax1D[1].set_ylabel("")
    ax1D[1].set_title(f"Input weights from CA3 to CA1 soma{target_CA1_idx_str}")

    # plot time series
    CA1s.plot_rate_timeseries(
        t_start=t_start,
        t_end=t_end,
        ax=ax1D[2:],
        adjust_xlim=True,
        autosave=False,
        separate_axes=True,
        **CA1_kwargs,
    )

    t = CA1s.SomaCompartment.get_plotting_times(t_start, t_end)[0]
    actual_t_end = t[-1] if len(t) else None
    fix_xlims_and_ticks(
        ax1D[2:],
        t_start,
        t_end,
        actual_t_end=actual_t_end,
        dt=CA1s.Agent.dt,
        convert_to_min=True,
    )

    ax1D[0].set_xlabel("")
    ax1D[2].set_ylabel("Soma")
    ax1D[3].set_ylabel("Apical dendrite")
    ax1D[4].set_ylabel("Interneuron")

    return axes


def animate(
    Ag,
    CA1s,
    CA1s_weights,
    CA1s_weights_t,
    target_CA1_idx=0,
    t_start=None,
    t_end=None,
    fps=5,
    speed_up=6,
    savename="animation",
    embed_limit=None,
    environment="linear_track",
    traj_kwargs=dict(),
    CA1_kwargs=dict(),
    autosave=None,
):
    """
    animate(Ag, CA1s, CA1s_weights, CA1s_weights_t)

    Animates the agent's trajectory and the activity of the CA1 neurons over time.

    Args:
    - Ag (agent.ResetableAgent): Agent.
    - CA1s (two_comp_neurons.TwoComp): CA1 neurons.
    - CA1s_weights (list): List of input weights from CA3 place cells to CA1 somata,
        across time, where input weights have shape (n_CA1, n_CA3).
    - CA1s_weights_t (list): List of timepoints for the input weights from CA3 place
        cells to CA1 somata.
    - target_CA1_idx (int, optional): Index of the target CA1 neuron for which to plot
        input CA3 weights. Default is 0.
    - t_start (float, optional): Start timepoint for the animation. Default is None.
    - t_end (float, optional): End timepoint for the animation. Default is None.
    - fps (int, optional): Frames per second. Default is 5.
    - speed_up (int, optional): Speed up factor. Default is 6.
    - savename (str, optional): Name of the file to save the animation. Default is
        "animation".
    - embed_limit (int, optional): Limit for embedding the animation. Default is None.
    - environment (str, optional): Environment in which the agent is moving. Default
        is "linear_track".
    - traj_kwargs (dict, optional): Keyword arguments passed to
        agent.ResetableAgent.plot_trajectories(). Default is dict().
    - CA1_kwargs (dict, optional): Keyword arguments passed to
        two_comp_neurons.TwoComp.plot_rate_timeseries(). Default is dict().
    - autosave (bool, optional): Whether to autosave the animation. If None, the
        global autosave setting for ratinabox is used. Default is None.

    Returns:
    - anim (matplotlib.animation.FuncAnimation): Animation object.
    """

    if embed_limit is not None:
        matplotlib.rcParams["animation.embed_limit"] = embed_limit

    if environment == "linear_track":
        init_fct = init_linear_track_fig
        plot_fct = plot_linear_track
    elif environment == "openfield":
        init_fct = init_openfield_fig
        plot_fct = plot_openfield
    else:
        raise ValueError(f"Unknown environment: {environment}")

    fig, axes = init_fct()

    plt.rcParams["animation.html"] = "jshtml"  # for animation rendering in juypter

    dt = 1 / fps
    if t_start == None:
        t_start = Ag.history["t"][0]
    if t_end == None:
        t_end = Ag.history["t"][-1]

    def animate_(i, axes, t_start, speed_up, dt, traj_kwargs, CA1_kwargs):
        """
        animate_(i, axes, t_start, speed_up, dt, traj_kwargs, CA1_kwargs)

        Plots a single frame for an animation of the agent's trajectory and the
        activity of the CA1 neurons over time.

        Args:
        - i (int): Frame index.
        - axes (2D np.ndarray): Array of subplots.
        - t_start (float): Start timepoint for the animation frame. Default is None.
        - speed_up (int): Speed up factor.
        - dt (float): Time step for the animation frame.
        - traj_kwargs (dict): Keyword arguments passed to
            agent.ResetableAgent.plot_trajectories().
        - CA1_kwargs (dict): Keyword arguments passed to
            two_comp_neurons.TwoComp.plot_rate_timeseries().
        """

        t_end = t_start + (i + 1) * speed_up * dt

        ax1D = np.asarray(axes).ravel()
        if environment == "openfield":
            if len(ax1D[1].images) == 2:
                cbar = ax1D[1].images[0].colorbar
                if cbar is None:
                    raise RuntimeError("Colorbar not found.")
                cbar.set_label("")
                cbar.set_ticks([])

        for sub_ax in ax1D:
            sub_ax.clear()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="This figure was using"
            )
            plot_fct(
                Ag,
                CA1s,
                CA1s_weights,
                CA1s_weights_t,
                target_CA1_idx=target_CA1_idx,
                axes=axes,
                t_start=t_start,
                t_end=t_end,
                traj_kwargs=traj_kwargs,
                CA1_kwargs=CA1_kwargs,
            )

        if environment == "openfield":
            plot_util.remove_prev_handle_labels(ax1D[0])
            plt.close()
        return

    anim = mpl_animation.FuncAnimation(
        fig,
        animate_,
        interval=1000 * dt,
        frames=int((t_end - t_start) / (dt * speed_up)),
        blit=False,
        fargs=(axes, t_start, speed_up, dt, traj_kwargs, CA1_kwargs),
    )

    rutils.save_animation(anim, savename, anim_save_types=["gif", "mp4"], save=autosave)

    return anim
