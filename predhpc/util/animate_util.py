import time
import warnings

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as mpl_animation
import numpy as np
import tqdm

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

        Args:
        - Ag (Resetable.Agent): The agent whose target position should be temporarily
            moved.
        - temp_target_position (float or 1D np.ndarray):
            The temporary new target position. Single value for a 1D environment.
            [x, y] for a 2D environment.
        """

        self.Agent = Ag
        self.original_target_position = Ag.target_position
        self.temp_target_position = temp_target_position

    def __enter__(self):
        """
        Temporarily moves the target position of the agent.
        """

        self.Agent.set_target_position(self.temp_target_position)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Restores the target position of the agent to the original target position.
        """

        self.Agent.set_target_position(self.original_target_position)


class TemporarilyReshufflePlaceCells:
    """
    TemporarilyReshufflePlaceCells()

    Context manager to temporarily reshuffle place cell positions.
    """

    def __init__(self, PCs, sorter):
        """
        TemporarilyReshufflePlaceCells(PCs, sorter)

        Initialises the context manager.

        Args:
        - PCs (PlaceCells): Place cells.
        - sorter (np.ndarray): Array of indices to resort the place cell positions,
            based on original sorting.
        """

        self.PCs = PCs
        self.original_sorter = PCs._current_sorter

        if len(sorter) != PCs.n:
            raise ValueError(
                f"Length of 'sorter' ({len(sorter)}) must be equal to the number of "
                f"place cells ({PCs.n})."
            )
        self.temp_sorter = sorter

    def __enter__(self):
        """
        Temporarily shuffles the place cell order.
        """

        self.PCs.shuffle_place_cell_locations(
            shuffle_sorter=self.temp_sorter, record=False
        )

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Restores the original place cell order.
        """

        self.PCs.shuffle_place_cell_locations(
            shuffle_sorter=self.original_sorter, record=False
        )


def init_linear_track_fig(n=1):
    """
    init_linear_track_fig()

    Initialises a figure with subplots for a linear track environment.

    Args:
    - n (int, optional): Number of Pyr. neurons. Default is 1.

    Returns:
    - fig (mpl_figure.Figure): Figure.
    - axes (2D np.ndarray): Array of axes, with shape (6, 1).
    """

    hei_each = min(max(1, (n - 1) // 20), 3)

    gridspec_kw = {
        "height_ratios": [0.5, 1, 0.3, hei_each, hei_each, hei_each],
        "hspace": 0.24,
    }
    num_plots = len(gridspec_kw["height_ratios"])

    height = sum(gridspec_kw["height_ratios"]) + gridspec_kw["hspace"] * (num_plots - 1)
    fig, axes = plt.subplots(
        num_plots, 1, figsize=[6, height], gridspec_kw=gridspec_kw, squeeze=False
    )

    return fig, axes


def init_openfield_fig(n=1):
    """
    init_openfield_fig()

    Initialises a figure with subplots for an open field environment.

    Args:
    - n (int, optional): Number of Pyr. neurons. Default is 1.

    Returns:
    - fig (mpl_figure.Figure): Figure.
    - axes (2D np.ndarray): Array of subplots obtained from a mosaic, but sorted from
        top left to bottom right, and with shape (5, 1).
    """

    num_top = max(1, 3 - (n - 1) // 20)
    mosaic = [
        *[["left", "right"]] * num_top,
        ["top", "top"],
        ["middle", "middle"],
        ["bottom", "bottom"],
    ]

    height = 3.75 * (5 - num_top)
    fig, axd = plt.subplot_mosaic(mosaic, layout="constrained", figsize=[7, height])

    # adjust figure boundaries
    engine = fig.get_layout_engine()
    engine.set(rect=(0.01, 0.01, 0.96, 0.93))

    axes = np.asarray(
        [[axd["left"], axd["right"], axd["top"], axd["middle"], axd["bottom"]]]
    ).T  # shape (5, 1)

    return fig, axes


def get_previous_values(previous_values, change_times, t_end=0):
    """
    get_previous_values(previous_values, change_times)

    Returns the current value and the previous values.

    Args:
    - previous_values (list): Previous values, before the final value.
    - change_times (list): Timepoints values changed. Should have the same length as
        previous_values.
    - t_end (float, optional): End timepoint for the plot. Default is 0.

    Returns:
    - current_value (float): Current value.
    - previous_values (list): Previous values.
    """

    if len(previous_values) != len(change_times):
        raise ValueError(
            "Length of 'previous_values' "
            f"({len(previous_values)}) and 'change_times' "
            f"({len(change_times)}) must be the same."
        )
    change_times = np.asarray(change_times)

    if (np.argsort(change_times) != np.arange(len(change_times))).any():
        raise ValueError("'change_times' must be in ascending order.")
    past = np.where(t_end > change_times)[0]
    if len(past) == len(change_times):
        current_value = None
        previous_values = previous_values
    else:
        current_value = previous_values[len(past)]
        previous_values = previous_values[: len(past)]

    return current_value, previous_values


def fix_xlims_and_ticks(
    axes,
    t_start,
    t_end,
    actual_t_end=None,
    dt=0.03,
    convert_to_min=False,
    ticks_last_only=False,
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
    - ticks_last_only (bool, optional): Whether to set the ticks on the last subplot
        only. Default is False.
    """

    if actual_t_end is not None:
        if np.absolute(t_end - actual_t_end) < dt / 2:
            t_end = actual_t_end

    factor = 1
    if convert_to_min:
        factor = 1 / 60

    xticks = [t_start * factor, t_end * factor]
    xticklabels = [f"{float(str(x)):.2f}" for x in xticks]
    if xticks[0] == 0:
        xticklabels[0] = "0.0"

    for i, sub_ax in enumerate(axes.ravel()):
        sub_ax.set_xlim(xticks)
        if not ticks_last_only or i == len(axes) - 1:
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticklabels)


def plot_linear_track(
    Pyrs,
    Pyrs_weights,
    Pyrs_weights_t,
    target_Pyr_idx=0,
    axes=None,
    t_start=0,
    t_end=100,
    addendum=None,
    previous_target_positions=list(),
    target_move_times=list(),
    Pyr_kwargs=dict(),
):
    """
    plot_linear_track(Pyrs, Pyrs_weights, Pyrs_weights_t)

    Plots linear track experiment for a specific timepoint. The plot consists of
    the following subplots:
        (1) 1D environment with agent's trajectory and target position.
        (2) Place cell input weights to first Pyr. soma.
        (3) (Blank subplot.)
        (3) Time series of Pyr. soma.
        (4) Time series of Pyr. apical dendrite.
        (5) Time series of Pyr. interneuron.


    Args:
    - Pyrs (two_comp_neurons.TwoComp): Pyr. neurons.
    - Pyrs_weights (list): List of input weights from place cells to Pyr. somata,
        across time, where input weights have shape (n_Pyrs, n_PCs).
    - Pyrs_weights_t (list): List of timepoints for the input weights from place
        cells to Pyr. somata.
    - target_Pyr_idx (int, optional): Index of the target Pyr. neuron for which to plot
        timeseries and input place cell weights. If "all", max across weights is
        plotted. Default is 0.
    - axes (2D np.ndarray, optional): Array of 6 subplots. Default is None.
    - t_start (float, optional): Start timepoint for the plot. Default is 0.
    - t_end (float, optional): End timepoint for the plot. Default is 100.
    - addendum (str, optional): Addendum for the title. Default is None.
    - previous_target_positions (list): Previous target positions, before the
        final position. Default is list().
    - target_move_times (list): Timepoints target moved. Should have the same length as
        previous_target_positions. Default is list().
    - Pyr_kwargs (dict, optional): Keyword arguments dictionary passed to
        two_comp_neurons.TwoComp.plot_rate_timeseries(). Default is dict().

    Returns:
    - axes (2D np.ndarray): Array of 6 subplots. If input axes is None, shape is
        (6, 1).
    """

    if axes is None:
        fig, axes = init_linear_track_fig(Pyrs.n)
    else:
        fig = np.asarray(axes).ravel()[0].figure

    ax1D = np.asarray(axes).ravel()

    suptitle = "BTSP on a linear track"
    if addendum is not None and len(addendum):
        suptitle = f"{suptitle}\n({addendum})"

    fig.suptitle(suptitle, fontweight="bold", y=0.95)

    PCs = Pyrs.SomaCompartment.inputs["PCs"]["layer"]

    Ag = Pyrs.Agent
    current_target_position, previous_target_positions = get_previous_values(
        previous_target_positions, target_move_times, t_end=t_end
    )
    if current_target_position is None:
        current_target_position = Ag.target_position
    else:
        current_target_position = np.asarray(current_target_position).reshape(-1)

    ncol = 4
    with TemporarilyMoveTarget(Ag, current_target_position):
        plot_fcts.plot_1D_reset_environment(Ag, sub_ax=ax1D[0], autosave=False)
        _, t_idx_end = plot_util.get_plotting_times(Ag.history["t"], t_start, t_end)

        # plot previous target positions
        for p, plot_position in enumerate(previous_target_positions):
            plot_position = np.asarray(plot_position).reshape(-1)[0]
            label = None
            if p == 0:
                plural = "" if len(previous_target_positions) == 1 else "s"
                label = f"prev. target{plural}"
            target_kwargs = plot_util.get_plot_marker_kwargs("target")
            ax1D[0].scatter(
                plot_position, 0, zorder=10, alpha=0.2, label=label, **target_kwargs
            )
            ncol += 1

        agent_kwargs = plot_util.get_plot_marker_kwargs("agent")
        agent_kwargs["color"] = Pyrs.SomaCompartment.color
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

        # plot place cell input weights to Pyr.
        plot_fcts.plot_recorded_1D_input_place_cell_weights(
            np.asarray(Pyrs_weights)[:, target_Pyr_idx],
            input_centres=PCs.place_cell_centres,
            weights_t=Pyrs_weights_t,
            color=PCs.color,
            sub_ax=ax1D[1],
            t_start=t_start,
            t_end=t_end,
            plot_last_FWHM=(Pyrs.n == 1),
        )

        target_Pyr_idx_str = ""
        if target_Pyr_idx >= Pyrs.n:
            raise ValueError(
                f"weight_target_idx ({target_Pyr_idx}) must be less than "
                f"Pyrs.n ({Pyrs.n})"
            )
        elif Pyrs.n > 1:
            target_Pyr_idx_str = f" (#{target_Pyr_idx + 1})"

        if current_target_position is not None:
            current_target_position = np.asarray(current_target_position).reshape(-1)[0]
            ax1D[1].axvline(current_target_position, ls="dotted", color="k")
        ax1D[1].spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax1D[1].set_xticks([])
        ax1D[1].set_yticks([])
        ax1D[1].set_ylabel(
            f"Input weights\nplace cells to Pyr. soma{target_Pyr_idx_str}"
        )
        plot_util.pad_axis(ax1D[1], axis="y", pad_prop=0.2)

        # match x lims for subplots 0 and 1
        x_min = min([ax1D[i].get_xlim()[0] for i in range(2)])
        x_max = max([ax1D[i].get_xlim()[1] for i in range(2)])
        for i in range(2):
            ax1D[i].set_xlim([x_min, x_max])

        # turn off buffer axis
        ax1D[2].axis("off")

        # plot time series
        chosen_neurons = "all" if target_Pyr_idx == "all" else [target_Pyr_idx]
        Pyrs.plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            ax=ax1D[3:],
            adjust_xlim=True,
            norm_by="max_per",
            single_x_axis=True,
            chosen_neurons=chosen_neurons,
            autosave=False,
            separate_axes=True,
            **Pyr_kwargs,
        )

        t = Pyrs.SomaCompartment.get_plotting_times(t_start, t_end)[0]
        actual_t_end = t[-1] if len(t) else None
        fix_xlims_and_ticks(
            ax1D[3:],
            t_start,
            t_end,
            actual_t_end=actual_t_end,
            dt=Pyrs.Agent.dt,
            convert_to_min=True,
            ticks_last_only=True,
        )

        ax1D[3].set_ylabel("Soma")
        ax1D[4].set_ylabel("Apical dendrite")
        ax1D[5].set_ylabel("Interneuron")

    return axes


def plot_openfield(
    Pyrs,
    Pyrs_weights,
    Pyrs_weights_t,
    target_Pyr_idx=0,
    s=75,
    axes=None,
    t_start=0,
    t_end=100,
    addendum=None,
    traj_kwargs=dict(),
    Pyr_kwargs=dict(),
):
    """
    plot_openfield(Pyrs, Pyrs_weights, Pyrs_weights_t)

    Plots open field experiment for a specific timepoint. The plot consists of
    the following subplots:
        (1) Agent's trajectory (top left).
        (2) Place cell input weights to Pyr. soma (top right).
        (3) Time series of Pyr. soma (next top, full width).
        (4) Time series of Pyr. apical dendrite (mid, full width).
        (5) Time series of Pyr. interneuron (bottom, full width).

    Args:
    - Pyrs (Resetable.Neuron): Pyr. neuron.
    - Pyrs_weights (list): List of input weights from place cells to Pyr. somata,
        across time, where input weights have shape (n_Pyrs, n_PCs).
    - Pyrs_weights_t (list): List of timepoints for the input weights from place
        cells to Pyr. somata.
    - target_Pyr_idx (int, optional): Index of the target Pyr. neuron for which to plot
        timeseries and input place cell weights. If "all", max across weights is
        plotted. Default is 0.
    - s (int, optional): Marker size for the input weights. Default is 75.
    - axes (2D np.ndarray, optional): Array of 5 subplots. Default is None.
    - t_start (float, optional): Start timepoint for the plot. Default is 0.
    - t_end (float, optional): End timepoint for the plot. Default is 100.
    - addendum (str, optional): Addendum for the title. Default is None.
    - traj_kwargs (dict, optional): Keyword arguments dictionary passed to
        agent.ResetableAgent.plot_trajectories(). Default is dict().
    - Pyr_kwargs (dict, optional): Keyword arguments dictionary passed to
        two_comp_neurons.TwoComp.plot_rate_timeseries(). Default is dict().

    Returns:
    - axes (2D np.ndarray): Array of 5 subplots. If input axes is None,
        shape is (5, 1).
    """

    if axes is None:
        fig, axes = init_openfield_fig(Pyrs.n)
    else:
        fig = np.asarray(axes).ravel()[0].figure

    suptitle = "BTSP in an open field"
    if addendum is not None and len(addendum):
        suptitle = f"{suptitle}\n({addendum})"

    fig.suptitle(suptitle, fontweight="bold", y=0.98)

    ax1D = np.asarray(axes).ravel()
    Pyrs.Agent.plot_trajectories(
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

    # plot place cell input weights to Pyr.
    Pyrs_weights = np.asarray(Pyrs_weights)

    target_Pyr_idx_str = ""
    if target_Pyr_idx == "all":
        target_Pyr_idx = 0
        chosen_neurons = "all"
        if Pyrs.n > 1:
            Pyrs_weights = Pyrs_weights.max(axis=1, keepdims=True)
            target_Pyr_idx_str = f" (max)"
    elif target_Pyr_idx >= Pyrs.n:
        raise ValueError(
            f"weight_target_idx ({target_Pyr_idx}) must be less than "
            f"Pyrs.n ({Pyrs.n})"
        )
    else:
        chosen_neurons = [target_Pyr_idx]
        if Pyrs.n > 1:
            target_Pyr_idx_str = f" (#{target_Pyr_idx})"

    weight_idx = np.where(np.asarray(Pyrs_weights_t) < t_end)[0][-1]

    # handle any reshuffling of place cells
    PCs = Pyrs.SomaCompartment.inputs["PCs"]["layer"]

    previous_values = ([np.arange(PCs.n)] + PCs.shuffle_sorters)[:-1]
    current_sorter, _ = get_previous_values(
        previous_values, PCs.shuffle_times, t_end=t_end
    )
    if current_sorter is None:
        current_sorter = PCs._current_sorter

    with TemporarilyReshufflePlaceCells(PCs, current_sorter):
        plot_fcts.plot_2D_input_place_cell_weights(
            Pyrs.SomaCompartment,
            PCs_input_name="PCs",
            place_weights=Pyrs_weights[weight_idx][target_Pyr_idx : target_Pyr_idx + 1],
            alpha=0.3,
            s=s,
            chosen_neurons=chosen_neurons,
            t_end=t_end,
            plot_BTSP_events=True,
            no_legend=True,
            ax=ax1D[1],
        )
    ax1D[1].set_ylabel("")
    ax1D[1].set_title(f"Input weights from PCs to Pyr. soma{target_Pyr_idx_str}")

    # plot time series
    Pyrs.plot_rate_timeseries(
        t_start=t_start,
        t_end=t_end,
        ax=ax1D[2:],
        adjust_xlim=True,
        norm_by="max_per",
        single_x_axis=True,
        chosen_neurons=chosen_neurons,
        autosave=False,
        separate_axes=True,
        **Pyr_kwargs,
    )

    t = Pyrs.SomaCompartment.get_plotting_times(t_start, t_end)[0]
    actual_t_end = t[-1] if len(t) else None
    fix_xlims_and_ticks(
        ax1D[2:],
        t_start,
        t_end,
        actual_t_end=actual_t_end,
        dt=Pyrs.Agent.dt,
        convert_to_min=True,
        ticks_last_only=True,
    )

    ax1D[0].set_xlabel("")
    ax1D[2].set_ylabel("Soma")
    ax1D[3].set_ylabel("Apical dendrite")
    ax1D[4].set_ylabel("Interneuron")

    return axes


def animate(
    Pyrs,
    Pyrs_weights,
    Pyrs_weights_t,
    target_Pyr_idx=0,
    t_start=None,
    t_end=None,
    fps=8,
    speed_up=3,
    environment="linear_track",
    savename="animation",
    embed_limit=None,
    progress_bar=True,
    autosave=None,
    **kwargs,
):
    """
    animate(Pyrs, Pyrs_weights, Pyrs_weights_t)

    Animates the agent's trajectory and the activity of the Pyr. neurons over time.

    Args:
    - Pyrs (two_comp_neurons.TwoComp): Pyr. neurons.
    - Pyrs_weights (list): List of input weights from place cells to Pyr. somata,
        across time, where input weights have shape (n_Pyrs, n_PCs).
    - Pyrs_weights_t (list): List of timepoints for the input weights from place
        cells to Pyr. somata.
    - target_Pyr_idx (int, optional): Index of the target Pyr. neuron for which to plot
        timeseries and input place cell weights. If "all", max across weights is
        plotted. Default is 0.
    - t_start (float, optional): Start timepoint for the animation. Default is None.
    - t_end (float, optional): End timepoint for the animation. Default is None.
    - fps (int, optional): Frames per second. Default is 10.
    - speed_up (int, optional): Speed up factor. Default is 3.
    - environment (str, optional): Environment in which the agent is moving. Default
        is "linear_track".
    - savename (str, optional): Name of the file to save the animation. Default is
        "animation".
    - embed_limit (int, optional): Limit for embedding the animation. Default is None.
    - progress_bar (bool, optional): Whether to show a progress bar. Default is True.
    - autosave (bool, optional): Whether to autosave the animation. If None, the
        global autosave setting for ratinabox is used. Default is None.

    Keyword Args:
    - kwargs: Additional keyword arguments passed to plotting function.

    Returns:
    - anim (matplotlib.animation.FuncAnimation): Animation object.
    """

    start_time = time.perf_counter()

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

    n = Pyrs.n if target_Pyr_idx == "all" else 1
    fig, axes = init_fct(n)

    plt.rcParams["animation.html"] = "jshtml"  # for animation rendering in juypter

    dt = 1 / fps

    t = Pyrs.Agent.get_plotting_times(t_start, t_end)[0]
    t_start, t_end = t[0], t[-1]

    def animate_(i, axes, dt, t_start=0, speed_up=3, kwargs=dict()):
        """
        animate_(i, axes, dt)

        Plots a single frame for an animation of the agent's trajectory and the
        activity of the Pyr. neurons over time.

        Args:
        - i (int): Frame index.
        - axes (2D np.ndarray): Array of subplots.
        - dt (float): Time step for the animation frame.
        - t_start (float): Start timepoint for the animation frame. Default is 0.
        - speed_up (int): Speed up factor. Default is 3.
        - kwargs (dict): Dictionary of additional keyword arguments passed to plotting
            function. Default is dict().
        """

        t_end = t_start + (i + 1) * speed_up * dt

        ax1D = np.asarray(axes).ravel()
        if environment == "openfield":
            all_axes = ax1D[-1].figure.axes
            if len(all_axes) == 6:
                fig.delaxes(all_axes[5])

        for sub_ax in ax1D:
            sub_ax.clear()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="This figure was using"
            )

            plot_fct(
                Pyrs,
                Pyrs_weights,
                Pyrs_weights_t,
                target_Pyr_idx=target_Pyr_idx,
                axes=axes,
                t_start=t_start,
                t_end=t_end,
                **kwargs,
            )

        if environment == "openfield":
            plot_util.remove_duplicate_handle_labels(ax1D[0])
            plt.close()
        return

    frames = int((t_end - t_start) / (dt * speed_up))
    if progress_bar:
        frames = tqdm.tqdm(range(frames), position=0, leave=True)

    anim = mpl_animation.FuncAnimation(
        fig,
        animate_,
        interval=1000 * dt,
        frames=frames,
        blit=False,
        fargs=(axes, dt, t_start, speed_up, kwargs),
    )

    rutils.save_animation(anim, savename, anim_save_types=["mp4", "gif"], save=autosave)

    time_str = gen_util.get_duration_str(start_time)
    print(f"Animation took {time_str} to create.")

    return anim
