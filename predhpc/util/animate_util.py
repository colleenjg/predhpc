import time
import warnings

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as mpl_animation
import numpy as np
from tqdm.auto import tqdm

from ratinabox import utils as rutils  # type: ignore[import]

from predhpc import plot_fcts
from predhpc.util import plot_util, gen_util


class TemporarilyMoveObject:
    """
    TemporarilyMoveObject()

    Context manager to temporarily move the object position of an agent.
    """

    def __init__(self, Env, temp_env_object):
        """
        TemporarilyMoveObject(Env, temp_env_object)

        Initialises the context manager.

        Args:
        - Env (Resetable.Environment): The environment whose object position should be
            temporarily moved.
        - temp_env_object (float or 1D np.ndarray): The temporary new object
            position. Single value for a 1D environment.
        """

        if not hasattr(Env, "env_object"):
            raise ValueError("Environment must have 'env_object' attribute.")

        self.Env = Env
        self.original_env_object = Env.env_object
        self.temp_env_object = Env.format_position(temp_env_object)

    def __enter__(self):
        """
        Temporarily moves the object position of the agent.
        """

        self.Env.set_env_object(self.temp_env_object)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Restores the object position of the agent to the original object position.
        """

        self.Env.set_env_object(self.original_env_object)


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


def get_suptitle(environment="linear", addendum=None, speed_up=None):
    """
    get_suptitle()

    Returns the suptitle for the figure based on the environment and other parameters.

    Args:
    - environment (str, optional): Environment type. Default is "linear".
    - addendum (str, optional): Additional text for the suptitle, placed in parentheses.
        Default is None.
    - speed_up (float, optional): Speed up factor for the animation to include in
        title, in parentheses. Default is None.

    Returns:
    - suptitle (str): Suptitle for the figure.
    - y (float): y position for the suptitle.
    """

    if environment == "linear":
        suptitle = "Linear track"
    elif environment == "tmaze":
        suptitle = "T-maze"
    elif environment == "openfield":
        suptitle = "Open field"
    else:
        raise ValueError(f"Environment '{environment}' not recognized.")

    addendum = addendum or ""
    if len(addendum) or speed_up is not None:
        if len(addendum) > 20:
            separator = "\n"
            y = 0.99
        else:
            separator = " "
            y = 0.97
        if speed_up is not None:
            speed_up_str = f"{speed_up}x"
            if len(addendum):
                addendum = f"{addendum}, {speed_up_str}"
            else:
                addendum = speed_up_str
        suptitle = f"{suptitle}{separator}({addendum})"

    return suptitle, y


def get_target_Pyr_idx_str(target_Pyr_idx, Pyrs_n=None):
    """
    get_target_Pyr_idx_str(target_Pyr_idx)

    Returns a string to be included in the title based on the target Pyr. neuron index.

    Args:
    - target_Pyr_idx (int or str): Index of the target Pyr. neuron for which to plot
        timeseries and input place cell weights. If "all", max across weights is
        plotted. Default is 0.

    Returns:
    - target_Pyr_idx_str (str): String to be included in the title based on the target
        Pyr. neuron index.
    """

    target_Pyr_idx_str = ""
    if target_Pyr_idx == "all":
        if Pyrs_n is None or Pyrs_n > 1:
            target_Pyr_idx_str = f" (max)"
        else:
            target_Pyr_idx_str = ""
    elif Pyrs_n is not None and target_Pyr_idx >= Pyrs_n:
        raise ValueError(
            f"weight_target_idx ({target_Pyr_idx}) must be less than "
            f"number of Pyrs ({Pyrs_n})."
        )
    elif Pyrs_n is None or Pyrs_n > 1:
        target_Pyr_idx_str = f" (neuron #{target_Pyr_idx + 1})"

    return target_Pyr_idx_str


def get_linear_track_rate_timeseries_height(n=1):
    """
    get_linear_track_rate_timeseries_height()

    Returns the height for the rate time series subplots in the linear track subplots,
    based on the number of Pyr. neurons.

    Args:
    - n (int, optional): Number of Pyr. neurons. Default is 1.

    Returns:
    - hei_each (float): Height for each time series subplot.
    """

    hei_each = min(max(1, (n - 1) // 20), 3)

    return hei_each


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

    hei_each = get_linear_track_rate_timeseries_height(n)

    gridspec_kw = {
        "height_ratios": [1, 0.6, 0.3, hei_each, hei_each, hei_each],
        "hspace": 0.5,
    }
    num_plots = len(gridspec_kw["height_ratios"])

    height = sum(gridspec_kw["height_ratios"]) + gridspec_kw["hspace"] * (num_plots - 1)
    fig, axes = plt.subplots(
        num_plots, 1, figsize=[5.5, height], gridspec_kw=gridspec_kw, squeeze=False
    )

    return fig, axes


def get_openfield_num_per(n=1, num_top=6):
    """
    get_openfield_num_per()

    Returns the number of subplots per row for the open field figure, based on the
    number of Pyr. neurons.

    Args:
    - n (int, optional): Number of Pyr. neurons. Default is 1.

    Returns:
    - num_per (int): Number of subplots per row for the open field figure.
    """

    num_per = int(np.around(num_top * (1 / 3 + min(1, n / 60))))

    return num_per


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

    width = 10

    num_top = 6
    height_per = (width - 2) / (2 * num_top)

    num_per = get_openfield_num_per(n, num_top)
    height = height_per * (num_top + num_per * 3)

    mosaic = [
        *[["left", "right"]] * num_top,
        *[["top", "top"]] * num_per,
        *[["middle", "middle"]] * num_per,
        *[["bottom", "bottom"]] * num_per,
    ]

    gridspec_kw = {"hspace": 0.3}

    fig, axd = plt.subplot_mosaic(
        mosaic, layout="constrained", figsize=[width, height], gridspec_kw=gridspec_kw
    )

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


def plot_linear_track_trajectory(Ag, t=0, sub_ax=None):
    """
    plot_linear_track_trajectory(Ag, t=0)

    Plots the agent's trajectory and target position for a linear track environment.

    Args:
    - Ag (Resetable.Agent): Agent.
    - t (float, optional): Timepoint for the plot. Default is 0.
    - sub_ax (mpl_axes.Axes): Subplot to plot on. If None, a new figure and subplot
        are created. Default is None.

    Returns:
    - sub_ax (mpl_axes.Axes): Subplot with the trajectory plot.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=[5.5, 0.6])

    _, t_idx_end = plot_util.get_plotting_times(Ag.get_t_history(), None, t)

    current, previous = Ag.get_target_positions(t_end=t, unique=True)
    with TemporarilyMoveObject(Ag.Environment, temp_env_object=current):
        Ag.Environment.plot_environment(sub_ax=sub_ax, autosave=False)

    # plot previous object positions
    ncol = 4
    obj_name = Ag.Environment.object_type_name
    for p, plot_position in enumerate(previous):
        plot_position = np.asarray(plot_position).reshape(-1)[0]
        label = None
        if p == 0:
            plural = "" if len(previous) == 1 else "s"
            label = f"prev. {obj_name} location{plural}"
        target_kwargs = plot_util.get_plot_marker_kwargs(obj_name)
        sub_ax.scatter(
            plot_position, 0, zorder=10, alpha=0.2, label=label, **target_kwargs
        )
        ncol += 1

    agent_kwargs = plot_util.get_plot_marker_kwargs("agent")
    agent_kwargs["color"] = "k"
    sub_ax.scatter(
        Ag.history["pos"][t_idx_end],
        0,
        zorder=10,
        alpha=0.6,
        label="agent",
        **agent_kwargs,
    )
    sub_ax.set_title("Trajectory", y=1.02)
    sub_ax.set_xticks(list())
    sub_ax.set_yticks(list())
    sub_ax.set_xlabel("")
    sub_ax.set_ylim([-0.3, 2.2])
    sub_ax.legend(ncol=ncol, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.9))

    return sub_ax


def plot_linear_track_weights(
    Pyrs,
    Pyrs_weights,
    Pyrs_weights_t,
    sub_ax=None,
    target_Pyr_idx=0,
    t_start=None,
    t_end=None,
):
    """
    plot_linear_track_weights(Pyrs, Pyrs_weights, Pyrs_weights_t)

    Plots the weights from place cells to pyramidal neurons for a linear track
    environment.

    Args:
    - Pyrs (two_comp_neurons.TwoComp): Pyr. neurons.
    - Pyrs_weights (2D np.ndarray): Weights from place cells to pyramidal neurons.
    - Pyrs_weights_t (1D np.ndarray): Timepoints for the weights.
    - sub_ax (mpl_axes.Axes, optional): Subplot to plot on. If None, a new figure and
        subplot are created. Default is None.
    - target_Pyr_idx (int, optional): Index of the target pyramidal neuron for which to
        plot weights. Default is 0.
    - t_start (float, optional): Start timepoint for the plot. Default is None.
    - t_end (float, optional): End timepoint for the plot. Default is None.

    Returns:
    - sub_ax (mpl_axes.Axes): Subplot with the weights plot.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=[5.5, 1.0])

    PCs = Pyrs.ProximalCompartment.inputs["PCs"]["layer"]

    t_start, t_end = Pyrs.Agent.get_t_start_end(t_start=t_start, t_end=t_end)

    # Plot place cell input weights to Pyr.
    plot_fcts.plot_recorded_1D_PFs(
        np.asarray(Pyrs_weights)[:, target_Pyr_idx],
        PCs.place_cell_centers,
        PFs_t=Pyrs_weights_t,
        color=PCs.color,
        sub_ax=sub_ax,
        t_start=t_start,
        t_end=t_end,
        plot_last_width=(Pyrs.n == 1),
    )

    target_Pyr_idx_str = get_target_Pyr_idx_str(target_Pyr_idx, Pyrs_n=Pyrs.n)

    current, previous = Pyrs.Agent.get_target_positions(t_end=t_end, unique=True)
    if current is not None:
        sub_ax.axvline(current, ls="dotted", color="k")
    for target_position in previous:
        sub_ax.axvline(target_position, ls="dotted", color="k", alpha=0.5)

    sub_ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    sub_ax.set_xticks(list())
    sub_ax.set_yticks(list())
    sub_ax.set_xlabel("")
    sub_ax.set_ylabel("")
    sub_ax.set_title(f"Weights{target_Pyr_idx_str}", y=1.04)
    plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.2)

    return sub_ax


def plot_rate_timeseries(
    Pyrs,
    chosen_neurons="all",
    ax1D=None,
    t_start=None,
    t_end=None,
    shared_y_axis=True,
    fixed_y_axis=None,
    plot_positions=["target"],
    reached_only=False,
    Pyr_kwargs=dict(),
):
    """
    plot_rate_timeseries(Pyrs)

    Plots the rate time series for the experiment.

    Args:
    - Pyrs (two_comp_neurons.TwoComp): Pyr. neurons.
    - chosen_neurons (list or str, optional): List of indices of the chosen Pyr.
        neurons for which to plot timeseries. Default is "all".
    - ax1D (2D np.ndarray, optional): Array of subplots. Default is None.
    - t_start (float, optional): Start timepoint for the plot. Default is None.
    - t_end (float, optional): End timepoint for the plot. Default is None.
    - shared_y_axis (bool, optional): Whether to share y axis across Pyr. time series
        subplots. Default is True.
    - fixed_y_axis (tuple, optional): Tuple of (ymin, ymax) to set the y axis limits
        for all Pyr. time series subplots. Default is None.
    - plot_positions (list, optional): List of positions to plot as icons.
        Default is ["target"].
    - reached_only (bool, optional): Whether to plot only positions visited when they
        were targets. Default is False.
    - Pyr_kwargs (dict, optional): Keyword arguments for plotting the Pyr. neurons.
        Default is dict().

    Returns:
    - ax1D (2D np.ndarray): Array of subplots with the time series plot.
    """

    if ax1D is None:
        hei_each = get_linear_track_rate_timeseries_height(n=Pyrs.n)
        height = hei_each * 3 + 0.4 * 2
        _, ax1D = plt.subplots(3, 1, figsize=[5.5, height])

    ax1D = ax1D.ravel()

    if Pyrs.n == 1:
        norm_by = 1
    else:
        if fixed_y_axis:
            max_fr = Pyrs.get_min_max_firingrates()[1]
        else:
            max_fr = Pyrs.get_min_max_firingrates(t_start=t_start, t_end=t_end)[1]
        norm_by = max_fr / 3

    Pyrs.plot_rate_timeseries(
        t_start=t_start,
        t_end=t_end,
        ax=ax1D,
        adjust_xlim=True,
        single_x_axis=True,
        chosen_neurons=chosen_neurons,
        omit_target_reset=True,
        norm_by=norm_by,
        autosave=False,
        separate_axes=True,
        **Pyr_kwargs,
    )

    # move the titles up a bit
    y = 0.98 + 0.1 * (1 / Pyrs.n)
    for sub_ax in ax1D:
        sub_ax.set_title(sub_ax.get_title(), y=y)

    pad_prop = 0.06 + 0.09 * (1 / Pyrs.n)
    plot_util.pad_axis(ax1D[0], axis="y", pad_prop=pad_prop, prop_high=1.0)
    if fixed_y_axis is not None:
        for sub_ax in ax1D:
            ylims = sub_ax.get_ylim()
            ymin = min(ylims[0], fixed_y_axis[0])
            ymax = max(ylims[1], fixed_y_axis[1] / 2)  # only expand partially here
            sub_ax.set_ylim([ymin, ymax])

    for i, sub_ax in enumerate(ax1D):
        if i == 0:
            use_plot_positions = plot_positions
            plot_teleportation = True
        else:
            use_plot_positions = list()
            plot_teleportation = False

        plot_fcts.add_timeseries_markers(
            Pyrs.Agent,
            sub_ax,
            t_start=t_start,
            t_end=t_end,
            plot_teleportation=plot_teleportation,
            plot_positions=use_plot_positions,
            reached_only=reached_only,
            plot_reset_lines=True,
        )

    t = Pyrs.ProximalCompartment.get_plotting_times(t_start, t_end)[0]
    actual_t_end = t[-1] if len(t) else None
    fix_xlims_and_ticks(
        ax1D,
        t_start,
        t_end,
        actual_t_end=actual_t_end,
        dt=Pyrs.Agent.dt,
        convert_to_min=True,
        ticks_last_only=True,
    )

    for sub_ax in ax1D:
        sub_ax.set_ylabel("")

    if fixed_y_axis is not None:
        for sub_ax in ax1D:
            sub_ax.set_ylim(fixed_y_axis)

    elif shared_y_axis:
        plot_util.match_y_axis_scales(ax1D, match_ymins=True)

    return ax1D


def plot_linear_track(
    Pyrs,
    Pyrs_weights,
    Pyrs_weights_t,
    target_Pyr_idx=0,
    axes=None,
    t_start=None,
    t_end=None,
    shared_y_axis=True,
    fixed_y_axis=None,
    addendum=None,
    speed_up=None,
    plot_positions=["landmark"],
    reached_only=False,
    Pyr_kwargs=dict(),
):
    """
    plot_linear_track(Pyrs, Pyrs_weights, Pyrs_weights_t)

    Plots linear track experiment for a specific timepoint. The plot consists of
    the following subplots:
        (1) 1D environment with agent's trajectory and landmark position.
        (2) Place cell input weights to first Pyr. proximal compartment.
        (3) (Blank subplot.)
        (3) Time series of Pyr. proximal compartment.
        (4) Time series of Pyr. distal compartment.
        (5) Time series of Pyr. inhibitory interneuron.

    Args:
    - Pyrs (two_comp_neurons.TwoComp): Pyr. neurons.
    - Pyrs_weights (list): List of input weights from place cells to Pyr. proximal
        compartments, across time, where input weights have shape (n_Pyrs, n_PCs).
    - Pyrs_weights_t (list): List of timepoints for the input weights from place
        cells to Pyr. proximal compartments.
    - target_Pyr_idx (int, optional): Index of the target Pyr. neuron for which to plot
        timeseries and input place cell weights. If "all", max across weights is
        plotted. Default is 0.
    - axes (2D np.ndarray, optional): Array of 6 subplots. Default is None.
    - t_start (float, optional): Start timepoint for the plot. Default is None.
    - t_end (float, optional): End timepoint for the plot. Default is None.
    - shared_y_axis (bool, optional): Whether to share y axis across Pyr. time series
        subplots. Default is True.
    - fixed_y_axis (tuple, optional): Tuple of (ymin, ymax) to set the y axis limits
        for all Pyr. time series subplots. If None, y axis limits are determined
        automatically. Default is None.
    - addendum (str, optional): Addendum for the title. Default is None.
    - speed_up (int, optional): Speed up factor to be included in title.
        Default is None.
    - plot_positions (list or dict, optional): List of positions or position names to
        plot markers for or dictionary with both. Default is ["landmark"].
    - reached_only (bool, optional): Whether to plot only positions visited when they
        were targets in the timeseries. Default is False.
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

    suptitle, y = get_suptitle("linear", addendum=addendum, speed_up=speed_up)
    fig.suptitle(suptitle, fontweight="bold", fontsize=13, y=y)

    t_end = t_end or Pyrs.Agent.get_t_history()[-1]

    plot_linear_track_weights(
        Pyrs,
        Pyrs_weights,
        Pyrs_weights_t,
        sub_ax=ax1D[0],
        target_Pyr_idx=target_Pyr_idx,
        t_start=t_start,
        t_end=t_end,
    )

    plot_linear_track_trajectory(
        Pyrs.Agent,
        t=t_end,
        sub_ax=ax1D[1],
    )

    # match x lims for subplots 0 and 1
    x_min = min([ax1D[i].get_xlim()[0] for i in range(2)])
    x_max = max([ax1D[i].get_xlim()[1] for i in range(2)])
    for i in range(2):
        ax1D[i].set_xlim([x_min, x_max])

    # Turn off buffer axis
    ax1D[2].axis("off")

    chosen_neurons = "all" if target_Pyr_idx == "all" else [target_Pyr_idx]
    plot_rate_timeseries(
        Pyrs,
        chosen_neurons=chosen_neurons,
        ax1D=ax1D[3:],
        t_start=t_start,
        t_end=t_end,
        Pyr_kwargs=Pyr_kwargs,
        shared_y_axis=shared_y_axis,
        fixed_y_axis=fixed_y_axis,
        plot_positions=plot_positions,
        reached_only=reached_only,
    )

    return axes


def check_teleportation_disabled(step, teleportation_disabled=list()):
    """
    check_teleportation_disabled(step)

    Checks whether teleportation ports should be omitted from plots at a given
    step.

    Args:
    - step (int): Step to check.
    - teleportation_disabled (list): List of periods during which teleportation
        ports should not be plotted. Each period should be a tuple of (start, end).

    Returns:
    - no_teleport (bool): Whether teleportation ports should be omitted from plots at
        the given timepoint.
    """

    no_teleport = False
    for start, end in teleportation_disabled:
        if np.isfinite(start) and np.isfinite(end) and step >= start and step < end:
            no_teleport = True
            break

    return no_teleport


def add_teleport(Ag, sub_ax, t=None, legend=True):
    """
    add_teleport(Ag, sub_ax)

    Adds teleportation ports to the plot if applicable, either with full alpha or with
    low alpha if they are only temporarily disabled.

    Args:
    - Ag (Resetable.Agent): Agent.
    - sub_ax (plt.Axes): Subplot to plot on.
    - t (float, optional): Timepoint for the plot. Default is None.
    - legend (bool, optional): Whether to plot a legend. Default is True.
    """

    if not hasattr(Ag, "teleportation_disabled"):
        return

    t = t or Ag.t

    if len(Ag.teleportation_disabled) and Ag.teleportation_disabled[0][0] == 0:
        no_teleport = check_teleportation_disabled(
            int(t / Ag.dt), Ag.teleportation_disabled[:1]
        )
        if no_teleport:
            return

    alpha = 0.8
    if len(Ag.teleportation_disabled) > 1:
        low_alpha_teleport = check_teleportation_disabled(
            int(t / Ag.dt), Ag.teleportation_disabled[1:]
        )
        if low_alpha_teleport:
            alpha = 0.2

    skip_object_types = list()
    for label in sub_ax.get_legend_handles_labels()[1]:
        skip_object_types.append(label.replace(" ", "_"))

    Ag.Environment.add_objects_to_plot(
        sub_ax, skip_object_types=skip_object_types, alpha=alpha, fontsize=8
    )

    if not legend:
        legend = sub_ax.get_legend()
        if legend is not None:
            legend.remove()


def plot_openfield_weights(
    Pyrs,
    Pyrs_weights,
    Pyrs_weights_t,
    t=None,
    target_Pyr_idx=0,
    s=75,
    sub_ax=None,
):
    """
    plot_openfield_weights(Pyrs, Pyrs_weights, Pyrs_weights_t)

    Plots the weights from place cells to pyramidal neurons for an open field
    environment.

    Args:
    - Pyrs (two_comp_neurons.TwoComp): Pyr. neurons.
    - Pyrs_weights (list): List of input weights from place cells to Pyr. proximal
        compartments, across time, where input weights have shape (n_Pyrs, n_PCs).
    - Pyrs_weights_t (list): List of timepoints corresponding to the weights.
    - t (float, optional): Timepoint for the plot. Default is None.
    - target_Pyr_idx (int, optional): Index of the target pyramidal neuron for which
        to plot weights. Default is 0.
    - s (int, optional): Size of the scatter points. Default is 75.
    - sub_ax (matplotlib.axes.Axes, optional): Subplot to plot the weights on.
        Default is None.

    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=[3.5, 3.5])

    # plot place cell input weights to Pyr.
    target_Pyr_idx_str = get_target_Pyr_idx_str(target_Pyr_idx, Pyrs_n=Pyrs.n)
    chosen_neurons = "all" if target_Pyr_idx == "all" else [target_Pyr_idx]

    Pyrs_weights = np.asarray(Pyrs_weights)

    if target_Pyr_idx == "all":
        target_Pyr_idx = 0
        if Pyrs.n > 1:
            Pyrs_weights = Pyrs_weights.max(axis=1, keepdims=True)

    weight_idx = np.where(np.asarray(Pyrs_weights_t) < t)[0][-1]

    # handle any reshuffling of place cells
    PCs = Pyrs.ProximalCompartment.inputs["PCs"]["layer"]

    previous_values = ([np.arange(PCs.n)] + PCs.shuffle_sorters)[:-1]

    t = t or Pyrs.Agent.t
    current_sorter, _ = get_previous_values(previous_values, PCs.shuffle_times, t_end=t)
    if current_sorter is None:
        current_sorter = PCs._current_sorter

    with TemporarilyReshufflePlaceCells(PCs, current_sorter):
        plot_fcts.plot_2D_PFs(
            Pyrs.ProximalCompartment,
            PCs_input_name="PCs",
            PFs=Pyrs_weights[weight_idx][target_Pyr_idx : target_Pyr_idx + 1],
            PF_type="weights",
            alpha=0.3,
            s=s,
            chosen_neurons=chosen_neurons,
            t_end=t,
            skip_object_types=["teleport"],
            round_dec=1,
            plot_BTSP_events=True,
            cbar_outline=True,
            no_legend=True,
            ax=sub_ax,
        )
    add_teleport(Pyrs.Agent, sub_ax, t=t, legend=False)

    sub_ax.set_ylabel("")

    if len(target_Pyr_idx_str) > 0:
        sub_ax.set_title(f"Weights{target_Pyr_idx_str}", y=1.02)

    return sub_ax


def plot_openfield(
    Pyrs,
    Pyrs_weights,
    Pyrs_weights_t,
    target_Pyr_idx=0,
    s=75,
    axes=None,
    t_start=None,
    t_end=None,
    shared_y_axis=True,
    fixed_y_axis=None,
    addendum=None,
    speed_up=None,
    plot_positions=["target"],
    reached_only=False,
    traj_kwargs=dict(),
    Pyr_kwargs=dict(),
):
    """
    plot_openfield(Pyrs, Pyrs_weights, Pyrs_weights_t)

    Plots open field experiment for a specific timepoint. The plot consists of
    the following subplots:
        (1) Agent's trajectory (top left).
        (2) Place cell input weights to Pyr. proximal compartment (top right).
        (3) Time series of Pyr. proximal compartment (next top, full width).
        (4) Time series of Pyr. distal compartment (mid, full width).
        (5) Time series of Pyr. inhibitory interneuron (bottom, full width).

    Args:
    - Pyrs (Resetable.Neuron): Pyr. neuron.
    - Pyrs_weights (list): List of input weights from place cells to Pyr. proximal
        compartment, across time, where input weights have shape (n_Pyrs, n_PCs).
    - Pyrs_weights_t (list): List of timepoints for the input weights from place
        cells to Pyr. proximal compartment.
    - target_Pyr_idx (int, optional): Index of the target Pyr. neuron for which to plot
        timeseries and input place cell weights. If "all", max across weights is
        plotted. Default is 0.
    - s (int, optional): Marker size for the input weights. Default is 75.
    - axes (2D np.ndarray, optional): Array of 5 subplots. Default is None.
    - t_start (float, optional): Start timepoint for the plot. Default is None.
    - t_end (float, optional): End timepoint for the plot. Default is None.
    - addendum (str, optional): Addendum for the title. Default is None.
    - speed_up (int, optional): Speed up factor to be included in title.
        Default is None.
    - plot_positions (list or dict, optional): List of positions or position names to
        plot markers for or dictionary with both. Default is ["target"].
    - reached_only (bool, optional): Whether to plot only positions visited when they
        were targets in the timeseries. Default is False.
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

    suptitle, y = get_suptitle("openfield", addendum=addendum, speed_up=speed_up)
    y = y - (get_openfield_num_per(Pyrs.n) - 2) * 0.003
    fig.suptitle(suptitle, fontweight="bold", fontsize=13, y=y)

    ax1D = np.asarray(axes).ravel()

    ax1D[0].set_title("Trajectories", y=1.02)
    Pyrs.Agent.plot_trajectories(
        t_start=t_start,
        t_end=t_end,
        ax=ax1D[0],
        alpha=0.3,
        decay_point_size=True,
        decay_point_timescale=1,
        plot_head_direction=True,
        plot_traj_ends=False,
        skip_object_types=["teleport"],
        autosave=False,
        **traj_kwargs,
    )

    add_teleport(Pyrs.Agent, ax1D[0], t=t_end)
    ax1D[0].legend(
        loc="center left", fontsize=8, bbox_to_anchor=(1, 0.5), frameon=False
    )

    plot_openfield_weights(
        Pyrs,
        Pyrs_weights,
        Pyrs_weights_t,
        t=t_end,
        target_Pyr_idx=target_Pyr_idx,
        s=s,
        sub_ax=ax1D[1],
    )

    chosen_neurons = "all" if target_Pyr_idx == "all" else [target_Pyr_idx]
    plot_rate_timeseries(
        Pyrs,
        ax1D=ax1D[2:],
        t_start=t_start,
        t_end=t_end,
        chosen_neurons=chosen_neurons,
        shared_y_axis=shared_y_axis,
        fixed_y_axis=fixed_y_axis,
        plot_positions=plot_positions,
        reached_only=reached_only,
        Pyr_kwargs=Pyr_kwargs,
    )

    return axes


def get_speed_up(t_start, t_end, dt, speed_up=5):
    """
    get_speed_up(t_start, t_end, dt)

    Calculate the speed up factor for the animation, alloting equal amounts of
    animation time to each speed up factor.

    Args:
    - t_start (float): Start timepoint.
    - t_end (float): End timepoint.
    - dt (float): Time step.
    - speed_up (int or list): Speed up factor or list of speed up factors.

    Returns:
    - speed_up (list): List of speed up factors.
    """

    if isinstance(speed_up, list):
        time_left = t_end - t_start
        order = np.argsort(speed_up)[::-1]  # highest to lowest

        speed_up_vals = [None for _ in speed_up]

        done = list()
        for i in order:
            speed_up_val = speed_up[i]

            speed_up_left = [val for j, val in enumerate(speed_up) if j not in done]
            prop = speed_up_val / sum(speed_up_left)

            speed_up_time = time_left * prop

            num_frames = int(speed_up_time / (dt * speed_up_val))
            speed_up_vals[i] = [speed_up_val] * num_frames

            time_left -= num_frames * dt * speed_up_val
            done.append(i)

        speed_up = list(np.concatenate(speed_up_vals))

    return speed_up


def animate(
    Pyrs,
    Pyrs_weights,
    Pyrs_weights_t,
    target_Pyr_idx=0,
    t_start=None,
    t_end=None,
    fps=8,
    speed_up=5,
    hold=0,
    fixed_y_axis=True,
    environment="linear_track",
    savename="animation",
    anim_save_types=["mp4", "gif"],
    embed_limit=None,
    progress_bar=True,
    debug_frame_idx=None,
    autosave=None,
    **kwargs,
):
    """
    animate(Pyrs, Pyrs_weights, Pyrs_weights_t)

    Animates the agent's trajectory and the activity of the Pyr. neurons over time.

    Args:
    - Pyrs (two_comp_neurons.TwoComp): Pyr. neurons.
    - Pyrs_weights (list): List of input weights from place cells to Pyr. proximal
        compartments, across time, where input weights have shape (n_Pyrs, n_PCs).
    - Pyrs_weights_t (list): List of timepoints for the input weights from place
        cells to Pyr. proximal compartments.
    - target_Pyr_idx (int, optional): Index of the target Pyr. neuron for which to plot
        timeseries and input place cell weights. If "all", max across weights is
        plotted. Default is 0.
    - t_start (float, optional): Start timepoint for the animation. Default is None.
    - t_end (float, optional): End timepoint for the animation. Default is None.
    - fps (int, optional): Frames per second. Default is 8.
    - speed_up (int, optional): Speed up factor. Default is 5.
    - hold (int, optional): Number of seconds to hold the final frame for.
        Default is 0.
    - fixed_y_axis (bool or tuple, optional): Whether to set the y axis limits for all
        Pyr. time series subplots to the same fixed limits. If a tuple of (ymin, ymax)
        is provided, these limits are used. If True, limits are determined
        automatically from the final animation frame. Default is True.
    - environment (str, optional): Environment in which the agent is moving. Default
        is "linear_track".
    - savename (str, optional): Name of the file to save the animation. Default is
        "animation".
    - anim_save_types (list, optional): List of formats to save the animation in.
        Default is ["mp4", "gif"].
    - embed_limit (int, optional): Limit for embedding the animation. Default is None.
    - progress_bar (bool, optional): Whether to show a progress bar. Default is True.
    - debug_frame_idx (int, optional): If not None, index of the frame to plot for
        debugging purposes instead of creating the animation. Default is None.
    - autosave (bool, optional): Whether to autosave the animation. If None, the
        global autosave setting for ratinabox is used. Default is None.

    Keyword Args:
    - kwargs: Additional keyword arguments passed to plotting function.

    Returns:
    - anim (matplotlib.animation.FuncAnimation): Animation object
        (or figure if debug_frame_idx is not None).
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
        - speed_up (int or list): Speed up factor or list of factors. Default is 3.
        - kwargs (dict): Dictionary of additional keyword arguments passed to plotting
            function. Default is dict().
        """

        if isinstance(speed_up, list):
            if i >= len(speed_up):
                raise ValueError(
                    f"Index i ({i}) exceeds length of speed_up list ({len(speed_up)})."
                )
            t_end = t_start + sum(speed_up[: i + 1]) * dt
            speed_up = speed_up[i]
        else:
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
                speed_up=speed_up,  # for title
                **kwargs,
            )

        if environment == "openfield":
            plot_util.remove_duplicate_handle_labels(ax1D[0])
            plt.close()

        return

    dt = 1 / fps

    t = Pyrs.Agent.get_plotting_times(t_start, t_end)[0]
    t_start, t_end = t[0], t[-1]

    speed_up = get_speed_up(t_start, t_end, dt, speed_up=speed_up)
    if isinstance(speed_up, list):
        num_frames = len(speed_up)
    else:
        num_frames = int((t_end - t_start) / (dt * speed_up)) - 1

    if num_frames == 0:
        raise ValueError(
            f"No frames for animation. Consider decreasing speed_up ({speed_up})."
        )

    if fixed_y_axis and isinstance(fixed_y_axis, bool):
        dummy_fig, dummy_axes = init_fct(n)
        animate_(
            num_frames - 1,
            dummy_axes,
            dt,
            t_start=t_start,
            speed_up=speed_up,
            kwargs=kwargs,
        )
        ylims = np.asarray([ax.get_ylim() for ax in dummy_axes.ravel()[-3:]])
        fixed_y_axis = (ylims[:, 0].min(), ylims[:, 1].max())
        plt.close(dummy_fig)

    kwargs["fixed_y_axis"] = fixed_y_axis

    if debug_frame_idx is None:
        print("Creating animation...")

    frames = range(num_frames)
    if hold:
        hold_n = int(hold * fps)
        frames = np.concatenate([np.asarray(frames), np.repeat(num_frames - 1, hold_n)])

    fargs = (axes, dt, t_start, speed_up, kwargs)

    if debug_frame_idx is None:
        if progress_bar:
            frames = tqdm(frames, position=0, leave=True)

        anim = mpl_animation.FuncAnimation(
            fig,
            animate_,
            interval=1000 * dt,
            frames=frames,
            blit=False,
            fargs=(axes, dt, t_start, speed_up, kwargs),
        )

        rutils.save_animation(
            anim, savename, anim_save_types=anim_save_types, save=autosave
        )

        time_str = gen_util.get_duration_str(start_time)

        if autosave:
            print(f"Animation took {time_str} to create.")

    else:
        if debug_frame_idx >= num_frames:
            raise ValueError(
                f"debug_frame_idx ({debug_frame_idx}) exceeds number of frames "
                f"({num_frames})."
            )
        animate_(frames[debug_frame_idx], *fargs)

        anim = fig

    return anim
