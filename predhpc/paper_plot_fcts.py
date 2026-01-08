#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpl_patches
from matplotlib import colors as mpl_colors
from matplotlib import cm as mpl_cm
import scipy.stats
import ratinabox

from predhpc import plot_fcts
from predhpc.experiments import metrics
from predhpc.util import gen_util, params_util, plot_util, ext_util

LW = 1.6

BTSP_ASTERISK = (6, 2, 0)  # asterisk
BTSP_S = 35  # size of BTSP markers


def stylize_plots_for_paper(fs=12, lw=1.7, tick_length=6, notebook=False):
    """
    stylize_plots_for_paper()

    Stylize plots for paper.

    Args:
    - fs (int): Font size for the plots. Default is 12.
    - lw (float): Line width for the plots. Default is 1.7.
    - tick_length (int): Length of the ticks in the plots. Default is 6
    - notebook (bool): Whether to stylize for a Jupyter notebook. Default is False.
    """

    ratinabox.stylize_plots()
    if notebook:
        plot_util.stylize_plots_for_notebook()

    from matplotlib import rcParams as mpl_rcParams

    rcParams = {
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "axes.linewidth": lw,
        "xtick.labelsize": fs - 1,
        "xtick.major.size": tick_length,
        "xtick.major.width": lw,
        "ytick.labelsize": fs - 1,
        "ytick.major.size": tick_length,
        "ytick.major.width": lw,
    }
    mpl_rcParams.update(rcParams)
    plot_util.set_plot_font()


def initialize_paper_parameters(gen_dir=".", notebook=False):
    """
    initialize_paper_parameters()

    Initializes parameters for paper.

    Args:
    - gen_dir (str): Directory for generated plots. Default is ".".
    - notebook (bool): Whether to stylize for a Jupyter notebook. Default is False.
    """

    ratinabox.autosave_plots = False
    ratinabox.figure_directory = str(Path(gen_dir, "results", "paper"))
    stylize_plots_for_paper(notebook=notebook)


def get_somatic_compartment(Pyrs):
    """
    get_somatic_compartment(Pyrs)

    Get the somatic compartment of the Pyr object.

    Args:
    - Pyrs (Pyr): Pyramidal neuron layer object.

    Returns:
    - Pyrs (SomaticCompartment): Somatic compartment of the Pyr object.
    """

    if gen_util.attribute_type_checker(Pyrs, "TwoCompLayer"):
        Pyrs = Pyrs.SomaticCompartment
    return Pyrs


def get_PF_label(PF_type="history", title=False):
    """
    get_PF_label(PF_type)

    Get the label for the place field type.

    Args:
    - PF_type (str, optional): Type of place field. Default is "history".
    - title (bool, optional): Whether the label is for a title. Default is False.

    Returns:
    - label (str): Label for the place field type.
    """

    if PF_type == "history":
        label = "Neural activity"
    else:
        label = plot_fcts.get_PF_label(PF_type, title=title)

    return label


def get_learner_remap_step(learner, idx=0, num_total=None):
    """
    get_learner_remap_step(learner)

    Retrieves a specific remap step from the learner object.

    Args:
    - learner (Learner): Learner object containing remap steps.
    - idx (int, optional): Index of the remap step to retrieve. Default is 0.
    - num_total (int, optional): Expected total number of remap steps recorded by the
        learner. Default is None.

    Returns:
    - remap_step: The remap step at the specified index.
    """

    remap_steps = getattr(learner, "remap_steps", list())

    if num_total is not None and len(remap_steps) != num_total:
        raise RuntimeError(
            f"Expected the learner to have exactly {num_total} remap step(s), "
            f"but found {len(remap_steps)}."
        )

    if idx >= len(remap_steps):
        raise ValueError(
            "idx must be less than number of remap steps recorded "
            f"({len(remap_steps)}), but got idx={idx}."
        )

    remap_step = remap_steps[idx]

    return remap_step


def format_1D_PF_xaxis(
    sub_ax, scale=params_util.SCALE_LINEAR, num_ticks=7, PF_type="history"
):
    """
    format_1D_PF_xaxis(sub_ax)

    Formats the x-axis of a 1D place field plot.

    Args:
    - sub_ax (matplotlib.axes.Axes): The axes to format.
    - scale (float, optional): The scale of the environment.
        Default is params_util.SCALE_LINEAR.
    - num_ticks (int, optional): The number of ticks to display on the x-axis.
        Default is 7.
    - PF_type (str, optional): If "weights", the y-axis label is set to
        "Input weight". If "history", it is set to "Neural activity".
        Default is "history".
    """

    sub_ax.set_xlim([0, scale])
    sub_ax.set_xticks([0, scale])
    sub_ax.spines[["top", "right"]].set_visible(False)
    plot_util.expand_ticks(
        sub_ax, axis="x", num_ticks=num_ticks, alternating=True, round_dec=0
    )

    if "weights" in PF_type:
        xlabel = "Input place field center (m)"
    else:
        xlabel = "Position (m)"

    sub_ax.set_xlabel(xlabel)

    ylabel = get_PF_label(PF_type, title=False)

    sub_ax.set_ylabel(ylabel)


def mark_1D_target(sub_ax, Ag, target_shift=0, alpha=0.8):
    """
    mark_1D_target(sub_ax, Ag)

    Adds vertical dashed line for target position on the 1D track to a subplot.

    Args:
    - sub_ax (plt.Axes): Subplot on which to add positions.
    - Ag (Agent, optional): Agent object to plot.
    - target_shift (float, optional): Shift to apply to the target position.
        Default is 0.
    - alpha (float, optional): Alpha value for the vertical line. Default is 0.8.
    """

    target_pos = Ag.get_position("target", dim_idx=0)
    positions = [target_pos]
    if target_shift != 0:
        positions.append(target_pos + target_shift)

    for i, pos in enumerate(positions):
        use_alpha = alpha if i == len(positions) - 1 else alpha / 2
        sub_ax.axvline(pos, ls="dotted", color="k", alpha=use_alpha, zorder=-5, lw=LW)


def add_1D_position_markers(
    sub_ax,
    Ag,
    y_1D=1.0,
    base_s=30,
    pos_shift=0,
    target_shift=0,
    target_alpha=1.0,
    **kwargs,
):
    """
    add_1D_position_markers(sub_ax, Ag)

    Adds markers for positions along the 1D track to a subplot.

    Args:
    - sub_ax (plt.Axes): Subplot on which to add positions.
    - Ag (Agent, optional): Agent object to plot.
    - y_1D (float, optional): Y-coordinate for the 1D positions. Default is 1.0.
    - base_s (float, optional): Base size for the 1D positions. Default is 30.
    - pos_shift (float, optional): Shift to apply to the position. Default is 0.
    - target_shift (float, optional): Shift to apply to the target position.
        Default is 0.
    - alpha (float, optional): Alpha value for the position markers. Default is 1.0.

    Keyword args:
    - **kwargs: Keyword arguments passed to Ag.add_positions_spatially_to_plot()
    """

    positions = ["start", "reset", "target"]
    if target_shift != 0:
        positions.append("target_shifted")

    for position in positions:
        use_alpha = target_alpha if position in ["target", "target_shifted"] else 1.0
        extra_shift = target_shift if position == "target_shifted" else 0
        if position == "target" and "target_shifted" in positions:
            use_alpha = target_alpha / 2
        position = "target" if position == "target_shifted" else position

        Ag.add_positions_spatially_to_plot(
            sub_ax,
            position_name=position,
            pos_shift=extra_shift + pos_shift,
            y_1D=y_1D,
            base_s=base_s,
            alpha=use_alpha,
            lw=LW,
            **kwargs,
        )


def add_regression_line(sub_ax, x_data, y_data, prop_x=0.2, prop_y=0.95, log=True):
    """
    add_regression_line(sub_ax, x_data, y_data)

    Adds a linear regression line to the given subplot.

    Args:
    - sub_ax (plt.Axes): Subplot to which to add the regression line.
    - x_data (1D np.ndarray): X data for the regression.
    - y_data (1D np.ndarray): Y data for the regression.
    - prop_x (float, optional): Proportion along the x-axis to place the regression
        equation text. Default is 0.2.
    - prop_y (float, optional): Proportion along the y-axis to place the regression
        equation text. Default is 0.95.
    - log (bool, optional): If True, prints R^2 and p-value of the regression.
        Default is True.

    Returns:
    - regr (LinregressResult): Result of the linear regression.
    """

    regr = scipy.stats.linregress(x_data, y_data)
    x = np.asarray(sub_ax.get_xlim())
    y = x * regr.slope + regr.intercept
    sub_ax.plot(x, y, alpha=0.6, color="k", ls="dashed", zorder=-5, lw=LW)

    regr_str = f"y = {regr.slope:.2f}x + {regr.intercept:.2f}"
    x_text = gen_util.get_proportion_edges(sub_ax.get_xlim(), prop=prop_x)
    y_text = gen_util.get_proportion_edges(sub_ax.get_ylim(), prop=prop_y)
    sub_ax.text(x_text, y_text, regr_str, fontsize=12)

    if log:
        print(f"R^2: {regr.rvalue**2:.2f}, p-value: {regr.pvalue:.4f}")

    return regr


def get_BTSP_times_and_cmap(NeuronLayer, chosen_neuron=0, BTSP_idx=0):
    """
    get_BTSP_times_and_cmap(NeuronLayer)

    Retrieves BTSP event times and colormap for the given NeuronLayer object.

    Args:
    - NeuronLayer (NeuronLayer): The NeuronLayer for which to retrieve BTSP times and
        colormap.
    - chosen_neuron (int, optional): Index of the neuron to retrieve BTSP times for.
        Default is 0.
    - BTSP_idx (int, optional): Index of the BTSP event to retrieve times for.
        Default is 0.

    Returns:
    - t_start (float): Start time for plotting.
    - t_end (float): End time for plotting.
    - cmap (ListedColormap): Colormap for BTSP events.
    """

    BTSP_times = (
        NeuronLayer.get_BTSP_steps(chosen_neurons=[chosen_neuron])
        * NeuronLayer.Agent.dt
    )
    if len(BTSP_times) == 0:
        raise RuntimeError("No BTSP events found for the neuron layer.")
    if BTSP_idx >= len(BTSP_times):
        raise ValueError(
            f"BTSP index of {BTSP_idx} to high for the number of BTSP events: "
            f"{len(BTSP_times)}."
        )
    pre, post = NeuronLayer.get_estimated_num_steps_pre_post_BTSP(as_time=True)
    t_start = max(0, BTSP_times[BTSP_idx] - pre)
    t_end = BTSP_times[BTSP_idx] + post
    cmap, _, _ = NeuronLayer.get_BTSP_kernel_based_cmap(
        t_pre=pre,
        t_post=post,
        use_alpha=True,
    )

    return t_start, t_end, cmap


def configure_neural_activity_axis(
    sub_ax, ymin=0, ymax=10, xmin=None, norm=1, right=False, label=True
):
    """
    configure_neural_activity_axis(sub_ax)

    Configures axes for neural activity plots.

    Args:
    - sub_ax (plt.Axes): Subplot for which to configure axis.
    - ymin (float, optional): Minimum value for y axis spine. Default is 0.
    - ymax (float, optional): Maximum value for y axis spine. Default is 10.
    - xmin (float, optional): Minimum value for x axis spine. Default is None.
    - norm (float, optional): Normalization factor for y axis ticks. Default is 1.
    - right (bool, optional): IF True, y axis spine and labels are moved to the right.
        Default is False.
    - label (bool, optional): If True, y axis label is added. Default is True.
    """

    sub_ax.set_yticks([ymin / norm, ymax / norm])
    if norm != 1:
        sub_ax.set_yticklabels([ymin, ymax])

    spine = "left"
    labelpad = 0
    if right:
        spine = "right"
        sub_ax.yaxis.tick_right()
        sub_ax.yaxis.set_label_position("right")
        labelpad = 6

    sub_ax.spines[spine].set_visible(True)
    sub_ax.spines[spine].set_bounds(ymin / norm, ymax / norm)

    xmin = xmin or 0
    sub_ax.spines["bottom"].set_bounds(xmin, sub_ax.get_xlim()[1])
    if label:
        sub_ax.set_ylabel("Neural activity", labelpad=labelpad)


def plot_1D_PFs(
    PF_centers,
    PFs,
    shift=0,
    scale_y=4,
    alpha=0.8,
    shade_factor=1.0,
    num_hatches=6,
    color="k",
    lw=LW,
    base=None,
    mark_original=False,
    sub_ax=None,
):
    """
    plot_1D_PFs(PF_centers, PFs)

    Plots place fields.

    Args:
    - PF_centers (1D np.ndarray): Centers of the place cells.
    - PFs (1D np.ndarray): Place fields to plot.
    - shift (float, optional): Shift to apply to the place fields. Default is 0.
    - scale_y (float, optional): Scale factor for the y-axis. Default is 4.
    - alpha (float, optional): Alpha value for the plot line. Default is 0.8.
    - shade_factor (float, optional): Factor by which to adjust the shading alphas.
        Default is 1.0.
    - num_hatches (int, optional): Number of hatches to use for shading. Default is 6.
    - color (str, optional): Color of the place cell plot lines. Default is "k".
    - lw (float, optional): Line width for the plot line. Default is LW.
    - base (1D np.ndarray, optional): Base place fields to compare against.
    Default is None.
    - mark_original (bool, optional): Whether to mark the original place fields.
        Default is False.
    - sub_ax (plt.Axes, optional): Axes to plot on. If None, a new figure and subplot
        are created. Default is None.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted weights.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(5, 4))

    plot_PFs = PFs * scale_y + shift

    alphas = [0.3 * shade_factor, 0.5 * shade_factor]
    hatches = ["/" * num_hatches, "\\" * num_hatches]
    if base is not None:
        base = base * scale_y + shift
        diff = plot_PFs - base
        for i, fill_color in enumerate([color, "dimgray"]):
            diff = -diff if i == 1 else diff
            cond = plot_util.get_greater_condition_for_fill_between(diff)
            sub_ax.fill_between(
                PF_centers,
                base,
                plot_PFs,
                color=fill_color,
                hatch=hatches[i],
                alpha=alphas[i],
                lw=0,
                where=cond,
                rasterized=True,  # for hatches to export
            )
    if mark_original:
        sub_ax.plot(
            PF_centers,
            base,
            color=color,
            alpha=alpha * 0.7,
            lw=lw * 0.8,
            ls="dashed",
        )

    sub_ax.plot(PF_centers, plot_PFs, color=color, alpha=alpha, lw=lw)

    return sub_ax


def add_BTSP_kernel_to_timeseries(
    NeuronLayer,
    sub_ax,
    chosen_neuron=0,
    t_start=None,
    t_end=None,
    BTSP_kernel_s=60,
    BTSP_kernel_lw=1,
    plot_colorbar=True,
    y=0,
):
    """
    add_BTSP_kernel_to_timeseries(NeuronLayer, sub_ax)

    Plots BTSP kernel in a neuron rate timeseries plot.

    Args:
    - NeuronLayer (NeuronLayer): The NeuronLayer for which to plot the BTSP kernel.
    - sub_ax (plt.Axes): The subplot to plot on.
    - chosen_neuron (int, optional): The index of the neuron to plot the BTSP kernel
        for. Default is 0.
    - t_start (float, optional): Start time for the plot. Default is None.
    - t_end (float, optional): End time for the plot. Default is None.
    - BTSP_kernel_s (int, optional): Size of the BTSP kernel markers. Default is 60.
    - BTSP_kernel_lw (float, optional): Line width of the BTSP kernel markers.
        Default is 1.
    - plot_colorbar (bool, optional): Whether to plot a colorbar for the BTSP kernel.
        Default is True.
    - y (float, optional): Y-coordinate at which to plot the BTSP kernel. Default is 0.
    """

    num_lines = None
    num_BTSP = len(NeuronLayer.get_BTSP_steps())
    for BTSP_idx in range(num_BTSP):
        BTSP_t_start, BTSP_t_end, cmap = get_BTSP_times_and_cmap(
            NeuronLayer, chosen_neuron, BTSP_idx=BTSP_idx
        )
        if num_lines is None:
            num_lines = int((BTSP_t_end - BTSP_t_start) / NeuronLayer.Agent.dt)

        times = np.linspace(BTSP_t_start, BTSP_t_end, num_lines)
        mask = np.ones_like(times).astype(bool)
        if t_start is not None:
            mask[times < t_start] = False
        if t_end is not None:
            mask[times > t_end] = False

        times = times[mask]
        colors = cmap(np.linspace(0, 1, num_lines))[mask]

        sub_ax.scatter(
            times / 60,
            [y] * mask.sum(),
            color=colors,
            marker="|",
            s=BTSP_kernel_s,
            lw=BTSP_kernel_lw,
        )

    if plot_colorbar and num_BTSP:
        cmap, vmin, vmax = NeuronLayer.get_BTSP_kernel_based_cmap(for_colorbar=True)
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
        plot_util.add_colorbars(
            sub_ax,
            cbar,
            vmin=vmin,
            vmax=vmax,
            label="BTSP kernel\nstrength",
            outline=True,
            size="1.5%",
            pad=0.2,
        )


def plot_single_neuron_rate_timeseries(
    NeuronLayer,
    chosen_neuron=0,
    t_start=None,
    t_end=None,
    in_min=True,
    sub_ax=None,
    fig_size=(10, 1.7),
    num_ticks=11,
    lw=LW,
    mark_traj_idxs=None,
    plot_BTSP_kernel=True,
    BTSP_s=BTSP_S,
    plot_teleportation=True,
    plot_reward=True,
    **kwargs,
):
    """
    plot_single_neuron_rate_timeseries(NeuronLayer)

    Plots the rate timeseries of a single neuron over time.

    Args:
    - NeuronLayer (NeuronLayer): The NeuronLayer for which to plot activity.
    - chosen_neuron (int, optional): The index of the neuron to plot activity for.
        Default is 0.
    - t_start (float, optional): Start time for the plot. Default is None.
    - t_end (float, optional): End time for the plot. Default is None.
    - in_min (bool, optional): Whether to plot time in minutes. Default is True.
    - sub_ax (plt.Axes, optional): The subplot to plot on. Default is None.
    - fig_size (tuple, optional): Size of the figure if sub_ax is None.
        Default is (10, 1.7).
    - num_ticks (int, optional): Number of ticks on the x-axis. Default is 11.
    - lw (float, optional): Line width for the plot. Default is LW.
    - mark_traj_idxs (list, optional): List of trajectory indices to mark on the plot.
        Default is None.
    - plot_BTSP_kernel (bool, optional): Whether to mark the BTSP kernel on the plot.
        Default is True.
    - BTSP_s (int, optional): Size of the BTSP markers. Default is BTSP_S.
    - plot_teleportation (bool, optional): Whether to plot teleportation markers.
        Default is True.
    - plot_reward (bool, optional): Whether to plot reward position markers.
        Default is True.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to
        add_BTSP_kernel_to_timeseries().

    Returns:
    - sub_ax (plt.Axes): The axes with the plotted neuron activity.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=fig_size)

    NeuronLayer.plot_rate_timeseries(
        sub_ax=sub_ax,
        norm_by="none",
        plot_BTSP_events=False,
        t_start=t_start,
        t_end=t_end,
        lw=lw,
        chosen_neurons=[chosen_neuron],
    )

    if plot_BTSP_kernel:
        ylow = -1.0
    elif mark_traj_idxs:
        ylow = -0.3
    else:
        ylow = -0.2

    ymin = min(sub_ax.get_ylim()[0], ylow) + ylow
    ymax = max(sub_ax.get_ylim()[1], 11)
    sub_ax.set_ylim(ymin, ymax)

    sub_ax.set_ylabel("")
    xticks = plot_util.expand_ticks(
        sub_ax, axis="x", num_ticks=num_ticks, alternating=True, round_dec=1
    )

    configure_neural_activity_axis(sub_ax, xmin=xticks[0])
    plot_util.pad_axis(sub_ax, axis="x", pad_prop=0.015, prop_high=0)

    if not in_min:
        xticks = np.around(sub_ax.get_xticks() * 60, 0) / 60
        sub_ax.set_xticks(xticks)
        xticklabels = [f"{tick * 60:.0f}" for tick in xticks]
        sub_ax.set_xticklabels(xticklabels)
        sub_ax.set_xlabel("Time (s)")

    if mark_traj_idxs is not None:
        t, _, _ = NeuronLayer.Agent.get_trajectory_plotting_times(
            traj_idxs=mark_traj_idxs, t_start=t_start, t_end=t_end
        )
        sub_ax.plot(t[[0, -1]] / 60, [ymin / 2] * 2, color="k", lw=2, alpha=0.8)

    if gen_util.attribute_type_checker(NeuronLayer, "BTSPLayer"):
        sub_ax.set_ylim(None, max(sub_ax.get_ylim()[1], ymax * 1.3))
        NeuronLayer.add_BTSP_markers_to_plots(
            ax=sub_ax,
            s=BTSP_s,
            prop_y=0.8,
            marker=BTSP_ASTERISK,
            timeseries=True,
            t_start=t_start,
            t_end=t_end,
            lw=lw,
            chosen_neurons=[chosen_neuron],
        )

    if plot_reward:
        sub_ax.set_ylim(None, max(sub_ax.get_ylim()[1], ymax * 1.3))
        NeuronLayer.Agent.add_position_across_time_to_plot(
            sub_ax=sub_ax,
            position_name="reward",
            alpha=0.8,
            y=13.6,
            t_start=t_start,
            t_end=t_end,
            raise_error=False,
        )

    if plot_teleportation and len(NeuronLayer.Agent.teleportation_df) > 0:
        if plot_reward:
            plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.1, prop_high=1.0)
        else:
            sub_ax.set_ylim(None, max(sub_ax.get_ylim()[1], ymax * 1.3))
        NeuronLayer.Agent.add_teleportation_markers_to_plots(
            ax=sub_ax,
            timeseries=True,
            plot_lines=True,
            t_start=t_start,
            t_end=t_end,
            y_prop=0.96,
            lw=LW,
            no_legend=True,
        )

    if plot_BTSP_kernel:
        add_BTSP_kernel_to_timeseries(
            NeuronLayer,
            sub_ax,
            chosen_neuron=chosen_neuron,
            t_start=t_start,
            t_end=t_end,
            y=ymin / 2,
            **kwargs,
        )

    return sub_ax


def plot_linear_environment(Ag):
    """
    plot_linear_environment()

    Plots the environment for the linear experiment.

    Args:
    - Ag (Agent, optional): Agent object to plot.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted environment.
    """

    _, sub_ax = plt.subplots(figsize=(6.2, 1.0))
    sub_ax = plot_fcts.plot_1D_reset_environment(
        Ag, minimalist=True, title="", base_s=50, obj_lw=LW, sub_ax=sub_ax
    )
    sub_ax.spines["bottom"].set_linewidth(2.5)
    leg = sub_ax.get_legend()
    for text in leg.get_texts():
        text_str = text.get_text()
        add_str = "object" if text_str == "target" else "position"
        text.set_text(f"{text_str} {add_str}")
        text.set_fontsize(12)

    return sub_ax


def plot_BTSP_kernel(Pyrs, xlims=None):
    """
    plot_BTSP_kernel()

    Plots the BTSP kernel for the given Pyrs object.

    Args:
    - Pyrs (Pyr): Pyr object containing the agent and place cells.
    - xlims (list, optional): X-axis limits for the plot. Default is None.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted BTSP kernel.
    """

    _, Ag, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    _, sub_ax = plt.subplots(figsize=(6, 2.28))
    BTSP_param_dict = params_util.get_default_BTSP_filter_param_dict(
        incl_BTSP_str=False
    )

    if xlims is None:
        xlims = [-14, 10]

    sub_ax = plot_util.plot_summed_exp_kernel(
        dt=Ag.dt,
        color=PCs.color,
        xlims=xlims,
        minimalist=True,
        lw=LW,
        sub_ax=sub_ax,
        **BTSP_param_dict,
    )[0]

    xticks = np.arange(xlims[0], xlims[1] + 2, 2)
    plot_util.set_alternating_ticks(sub_ax, xticks, round_dec=0)
    plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.15, prop_high=0.9)
    sub_ax.set_xlabel("Time relative to BTSP event (s)")

    return sub_ax


def plot_normalization_values(
    Pyrs, sub_ax=None, skip_initial=False, fig_width=6, shift_time=None
):
    """
    plot_normalization_values(Pyrs)

    Plots the normalization values for the given Pyrs object.

    Args:
    - Pyrs (Pyr): Pyr object containing the agent and place cells.
    - sub_ax (plt.Axes, optional): The subplot to plot on. If None, a new figure
        and subplot are created. Default is None.
    - skip_initial (bool, optional): Whether to skip initial normalization values.
        Default is False.
    - fig_width (float, optional): Width of the figure if sub_ax is None. Default is 6.
    - shift_time (float, optional): Time as of which to shift the normalization values.
        Default is None.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(fig_width, 2))

    Pyrs = get_somatic_compartment(Pyrs)

    if Pyrs.n == 1:
        by_neuron = False
        plot_BTSP_events = True
    else:
        by_neuron = True
        plot_BTSP_events = False

    Pyrs.plot_normalization_values(
        "PCs",
        skip_initial=skip_initial,
        shift_time=shift_time,
        by_neuron=by_neuron,
        plot_BTSP_events=plot_BTSP_events,
        sub_ax=sub_ax,
    )

    # hacky: shift from neuron indices to neuron number
    xticks, xtick_labels = list(), list()
    for xtick, xtick_label in zip(sub_ax.get_xticks(), sub_ax.get_xticklabels()):
        if xtick >= sub_ax.get_xlim()[0] and xtick <= sub_ax.get_xlim()[1] + 1:
            xticks.append(xtick - 1)
            xtick_labels.append(xtick_label.get_text())

    sub_ax.set_xticks(xticks)
    sub_ax.set_xticklabels(xtick_labels)

    sub_ax.set_xlabel("Pyramidal neuron number")
    sub_ax.set_ylabel("Norm. value")
    sub_ax.set_title("")

    return sub_ax


def plot_BTSP_ramp(Pyrs, sub_ax=None):
    """
    plot_BTSP_ramp(Pyrs)

    Plots the BTSP ramp for the given Pyrs object.

    Args:
    - Pyrs (Pyr): Pyr object containing the agent and place cells.
    - sub_ax (plt.Axes, optional): The subplot to plot on. If None, a new figure
        and subplot are created. Default is None.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted BTSP ramp.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(8.8, 1.3))

    Pyrs = get_somatic_compartment(Pyrs)

    if Pyrs.n > 1:
        raise NotImplementedError(
            "BTSP ramp plotting only implemented for single neuron layers."
        )

    Pyrs.plot_BTSP_ramp(
        sub_ax=sub_ax,
        lw=LW,
        mark_threshold=False,
        plot_BTSP_events=True,
        s=BTSP_S,
        marker=BTSP_ASTERISK,
    )
    xmax = Pyrs.Agent.t / 60
    sub_ax.plot([0, xmax], [1, 1], ls="dotted", lw=LW, color=Pyrs.color, alpha=1.0)

    plot_fcts.mark_target_and_reset_points(Pyrs, sub_ax=sub_ax, lw=LW, alpha=0.5)
    sub_ax.set_title("Proportion of BTSP criterion reached", y=1.15)

    sub_ax.set_ylabel("Prop.")
    max_val = max(1, int(sub_ax.get_ylim()[1]))
    sub_ax.set_yticks([0, max_val])
    sub_ax.spines["left"].set_bounds(0, max_val)
    plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.1, prop_high=0.5)

    sub_ax.set_xlabel("")
    sub_ax.spines["bottom"].set_visible(False)
    sub_ax.tick_params(axis="x", bottom=False)

    return sub_ax


def plot_BTSP_responses(Pyrs, sub_ax=None, **kwargs):
    """
    plot_BTSP_responses(Pyrs)

    Plots the BTSP responses for the given Pyrs object.

    Args:
    - Pyrs (Pyr): Pyr object containing the agent and place cells.
    - sub_ax (plt.Axes, optional): The subplot to plot on. If None, a new figure
        and subplot are created. Default is None.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to Pyrs.plot_BTSP_responses().

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted BTSP responses.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(3, 2))

    Pyrs = get_somatic_compartment(Pyrs)

    Pyrs.plot_BTSP_responses(split=False, fill=False, post=1, ax=sub_ax, **kwargs)
    sub_ax.set_title("")
    sub_ax.set_ylabel("Neural activity")

    return sub_ax


def plot_BTSP_counts_vs_target_visits(
    Pyrs, t_start=0, hline=None, xmin=None, plot_regression=False, height=1.8
):
    """
    plot_BTSP_counts_vs_target_visits(Pyrs)

    Plots the number of BTSP events versus the number of object visits.

    Args:
    - Pyrs (Pyr): Pyr object.
    - t_start (float, optional): Start time for the plot. Default is 0.
    - hline (float, optional): Y-value at which to draw a horizontal line.
        Default is None.
    - xmin (float, optional): Minimum x-value for the plot. Default is None.
    - plot_regression (bool, optional): Whether to plot a regression line.
        Default is False.
    - height (float, optional): Height of the figure. Default is 1.8.

    Returns:
    - sub_ax (plt.Axes): Subplot with the plotted BTSP counts versus target visits.
    """

    _, sub_ax = plt.subplots(figsize=(2.5, height))

    num_BTSP = Pyrs.SomaticCompartment.get_BTSP_counts(t_start=t_start).max()

    max_spread = max(0.1, min(0.3, 0.05 * num_BTSP))

    Pyrs.plot_BTSP_counts_vs_target_visits(
        sub_ax=sub_ax,
        t_start=t_start,
        alpha=0.8,
        max_spread=max_spread,
        spread_bin_width_prop=0.1,
        hline=hline,
        xmin=xmin,
    )
    sub_ax.set_title("")
    sub_ax.set_xlabel("Object visits")
    sub_ax.set_ylabel("BTSP events")

    if plot_regression:
        nbr_visits = Pyrs.get_nbr_visits_per_target()
        BTSP_counts = Pyrs.SomaticCompartment.get_BTSP_counts(t_start=t_start)
        add_regression_line(sub_ax, nbr_visits, BTSP_counts, prop_x=0.1, prop_y=0.8)

    return sub_ax


def plot_linear_neural_activity(learner):
    """
    plot_linear_neural_activity()

    Plots neural activity for linear experiment.

    Args:
    - learner (Learner): Learner object.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with neural activity plotted.
    """

    _, ax1D = plt.subplots(
        nrows=4,
        figsize=(10, 3.2),
        gridspec_kw={"hspace": 0.7},
        sharex=True,
        squeeze=True,
    )

    learner.Pyrs.plot_rate_timeseries(
        ax=ax1D[1:],
        chosen_neurons="all",
        shift=-10,
        overlap=1,
        lw=LW,
        BTSP_s=BTSP_S,
        BTSP_lw=LW,
        BTSP_marker=BTSP_ASTERISK,
        separate_axes=True,
        norm_by="none",
        no_legend=True,
        autosave=False,
    )
    ax1D[1].set_title("Pyramidal neuron", y=1.15)
    ax1D[2].set_title("")
    ax1D[3].set_title("Inhibitory interneuron", y=1.15)

    plot_util.match_y_axis_scales(ax1D[1:])

    plot_util.expand_ticks(
        ax1D[-1], axis="x", num_ticks=7, alternating=True, round_dec=1
    )
    plot_util.pad_axis(ax1D[0], axis="x", pad_prop=0.02, prop_high=0)
    ax1D[0].set_xlim([None, ax1D[0].get_xticks()[-1]])

    for i, comp in enumerate(["somatic", "apical"]):
        learner.Pyrs.add_compartment_legend(
            ax1D[i + 1],
            compartment=comp,
            lw=LW,
            loc=(0.725, 0.65),
            handlelength=0.8,
            handletextpad=0.5,
            frameon=False,
            fontsize=11,
        )

    # mark axes for neural activity
    for i, sub_ax in enumerate(ax1D[1:]):
        label = True if i == 1 else False
        sub_ax.set_ylabel("")
        configure_neural_activity_axis(sub_ax, label=label)

    plot_BTSP_ramp(learner.Pyrs, sub_ax=ax1D[0])

    return ax1D


def plot_linear_summary(learner):
    """
    plot_linear_summary()

    Plots summary of linear experiment.

    Args:
    - learner (Learner): Learner object.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with linear data plotted.
    """

    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(learner.Pyrs)

    plot_kwargs = {
        "hspace": 0.38,
        "height_ratios": [0.1, 0.1, 0.45, 0.13, 0.13, 0.13],
        "figsize": (5.8, 8.3),
        "lw": LW,
        "s": 1.2,
        "base_obj_s": 25,
    }

    Pyr_kwargs = {
        "BTSP_s": BTSP_S,
        "BTSP_lw": LW,
        "BTSP_marker": BTSP_ASTERISK,
        "separate_axes": True,
        "no_legend": True,
    }

    axes = plot_fcts.plot_1D_time_info(
        learner.Pyrs, Pyrs_spikes=False, Pyr_kwargs=Pyr_kwargs, **plot_kwargs
    )
    ax1D = axes[:, 0]

    ax1D[0].set_yticks([0, 3, 6])
    ax1D[0].spines["left"].set_bounds(0, 6)
    plot_util.expand_ticks(
        ax1D[-1], axis="x", num_ticks=7, alternating=True, round_dec=1
    )
    plot_util.pad_axis(ax1D[0], axis="x", pad_prop=0.01, prop_high=0)
    ax1D[0].set_xlim([None, ax1D[0].get_xticks()[-1]])

    titles = [
        "Trajectories",
        "Object cell",
        f"Place cells ({PCs.n})",
        "Pyramidal neuron",
        "",
        "",
    ]
    for i, title in enumerate(titles):
        y = 1.02 if "Place" in title else 1.06
        ax1D[i].set_title(title, y=y)
        if i > 0:
            ax1D[i].set_ylabel("")

    for i, comp in enumerate(["somatic", "apical"]):
        learner.Pyrs.add_compartment_legend(
            ax1D[3 + i],
            compartment=comp,
            lw=plot_kwargs["lw"],
            loc=(0.725, 0.65),
            handlelength=0.8,
            handletextpad=0.5,
            frameon=False,
            fontsize=11,
        )

    ax1D[-1].set_title("Inhibitory interneuron", y=1.02)

    # mark axes for neural activity
    for i, sub_ax in enumerate(ax1D[1:]):
        label = True if i == 2 else False
        norm = PCs.n / plot_fcts.DIV_MULTI_NEURON if i == 1 else 1
        configure_neural_activity_axis(sub_ax, label=label, norm=norm)

    return ax1D


def plot_linear_place_fields(learner):
    """
    plot_linear_place_fields(learner)

    Plots the place fields and weights for the given learner object.

    Args:
    - learner (Learner): Learner object.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with the linear place fields plotted.
    """

    _, Ag, PCs, _ = ext_util.extract_objects_from_Pyrs(learner.Pyrs)

    _, ax1D = plt.subplots(
        2, 1, figsize=(3, 3.1), sharex=True, gridspec_kw={"hspace": 0.28}
    )

    for i, PF_type in enumerate(["weights", "history"]):
        if "weights" in PF_type:
            data = learner.get_recorded_weights()["weights"][:, 0]
            PF_centers = PCs.place_cell_centers
            color = PCs.color
            if PF_type == "smoothed_weights":
                data, PF_centers = metrics.get_smoothed_1D_weights(
                    data, PF_centers, PCs.widths
                )
            elif PF_type != "weights":
                raise ValueError(f"PF_type '{PF_type}' not recognized.")
        else:
            t_start, t_end = ext_util.get_times_for_each_BTSP_event(
                learner.Pyrs.SomaticCompartment, next_trajectory=True, use_nans=True
            )[-1]
            if np.isnan(t_start) or np.isnan(t_end):
                raise RuntimeError(
                    "The last BTSP event did not have valid start and end times for "
                    "place field evaluation."
                )
            data, PF_centers = metrics.evaluate_PFs(
                learner.Pyrs, method="history", t_start=t_start, t_end=t_end
            )
            color = learner.Pyrs.SomaticCompartment.color

        plot_fcts.plot_recorded_1D_PFs(
            data,
            PF_centers,
            color=color,
            marker="none",
            lw=LW,
            plot_last_width=False,
            sub_ax=ax1D[i],
        )

        if PF_type == "history":
            ymin, ymax = ax1D[i].get_ylim()
            ax1D[i].set_ylim(min(0, ymin), max(6, ymax))
            learner.Pyrs.SomaticCompartment.add_BTSP_markers_to_plots(
                ax=ax1D[i],
                s=BTSP_S,
                marker=BTSP_ASTERISK,
                lw=1.5,
                prop_y=0.95,
            )

        format_1D_PF_xaxis(ax1D[i], PF_type=PF_type)
        mark_1D_target(ax1D[i], Ag=Ag)

        plot_util.expand_ticks(
            ax1D[i], axis="y", num_ticks=5, alternating=True, round_dec=1
        )
    add_1D_position_markers(ax1D[0], Ag=Ag, y_1D=0.2)

    ax1D[0].set_xlabel("")
    ax1D[0].xaxis.set_tick_params(bottom=False)
    ax1D[0].spines["bottom"].set_visible(False)

    ax1D[1].set_xlabel("Position (m)")
    ax1D[1].set_ylabel("Neural activity")

    return ax1D


def plot_linear_binned_rates(learner, num_bins=150):
    """position
    plot_linear_binned_rates(learner)

    Plots binned rates for linear experiment.

    Args:
    - learner (Learner): Learner object.
    - num_bins (int, optional): Number of bins to use for the histogram. Default is 150.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with linear binned rates plotted.
    """

    _, ax1D = plt.subplots(3, 1, figsize=(3.7, 4.7), squeeze=True)

    kwargs = {
        "num_bins": num_bins,
        "vmin": 0,
        "vmax": 10,
        "cbar_aspect": 10,
        "plot_occ": False,
        "mark_runs": True,
        "shared_range": True,
        "cbar_label": "Neural activity",
    }

    learner.Pyrs.plot_binned_rates(axes=ax1D.reshape(-1, 1), **kwargs)
    env_scale = learner.Pyrs.Environment.scale
    for sub_ax in ax1D:
        add_1D_position_markers(
            sub_ax,
            Ag=learner.Pyrs.Agent,
            y_1D=-0.4,
            pos_factor=num_bins / env_scale,
            pos_shift=-0.5,
        )

    labels = ["Somatic", "Apical", "Inhibitory"]
    for i, sub_ax in enumerate(ax1D):
        sub_ax.set_title("")
        sub_ax.set_ylabel(labels[i])

    # Add position xaxis
    ax1D[-1].set_xlabel("")
    pos_sub_ax = ax1D[-1].twiny()
    pos_sub_ax.set_xlim(0, env_scale)
    pos_sub_ax.spines[["top", "right", "left"]].set_visible(False)
    pos_sub_ax.xaxis.set_ticks_position("bottom")
    pos_sub_ax.xaxis.set_label_position("bottom")
    pos_sub_ax.set_xlabel(f"Position (m)")  # , labelpad=8)

    # ax1D[-1].set_xlabel(f"Spatial bin ({num_bins})", labelpad=12)

    return ax1D


def retrieve_PF_data(data_dict, PF_type="history", width=True):
    """
    retrieve_PF_data(data_dict)

    Retrieves place field data from the given data dictionary.

    Args:
    - data_dict (dict): Dictionary containing place field data.
    - PF_type (str, optional): PF evaluation method to retrieve. Default is "history".
    - width (bool, optional): Whether to retrieve place field widths. Default is True.

    Returns:
    - PFs (2D np.ndarray): Place fields.
    - PF_centers (1D np.ndarray): Centers of the place fields.
    if width:
    - PF_widths (1D np.ndarray): Widths of the last recorded place fields.
    """

    if PF_type == "weights":
        data_key = "PC_weights"
        center_key = "PC_place_centers"
        width_key = "PC_weight_widths"
    elif PF_type == "smoothed_weights":
        data_key = "PC_smoothed_weights"
        center_key = "PC_place_centers"
        width_key = "PC_smoothed_weight_widths"
    elif PF_type == "history":
        data_key = "PFs"
        center_key = "PF_centers"
        width_key = "PF_widths"
    else:
        raise ValueError(f"PF_type '{PF_type}' not recognized.")

    for key in [data_key, center_key, width_key]:
        if key not in data_dict:
            if key == width_key and not width:
                continue
            raise ValueError(f"Key '{key}' not found in data_dict.")

    PFs = data_dict[data_key]
    PF_centers = data_dict[center_key]

    if width:
        PF_widths = data_dict[width_key]
        return PFs, PF_centers, PF_widths

    else:
        return PFs, PF_centers


def plot_linear_speed_PF_examples(
    speed_data, Ag=None, color=None, PF_type="history", k=1, show_unsmoothed=True
):
    """
    plot_linear_speed_PF_examples()

    Plots examples of place fields for different speeds on the linear track. If k is
    not 1, smoothed signal is visualized.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see paper.run_linear_speeds()).
    - Ag (Agent, optional): Agent object. If provided, it is used to add markers to
        subplot. Default is None.
    - color (str, optional): Color for PF lines. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - k (int, optional): Smoothing factor for visualizing the place fields. Default is 1.
    - show_unsmoothed (bool, optional): If True, the unsmoothed place fields are also
        plotted. Default is True.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with example place fields plotted.
    """

    num_examples = len(speed_data["speed_means"])

    _, axes = plt.subplots(
        num_examples,
        1,
        figsize=(5.7, 1.3 * num_examples),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.23},
        squeeze=False,
    )
    ax1D = axes[:, 0]

    PFs, PF_centers, PF_widths = retrieve_PF_data(speed_data, PF_type=PF_type)
    if "weights" in PF_type:
        color = color or params_util.PC_COLOR
        ytick_max, ymax = 0.26, 0.29
        round_dec = 2
    elif PF_type == "history":
        color = color or params_util.PYR_SOMATIC_COLOR
        ytick_max, ymax = 6, 6.5
        round_dec = 0
    else:
        raise ValueError(f"PF_type '{PF_type}' not recognized.")

    show_unsmoothed = show_unsmoothed and k > 1
    for i, speed_mean in enumerate(speed_data["speed_means"]):
        sub_ax = ax1D[i]
        for j in range(int(show_unsmoothed) + 1):
            if show_unsmoothed and j == 0:
                use_k = 1
                plot_last_width = False
                ls, lw = "dashed", LW * 0.85
            else:
                use_k = k
                plot_last_width = True
                ls, lw = None, LW

            PF_idxs = np.where(np.isfinite(PFs[i]).any(axis=1))[0]
            if not len(PF_idxs):
                raise ValueError("Cannot plot PF example: PF contains only NaNs.")

            plot_fcts.plot_recorded_1D_PFs(
                PFs[i, PF_idxs[-1]].reshape(1, -1),
                PF_centers,
                color=color,
                marker="none",
                plot_last_width=plot_last_width,
                k=use_k,
                plot_smoothed=True,
                lw=lw,
                ls=ls,
                no_legend=True,
                sub_ax=sub_ax,
            )
        sub_ax.text(5, ymax * 0.83, f"{speed_mean:.2f} m/s", ha="center", fontsize=12)

        rect = mpl_patches.Rectangle(
            (4.59, ymax * 0.57), 0.8, ymax * 0.22, color=color, alpha=0.3, lw=0
        )
        sub_ax.add_patch(rect)

        width = np.around(PF_widths[i], 2)
        sub_ax.text(5, ymax * 0.62, f"{width} m", ha="center", fontsize=12)

    for i, sub_ax in enumerate(ax1D):
        sub_ax.set_ylim([0, ymax])
        sub_ax.set_yticks([0, ytick_max])
        plot_util.expand_ticks(
            sub_ax, axis="y", num_ticks=5, alternating=True, round_dec=round_dec
        )
        if Ag is not None:
            mark_1D_target(sub_ax, Ag=Ag)
        format_1D_PF_xaxis(sub_ax, PF_type=PF_type)
        if i != len(ax1D) // 2:
            sub_ax.set_ylabel("")

    for sub_ax in ax1D[:-1]:
        sub_ax.set_xlabel("")
        sub_ax.xaxis.set_tick_params(bottom=False)
        sub_ax.spines["bottom"].set_visible(False)

    if Ag is not None:
        add_1D_position_markers(ax1D[0], Ag=Ag, y_1D=ymax * 0.98)

    return ax1D


def plot_linear_speed_PF_widths(
    speed_data, mark_examples=list(), color=None, PF_type="history"
):
    """
    plot_linear_speed_PF_widths()

    Plots the relationship between speed means and place field widths for the linear
    experiment.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see paper.run_linear_speeds()).
    - mark_examples (list): List of speed means to mark. Default is an empty list.
    - color (str, optional): Color for PF lines. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".

    Returns:
    - sub_ax (plt.Axes): The subplot with the speed means and place field widths
        plotted.
    """

    _, sub_ax = plt.subplots(figsize=(3.3, 3.3))

    xtick_max = np.ceil(speed_data["speed_means"].max() * 10) / 10

    _, _, PF_widths = retrieve_PF_data(speed_data, PF_type=PF_type)
    if "weights" in PF_type:
        color = color or params_util.PC_COLOR
        ytick_min, ytick_max, num_yticks = 0.2, 0.8, 5
        start_y_idx = 0
    elif PF_type == "history":
        color = color or params_util.PYR_SOMATIC_COLOR
        ytick_min, ytick_max, num_yticks = 0.2, 0.6, 5
        start_y_idx = 0
    else:
        raise ValueError(f"PF_type '{PF_type}' not recognized.")

    num_BTSP = np.isfinite(speed_data["BTSP_times"]).sum(axis=1).astype(int)
    alphas = np.linspace(0.5, 0.8, num_BTSP.max())[num_BTSP - 1]
    sub_ax.scatter(speed_data["speed_means"], PF_widths, s=20, alpha=alphas, color="k")

    # Format axes
    for axis in ["x", "y"]:
        plot_util.pad_axis(sub_ax=sub_ax, axis=axis)
    sub_ax.spines[["top", "right"]].set_visible(False)

    sub_ax.set_xlabel("Running velocity (m/s)")
    if xtick_max < 0.4:
        xtick_min = 0.05
        num_xticks = (xtick_max - xtick_min) * 20 + 1
        start_x_idx = 1
    else:
        xtick_min = 0
        num_xticks = (xtick_max - xtick_min) * 10 + 1
        start_x_idx = 0
    sub_ax.set_xticks([xtick_min, xtick_max])
    plot_util.expand_ticks(
        sub_ax,
        axis="x",
        num_ticks=int(num_xticks),
        alternating=True,
        round_dec=2,
        start_idx=start_x_idx,
    )

    sub_ax.set_ylabel("Place field width (m)")
    sub_ax.set_yticks([ytick_min, ytick_max])
    plot_util.expand_ticks(
        sub_ax,
        axis="y",
        num_ticks=num_yticks,
        alternating=True,
        round_dec=1,
        start_idx=start_y_idx,
    )

    add_regression_line(sub_ax, speed_data["speed_means"], PF_widths)

    mark_kwargs = {
        "color": color,
        "ls": "dotted",
        "lw": LW,
        "alpha": 1.0,
        "zorder": -5,
    }

    xmin = sub_ax.get_xlim()[0]
    ymin = sub_ax.get_ylim()[0]
    for speed_mean in mark_examples:
        if speed_mean not in speed_data["speed_means"]:
            raise RuntimeError(f"{speed_mean} speed mean not found in data dictionary.")
        idx = np.where(speed_data["speed_means"] == speed_mean)[0][0]
        width = PF_widths[idx]
        sub_ax.plot(
            [speed_mean, speed_mean],
            [ymin, width],
            **mark_kwargs,
        )
        sub_ax.plot(
            [xmin, speed_mean],
            [width, width],
            **mark_kwargs,
        )

    return sub_ax


def plot_PF_cmap(
    PF_cmap_data,
    sub_ax=None,
    vmax=None,
    x_vals=None,
    y_vals=None,
    PF_type="history",
    cmap_x_corr=1.0027,
    plot_colorbar=True,
    mark_maxes=True,
    keep_yticks=False,
):
    """
    plot_PF_cmap(PF_cmap_data)

    Plots a 2D place field weight map, with each row representing a place field.

    Args:
    - PF_cmap_data (2D np.ndarray): Place fields to plot (number of place fields
        x number of inputs).
    - sub_ax (plt.Axes, optional): Subplot on which to plot the data. If None, a new
        figure and subplot are created. Default is None.
    - vmax (float, optional): Maximum value for the color scale. If None, it is set to
        the maximum value in PF_cmap_data. Default is None.
    - x_vals (1D np.ndarray, optional): X values for the place field colormap. If None,
        the x-axis is set to the range of the number of inputs. Default is None.
    - y_vals (1D np.ndarray, optional): Y values for the place field colormap. If None,
        the y-axis is set to the range of the number of place fields. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - cmap_x_corr (float, optional): Correction factor along the x axis for plotting
        scatterplot elements. Default is 1.0022.
    - plot_colorbar (bool, optional): Whether to plot a colorbar. Default is True.
    - mark_maxes (bool, optional): Whether to mark the maximum values along the x and
        y axes with horizontal and vertical lines, respectively. Default is True.
    - keep_yticks (bool, optional): Whether to keep the y-axis ticks, hiding them
        instead of removing them. Default is False.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted place field weights.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(2, 2))

    if x_vals is None:
        x_vals = np.arange(PF_cmap_data.shape[1])
        extent_x = [x_vals.min() - 0.5, x_vals.max() + 0.5]
    else:
        extent_x = plot_util.get_cmap_extent(x_vals)

    if y_vals is None:
        y_vals = np.arange(PF_cmap_data.shape[0])
        extent_y = [y_vals.min() - 0.5, y_vals.max() + 0.5]
    else:
        extent_y = plot_util.get_cmap_extent(y_vals)

    if vmax is None:
        vmax = np.nanmax(PF_cmap_data)

    im = sub_ax.imshow(
        PF_cmap_data[::-1],
        interpolation="none",
        aspect="auto",
        cmap="viridis",
        extent=[*extent_x, *extent_y],
        vmin=0,
        vmax=vmax,
    )

    if mark_maxes:
        scatter_kwargs = {"color": "k", "s": 8, "alpha": 0.6}

        if len(x_vals) > 1:
            horiz_max = plot_util.get_cmap_max(PF_cmap_data, axis=1, indexed=x_vals)
            sub_ax.scatter(
                horiz_max * cmap_x_corr, y_vals, marker="_", **scatter_kwargs
            )

        if len(y_vals) > 1:
            vert_max = plot_util.get_cmap_max(PF_cmap_data, axis=0, indexed=y_vals)
            sub_ax.scatter(x_vals * cmap_x_corr, vert_max, marker="|", **scatter_kwargs)

    sub_ax.spines[["top", "left", "right"]].set_visible(False)
    if keep_yticks:
        sub_ax.tick_params(axis="y", length=0)
    else:
        sub_ax.set_yticks([])

    if plot_colorbar:
        cbar = plt.colorbar(im, ax=sub_ax, pad=0.15, aspect=25)
        clabel = get_PF_label(PF_type, title=False)
        cbar.set_label(clabel)
        cbar.ax.yaxis.set_label_position("left")

    if "weights" in PF_type:
        xlabel = "Input place field center (m)"
    else:
        xlabel = "Position (m)"
    sub_ax.set_xlabel(xlabel)

    return sub_ax


def plot_PF_peak_shift(
    PF_data,
    sub_ax,
    initial_peak_idx=0,
    initial_peak_value=None,
    s=10,
    lw=LW,
    alpha=0.7,
    x_vals=None,
    y_vals=None,
    PF_type="history",
    keep_yticks=False,
    target_positions=None,
    **arrow_kwargs,
):
    """
    plot_PF_peak_shift(PF_data, sub_ax)

    Plots the peak shifts of place fields in a 1D environment.

    Args:
    - PF_data (2D np.ndarray): Place fields to plot (number of place fields x
        number of PF centers).
    - sub_ax (plt.Axes): Subplot on which to plot the data.
    - initial_peak_idx (int, optional): Index of the initial peak in the place fields.
        Default is 0.
    - initial_peak_value (float, optional): Value of the initial peak. Default is None.
    - s (float, optional): Size of the scatter points for the initial peak.
        Default is 10.
    - lw (float, optional): Line width for the arrows. Default is LW.
    - alpha (float, optional): Alpha value for the plot. Default is 0.7.
    - x_vals (1D np.ndarray, optional): X values for the place fields. If None,
        the x-axis is set to the range of the number of inputs. Default is None.
    - y_vals (1D np.ndarray, optional): Y values for the place fields. If None,
        the y-axis is set to the range of the number of place fields. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - keep_yticks (bool, optional): Whether to keep the y-axis ticks, hiding them
        instead of removing them. Default is False.
    - target_positions (float, optional): Position of the target to mark on the plot.
        Default is None.

    Keyword args:
    - **arrow_kwargs: Additional keyword arguments for the arrow properties.
    """

    if x_vals is None:
        x_vals = np.arange(PF_data.shape[1])

    if y_vals is None:
        y_vals = np.arange(PF_data.shape[0])

    arrow_sources = [x_vals[initial_peak_idx]] * len(y_vals)
    if target_positions is not None:
        if len(target_positions) != len(y_vals):
            raise ValueError("target_positions and y_vals must have the same length.")
        arrow_sources = target_positions

    if "weights" in PF_type:
        xlabel = "Input place field center (m)"
    else:
        xlabel = "Position (m)"

    y_diff = 0
    for i, data in enumerate(PF_data):
        if np.isfinite(data).any():
            new_peak_idx = np.argmax(data)
            sub_ax.scatter(
                arrow_sources[i],
                y_vals[i],
                marker=".",
                color="k",
                alpha=alpha,
                s=s,
            )

            if new_peak_idx != initial_peak_idx:
                x_start = arrow_sources[i]
                x_end = x_vals[new_peak_idx]
                if initial_peak_value is not None:
                    y_diff = data[initial_peak_idx] - initial_peak_value

                sub_ax.arrow(
                    x_start,
                    y_vals[i],
                    x_end - x_start,
                    y_diff,
                    facecolor="k",
                    edgecolor="k",
                    length_includes_head=True,
                    alpha=alpha,
                    lw=lw,
                    **arrow_kwargs,
                )

        sub_ax.spines[["top", "left", "right"]].set_visible(False)
        if keep_yticks:
            sub_ax.tick_params(axis="y", length=0)
        else:
            sub_ax.set_yticks([])

        sub_ax.set_xlabel(xlabel)


def get_shift_baseline_idx(target_shifts, num_BTSP_applied=None, seeds=None):
    """
    get_shift_baseline_idx(target_shifts)

    Gets the index of the baseline (0 target shift) in the shift data dictionary.
    Optionally checks whether the number of applied BTSP events are as expected
    (minimum 1, maximum 2, with one 1 for the baseline) and whether all shifts have the
    same random seed.

    Args:
    - target_shifts (1D np.ndarray): Array of target shifts.
    - num_BTSP_applied (1D np.ndarray, optional): Array of number of BTSP events
        applied per shift. Default is None.
    - seeds (1D np.ndarray, optional): Array of random seeds used for each shift.
        Default is None.

    Returns:
    - base_idx (int): Index of the baseline (0 target shift) in the shift data
        dictionary.
    """

    if num_BTSP_applied is not None:
        if len(num_BTSP_applied) != len(target_shifts):
            raise ValueError(
                "num_BTSP_applied and target_shifts must have the same length."
            )
        if num_BTSP_applied.min() != 1 and num_BTSP_applied.max() != 2:
            raise RuntimeError(
                "Expected at least 1 and at most 2 BTSP events per shift."
            )

    if seeds is not None:
        if len(seeds) != len(target_shifts):
            raise ValueError("seeds and target_shifts must have the same length.")
        if len(np.unique(seeds)) != 1:
            raise RuntimeError("Expected all shifts to use the same seed.")

    if np.isclose(target_shifts, 0).sum() == 0:
        raise NotImplementedError(
            "Plotting not implemented if shift of 0 is not in the data dictionary."
        )
    base_idx = np.where(np.isclose(target_shifts, 0))[0][0]

    if num_BTSP_applied is not None and num_BTSP_applied[base_idx] != 1:
        raise RuntimeError("Expected exactly 1 BTSP event for 0 target shift.")

    return base_idx


def plot_linear_shift_PF_examples(
    shift_data,
    Ag=None,
    color=None,
    plot_cmap=False,
    PF_type="history",
    mark_pos_range=None,
):
    """
    plot_linear_shift_PF_examples()

    Plots examples of place fields for different target shifts on the linear track.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see paper.run_linear_shifts()).
    - Ag (Agent, optional): Agent object. If provided, it is used to add markers to
        subplot. Default is None.
    - color (str, optional): Color of the place field plot lines. Default is None.
    - plot_cmap (bool): Whether to plot the place field colormap instead of a peak shift
        plot. Default is False.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - mark_pos_range (tuple, optional): Range of target shift positions to highlight
        on the plot (min_shift, max_shift). Default is None.

    Returns:
    - axes (2D np.ndarray of plt.Axes): Subplots with example place fields plotted.
    """

    # identify the 0 shift
    base_idx = get_shift_baseline_idx(
        shift_data["target_shifts"],
        num_BTSP_applied=shift_data["num_BTSP_applied"],
        seeds=shift_data.get("seeds", None),
    )

    shifts_to_plot = [
        shift for i, shift in enumerate(shift_data["target_shifts"]) if i != base_idx
    ]
    num_plots = len(shifts_to_plot)

    height = 1.3 * num_plots
    figsize = (8.8, height) if plot_cmap else (10.1, height)
    width_ratios = [1.7, 1] if plot_cmap else [3, 1]
    height_ratios = [1] * num_plots

    offset = 0
    if mark_pos_range is not None:
        num_plots += 1
        figsize = (figsize[0], figsize[1] + 0.3)
        height_ratios = [0.1] + height_ratios
        offset = 1

    _, axes = plt.subplots(
        num_plots,
        2,
        figsize=figsize,
        sharex=True,
        sharey="col",
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        gridspec_kw={"wspace": 0.1, "hspace": 0.4},
        squeeze=False,
    )

    PFs, PF_centers, _ = retrieve_PF_data(shift_data, PF_type=PF_type)

    if "weights" in PF_type:
        ytick_max, ymax = 0.25, 0.28
        round_dec = 2
        color = color or params_util.PC_COLOR
        alpha = 0.8
    elif PF_type == "history":
        ytick_max, ymax = 6, 6.5
        round_dec = 0
        color = color or params_util.PYR_SOMATIC_COLOR
        alpha = 1.0
    else:
        raise ValueError(f"PF_type '{PF_type}' not recognized.")

    base = PFs[base_idx, -2]
    peak_idx = np.argmax(base)

    PFs = PFs[:, -1].copy()
    PFs[base_idx] = base

    vmax = np.nanmax(PFs)
    for i, target_shift in enumerate(shifts_to_plot):
        if np.isclose(target_shift, 0):
            continue

        idx = np.where(shift_data["target_shifts"] == target_shift)[0][0]
        ax_idx = i + offset

        plot_1D_PFs(
            PF_centers,
            PFs[idx],
            scale_y=1,
            base=base,
            color=color,
            sub_ax=axes[ax_idx, 0],
            alpha=alpha,
            mark_original=True,
        )
        axes[ax_idx, 0].text(
            1.8, ymax * 0.78, f"{target_shift:.1f} m shift", ha="center", fontsize=12
        )

        if Ag is not None:
            mark_1D_target(axes[ax_idx, 0], Ag=Ag, target_shift=target_shift)

        mark_1D_target(axes[ax_idx, 1], Ag=Ag, target_shift=target_shift)
        if plot_cmap:
            plot_PF_cmap(
                PFs[idx].reshape(1, -1),
                sub_ax=axes[ax_idx, 1],
                vmax=vmax,
                x_vals=PF_centers,
                PF_type=PF_type,
                plot_colorbar=False,
            )

        else:
            plot_PF_peak_shift(
                PFs[idx].reshape(1, -1),
                sub_ax=axes[ax_idx, 1],
                initial_peak_idx=peak_idx,
                # initial_peak_value=base[peak_idx],
                s=35,
                lw=LW,
                x_vals=PF_centers,
                PF_type=PF_type,
                head_width=0.1,
                head_length=0.1,
            )

    for sub_ax in axes[:, 0]:
        sub_ax.set_ylim([0, ymax])
        sub_ax.set_yticks([0, ytick_max])
        sub_ax.set_ylabel("")
        plot_util.expand_ticks(
            sub_ax, axis="y", num_ticks=3, alternating=True, round_dec=round_dec
        )

    for i, sub_ax in enumerate(axes[offset:].T.ravel()):
        format_1D_PF_xaxis(sub_ax, PF_type=PF_type)
        if i != axes.size // 4:
            sub_ax.set_ylabel("")

    for sub_ax in axes[:, 1]:
        sub_ax.set_ylim(-0.8, 0.8)

    if Ag is not None:
        for i, target_shift in enumerate(shifts_to_plot):
            for j, y_1D in enumerate([ymax * 0.95, 0.8]):
                add_1D_position_markers(
                    axes[i + offset, j], Ag=Ag, y_1D=y_1D, target_shift=target_shift
                )

    if mark_pos_range is not None:
        start, end = mark_pos_range
        print(
            "No second BTSP event recorded for shifts to positions between "
            f"{start:.1f}m and {end:.1f}m."
        )
        color = params_util.OBJ_COLOR
        for i in range(2):
            axes[0, i].axvspan(start, end, color=color, alpha=0.8, lw=0)
            axes[0, i].axis("off")

    for ax1D in axes.T:
        plot_util.clear_bottom(ax1D[:-1])

    return axes


def plot_target_shift_PFs(
    shift_data,
    Ag=None,
    mark_examples=list(),
    plot_cmap=False,
    color=None,
    PF_type="history",
    mark_one_BTSP_range=True,
):
    """
    plot_target_shift_PFs()

    Plots the relationship between target shifts and place fields for the linear
    experiment.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see paper.run_linear_shifts()).
    - Ag (Agent, optional): Agent object. If provided, it is used to add markers to
        subplot. Default is None.
    - mark_examples (list): List of target shifts to plot arrows for.
        Default is an empty list.
    - plot_cmap (bool): Whether to plot the place field colormap instead of a peak shift
        plot. Default is False.
    - color (str): Color to use for place cell plots. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - mark_one_BTSP_range (bool): Whether to highlight the range of target shifts
        that received only one BTSP event. Default is True.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with target shifts and place field
        weights plotted.
    """

    width_ratios = [1.2, 1] if plot_cmap else [3, 1]
    _, axes = plt.subplots(
        1,
        2,
        figsize=(10.1, 4.8),
        sharey=True,
        squeeze=False,
        width_ratios=width_ratios,
        gridspec_kw={"wspace": 0.1, "hspace": 0.4},
    )

    ax1D = axes[0, :]

    base_idx = get_shift_baseline_idx(
        shift_data["target_shifts"],
        num_BTSP_applied=shift_data["num_BTSP_applied"],
        seeds=shift_data.get("seeds", None),
    )

    PFs, PF_centers, _ = retrieve_PF_data(shift_data, PF_type=PF_type)

    base = PFs[base_idx, -2].copy()
    base_shift = np.nanmin(base)
    base_shifted = base - base_shift

    PFs = PFs[:, -1].copy()
    PFs[base_idx] = base

    if "weights" in PF_type:
        scale_y = 4
        color = color or params_util.PC_COLOR
        alpha = 0.8
    elif PF_type == "history":
        scale_y = 0.07
        color = color or params_util.PYR_SOMATIC_COLOR
        alpha = 1.0
    else:
        raise ValueError(f"PF_type '{PF_type}' not recognized.")

    plot_1D_PFs(
        PF_centers,
        base_shifted,
        lw=1.8,
        sub_ax=ax1D[0],
        scale_y=scale_y,
        color=color,
        alpha=alpha,
    )

    peak_idx = np.argmax(base)
    for i, shift in enumerate(shift_data["target_shifts"]):
        shifted = PFs[i] - base_shift
        if np.isfinite(shifted.all()):
            plot_1D_PFs(
                PF_centers,
                shifted,
                shift=shift,
                base=base_shifted,
                lw=1.0,
                sub_ax=ax1D[0],
                scale_y=scale_y,
                color=color,
                alpha=alpha,
            )

    for sub_ax in ax1D:
        format_1D_PF_xaxis(sub_ax, PF_type=PF_type)
    ax1D[0].set_ylabel("Target object shift (m)", labelpad=13)
    ax1D[1].set_ylabel("")

    ymin = shift_data["target_shifts"].min()
    ymax = shift_data["target_shifts"].max()
    yticks = [y for y in ax1D[0].get_yticks() if y >= ymin and y <= ymax]
    ax1D[0].spines["left"].set_bounds(ymin, ymax)
    ax1D[0].set_yticks(yticks)

    if mark_one_BTSP_range:
        one_BTSP_range_idxs = gen_util.get_value_index_range(
            shift_data["num_BTSP_applied"], 1, single_range_only=True
        )

        one_BTSP_shift_range = [
            shift_data["target_shifts"][one_BTSP_range_idxs[0]],
            shift_data["target_shifts"][one_BTSP_range_idxs[1] - 1],
        ]

        ax1D[0].axhspan(
            *one_BTSP_shift_range,
            color=params_util.OBJ_COLOR,
            alpha=0.8,
            lw=0,
            xmin=0.99,
        )
        print(
            "No second BTSP event recorded for shifts between {:.1f}m and {:.1f}m.".format(
                *one_BTSP_shift_range
            )
        )

    for idx in np.where(shift_data["target_shifts"] == 0)[0]:
        PFs[idx] = PFs[idx]

    if plot_cmap:
        plot_PF_cmap(
            PFs,
            sub_ax=ax1D[1],
            x_vals=PF_centers,
            y_vals=shift_data["target_shifts"],
            PF_type=PF_type,
            keep_yticks=True,
        )
    else:
        plot_PF_peak_shift(
            PFs * scale_y,
            sub_ax=ax1D[1],
            initial_peak_idx=peak_idx,
            # initial_peak_value=base[peak_idx] * scale_y,
            x_vals=PF_centers,
            y_vals=shift_data["target_shifts"],
            PF_type=PF_type,
            lw=LW * 0.8,
            head_width=0.04,
            head_length=0.04,
            keep_yticks=True,
        )

        x_loc = gen_util.get_proportion_edges(PF_centers, 0.01)
        for target_shift in mark_examples:
            if np.isclose(target_shift, 0):
                continue
            if target_shift not in shift_data["target_shifts"]:
                raise RuntimeError(
                    f"{target_shift} shift not found in data dictionary."
                )
            idx = np.where(shift_data["target_shifts"] == target_shift)[0][0]
            ax1D[1].scatter(
                x_loc,
                shift_data["target_shifts"][idx],
                color=color,
                marker=">",
                s=10,
                zorder=5,
            )

    if Ag is not None:
        for i, sub_ax in enumerate(ax1D):
            add_1D_position_markers(
                sub_ax, Ag=Ag, y_1D=sub_ax.get_ylim()[1], target_alpha=0.5
            )
            if i == 0:
                mark_1D_target(sub_ax, Ag=Ag, alpha=0.4)

    plot_util.pad_axis(ax1D[0], axis="y", pad_prop=0.05)

    return ax1D


def plot_openfield_components(Pyrs, titles=False, traj_idx=8, PC_to_plot=206):
    """
    plot_openfield_components(Pyrs)

    Plots the openfield components for an experiment.

    Args:
    - Pyrs (Pyr): Pyr object containing the environment, agent, object and place cells.
    - titles (bool, optional): Whether to add titles to each subplot. Default is False.
    - traj_idx (int, optional): Index of the trajectory to plot. Default is 8.
    - PC_to_plot (int, optional): Index of the place cell to plot. Default is 206.

    Returns:
    - axes (np.ndarray of plt.Axes): Array of subplots with openfield components plotted.
    """

    _, axes = plt.subplots(
        2, 2, figsize=(4, 4), gridspec_kw={"wspace": 0.015, "hspace": 0.19}
    )
    Env, Ag, PCs, Objs = ext_util.extract_objects_from_Pyrs(Pyrs)
    if Env.D != 2:
        raise ValueError("2D plotting is only supported for 2D environments.")

    env_kwargs = {"skip_object_types": ["teleport"], "no_legend": True}
    y = 1.02
    chosen_PCs = [PC_to_plot]

    env_sub_ax, traj_sub_ax = axes[0]
    Obj_sub_ax, PC_sub_ax = axes[1]

    # top row
    if titles:
        env_sub_ax.set_title("Environment", y=y)
    Env.plot_environment(
        sub_ax=env_sub_ax, scale_loc=(1.62, 1.88), scale_length=0.5, **env_kwargs
    )

    if titles:
        traj_sub_ax.set_title("Example trajectory", y=y)

    Ag.plot_trajectories(
        ax=traj_sub_ax,
        alpha=0.4,
        traj_idxs=[traj_idx],
        cmap_per=True,
        framerate=1 / Ag.dt,
        s_2D=5,
        plot_target=False,
        plot_agent=False,
        plot_traj_ends=True,
        **env_kwargs,
    )

    # bottom row
    if titles:
        Obj_sub_ax.set_title("Object field", y=y, color=Objs.color)
    Objs.plot_rate_map(
        sub_ax=Obj_sub_ax, plot_objects=False, colorbar=False, **env_kwargs
    )

    if titles:
        PC_sub_ax.set_title(
            f"Place fields ({len(chosen_PCs)}/{PCs.n})", y=y, color=PCs.color
        )
    Env.plot_environment(sub_ax=PC_sub_ax, plot_objects=True, **env_kwargs)
    plot_fcts.plot_overlayed_rate_maps(
        PCs, sub_ax=PC_sub_ax, method="max", chosen_neurons=chosen_PCs, colorbar=False
    )
    PCs.plot_place_cell_locations(
        sub_ax=PC_sub_ax, plot_objects=True, s=3, alpha=0.8, marker=".", **env_kwargs
    )

    vmin, vmax = plot_util.match_clims([Obj_sub_ax, PC_sub_ax])

    cbar_axis = axes.ravel()[-1]
    plot_util.add_colorbars(
        cbar_axis,
        cbar_axis.get_images()[-1],
        vmin=vmin,
        vmax=vmax,
        label="Neural activity",
        outline=True,
    )

    for sub_ax in axes.ravel()[:-1]:
        plot_util.add_dummy_colorbar_axis(sub_ax)

    return axes


def plot_last_openfield_PF(
    Pyrs,
    i=0,
    fig_side=3.0,
    lw=LW,
    alpha=0.8,
    obj_s=30,
    BTSP_s=BTSP_S,
    PF_type="history",
    t_end=None,
    sub_ax=None,
    no_teleport=True,
    plot_colorbar=True,
    plot_BTSP_events=True,
    **kwargs,
):
    """
    plot_openfield_PF(Pyrs)

    Plots the last recorded place field of the Pyr neuron in the openfield corridor.

    Args:
    - Pyrs (Pyr): Pyr object for openfield.
    - i (int, optional): Index of the Pyr neuron to plot. Default is 0.
    - fig_side (float, optional): Size of the figure. Default is 3.0.
    - lw (float, optional): Line width. Default is LW.
    - alpha (float, optional): Transparency level. Default is 0.8.
    - obj_s (float, optional): Size of the object markers. Default is 30.
    - BTSP_s (float, optional): Size of the BTSP event markers. Default is BTSP_S.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - t_end (float, optional): End time for identifying last PF. Only applies if
        PF_type is "history". Default is None.
    - sub_ax (plt.Axes, optional): Subplot on which to plot the data. If None, a new
        figure and subplot are created. Default is None.
    - no_teleport (bool, optional): Whether to skip plotting teleportation ports in the
        plot. Default is True.
    - plot_colorbar (bool, optional): Whether to plot the colorbar. Default is True.
    - plot_BTSP_events (bool, optional): Whether to plot BTSP event markers. Default is
        True.

    Keyword args:
    - **kwargs: Additional keyword arguments for the plot_2D_PFs function.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted PFs.
    """

    if sub_ax is None:
        fig, sub_ax = plt.subplots(figsize=(fig_side, fig_side))
    else:
        fig = sub_ax.figure

    Pyrs = get_somatic_compartment(Pyrs)

    PF_t_start, PF_t_end = None, None
    if PF_type == "history":
        round_dec = 0
        vmax = 10
        PF_times = ext_util.get_times_for_each_BTSP_event(
            Pyrs, i=i, use_nans=False, t_end=t_end
        )
        if len(PF_times):
            PF_t_start, PF_t_end = PF_times[-1]
    else:
        round_dec = 2
        vmax = None
        if t_end is not None:
            raise NotImplementedError("t_end is only applicable for PF_type 'history'.")

    skip_object_types = list()
    if no_teleport:
        skip_object_types.append("teleport")

    plot_fcts.plot_2D_PFs(
        Pyrs,
        PF_type=PF_type,
        PF_t_start=PF_t_start,
        PF_t_end=PF_t_end,
        chosen_neurons=[i],
        alpha=alpha,
        obj_s=obj_s,
        lw=lw,
        plot_BTSP_events=plot_BTSP_events,
        BTSP_s=BTSP_s,
        BTSP_marker=BTSP_ASTERISK,
        skip_object_types=skip_object_types,
        no_legend=True,
        vmin=0,
        vmax=vmax,
        round_dec=round_dec,
        ax=sub_ax,
        marker="s",
        plot_colorbar=plot_colorbar,
        cbar_side="right",
        cbar_outline=True,
        **kwargs,
    )

    if plot_colorbar:
        cax = fig.axes[-1]
        clabel = get_PF_label(PF_type, title=False)
        cax.set_ylabel(clabel)

    return sub_ax


def plot_openfield_corridor_PFs(
    Pyrs,
    PFs,
    PF_centers,
    PF_type="history",
    num_BTSP=None,
    num_teleportations=None,
    fig_side=1.5,
    obj_s=1,
    num_cols=5,
    lw=LW,
    alpha=1.0,
    no_teleport=True,
    axes=None,
    **kwargs,
):
    """
    plot_openfield_corridor_PFs(Pyrs)

    Plots series of openfield corridor place fields.

    Args:
    - Pyrs (Pyr): Pyr object for openfield.
    - PFs (2D np.ndarray): Place fields to plot (number of place fields x
        number of PF centers).
    - PF_centers (1D np.ndarray): Centers of the input place fields.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - num_BTSP (int, optional): Number of BTSP events per plot, to use in plot titles.
        Default is None.
    - num_teleportations (int, optional): Number of teleportation events per plot, to
        use in plot titles. Default is None.
    - fig_side (float, optional): Size of the figure. Default is 1.5.
    - obj_s (float, optional): Size of the object markers. Default is 1.
    - num_cols (int, optional): Number of columns in the subplot grid. Default is 5.
    - lw (float, optional): Line width. Default is LW.
    - alpha (float, optional): Transparency level. Default is 1.0.
    - no_teleport (bool, optional): Whether to skip plotting teleportation ports in the
        plot. Default is True.
    - axes (np.ndarray of plt.Axes, optional): Array of subplots to plot on. If None,
        a new array is created. If provided, it should either have the same number of
        subplots as the number of PFs to plot, or one extra per row for the colorbar.
        Default is None.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to plot_fcts.plot_2D_PFs().

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted PFs.
    """

    num_PFs = len(PFs)

    if num_BTSP is not None and len(num_BTSP) != num_PFs:
        raise ValueError("num_BTSP and PFs must have the same length.")
    if num_teleportations is not None and len(num_teleportations) != num_PFs:
        raise ValueError("num_teleportations and PFs must have the same length.")

    if axes is None:
        num_cols = min(num_cols, num_PFs)
        num_rows = int(np.ceil(num_PFs / num_cols))
        width_ratios = [1] * num_cols + [0.1]
        hspace = (
            0.28 if num_BTSP is not None and num_teleportations is not None else 0.2
        )

        _, axes = plt.subplots(
            num_rows,
            num_cols + 1,
            figsize=(fig_side * num_cols * 1.04, fig_side * num_rows),
            gridspec_kw={"width_ratios": width_ratios, "hspace": hspace},
            squeeze=False,
        )

        PF_axes = axes[:, :-1].ravel()
        caxes = axes[:, -1]

    else:
        axes = np.asarray(axes)
        if axes.size < num_PFs:
            raise ValueError(
                f"Provided array of {axes.size} subplots, but {num_PFs} PFs."
            )
        elif axes.size == num_PFs:
            PF_axes = axes.ravel()
            caxes = plot_util.add_colorbar_axes(axes, end_only=True, size="5%")
        elif axes.reshape(1, -1)[:, :-1].size <= num_PFs:
            PF_axes = axes.reshape(1, -1)[:, :-1].ravel()[:num_PFs]
            caxes = axes.reshape(1, -1)[:, -1:]
        else:
            raise NotImplementedError(
                "Unclear how to include colorbar subplots given array of subplots with "
                f"shape {axes.shape} and {num_PFs} PFs to plot."
            )

    cmap = "inferno"
    vmin = 0
    if PF_type == "history":
        vmax = 10
    else:
        vmax = np.ceil(np.nanmax(PFs) * 10**2) / 10**2

    skip_object_types = list()
    if no_teleport:
        skip_object_types.append("teleport")

    if isinstance(Pyrs, list):
        if len(Pyrs) != num_PFs:
            raise ValueError("Length of Pyrs list must match number of PFs.")
        PFs = PFs.reshape(PFs.shape[0], 1, PFs.shape[1])
        use_PF_axes = PF_axes
    else:
        Pyrs = [Pyrs]
        use_PF_axes = [PF_axes]
        PFs = [PFs]

    for i, use_Pyrs in enumerate(Pyrs):
        plot_fcts.plot_2D_PFs(
            get_somatic_compartment(use_Pyrs),
            PF_type=PF_type,
            PFs=PFs[i],
            PF_centers=PF_centers,
            alpha=alpha,
            obj_s=obj_s,
            lw=lw,
            plot_BTSP_events=False,
            skip_object_types=skip_object_types,
            no_legend=True,
            vmin=vmin,
            vmax=vmax,
            ax=use_PF_axes[i],
            marker="s",
            plot_colorbar=False,
            cbar_side="right",
            cbar_outline=True,
            **kwargs,
        )

    for i, sub_ax in enumerate(PF_axes):
        title = f"#{i + 1}"
        add_strs = list()
        if num_BTSP is not None:
            event_str = "event" if num_BTSP[i] == 1 else "events"
            add_strs.append(f"{num_BTSP[i]} BTSP {event_str}")
        if num_teleportations is not None:
            event_str = (
                "teleportation" if num_teleportations[i] == 1 else "teleportations"
            )
            add_strs.append(f"{num_teleportations[i]} {event_str}")
        if len(add_strs):
            add_str = ", \n".join(add_strs)
            title = f"#{i + 1} ({add_str})"
        sub_ax.set_title(title, fontsize=10)

    for i, cax in enumerate(caxes):
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        im = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(im, cax=cax)
        if i == len(caxes) - 1:
            clabel = get_PF_label(PF_type, title=False)
            cax.set_ylabel(clabel)

    return axes


def plot_openfield_corridor_BTSP_trajectory(
    Pyrs,
    BTSP_idx=0,
    sub_ax=None,
    x_lims=None,
    y_lims=(None, 0.6),
    no_teleport=True,
    plot_colorbar=True,
    clabel_length=None,
    s_2D=7,
    obj_s=20,
    **kwargs,
):
    """
    plot_openfield_corridor_BTSP_trajectory(Pyrs)

    Plots the trajectory around the first BTSP event in an openfield corridor experiment.

    Args:
    - Pyrs (Pyr): Pyr object containing the environment, agent, object and place cells.
    - BTSP_idx (int, optional): Index of the BTSP event to plot. Default is 0.
    - sub_ax (plt.Axes, optional): Subplot on which to plot the data. If None, a new
        subplot is created. Default is None.
    - x_lims (list, optional): X-axis limits for the plot. Default is None.
    - y_lims (list, optional): Y-axis limits for the plot. Default is (None, 0.5).
    - no_teleport (bool, optional): Whether to skip plotting teleportation ports in the
        plot. Default is True.
    - plot_colorbar (bool, optional): Whether to plot a colorbar showing BTSP
        kernel strength. Default is True.
    - clabel_length (bool, optional): Maximum length of the colorbar label before
        splitting into two lines. Default is None.
    - s_2D (float, optional): Size of the trajectory markers. Default is 7.
    - obj_s (float, optional): Size of the object markers. Default is 20.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to Ag.plot_trajectories().

    Returns:
    - sub_ax (plt.Axes): Subplot with the openfield BTSP trajectory.
    """

    Env, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)
    if Env.D != 2:
        raise ValueError("2D plotting is only supported for 2D environments.")

    Pyrs = get_somatic_compartment(Pyrs)

    split_clabel = False
    if sub_ax is None:
        x_prop = 1.0
        if x_lims is not None:
            xmin = x_lims[0] or Env.extent[0]
            xmax = x_lims[1] or Env.extent[1]
            x_prop = (xmax - xmin) / (Env.extent[1] - Env.extent[0])

        y_prop = 1.0
        if y_lims is not None:
            ymin = y_lims[0] or Env.extent[2]
            ymax = y_lims[1] or Env.extent[3]
            y_prop = (ymax - ymin) / (Env.extent[3] - Env.extent[2])
            if y_prop < 0.5:
                split_clabel = True

        _, sub_ax = plt.subplots(figsize=(4.2 * x_prop, 2.85 * y_prop))

    kwargs["no_legend"] = True
    if no_teleport:
        kwargs["skip_object_types"] = ["teleport"]

    t_start, t_end, cmap = get_BTSP_times_and_cmap(Pyrs, BTSP_idx=BTSP_idx)

    Ag.plot_trajectories(
        ax=sub_ax,
        t_start=t_start,
        t_end=t_end,
        framerate=1 / Ag.dt,
        s_2D=s_2D,
        s=obj_s,
        colormap=cmap,
        alpha=None,
        cmap_per=True,
        plot_target=False,
        plot_agent=False,
        plot_traj_ends=False,
        **kwargs,
    )

    factor = min(1, obj_s / 20)
    BTSP_kwargs = {
        "t_start": t_start,
        "t_end": t_end,
        "s": BTSP_S * factor,
        "marker": BTSP_ASTERISK,
        "lw": LW,
    }

    Pyrs.add_BTSP_markers_to_plots(ax=sub_ax, **BTSP_kwargs)

    if x_lims is not None:
        sub_ax.set_xlim(x_lims)
    if y_lims is not None:
        sub_ax.set_ylim(y_lims)

    if plot_colorbar:
        cmap, vmin, vmax = Pyrs.get_BTSP_kernel_based_cmap(for_colorbar=True)
        norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
        clabel = "BTSP kernel strength"
        if clabel_length is not None:
            split_clabel = False
            if clabel_length >= len(clabel):
                pass
            elif clabel_length < len("strength"):
                raise ValueError(
                    "clabel_length too small to split 'BTSP kernel strength' label."
                )
            elif clabel_length < len("BTSP kernel"):
                split_clabel = True
            else:
                clabel = "BTSP kernel\nstrength"

        if split_clabel:
            clabel = clabel.replace(" ", "\n")

        plot_util.add_colorbars(
            sub_ax,
            cbar,
            vmin=vmin,
            vmax=vmax,
            label=clabel,
            outline=True,
            size="5%",
        )

    return sub_ax


def plot_openfield_corridor_timelines(
    BTSP_times,
    visit_times,
    PF_times,
    end_times=None,
    teleportation_times=None,
    num_teleportation_pairs=1,
    in_min=True,
    factor=1.2,
    fig_width=6,
):
    """
    plot_openfield_corridor_timelines(BTSP_times, visit_times, PF_times)

    Plots timelines of BTSP events, object visits, and place field evaluations in an
    openfield corridor experiment.

    Args:
    - BTSP_times (2D np.ndarray): Array of BTSP event times for each repeat.
    - visit_times (2D np.ndarray): Arrays of object visit times for each repeat.
    - PF_times (3D np.ndarray): Arrays of place field evaluation start and end times
        for each repeat.
    - teleportation_times (2D np.ndarray, optional): Arrays of teleportation event
        times for each repeat, if applicable. Default is None.
    - end_times (1D np.ndarray, optional): End time of the experiment for each repeat.
        Default is None.
    - num_teleportation_pairs (int, optional): Number of teleportation pairs per
        timeline. Default is None.
    - in_min (bool, optional): Whether to plot time in minutes instead of seconds.
        Default is True.
    - factor (float, optional): Scaling factor for marker sizes and line widths.
        Default is 1.2.
    - width (float, optional): Width of the figure. Default is 6.

    Returns:
    - sub_ax (plt.Axes): Subplot with the plotted timelines.
    """

    _, sub_ax = plt.subplots(figsize=(fig_width, 3.2))

    time_factor = 1 / 60 if in_min else 1
    time_unit = "min" if in_min else "s"

    num_repeats = len(BTSP_times)
    for i, repeat_BTSP_times in enumerate(BTSP_times):
        sub_ax.scatter(
            repeat_BTSP_times * time_factor,
            [i + 0.6] * len(repeat_BTSP_times),
            marker=BTSP_ASTERISK,
            color=params_util.PYR_SOMATIC_COLOR,
            s=16 * factor,
            lw=0.7 * factor,
            zorder=5,
        )

    for i, repeat_visit_times in enumerate(visit_times):
        sub_ax.scatter(
            repeat_visit_times * time_factor,
            [i + 1] * len(repeat_visit_times),
            marker="o",
            color=params_util.OBJ_COLOR,
            s=4 * factor,
            lw=0.7 * factor,
            zorder=5,
        )

    for i, repeat_PF_times in enumerate(PF_times):
        idx = np.where(np.isfinite(repeat_PF_times).all(axis=1))[0][-1]
        start, end = repeat_PF_times[idx] * time_factor
        rect = mpl_patches.Rectangle(
            (start, i + 0.77),
            end - start,
            0.5,
            color=params_util.PYR_SOMATIC_COLOR,
            alpha=0.4,
            lw=0,
            zorder=3,
        )
        sub_ax.add_patch(rect)

    if end_times is not None:
        for i, end_time in enumerate(end_times):
            sub_ax.plot(
                [end_time * time_factor] * 2,
                [i + 0.77, i + 0.77 + 0.5],
                color="k",
                lw=1.3 * factor,
                zorder=4,
            )

    if teleportation_times is not None:
        if num_teleportation_pairs != 1:
            raise NotImplementedError(
                "Plotting for multiple teleportation port pairs not implemented."
            )
        teleport_color = params_util.get_teleportation_colors()[0]
        for i, times in enumerate(teleportation_times):
            for time in times[~np.isnan(times)]:
                sub_ax.plot(
                    [time * time_factor] * 2,
                    [i + 0.7, i + 1.3],
                    color=teleport_color,
                    lw=0.8 * factor,
                    ls="dashed",
                    zorder=4,
                )

    sub_ax.set_xlim(0, None)
    max_xtick = sub_ax.get_xticks()[-1]
    if sub_ax.get_xlim()[1] < max_xtick:
        sub_ax.set_xlim(None, max_xtick)
    sub_ax.spines["bottom"].set_bounds(0, sub_ax.get_xlim()[1])
    plot_util.pad_axis(sub_ax, axis="x", pad_prop=0.02, prop_high=1.0)

    sub_ax.set_ylim(0.2, num_repeats + 0.8)
    sub_ax.set_yticks(np.arange(num_repeats) + 1)
    sub_ax.yaxis.set_ticks_position("right")
    sub_ax.yaxis.set_tick_params(length=3)
    sub_ax.yaxis.set_inverted(True)

    sub_ax.spines[["top", "left", "right"]].set_visible(False)

    sub_ax.set_xlabel(f"Time ({time_unit})")

    return sub_ax


def plot_openfield_teleportation_summary(learner, num_sec=4, width_per=1.875):
    """
    plot_openfield_teleportation_summary(learner)

    Plots a summary figure for openfield teleportation experiments.

    Args:
    - learner (Learner): Learner object containing the experiment data.
    - num_sec (int, optional): Number of seconds to plot around each teleportation
        event. Default is 6.
    - width_per (float, optional): Width per teleportation subplot. Default is 1.875.

    Returns:
    - axes (np.ndarray of plt.Axes): Array of subplots with the plotted summary.
    """

    num_teleportations = len(learner.Agent.teleportation_df)

    BTSP_steps = learner.Pyrs.SomaticCompartment.get_BTSP_steps()
    num_steps_after = num_sec / learner.Agent.dt
    BTSP_teleportation_idxs = list()
    PF_idxs = [1]
    for tel_idx, step in enumerate(learner.Agent.teleportation_df["step_num"]):
        mask = (BTSP_steps >= step) & (BTSP_steps < step + num_steps_after)
        if mask.sum():
            BTSP_teleportation_idxs.append(tel_idx)
            PF_idxs.append(np.where(mask)[0][0] + 1)

    num_plots = num_teleportations
    if len(BTSP_teleportation_idxs):
        if BTSP_teleportation_idxs[-1] == num_teleportations - 1:
            num_plots += 1

    _, axes = plt.subplots(
        6,
        num_plots,
        figsize=(num_plots * width_per, 6.5),
        gridspec_kw={"wspace": 0.25, "hspace": 0.08},
        height_ratios=[1, 1, 1, 0.5, 0.5, 0.5],
        squeeze=False,
        sharey="row",
    )

    # first row: PFs
    PF_info = metrics.gather_PF_info(learner, position_name="reward")
    PF_axes = [axes[0, 0]]
    for i in range(num_teleportations):
        if i in BTSP_teleportation_idxs:
            PF_axes.append(axes[0, i + 1])
        elif i + 1 < num_plots:
            axes[0, i + 1].axis("off")

    plot_openfield_corridor_PFs(
        learner.Pyrs,
        PFs=PF_info["PFs"][np.asarray(PF_idxs)],
        PF_centers=PF_info["PF_centers"],
        axes=PF_axes,
        no_teleport=False,
        obj_s=10,
    )

    for i in range(num_teleportations):
        BTSP_str = " (BTSP)" if i in BTSP_teleportation_idxs else " (no BTSP)"
        title_str = f"Teleport #{i + 1}{BTSP_str}"
        axes[0, i].set_title(title_str, y=1.04)

    comps = [
        learner.Pyrs.SomaticCompartment,
        learner.Pyrs.ApicalCompartment,
        learner.Pyrs.ApicalInhibition,
    ]

    framerate = 1 / learner.Agent.dt
    for i, time in enumerate(learner.Agent.teleportation_df["time"]):
        # second row: trajectories around teleportation
        sub_ax = axes[1, i]
        t_start = np.around(time, 0) - num_sec
        t_end = t_start + num_sec * 2
        learner.Agent.plot_trajectories(
            t_start=t_start,
            t_end=t_end,
            ax=sub_ax,
            s_2D=3,
            s=10,  # objects
            no_legend=True,
            plot_target=False,
            cmap_per=True,
            framerate=framerate,
        )

        xticks = np.linspace(t_start, t_end, 5)
        xtick_labels = [
            str(int(xtick)) if i % 2 == 0 else "" for i, xtick in enumerate(xticks)
        ]
        for j, comp in enumerate(comps):
            sub_ax = axes[3 + j, i]
            # bottom rows: trajectories around teleportation
            plot_single_neuron_rate_timeseries(
                comp,
                t_start=t_start,
                t_end=t_end,
                BTSP_s=BTSP_S / 2,
                plot_BTSP_kernel=False,
                plot_reward=(j == 0),
                plot_teleportation=(j == 0),
                sub_ax=sub_ax,
                num_ticks=5,
                in_min=False,
                lw=LW * 0.8,
            )

            # axis is in min
            sub_ax.set_xlim(t_start / 60, t_end / 60)
            sub_ax.spines["bottom"].set_bounds(t_start / 60, t_end / 60)
            sub_ax.set_xticks(xticks / 60)
            sub_ax.set_xticklabels(xtick_labels)  # label in seconds

            if j != 0:
                learner.Agent.add_teleportation_markers_to_plots(
                    sub_ax,
                    t_start=t_start,
                    t_end=t_end,
                    timeseries=True,
                    plot_markers=False,
                    lw=LW * 0.8,
                    no_legend=True,
                )
            if j != len(comps) - 1:
                plot_util.clear_bottom(sub_ax)
                sub_ax.set_ylabel("")

            if i == 0:
                sub_ax.spines["left"].set_position(("outward", 4))
            else:
                sub_ax.spines["left"].set_visible(False)
                sub_ax.tick_params(axis="y", left=False)
                sub_ax.set_ylabel("")

        plot_util.match_y_axis_scales(axes[3:].ravel())

    # third row: BTSP trajectories
    for i in range(num_teleportations):
        if i in BTSP_teleportation_idxs:
            BTSP_idx = BTSP_teleportation_idxs.index(i) + 1
            plot_openfield_corridor_BTSP_trajectory(
                learner.Pyrs,
                sub_ax=axes[2, i],
                BTSP_idx=BTSP_idx,
                y_lims=None,
                no_teleport=False,
                clabel_length=15,
                obj_s=10,  # objects
                s_2D=3,
            )
        else:
            axes[2, i].axis("off")

    if num_plots > num_teleportations:
        for sub_ax in axes[1:, -1]:
            sub_ax.axis("off")

    return axes


def plot_openfield_overlayed_last_PFs(learner, PF_type="history", sub_ax=None):
    """
    plot_openfield_overlayed_last_PFs(learner)

    Plots the last place fields for an openfield experiment, overlayed.

    Args:
    - learner (Learner): Learner object containing the experiment data.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - sub_ax (plt.Axes, optional): Subplot on which to plot the data. If None, a new
        subplot is created. Default is None.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted overlayed PFs.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(2, 2))

    data_dict = metrics.gather_PF_info(learner, position_name="reward")
    PFs, PF_centers = retrieve_PF_data(data_dict, PF_type=PF_type, width=False)

    overlayed_PF = PFs[-1].max(axis=0)
    plot_openfield_corridor_PFs(
        learner.Pyrs,
        PFs=overlayed_PF.reshape(1, -1),
        PF_centers=PF_centers,
        PF_type=PF_type,
        axes=[sub_ax],
        plot_objects=False,
    )
    sub_ax.set_title("")

    return sub_ax


def plot_openfield_multitarget_summary(learner, PF_type="history", after_remap=False):
    """
    plot_openfield_multitarget_summary(learner)

    Plots a summary figure for openfield multitarget experiments.

    Args:
    - learner (Learner): Learner object containing the experiment data.
    - t_start (float, optional): Start time for trajectory plotting. Default is 0.
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".

    Returns:
    - ax1D (np.ndarray of plt.Axes): Array of subplots with the plotted summary.
    """

    gridspec_kw = {"width_ratios": [1.0, 1.0, 1.08], "wspace": 0.07}
    _, ax1D = plt.subplots(1, 3, figsize=(5, 2), gridspec_kw=gridspec_kw)

    if after_remap:
        step = get_learner_remap_step(learner, idx=0, num_total=1)
    else:
        step = learner.agent_start_step
    t_start = step * learner.Agent.dt

    learner.Agent.plot_trajectories(
        ax=ax1D[0],
        t_start=t_start,
        t_end=t_start + 120,
        framerate=1 / learner.Agent.dt,
        alpha=0.4,
        s_2D=3,
        s=15,
        cmap_per=True,
        plot_target=False,
        plot_agent=False,
        plot_traj_ends=True,
        no_legend=True,
    )

    learner.Pyrs.SomaticCompartment.plot_BTSP_locations(
        sub_ax=ax1D[1],
        t_start=t_start,
        s=BTSP_S / 2,
        max_alpha=0.8,
        marker=BTSP_ASTERISK,
        plot_objects=False,
        plot_before=False,
        no_legend=True,
    )

    plot_openfield_overlayed_last_PFs(learner, PF_type=PF_type, sub_ax=ax1D[-1])

    return ax1D


def plot_openfield_multitarget_PFs(
    Pyrs,
    n="all",
    PF_type="history",
    plot_colorbar=False,
    width_per=0.9,
    split_time=None,
):
    """
    plot_openfield_multitarget_PFs(Pyrs)

    Plots place fields for multiple Pyr neurons in an openfield multitarget experiment.

    Args:
    - Pyrs (Pyr): Pyr object for openfield.
    - n (int or str, optional): Number of Pyr neurons to plot. If "all", plots all
        neurons. Default is "all".
    - PF_type (str, optional): PF evaluation method to plot. Default is "history".
    - plot_colorbar (bool, optional): Whether to plot the colorbar. Default is False.
    - width_per (float, optional): Width per subplot. Default is 0.9.
    - split_time (float, optional): If provided, plots last PFs calculated before and
        after the time provided. Otherwise, plots object cell rate maps and
        corresponding pyramidal neuron PFs. Default is None.

    Returns:
    - axes (np.ndarray of plt.Axes): Array of subplots with the plotted place fields.
    """

    if n == "all":
        n = Pyrs.n

    ncols = min(n, 10)
    nrows = int(np.ceil(n / ncols)) * 2
    _, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * width_per, nrows * width_per * 1.1),
        gridspec_kw={"hspace": 0.11 / width_per, "wspace": 0.09 / width_per},
    )

    event_str = "ev." if width_per < 0.9 else "event"

    top_axes = axes[::2].ravel()
    bottom_axes = axes[1::2].ravel()

    if split_time is None:
        axis_sets = [bottom_axes]
        t_ends = [None]
    else:
        axis_sets = [top_axes, bottom_axes]
        t_ends = [split_time, None]

    for i in range(n):
        if split_time is None:
            Objs = Pyrs.get_main_apical_input_layer(src_name="Obj")
            dend_idx = Pyrs.get_index_of_main_apical_input(neuron_idx=i, src_name="Obj")
            Objs.plot_rate_map(
                ax=top_axes[i],
                chosen_neurons=[dend_idx],
                no_legend=True,
                s=2,
                colorbar=False,
                wall_lw=2,
            )
            top_axes[i].set_title(f"#{i + 1}", fontsize="medium")

        use_colorbar = plot_colorbar and (i % ncols == ncols - 1)
        for j, axis_set in enumerate(axis_sets):
            plot_last_openfield_PF(
                Pyrs,
                i=i,
                PF_type=PF_type,
                sub_ax=axis_set[i],
                t_end=t_ends[j],
                plot_colorbar=use_colorbar,
                obj_s=2,
                wall_lw=2,
                plot_BTSP_events=False,
            )
            num_BTSP = len(
                Pyrs.SomaticCompartment.get_BTSP_steps(
                    chosen_neurons=[i], t_end=t_ends[j]
                )
            )
            if j == 1:
                num_BTSP = num_BTSP - prev_num_BTSP
                title_str = f"+{num_BTSP} BTSP ev."
            else:
                if width_per < 0.9:
                    event_str = "ev."
                else:
                    event_str = "event" if num_BTSP == 1 else "events"
                title_str = f"{num_BTSP} BTSP {event_str}"
            axis_set[i].set_title(title_str, fontsize="small", y=0.99)

            prev_num_BTSP = num_BTSP

    plot_util.match_clims(axes.ravel())

    return axes


def plot_openfield_remapping_pre_post_weights(learner):
    """
    plot_openfield_remapping_pre_post_weights(learner)

    Plots pre- and post-remapping synaptic weights for an openfield remapping experiment.

    Args:
    - learner (Learner): Learner object containing the experiment data.

    Returns:
    - ax1D (np.ndarray of plt.Axes): Array of subplots with the plotted weights.
    """

    fig, ax1D = plt.subplots(1, 2, figsize=(2.7, 1.5), gridspec_kw={"wspace": 0.26})

    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(learner.Pyrs)

    remap_step = get_learner_remap_step(learner, idx=0, num_total=1)

    recorded_weights = learner.get_recorded_weights()
    idx = np.where(recorded_weights["steps"] < remap_step)[0][-1]

    PF_weights_overlayed = recorded_weights["weights"][idx : idx + 1].max(axis=1)

    if not gen_util.attribute_type_checker(PCs, "PlaceCells"):
        raise ValueError("PCs is not a PlaceCells object.")
    if not hasattr(PCs, "_current_sorter"):
        raise ValueError("PCs does not have a current sorter used for remapping.")

    shuffle_idxs = [-2, -1]

    for i, shuffle_idx in enumerate(shuffle_idxs):
        PF_centers = PCs.get_place_cell_centers(shuffle_idx)
        plot_fcts.plot_2D_PFs(
            get_somatic_compartment(learner.Pyrs),
            PF_type="weights",
            PFs=PF_weights_overlayed,
            PF_centers=PF_centers,
            round_dec=2,
            wall_lw=2,
            obj_s=10,
            plot_BTSP_events=False,
            no_legend=True,
            ax=ax1D[i],
            marker="s",
            plot_objects=True,
            plot_colorbar=(i == 1),
            cbar_side="bottom",
            cbar_size="7%",
            cbar_outline=True,
        )

    cax = fig.axes[-1]
    clabel = get_PF_label("weights", title=False)
    cax.set_label(clabel)

    return ax1D


def plot_remapping_pre_post_BTSP(
    learner, applied_only=False, alpha=0.5, plot_regression=False
):
    """
    plot_remapping_pre_post_BTSP(learner)

    Plots number of BTSP events pre- and post-remapping for a remapping experiment.

    Args:
    - learner (Learner): Learner object containing the experiment data.

    Returns:
    - sub_ax (plt.Axes): Subplot with the plotted BTSP events pre and post remapping.
    """

    _, sub_ax = plt.subplots(figsize=(2.5, 1.8))
    remap_step = get_learner_remap_step(learner, idx=0, num_total=1)
    remap_time = remap_step * learner.Agent.dt

    times = [(None, remap_time), (remap_time, None)]
    BTSP_counts = list()
    for t_start, t_end in times:
        num_BTSP = learner.Pyrs.SomaticCompartment.get_BTSP_counts(
            t_start=t_start, t_end=t_end, applied_only=applied_only
        )
        BTSP_counts.append(num_BTSP)

    BTSP_pre, BTSP_post = BTSP_counts

    max_spread = max(0.1, min(0.5, 0.1 * BTSP_pre.max()))
    BTSP_pre_plot = gen_util.spread_data(BTSP_post, BTSP_pre, max_spread=max_spread)

    sub_ax.scatter(
        BTSP_pre_plot,
        BTSP_post,
        color=learner.Pyrs.SomaticCompartment.color,
        alpha=alpha,
        s=10,
    )

    for i, axis in enumerate(["x", "y"]):
        ticks = np.arange(BTSP_counts[i].max() + 1)
        if len(ticks) > 5:
            ticks = ticks[:: int(len(ticks) / 5) + 1]
        tick_setter = sub_ax.set_xticks if axis == "x" else sub_ax.set_yticks
        tick_setter(ticks)
        lim_getter = sub_ax.get_xlim if axis == "x" else sub_ax.get_ylim
        if lim_getter()[0] > 0:
            lim_setter = sub_ax.set_xlim if axis == "x" else sub_ax.set_ylim
            lim_setter(0, None)

    plot_util.pad_axis(sub_ax, axis="x", pad_prop=0.2)
    plot_util.pad_axis(sub_ax, axis="y")

    sub_ax.spines[["right", "top"]].set_visible(False)

    sub_ax.set_title("")
    sub_ax.set_xlabel("BTSP events (pre)")
    sub_ax.set_ylabel("BTSP events (post)")

    if plot_regression:
        add_regression_line(sub_ax, BTSP_pre, BTSP_post, prop_x=0.1, prop_y=0.75)

    return sub_ax


def plot_remapping_correlation_matrices(learner, num_periods=40, approximate=True):
    """
    plot_remapping_correlation_matrices(learner)

    Plots correlation matrices showing neural correlations across time in a remapping
    experiment.

    Args:
    - learner (Learner): Learner object containing the experiment data.
    - num_periods (int or list of int, optional): Number of periods or list of number
        of periods to use for correlation matrix calculation. Default is 40.
    - approximate (bool, optional): Whether to calculate a number of periods near the
        values provided to place remapping transition between periods, instead of
        using the exact values. Default is True.

    Returns:
    - ax1D (np.ndarray of plt.Axes): Array of subplots with the plotted correlation
        matrices.
    """

    num_steps = learner.Agent.num_steps_total
    remap_step = get_learner_remap_step(learner, idx=0, num_total=1)

    BTSP_steps = learner.Pyrs.SomaticCompartment.get_BTSP_steps(
        apply_step=True, applied_only=True
    )
    pre_remap_step = BTSP_steps[BTSP_steps < remap_step].max()
    post_remap_step = BTSP_steps[BTSP_steps > remap_step].max()

    if isinstance(num_periods, int):
        num_periods = [num_periods]

    if approximate:
        num_periods_approx = list()
        for num in num_periods:
            _, denom = gen_util.get_integer_fraction(
                remap_step / num_steps,
                denom_min=int(np.min(num * 0.7)),
                denom_max=int(np.max(num * 1.3)),
            )
            if denom is not None:
                num = denom
            num_periods_approx.append(num)
        num_periods = num_periods_approx

    _, axes = plt.subplots(
        1, len(num_periods), figsize=(len(num_periods) * 3.6, 2.5), squeeze=False
    )
    ax1D = axes.ravel()
    for i, num in enumerate(num_periods):
        clabel = "Correlation" if i == len(num_periods) - 1 else ""
        learner.Pyrs.SomaticCompartment.get_firingrate_CC_matrix(
            num_periods=num, plot=True, sub_ax=ax1D[i], vmin=None, clabel=clabel
        )
        label = f"Time bins ({num})"
        ax1D[i].set_xlabel(label)
        ax1D[i].set_ylabel(label)
        ax1D[i].set_title("")

        # plot remap step
        for axline in [ax1D[i].axhline, ax1D[i].axvline]:
            axline(remap_step / num_steps * num - 0.5, color="k", lw=0.7)

        # mark time span after last BTSP (pre/post remapping)
        for start, end in [(pre_remap_step, remap_step), (post_remap_step, num_steps)]:
            x = start / num_steps * num - 0.5
            y = end / num_steps * num - 0.5
            rect = mpl_patches.Rectangle(
                (x, num_periods[i] / 30),
                y - x,
                num_periods[i] / 30,
                color="k",
                alpha=0.5,
                lw=0,
            )
            ax1D[i].add_patch(rect)

    return ax1D
