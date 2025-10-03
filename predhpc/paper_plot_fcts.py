#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpl_patches
import scipy.stats
import ratinabox

from predhpc import plot_fcts
from predhpc.experiments import metrics
from predhpc.util import gen_util, params_util, plot_util, ext_util

LW = 1.6

BTSP_ASTERISK = (5, 2, 0)  # asterisk
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


def get_PF_label(PF_type="weights", title=False):
    """
    get_PF_label(PF_type)

    Get the label for the place field type.

    Args:
    - PF_type (str, optional): Type of place field. Default is "weights".
    - title (bool, optional): Whether the label is for a title. Default is False.

    Returns:
    - label (str): Label for the place field type.
    """

    if PF_type == "history":
        label = "Neural activity"
    else:
        label = plot_fcts.get_PF_label(PF_type, title=title)

    return label


def format_1D_PF_xaxis(
    sub_ax, scale=params_util.SCALE_LINEAR, num_ticks=7, PF_type="weights"
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
        Default is "weights".
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


def configure_neural_activity_axis(
    sub_ax, ymin=0, ymax=10, norm=1, right=False, label=True
):
    """
    configure_neural_activity_axis(sub_ax)

    Configures axes for neural activity plots.

    Args:
    - sub_ax (plt.Axes): Subplot for which to configure axis.
    - ymin (float, optional): Minimum value for y axis spine. Default is 0.
    - ymax (float, optional): Maximum value for y axis spine. Default is 10.
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
    sub_ax.spines["bottom"].set_bounds(0, sub_ax.get_xlim()[1])
    if label:
        sub_ax.set_ylabel("Neural activity", labelpad=labelpad)


def plot_1D_PFs(
    PF_centers,
    PFs,
    shift=0,
    scale_y=4,
    alpha=0.8,
    shade_fact=1.0,
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
    - shade_fact (float, optional): Factor by which to adjust the shading alphas.
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

    alphas = [0.3 * shade_fact, 0.5 * shade_fact]
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


def plot_single_neuron_rate_timeseries(
    NeuronLayer, chosen_neuron=0, sub_ax=None, lw=LW
):
    """
    plot_single_neuron_rate_timeseries(NeuronLayer)

    Plots the rate timeseries of a single neuron over time.

    Args:
    - NeuronLayer (NeuronLayer): The NeuronLayer for which to plot activity.
    - chosen_neuron (int, optional): The index of the neuron to plot activity for.
        Default is 0.
    - sub_ax (plt.Axes, optional): The subplot to plot on. Default is None.
    - lw (float, optional): Line width for the plot. Default is LW.

    Returns:
    - sub_ax (plt.Axes): The axes with the plotted neuron activity.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(10, 1.7))
    NeuronLayer.plot_rate_timeseries(
        sub_ax=sub_ax,
        lw=lw,
        norm_by="none",
        mark_BTSP=False,
        chosen_neurons=[chosen_neuron],
    )

    ymin = min(sub_ax.get_ylim()[0], -0.2) - 0.2
    ymax = max(sub_ax.get_ylim()[1], 11) * 1.3
    sub_ax.set_ylim(ymin, ymax)

    NeuronLayer.add_BTSP_markers_to_plots(
        ax=sub_ax, s=BTSP_S, prop_y=0.8, lw=lw, marker=BTSP_ASTERISK, timeseries=True
    )
    sub_ax.set_ylabel("")
    plot_util.expand_ticks(sub_ax, axis="x", num_ticks=7, alternating=True, round_dec=1)
    NeuronLayer.Agent.add_position_across_time_to_plot(
        sub_ax=sub_ax, position_name="reward", alpha=0.8, y=13.6
    )

    configure_neural_activity_axis(sub_ax)
    plot_util.pad_axis(sub_ax, axis="x", pad_prop=0.015, prop_high=0)

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
        "base_s": 25,
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

    _, ax1D = plt.subplots(2, 1, figsize=(3, 3), sharex=True)

    for i, PF_type in enumerate(["weights", "history"]):
        if "weights" in PF_type:
            data = learner.get_recorded_weights()["weights"][:, 0]
            PF_centers = PCs.place_cell_centers
            color = PCs.color
            if PF_type == "smoothed_weights":
                data, PF_centers = metrics.get_smoothed_weights(
                    data, PF_centers, PCs.widths
                )
            elif PF_type != "weights":
                raise ValueError(f"PF_type '{PF_type}' not recognized.")
        else:
            t_start = ext_util.choose_t_start_after_BTSP(
                learner.Pyrs.SomaticCompartment, next_trajectory=True
            )
            data, PF_centers = metrics.evaluate_PFs(
                learner.Pyrs, method="history", t_start=t_start
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

        format_1D_PF_xaxis(ax1D[i], PF_type=PF_type)
        mark_1D_target(ax1D[i], Ag=Ag)

    plot_util.expand_ticks(
        ax1D[0], axis="y", num_ticks=5, alternating=True, round_dec=1
    )
    add_1D_position_markers(ax1D[0], Ag=Ag, y_1D=0.2)

    ax1D[0].set_xlabel("")
    ax1D[0].xaxis.set_tick_params(bottom=False)
    ax1D[0].spines["bottom"].set_visible(False)

    ax1D[1].set_xlabel("Position (m)")
    ax1D[1].set_ylabel("Neural activity")

    return ax1D


def plot_linear_binned_rates(learner, num_bins=100):
    """
    plot_linear_binned_rates(learner)

    Plots binned rates for linear experiment.

    Args:
    - learner (Learner): Learner object.
    - num_bins (int, optional): Number of bins to use for the histogram. Default is 100.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with linear binned rates plotted.
    """

    _, ax1D = plt.subplots(3, 1, figsize=(3.5, 4.8), squeeze=True)

    kwargs = {
        "num_bins": num_bins,
        "vmin": 0,
        "cbar_aspect": 10,
        "plot_occ": False,
        "mark_runs": True,
        "shared_range": True,
        "cbar_label": "Neural activity",
    }

    learner.Pyrs.plot_binned_rates(axes=ax1D.reshape(-1, 1), **kwargs)
    for sub_ax in ax1D:
        add_1D_position_markers(
            sub_ax,
            Ag=learner.Pyrs.Agent,
            y_1D=3.4,
            pos_fact=100 / 6,
            pos_shift=-0.5,
        )

    labels = ["Somatic", "Apical", "Inhibitory"]
    for i, sub_ax in enumerate(ax1D):
        sub_ax.set_title("")
        sub_ax.set_ylabel(labels[i])
    ax1D[-1].set_xlabel(f"Spatial bin ({num_bins})", labelpad=12)

    return ax1D


def retrieve_PF_data(data_dict, PF_type="weights"):
    """
    retrieve_PF_data(data_dict)

    Retrieves place field data from the given data dictionary.

    Args:
    - data_dict (dict): Dictionary containing place field data.
    - PF_type (str, optional): PF evaluation method to retrieve. Default is "weights".

    Returns:
    - PFs (2D np.ndarray): Place fields.
    - PF_centers (1D np.ndarray): Centers of the place fields.
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
            raise ValueError(f"Key '{key}' not found in data_dict.")

    PFs = data_dict[data_key]
    PF_centers = data_dict[center_key]
    PF_widths = data_dict[width_key]

    return PFs, PF_centers, PF_widths


def plot_linear_speed_PF_examples(
    speed_data, Ag=None, color=None, PF_type="weights", k=1, show_unsmoothed=True
):
    """
    plot_linear_speed_PF_examples()

    Plots examples of place fields for different speeds on the linear track. If k is
    not 1, smoothed signal is visualized.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see run_linear_speeds()).
    - Ag (Agent, optional): Agent object. If provided, it is used to add markers to
        subplot. Default is None.
    - color (str, optional): Color for PF lines. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "weights".
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
        ytick_max, ymax = 0.25, 0.28
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

            plot_fcts.plot_recorded_1D_PFs(
                PFs[i],
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
    speed_data, mark_examples=list(), color=None, PF_type="weights"
):
    """
    plot_linear_speed_PF_widths()

    Plots the relationship between speed means and place field widths for the linear
    experiment.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see run_linear_speeds()).
    - mark_examples (list): List of speed means to mark. Default is an empty list.
    - color (str, optional): Color for PF lines. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "weights".

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

    sub_ax.scatter(speed_data["speed_means"], PF_widths, s=20, alpha=0.5, color="k")

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

    # Add regression line last
    regr = scipy.stats.linregress(speed_data["speed_means"], PF_widths)
    x = np.asarray(sub_ax.get_xlim())
    y = x * regr.slope + regr.intercept
    sub_ax.plot(x, y, alpha=0.6, color="k", ls="dashed", zorder=-5, lw=LW)

    regr_str = f"y = {regr.slope:.2f}x + {regr.intercept:.2f}"
    x_text = sub_ax.get_xlim()[1] * 0.2
    y_text = np.diff(sub_ax.get_ylim()) * 0.95 + sub_ax.get_ylim()[0]
    sub_ax.text(x_text, y_text, regr_str, fontsize=12)

    regr_kwargs = {
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
            **regr_kwargs,
        )
        sub_ax.plot(
            [xmin, speed_mean],
            [width, width],
            **regr_kwargs,
        )

    return sub_ax


def plot_PF_cmap(
    PF_cmap_data,
    sub_ax=None,
    vmax=None,
    x_vals=None,
    y_vals=None,
    PF_type="weights",
    cmap_x_corr=1.0027,
    plot_colorbar=True,
    mark_maxes=True,
    keep_yticks=False,
):
    """
    plot_PF_cmap(PF_cmap_data)

    Plots a 2D place field weight map, with each row representing a place field.

    Args:
    - PF_cmap_data (2D np.ndarray): Place field weights to plot (number of place fields
        x number of inputs).
    - sub_ax (plt.Axes, optional): Subplot on which to plot the data. If None, a new
        figure and subplot are created. Default is None.
    - vmax (float, optional): Maximum value for the color scale. If None, it is set to
        the maximum value in PF_cmap_data. Default is None.
    - x_vals (1D np.ndarray, optional): X values for the place field colormap. If None,
        the x-axis is set to the range of the number of inputs. Default is None.
    - y_vals (1D np.ndarray, optional): Y values for the place field colormap. If None,
        the y-axis is set to the range of the number of place fields. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "weights".
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
    PF_type="weights",
    keep_yticks=False,
    target_positions=None,
    **arrow_kwargs,
):
    """
    plot_PF_peak_shift(PF_data, sub_ax)

    Plots the peak shifts of place fields in a 1D environment.

    Args:
    - PF_data (2D np.ndarray): Place fields to plot (number of place fields x
        number of PF positions).
    - sub_ax (plt.Axes): Subplot on which to plot the data.
    - initial_peak_idx (int, optional): Index of the initial peak in the place field
        weights. Default is 0.
    - initial_peak_value (float, optional): Value of the initial peak. Default is None.
    - s (float, optional): Size of the scatter points for the initial peak.
        Default is 10.
    - lw (float, optional): Line width for the arrows. Default is LW.
    - alpha (float, optional): Alpha value for the plot. Default is 0.7.
    - x_vals (1D np.ndarray, optional): X values for the place field weights. If None,
        the x-axis is set to the range of the number of inputs. Default is None.
    - y_vals (1D np.ndarray, optional): Y values for the place field weights. If None,
        the y-axis is set to the range of the number of place fields. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "weights".
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
    get_shift_baseline_idx(shift_data)

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
    PF_type="weights",
):
    """
    plot_linear_shift_PF_examples()

    Plots examples of place fields for different target shifts on the linear track.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see run_linear_shifts()).
    - Ag (Agent, optional): Agent object. If provided, it is used to add markers to
        subplot. Default is None.
    - color (str, optional): Color of the place field plot lines. Default is None.
    - plot_cmap (bool): Whether to plot the place field colormap instead of a peak shift
        plot. Default is False.
    - PF_type (str, optional): PF evaluation method to plot. Default is "weights".

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

    height = 1.2 * len(shifts_to_plot)
    figsize = (8.7, height) if plot_cmap else (10.01, height)
    width_ratios = [1.7, 1] if plot_cmap else [3, 1]

    _, axes = plt.subplots(
        len(shifts_to_plot),
        2,
        figsize=figsize,
        sharex=True,
        sharey="col",
        width_ratios=width_ratios,
        gridspec_kw={"wspace": 0.08, "hspace": 0.4},
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

        plot_1D_PFs(
            PF_centers,
            PFs[idx],
            scale_y=1,
            base=base,
            color=color,
            sub_ax=axes[i, 0],
            alpha=alpha,
            mark_original=True,
        )
        axes[i, 0].text(
            1.8, ymax * 0.78, f"{target_shift:.1f} m shift", ha="center", fontsize=12
        )

        if Ag is not None:
            mark_1D_target(axes[i, 0], Ag=Ag, target_shift=target_shift)

        mark_1D_target(axes[i, 1], Ag=Ag, target_shift=target_shift)
        if plot_cmap:
            plot_PF_cmap(
                PFs[idx],
                sub_ax=axes[i, 1],
                vmax=vmax,
                x_vals=PF_centers,
                PF_type=PF_type,
                plot_colorbar=False,
            )

        else:
            plot_PF_peak_shift(
                PFs[idx].reshape(1, -1),
                sub_ax=axes[i, 1],
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

    for i, sub_ax in enumerate(axes.T.ravel()):
        format_1D_PF_xaxis(sub_ax, PF_type=PF_type)
        if i != axes.size // 4:
            sub_ax.set_ylabel("")

    for sub_ax in axes[:, 1]:
        sub_ax.set_ylim(-0.8, 0.8)

    if Ag is not None:
        for i, target_shift in enumerate(shifts_to_plot):
            for j, y_1D in enumerate([ymax * 0.95, 0.8]):
                add_1D_position_markers(
                    axes[i, j], Ag=Ag, y_1D=y_1D, target_shift=target_shift
                )

    for ax1D in axes.T:
        plot_util.clear_bottom(ax1D[:-1])

    return axes


def plot_target_shift_PFs(
    shift_data,
    Ag=None,
    mark_examples=list(),
    plot_cmap=False,
    color=None,
    PF_type="weights",
):
    """
    plot_target_shift_PFs()

    Plots the relationship between target shifts and place field weights for the linear
    experiment.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see run_linear_shifts()).
    - Ag (Agent, optional): Agent object. If provided, it is used to add markers to
        subplot. Default is None.
    - mark_examples (list): List of target shifts to plot arrows for.
        Default is an empty list.
    - plot_cmap (bool): Whether to plot the place field colormap instead of a peak shift
        plot. Default is False.
    - color (str): Color to use for place cell plots. Default is None.
    - PF_type (str, optional): PF evaluation method to plot. Default is "weights".

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with target shifts and place field
        weights plotted.
    """

    width_ratios = [1.2, 1] if plot_cmap else [3, 1]
    _, axes = plt.subplots(
        1,
        2,
        figsize=(10.01, 4.8),
        sharey=True,
        squeeze=False,
        width_ratios=width_ratios,
        gridspec_kw={"wspace": 0.07, "hspace": 0.4},
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

        # ax1D[1].plot(
        #     target_positions,
        #     shift_data["target_shifts"],
        #     color="k",
        #     lw=LW,
        #     alpha=0.8,
        #     ls="dotted",
        #     zorder=-5,
        # )

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


def plot_openfield_components(Pyrs, titles=False, BTSP_trajectory=True):
    """
    plot_openfield_components(Pyrs)

    Plots the openfield components for an experiment.

    Args:
    - Pyrs (Pyr): Pyr object containing the environment, agent, object and place cells.
    - titles (bool, optional): Whether to add titles to each subplot. Default is False.

    Returns:
    - axes (np.ndarray of plt.Axes): Array of subplots with openfield components plotted.
    """

    _, axes = plt.subplots(
        2, 2, figsize=(3.8, 3.8), gridspec_kw={"wspace": 0.015, "hspace": 0.19}
    )
    Env, Ag, PCs, Objs = ext_util.extract_objects_from_Pyrs(Pyrs)
    if Env.D != 2:
        raise ValueError("2D plotting is only supported for 2D environments.")

    kwargs = {"skip_object_types": ["teleport"], "no_legend": True}
    y = 1.02
    chosen_PCs = [206]

    env_sub_ax, traj_sub_ax = axes[0]
    Obj_sub_ax, PC_sub_ax = axes[1]

    # top row
    if titles:
        env_sub_ax.set_title("Environment", y=y)
    Env.plot_environment(
        sub_ax=env_sub_ax, scale_loc=(1.62, 1.88), scale_length=0.5, **kwargs
    )
    if BTSP_trajectory:
        BTSP_times = Pyrs.SomaticCompartment.get_BTSP_steps() * Ag.dt
        if len(BTSP_times) == 0:
            raise RuntimeError("No BTSP events found for Pyrs.SomaticCompartment.")
        pre, post = Pyrs.SomaticCompartment.get_estimated_num_steps_pre_post_BTSP(
            as_time=True
        )
        kwargs["t_start"] = max(0, BTSP_times[0] - pre)
        kwargs["t_end"] = BTSP_times[0] + post
        kwargs["colormap"] = Pyrs.SomaticCompartment.get_BTSP_kernel_based_cmap(
            t_pre=pre, t_post=post
        )
    else:
        kwargs["traj_idxs"] = [0]

    if titles:
        title = "Trajectory used for BTSP" if BTSP_trajectory else "Example trajectory"
        traj_sub_ax.set_title(title, y=y)

    Ag.plot_trajectories(
        ax=traj_sub_ax,
        framerate=8,
        alpha=0.4,
        s_2D=5,
        cmap_per=True,
        plot_target=False,
        plot_agent=False,
        **kwargs,
    )

    # bottom row
    if titles:
        Obj_sub_ax.set_title("Object field", y=y, color=Objs.color)
    Objs.plot_rate_map(sub_ax=Obj_sub_ax, plot_objects=False, colorbar=False, **kwargs)

    if titles:
        PC_sub_ax.set_title(
            f"Place fields ({len(chosen_PCs)}/{PCs.n})", y=y, color=PCs.color
        )
    Env.plot_environment(sub_ax=PC_sub_ax, plot_objects=True, **kwargs)
    plot_fcts.plot_overlayed_rate_maps(
        PCs, sub_ax=PC_sub_ax, method="max", chosen_neurons=chosen_PCs, colorbar=False
    )
    PCs.plot_place_cell_locations(
        sub_ax=PC_sub_ax, plot_objects=True, s=3, alpha=0.8, marker=".", **kwargs
    )

    cbar_axlist = [Obj_sub_ax, PC_sub_ax]
    vmin = min([sub_ax.get_images()[-1].get_clim()[0] for sub_ax in cbar_axlist])
    vmax = max([sub_ax.get_images()[-1].get_clim()[1] for sub_ax in cbar_axlist])
    for sub_ax in cbar_axlist:
        im = sub_ax.get_images()[-1]
        im.set_clim(vmin, vmax)

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


def get_s_openfield_PFs(fig_side=4.0, PF_type="weights"):
    """
    get_s_openfield_PFs(fig_side=4.0, num_x=60)

    Returns the place field square size (s) needed to contiguously plot openfield place
    fields without overlap, based on the figure size and type of place field to plot.

    For weights, expects the place field plot to consist of a square grid of 60x60
    input place fields. For history, expects a square grid of 120x120 positions.

    Sizes were identified manually.

    Args:
    - fig_side (float, optional): Size of the figure. Default is 4.
    - PF_type (str, optional): PF evaluation method to plot. Default is "weights".

    Returns:
    - s (float): Suggested starting estimate for the place field square size.
    """

    approx = False
    if "weights" in PF_type:
        if fig_side == 1:
            s = 10.2
        elif fig_side == 2:
            s = 42.8
        elif fig_side == 3:
            s = 99
        elif fig_side == 4:
            s = 176.3
        elif fig_side == 5:
            s = 278
        else:
            s = max(1, 66.8 * np.exp(fig_side * 0.34) - 85.6)
            approx = True

    elif PF_type == "history":
        if fig_side == 1:
            s = 2.55
        elif fig_side == 2:
            s = 10.7
        elif fig_side == 3:
            s = 24.5
        elif fig_side == 4:
            s = 44.08
        elif fig_side == 5:
            s = 69
        else:
            s = max(1, 17 * np.exp(fig_side * 0.34) - 21.8)
            approx = True

    else:
        raise NotImplementedError(
            f"No proposed place field square size available for {PF_type} PF_type."
        )

    if approx:
        print(
            f"No place field square size recorded for fig_side of {fig_side}. "
            f"Starting estimate of {s} suggested."
        )

    return s


def plot_openfield_PFs(Pyrs, fig_side=4.0, lw=LW, alpha=0.8, PF_type="weights"):
    """
    plot_openfield_PFs(Pyrs)

    Plots the place field of the Pyr neuron in the openfield corridor.

    Args:
    - Pyrs (Pyr): Pyr object for openfield.
    - fig_side (float, optional): Size of the figure. Default is 3.3.
    - lw (float, optional): Line width. Default is LW.
    - alpha (float, optional): Transparency level. Default is 0.8.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted weights.
    """

    fig, sub_ax = plt.subplots(figsize=(fig_side, fig_side))

    if Pyrs.SomaticCompartment.n != 1:
        raise ValueError("Only single neuron plotting is supported.")

    PF_t_start = None
    if PF_type == "history":
        round_dec = 0
        BTSP_applied = Pyrs.SomaticCompartment.get_BTSP_steps(
            applied_only=True, apply_step=True
        )
        vmax = 10
        if len(BTSP_applied):
            PF_t_start = BTSP_applied[-1] * Pyrs.SomaticCompartment.Agent.dt
    else:
        round_dec = 2
        vmax = None

    s = get_s_openfield_PFs(fig_side=fig_side, PF_type=PF_type)

    plot_fcts.plot_2D_PFs(
        Pyrs.SomaticCompartment,
        PF_type=PF_type,
        PF_t_start=PF_t_start,
        alpha=alpha,
        s=s,
        obj_s=30,
        BTSP_s=30,
        BTSP_marker=(5, 2, 0),
        lw=lw,
        round_dec=round_dec,
        plot_BTSP_events=True,
        skip_object_types=["teleport"],
        no_legend=True,
        vmax=vmax,
        ax=sub_ax,
        marker="s",
        cbar_side="right",
        cbar_outline=True,
        vmin=0,
    )

    cax = fig.axes[-1]
    clabel = get_PF_label(PF_type, title=False)
    cax.set_ylabel(clabel)

    return sub_ax
