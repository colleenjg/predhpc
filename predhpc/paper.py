#!/usr/bin/env python3

import copy
from pathlib import Path
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpl_patches
import scipy.stats
import ratinabox
from tqdm import tqdm

from predhpc import run_manager, plot_fcts
from predhpc.util import gen_util, params_util, plot_util, ext_util
from predhpc.experiments import metrics

PAPER_SEED = 18
gen_util.seed_all(PAPER_SEED)

LW = 1.6


def suppress_warnings():
    """
    suppress_warnings()

    Suppress expected warnings.
    """

    warnings.filterwarnings("ignore", message="solid 1D boundary", category=UserWarning)
    warnings.filterwarnings(
        "ignore", message="invalid value encountered", category=RuntimeWarning
    )


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
    - notebook (bool): Whether to stylize for a Jupyter notebook. Default is False.
    """

    ratinabox.autosave_plots = False
    ratinabox.figure_directory = str(Path(gen_dir, "results", "paper"))
    suppress_warnings()
    stylize_plots_for_paper(notebook=notebook)


def format_1D_PF_xaxis(sub_ax, scale=params_util.SCALE_LINEAR, num_ticks=7):
    """
    format_1D_PF_xaxis(sub_ax)

    Formats the x-axis of a 1D place field plot.

    Args:
    - sub_ax (matplotlib.axes.Axes): The axes to format.
    - scale (float, optional): The scale of the environment.
        Default is params_util.SCALE_LINEAR.
    - num_ticks (int, optional): The number of ticks to display on the x-axis.
        Default is 7.
    """

    sub_ax.set_xlim([0, scale])
    sub_ax.set_xticks([0, scale])
    sub_ax.spines[["top", "right"]].set_visible(False)
    plot_util.expand_ticks(
        sub_ax, axis="x", num_ticks=num_ticks, alternating=True, round_dec=0
    )
    sub_ax.set_xlabel("Input place field center")


def mark_1D_target(sub_ax, Ag=None, target_shift=0, alpha=0.8):
    """
    mark_1D_target(sub_ax)

    Adds vertical dashed line for target position on the 1D track to a subplot.

    Args:
    - sub_ax (plt.Axes): Subplot on which to add positions.
    - Ag (Agent, optional): Agent object to plot. If None, a new Agent object is created.
        Default is None.
    - target_shift (float, optional): Shift to apply to the target position.
        Default is 0.
    - alpha (float, optional): Alpha value for the vertical line. Default is 0.8.
    """
    if Ag is None:
        Pyrs = get_linear_Pyrs()
        _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    target_pos = Ag.get_position("target", dim_idx=0)
    positions = [target_pos]
    if target_shift != 0:
        positions.append(target_pos + target_shift)

    for i, pos in enumerate(positions):
        use_alpha = alpha if i == len(positions) - 1 else alpha / 2
        sub_ax.axvline(pos, ls="dotted", color="k", alpha=use_alpha, zorder=-5, lw=LW)


def add_1D_position_markers(
    sub_ax,
    Ag=None,
    y_1D=1.0,
    base_s=30,
    pos_shift=0,
    target_shift=0,
    target_alpha=1.0,
    **kwargs,
):
    """
    add_1D_position_markers(sub_ax)

    Adds markers for positions along the 1D track to a subplot.

    Args:
    - sub_ax (plt.Axes): Subplot on which to add positions.
    - Ag (Agent, optional): Agent object to plot. If None, a new Agent object is created.
        Default is None.
    - y_1D (float, optional): Y-coordinate for the 1D positions. Default is 1.0.
    - base_s (float, optional): Base size for the 1D positions. Default is 30.
    - pos_shift (float, optional): Shift to apply to the position. Default is 0.
    - target_shift (float, optional): Shift to apply to the target position.
        Default is 0.
    - alpha (float, optional): Alpha value for the position markers. Default is 1.0.

    Keyword args:
    - **kwargs: Keyword arguments passed to Ag.add_positions_spatially_to_plot()
    """

    if Ag is None:
        Pyrs = get_linear_Pyrs()
        _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

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
            **kwargs,
        )


def plot_1D_PF_weights(
    PCs, weights, shift=0, scale_y=4, alpha=0.8, lw=LW, base=None, sub_ax=None
):
    """
    plot_1D_PF_weights(PCs, weights)

    Plots the weights of place cells.

    Args:
    - PCs (PlaceCells): PlaceCells object containing place cell data.
    - weights (1D np.ndarray): Weights to plot.
    - shift (float, optional): Shift to apply to the weights. Default is 0.
    - scale_y (float, optional): Scale factor for the y-axis. Default is 4.
    - alpha (float, optional): Alpha value for the plot line. Default is 0.8.
    - lw (float, optional): Line width for the plot line. Default is LW.
    - base (1D np.ndarray, optional): Base weights to compare against. Default is None.
    - sub_ax (plt.Axes, optional): Axes to plot on. If None, a new figure and subplot
        are created. Default is None.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted weights.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(5, 4))

    PC_centers = PCs.place_cell_centres[:, 0]
    plot_weights = weights * scale_y + shift

    if base is not None:
        base = base * scale_y + shift
        diff = plot_weights - base
        for i, color in enumerate([PCs.color, "royalblue"]):
            diff = -diff if i == 1 else diff
            cond = plot_util.get_greater_condition_for_fill_between(diff)
            sub_ax.fill_between(
                PC_centers, base, plot_weights, color=color, alpha=0.3, lw=0, where=cond
            )

    sub_ax.plot(PC_centers, plot_weights, color=PCs.color, alpha=alpha, lw=lw)
    return sub_ax


def get_linear_Pyrs(
    scale=params_util.SCALE_LINEAR,
    speed_mean=params_util.SPEED_MEAN_LINEAR,
    speed_std=params_util.SPEED_STD,
    wait_at_end=int(15 / params_util.DT),
    log_BTSP=True,
    seed=True,
):
    """
    get_linear_Pyrs()

    Initializes Pyr parameters for linear environment.

    Args:
    - scale (float): Scale of the environment. Default is params_util.SCALE_LINEAR.
    - speed_mean (float): Mean speed of the agent. Default is
        params_util.SPEED_MEAN_LINEAR.
    - speed_std (float): Standard deviation of the agent's speed. Default is
        params_util.SPEED_STD.
    - wait_at_end (int): Number of steps to wait at the end of the environment.
        Default is 15 seconds converted to steps.
    - log_BTSP (bool): Whether to log BTSP events. Default is True.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - Pyrs (Pyr): Pyr object initialized with the specified parameters.
    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

    target_position = params_util.get_target_position()

    env_params = params_util.get_env_params(
        environment="linear",
        scale=scale,
    )

    agent_params = params_util.get_agent_params(
        environment="linear",
        scale=scale,
        target_position=target_position,
        speed_mean=speed_mean,
        speed_std=speed_std,
        wait_at_end=wait_at_end,
    )

    PC_params = params_util.get_PC_params(
        environment="linear",
    )

    Pyr_params = params_util.get_Pyr_params(
        environment="linear",
        log_BTSP=log_BTSP,
    )

    Obj_params = params_util.get_Obj_params(
        environment="linear",
    )

    Pyrs = run_manager.init_env_objects(
        env_params=env_params,
        agent_params=agent_params,
        PC_params=PC_params,
        Pyr_params=Pyr_params,
        Obj_params=Obj_params,
        environment="linear",
        plot=False,
    )

    return Pyrs


def plot_environment(Ag=None):
    """
    plot_environment()

    Plots the environment for the linear experiment.

    Args:
    - Ag (Agent, optional): Agent object to plot. If None, a new Agent object is created.
        Default is None.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted environment.
    """

    if Ag is None:
        Pyrs = get_linear_Pyrs()
        _, Ag, _, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    _, sub_ax = plt.subplots(figsize=(6.2, 0.8))
    sub_ax = plot_fcts.plot_1D_reset_environment(
        Ag, minimalist=True, title="", base_s=50, sub_ax=sub_ax
    )
    sub_ax.spines["bottom"].set_linewidth(2.5)
    leg = sub_ax.get_legend()
    for text in leg.get_texts():
        text_str = text.get_text()
        add_str = "object" if text_str == "target" else "position"
        text.set_text(f"{text_str} {add_str}")
        text.set_fontsize(12)

    return sub_ax


def plot_BTSP_kernel(Pyrs=None, xlims=None):
    """
    plot_BTSP_kernel()

    Plots the BTSP kernel for the given Pyrs object.

    Args:
    - Pyrs (Pyr): Pyr object containing the agent and place cells.

    Returns:
    - sub_ax (plt.Axes): The subplot with the plotted BTSP kernel.
    """

    if Pyrs is None:
        Pyrs = get_linear_Pyrs()

    _, Ag, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    _, sub_ax = plt.subplots(figsize=(6, 2.28))
    BTSP_param_dict = params_util.get_default_BTSP_filter_param_dict(
        incl_BTSP_str=False
    )

    if xlims is None:
        xlims = [-10, 10]

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


def run_linear(Pyrs=None, max_num_steps=3800, max_time_min=None, seed=True, **kwargs):
    """
    run_linear()

    Runs a linear environment with the specified Pyr parameters.

    Args:
    - Pyrs (Pyr, optional): Pyr object with initialized parameters. If None,
        a new Pyr object is created with default parameters.
    - max_num_steps (int): Maximum number of steps to run the environment.
        Default is 3800.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is None.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Keyword Args:
    - **kwargs: Additional keyword arguments passed to run_manager.learn_1D_BTSP().

    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

    if Pyrs is None:
        Pyrs = get_linear_Pyrs()

    if max_time_min is not None:
        max_num_steps = int(max_time_min * 60 / Pyrs.Agent.dt)

    learner = run_manager.learn_1D_BTSP(
        Pyrs, BTSP_on=0, max_num_steps=max_num_steps, plot=False, **kwargs
    )

    return learner


def plot_linear_summary(learner=None, max_time_min=1.8):
    """
    plot_linear_summary()

    Plots summary of linear experiment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is 1.8.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with linear data plotted.
    """

    if learner is None:
        Pyrs = get_linear_Pyrs()
        learner = run_linear(Pyrs, max_time_min=max_time_min)

    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(learner.Pyrs)

    plot_kwargs = {
        "hspace": 0.35,
        "height_ratios": [0.1, 0.12, 0.36, 0.14, 0.14, 0.14],
        "figsize": (6.1, 8.3),
        "lw": LW,
        "s": 1.2,
        "base_s": 25,
    }

    Pyr_kwargs = {
        "norm_by": "shared_max",
        "BTSP_s": 40,
        "separate_axes": True,
        "no_legend": True,
    }

    axes = plot_fcts.plot_1D_time_info(
        learner.Pyrs, Pyrs_spikes=False, Pyr_kwargs=Pyr_kwargs, **plot_kwargs
    )
    ax1D = axes[:, 0]

    ax1D[0].set_yticks([0, 3, 6])
    plot_util.expand_ticks(
        ax1D[-1], axis="x", num_ticks=9, alternating=True, round_dec=1
    )
    plot_util.pad_axis(ax1D[0], axis="x", pad_prop=0.02, prop_high=1.0)

    titles = [
        "Trajectories",
        "Object cell",
        f"Place cells ({PCs.n})",
        "Pyramidal neuron",
        "",
        "",
    ]
    for i, title in enumerate(titles):
        ax1D[i].set_title(title, y=1.06)
        if i == 3:
            ax1D[i].set_ylabel("Neural activity", labelpad=12)
        elif i > 0:
            ax1D[i].set_ylabel("")

    for i, comp in enumerate(["soma", "dend", "inhibit"]):
        learner.Pyrs.add_compartment_legend(
            ax1D[3 + i],
            compartment=comp,
            lw=plot_kwargs["lw"],
            loc=(0.72, 0.8),
            frameon=False,
            fontsize=11,
        )
        ax1D[3 + i].set_ylim(0.95, None)

    return ax1D


def plot_linear_PFs(learner, max_time_min=1.8):
    """
    plot_linear_PFs(learner)

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is 1.8.

    Returns:
    - sub_ax (plt.Axes): The subplot with the linear place fields plotted.
    """

    if learner is None:
        Pyrs = get_linear_Pyrs()
        learner = run_linear(Pyrs, max_time_min=max_time_min)

    _, Ag, PCs, _ = ext_util.extract_objects_from_Pyrs(learner.Pyrs)

    _, sub_ax = plt.subplots(figsize=(3, 2))

    plot_fcts.plot_recorded_1D_input_place_cell_weights(
        learner.get_recorded_weights()["weights"][:, 0],
        PCs.place_cell_centres,
        color=PCs.color,
        marker="none",
        lw=LW,
        plot_last_FWHM=False,
        sub_ax=sub_ax,
    )

    format_1D_PF_xaxis(sub_ax)
    plot_util.expand_ticks(sub_ax, axis="y", num_ticks=5, alternating=True, round_dec=1)
    add_1D_position_markers(sub_ax, Ag=Ag, y_1D=0.2)
    mark_1D_target(sub_ax, Ag=Ag)

    return sub_ax


def plot_linear_binned_rates(learner, num_bins=100, max_time_min=1.8):
    """
    plot_linear_binned_rates(learner)

    Plots binned rates for linear experiment.

    Args:
    - learner (Learner): Learner object.
    - max_time_min (float, optional): Maximum time in minutes to run the environment.
        If specified, it overrides max_num_steps based on the agent's time step.
        Default is 1.8.

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with linear binned rates plotted.
    """

    if learner is None:
        Pyrs = get_linear_Pyrs()
        learner = run_linear(Pyrs, max_time_min=max_time_min)

    _, ax1D = plt.subplots(3, 1, figsize=(3.7, 5.6), squeeze=True)

    kwargs = {
        "num_bins": num_bins,
        "vmin": 0,
        "cbar_aspect": 10,
        "plot_occ": False,
        "mark_runs": True,
        "shared_range": True,
    }

    learner.Pyrs.plot_binned_rates(axes=ax1D.reshape(-1, 1), **kwargs)
    for sub_ax in ax1D:
        add_1D_position_markers(
            sub_ax,
            learner.Pyrs.Agent,
            y_1D=3.4,
            pos_fact=100 / 6,
            pos_shift=-0.5,
        )

    labels = ["Soma", "Apical dend.", "Dend. inhib."]
    for i, sub_ax in enumerate(ax1D):
        sub_ax.set_title("")
        sub_ax.set_ylabel(labels[i])
    ax1D[-1].set_xlabel(f"Spatial bin ({num_bins})", labelpad=12)

    return ax1D


def run_linear_speeds(seed=True):
    """
    run_linear_speeds()

    Runs a linear environment with varying speeds and collects data on place field
    widths and weights.

    Args:
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - speed_data (dict): Dictionary containing:
        - "speed_means": Array of speed means used in the experiment.
        - "PF_widths": List of place field widths for each speed mean.
        - "PF_weights": List of place field weights for each speed mean.
    """

    speed_data = {
        "speed_means": gen_util.get_rounded_linspace(0.05, 0.4, 29),
        "PF_widths": list(),
        "PF_weights": list(),
    }
    for speed_mean in tqdm(speed_data["speed_means"]):
        if seed:
            gen_util.seed_all(PAPER_SEED)
        Pyrs = get_linear_Pyrs(
            speed_mean=speed_mean,
            speed_std=0,
            log_BTSP=False,
            wait_at_end=0,
            seed=False,
        )
        learner = run_linear(Pyrs, max_time_min=1.2, no_logs=True, seed=False)
        speed_data["PF_widths"].append(metrics.compute_PC_FWHM(Pyrs))
        speed_data["PF_weights"].append(learner.get_recorded_weights()["weights"][:, 0])

    speed_data = {key: np.asarray(val) for key, val in speed_data.items()}

    return speed_data


def plot_linear_speed_PF_examples(speed_data=None, to_plot=[0.15, 0.35]):
    """
    plot_linear_speed_PF_examples()

    Plots examples of place fields for different speeds on the linear track.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see run_linear_speeds()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - to_plot (list): List of speed means to plot. Default is [0.15, 0.35]

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with example place fields plotted.
    """

    if speed_data is None:
        speed_data = run_linear_fct("linear_speeds", overwrite=False)

    Pyrs = get_linear_Pyrs()
    _, Ag, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    _, axes = plt.subplots(
        len(to_plot),
        1,
        figsize=(5.7, 3.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    ax1D = axes[:, 0]

    for i, speed_mean in enumerate(to_plot):
        sub_ax = ax1D[i]
        if speed_mean not in speed_data["speed_means"]:
            raise RuntimeError(f"{speed_mean} speed mean not found in data dictionary.")
        idx = np.where(speed_data["speed_means"] == speed_mean)[0][0]
        plot_fcts.plot_recorded_1D_input_place_cell_weights(
            speed_data["PF_weights"][idx],
            PCs.place_cell_centres,
            color=PCs.color,
            marker="none",
            plot_last_FWHM=True,
            lw=LW,
            no_legend=True,
            sub_ax=sub_ax,
        )
        sub_ax.text(5, 0.223, f"{to_plot[i]} m/s", ha="center", fontsize=12)

        width = speed_data["PF_widths"][idx]
        width = np.around(width, 2)
        rect = mpl_patches.Rectangle(
            (4.59, 0.17), 0.8, 0.043, color=PCs.color, alpha=0.3, lw=0
        )
        sub_ax.add_patch(rect)
        sub_ax.text(5, 0.18, f"{width} m", ha="center", fontsize=12)

    for sub_ax in ax1D:
        sub_ax.set_ylim([0, 0.28])
        sub_ax.set_yticks([0, 0.25])
        plot_util.expand_ticks(
            sub_ax, axis="y", num_ticks=6, alternating=True, round_dec=2
        )
        mark_1D_target(sub_ax, Ag=Ag)
        format_1D_PF_xaxis(sub_ax)

    ax1D[0].set_xlabel("")
    add_1D_position_markers(ax1D[0], Ag, y_1D=0.27)

    return ax1D


def plot_linear_speed_PF_widths(speed_data=None, examples=[0.15, 0.35]):
    """
    plot_linear_speed_PF_widths()

    Plots the relationship between speed means and place field widths for the linear
    experiment.

    Args:
    - speed_data (dict): Dictionary containing speed-related data
        (see run_linear_speeds()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - examples (list): List of speed means to plot lines for. Default is [0.15, 0.35].

    Returns:
    - sub_ax (plt.Axes): The subplot with the speed means and place field widths
        plotted.
    """

    if speed_data is None:
        speed_data = run_linear_fct("linear_speeds", overwrite=False)

    Pyrs = get_linear_Pyrs()  # with default parameters
    _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    _, sub_ax = plt.subplots(figsize=(3.3, 3.3))

    sub_ax.scatter(
        speed_data["speed_means"], speed_data["PF_widths"], s=20, alpha=0.5, color="k"
    )

    # Format axes
    for axis in ["x", "y"]:
        plot_util.pad_axis(sub_ax=sub_ax, axis=axis)
    sub_ax.spines[["top", "right"]].set_visible(False)

    sub_ax.set_xlabel("Speed mean (m/s)")
    sub_ax.set_xticks([0.05, 0.4])
    plot_util.expand_ticks(
        sub_ax, axis="x", num_ticks=8, alternating=True, round_dec=2, start_idx=1
    )

    sub_ax.set_ylabel("Place field width (m)")
    sub_ax.set_yticks([0.2, 0.7])
    plot_util.expand_ticks(
        sub_ax, axis="y", num_ticks=6, alternating=True, round_dec=1, start_idx=1
    )

    # Add regression line last
    regr = scipy.stats.linregress(speed_data["speed_means"], speed_data["PF_widths"])
    x = np.asarray(sub_ax.get_xlim())
    y = x * regr.slope + regr.intercept
    sub_ax.plot(x, y, alpha=0.6, color="k", ls="dashed", zorder=-5, lw=LW)

    regr_str = f"y = {regr.slope:.2f}x + {regr.intercept:.2f}"
    sub_ax.text(0.11, 0.7, regr_str, fontsize=12)

    regr_kwargs = {
        "color": PCs.color,
        "ls": "dotted",
        "lw": LW,
        "alpha": 1.0,
        "zorder": -5,
    }

    xmin = sub_ax.get_xlim()[0]
    ymin = sub_ax.get_ylim()[0]
    for speed_mean in examples:
        if speed_mean not in speed_data["speed_means"]:
            raise RuntimeError(f"{speed_mean} speed mean not found in data dictionary.")
        idx = np.where(speed_data["speed_means"] == speed_mean)[0][0]
        width = speed_data["PF_widths"][idx]
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


def run_linear_shifts(seed=True):
    """
    run_linear_shifts()

    Runs a linear environment with varying target position shifts and collects data
    on place field widths and weights.

    Args:
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - shift_data (dict): Dictionary containing:
        - "target_shifts": Array of target position shifts used in the experiment.
        - "PF_widths": List of place field widths for each target shift.
        - "PF_weights": List of place field weights for each target shift.
    """

    if seed:
        gen_util.seed_all(PAPER_SEED)

    shift_data = {
        "target_shifts": gen_util.get_rounded_linspace(-3.6, 2.4, 61),
        "PF_widths": list(),
        "PF_weights": list(),
    }

    Pyrs = get_linear_Pyrs(speed_std=0, log_BTSP=False, wait_at_end=0)
    orig_learner = run_linear(Pyrs, max_time_min=1.2, no_logs=True)
    for shift in tqdm(shift_data["target_shifts"]):
        if seed:
            gen_util.seed_all(PAPER_SEED)
        learner = copy.deepcopy(orig_learner)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="Target position"
            )
            learner.Pyrs.Agent.move_target_position(shift)
        learner = run_linear(learner, max_time_min=1.2, no_logs=True)
        shift_data["PF_widths"].append(metrics.compute_PC_FWHM(Pyrs))
        shift_data["PF_weights"].append(learner.get_recorded_weights()["weights"][:, 0])

    num = max([len(wei) for wei in shift_data["PF_weights"]])
    shift_data["PF_weights"] = [
        np.pad(wei, ((0, num - len(wei)), (0, 0)), "constant", constant_values=np.nan)
        for wei in shift_data["PF_weights"]
    ]
    shift_data = {key: np.asarray(val) for key, val in shift_data.items()}

    return shift_data


def plot_PF_cmap(
    PF_cmap_data,
    sub_ax=None,
    vmax=None,
    x_vals=None,
    y_vals=None,
    cmap_x_corr=1.0027,
    plot_colorbar=True,
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
    - cmap_x_corr (float, optional): Correction factor along the x axis for plotting
        scatterplot elements. Default is 1.0022.
    - plot_colorbar (bool, optional): Whether to plot a colorbar. Default is True.
    - keep_yticks (bool, optional): Whether to keep the y-ticks on the plot.
        Default is False.

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

    scatter_kwargs = {"color": "k", "s": 8, "alpha": 0.6}

    if len(x_vals) > 1:
        horiz_max = plot_util.get_cmap_max(PF_cmap_data, axis=1, indexed=x_vals)
        sub_ax.scatter(horiz_max * cmap_x_corr, y_vals, marker="_", **scatter_kwargs)

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
        cbar.set_label("Input weight")
        cbar.ax.yaxis.set_label_position("left")

    sub_ax.set_xlabel("Input place field center")

    return sub_ax


def plot_linear_shift_PF_examples(shift_data=None, to_plot=[0.4, -2]):
    """
    plot_linear_shift_PF_examples()

    Plots examples of place fields for different target shifts on the linear track.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see run_linear_shifts()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - to_plot (list): List of target shifts to plot. Default is [0.4, -2].

    Returns:
    - axes (2D np.ndarray of plt.Axes): Subplots with example place fields plotted.
    """

    if shift_data is None:
        shift_data = run_linear_fct("linear_shifts", overwrite=False)

    Pyrs = get_linear_Pyrs()
    _, Ag, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    _, axes = plt.subplots(
        len(to_plot),
        2,
        figsize=(8.7, 2.4),
        sharex="col",
        sharey="col",
        width_ratios=[1.7, 1],
        gridspec_kw={"wspace": 0.06, "hspace": 0.4},
        squeeze=False,
    )

    base = shift_data["PF_weights"][0][1]
    vmax = np.nanmax(shift_data["PF_weights"])
    for i, target_shift in enumerate(to_plot):
        if target_shift not in shift_data["target_shifts"]:
            raise RuntimeError(f"{target_shift} shift not found in data dictionary.")
        idx = np.where(shift_data["target_shifts"] == target_shift)[0][0]
        plot_1D_PF_weights(
            PCs,
            shift_data["PF_weights"][idx][-1],
            scale_y=1,
            base=base,
            sub_ax=axes[i, 0],
        )
        axes[i, 0].text(5.0, 0.2, f"{to_plot[i]} m shift", ha="center", fontsize=12)

        data = shift_data["PF_weights"][idx : idx + 1, -1]
        plot_PF_cmap(
            data,
            sub_ax=axes[i, 1],
            vmax=vmax,
            x_vals=PCs.place_cell_centres[:, 0],
            plot_colorbar=False,
        )
        for sub_ax in axes[i]:
            mark_1D_target(sub_ax, Ag=Ag, target_shift=target_shift)

    for sub_ax in axes[:, 0]:
        sub_ax.set_ylim([0, 0.28])
        sub_ax.set_yticks([0, 0.25])
        sub_ax.set_ylabel("")
        plot_util.expand_ticks(
            sub_ax, axis="y", num_ticks=3, alternating=True, round_dec=2
        )
        format_1D_PF_xaxis(sub_ax)

    axes[-1, 0].set_ylabel("Input weight", va="top", labelpad=15)
    for sub_ax in axes[:, 1]:
        sub_ax.set_ylim(-0.8, 0.8)

    for i, target_shift in enumerate(to_plot):
        add_1D_position_markers(axes[i, 0], Ag=Ag, y_1D=0.27, target_shift=target_shift)
        add_1D_position_markers(axes[i, 1], Ag=Ag, y_1D=0.8, target_shift=target_shift)

    for ax1D in axes.T:
        plot_util.clear_bottom(ax1D[:-1])

    return axes


def plot_target_shift_PFs(shift_data=None, examples=[0.4, -2]):
    """
    plot_target_shift_PFs()

    Plots the relationship between target shifts and place field weights for the linear
    experiment.

    Args:
    - shift_data (dict): Dictionary containing target shift-related data
        (see run_linear_shifts()). If not provided, data is loaded or experiment is run
        from scratch. Default is None.
    - examples (list): List of target shifts to plot arrows for. Default is [0.4, -2].

    Returns:
    - ax1D (1D np.ndarray of plt.Axes): Subplots with target shifts and place field
        weights plotted.
    """

    if shift_data is None:
        shift_data = run_linear_fct("linear_shifts", overwrite=False)

    Pyrs = get_linear_Pyrs()
    _, Ag, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)

    _, ax1D = plt.subplots(
        1,
        2,
        figsize=(10.01, 4),
        width_ratios=[1.2, 1],
        gridspec_kw={"wspace": 0.05},
        sharey=True,
    )

    base = shift_data["PF_weights"][0][1]
    base_shift = np.nanmin(base)
    base_shifted = base - base_shift
    plot_1D_PF_weights(PCs, base_shifted, lw=1.8, sub_ax=ax1D[0])

    for i, shift in enumerate(shift_data["target_shifts"]):
        last_shifted = shift_data["PF_weights"][i, -1] - base_shift
        if np.isfinite(last_shifted.all()):
            plot_1D_PF_weights(
                PCs,
                last_shifted,
                shift=shift,
                base=base_shifted,
                lw=1.0,
                sub_ax=ax1D[0],
            )
    format_1D_PF_xaxis(ax1D[0])
    ax1D[0].set_ylabel("Target object shift", labelpad=13)

    ymin = shift_data["target_shifts"].min()
    ymax = shift_data["target_shifts"].max()
    yticks = [y for y in ax1D[0].get_yticks() if y >= ymin and y <= ymax]
    ax1D[0].spines["left"].set_bounds(ymin, ymax)
    ax1D[0].set_yticks(yticks)

    cmap_data = shift_data["PF_weights"][:, -1]
    for idx in np.where(shift_data["target_shifts"] == 0)[0]:
        cmap_data[idx] = shift_data["PF_weights"][idx, 1]

    plot_PF_cmap(
        cmap_data,
        sub_ax=ax1D[1],
        x_vals=PCs.place_cell_centres[:, 0],
        y_vals=shift_data["target_shifts"],
        keep_yticks=True,
    )

    x_loc = gen_util.get_proportion_edges(PCs.place_cell_centres, 0.01)
    for target_shift in examples:
        if target_shift not in shift_data["target_shifts"]:
            raise RuntimeError(f"{target_shift} shift not found in data dictionary.")
        idx = np.where(shift_data["target_shifts"] == target_shift)[0][0]
        ax1D[1].scatter(
            x_loc,
            shift_data["target_shifts"][idx],
            color=PCs.color,
            marker=">",
            s=10,
            zorder=5,
        )

    for sub_ax in ax1D:
        add_1D_position_markers(
            sub_ax, Ag=Ag, y_1D=sub_ax.get_ylim()[1], target_alpha=0.5
        )
        mark_1D_target(sub_ax, Ag=Ag, alpha=0.4)

    plot_util.pad_axis(ax1D[0], axis="y", pad_prop=0.05)

    return ax1D


def run_linear_fct(fct_name="linear_speeds", overwrite=False, seed=True):
    """
    run_linear_fct()

    Runs a specified linear function (either 'linear_speeds' or 'linear_shifts'),
    loading an existing data dictionary if it exists or rerunning the experiment.

    Args:
    - fct_name (str): Name of the function to run. Options are 'linear_speeds' or
        'linear_shifts'. Default is 'linear_speeds'.
    - overwrite (bool): Whether to overwrite existing data. Default is False.
    - seed (bool): Whether to seed the random number generator with the paper seed.
        Default is True.

    Returns:
    - data_dict (dict): Dictionary containing the results of the experiment.
    """

    if fct_name == "linear_speeds":
        fct = run_linear_speeds
        data_name = "speed_data"
    elif fct_name == "linear_shifts":
        fct = run_linear_shifts
        data_name = "shift_data"
    else:
        raise ValueError(f"fct_name '{fct_name}' not recognized.")

    save_path = Path(ratinabox.figure_directory, f"{data_name}_{PAPER_SEED}.npz")
    if overwrite:
        gen_util.delete_np_dict(save_path)
    data_dict = gen_util.load_np_dict(save_path)

    if data_dict is None:
        print("Running...")
        data_dict = fct(seed=seed)
        gen_util.save_np_dict(save_path, data_dict)

    return data_dict
