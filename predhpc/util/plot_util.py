from typing import Any
import warnings

import numpy as np
from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import markers as mpl_markers
from matplotlib import colorbar as mpl_cbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns  # type: ignore[import]

from ratinabox import utils as rutils  # type: ignore[import]
from ratinabox import stylize_plots as rstylize_plots  # type: ignore[import]

from predhpc.util import ext_util, gen_util


def save_figure(fig, name: str = "figure", save: bool | None = None):
    """
    save_figure(fig)

    Save figure.

    Args:
    - fig (mpl_figure.Figure): Figure to save.
    - name (str, optional): Name of figure. Default is "figure".
    - save (bool, optional): If True, saves the figure. If False, does not save the
        figure. If None, the global variable for ratinabox is checked.
        Default is None.
    """

    rutils.save_figure(fig, name, save=save)  # type: ignore[arg-type]


def stylize_plots_for_notebook(dpi: int = 150):
    """
    stylize_plots_for_notebook()

    Stylize plots for notebook, but at a lower default DPI than ratinabox.
    """

    rstylize_plots()

    from matplotlib import rcParams as mpl_rcParams

    mpl_rcParams["figure.dpi"] = dpi


def get_nrows_ncols(n, num_cols=3):
    """
    get_nrows_ncols(n)

    Obtain the number of rows and columns for a subplot grid.

    Args:
    - n (int): Number of subplots.
    - num_cols (int or str, optional): Number of columns for the subplot grid.
        If "square", the grid will be as square as possible. Default is 3.

    Returns:
    - num_rows (int): Number of rows for the subplot grid.
    - num_cols (int): Number of columns for the subplot grid.
    """

    if num_cols == "square":
        num_rows = int(np.ceil(np.sqrt(n)))
        num_cols = int(np.ceil(n / num_rows))
    else:
        if num_cols > n:
            num_cols = n
        num_rows = int(np.ceil(n / num_cols))

    return num_rows, num_cols


def init_subplot_grid(n, num_cols=3, width=3, height=2, sharex=True, sharey=True):
    """
    init_subplot_grid(n)

    Initialise a subplot grid.

    Args:
    - n (int): Number of subplots.
    - num_cols (int or str, optional): Number of columns for the subplot grid.
        If "square", the grid will be as square as possible. Default is 3.
    - width (int, optional): Width of the subplots. Default is 3.
    - height (int, optional): Height of the subplots. Default is 2.
    - sharex (bool, optional): Whether to share the x-axis. Default is True.
    - sharey (bool, optional): Whether to share the y-axis. Default is True.

    Returns:
    - fig (mpl_figure.Figure): Figure object.
    - axes (2D np.ndarray): 2D array of subplots.
    """

    num_rows, num_cols = get_nrows_ncols(n, num_cols=num_cols)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(width * num_cols, height * num_rows),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )

    for s, sub_ax in enumerate(axes.ravel()):
        if s >= n:
            sub_ax.axis("off")

    return fig, axes


def remove_redundant_subplot_axis_labels(axes):
    """
    remove_redundant_subplot_axis_labels(axes)

    Remove redundant axis x and y axis labels for a subplot grid.

    Args:
    - axes (2D np.ndarray): 2D array of subplots.
    """

    if len(axes.shape) != 2:
        raise ValueError("Expected a 2D array of subplots.")

    for r, ax_row in enumerate(axes):
        for c, sub_ax in enumerate(ax_row):
            if r != len(axes) - 1:
                sub_ax.set_xlabel("")
            if c != 0:
                sub_ax.set_ylabel("")


def shape_ax(ax_to_reshape, src_ax=None):
    """
    shape_ax(ax_to_reshape)

    Reshape the axes to match the source axes.

    Args:
    - ax_to_reshape (plt.Axes or np.ndarray): Axes to reshape.
    - src_ax (plt.Axes or np.ndarray, optional): Source axes to reshape to.
        Default is None.

    Returns:
    - ax_to_reshape (plt.Axes or np.ndarray): Reshaped axes.
    """

    if src_ax is None:
        return ax_to_reshape

    if isinstance(src_ax, np.ndarray):
        ax_to_reshape = np.asarray(ax_to_reshape).reshape(src_ax.shape)
    elif isinstance(src_ax, plt.Axes):
        if len(np.asarray(ax_to_reshape).ravel()) == 1:
            ax_to_reshape = np.asarray(ax_to_reshape).ravel()[0]

    return ax_to_reshape


def organize_fig_ax_kwargs(fig=None, sub_ax=None, ax=None, axes=None, **kwargs):
    """
    organize_fig_ax_kwargs()

    Obtain keyword arguments with figure and subplot keywords organized.

    Args:
    - fig (mpl_figure.Figure, optional): Figure. Default is None.
    - sub_ax (plt.Axes, optional): Subplot. Default is None.
    - ax (plt.Axes or np.ndarray, optional): Subplot or array of subplots.
        Default is None.
    - axes (np.ndarray, optional): Array of subplots. Default is None.

    Keyword args:
    - **kwargs: All other keyword arguments to return.

    Returns:
    - kwargs (dict): Keyword argument dictionary, with, at most, the following
        figure-related keys and values:
        - "fig" (plt.Figure): Figure.
        - "ax" (plt.Axes): Axes
    """

    fig_ax_kwargs = dict()

    if sub_ax is not None:
        if ax is not None:
            raise ValueError("Cannot specify sub_ax and ax.")
        if axes is not None:
            raise ValueError("Cannot specify sub_ax and axes.")
        fig_ax_kwargs["ax"] = sub_ax
    elif ax is not None:
        if axes is not None:
            raise ValueError("Cannot specify ax and axes.")
        fig_ax_kwargs["ax"] = ax
    elif axes is not None:
        fig_ax_kwargs["ax"] = axes

    if "ax" in fig_ax_kwargs.keys():
        ax_fig = np.asarray(fig_ax_kwargs["ax"]).ravel()[0].figure
        if fig is not None and fig != ax_fig:
            raise ValueError(
                "If a figure is passed, it must be associated with the provided subplot(s)."
            )
        fig_ax_kwargs["fig"] = np.asarray(fig_ax_kwargs["ax"]).ravel()[0].figure
    elif fig is not None:
        raise ValueError("If a figure is passed, sub_ax, ax or axes must be provided.")

    kwargs.update(fig_ax_kwargs)

    return kwargs


def scale_alpha(n, max_alpha=1.0, decay_rate=0.01):
    """
    scale_alpha(n)

    Scale the alpha value of a plot.

    Args:
    - n (int): Number of decay steps to apply to max alpha.
    - max_alpha (float, optional): Maximum alpha value. Default is 1.0.
    - decay_rate (float, optional): Rate of decay. Default is 0.01.

    Returns:
    - alpha (float): Alpha value.
    """

    alpha = max_alpha * np.exp(-decay_rate * n)
    return alpha


def get_figsize(prop_of_default_height=1, squat_height=False) -> tuple[float, float]:
    """
    get_figsize()

    Obtain figure size for a plot.

    Args:
    - prop_of_default_height (float, optional): Proportion of the height of the plot.
        Default is 1.
    - squat_height (bool, optional): Whether to make the plot squat_height.
        Default is False.

    Returns:
    - figsize (tuple): Figure size (width, height).
    """

    width = 5
    height = 1.5 if squat_height else 3
    figsize = (width, height * prop_of_default_height)

    return figsize


def get_plot_shape(n: int, target_num_col: int = 10) -> tuple[int, int]:
    """
    get_plot_shape(n)

    Obtain the shape of a plot with n subplots, allowing only exact
    divisions of n.

    Args:
    - n (int): Number of subplots
    - target_num_col (int, optional): Number of columns to aim for. Default is 10.

    Returns:
    - shape (tuple): Shape of plot (num_rows, num_cols)
    """

    divisors = np.asarray(gen_util.get_divisors(n))
    if len(divisors) == 2:
        ncol = n
        nrow = 1
    else:
        closest = np.argmin(np.abs(divisors - target_num_col))
        ncol = divisors[closest]
        nrow = n // ncol
    shape = (nrow, ncol)

    return shape


def pad_axis(sub_ax, axis="y", pad_prop=0.1, end="both"):
    """
    pad_axis(sub_ax)

    Pads the axis limits of a subplot.

    Args:
    - sub_ax (plt.Axes): Subplot for which to pad an axis.
    - axis (str, optional): Subplot to pad. Defaut is "y".
    - pad_prop (float, optional): Proportion of the axis range to pad by.
        Default is 0.1.
    - end (str, optional): End to pad. Defaut is "both".
    """

    if axis not in ["x", "y", "both"]:
        raise ValueError(f"Subplot {axis} is not recognized.")
    if end not in ["both", "low", "high"]:
        raise ValueError(f"End {end} is not recognized.")

    if axis == "both":
        axes = ["x", "y"]
    else:
        axes = [axis]

    for axis in axes:
        if axis == "x":
            min_val, max_val = sub_ax.get_xlim()
            set_fct = sub_ax.set_xlim
        else:
            min_val, max_val = sub_ax.get_ylim()
            set_fct = sub_ax.set_ylim

        pad = (max_val - min_val) * pad_prop

        if end == "both":
            min_val -= pad / 2
            max_val += pad / 2
        elif end == "low":
            min_val -= pad
        elif end == "high":
            max_val += pad

        set_fct(min_val, max_val)


def convert_to_rgb(hex_color):
    """
    convert_to_rgb(hex_color)

    Convert a hex color to an RGB tuple.

    Args:
    - hex_color (str): Hex color string.

    Returns:
    - rgb_color (tuple): RGB color tuple.
    """

    rgb_color = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    return rgb_color


def complete_legend_kwargs(legend, **legend_kwargs):
    """
    complete_legend_kwargs(legend)

    Obtain a dictionary with legend keyword argument values retrieved from the
    legend, if they were not already provided.

    Args:
    - **legend_**kwargs: Legend arguments to use, instead of ones extracted from the
        legend.

    Returns:
    - legend_kwargs (dict): Completed legend keyword dictionary with keys and values:
        - "frameon" (bool): Whether the frame is on.
        - "bbox_to_anchor" (tuple): Bounding box to anchor.
        - "fontsize" (int): Font size.
        - "loc" (tuple or str): Location of the legend.
    """

    if "frameon" not in legend_kwargs.keys():
        legend_kwargs["frameon"] = legend.get_frame_on()
    if "bbox_to_anchor" not in legend_kwargs.keys():
        legend_kwargs["bbox_to_anchor"] = legend.get_bbox_to_anchor()._bbox
    if "fontsize" not in legend_kwargs.keys():
        texts = legend.get_texts()
        if len(texts) > 0:
            legend_kwargs["fontsize"] = texts[0].get_fontsize()
    if "loc" not in legend_kwargs.keys():
        legend_kwargs["loc"] = legend._loc

    return legend_kwargs


def remove_prev_handle_labels(sub_ax: plt.Axes, **legend_kwargs):
    """
    remove_prev_handle_labels(sub_ax)

    Remove previous handle labels from the legend.

    Args:
    - sub_ax (plt.Axes): Subplot from which to remove handle labels.

    Keyword args:
    - **legend_kwargs: Keyword arguments passed to the legend.
    """

    legend = sub_ax.get_legend()
    if legend is None:
        return

    handles, labels = sub_ax.get_legend_handles_labels()
    if len(labels) == 0:
        return

    num_unique = len(set(labels))
    handles = handles[:-num_unique]
    labels = labels[:-num_unique]

    order = np.argsort(labels)
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    # gather a few kwargs from the current legend
    legend_kwargs = complete_legend_kwargs(legend, **legend_kwargs)

    sub_ax.legend(handles=handles, labels=labels, **legend_kwargs)


def set_violinplot_colors(violin_parts, color="grey"):
    """
    set_violinplot_colors(violin_parts)

    Set the face and edge colors of violinplot components.

    Args:
    - violin_parts (dict): Dictionary of violinplot components.
    - color (str, optional): Color to set the face and edge colors to.
        Default is "grey".
    """

    for subpart in violin_parts.values():
        if not isinstance(subpart, list):
            subpart = [subpart]
        for component in subpart:
            component.set_facecolor(color)
            component.set_edgecolor(color)


def plot_activation_function(
    activation_function,
    min_input_fr=-15,
    max_input_fr=15,
    sub_ax=None,
    color=None,
):
    """
    plot_activation_function(activation_function)

    Plot an activation function.

    Args:
    - activation_function (function): Activation function to plot.
    - min_input_fr (float, optional): Minimum input firing rate. Default is -15.
    - max_input_fr (float, optional): Maximum input firing rate. Default is 15.
    - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
    - color (str, optional): Line color. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with activation function plotted.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(4, 2))

    x = np.linspace(min_input_fr, max_input_fr, 1000)
    extrema = np.asarray([-1e2, 1e2])
    if isinstance(activation_function, dict):
        y = rutils.activate(x, other_args=activation_function)
        min_y, max_y = rutils.activate(extrema, other_args=activation_function)
    else:
        y = activation_function(x, deriv=False)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="overflow"
            )
            min_y, max_y = activation_function(extrema, deriv=False)

    sub_ax.plot(x, y, color=color, lw=1.5)
    sub_ax.axhline(0, color="k", lw=1, ls="dashed")
    sub_ax.axhline(min_y + (max_y - min_y) / 2, color="k", lw=1, ls="dashed")
    sub_ax.axvline(0, color="k", lw=1, ls="dashed")

    sub_ax.set_xlabel("Input firing rate")
    sub_ax.set_ylabel("Output value")
    sub_ax.set_title("Activation function")

    sub_ax.spines[["right", "top"]].set_visible(False)

    return sub_ax


def get_time_indices(
    pre: int = 1,
    post: int = 1,
    dt: float = 0.03,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """
    get_time_indices()

    Obtain time indices for a given pre and post time.

    Args:
    - pre (int, optional): Pre time. Default is 1.
    - post (int, optional): Post time. Default is 1.
    - dt (float, optional): Time step. Default is 0.03.

    Returns:
    - time_indices (1D np.ndarray): Array of time indices.
    """

    num_pre = int(np.around(pre / dt))
    num_post = int(np.around(post / dt))

    time_indices = np.arange(-num_pre, num_post + 1)

    return time_indices


def get_plotting_times(
    times: np.ndarray[tuple[int], np.dtype[np.float64]] | list,
    t_start: float | None = None,
    t_end: float | None = None,
    raise_error: bool = True,
) -> tuple[int, int]:
    """
    get_plotting_times(times)

    Obtain the times to plot.

    Args:
    - times (1D np.ndarray): Array of timepoints.
    - t_start (float, optional): Start time. Default is None.
    - t_end (float, optional): End time. Default is None.
    - raise_error (bool, optional): Whether to raise an error if the end time is
        less than or equal to the start time. Default is True.

    Returns:
    - startid (int): Index of the start time.
    - endid (int): Index of the end time.
    """

    times = np.asarray(times)

    # times to plot
    start_time = t_start if t_start is not None else times[0]
    if start_time < 0:
        start_time = times[-1] + t_start

    end_time = t_end if t_end is not None else times[-1]
    if end_time < 0:
        end_time = times[-1] + t_end

    if end_time <= start_time and raise_error:
        raise ValueError("End time must be greater than start time.")

    startid = int(np.argmin(np.abs(times - (start_time))))
    endid = int(np.argmin(np.abs(times - (end_time))))

    if endid == startid and raise_error:
        raise RuntimeError("End index is the same as start index.")

    return startid, endid


def init_rate_map_axes(
    num_plots=1, num_cols=None, size_per=1.5, target_neurons=None, **kwargs
):
    """
    init_rate_map_axes()

    Initialize rate map axes.

    Args:
    - num_plots (int, optional): Number of plots. Default is 1.
    - num_cols (int, optional): Number of columns. Default is None.
    - size_per (float, optional): Size per plot. Default is 1.5.
    - target_neurons (PlaceCells, optional): Target neurons. Default is None.

    Keyword args:
    - **kwargs: Keyword arguments passed to the environment plot.

    Returns:
    - axes (2D np.ndarray): Array of axes.
    """

    num_cols = np.min([num_plots, num_cols])
    num_rows = np.ceil(num_plots / num_cols).astype(int)
    figsize = (num_cols * size_per, num_rows * size_per)

    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    if target_neurons is not None:
        for i, sub_ax in enumerate(axes.ravel()):
            if i < num_plots:
                target_neurons.Agent.Environment.plot_environment(
                    sub_ax=sub_ax,
                    autosave=False,
                    **kwargs,
                )

    for sub_ax in axes.ravel()[num_plots:]:
        sub_ax.axis("off")

    return axes


def add_colorbars(axes, im, vmin=None, vmax=None, label=None, end_only=False, round=2):
    """
    add_colorbars(axes, im)

    Add a colorbar to the end of each row of subplots.

    Args:
    - axes (list or np.ndarray): List or array of axes to add the colorbar to.
    - im (mpl.image.AxesImage): Image to add the colorbar(s) to.
    - vmin (float, optional): Minimum value. Default is None.
    - vmax (float, optional): Maximum value. Default is None.
    - label (str, optional): Label for the colorbar. Default is None.
    - end_only (bool, optional): Whether to add colorbars only to the end of each row.
        Default is False.
    - round (int, optional): Number of decimal places to round to. Default is 2.

    Returns:
    - cbars (list): Colorbars.
    """

    if label is None:
        label = "Firing rate / Hz"

    axes = np.asarray(axes)
    if len(axes.shape) < 2 or end_only:
        dividers = [make_axes_locatable(axes.ravel()[-1])]
    else:
        dividers = [make_axes_locatable(ax_row[-1]) for ax_row in axes]

    if vmin is None:
        vmin = im.get_array().min()
    if vmax is None:
        vmax = im.get_array().max()

    if round is None:
        round = -int(np.floor(np.log10(np.absolute(vmax - vmin))))

    cbars = list()
    for divider in dividers:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(length=0)
        if label is not None:
            cbar.set_label(label, labelpad=-10)

        vmin_tick = np.around(vmin, round)
        vmax_tick = np.around(vmax, round)
        cbar.set_ticks([vmin_tick, vmax_tick])
        cbar.outline.set_visible(False)
        cbars.append(cbar)

    return cbars


def normalize_cmaps(
    axes: list[plt.Axes],
    round_order: int = 1,
    shrink: float = 0.7,
    colorbar: bool = True,
) -> mpl_cbar.Colorbar | None:
    """
    normalize_cmaps(axes)

    Normalize colormaps across subplots.

    Args:
    - axes (list): List of axes to normalize together.
    - round_order (int, optional): Order of magnitude to round to. Default is 1.
    - shrink (float, optional): Shrink factor for colorbar. Default is 0.7.
    - colorbar (bool, optional): Whether to add a colorbar. Default is True.

    Returns:
    - cbar (mpl.colorbar.Colorbar): Colorbar or None.
    """

    images = []
    vmin, vmax = np.inf, -np.inf
    for sub_ax in axes:
        ax_images = sub_ax.get_images()
        if len(ax_images) == 0:
            raise ValueError("No images found in subplot.")
        else:
            im = ax_images[0]
        vmin = min(vmin, im.get_array().min())
        vmax = max(vmax, im.get_array().max())
        images.append(im)

    if colorbar:
        vmin = np.floor(vmin * 10**round_order) / 10**round_order
        vmax = np.ceil(vmax * 10**round_order) / 10**round_order

    for im in images:
        im.set_clim(vmin, vmax)

    cbar = None
    if colorbar:
        cbar = axes[0].get_figure().colorbar(images[-1], ax=axes, shrink=shrink)
        ticks = [vmin, vmax]
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(length=0)
        cbar.outline.set_visible(False)  # type: ignore[attr-defined]

    return cbar


def get_trajectory_cmap_colors(
    trajectory_lengths,
    colormap=None,
    cmap_per=False,
    scale_cmap_per=False,
    time_idx=None,
):
    """
    get_trajectory_cmap_colors(trajectory_lengths)

    Obtain colormap colors for trajectories.

    Args:
    - trajectory_lengths (1D np.ndarray): Array of trajectory lengths.
    - colormap (str, optional): Colormap to use. Default is None.
    - cmap_per (bool, optional): Whether to use a colormap per trajectory.
        Default is False.
    - scale_cmap_per (bool, optional): Whether to scale the colormap per trajectory.
        Default is False.
    - time_idx (int, optional): Time index for restricting steps for which to obtain
        colors. Default is None.

    Returns:
    - cmap_colors (1D np.ndarray): Array of colormap colors.
    """

    traj_idx = [np.full(steps, i) for i, steps in enumerate(trajectory_lengths)]

    if cmap_per:
        if scale_cmap_per:
            cmap_vals = [np.linspace(0, 1, steps) for steps in trajectory_lengths]
        else:
            cmap_vals = [
                np.arange(steps, dtype=np.int64) for steps in trajectory_lengths
            ]
    else:
        cmap_vals = traj_idx[:]
    cmap_vals_np = np.concatenate(cmap_vals).astype(float)

    if time_idx is not None:
        if len(cmap_vals_np) <= time_idx.max():
            raise RuntimeError("Time index is out of bounds.")
        cmap_vals_np = cmap_vals_np[time_idx]

    cmap_min, cmap_max = cmap_vals_np.min(), cmap_vals_np.max()
    if cmap_min == cmap_max:
        cmap_vals_np[:] = 0.5  # mid point of the colormap
    else:
        cmap_vals_np = (cmap_vals_np - cmap_min) / (cmap_max - cmap_min)

    if colormap is None:
        colormap = "crest"
    cmap_colors = sns.color_palette(colormap, as_cmap=True)(cmap_vals_np)  # type: ignore[callable]

    return cmap_colors


def get_trajectory_dict(
    trajectory_lengths: np.ndarray[tuple[int], np.dtype[np.int64]] | list | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    get_trajectory_dict(trajectory_lengths)

    Obtain a dictionary of trajectory lengths.

    Args:
    - trajectory_lengths (1D np.ndarray, optional): Trajectory lengths. Default is None.

    Keyword args:
    - **kwargs: Keyword arguments passed to ext_util.get_trajectory_lengths().

    Returns:
    - trajectory_dict (dict): Dictionary of trajectory lengths with keys:
        - "trajectory_lengths" (1D np.ndarray): Trajectory lengths.
        - "min_trajectory_length" (int): Minimum trajectory length.
        - "max_trajectory_length" (int): Maximum trajectory length.
        - "midpoint" (int): Midpoint of trajectory lengths.
        - "midpoint_idx" (int): Index of the midpoint.
    """

    if trajectory_lengths is None:
        trajectory_lengths = ext_util.get_trajectory_lengths(**kwargs)
    else:
        trajectory_lengths = np.asarray(trajectory_lengths)

    min_trajectory_length, max_trajectory_length = [
        trajectory_lengths.min(),
        trajectory_lengths.max(),
    ]

    true_midpoint = (
        min_trajectory_length + (max_trajectory_length - min_trajectory_length) / 2
    )
    midpoint_idx = np.argmin(np.abs(trajectory_lengths - true_midpoint))
    midpoint = trajectory_lengths[midpoint_idx]

    trajectory_dict = {
        "trajectory_lengths": trajectory_lengths,
        "min_trajectory_length": min_trajectory_length,
        "max_trajectory_length": max_trajectory_length,
        "midpoint": midpoint,
        "midpoint_idx": midpoint_idx,
    }

    return trajectory_dict


def get_plot_marker_kwargs(position_name: str = "reset", base_s: float = 15) -> dict:
    """
    get_plot_marker_kwargs()

    Obtain the marker style and color for a position.

    Args:
    - position_name (str): Position name to plot. Must be 'start', 'reset', 'target'
        or 'agent'.

    Returns:
    - marker_kwargs (dict): Dictionary with keyword arguments for plt.scatter(),
        with keys and values:
        - "color" (str): Marker color.
        - "marker" (str): Marker style.
        - "s" (float): Marker size.
    """

    if position_name == "start":
        color = "gold"
        marker = mpl_markers.MarkerStyle("^")
        s = base_s

    elif position_name == "reset":
        color = "red"
        marker = mpl_markers.MarkerStyle("x")
        s = base_s * 1.2

    elif position_name == "target":
        color = "blue"
        marker = mpl_markers.MarkerStyle("o")
        s = base_s * 1.3

    elif position_name == "agent":
        color = "black"
        marker = mpl_markers.MarkerStyle("d")
        s = base_s * 1.3

    else:
        raise NotImplementedError(
            "Position name must be 'start', 'reset' or 'target', "
            f"but got {position_name}."
        )

    marker_kwargs = {"color": color, "marker": marker, "s": s}

    return marker_kwargs


def get_closest_step_marker_kwargs(step_type="steps_before", plot_line=False):
    """
    get_closest_step_marker_properties()

    Get the marker properties for plotting the closest steps to the target.

    Args:
    - step_type (str, optional): Type of closest steps to plot
        ("steps_before", "steps_near_BTSP", "steps_of_nearest_BTSP", "steps_other",
        "other_BTSP_steps"). Default is "steps_before".

    Returns:
    - marker_kwargs (dict): Marker properties for plotting the closest steps,
        with keys: "color", "marker", "zorder".
    """

    step_types = [
        "steps_before",
        "steps_near_BTSP",
        "steps_of_nearest_BTSP",
        "steps_other",
        "other_BTSP_steps",
    ]

    if step_type == "steps_of_nearest_BTSP":
        color = "crimson"
        marker = "d"
        zorder = 1
        s = 15
    elif step_type == "other_BTSP_steps":
        color = "crimson"
        marker = "x"
        zorder = 1
        s = 12
    else:
        marker = "o"
        zorder = 2
        s = 12
        if step_type == "steps_before":
            color = "grey"
        elif step_type == "steps_near_BTSP":
            color = "crimson"
        elif step_type == "steps_other":
            color = "darkslateblue"
        else:
            raise ValueError(
                f"step_type '{step_type}' not recognized. Must be in: {step_types}."
            )

    marker_kwargs = {
        "color": color,
        "alpha": 0.5,
    }

    if plot_line:
        marker_kwargs["zorder"] = 0
        marker_kwargs["ls"] = "dotted"
    else:
        marker_kwargs["zorder"] = zorder
        marker_kwargs["marker"] = marker
        marker_kwargs["s"] = s

    return marker_kwargs


def plot_rate_correlations(firingrates, sub_ax=None, cut_off_thr=None):
    """
    plot_rate_correlations(firingrates)

    Plot the correlation matrix of firing rates.

    Args:
    - firingrates (2D np.ndarray): Firing rates of neurons (num_neurons, num_steps).

    Returns:
    - sub_ax (plt.Axes): Subplot with the correlation matrix plotted.
    """

    CC = np.corrcoef(np.asarray(firingrates).T)

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(3, 3))

    if cut_off_thr is not None:
        sorter, groups = gen_util.get_CC_sorter(CC, cut_off_thr=70, log=False)
        CC = CC[sorter][:, sorter]

    im = sub_ax.imshow(CC, origin="upper", vmin=0, vmax=1)
    sub_ax.figure.colorbar(im)

    if cut_off_thr:
        sub_ax.set_title("Pairwise CCs, grouped and sorted", y=1.1)
        n = 0
        for num_in_grp, _ in groups.values():
            n += num_in_grp
            plt.axvline(n - 0.5, color="white", alpha=0.8, lw=0.3)
            plt.axhline(n - 0.5, color="white", alpha=0.8, lw=0.3)
    else:
        sub_ax.set_title("Pairwise CCs", y=1.1)

    sub_ax.set_xlabel("Neuron")
    sub_ax.set_ylabel("Neuron")

    return sub_ax


def plot_learning_kernel(Is, xs, kernel=None, kernel_xs=None):
    """
    plot_learning_kernel(Is, xs)

    Plot a Gaussian-filtered learning kernel.

    Args:
    - Is (2D np.ndarray): Inferred 2D input (exponential dim x Gaussian only dim).
    - xs (1D np.ndarray): Position coordinates along the first dimension of Is.
    - kernel (2D np.ndarray, optional): Full 2D input kernel. Default is None.
    - kernel_xs (1D np.ndarray, optional): Position coordinates along the first
        dimension of kernel, required if kernel is provided. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with the learning kernel plotted.
    """

    if len(Is) != len(xs):
        raise ValueError("Is and xs must have the same length.")

    _, sub_ax = plt.subplots(figsize=(6, 2.5))
    color = None
    for y in range(Is.shape[1]):
        alpha = Is[:, y].max() / Is.max()
        sc = sub_ax.scatter(xs, Is[:, y], s=6, alpha=alpha, color=color)
        if color is None:
            color = sc.get_facecolor()[0]

    if kernel is not None:
        if kernel_xs is None:
            raise ValueError("If providing kernel, must provide kernel_xs.")
        if len(kernel) != len(kernel_xs):
            raise ValueError("Kernel and kernel_xs must have the same length.")
        if xs[0] > kernel_xs.min():
            start_x = gen_util.get_index_of_closest(kernel_xs, xs[0], method="below")
            kernel = kernel[start_x:]
            kernel_xs = kernel_xs[start_x:]
        if xs[-1] < kernel_xs.max():
            stop_x = gen_util.get_index_of_closest(kernel_xs, xs[-1], method="above")
            kernel = kernel[: stop_x + 1]
            kernel_xs = kernel_xs[: stop_x + 1]

        sub_ax.plot(
            kernel_xs,
            np.max(kernel, axis=1),
            color="k",
            # lw=1.5,
            alpha=0.8,
        )

    # format
    sub_ax.axhline(0, ls="dashed", color="k", alpha=0.6)
    pad = 0.05 * (xs[-1] - xs[0])
    sub_ax.set_xlim(xs[0] - pad, xs[-1] + pad)
    sub_ax.set_ylim(-0.5, Is.max() + 0.5)
    sub_ax.set_title("Learning kernel")
    sub_ax.set_ylabel("Kernel-inferred input")
    sub_ax.set_xlabel("Position")
    sub_ax.spines[["top", "right"]].set_visible(False)

    return sub_ax


def plot_skewed_gaussian_kernel(
    wid_half_max: float = 1.5,
    prop: float = 4.0,
    atol: float = 1e-6,
    dt: float = 0.03,
    num_estimate_pts: int = 5000,
    sub_ax: plt.Axes | None = None,
    autosave: bool | None = None,
) -> tuple[
    plt.Axes,
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.int64,
]:
    """
    plot_skewed_gaussian_kernel()

    Plot a Gaussian skewed kernel.

    Args:
    - wid_half_max (float, optional): Width at half max of the kernel.
        Default is 1.5.
    - prop (float, optional): Right / left width half max. Default is 4.0.
    - atol (float, optional): Absolute tolerance for determining the edges of the
        distribution. Default is 1e-6.
    - dt (float, optional): Time step size in seconds. Default is 0.03.
    - num_estimate_pts (int, optional): Number of points to use for estimating the
        kernel parameters. Default is 5000.
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.
    - autosave (bool, optional): Whether to save the figure. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with skewed Gaussian kernel plotted.
    - skewed_Gaussian_kernel (function): Skewed Gaussian kernel function
    - max_idx (float): Index of the maximum value of the kernel
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots()

    skewed_Gaussian_kernel, max_value_idx = gen_util.get_skewed_Gaussian_kernel(
        wid_half_max, prop, atol, dt, num_estimate_pts=num_estimate_pts
    )
    pts = (
        np.arange(len(skewed_Gaussian_kernel)) - np.argmax(skewed_Gaussian_kernel)
    ) * dt

    act_wid_half_max_idxs = list()
    for i, sl in enumerate([slice(0, max_value_idx), slice(max_value_idx, None)]):
        range_mid = skewed_Gaussian_kernel[sl]

        # find the closest point to the half-max
        act_wid_half_max_idx = np.argmin(
            np.abs(range_mid - skewed_Gaussian_kernel.max() / 2)
        )
        if i == 0:
            act_wid_half_max_idx = max_value_idx - act_wid_half_max_idx

        act_wid_half_max_idxs.append(act_wid_half_max_idx)

    act_wid_half_maxs = [wid_idx * dt for wid_idx in act_wid_half_max_idxs]
    order = [0, 1] if prop < 1 else [1, 0]

    blue = "cornflowerblue"

    # center point
    sub_ax.axhline(0, color="black", ls="dashed", alpha=0.4)
    sub_ax.axvline(0, color="black", ls="dashed", alpha=0.4)

    # half-width
    sub_ax.axhline(skewed_Gaussian_kernel.max() / 2, color=blue, ls="dashed", alpha=0.6)
    for o, alpha in zip(order, [0.6, 0.3]):
        sign = np.sign(o - 0.5)
        plot_wid_half_max = sign * act_wid_half_maxs[o]
        label = (
            f"width at half max: {act_wid_half_maxs[o]:.3f}s"
            f"\n({act_wid_half_max_idxs[o]} pts"
        )
        if o == np.argmax(act_wid_half_maxs):
            label = f"{label}, target: {wid_half_max:.3f}s)"
        else:
            act_prop = act_wid_half_max_idxs[1] / act_wid_half_max_idxs[0]
            label = f"{label}, prop: {act_prop:.3f} vs target: {prop:.3f})"
        sub_ax.axvline(
            plot_wid_half_max, color=blue, ls="dashed", label=label, alpha=alpha
        )

    label = f"{pts.min():.2f}s to {pts.max():.2f}s ({len(skewed_Gaussian_kernel)} pts)"
    sub_ax.plot(pts, skewed_Gaussian_kernel, color=blue, label=label)
    sub_ax.set_xlabel("Time (s)")
    sub_ax.set_ylabel("Skewed Gaussian kernel")
    sub_ax.spines[["top", "right"]].set_visible(False)

    legend = sub_ax.legend()
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.5)

    fig = sub_ax.figure
    save_figure(fig, "skewed_gaussian_kernel", save=autosave)

    return sub_ax, skewed_Gaussian_kernel, max_value_idx


def plot_lr_factor_assessment(assessment_dict):
    """
    plot_lr_actor_assessment(assessment_dict)

    Plot the assessment of the learning rate actor.

    Args:
    - assessment_dict (dict): Learning factor assessment dictionary with initial and
        updated weights (under "ws"), computed output firing rates (under "Os"), and
        biases if applicable (under "bs").

    Returns:
    - axes (2D np.ndarray): 2D array of subplots with the assessment plots.
    """

    num_subplots = len(assessment_dict["ws"])
    num_rows, num_cols = get_nrows_ncols(num_subplots, num_cols=4)
    width = num_cols * 2 + 1
    if assessment_dict["ws"][0].shape[1] == 1:
        base_height = 1
        hspace = 0.75
    else:
        base_height = 2
        hspace = 0.4
    height = base_height * (num_rows + hspace * (num_rows - 1))
    _, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(width, height),
        gridspec_kw={"hspace": hspace},
        squeeze=False,
    )

    vmin = 0
    max_val = np.max(assessment_dict["ws"])
    vmax_round = max(3, -int(np.floor(np.log10(np.absolute(max_val)))))
    vmax = np.ceil(max_val * 10**vmax_round) / (10**vmax_round)
    for i, w in enumerate(assessment_dict["ws"]):
        sub_ax = axes.ravel()[i]
        im = sub_ax.imshow(
            w.T, vmin=vmin, vmax=vmax, aspect="auto", interpolation="none"
        )

        if i == 0:
            title_str = r"$\bf{Initial\ setting}$"
        else:
            title_str = r"$\bf{Update\ " + str(i) + "}$"

        num_max = (w == w.max()).sum()
        if num_max == w.size:
            max_str = "all weights"
        elif num_max < 5:
            max_str = f"{num_max} weight" if num_max == 1 else f"{num_max} weights"
        else:
            perc = f"{num_max/w.size * 100:2f}%"
            max_str = f"{num_max}/{w.size} ({perc}) weights"
        extr_strs = [
            f"{max_str} at max ({w.max():.4f})",
            f"weights sum to {w.sum():.4f}",
        ]
        if "bs" in assessment_dict.keys():
            extr_strs.append(f"bias: {assessment_dict['bs'][i]:.4f}")
        extr_strs.append(f"output: {assessment_dict['Os'][i]:.3f}")
        extr_str = "\n".join(extr_strs)
        title_str = f"{title_str}\n{extr_str}"

        sub_ax.axis("off")
        sub_ax.set_title(title_str)

    for s, sub_ax in enumerate(axes.ravel()):
        if s >= len(assessment_dict["ws"]):
            sub_ax.axis("off")

    add_colorbars(axes, im, vmin=0, vmax=vmax, label="Weights", round=vmax_round)

    return axes
