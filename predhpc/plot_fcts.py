from typing import Any, TYPE_CHECKING, Sequence

import numpy as np
from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import markers as mpl_markers
from matplotlib import colors as mpl_colors
from matplotlib import patches as mpl_patches
from matplotlib import cm as mpl_cm
import scipy  # type: ignore[import]
import seaborn as sns  # type: ignore[import]

from ratinabox import utils as rutils  # type: ignore[import]
from ratinabox import MOUNTAIN_PLOT_WIDTH_MM, MOUNTAIN_PLOT_SHIFT_MM

from predhpc.util import ext_util, gen_util, learn_util, signal_util, plot_util

if TYPE_CHECKING:
    from predhpc.agent import ResetableAgent
    from predhpc.neurons import riab_neurons, learning_neurons, two_comp_neurons


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

    if PF_type == "weights":
        label = "Input weights"
    elif PF_type == "smoothed_weights":
        label = "Smoothed input weights"
    elif PF_type == "applied_weights":
        label = "Applied input weights"
    elif PF_type == "history":
        label = "Firing rate"
    else:
        raise ValueError(f"PF_type '{PF_type}' not recognized.")

    if title:
        label = f"{label}s"

    return label


def add_time_axis(
    sub_ax,
    dt: float = 0.03,
    in_min: bool = False,
    max_num_minutes: float | int | None = None,
    trajectory_lengths: np.ndarray[tuple[int], np.dtype[np.int64]] | list | None = None,
    **kwargs,
):
    """
    add_time_axis(sub_ax)

    Plot trajectories on a twin axis with time information added to y axis.

    Args:
    - sub_ax (plt.Axes): Subplot to create a twin axis from (along x axis).
    - dt (float, optional): Time step. Default is 0.03.
    - in_min (bool, optional): Whether to plot in minutes. Default is False.
    - max_num_minutes (float, optional): Maximum number of minutes to plot.
        Default is None.
    - trajectory_lengths (1D np.ndarray, optional): Trajectory lengths. Default is None.


    Keyword args:
    - **kwargs: Keyword arguments passed to plot_util.get_trajectory_dict().

    Returns:
    - twin_sub_ax (plt.Axes): Twin subplot (shared x axis) with time information
        added with a new y axis.
    """

    # create a second axis label for time
    twin_sub_ax = sub_ax.twinx()

    trajectory_dict = plot_util.get_trajectory_dict(trajectory_lengths, **kwargs)
    trajectory_lengths = np.asarray(trajectory_dict["trajectory_lengths"])

    if in_min:
        dt = dt / 60
        long_str, short_str = "minutes", "min"
        max_time = max_num_minutes
    else:
        long_str, short_str = "seconds", "s"
        max_time = max_num_minutes * 60 if max_num_minutes is not None else None

    twin_sub_ax.set_ylabel(f"Time (in {long_str})")

    y_min, y_max = sub_ax.get_ylim()
    twin_sub_ax.set_ylim(y_min * dt, y_max * dt)

    min_trajectory_length_in_time = trajectory_dict["min_trajectory_length"] * dt
    max_trajectory_length_in_time = trajectory_dict["max_trajectory_length"] * dt
    trajectory_length_label = (
        f"{min_trajectory_length_in_time:.1f}-"
        f"{max_trajectory_length_in_time:.1f} {short_str}"
    )
    twin_sub_ax.plot(
        [], [], label=trajectory_length_label, lw=0, marker=mpl_markers.MarkerStyle(".")
    )

    midpoint_in_time = trajectory_dict["midpoint"] * dt
    midpoint_label = f"Midpoint: ({midpoint_in_time:.1f} {short_str})"
    half_pt_kwargs = {"lw": 2, "ls": "dotted", "color": "k", "alpha": 0.5, "zorder": -1}
    twin_sub_ax.plot([], [], label=midpoint_label, **half_pt_kwargs)
    twin_sub_ax.spines[["top", "left"]].set_visible(False)

    trajectory_length_str = f" ({len(trajectory_lengths)} traj.)"
    if max_time is not None:
        total_time = np.cumsum(trajectory_lengths) * dt
        time_past_max = total_time > max_time

        if time_past_max.any():  # less than a cycle completed
            time_past_max_idx = int(np.argmax(time_past_max))
            twin_sub_ax.axvspan(
                time_past_max_idx,
                len(trajectory_lengths),
                color="red",
                alpha=0.3,
                zorder=-1,
                lw=0,
            )
            num_trajectories = time_past_max_idx + (
                max_time - total_time[time_past_max_idx - 1]
            ) / (total_time[time_past_max_idx] - total_time[time_past_max_idx - 1])

            trajectory_length_str = (
                f" ({num_trajectories:.1f} / {len(trajectory_lengths)} traj.)"
            )

        else:
            num_full_cycles = max_time // total_time[-1]
            last_cycle_proportion = max_time / total_time[-1] - num_full_cycles
            if last_cycle_proportion == 0:
                num_trajectories = len(trajectory_lengths) * num_full_cycles
            else:
                last_cycle_time = last_cycle_proportion * total_time[-1]
                time_past_max_idx = np.where(total_time > last_cycle_time)[0][0]
                last_trajectory = (
                    last_cycle_time - total_time[time_past_max_idx - 1]
                ) / (total_time[time_past_max_idx] - total_time[time_past_max_idx - 1])

                num_trajectories = (
                    len(trajectory_lengths) * num_full_cycles
                    + time_past_max_idx
                    + last_trajectory
                )

            num_cycles = max_time / total_time[-1]
            trajectory_length_str = (
                f" ({num_trajectories:.1f} traj., "
                f"{len(trajectory_lengths)} per cycle, ~{num_cycles:.1f} cycles)"
            )

    legend = twin_sub_ax.legend(loc="lower right", fontsize="small")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.5)
    sub_ax.set_title(f"Trajectory lengths{trajectory_length_str}")

    return twin_sub_ax


def plot_trajectory_lengths(
    dt: float | None = None,
    in_min: bool = False,
    max_num_minutes: float | int | None = None,
    trajectory_lengths: np.ndarray[tuple[int], np.dtype[np.int64]] | list | None = None,
    **kwargs,
) -> tuple[plt.Axes, np.ndarray]:
    """
    plot_trajectory_lengths()

    Plot trajectory lengths.

    Args:
    - dt (float, optional): Time step. If None, time axis is not added. Default is None.
    - in_min (bool, optional): Whether to plot time axis in minutes instead of
        seconds. Default is False.
    - max_num_minutes (float, optional): Maximum time in minutes. Default is None.

    Keyword args:
    - **kwargs: Keyword arguments passed to plot_util.get_trajectory_dict().

    Returns:
    - sub_ax (plt.Axes): Subplot with trajectory lengths plotted.
    - trajectory_lengths (1D np.ndarray): Array of trajectory lengths.
    """

    _, sub_ax = plt.subplots(figsize=(8, 4))

    trajectory_dict = plot_util.get_trajectory_dict(trajectory_lengths, **kwargs)
    trajectory_lengths = np.asarray(trajectory_dict["trajectory_lengths"])

    steps_label = (
        f"{trajectory_dict['min_trajectory_length']}-"
        f"{trajectory_dict['max_trajectory_length']} steps"
    )

    sub_ax.plot(
        trajectory_lengths,
        label=steps_label,
        lw=0,
        marker=mpl_markers.MarkerStyle("."),
        markersize=10,
    )

    midpoint_idx = trajectory_dict["midpoint_idx"]
    midpoint = trajectory_dict["midpoint"]
    half_pt_kwargs = {"lw": 2, "ls": "dotted", "color": "k", "alpha": 0.5, "zorder": -1}
    sub_ax.plot(
        [0, midpoint_idx],
        [midpoint, midpoint],
        label=f"Midpoint: ({midpoint:.0f} steps)",
        **half_pt_kwargs,
    )
    sub_ax.plot([midpoint_idx, midpoint_idx], [0, midpoint], **half_pt_kwargs)

    # plot aesthetics
    sub_ax.spines[["top", "right"]].set_visible(False)
    legend = sub_ax.legend(loc="upper left", fontsize="small")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.5)
    sub_ax.set_xlabel("Trajectory number")
    sub_ax.set_ylabel("Trajectory length (in steps)")

    # set y-axis limits
    if len(trajectory_lengths) > 0:
        min_val = trajectory_lengths.min()
        max_val = trajectory_lengths.max()
        pad = max(10, 0.05 * (max_val - min_val))
        new_min = min_val - pad
        if new_min > 0 and min_val - pad * 4 < 0:  # if close to 0
            new_min = 0
        sub_ax.set_ylim(new_min, max_val + pad)

    # set x-axis limits
    if len(trajectory_lengths) > 0:
        pad = 0.05 * len(trajectory_lengths)
        sub_ax.set_xlim(0 - pad, len(trajectory_lengths) + pad)

    if dt is not None:
        add_time_axis(sub_ax, dt, in_min, max_num_minutes, trajectory_lengths)

    return sub_ax, trajectory_lengths


def mark_target_and_reset_points(
    Pyrs: "riab_neurons.Neurons",
    sub_ax: plt.Axes,
    restore_xlims: bool = True,
    lw: float = 1.0,
    omit_reset: bool = False,
    min_steps_btw: float = 0.0,
):
    """
    mark_target_and_reset_points(Pyrs, sub_ax)

    Add target and reset points to a timeseries plot.

    Args:
    - Pyrs (riab_neurons.Neurons): Pyr. layer to plot.
    - sub_ax (plt.Axes): Subplot to add target and reset points to.
    - restore_xlims (bool, optional): Whether to restore x limits. Default is True.
    - lw (float, optional): Line width of the vertical lines. Default is 1.0.
    - omit_reset (bool, optional): Whether to omit reset points. Default is False.
    - min_steps_btw (float, optional): Minimum steps between points for plotting.
        Default is 0.0.
    """

    if restore_xlims:
        xlims = sub_ax.get_xlim()

    for position_name, ls in [
        ("reset", "dashed"),
        ("target", "dotted"),
    ]:
        if position_name == "reset" and omit_reset:
            continue

        steps = Pyrs.Agent.get_reached_position_steps(
            position_name=position_name, min_steps_btw=min_steps_btw
        )

        last_step_idxs = np.where(steps == len(Pyrs.Agent.history["t"]))[0]
        if len(last_step_idxs):
            steps[last_step_idxs] -= 1
            steps = np.unique(steps)

        time_min = Pyrs.Agent.dt * steps / 60
        for t in time_min:
            sub_ax.axvline(
                t,
                alpha=0.7,
                zorder=-1,
                lw=lw,
                ls=ls,
                color="k",
            )

    if restore_xlims:
        sub_ax.set_xlim(xlims)


def plot_loss(
    t: np.ndarray[tuple[int], np.dtype[np.float64]],
    loss: np.ndarray[tuple[int], np.dtype[np.float64]],
    mark_ts: list[float] | None = None,
    t_start: float | None = None,
    t_end: float | None = None,
    sub_ax: plt.Axes | None = None,
    color: str | None = None,
    alpha: float = 0.7,
    xlim: tuple[float, float] | None = None,
    k_prop_to_loss_length: float = 0.05,
    k_max: int = 10000,
    in_min: bool = True,
    loss_type: str = "MSE",
    test_p: float | None = None,
) -> plt.Axes:
    """
    plot_loss(t, loss)

    Plot the loss of the layer over time.

    Args:
    - t (1D np.ndarray): Time array.
    - loss (1D np.ndarray): Loss array.
    - mark_ts (list of floats, optional): Times to mark on the plot. Default is None.
    - t_start (float, optional): Start time of the plot. Default is None.
    - t_end (float, optional): End time of the plot. Default is None.
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.
    - color (str, optional): Color of the plot. Default is None.
    - alpha (float, optional): Alpha of the plot. Default is 0.7.
    - xlim (tuple, optional): x limits of the plot. Default is None.
    - k_prop_to_loss_length (float, optional): Smoothing factor, proportional to
        length of loss array. Default is 0.15.
    - k_max (int, optional): Maximum smoothing factor. Default is 10000.
    - in_min (bool, optional): Whether to plot time in minutes. Default is True.
    - loss_type (str, optional): Type of loss. Default is "MSE".
    - test_p (int, optional): Proportion of time at which the test set starts.
        Default is 0.

    Returns:
    - sub_ax (plt.Axes): Subplot with loss plotted.
    """

    t = np.asarray(t)

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=[6, 4])

    startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
    if in_min:
        t = t / 60

    loss = np.asarray(loss[startid : endid + 1])

    if np.isfinite(loss).any():
        nan_mask = np.isfinite(loss)
        sub_ax.plot(t[nan_mask], loss[nan_mask], color=color, alpha=alpha / 2)

        # smoothed the loss, accounting for NaNs
        k = max(1, int(k_prop_to_loss_length * len(loss[nan_mask])))
        if k_max > 10000:
            k = k_max
            new_k_prop_to_loss_length = k / len(loss[nan_mask])
            print(
                "Reducing 'k_prop_to_loss_length' from "
                f"{k_prop_to_loss_length} to {new_k_prop_to_loss_length} to obey "
                f"max k={k_max}."
            )

        if xlim is not None:
            sub_ax.set_xlim(xlim)

        test_loss_label = None
        if test_p is None:
            loss_smoothed = scipy.ndimage.convolve(
                loss[nan_mask], np.ones(k) / k, mode="reflect"
            )
            best_loss_idx = np.argmin(loss_smoothed)
            best_loss_smoothed = loss_smoothed[best_loss_idx]
            best_loss_proportional_to_start = (
                best_loss_smoothed / loss_smoothed[0] * 100
            )
            label = (
                f"min. MSE={best_loss_smoothed:.2f}\n"
                f"({best_loss_proportional_to_start:.2f}%)"
            )
            sub_ax.axvline(
                t[nan_mask][best_loss_idx], ls="dashed", lw=2, label=label, color=color
            )
        else:
            test_start_idx = len(t) - int(test_p * len(t))
            test_start_idx = max(0, test_start_idx - startid)

            test_mask = np.arange(len(loss)) > test_start_idx
            loss_smoothed = np.concatenate(
                [
                    scipy.ndimage.convolve(
                        loss[nan_mask * ~test_mask], np.ones(k) / k, mode="reflect"
                    ),
                    scipy.ndimage.convolve(
                        loss[nan_mask * test_mask], np.ones(k) / k, mode="reflect"
                    ),
                ]
            )
            num_test_pts = int(np.sum(nan_mask * test_mask))
            mean_prev_loss_smoothed = loss_smoothed[:-num_test_pts].mean()
            mean_test_loss_smoothed = loss_smoothed[-num_test_pts:].mean()
            test_percentage = mean_test_loss_smoothed / mean_prev_loss_smoothed * 100
            test_loss_label = (
                f"test MSE={mean_test_loss_smoothed:.2f} ({test_percentage:.1f}%)"
            )
            sub_ax.axvline(
                t[test_start_idx], ls="dashed", lw=2, label="test start", color=color
            )

        sub_ax.plot(
            t[nan_mask], loss_smoothed, color=color, alpha=alpha, label=test_loss_label
        )

        sub_ax.axhline(0, ls="dashed", lw=2, color="k", zorder=-10)
        legend = sub_ax.legend(fontsize="small")
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.8)

        if mark_ts is not None:
            for mark_t in mark_ts:
                if mark_t > t[startid] and mark_t < t[endid]:
                    sub_ax.axvline(
                        mark_t, ls="dotted", color="k", alpha=0.3, zorder=-13
                    )
    else:
        raise RuntimeError(f"No loss data to plot from {t[0]}s to {t[-1]}s.")

    sub_ax.spines[["top", "right"]].set_visible(False)

    xlabel = "Time (in min)" if in_min else "Time (in s)"
    sub_ax.set_xlabel(xlabel)
    sub_ax.set_ylabel(loss_type)

    plot_util.pad_axis(sub_ax)

    return sub_ax


def plot_oscillation_events(
    oscillation_df,
    firingrates,
    order_by="neuron_idx",
    reverse=False,
    max_num=1000,
    num_cols=15,
    sharey=True,
    axes=None,
    color="k",
):
    """
    plot_oscillation_events(oscillation_df, firingrates)

    Plot oscillation events in individual subplots.

    Args:
    - oscillation_df (pd.DataFrame): DataFrame of oscillation events.
    - firingrates (2D np.ndarray): Firing rates.
    - order_by (str, optional): Column to order by. Default is "neuron_idx".
    - reverse (bool, optional): Whether to reverse the order. Default is False.
    - max_num (int, optional): Maximum number of oscillations to plot. Default is 1000.
    - num_cols (int, optional): Number of columns. Default is 15.
    - sharey (bool, optional): Whether to share y axes. Default is True.
    - axes (np.ndarray, optional): Array of subplots to plot on. Default is None.
    - color (str, optional): Firing rate trace color. Default is "k".

    Returns:
    - sub_ax (plt.Axes): Subplot with oscillation events plotted.
    """

    num_osc_total = len(oscillation_df)

    num_osc = min(num_osc_total, max_num)
    if axes is None:
        num_cols = max(1, min(num_osc, 15))
        num_rows = max(1, int(np.ceil(num_osc / num_cols)))
        figsize = (num_cols * 0.6, num_rows * 0.6)
        _, axes = plt.subplots(
            num_rows, num_cols, figsize=figsize, sharey=sharey, squeeze=False
        )
    else:
        if np.asarray(axes).shape == 0:
            raise ValueError("axes must be a 1 or 2D array.")
        num_rows = len(axes)
        if len(axes.ravel()) < num_osc:
            raise RuntimeError(
                f"Insufficient number of subplots ({len(axes.ravel())}) provided "
                f"for {num_osc} oscillations."
            )

    if not isinstance(order_by, list):
        order_by = [order_by]
    for column in order_by:
        if column not in oscillation_df.columns:
            raise ValueError(f"{column} not found in oscillation_df.")
    if "start_frame" not in order_by:
        order_by.append("start_frame")  # avoids arbitrary sub-sorting

    oscillation_df = oscillation_df.sort_values(order_by)
    indices = oscillation_df.index
    if reverse:
        indices = indices[::-1]

    if num_osc_total > max_num:
        title = f"First {num_osc}/{num_osc_total} oscillations"
    else:
        title = f"{num_osc} oscillations"
    num_neurons = len(oscillation_df.loc[indices[:num_osc], "neuron_idx"].unique())
    num_neurons_total = firingrates.shape[1]

    title = f"{title} (from {num_neurons}/{num_neurons_total} neurons)"
    y = 0.55 * np.exp(-0.25 * num_rows) + 0.89  # works ok to about 80 rows
    axes.ravel()[-1].figure.suptitle(title, y=y)

    legend_kwargs = {
        "handletextpad": -0.1,
        "handlelength": 0,
        "loc": "upper center",
        "fontsize": 5,
    }

    prev_neuron_idx = -1
    for i, row in enumerate(np.arange(num_osc)):
        sub_ax = axes.ravel()[i]
        df_row = oscillation_df.loc[indices[row]]
        neuron_idx = df_row["neuron_idx"]
        neuron_sub_idx = df_row["neuron_sub_idx"]
        start, stop = df_row["start_frame"], df_row["stop_frame"]

        sub_ax.plot(firingrates[start:stop, neuron_idx], color=color)

        if order_by[0] == "neuron_idx" and neuron_idx != prev_neuron_idx:
            num = len(oscillation_df.loc[oscillation_df["neuron_idx"] == neuron_idx])
            sub_ax.plot([], [], label=f"#{neuron_idx} ({num})")
            sub_ax.legend(**legend_kwargs)
            prev_neuron_idx = neuron_idx

    for sub_ax in axes.ravel():
        sub_ax.axis("off")

    return axes


def plot_with_marked_oscillations(
    oscillation_df,
    firingrates,
    t=None,
    norm_height=0.8,
    sub_ax=None,
    color="k",
):
    """
    plot_with_marked_oscillations(oscillation_df, firingrates)

    Plot firing rates with oscillations marked with shading.

    Args:
    - oscillation_df (pd.DataFrame): DataFrame of oscillations.
    - firingrates (2D np.ndarray): Firing rates.
    - t (1D np.ndarray, optional): Time array. Default is None.
    - norm_height (float, optional): Normalized height of the plot. Default is 0.8.
    - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
    - color (str, optional): Firing rate trace and shading color. Default is "k".

    Returns:
    - sub_ax (plt.Axes): Subplot with firing rates and oscillations plotted.
    """

    norm_firingrates = signal_util.get_norm_data(firingrates, axis=0) * norm_height
    num_frames, num_neurons = firingrates.shape

    if t is None:
        t = np.arange(num_frames)
        xlabel = "Frame"
    elif len(t) != num_frames:
        raise ValueError("Time array must have the same length as the firing rates.")
    else:
        xlabel = "Time (s)"

    if sub_ax is None:
        height = np.min([8, np.max([2, firingrates.shape[1] * 0.2])])
        _, sub_ax = plt.subplots(figsize=(8, height))
    elif not isinstance(sub_ax, plt.Axes):
        raise ValueError("sub_ax must be a matplotlib Axes object.")

    oscillation_df = oscillation_df.loc[
        (oscillation_df["start_frame"] < len(t)) & (oscillation_df["stop_frame"] > 0)
    ]

    if len(oscillation_df):
        oscillation_df.loc[oscillation_df["start_frame"] < 0, "start_frame"] = 0
        oscillation_df.loc[oscillation_df["stop_frame"] > len(t), "stop_frame"] = len(t)

    for i in range(num_neurons):
        sub_df = oscillation_df[oscillation_df["neuron_idx"] == i]

        data_to_plot = norm_firingrates[:, i] + i
        sub_ax.plot(t, data_to_plot, c=color, alpha=1.0)

        sub_ax.fill_between(
            t,
            np.full_like(data_to_plot, i),
            data_to_plot,
            color=color,
            alpha=0.3,
            lw=0,
        )

        for startid, endid in zip(sub_df["start_frame"], sub_df["stop_frame"]):
            startid = max(0, startid)
            endid = min(len(t), endid)

            width = t[endid - 1] - t[startid]
            rectangle = mpl_patches.Rectangle(
                (t[startid], i),
                width,
                norm_height,
                alpha=0.3,
                color="red",
                lw=0,
                zorder=-2,
            )
            sub_ax.add_patch(rectangle)

    sub_ax.set_title("Firing rates with oscillations marked")
    sub_ax.set_ylabel("Normalized firing rates")
    sub_ax.set_xlabel(xlabel)
    sub_ax.set_yticks([])

    if len(oscillation_df):
        sub_ax.set_ylim(0, num_neurons)
        plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.05)

    sub_ax.spines[["top", "right", "left"]].set_visible(False)

    return sub_ax


def plot_oscillations(
    oscillation_df,
    firingrates,
    norm_height=0.8,
    aligned=True,
    pad_prop=0.3,
    sub_ax=None,
    color="k",
):
    """
    plot_oscillations(oscillation_df, firingrates)

    Plot oscillations on a firing rate plot.

    Args:
    - oscillation_df (pd.DataFrame): DataFrame of oscillations.
    - firingrates (2D np.ndarray): Firing rates.
    - norm_height (float, optional): Normalized height of the plot. Default is 0.8.
    - aligned (bool, optional): Whether to align the oscillations. Default is True.
    - pad_prop (float, optional): Proportion of padding to use. Default is 0.1.
    - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
    - color (str, optional): Firing rate trace and shading color. Default is "k".

    Returns:
    - sub_ax (plt.Axes): Subplot with oscillations plotted
    """

    norm_firingrates = signal_util.get_norm_data(firingrates, axis=0) * norm_height
    num_frames, num_neurons = firingrates.shape

    if sub_ax is None:
        height = np.min([8, np.max([2, firingrates.shape[1] * 0.2])])
        _, sub_ax = plt.subplots(figsize=(8, height))
    elif not isinstance(sub_ax, plt.Axes):
        raise ValueError("sub_ax must be a matplotlib Axes object.")

    oscillation_df = oscillation_df.loc[
        (oscillation_df["start_frame"] < num_frames)
        & (oscillation_df["stop_frame"] > 0)
    ]
    oscillation_df.loc[oscillation_df["start_frame"] < 0, "start_frame"] = 0
    oscillation_df.loc[oscillation_df["stop_frame"] > num_frames, "stop_frame"] = (
        num_frames
    )

    frames_to_plot = list()
    for i in range(num_neurons):
        sub_df = oscillation_df[oscillation_df["neuron_idx"] == i]
        start_frames, stop_frames = (
            sub_df["start_frame"].to_numpy(),
            sub_df["stop_frame"].to_numpy(),
        )
        if len(start_frames) == 0:
            continue

        if aligned:
            frames = np.concatenate(
                [np.arange(st, en) for st, en in zip(start_frames, stop_frames)]
            )
            frames_to_plot.append(frames)
        else:
            frames_to_plot.append(np.sum(stop_frames - start_frames))

    if len(frames_to_plot):
        if aligned:
            frames_to_plot = np.sort(np.unique(np.concatenate(frames_to_plot)))
            frames_to_plot = signal_util.pad_throughout(
                frames_to_plot, pad_prop=pad_prop, min_val=0, max_val=num_frames
            )
            num_frames_to_plot = len(frames_to_plot)
        else:
            num_frames_to_plot = int(np.ceil(np.max(frames_to_plot) * (1 + pad_prop)))
    else:
        num_frames_to_plot = 0

    full_array = np.full((num_frames_to_plot, num_neurons), np.nan)
    for i in range(num_neurons):
        if num_frames_to_plot == 0:
            continue
        sub_df = oscillation_df[oscillation_df["neuron_idx"] == i]

        indices = list()
        for startid, endid in zip(sub_df["start_frame"], sub_df["stop_frame"]):
            startid = max(0, startid)
            endid = min(num_frames, endid)
            indices.append(np.arange(startid, endid))

        if not len(indices):
            continue

        indices = np.sort(np.unique(np.concatenate(indices)))
        if aligned:
            sub_idxs = np.where(np.isin(frames_to_plot, indices))[0]
        else:
            pad_prop = num_frames_to_plot / len(indices) - 1
            padded_idxs = signal_util.pad_throughout(
                indices, pad_prop=pad_prop, min_val=0, max_val=num_frames
            )

            if len(padded_idxs) > num_frames_to_plot:
                raise RuntimeError("Padding extended number of frames to plot too far.")

            sub_idxs = np.where(np.isin(padded_idxs, indices))[0]

        full_array[sub_idxs, i] = norm_firingrates[indices, i]

    x = np.arange(num_frames_to_plot)
    for i in range(num_neurons):
        if num_frames_to_plot == 0:
            continue

        data_to_plot = full_array[:, i] + i
        if np.isnan(data_to_plot).all():
            continue

        sub_ax.plot(x, data_to_plot, c=color, alpha=1.0)

        data_to_plot[np.isnan(data_to_plot)] = i
        sub_ax.fill_between(
            x,
            np.full_like(data_to_plot, i),
            data_to_plot,
            color=color,
            alpha=0.3,
            lw=0,
        )

    xlabel = f"Frame ({100 * num_frames_to_plot / num_frames:.1f} %)"

    sub_ax.set_title("Oscillation segments")
    sub_ax.set_ylabel("Normalized firing rates")
    sub_ax.set_xlabel(xlabel)
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])

    if num_frames_to_plot:
        sub_ax.set_ylim(0, num_neurons)
        plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.05)

    sub_ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    return sub_ax


def plot_property_at_BTSP_and_closest_to_target_steps(
    steps_dict,
    y_data,
    x_data=None,
    x_data_type="Step",
    y_data_type="Distance",
    sub_ax=None,
    legend=True,
    scale_size=True,
):
    """
    plot_property_at_BTSP_and_closest_to_target_steps(steps_dict, y_data)

    Plot the property at the BTSP and the closest steps to the target.

    Args:
    - steps_dict (dict): Dictionary of steps with keys:
        - "steps_before" (list): Steps before the first BTSP.
        - "steps_near_BTSP" (list): Steps near a BTSP event.
        - "steps_of_nearest_BTSP" (list): Steps of the nearest BTSP.
        - "steps_other" (list): Other steps.
        - "other_BTSP_steps" (list): Other BTSP steps.
    - y_data (1D np.ndarray): Data to plot.
    - x_data (1D np.ndarray, optional): x data to plot. Default is None.
    - x_data_type (str, optional): x data type. Default is "Step".
    - y_data_type (str, optional): y data type. Default is "Distance".
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.
    - legend (bool, optional): Whether to add a legend. Default is True.
    - scale_size (bool, optional): Whether to scale the marker size. Default is True.

    Returns:
    - sub_ax (plt.Axes): Subplot with the property at the BTSP and the closest steps
        to the target plotted.
    """

    all_steps = np.sort(np.concatenate([steps for steps in steps_dict.values()]))

    if scale_size:
        if len(all_steps) < 2 or (all_steps.max() == all_steps.min()):
            factors = np.array([1])
        else:
            factors = (all_steps - all_steps.min()) / (
                all_steps.max() - all_steps.min()
            ) + 0.5

    if x_data is None and len(all_steps):
        x_data = np.arange(all_steps.max() + 1)

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=[2.6, 2])

    for key, steps in steps_dict.items():
        plot_kwargs = plot_util.get_closest_step_marker_kwargs(key)
        base_s = plot_kwargs["s"]
        for step in steps:
            if scale_size:
                plot_kwargs["s"] = base_s * factors[np.where(all_steps == step)[0][0]]

            sub_ax.scatter(x_data[step], y_data[step], **plot_kwargs)

        if legend and len(steps):
            label = key.replace("_", " ").replace("steps ", "")
            plot_kwargs["s"] = base_s
            sub_ax.scatter([], [], label=label, **plot_kwargs)

    line_kwargs = plot_util.get_closest_step_marker_kwargs(
        "steps_of_nearest_BTSP", plot_line=True
    )
    for steps in zip(
        steps_dict["steps_near_BTSP"], steps_dict["steps_of_nearest_BTSP"]
    ):
        sub_ax.plot(
            [x_data[step] for step in steps],
            [y_data[step] for step in steps],
            **line_kwargs,
        )

    sub_ax.spines[["right", "top"]].set_visible(False)
    plot_util.pad_axis(sub_ax, axis="x")
    plot_util.pad_axis(sub_ax, axis="y")

    sub_ax.set_xlabel(x_data_type)
    sub_ax.set_ylabel(y_data_type)

    if legend and len(all_steps):
        sub_ax.legend(fontsize=5, loc="upper right")

    return sub_ax


def plot_time_series_with_BTSP_events(
    Pyrs: "learning_neurons.BTSPLayer",
    ax1D: np.ndarray | None = None,
) -> np.ndarray:
    """
    plot_time_series_with_BTSP_events(Pyrs)

    Plot the time series of the Pyrs layer with BTSP events marked.

    Args:
    - Pyrs (learning_neurons.BTSPLayer): Pyrs layer.
    - ax1D (np.ndarray, optional): 1D array of subplots. Default is None.

    Returns:
    - ax1D (np.ndarray, optional): 1D array of subplots with time series and BTSP
        events plotted.
    """

    Objs = ext_util.extract_objects_from_Pyrs(Pyrs)[-1]
    num_rows = 2 if Objs is None else 3

    if ax1D is None:
        height = 1.3**Pyrs.n + (num_rows - 1) / 2
        _, ax1D = plt.subplots(num_rows, figsize=(6, height), sharex=True)

    title_str = "Pyr."
    i = 0
    if Objs is not None:
        Objs.plot_rate_timeseries(sub_ax=ax1D[0])
        ax1D[0].axis("off")
        title_str = "Object and Pyr."
        i = 1

    ax1D[0].set_title(
        f"{title_str} time series with BTSP events (with proximity to target)", y=1.1
    )

    Pyrs.plot_rate_timeseries(chosen_neurons="all", spikes=True, sub_ax=ax1D[i])
    ax1D[i].set_xlabel("")
    lo, hi = ax1D[i].get_ylim()

    target_reached_step = Pyrs.Agent.target_df["reached_step"].to_numpy()  # type: ignore[attr-defined]
    if np.isnan(target_reached_step[-1]):
        target_reached_step = target_reached_step[:-1]
    target_reached_step = target_reached_step.astype(int)

    for t in target_reached_step:
        y_hei = lo + (hi - lo) * 0.82
        ax1D[i].scatter(
            Pyrs.Agent.history["t"][t] / 60,
            y_hei,
            marker=mpl_markers.MarkerStyle("o"),
            s=6,
            color="k",
            alpha=0.7,
        )

    # add distance from target below
    Pyrs.Agent.plot_distance_to_target(
        norm=True, flipped=True, sub_ax=ax1D[-1], autosave=False
    )
    ax1D[-1].set_title("")
    ax1D[-1].set_yticks([])

    return ax1D


def plot_timeseries(
    NeuronLayer: "riab_neurons.Neurons",
    t_start: float | None = None,
    t_end: float | None = None,
    chosen_neurons: (
        str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
    ) = "all",
    spikes: bool = False,
    imshow: bool = False,
    sub_ax: plt.Axes | None = None,
    xlim: tuple[float, float] | None = None,
    color: str | None = None,
    background_color: str | None = None,
    trace_name: str = "firingrate",
    in_min: bool = True,
    autosave=None,
    **kwargs,
):
    """
    plot_timeseries(NeuronLayer)

    Plot the rate timeseries of a layer of neurons.

    Args:
    - NeuronLayer (riab_neurons.Neurons): The layer of neurons to plot.
    - t_start (float, optional): The start time of the plot. Default is None.
    - t_end (float, optional): The end time of the plot. Default is None.
    - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
        Default is "all".
    - spikes (bool, optional): Whether to plot spikes or firing rate. Default is False.
    - imshow (bool, optional): Whether to plot the timeseries as an image.
        Default is False.
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.
    - xlim (tuple[float, float], optional): The x limits of the plot. Default is None.
    - color (str, optional): The color of the plot. Default is None.
    - background_color (str, optional): The background color of the plot.
        Default is None.
    - trace_name (str, optional): The name of the trace to plot.
        Default is "firingrate".
    - in_min (bool, optional): Whether to plot time in minutes. Default is True.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with timeseries of the layer plotted.
    """

    t = np.asarray(NeuronLayer.history["t"])
    startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
    if in_min:
        t = t / 60

    t_start = t[startid]
    t_end = t[endid]

    rate_timeseries = np.asarray(NeuronLayer.history[trace_name][startid : endid + 1])

    # neurons to plot
    if not isinstance(chosen_neurons, np.ndarray):
        chosen_neurons = np.asarray(NeuronLayer.return_list_of_neurons(chosen_neurons))  # type: ignore[arg-type]
    rate_timeseries = rate_timeseries[:, chosen_neurons]

    was_ax = (
        sub_ax is None
    )  # remember whether a subplot was provided as xlims depend on this
    fig = None if sub_ax is None else sub_ax.figure

    if color is None:
        color = NeuronLayer.color  # type: ignore[attr-defined]

    time_str = "Time (s)" if not in_min else "Time (min)"
    if imshow == False:
        firingrates = rate_timeseries.T
        _, sub_ax = rutils.mountain_plot(
            X=t,
            NbyX=firingrates,
            color=color,  # type: ignore[assignment]
            xlabel=time_str,
            ylabel="Neurons",
            xlim=None,
            fig=fig,
            ax=sub_ax,
            **kwargs,
        )

        if sub_ax is None:
            raise RuntimeError("sub_ax is None.")

        if spikes == True:
            spike_data = np.asarray(NeuronLayer.history["spikes"][startid : endid + 1])[
                :, chosen_neurons
            ]
            for i in range(len(chosen_neurons)):
                time_when_spiked = t[spike_data[:, i]]
                h = (i + 1 - 0.1) * np.ones_like(time_when_spiked)
                sub_ax.scatter(
                    time_when_spiked,
                    h,
                    color=(NeuronLayer.color or "C1"),  # type: ignore[attr-defined]
                    alpha=0.5,
                    s=5,
                    lw=0,
                )

        xmin = t[0] if was_ax else min(t[0], sub_ax.get_xlim()[0])  # type: ignore[operator]
        xmax = t[-1] if was_ax else max(t[-1], sub_ax.get_xlim()[1])  # type: ignore[operator]
        sub_ax.set_xlim(xmin, xmax)
        sub_ax.set_xticks([xmin, xmax])
        sub_ax.set_xticklabels([round(xmin, 2), round(xmax, 2)])
        if xlim is not None:
            sub_ax.set_xlim(right=xlim)  # type: ignore[operator]
            sub_ax.set_xticks([round(t_start, 2), round(xlim, 2)])  # type: ignore[operator]
            sub_ax.set_xticklabels([round(t_start, 2), round(xlim, 2)])  # type: ignore[operator]

        if background_color is not None:
            sub_ax.set_facecolor(background_color)
            sub_ax.figure.patch.set_facecolor(background_color)  # type: ignore[attr-defined]

    elif imshow == True:
        if sub_ax is None:
            _, sub_ax = plt.subplots(
                figsize=(
                    MOUNTAIN_PLOT_WIDTH_MM / 25,
                    0.5 * MOUNTAIN_PLOT_WIDTH_MM / 25,
                )
            )

        data = rate_timeseries.T
        sub_ax.imshow(
            data[::-1],
            aspect="auto",
            # aspect=0.5 * data.shape[1] / data.shape[0],
            extent=(t_start, t_end, 0, 1),  # type: ignore[assignment]
        )
        sub_ax.spines[["right", "top", "left"]].set_visible(False)
        sub_ax.set_xlabel(time_str)
        sub_ax.set_xticks([t_start, t_end])
        sub_ax.set_xticklabels([round(t_start, 2), round(t_end, 2)])  # type: ignore[operator]
        sub_ax.set_yticks([])
        sub_ax.set_ylabel("Neurons")

    fig = sub_ax.figure
    plot_util.save_figure(fig, f"{NeuronLayer.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

    return sub_ax


def plot_1D_reset_environment(
    Ag: "ResetableAgent",
    title: str = "Environment",
    minimalist: bool = False,
    base_s: float = 15,
    obj_lw: float = 1,
    sub_ax: plt.Axes | None = None,
    autosave: bool | None = None,
) -> plt.Axes:
    """
    plot_1D_reset_environment(Ag)

    Plot the 1D environment of the agent.

    Args:
    - Ag (Agent): Agent for which to plot the environment, with reset and
        start points marked.
    - title (str, optional): Title of the plot. Default is "Environment".
    - minimalist (bool, optional): Whether to create minimalist reset environment plot.
        Default is False.
    - base_s (float, optional): Base size for markers. Default is 15.
    - obj_lw (float, optional): Line width for objects. Default is 1.
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.
    - autosave (bool, optional): Whether to save the figure. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with 1D reset environment plotted.
    """

    plotting_dict = {
        "start": {"data": Ag.start_position},
        "reset": {"data": Ag.reset_position},
        "target": {"data": Ag.target_position},
    }

    for label in plotting_dict.keys():
        plotting_dict[label]["kwargs"] = plot_util.get_plot_marker_kwargs(
            label, base_s=base_s
        )

    if minimalist and sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(4, 0.7))

    sub_ax = Ag.Environment.plot_environment(
        sub_ax=sub_ax, plot_objects=False, autosave=False
    )

    if sub_ax is None:
        raise RuntimeError("sub_ax is None.")

    edges = list()
    for label, sub_dict in plotting_dict.items():
        if sub_dict["data"] is None:
            continue
        if minimalist and label in ["start", "reset"]:
            edges.append(sub_dict["data"][0])
        sub_ax.scatter(
            sub_dict["data"],
            0,
            zorder=5,
            label=label,
            lw=obj_lw,
            **sub_dict["kwargs"],
        )
    sub_ax.legend(ncol=len(plotting_dict), frameon=False)
    sub_ax.set_title(title)

    if len(edges) == 2:
        sub_ax.spines["bottom"].set_bounds(sorted(edges))
        plot_util.pad_axis(sub_ax, axis="x", pad_prop=0.05)

    if minimalist:
        sub_ax.set_xticks([])
        sub_ax.set_xlabel("")
        sub_ax.spines["bottom"].set_linewidth(1.5)

    fig = sub_ax.figure
    plot_util.save_figure(fig, "1D_reset_environment", save=autosave)

    return sub_ax


def plot_1D_rate_map_across_learning(
    Pyrs: "riab_neurons.Neurons",
    axes: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | None = None,
    plot_proportion: float = 0.3,
    min_num_steps: int = 100,
    norm_by: str | float | None = None,
    autosave: bool | None = None,
) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
    """
    plot_1D_rate_map_across_learning(Pyrs)

    Plot the rate map of a layer across learning from stimulated activity in the
    first, mid and final steps.

    Args:
    - Pyrs (riab_neurons.Neurons): Pyr. layer to plot.
    - axes (1 or 2D np.ndarray, optional): Array of subplots to plot on (3 total).
        Default is None.
    - plot_proportion (float, optional): Proportion of the total steps to plot.
        Default is 0.3.
    - min_num_steps (int, optional): Minimum number of steps to plot. Default is 100.
    - norm_by (str, optional): Normalization method for rate maps. If None, default is
        used. Default is None.
    - autosave (bool, optional): Whether to save the figure. Default is None.

    Returns:
    - axes (2D np.ndarray, optional): Array of subplots to plot on. If input axes
        was None, shape is 2D (3, 1).
    """

    map_start_pts = ["first", "mid", "final"]  # each subplot

    if axes is None:
        figsize = plot_util.get_figsize(len(map_start_pts), squat_height=True)
        _, axes = plt.subplots(
            nrows=len(map_start_pts), figsize=figsize, sharex=True, squeeze=False
        )

    ax1D = np.asarray(axes).ravel()

    if len(ax1D) != len(map_start_pts):
        raise ValueError(
            f"Number of axes must match number of start points ({len(map_start_pts)})."
        )

    kwargs = dict()
    if norm_by is not None:
        kwargs["norm_by"] = norm_by

    suptitle = f"Pyramidal rate maps across learning: "

    Ag = Pyrs.Agent

    dt = float(Ag.dt)
    for s, start in enumerate(map_start_pts):
        t = Ag.num_steps_total * dt
        num_seconds_to_plot = t * plot_proportion
        if start == "first":
            t_start = 0
        elif start == "mid":
            t_start = min((t - num_seconds_to_plot) / 2, t - min_num_steps * dt)
        else:
            t_start = min(t - num_seconds_to_plot, t - min_num_steps * dt)
        t_start = max(0, t_start)
        t_end = min(Ag.num_steps_total + 1, t_start + num_seconds_to_plot)

        no_legend = s != len(map_start_pts) - 1
        Pyrs.plot_rate_map(
            chosen_neurons="all",
            ax=ax1D[s],
            t_start=t_start,  # type: ignore[type-arg]
            t_end=t_end,
            shift=-10,
            overlap=1,
            method="history",
            autosave=False,
            no_legend=no_legend,
            **kwargs,
        )

        lead = suptitle if s == 0 else ""
        ax1D[s].set_title(f"{lead}{start} ({t_start:.1f} to {t_end:.1f}s)")

        if s != len(map_start_pts) - 1:
            ax1D[s].set_xlabel("")
            # turn off x-axis
            ax1D[s].spines["bottom"].set_visible(False)
            ax1D[s].xaxis.set_visible(False)

        if s != len(map_start_pts) // 2:
            ax1D[s].set_ylabel("")

    fig = ax1D[0].figure
    plot_util.save_figure(fig, f"{Pyrs.name}_1D_rate_map_across_learning", save=autosave)  # type: ignore[attr-defined]

    return axes


def add_1D_PF_widths(
    sub_ax,
    PF: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    PF_positions: np.ndarray | None = None,
    PCs: "riab_neurons.PlaceCells" = None,
    Ag: "ResetableAgent" = None,
    prop_peak: float = signal_util.DFT_PROP_PEAK,
    time_axis: bool = False,
    time_shift: float = 0.0,
    **kwargs,
):
    """
    add_1D_PF_widths(sub_ax, PF)

    Add the width of the place fields to a 1D plot.

    Args:
    - sub_ax (plt.Axes): Subplot to add the width to.
    - PF (1D np.ndarray): Place field.
    - PF_positions (1D np.ndarray, optional): Positions for the place fields. If None,
        place cell centers are used. Default is None.
    - PCs (riab_neurons.PlaceCells): Place cells of the layer. Only required if
        PF_positions is not provided. Default is None.
    - Ag (ResetableAgent, optional): Agent for which to plot the environment. Only
        required if time_axis is True and PCs is not provided. Default is None.
    - prop_peak (float): Proportion of the peak to use for width computation.
        Default is signal_util.DFT_PROP_PEAK.
    - time_axis (bool, optional): Whether to plot the x-axis in time. Default is False.
    - time_shift (float, optional): Time shift to apply to the x-axis if time_axis is
        True. Default is 0.0.

    Keyword args:
    - **kwargs: Additional keyword arguments passed to plt.axvspan().
    """

    if Ag is None and PCs is not None:
        Ag = PCs.Agent

    if Ag is not None and Ag.Environment.D != 1:
        raise ValueError("Function is implemented for 1D environments.")

    PF = np.asarray(PF)
    if len(PF.shape) != 1:
        raise ValueError("PF must be a 1D array.")

    if PF_positions is None:
        if PCs is None:
            raise ValueError("PCs must be provided if PF_positions is None.")
        PF_positions = PCs.place_cell_centers[:, 0]

    if len(PF_positions) != len(PF):
        raise ValueError(
            f"PF_positions must have the same length ({len(PF_positions)}) as the "
            f"number of values per place field ({len(PF)})."
        )

    if Ag is None:
        max_pos = PF_positions.max()
    else:
        max_pos = Ag.Environment.scale

    unit = "m"
    if time_axis:
        if Ag is None:
            raise ValueError("Ag must be provided if time_axis is True.")
        speed_mean = Ag.speed_mean  # m/s
        PF_positions = PF_positions / speed_mean - time_shift
        max_pos = max_pos / speed_mean
        unit = "s"

    width, edges = signal_util.compute_signal_width(
        PF, PF_positions, max_x=max_pos, prop_peak=prop_peak, return_edges=True
    )
    label = f"Width={width:.2f} {unit}"

    end_pts = [0, max_pos]
    if np.isclose(edges[0], edges[1]):
        edges = end_pts
    plot_util.plot_vspan_circular(
        sub_ax=sub_ax, edges=edges, end_pts=end_pts, label=label, lw=0, **kwargs
    )


def plot_1D_PFs(
    PFs: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    PCs: "riab_neurons.PlaceCells" = None,
    PF_type: str = "weights",
    PF_positions: np.ndarray | None = None,
    Ag: "ResetableAgent" = None,
    cmap: str = "crest",
    time_axis: bool = False,
    plot_last_width: bool = False,
    color: str | None = None,
    sub_ax: plt.Axes | None = None,
    autosave: bool | None = None,
) -> plt.Axes:
    """
    plot_1D_PFs(PFs)

    Plot the place fields of a layer.

    Args:
    - PFs (2 or 3D np.ndarray): Place fields across epochs with shape
        ((num_epochs,) num_cells, num_PCs).
    - PCs (riab_neurons.PlaceCells): Place cells of the layer. Only required if
        PF_positions is not provided. Default is None.
    - PF_type (str, optional): Type of place field to plot. Default is "weights".
    - PF_positions (1D np.ndarray, optional): Positions for the place fields. If None,
        place cell centers are used. Default is None.
    - Ag (ResetableAgent, optional): Agent for which to plot the environment. Only
        required if time_axis is True and PCs is not provided. Default is None.
    - cmap (str, optional): Colormap to use. Default is "crest".
    - time_axis (bool, optional): Whether to plot the x-axis in time. Default is False.
    - plot_last_width (bool, optional): Whether to plot the last width of the place cells.
        Default is False.
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.
    - autosave (bool, optional): Whether to save the figure. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with 1D place fields plotted.
    """

    PFs = np.asarray(PFs)

    if Ag is None and PCs is not None:
        Ag = PCs.Agent

    if len(PFs.shape) == 2:
        single_sample = True
        PFs = PFs[np.newaxis]
    else:
        single_sample = False

    num_samples, num_cells, _ = PFs.shape
    if num_cells > 1 and plot_last_width:
        raise NotImplementedError(
            "plot_last_width is not implemented for multiple cells."
        )

    if sub_ax is None:
        figsize = (5, num_cells * 2)
        _, sub_ax = plt.subplots(figsize=figsize, sharex=True)

    title = get_PF_label(PF_type, title=True)
    ylabel = get_PF_label(PF_type, title=False)

    if "weights" in PF_type and PCs is not None and color is None:
        color = PCs.color
    color = color or "k"

    if single_sample:
        colors = [color]
        alpha = 0.8
    else:
        cmap_vals = np.linspace(0, 1, num_samples)
        mpl_cmap = sns.color_palette(cmap, as_cmap=True)
        colors = mpl_cmap(cmap_vals)  # type: ignore[callable]
        alpha = 0.8 ** (len(colors) / 6)

    xlabel = "Position (m)"
    if PF_positions is None:
        if PCs is None:
            raise ValueError("PCs must be provided if PF_positions is provided.")
        PF_positions = PCs.place_cell_centers[:, 0]
        xlabel = "Input place cell center (m)"

    if len(PF_positions) != PFs.shape[2]:
        raise ValueError(
            f"PF_positions must have the same length ({PF_positions.shape[0]}) as the "
            f"number of values per place field ({PFs.shape[2]})."
        )

    target_pos = None
    if Ag is not None:
        target_pos = Ag.target_position
    time_shift = 0
    if time_axis:
        if Ag is None:
            raise ValueError("Ag must be provided if time_axis is True.")
        xlabel = "Average time (s)"
        speed_mean = Ag.speed_mean  # m/s
        PF_positions = PF_positions / speed_mean
        if target_pos is not None:
            time_shift = target_pos / speed_mean
            PF_positions = PF_positions - time_shift
            target_pos = 0

    legend = plot_last_width
    if not single_sample:
        if len(colors) > 1:
            legend = True
            sub_ax.plot([], color=colors[0], label="first", alpha=0.8)
            sub_ax.plot([], color=colors[-1], label="last", alpha=0.8)

        title = f"{title} across learning"

    spacing = (np.nanmax(PFs) - np.nanmin(PFs)) * 1.1
    for n in range(num_cells):
        offset = None
        for i, color in enumerate(colors):
            offset = spacing * n
            sub_ax.plot(
                PF_positions,
                PFs[i, n] + offset,
                color=color,
                alpha=alpha,
                marker=mpl_markers.MarkerStyle("."),
                ms=4,
            )

            if plot_last_width and i == len(colors) - 1:
                add_1D_PF_widths(
                    sub_ax,
                    PFs[i, n],
                    PF_positions=PF_positions,
                    PCs=PCs,
                    time_axis=time_axis,
                    time_shift=time_shift,
                    color=color,
                    alpha=0.2,
                    zorder=-3,
                )

        if offset is not None:
            lw = 2 * 0.8**num_cells
            sub_ax.axhline(offset, color="k", alpha=0.3, lw=lw, ls="dotted", zorder=-12)

    if target_pos is not None:
        sub_ax.axvline(
            target_pos,
            alpha=0.7,
            zorder=-1,
            lw=1,
            ls="dotted",
            color="k",
        )

    sub_ax.set_xlabel(xlabel)
    sub_ax.set_ylabel(ylabel)
    sub_ax.spines[["top", "right"]].set_visible(False)

    sub_ax.set_title(title)
    if legend:
        sub_ax.legend(ncol=2, frameon=False)

    # pad the y-axis
    ylims = sub_ax.get_ylim()
    pad = (ylims[1] - ylims[0]) * 0.05
    sub_ax.set_ylim(ylims[0] - pad, ylims[1] + pad)

    fig = sub_ax.figure
    plot_util.save_figure(fig, f"1D_{PF_type}_PFs", save=autosave)  # type: ignore[attr-defined]

    return sub_ax


def plot_recorded_1D_PFs(
    PFs,
    PF_positions,
    PFs_t=None,
    PF_type="weights",
    color="k",
    marker="o",
    ms=2,
    lw=1.2,
    ls=None,
    t_start=None,
    t_end=None,
    plot_last_width=False,
    k=1,
    plot_smoothed=False,
    no_legend=False,
    sub_ax=None,
):
    """
    plot_recorded_1D_PFs(PFs, PF_positions)

    Plots the place fields of a Pyr. soma, across time, for a linear track environment.

    Args:
    - PFs (2D np.ndarray): Place fields of a target neuron, for each timepoint,
        with shape (timepoints, num_inputs).
    - PF_positions (1D np.ndarray): Positions for the place field values.
    - PFs_t (list): List of timepoints for the PFs. Must have the same
        length as PFs if provided. Default is None.
    - color (str, optional): Color. Default is "k".
    - marker (str, optional): Marker style. Default is "o".
    - ms (int, optional): Marker size. Default is 2.
    - lw (float, optional): Line width. Default is 1.2.
    - ls (str, optional): Line style. Default is None.
    - t_start (float, optional): Start timepoint for which to plot PFs.
        Default is None.
    - t_end (float, optional): End timepoint for which to plot PFs. Default is None.
    - plot_last_width (bool, optional): Whether to plot the last width of the PFs.
        Default is False.
    - k (int, optional): Smoothing factor for calculating width of PFs. Default is 1.
    - plot_smoothed (bool, optional): Whether to plot a smoothed version of the PFs.
        Default is False.
    - no_legend (bool, optional): Whether to not plot the legend. Default is False.
    - sub_ax (plt.Axes, optional): Subplot. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot on which PFs are plotted.
    """

    if len(PFs.shape) != 2:
        raise ValueError("PFs must be a 2D array.")

    if PFs_t is not None and len(PFs_t) != len(PFs):
        raise ValueError(
            f"Length of 'PFs' ({len(PFs)}) and 'PFs_t' ({len(PFs_t)}) "
            "must be the same."
        )

    if len(PF_positions.shape) == 2:
        if PF_positions.shape[1] != 1:
            raise ValueError(
                "PF_positions must be a 1D array or a 2D array with one column."
            )
        PF_positions = PF_positions[:, 0]
    elif len(PF_positions.shape) > 2:
        raise ValueError("PF_positions must be a 1D or 2D array.")

    if PF_type == "weights":
        xlabel = "Input place field center (m)"
        ylabel = "Input weight"
    elif PF_type == "history":
        xlabel = "Position (m)"
        ylabel = "Firing rate"
    else:
        raise ValueError("PF_type must be 'weights' or 'history'.")

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=[6, 1])

    if t_start is None or PFs_t is None:
        start_idx = 0
    else:
        start_idx = max(0, np.where(np.asarray(PFs_t) > t_start)[0][0] - 1)

    if t_end is None or PFs_t is None:
        end_idx = len(PFs)
    else:
        end_idx = np.where(np.asarray(PFs_t) < t_end)[0][-1] + 1
    alphas = np.linspace(0.3, 0.9, len(PFs[start_idx:end_idx]))

    place_PF_kwargs = {
        "color": color,
        "lw": lw,
        "marker": marker,
        "ms": ms,
        "ls": ls,
    }

    num_plotted = 0
    prev_PF = None
    for a, alpha in enumerate(alphas):
        plot_PF = PFs[a + start_idx]
        if prev_PF is not None and (prev_PF == plot_PF).all():
            continue
        prev_PF = plot_PF

        if plot_smoothed and k != 1:
            plot_PF = signal_util.smooth_circularly(plot_PF, k=k)

        sub_ax.plot(
            PF_positions,
            plot_PF,
            alpha=alpha,
            **place_PF_kwargs,
        )

        num_plotted += 1
        last_PF = plot_PF

    if num_plotted == 0:
        sub_ax.plot(list(), list())

    elif num_plotted == 1:  # only one
        sub_ax.plot(
            PF_positions,
            last_PF,
            alpha=0.9,
            **place_PF_kwargs,
        )

    if plot_last_width and num_plotted > 0:
        width, edges = signal_util.compute_signal_width(
            last_PF, PF_positions, k=k, return_edges=True
        )
        label = f"width={width:.2f} m"

        end_pts = [0, PF_positions.max()]
        if np.isclose(edges[0], edges[1]):
            edges = end_pts
        plot_util.plot_vspan_circular(
            sub_ax=sub_ax,
            edges=edges,
            end_pts=end_pts,
            label=label,
            lw=0,
            color=color,
            alpha=0.2,
            zorder=-3,
        )
        if not no_legend:
            sub_ax.legend(frameon=False, loc="upper right")

    sub_ax.set_xlabel(xlabel)
    sub_ax.set_ylabel(ylabel)
    sub_ax.set_ylim(0, sub_ax.get_ylim()[1] * 1.1)
    sub_ax.spines[["top", "right"]].set_visible(False)

    return sub_ax


def plot_1D_BTSP_stats(
    Pyrs,
    recorded_PFs,
    PF_positions=None,
    target_position=None,
    other_positions=list(),
    in_min=True,
    PF_type="weights",
    plot_last_width=False,
    color="None",
):
    """
    plot_1D_BTSP_stats(Pyrs, recorded_PFs)

    Plot the BTSP stats of a Pyramidal neuron layer.

    Args:
    - Pyrs (learning_neurons.BTSPLayer): Pyramidal neurons.
    - recorded_PFs (1D np.ndarray): Recorded place fields.
    - PF_positions (1D np.ndarray, optional): Positions for the place fields. If None,
        place cell centers are used. Default is None.
    - target_position (float, optional): Target position. Default is None.
    - other_positions (list, optional): Other positions to plot. Default is empty list.
    - in_min (bool, optional): Whether to plot time in minutes. Default is True.
    - PF_type (str, optional): Type of place field to plot. Default is "weights".
    - plot_last_width (bool, optional): Whether to plot the last width of the place cells.
        Default is False.
    - color (str, optional): Color for the place fields. Default is None.

    Returns:
    - BTSP_ramp_ax1D (np.ndarray): Array of subplots with BTSP ramp stats plotted.
    - PCs_sub_ax (plt.Axes): Subplot with place fields plotted.
    """

    _, BTSP_ramp_ax1D = plt.subplots(3, 1, figsize=(7, 6))
    _, PCs_sub_ax = plt.subplots(figsize=(7, 2))

    if hasattr(Pyrs, "DendriteCompartment"):
        Pyrs.SomaCompartment.plot_BTSP_ramp(axes=BTSP_ramp_ax1D, in_min=in_min)
        BTSP_ramp_ax1D[1].get_lines()[-1].set_label("soma")
        for i, comp in enumerate([Pyrs.DendriteCompartment, Pyrs.DendriteInhibition]):
            t = np.asarray(comp.history["t"])
            if in_min:
                t = t / 60
            alpha = 0.7 if i == 0 else 0.5
            label = "dend" if i == 0 else "inhib."
            BTSP_ramp_ax1D[1].plot(
                t,
                comp.history["firingrate"],
                lw=1.2,
                alpha=alpha,
                color=comp.color,
                label=label,
            )
        BTSP_ramp_ax1D[1].legend(loc="upper right", ncols=2)
    else:
        Pyrs.plot_BTSP_ramp(axes=BTSP_ramp_ax1D, in_min=in_min)

    if PF_positions is None:
        _, _, PCs, _ = ext_util.extract_objects_from_Pyrs(Pyrs)
        PF_positions = PCs.place_cell_centers
        color = color or PCs.color
    elif color is None:
        color = "k"

    plot_recorded_1D_PFs(
        recorded_PFs,
        PF_positions,
        PF_type=PF_type,
        color=color,
        marker="none",
        plot_last_width=plot_last_width,
        sub_ax=PCs_sub_ax,
    )

    if target_position is not None:
        PCs_sub_ax.axvline(target_position, ls="dashed", color="k")
    for position in other_positions:
        PCs_sub_ax.axvline(position, ls="dashed", color="k", alpha=0.6)

    return BTSP_ramp_ax1D, PCs_sub_ax


def plot_pre_post_responses(Pyrs, Objs, ref_time=0, pre=60, post=None, axes=None):
    """
    plot_pre_post_responses(Pyrs, Objs)

    Plot the Pyramidal and Object neuron layer firing rates before and after a
    reference time.

    Args:
    - Pyrs (riab_neurons.Neurons): Pyramidal neurons.
    - Objs (riab_neurons.Neurons): Object neurons.
    - ref_time (float, optional): Reference time in seconds. Default is 0.
    - pre (float, optional): Time before the reference time in seconds. Default is 60.
    - post (float, optional): Time after the reference time in seconds. If None,
        pre is used. Default is None.
    - axes (2D np.ndarray, optional): Array of subplots to plot on. Default is None.

    Returns:
    - axes (2D np.ndarray): Array of subplots with Pyr. and Obj. rates plotted.
    """

    if axes is None:
        _, axes = plt.subplots(4, 2, sharex="col", sharey=True, figsize=(8, 4))
    elif axes.shape != (4, 2):
        raise ValueError("If provided, axes must have shape (4, 2).")

    post = post or pre
    col_times = [(ref_time - pre, ref_time), (ref_time, ref_time + post)]

    for c, (start, end) in enumerate(col_times):
        Objs.plot_rate_timeseries(t_start=start, t_end=end, sub_ax=axes[0, c])
        axes[0, c].set_title("Object input")
        mark_target_and_reset_points(Objs, sub_ax=axes[0, c])

        Pyrs.plot_rate_timeseries(
            separate_axes=True,
            t_start=start,
            t_end=end,
            ax=axes[1:, c],
            norm_by="shared_max",
        )

        plot_util.clear_bottom(axes[:, c][:-1])

    return axes


def plot_pre_post_dendrite_peaks(
    Pyrs,
    ref_time=0,
    pre=60,
    post=None,
    pts_btw=100,
    label=None,
    ylims=None,
    together=False,
    ax=None,
):
    """
    plot_pre_post_dendrite_peaks(Pyrs)

    Plot the Pyramidal dendrite response peaks before and after a reference time.

    Args:
    - Pyrs (riab_neurons.Neurons): Pyramidal neurons.
    - ref_time (float, optional): Reference time in seconds. Default is 0.
    - pre (float, optional): Time before the reference time in seconds. Default is 60.
    - post (float, optional): Time after the reference time in seconds. If None,
        pre is used. Default is None.
    - pts_btw (int, optional): Minimum number of points between peaks. Default is 100.
    - label (str, optional): Label for the plot. Default is None.
    - ylims (tuple, optional): y-axis limits to use. Default is None.
    - together (bool, optional): Whether to plot the before and after responses
        on the same plot. Default is False.
    - ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is created.
        Default is None.

    Returns:
    - ax (plt.Axes or 1D np.ndarray): Subplot or array of subplots with Pyr. dendrite
        response peaks plotted.
    """

    if ax is None:
        n_cols = 1 if together else 2
        fig, ax = plt.subplots(1, n_cols, figsize=(6, 4), sharex=True, sharey=True)
    else:
        num_subplots = np.asarray(ax).size
        if num_subplots != (2 - int(together)):
            raise ValueError(
                "If ax is provided, it must include 1 subplot if together is True, "
                "and 2 otherwise."
            )
        fig = np.asarray(ax).ravel()[0].figure

    dt = Pyrs.Agent.dt
    t = np.asarray(Pyrs.DendriteCompartment.history["t"])
    firingrates = np.asarray(Pyrs.DendriteCompartment.history["firingrate"])

    # pre
    pre_ids = plot_util.get_plotting_times(
        t,
        t_start=ref_time - pre,
        t_end=ref_time,
    )

    # post
    post = post or pre
    post_ids = plot_util.get_plotting_times(
        t,
        t_start=ref_time,
        t_end=ref_time + post,
    )

    pts_per_sec = int(0.5 / dt)

    labels = ["before", "after"]
    if label is not None:
        labels = [f"{label_start} {label}" for label_start in labels]

    colors = ["red", "grey"]
    for i, (start_id, end_id) in enumerate(zip(pre_ids, post_ids)):
        s = 0 if together else i
        sub_ax = np.asarray(ax).ravel()[s]
        rates = firingrates[start_id:end_id, 0]
        peak_pts = gen_util.get_minima_indices(
            -rates, minimum=-0.1, min_pts_btw=pts_btw // 6
        )
        # remove second part of double peaks
        rem_idxs = np.where(np.diff(peak_pts) < pts_btw)[0] + 1
        peak_pts = np.delete(peak_pts, rem_idxs)

        exp_num_pts = 6 * pts_per_sec
        responses = np.full((exp_num_pts, len(peak_pts)), np.nan)
        for p, peak_pt in enumerate(peak_pts):
            pre_peak_pt = max(0, peak_pt - pts_per_sec)
            post_peak_pt = min(peak_pt + 5 * pts_per_sec, len(rates))
            num_pts = post_peak_pt - pre_peak_pt
            time = np.linspace(-0.5, 2.5, exp_num_pts)
            response = rates[pre_peak_pt:post_peak_pt]
            if pre_peak_pt == 0 and num_pts < exp_num_pts:
                time = time[-num_pts:]

            sub_ax.plot(
                time[:num_pts],
                response,
                color=colors[i],
                alpha=0.4,
                lw=1,
            )

            start = pre_peak_pt - peak_pt + pts_per_sec
            responses[start : start + len(response), p] = response

        time = np.linspace(-0.5, 2.5, len(responses))
        response_mean = np.nanmean(responses, axis=1)

        sub_ax.plot(
            time,
            response_mean,
            color=colors[i],
            alpha=0.6,
            lw=3,
            ls="dashed",
            label=labels[i],
        )

        if i == 1 and not together:
            ax[0].fill_between(
                time,
                np.zeros_like(response_mean),
                response_mean,
                color=colors[i],
                alpha=0.4,
                lw=0,
            )

        sub_ax.axvline(0, color="k", zorder=-3, alpha=0.5, lw=1.5, ls="dashed")
        sub_ax.set_xlabel("Time (s)")
        sub_ax.spines[["top", "right"]].set_visible(False)
        if ylims is not None:
            sub_ax.set_ylim(*ylims)

        sub_ax.set_ylabel("Firing rates")

    sub_ax.legend()
    fig.suptitle(
        f"Comparing Pyr. dendrite response peaks before and after {label}", y=1.0
    )

    return ax


def plot_1D_initial_conditions(Pyrs, axes=None):
    """
    plot_1D_initial_conditions(Pyrs)

    Plot initial conditions for a 1D environment experiment.

    Args:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer
    - axes (list, optional): List of axes to plot on. If None, new axes are created.
        Default is None.

    Returns:
    - axes (list): List of axes with 1D environment information plotted.
    """

    Env, Ag, PCs, Objs = ext_util.extract_objects_from_Pyrs(Pyrs)

    if axes is None:
        height_ratios = [1, 1.2, 1.5]
        if Objs is not None:
            height_ratios.insert(1, 0.6)  # add height ratio for object rate map
        gridspec_kw = {"height_ratios": height_ratios}
        figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
        _, axes = plt.subplots(
            nrows=len(height_ratios),
            figsize=figsize,
            sharex=True,
            gridspec_kw=gridspec_kw,
            squeeze=False,
        )
    ax1D = np.asarray(axes).ravel()
    if len(ax1D) != 3 + int(Objs is not None):
        raise ValueError(f"Expected 3 + {int(Objs is not None)} axes, got {len(ax1D)}.")

    # Plot environment
    plot_1D_reset_environment(Ag, sub_ax=ax1D[0], autosave=False)

    # Plot object cell rate map, if applicable
    i = 1
    if Objs is not None:
        Objs.plot_rate_map(chosen_neurons="all", ax=ax1D[1], autosave=False)
        ax1D[1].set_title("Object cell rate map")
        i = 2

    # Plot place cell locations
    PCs.plot_place_cell_locations(sub_ax=ax1D[i], autosave=False, plot_objects=False)
    plot_overlayed_rate_maps(PCs, sub_ax=ax1D[i], method="max", autosave=False)
    ymin, ymax = ax1D[i].get_ylim()
    ymin = min(ymin, 0)
    ax1D[i].set_ylim((ymin - 0.05 * (ymax - ymin)), ymax)
    ax1D[i].set_title("Place cell locations")

    # Plot place cell rate map
    PCs.plot_rate_map(chosen_neurons="all", ax=ax1D[i + 1], autosave=False)
    ax1D[i + 1].set_title("Place cell rate map")

    ax1D[0].set_xticks([0, Env.scale])
    for a, sub_ax in enumerate(ax1D[:-1]):
        sub_ax.set_xlabel("")
        if a > 0 and a != 1 + int(Objs is not None):
            sub_ax.spines["bottom"].set_visible(False)
            sub_ax.xaxis.set_visible(False)

    return axes


def plot_1D_spatial_info(
    Pyrs: "learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer",
    Pyr_weights: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] | None = None,
    Pyrs_norm_by: str | float | None = None,
    autosave: bool | None = None,
) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
    """
    plot_1D_spatial_info(Pyrs)

    Plot spatial info for a 1D environment experiment:
        (1) Environment,
        (2, optional): Object cell rate maps (if applicable),
        (3) Place cell locations,
        (4) Pyr. overlayed rate map,
        (5, optional) Pyr. input weights (if provided),
        (6-8) Pyr. rate map across learning
        (9) Environment (again).

    Args:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer): Pyr. neurons.
    - Pyr_weights (list): List of Pyr. weights with shape (num_epochs, num_cells, num_PCs).
        Default is None.
    - Pyrs_norm_by (str, optional): Normalization method for rate maps. If None,
        default is used. Default is None.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - Axes (2D np.ndarray): Array of subplots with 1D environment experiment info
        plotted, with shape (7 or 8 or 9, 1). See description for details.
    """

    _, Ag, PCs, Objs = ext_util.extract_objects_from_Pyrs(Pyrs)

    # 7 or 8 plots
    height_ratios = [1, 1.2, 1.5, 1, 1, 1, 1]
    if Pyr_weights is not None:
        height_ratios.insert(3, 2)  # add height ratio for weights
    if Objs is not None:
        height_ratios.insert(1, 0.6)  # add height ratio for object rate map
    gridspec_kw = {"height_ratios": height_ratios}
    figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
    fig, axes = plt.subplots(
        nrows=len(height_ratios),
        figsize=figsize,
        sharex=True,
        gridspec_kw=gridspec_kw,
        squeeze=False,
    )
    ax1D = np.asarray(axes).ravel()

    i = 3 + int(Objs is not None)
    plot_1D_initial_conditions(Pyrs, axes=ax1D[:i])

    # Plot Pyr. weights
    if Pyr_weights is not None:
        plot_1D_PFs(
            np.asarray(Pyr_weights),
            PCs,
            sub_ax=ax1D[i],
            plot_last_width=(Pyrs.n == 1),
            autosave=False,
        )
        i += 1

    # Plot Pyr. rate maps across learning
    plot_1D_rate_map_across_learning(
        Pyrs, axes=ax1D[i : i + 3], norm_by=Pyrs_norm_by, autosave=False  # type: ignore[arg-type]
    )

    # Plot environment
    plot_1D_reset_environment(Ag, sub_ax=ax1D[i + 3], autosave=False)

    for a, sub_ax in enumerate(ax1D[:-1]):
        sub_ax.set_xlabel("")
        if a > 1:
            sub_ax.spines["bottom"].set_visible(False)
            sub_ax.xaxis.set_visible(False)

    plot_util.save_figure(fig, "1D_env_info", save=autosave)

    return axes


def plot_1D_time_info(
    Pyrs: "learning_neurons.BTSPLayer | two_comp_neurons.TwoCompLayer",
    Pyrs_spikes: bool = True,
    Pyr_kwargs: dict[str, Any] = dict(),
    height_ratios: list[float] | None = None,
    lw: float = 1,
    s: float = 0.02,
    base_s: float = 10,
    figsize: tuple[float, float] | None = None,
    autosave: bool | None = None,
    **gridspec_kw,
) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
    """
    plot_1D_time_info(Pyrs)

    Plot time info for a 1D experiment:
        (1) Trajectories,
        (2) Place cell rate timeseries,
        (3) Pyr. rate timeseries

    Args:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer): Pyr. neurons.
    - Pyrs_spikes (bool, optional): Whether to plot spikes in the Pyr. rate timeseries.
        Default is True.
    - Pyr_kwargs (dict, optional): Additional keyword arguments for the Pyr.
        rate timeseries. Default is an empty dict.
    - height_ratios (list[float], optional): Height ratios for the subplots. If None,
        default ratios are used. Default is None.
    - lw (float, optional): Line width for the plots. Default is 1.
    - s (float, optional): Size of agent trajectory scatterplot markers. If None,
        defaults are used. Default is None.
    - base_s (float, optional): Base size of scatterplot markers for objects in
        environment. If None, defaults are used. Default is None.
    - figsize (tuple[float, float], optional): Figure size. If None, default size is
        used. Default is None.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Keyword Args:
    - **gridspec_kw: Additional keyword arguments for the gridspec.

    Returns:
    - axes (2D np.ndarray): Array of subplots with 1D time info plotted,
        with shape (3, 1). See description for details.
    """

    _, Ag, PCs, Objs = ext_util.extract_objects_from_Pyrs(Pyrs)

    # 3 or 4 plots
    if height_ratios is None:
        height_ratios = [1.5, 1, 1.1**Pyrs.n]
        if Objs is not None:
            height_ratios.insert(1, 0.6)

    num_Pyr_ax = 1
    mark_all = True
    ax_key = "sub_ax"
    if hasattr(Pyrs, "DendriteCompartment"):
        ax_key = "ax"
        if "separate_axes" in Pyr_kwargs.keys() and Pyr_kwargs["separate_axes"]:
            plot_lateral = "plot_lateral" in Pyr_kwargs and Pyr_kwargs["plot_lateral"]
            mark_all = False
            num_Pyr_ax = len(Pyrs.get_compartments("all", incl_lateral=plot_lateral))

    num_plots = 2 + num_Pyr_ax + int(Objs is not None)
    if len(height_ratios) != num_plots:
        raise ValueError(
            f"Expected {num_plots} height_ratios, but got {len(height_ratios)}."
        )

    gridspec_kw["height_ratios"] = height_ratios
    if figsize is None:
        figsize = plot_util.get_figsize(sum(height_ratios), squat_height=True)
    fig, axes = plt.subplots(
        nrows=len(height_ratios),
        figsize=figsize,
        sharex=True,
        gridspec_kw=gridspec_kw,
        squeeze=False,
    )
    ax1D = np.asarray(axes).ravel()

    # Plot trajectories
    Ag.plot_trajectories_across_time(
        framerate=1 / Ag.dt,
        s=s,
        base_s=base_s,
        obj_lw=lw,
        sub_ax=ax1D[0],
        autosave=False,
    )
    ax1D[0].set_title("Trajectories")

    # Plot object cell rate timeseries
    i = 1
    if Objs is not None:
        Objs.plot_rate_timeseries(
            chosen_neurons="all",
            spikes=False,
            sub_ax=ax1D[1],
            lw=lw,
            norm_by="none",
            autosave=False,
        )
        plot_util.pad_axis(ax1D[1], axis="y", pad_prop=0.15, prop_high=1.0)
        ax1D[1].set_title("Object cell rate timeseries")
        i = 2

    # Plot place cell rate timeseries
    PCs.plot_rate_timeseries(
        chosen_neurons="all",
        spikes=False,
        sub_ax=ax1D[i],
        lw=lw,
        autosave=False,
        norm_by=PCs.n / 20,
        overlap=1,
        global_shift=-1,
        shade_kwargs={"rasterized": True},  # svg too big, othersize
    )
    plot_util.pad_axis(ax1D[i], pad_prop=0.1, axis="y")
    ax1D[i].set_title("Place cell rate timeseries")

    # Plot Pyr. rate timeseries
    Pyr_kwargs[ax_key] = ax1D[i + 1 :]

    Pyrs.plot_rate_timeseries(
        chosen_neurons="all",
        spikes=Pyrs_spikes,
        shift=-10,
        overlap=1,
        lw=lw,
        norm_by="none",
        autosave=False,
        **Pyr_kwargs,
    )
    ax1D[i + 1].set_title("Pyr. rate timeseries")

    # set y axes so that y axis is comparable across subplots
    ymin = min([sub_ax.get_ylim()[0] for sub_ax in ax1D[i + 1 :]])
    for sub_ax in ax1D[i + 1 :]:
        sub_ax.set_ylim(ymin, None)

    idxs = np.arange(i + 1, len(ax1D))
    if hasattr(Pyrs, "DendriteCompartment"):
        idxs = np.insert(idxs, 0, [1])
    axlist = [ax1D[i] for i in idxs]
    plot_util.match_y_axis_scales(axlist, [height_ratios[i] for i in idxs])

    for i, sub_ax in enumerate(ax1D):
        if mark_all or i < len(ax1D) - num_Pyr_ax:
            mark_target_and_reset_points(Pyrs, sub_ax=sub_ax, lw=lw)
        if i != len(ax1D) - 1:
            sub_ax.set_xlabel("")
            sub_ax.spines["bottom"].set_visible(False)
            sub_ax.tick_params(axis="x", bottom=False)

    plot_util.pad_axis(ax1D[0], "x", pad_prop=0.03)

    plot_util.save_figure(fig, "time_info", save=autosave)

    return axes


def plot_2D_PFs(
    target_neurons: "riab_neurons.Neurons",
    PCs_input_name: str = "PCs",
    PFs: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    PF_positions: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    chosen_neurons: (
        str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
    ) = "all",
    PF_type: str = "weights",
    PF_t_start: float | None = None,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    round_dec: int = 2,
    marker: str = "o",
    alpha: float = 0.7,
    lw: float = 0,
    s: float = 75,
    obj_s: float = 20,
    BTSP_s: float = 10,
    BTSP_marker: str = "x",
    zorder: int = 1,
    title: str | None = None,
    y: float = 1,
    single_colorbar: bool = False,
    cbar_side: str = "right",
    plot_BTSP_events: bool = False,
    t_start: float | None = None,
    t_end: float | None = None,
    color: str | None = None,
    ax: plt.Axes | np.ndarray | None = None,
    autosave: bool | None = None,
    **kwargs,
) -> plt.Axes:
    """
    plot_2D_PFs(target_neurons)

    Plot the place fields of a layer.

    Args:
    - target_neurons (riab_neurons.Neurons): Target neurons.
    - PCs_input_name (str, optional): Name of the input place cell layer.
    - PFs (2D np.ndarray, optional): Place fields to plot with shape
        (number of neurons, number of place field positions). If None, extracts the
        values from target_neurons. Default is None.
    - PF_positions (2D np.ndarray, optional): Positions for the place fields with
        shape (number of place field positions, 2). If None, the place cell centers
        from the PCs_input_name input layer are used. Exact PF positions may differ if
        PF_type is "history" and PFs are not provided. Default is None.
    - PF_type (str, optional): Type of place field to plot. Either "weights" or
        "history". If "weights", the weights from the PCs_input_name input layer are
        used. If "history", the firing rate history of the target_neurons is used.
        Default is "weights".
    - PF_t_start (float, optional): Start time for the PFs if PF_type is "history".
        Default is None.
    - chosen_neurons (str, int, list, or np.ndarray, optional): Neurons to plot.
        Default is "all". If "all", all neurons are plotted. If int, a single neuron
        is plotted. If list or np.ndarray, a list of neurons is plotted.
        Default is "all".
    - cmap (str, optional): Colormap to use. Default is "inferno".
    - vmin (float, optional): Minimum value for the colorbar. Default is None.
    - vmax (float, optional): Maximum value for the colorbar. Default is None.
    - round_dec (int, optional): Number of decimal places to round vmin and vmax to.
        Default is 2.
    - marker (str, optional): Marker to use. Default is "o".
    - alpha (float, optional): Alpha of the marker. Default is 0.7.
    - lw (float, optional): Linewidth of the marker. Default is 0.
    - s (float, optional): Size of scatterplot markers. Default is 75.
    - obj_s (float, optional): Size of object markers. If None, defaults are used.
        Default is None.
    - BTSP_s (float, optional): Size of BTSP markers. Default is 10.
    - BTSP_marker (str, optional): Marker for BTSP events. Default is "x".
    - zorder (int, optional): Zorder of the marker. Default is 1.
    - title (str, optional). Figure title. Default is None.
    - y (float, optional): Y position of the title. Default is 1.
    - single_colorbar (bool, optional): Whether to use a single colorbar. Default is
        False.
    - cbar_side (str, optional): Side of the colorbar. Default is "right".
    - plot_BTSP_events (bool, optional): Whether to plot the BTSP events.
        Default is False.
    - t_start (float, optional): Start time of the plot. Default is None.
    - t_end (float, optional): End time of the plot. Default is None.
    - color (str, optional): Color of the BTSP markers. Default is None.
    - ax (plt.Axes or np.ndarray, optional): Subplot or array of subplots to plot on
       (one per ROI). Default is None.
    - autosave (bool, optional): Whether to save the figure. Default is None.

    Raises:
    - ValueError: If PCs_input_name is not found among the target_neurons inputs.

    Keyword args:
    - **kwargs: Keyword arguments passed to riab_neurons.Neurons.plot_environment().

    Returns:
    - axes (np.ndarray or plt.Axes): Subplot or array of subplots
       (one per plotted ROI, if environment is 2D).

    """

    if PFs is None or PF_positions is None:
        if PCs_input_name not in target_neurons.inputs.keys():
            raise ValueError(
                f"Input layer '{PCs_input_name}' not found among target_neurons inputs."
            )

    if PF_positions is None:
        PF_positions = target_neurons.inputs[PCs_input_name]["layer"].place_cell_centers

    if len(PF_positions.shape) != 2 or PF_positions.shape[1] != 2:
        raise ValueError(
            "PF_positions must be a 2D array with shape (num_PF_positions, 2)."
        )

    chosen_neurons = np.asarray(
        target_neurons.return_list_of_neurons(chosen_neurons=chosen_neurons)
    )  # type: ignore[arg-type]

    clabel = get_PF_label(PF_type, title=False)

    BTSP_per = False
    if PFs is None:
        if PF_type == "weights":
            PFs = target_neurons.inputs[PCs_input_name]["w"][chosen_neurons]
        elif PF_type == "history":
            dist = np.inf
            for i in range(2):
                dist = min(
                    dist,
                    np.absolute(np.diff(np.sort(np.unique(PF_positions[:, i])))).min(),
                )
            PFs, PF_positions = target_neurons.get_history_ratemap(
                t_start=PF_t_start, bin_size=dist / 2
            )
        else:
            raise ValueError("PF_type must be either 'weights' or 'history'.")
        BTSP_per = True

    if PFs.shape[1] != PF_positions.shape[0]:
        raise ValueError(
            f"Number of PF positions ({PF_positions.shape[0]}) must match the number "
            f"of columns in PFs ({PFs.shape[1]})."
        )

    orig_vmin = vmin
    fact = 10**round_dec
    if vmin is None:
        vmin = np.nanmin(PFs)
        vmin = np.floor(vmin * fact) / fact

    if vmax is None:
        vmax = max(np.nanmax(PFs), 0.02)
        vmax = np.ceil(vmax * fact) / fact

    if vmax == vmin and orig_vmin is None:
        vmax = np.ceil(vmin * 2 * fact) / fact
        vmin = 0

    if ax is None:
        ax = plot_util.init_rate_map_axes(
            num_plots=len(PFs),
            num_cols=10,
            **kwargs,
        )

    ax1D = np.asarray(ax).ravel()
    if len(ax1D) < len(PFs):
        raise ValueError(
            f"Number of axes ({len(ax1D)}) must be at least as high as the number of "
            f"place fields ({len(PFs)})."
        )

    is_tmaze = hasattr(target_neurons.Agent.Environment, "T_ends")
    obj_s_kwarg = dict()
    if obj_s is not None:
        key = "base_s" if is_tmaze else "s"
        obj_s_kwarg[key] = obj_s

    for i in range(len(PFs)):
        target_neurons.Agent.Environment.plot_environment(
            sub_ax=ax1D[i], alpha=0.8, autosave=False, **kwargs, **obj_s_kwarg
        )

        ax1D[i].scatter(
            *PF_positions.T,
            c=PFs[i],
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            s=s,
            alpha=alpha,
            lw=0,
            zorder=zorder,
            cmap=cmap,
        )

    if plot_BTSP_events:
        _, startid, endid = target_neurons.get_plotting_times(
            t_start=t_start, t_end=t_end, raise_error=False
        )

        if endid > startid:
            BTSP_kwargs = {
                "ax": ax1D,
                "t_start": t_start,
                "t_end": t_end,
                "color": color,
                "s": BTSP_s,
                "lw": lw,
                "marker": BTSP_marker,
            }
            if BTSP_per:
                target_neurons.add_BTSP_markers_to_plots(
                    chosen_neurons=chosen_neurons, **BTSP_kwargs
                )
            else:
                for i in chosen_neurons:
                    target_neurons.add_BTSP_markers_to_plots(
                        chosen_neurons=[i], **BTSP_kwargs
                    )

    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    im = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    cbars = plot_util.add_colorbars(
        ax,
        im,
        vmin=vmin,
        vmax=vmax,
        label=clabel,
        end_only=single_colorbar,
        side=cbar_side,
    )

    v_ticks = [np.around(vmin, 2), np.around(vmax, 2)]
    for cbar in cbars:
        if np.diff(v_ticks) < 0.1:
            tick_labels = [f"{tick:.2f}" for tick in v_ticks]
        else:
            tick_labels = [f"{tick:.1f}" for tick in v_ticks]
        cbar.set_ticklabels(tick_labels)

    fig = ax1D[-1].figure

    if title is not None:
        fig.suptitle(title, y=y)

    plot_util.save_figure(fig, f"2D_{PF_type}_PFs", save=autosave)  # type: ignore[attr-defined]

    return ax


def plot_series_of_2D_PFs(
    target_neurons: "riab_neurons.Neurons",
    PF_series: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    steps: np.ndarray | None = None,
    ratio: float = 8,
    title: str | None = None,
    y: float = 1,
    split: bool = True,
    **kwargs,
):
    """
    Plot a series of 2D place field weights.

    Args:
    - target_neurons (riab_neurons.Neurons): Target neurons.
    - PF_series (list): Series of place fields to plot.
    - steps (np.ndarray, optional): Steps numbers for each set of weights. Step
        numbers are included in the title of each figure. Default is None.
    - ratio (float, optional): Ratio of the figure width to height. Default is 8.
    - title (str or list, optional): Title or titles to use for the figures.
        Default is None.
    - y (float, optional): Y position of the title. Default is 1.
    - split (bool, optional): Whether to split the figure into subplots. Default is True.

    Keyword args:
    - **kwargs: Keyword arguments passed to plot_2D_PFs().

    Returns:
    if split:
    - all_axes (list): List of subplots with the 2D input place cell weights plotted
        for each series.
    else:
    - axes (np.ndarray): Subplot with the 2D input place cell weights plotted
        for all series.
    """

    if steps is not None:
        if len(PF_series) != len(steps):
            raise ValueError(
                "Number of steps must match the number of place fields in the series "
                "provided."
            )

    if title is None or isinstance(title, str):
        titles = [title] * len(PF_series)
    else:
        titles = title

    if len(titles) != len(PF_series):
        raise ValueError(
            "Number of titles must match the number of place field series provided."
        )

    if not split:
        num_plots = len(PF_series)
        num_cols = min(5, num_plots)
        num_rows = int(np.ceil(num_plots / num_cols))
        axes = plot_util.init_rate_map_axes(
            num_plots=num_plots,
            num_cols=num_cols,
            size_per=2,
            **kwargs,
        )

    all_axes = list()
    for i, PFs in enumerate(PF_series):
        if steps is None or steps[i] is None:
            title = titles[i]
            t_end = None
        else:
            step = steps[i]
            t_end = step * target_neurons.Agent.dt
            if titles[i] is None:
                title = f"Step {step} at {t_end:.2f} s"
            else:
                title = f"{titles[i]} (step {step} at {t_end:.2f} s)"

        if split:
            num_plots = len(PFs)
            num_rows = int(np.max([1, np.floor(np.sqrt(num_plots / ratio))]))
            num_cols = int(np.ceil(num_plots / num_rows))
            use_ax = plot_util.init_rate_map_axes(
                num_plots=num_plots,
                num_cols=num_cols,
                **kwargs,
            )
        else:
            PFs = np.asarray(PFs).max(axis=0)[np.newaxis]
            use_ax = axes.ravel()[i]

        plot_2D_PFs(
            target_neurons,
            PFs=PFs,
            t_end=t_end,
            ax=use_ax,
            **kwargs,
        )

        if split:
            for sub_ax in use_ax.ravel()[num_plots:]:
                sub_ax.axis("off")

            if title is not None:
                use_ax.ravel()[-1].figure.suptitle(title, y=y, fontsize=20)

            all_axes.append(use_ax)
        else:
            if title is not None:
                use_ax.set_title(title)

            all_axes.append(use_ax)

    if split:
        return all_axes
    else:
        return axes


def plot_place_cell_inputs_over_time(
    target_neurons: "riab_neurons.Neurons",
    PCs_input_name: str = "PCs",
    filter_key: str | None = None,
    plot_position: bool = True,
    downsample_pos: int = 1,
    target_num_samples: int = 100,
    max_pos_sec: float = 20,
    num_cols: int = 10,
    obj_s: float | None = None,
    **kwargs,
):
    """
    plot_place_cell_inputs()

    Plot the place cell inputs of a layer in a 2D environment.

    Args:
    - target_neurons (riab_neurons.Neurons): Target neurons.
    - PCs_input_name (str, optional): Name of the input place cell layer. Default is
        "PCs".
    - filter_key (str, optional): Filter key to use. Default is None.
    - plot_position (bool, optional): Whether to plot the agent's position.
        Default is True.
    - downsample_pos (int, optional): Downsample the position plot. Default is 1.
    - target_num_samples (int, optional): Number of samples to plot. Default is 100.
    - max_pos_sec (float, optional): Maximum number of positions to plot, in seconds.
        Default is 2.
    - num_cols (int, optional): Number of columns in the plot. Default is 10.
    - obj_s (float, optional): Size of the object markers. Default is None.

    Keyword args:
    - **kwargs: Keyword arguments passed to plot_util.init_rate_map_axes() and
        plot_2D_PFs().

    Returns:
    - axes (np.ndarray): Subplot with 1D place cell inputs plotted.
    """

    if PCs_input_name not in target_neurons.inputs.keys():
        raise ValueError(
            f"Input layer '{PCs_input_name}' not found among target_neurons inputs."
        )

    if filter_key is None:
        data = np.asarray(
            target_neurons.inputs[PCs_input_name]["layer"].history["firingrate"]
        )
    else:
        if filter_key not in target_neurons.history.keys():
            raise ValueError(
                f"Filter key '{filter_key}' not found in {target_neurons}'s history."
            )
        data = np.asarray(target_neurons.history[filter_key][PCs_input_name])

    steps_dist = int(np.around(len(data) / target_num_samples))
    steps = np.arange(0, len(data), steps_dist)
    data = data[steps]
    num_steps = len(steps)

    num_cols = np.min([num_cols, num_steps])

    axes = plot_util.init_rate_map_axes(
        num_plots=num_steps,
        num_cols=num_cols,
        **kwargs,
    )

    plot_2D_PFs(
        target_neurons,
        PFs=data,
        ax=axes,
        obj_s=obj_s,
        **kwargs,
    )

    for i, step in enumerate(steps):
        t_end = step * target_neurons.Agent.dt
        title = f"Step {step} at {t_end:.2f} s"
        axes.ravel()[i].set_title(title)

    pos_kwargs = {
        "color": "white",
        "zorder": 5,
    }

    if plot_position:
        positions = np.asarray(target_neurons.Agent.history["pos"])
        max_num_pos = int(max_pos_sec / target_neurons.Agent.dt)
        s_values = np.linspace(1, 10, min([max_num_pos, len(positions)]))
        for i, step in enumerate(steps):
            if step == 0:
                continue
            axes.ravel()[i].scatter(
                *positions[:step][-max_num_pos::downsample_pos].T,
                marker=".",
                s=s_values[-step:][-max_num_pos::downsample_pos],
                alpha=0.6,
                **pos_kwargs,
            )

            axes.ravel()[i].scatter(
                *positions[step - 1], marker="x", s=20, alpha=0.8, **pos_kwargs
            )

    return axes


def plot_overlayed_rate_maps(
    NeuronLayer: "riab_neurons.Neurons",
    chosen_neurons: (
        str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
    ) = "all",
    sub_ax: plt.Axes | None = None,
    method: str = "max",
    colorbar: bool = True,
    plot_env: bool = False,
    autosave: bool | None = None,
    **kwargs,
) -> plt.Axes:
    """
    plot_overlayed_rate_maps(NeuronLayer)

    Plot the rate maps of the neurons in a layer.

    Args:
    - NeuronLayer (riab_neurons.Neurons): NeuronLayer object
    - chosen_neurons (str, int, list or 1D np.ndarray, optional):
        - If "all" or None plots all neurons in the layer.
        - If 15 or "15", selects 15 neurons evenly spread from index 0 to n
        - If "15rand": randomly selects 15 neurons
        Default is "all".
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.
    - method (str, optional): Method for plotting the rate maps. Default is "max".
    - colorbar (bool, optional): Whether to plot the colorbar. Default is True.
    - plot_env (bool, optional): Whether to plot the environment if axis provided.
        Default is False.
    - autosave (bool, optional): Whether to save the figure. Default is None.

    Keyword args:
    - **kwargs: Keyword arguments passed to NeuronLayer.get_state().

    Returns:
    - sub_ax (plt.Axes): Subplot with overlayed rate maps for all neurons overlayed.
    """
    rate_maps = NeuronLayer.get_state(evaluate_at="all", **kwargs)  # type: ignore[has-method]

    color = NeuronLayer.color  # type: ignore[attr-defined]
    if color is None:
        coloralpha = None
    else:
        coloralpha = list(mpl_colors.to_rgba(color))
        coloralpha[-1] = 0.5

    chosen_neurons = NeuronLayer.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type]
    chosen_neurons = np.asarray(chosen_neurons)
    N_neurons = len(chosen_neurons)

    if sub_ax is None:
        if NeuronLayer.Agent.Environment.dimensionality == "1D":
            _, sub_ax = plt.subplots(
                figsize=(
                    MOUNTAIN_PLOT_WIDTH_MM / 25,
                    N_neurons * MOUNTAIN_PLOT_SHIFT_MM / 25,
                )
            )
        sub_ax = NeuronLayer.Agent.Environment.plot_environment(
            alpha=0.6, autosave=False
        )
    elif plot_env:
        sub_ax = NeuronLayer.Agent.Environment.plot_environment(
            sub_ax=sub_ax, alpha=0.6, autosave=False
        )

    if sub_ax is None:
        raise RuntimeError("sub_ax is None.")

    if NeuronLayer.Agent.Environment.dimensionality == "2D":
        reshape = NeuronLayer.Agent.Environment.discrete_coords.shape[:2]
    else:
        reshape = [-1]

    rate_maps = np.asarray(rate_maps)[chosen_neurons].reshape(N_neurons, *reshape)

    if method == "mean":
        rate_map = np.nanmean(rate_maps, axis=0)
    elif method == "median":
        rate_map = np.nanmedian(rate_maps, axis=0)
    elif method == "max":
        rate_map = np.nanmax(rate_maps, axis=0)
    elif method == "min":
        rate_map = np.nanmin(rate_maps, axis=0)
    elif method == "sum":
        rate_map = np.nansum(rate_maps, axis=0)
    else:
        raise ValueError(f"method {method} not recognized.")

    # PLOT 2D
    if NeuronLayer.Agent.Environment.dimensionality == "2D":
        ex = NeuronLayer.Agent.Environment.extent
        im = sub_ax.imshow(rate_map, extent=ex, zorder=0, cmap="inferno")

        if colorbar == True:
            plot_util.add_colorbars(axes=sub_ax, im=im, label="")

    # PLOT 1D
    elif NeuronLayer.Agent.Environment.dimensionality == "1D":
        x = NeuronLayer.Agent.Environment.flattened_discrete_coords[:, 0]
        _, sub_ax = rutils.mountain_plot(
            X=x,
            NbyX=rate_map.reshape(1, -1),
            color=color,
            fig=sub_ax.figure,
            ax=sub_ax,
            shift=0,
            **kwargs,
        )

        if sub_ax is None:
            raise RuntimeError("sub_ax is None.")

        sub_ax.set_xlabel("Position (m)")
        sub_ax.set_ylabel("Neurons")

    fig = sub_ax.figure
    plot_util.save_figure(fig, f"{NeuronLayer.name}_overlayed_rate_maps", save=autosave)  # type: ignore[attr-defined]

    return sub_ax


def plot_2D_initial_conditions(Pyrs, num_samples=10, autosave: bool | None = None):
    """
    plot_2D_initial_conditions(Pyrs)

    Plot initial conditions for a 2D environment experiment.

    Args:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer
    - num_samples (int, optional): Number of samples to plot. Default is 10.
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - fields_axes (2D np.ndarray): Array of subplots with place fields plotted, with
        shape (num_layers, num_samples).
    - aggreg_ax1D (1D np.ndarray): Array of subplots with environment and aggregated
        fields plotted, with shape (3,).
    """

    Env, _, PCs, Objs = ext_util.extract_objects_from_Pyrs(Pyrs)

    # Plot fields
    if Objs is None:
        num_cols = min(PCs.n, num_samples)
        neurons = [PCs]
    else:
        num_cols = min(min(PCs.n, Objs.n), num_samples)
        neurons = [Objs, PCs]

    fields_fig, fields_axes = plt.subplots(
        len(neurons), num_cols, figsize=(num_cols * 2, len(neurons) * 2), squeeze=False
    )
    title_i = max(0, num_cols // 2 - 1)

    for i, NeuronLayer in enumerate(neurons):
        if num_cols >= NeuronLayer.n:
            chosen_neurons = np.arange(NeuronLayer.n)
        else:
            chosen_neurons = np.sort(np.random.choice(NeuronLayer.n, num_cols))
        ax1D = fields_axes[i, : len(chosen_neurons)]
        NeuronLayer.plot_rate_map(
            chosen_neurons=chosen_neurons, ax=ax1D, no_legend=True, autosave=False
        )
        name = NeuronLayer.name.replace("_", " ")
        ax1D[title_i].set_title(f"{name} rate maps", fontsize="x-large")  # type: ignore[attr-defined]

    # Plot aggregated fields
    aggreg_fields, aggreg_ax1D = plt.subplots(1, 3, figsize=(9, 3))

    for sub_ax in aggreg_ax1D[:2]:
        Env.plot_environment(sub_ax=sub_ax, alpha=0.6, no_legend=True, autosave=False)
    aggreg_ax1D[0].set_title("Environment")

    plot_overlayed_rate_maps(
        PCs, method="max", colorbar=False, sub_ax=aggreg_ax1D[1], autosave=False
    )
    aggreg_ax1D[1].set_title("Overlayed place fields")

    PCs.plot_place_cell_locations(sub_ax=aggreg_ax1D[2], autosave=False)
    aggreg_ax1D[2].set_title("Place cell centers")

    if autosave:
        plot_util.save_figure(fields_fig, "initial_fields", save=autosave)
        plot_util.save_figure(aggreg_fields, "initial_aggreg", save=autosave)

    return fields_axes, aggreg_ax1D


def plot_interleaved_openfield_rate_maps(
    Pyrs, Objs, num_cols=10, size_per=1.4, obj_s=None, **kwargs
):
    """
    plot_interleaved_openfield_rate_maps(Pyrs, Objs)

    Plot interleaved open field rate maps for Pyr. neuron somata and the object neurons
    thattarget their dendrites.

    Rate maps are computed theoretically based on place cell inputs.

    Args:
    - Pyrs (two_comp_neurons.TwoComp): Pyr. neurons.
    - Objs (object_neurons.ObjectCells): Object neurons.
    - num_cols (int, optional): Number of columns in the plot. Default is 10.
    - size_per (float, optional): Size per subplot. Default is 1.4.

    Keyword args:
    - **kwargs: Keyword arguments passed to
        plot_2D_PFs().

    Returns:
    - axes (np.ndarray): Array of subplots with interleaved rate maps plotted.
    """

    if Pyrs.n != Objs.n:
        raise ValueError("Pyrs and Objs should have the same number of neurons.")

    num_cols = min(10, Objs.n)
    num_rows = int(np.ceil(Objs.n / num_cols)) * 2

    _, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(size_per * num_cols, size_per * num_rows),
        squeeze=False,
    )

    is_tmaze = hasattr(Pyrs.Agent.Environment, "T_ends")
    obj_s_kwarg = dict()
    no_legend = False if is_tmaze else True
    if obj_s is not None:
        key = "base_s" if is_tmaze else "s"
        obj_s_kwarg[key] = obj_s

    Objs.plot_rate_map(
        ax=axes[::2].ravel()[: Objs.n], no_legend=no_legend, **obj_s_kwarg
    )
    plot_2D_PFs(
        Pyrs.SomaCompartment,
        ax=axes[1::2].ravel()[: Objs.n],
        alpha=0.5,
        plot_BTSP_events=True,
        no_legend=no_legend,
        obj_s=obj_s,
        **kwargs,
    )

    # turn off extra subplots
    for sub_ax in axes[::2].ravel()[Objs.n :]:
        sub_ax.axis("off")
    for sub_ax in axes[1::2].ravel()[Objs.n :]:
        sub_ax.axis("off")

    return axes


def compare_theoretical_and_true_weights(
    Pyrs, PCs_name="PCs", target_position=None, output_fr=4, chosen_neuron=0, **kwargs
):
    """
    compare_theoretical_and_true_weights(Pyrs)

    Compare the theoretical and true weights of a layer.

    Args:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer for which to compare theoretical and true weights.
    - PCs_name (str, optional): Name of the input place cell layer. Default is "PCs".
    - target_position (float, optional): Target position to use to roll the theoretical
        weights. Inferred from Agent if not provided. If Agent has not target position,
        the weights are rolled approximately, based on peak difference. Default is None.
    - output_fr (int, optional): Output framerate to use to compute theoretical weights.
        Default is 4.
    - chosen_neuron (int, optional): Neuron to plot. Default is 0.

    Keyword args:
    - **kwargs: Keyword arguments passed to
        learn_util.assess_Pyrs_learning_rates_spatially().

    Returns:
    - norm_theor_axes (np.ndarray): Array of subplots with normalized theoretical
        weights plotted.
    - no_norm_theor_axes (np.ndarray): Array of subplots with non-normalized
        theoretical weights plotted.
    - act_sub_ax (plt.Axes): Subplot with actual weights plotted. If the environment is
        1D, the theoretical weights are also plotted.
    """

    if PCs_name not in Pyrs.inputs.keys():
        raise RuntimeError(f"{PCs_name} not found in inputs to Pyrs.")
    PCs = Pyrs.inputs[PCs_name]["layer"]

    y = 1.7 if Pyrs.Agent.Environment.dimensionality == "1D" else 1.3

    theor_axes = list()
    for normalize_weights_divisively in [True, False]:
        assessment_dict = learn_util.assess_Pyrs_learning_rates_spatially(
            Pyrs,
            normalize_weights_divisively=normalize_weights_divisively,
            output_fr=output_fr,
            log_reg=True,
            **kwargs,
        )
        axes = plot_util.plot_learning_rate_assessment(assessment_dict)
        title_str = "With" if normalize_weights_divisively else "No"
        axes.ravel()[0].figure.suptitle(f"{title_str} normalization", y=y)
        theor_axes.append(axes)

    norm_theor_axes, no_norm_theor_axes = theor_axes

    if Pyrs.Agent.Environment.dimensionality == "1D":
        true_ws = Pyrs.inputs[PCs_name]["w"]
        act_sub_ax = plot_1D_PFs(true_ws, PCs)

        # roll theoretical weights
        theor_ws = assessment_dict["ws"][-1][:, 0]
        if target_position is None:
            target_position = Pyrs.Agent.target_position[0]
        if target_position is not None:
            rel_pos = target_position / Pyrs.Agent.Environment.scale
            peak_pt_diff = int(np.argmax(theor_ws) - len(theor_ws) * rel_pos)
        else:
            peak_pt_diff = np.argmax(theor_ws) - np.argmax(true_ws)
        rolled_theor_ws = np.roll(theor_ws, -peak_pt_diff)

        x = np.linspace(0, Pyrs.Agent.Environment.scale, len(theor_ws) + 1)[:-1]
        act_sub_ax.plot(x, rolled_theor_ws, color="k", alpha=0.7, label="theoret.")
        act_sub_ax.set_title("Actual weights vs theoretical weights")
        act_sub_ax.legend()
        act_sub_ax.autoscale(axis="y")
        plot_util.pad_axis(act_sub_ax, axis="y", prop_high=1.0)

    else:
        dim = Pyrs.Agent.Environment.scale / 4 * 3
        _, act_sub_ax = plt.subplots(figsize=(dim, dim))

        is_not_tmaze = not hasattr(Pyrs.Agent.Environment, "T_ends")
        plot_2D_PFs(
            Pyrs,
            ax=act_sub_ax,
            chosen_neurons=[chosen_neuron],
            obj_s=10,
            no_legend=is_not_tmaze,
        )

    return norm_theor_axes, no_norm_theor_axes, act_sub_ax
