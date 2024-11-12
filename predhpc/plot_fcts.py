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

from predhpc.util import gen_util, plot_util

if TYPE_CHECKING:
    from predhpc.agent import ResetableAgent
    from predhpc.neurons import riab_neurons, learning_neurons


def add_time_axis(
    sub_ax,
    dt: float = 0.03,
    in_minutes: bool = False,
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
    - in_minutes (bool, optional): Whether to plot in minutes. Default is False.
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

    if in_minutes:
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
    in_minutes: bool = False,
    max_num_minutes: float | int | None = None,
    trajectory_lengths: np.ndarray[tuple[int], np.dtype[np.int64]] | list | None = None,
    **kwargs,
) -> tuple[plt.Axes, np.ndarray]:
    """
    plot_trajectory_lengths()

    Plot trajectory lengths.

    Args:
    - dt (float, optional): Time step. If None, time axis is not added. Default is None.
    - in_minutes (bool, optional): Whether to plot time axis in minutes instead of
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
        add_time_axis(sub_ax, dt, in_minutes, max_num_minutes, trajectory_lengths)

    return sub_ax, trajectory_lengths


def mark_target_and_reset_points(
    Ag: "ResetableAgent",
    CA1s: "riab_neurons.Neurons",
    sub_ax: plt.Axes,
    restore_xlims: bool = True,
):
    """
    mark_target_and_reset_points(Ag, CA1s, sub_ax)

    Add target and reset points to a timeseries plot.

    Args:
    - Ag (Agent): Agent for which to add target and reset points.
    - CA1s (riab_neurons.Neurons): CA1 layer to plot.
    - sub_ax (plt.Axes): Subplot to add target and reset points to.
    - restore_xlims (bool, optional): Whether to restore x limits. Default is True.
    """

    if restore_xlims:
        xlims = sub_ax.get_xlim()

    for end_point, ls in [
        ("reset", "dashed"),
        ("target", "dotted"),
    ]:
        if end_point == "reset":
            positions = Ag.trajectory_df["stop_step"].to_numpy()
        elif end_point == "target":
            positions = Ag.target_df["reached_step"].to_numpy()
        else:
            raise ValueError(f"Unknown end point: {end_point}")
        positions = positions[np.isfinite(positions)].astype(int)

        for t in positions:
            sub_ax.axvline(
                CA1s.Agent.history["t"][t] / 60,
                alpha=0.7,
                zorder=-1,
                lw=1,
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
    sub_ax.set_xlabel("Time (s)")
    sub_ax.set_ylabel(loss_type)

    plot_util.pad_axis(sub_ax)

    return sub_ax


def plot_oscillation_events(
    oscillation_df,
    firingrates,
    order_by="neuron_num",
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
    - order_by (str, optional): Column to order by. Default is "neuron_num".
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
    num_neurons = len(oscillation_df.loc[indices[:num_osc], "neuron_num"].unique())
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

    prev_neuron_num = -1
    for i, row in enumerate(np.arange(num_osc)):
        sub_ax = axes.ravel()[i]
        df_row = oscillation_df.loc[indices[row]]
        neuron_num = df_row["neuron_num"]
        neuron_idx = df_row["neuron_idx"]
        start, stop = df_row["start_frame"], df_row["stop_frame"]

        sub_ax.plot(firingrates[start:stop, neuron_idx], color=color)

        if order_by[0] == "neuron_num" and neuron_num != prev_neuron_num:
            num = len(oscillation_df.loc[oscillation_df["neuron_num"] == neuron_num])
            sub_ax.plot([], [], label=f"#{neuron_num} ({num})")
            sub_ax.legend(**legend_kwargs)
            prev_neuron_num = neuron_num

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

    norm_firingrates = gen_util.get_norm_data(firingrates, axis=0) * norm_height
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

    norm_firingrates = gen_util.get_norm_data(firingrates, axis=0) * norm_height
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
            frames_to_plot = gen_util.pad_throughout(
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
            padded_idxs = gen_util.pad_throughout(
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
        factors = (all_steps - all_steps.min()) / (
            all_steps.max() - all_steps.min()
        ) + 0.5

    if x_data is None:
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

    if legend:
        sub_ax.legend(fontsize=5, loc="upper right")

    return sub_ax


def plot_time_series_with_BTSP_events(
    CA1s: "learning_neurons.BTSPLayer",
    sub_ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    plot_time_series_with_BTSP_events(CA1s)

    Plot the time series of the CA1s layer with BTSP events marked.

    Args:
    - CA1s (learning_neurons.BTSPLayer): CA1s layer.
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with time series and BTSP events plotted.
    """

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=(6, 1.2**CA1s.n))

    CA1s.plot_rate_timeseries(chosen_neurons="all", spikes=True, sub_ax=sub_ax)
    lo, hi = sub_ax.get_ylim()

    target_reached_step = CA1s.Agent.target_df["reached_step"].to_numpy()  # type: ignore[attr-defined]
    if np.isnan(target_reached_step[-1]):
        target_reached_step = target_reached_step[:-1]
    target_reached_step = target_reached_step.astype(int)

    for t in target_reached_step:
        y_hei = lo + (hi - lo) * 0.82
        sub_ax.scatter(
            CA1s.Agent.history["t"][t] / 60,
            y_hei,
            marker=mpl_markers.MarkerStyle("o"),
            s=6,
            color="k",
            alpha=0.7,
        )

    # add distance from target below
    all_positions = np.asarray(CA1s.Agent.history["pos"])
    time_in_min = np.asarray(CA1s.Agent.history["t"]) / 60
    distances = np.linalg.norm(
        CA1s.Agent.target_position - all_positions, ord=2, axis=1  # type: ignore[attr-defined]
    )
    norm_dist = distances / distances.max()
    sub_ax.plot(time_in_min, -norm_dist, color="black", alpha=0.6, lw=1)
    sub_ax.set_ylim(-norm_dist.max() * 1.2, sub_ax.get_ylim()[1])

    sub_ax.set_title(
        "CA1 time series with BTSP events (with proximity to target)", y=1.1
    )

    return sub_ax


def plot_1D_reset_environment(
    Ag: "ResetableAgent",
    title: str = "Environment",
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
        plotting_dict[label]["kwargs"] = plot_util.get_plot_marker_kwargs(label)

    sub_ax = Ag.Environment.plot_environment(
        sub_ax=sub_ax, plot_objects=False, autosave=False
    )

    if sub_ax is None:
        raise RuntimeError("sub_ax is None.")

    for label, sub_dict in plotting_dict.items():
        if sub_dict["data"] is None:
            continue
        sub_ax.scatter(
            sub_dict["data"],
            0,
            zorder=5,
            label=label,
            **sub_dict["kwargs"],
        )
    sub_ax.legend(ncol=len(plotting_dict), frameon=False)
    sub_ax.set_title(title)

    fig = sub_ax.figure
    plot_util.save_figure(fig, "1D_reset_environment", save=autosave)

    return sub_ax


def plot_1D_rate_map_across_learning(
    Ag: "ResetableAgent",
    CA1s: "riab_neurons.Neurons",
    axes: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | None = None,
    plot_proportion: float = 0.3,
    min_num_steps: int = 100,
    autosave: bool | None = None,
) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
    """
    plot_1D_rate_map_across_learning(Ag, CA1s)

    Plot the rate map of a layer across learning from stimulated activity in the
    first, mid and final steps.

    Args:
    - Ag (Agent): Agent for which to plot the rate map.
    - CA1s (riab_neurons.Neurons): CA1 layer to plot.
    - axes (1 or 2D np.ndarray, optional): Array of subplots to plot on (3 total).
        Default is None.
    - plot_proportion (float, optional): Proportion of the total steps to plot.
        Default is 0.3.
    - min_num_steps (int, optional): Minimum number of steps to plot. Default is 100.
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

    suptitle = f"CA1 rate maps across learning: "

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
        CA1s.plot_rate_map(
            chosen_neurons="all",
            ax=ax1D[s],
            t_start=t_start,  # type: ignore[type-arg]
            t_end=t_end,
            shift=-10,
            overlap=1,
            method="history",
            autosave=False,
            no_legend=no_legend,
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
    plot_util.save_figure(fig, f"{CA1s.name}_1D_rate_map_across_learning", save=autosave)  # type: ignore[attr-defined]

    return axes


def plot_1D_input_place_cell_weights(
    place_weights: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    PCs: "riab_neurons.PlaceCells",
    cmap: str = "crest",
    sub_ax: plt.Axes | None = None,
    autosave: bool | None = None,
) -> plt.Axes:
    """
    plot_1D_input_place_cell_weights(place_weights, PCs)

    Plot the input place cell weights of a layer.

    Args:
    - place_weights (2 or 3D np.ndarray): Incoming place weights across epochs to a
        layer of cells with shape ((num_epochs,) num_cells, num_PCs).
    - PCs (riab_neurons.PlaceCells): Place cells of the layer.
    - cmap (str, optional): Colormap to use. Default is "crest".
    - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
        created. Default is None.
    - autosave (bool, optional): Whether to save the figure. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with 1D input place cell weights plotted.
    """

    place_weights = np.asarray(place_weights)

    if len(place_weights.shape) == 2:
        single_sample = True
        place_weights = place_weights[np.newaxis]
    else:
        single_sample = False

    num_samples, num_cells, _ = place_weights.shape

    if sub_ax is None:
        figsize = (5, num_cells * 2)
        _, sub_ax = plt.subplots(figsize=figsize, sharex=True)

    if single_sample:
        colors = [PCs.color or "k"]
        alpha = 0.8
    else:
        cmap_vals = np.linspace(0, 1, num_samples)
        mpl_cmap = sns.color_palette(cmap, as_cmap=True)
        colors = mpl_cmap(cmap_vals)  # type: ignore[callable]
        alpha = 0.8 ** (len(colors) / 6)

    spacing = (place_weights.max() - place_weights.min()) * 1.1
    for n in range(num_cells):
        offset = None
        for i, color in enumerate(colors):
            offset = spacing * n
            sub_ax.plot(
                PCs.place_cell_centres[:, 0],
                place_weights[i, n] + offset,
                color=color,
                alpha=alpha,
                marker=mpl_markers.MarkerStyle("."),
                ms=4,
            )
        if offset is not None:
            lw = 2 * 0.8**num_cells
            sub_ax.axhline(offset, color="k", alpha=0.3, lw=lw, ls="dotted", zorder=-12)

    sub_ax.axvline(
        PCs.Agent.target_position,
        alpha=0.7,
        zorder=-1,
        lw=1,
        ls="dotted",
        color="k",
    )
    if single_sample:
        title = "Input weights"
    else:
        sub_ax.plot([], color=colors[0], label="first", alpha=0.8)
        sub_ax.plot([], color=colors[-1], label="last", alpha=0.8)
        sub_ax.legend(ncol=2, frameon=False)
        title = "Input weights across learning"

    sub_ax.set_xlabel("Input place cell center / m")
    sub_ax.set_ylabel("Weights")
    sub_ax.spines[["top", "right"]].set_visible(False)

    sub_ax.set_title(title)

    # pad the y-axis
    ylims = sub_ax.get_ylim()
    pad = (ylims[1] - ylims[0]) * 0.05
    sub_ax.set_ylim(ylims[0] - pad, ylims[1] + pad)

    fig = sub_ax.figure
    plot_util.save_figure(fig, f"{PCs.name}_1D_input_place_cell_weights", save=autosave)  # type: ignore[attr-defined]

    return sub_ax


def plot_previous_1D_input_place_cell_weights(
    weights,
    weights_t,
    input_centres,
    color="k",
    t_start=None,
    t_end=None,
    sub_ax=None,
):
    """
    plot_previous_1D_input_place_cell_weights(weights_t, weights, input_centres)

    Plots the input weights from CA3 place cells to a CA1 soma, across time, for a
    linear track environment.

    Args:
    - weights (2D np.ndarray): Input weights from place cells to target neuron, for
        each timepoint, with shape (timepoints, num_inputs).
    - weights_t (list): List of timepoints for the input weights. Must have the same
        length as weights.
    - input_centres (2D np.ndarray): Centres of the input place cell locations.
    - color (str, optional): Color. Default is "k".
    - t_start (float, optional): Start timepoint for which to plot weights.
        Default is None.
    - t_end (float, optional): End timepoint for which to plot weights. Default is None.
    - sub_ax (plt.Axes, optional): Subplot. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot on which weights are plotted.
    """

    if len(weights_t) != len(weights):
        raise ValueError(
            f"Length of 'weights' ({len(weights)}) and 'weights_t' ({len(weights_t)}) "
            "must be the same."
        )

    if sub_ax is None:
        _, sub_ax = plt.subplots(figsize=[6, 1])

    if t_start is None:
        start_idx = 0
    else:
        start_idx = np.where(np.asarray(weights_t) > t_start)[0][0]

    if t_end is None:
        end_idx = len(weights_t)
    else:
        end_idx = np.where(np.asarray(weights_t) < t_end)[0][-1] + 1
    alphas = np.linspace(0.3, 0.9, len(weights[start_idx:end_idx]))

    place_weight_kwargs = {
        "color": color,
        "lw": 1.2,
        "marker": "o",
        "ms": 2,
    }
    num_plotted = 0
    prev_weights = None
    for a, alpha in enumerate(alphas):
        plot_weights = weights[a + start_idx]
        if prev_weights is not None and (prev_weights == plot_weights).all():
            continue
        sub_ax.plot(
            input_centres[:, 0],
            plot_weights,
            alpha=alpha,
            **place_weight_kwargs,
        )
        prev_weights = plot_weights
        num_plotted += 1

    if num_plotted == 0:
        sub_ax.plot(list(), list())

    elif num_plotted == 1:  # only one
        sub_ax.plot(
            input_centres[:, 0],
            prev_weights,
            alpha=0.9,
            **place_weight_kwargs,
        )

    sub_ax.set_xlabel("Position on track")
    sub_ax.spines[["top", "right"]].set_visible(False)
    sub_ax.set_ylabel(f"Input weights")

    return sub_ax


def plot_2D_input_place_cell_weights(
    target_neurons: "riab_neurons.Neurons",
    PCs_input_name: str = "CA3_PCs",
    place_weights: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    chosen_neurons: (
        str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
    ) = "all",
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    marker: str = "o",
    alpha: float = 0.7,
    lw: float = 0,
    s: float = 75,
    zorder: int = 1,
    title: str | None = None,
    y: float = 1,
    single_colorbar: bool = False,
    plot_BTSP_events: bool = False,
    t_start: float | None = None,
    t_end: float | None = None,
    color: str | None = None,
    ax: plt.Axes | np.ndarray | None = None,
    autosave: bool | None = None,
    **kwargs,
) -> plt.Axes:
    """
    plot_2D_input_place_cell_weights(target_neurons)

    Plot the input place cell weights of a layer.

    Args:
    - target_neurons (riab_neurons.Neurons): Target neurons.
    - PCs_input_name (str, optional): Name of the input place cell layer.
    - place_weights (2D np.ndarray, optional): Place weights to plot with shape
        (number of output neurons, number of input neurons). If None, uses the
        input weights to the target_neurons from the layer specified by PCs_input_name.
        Default is None.
    - cmap (str, optional): Colormap to use. Default is "inferno".
    - vmin (float, optional): Minimum value for the colorbar. Default is None.
    - vmax (float, optional): Maximum value for the colorbar. Default is None.
    - marker (str, optional): Marker to use. Default is "o".
    - alpha (float, optional): Alpha of the marker. Default is 0.7.
    - lw (float, optional): Linewidth of the marker. Default is 0.
    - s (float, optional): Size of scatterplot markers. Default is 75.
    - zorder (int, optional): Zorder of the marker. Default is 1.
    - title (str, optional). Figure title. Default is None.
    - y (float, optional): Y position of the title. Default is 1.
    - single_colorbar (bool, optional): Whether to use a single colorbar. Default is
        False.
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

    if PCs_input_name not in target_neurons.inputs.keys():
        raise ValueError(
            f"Input layer '{PCs_input_name}' not found among target_neurons inputs."
        )

    place_cell_centres = target_neurons.inputs[PCs_input_name][
        "layer"
    ].place_cell_centres

    if place_weights is None:
        chosen_neurons = np.asarray(target_neurons.return_list_of_neurons(chosen_neurons=chosen_neurons))  # type: ignore[arg-type]
        place_weights = target_neurons.inputs[PCs_input_name]["w"][chosen_neurons]

    if vmin is None:
        vmin = place_weights.min()
        vmin = min(vmin, np.around(vmin, 2))

    if vmax is None:
        vmax = place_weights.max()
        if vmax < 0.02:
            vmax = 0.02
        vmax = max(vmax, np.around(vmax, 2))

    if ax is None:
        ax = plot_util.init_rate_map_axes(
            num_plots=len(place_weights),
            num_cols=10,
            **kwargs,
        )

    ax1D = np.asarray(ax).ravel()
    if len(ax1D) < len(place_weights):
        raise ValueError(
            f"Number of axes ({len(ax1D)}) must be at least as high as the number of "
            f"place weights ({len(place_weights)})."
        )

    for i in range(len(place_weights)):
        target_neurons.Agent.Environment.plot_environment(
            sub_ax=ax1D[i],
            autosave=False,
            **kwargs,
        )

        ax1D[i].scatter(
            *place_cell_centres.T,
            c=place_weights[i],
            vmin=vmin,
            vmax=vmax,
            marker=marker,
            s=s,
            alpha=alpha,
            lw=lw,
            zorder=zorder,
            cmap=cmap,
        )

    if plot_BTSP_events:
        target_neurons.add_BTSP_markers_to_plots(
            ax=ax1D, t_start=t_start, t_end=t_end, color=color
        )

    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
    im = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    cbars = plot_util.add_colorbars(
        ax, im, vmin=vmin, vmax=vmax, label="Weights", end_only=single_colorbar
    )

    vmin_tick = np.around(vmin, 2)
    vmax_tick = np.around(vmax, 2)
    for cbar in cbars:
        cbar.set_ticklabels([f"{vmin_tick:.1f}", f"{vmax_tick:.1f}"])

    fig = ax1D[-1].figure

    if title is not None:
        fig.suptitle(title, y=y)

    plot_util.save_figure(fig, f"{PCs_input_name}_2D_input_place_cell_weights", save=autosave)  # type: ignore[attr-defined]

    return ax


def plot_series_of_2D_input_place_cell_weights(
    target_neurons: "riab_neurons.Neurons",
    place_weight_series: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    steps: np.ndarray | None = None,
    ratio: float = 8,
    title: str | None = None,
    y: float = 1,
    **kwargs,
):
    """
    Plot a series of 2D input place cell weights.

    Args:
    - target_neurons (riab_neurons.Neurons): Target neurons.
    - place_weight_series (list): Series of place weights to plot.
    - steps (np.ndarray, optional): Steps at which to plot the series. Step numbers are
        included in the title of each figure. Default is None.
    - ratio (float, optional): Ratio of the figure width to height. Default is 8.
    - title (str, optional): Title to use for all figures. Default is None.
    - y (float, optional): Y position of the title. Default is 1.

    Keyword args:
    - **kwargs: Keyword arguments passed to plot_2D_input_place_cell_weights().

    Returns:
    - all_axes (list): List of subplots with the 2D input place cell weights plotted
        for each series.
    """

    if steps is not None:
        if len(place_weight_series) != len(steps):
            raise ValueError(
                "Number of steps must match the number of place weight in the series "
                "provided."
            )

    all_axes = list()
    for i, weights in enumerate(place_weight_series):
        if steps is None:
            use_title = title
        else:
            step = steps[i]
            t_end = step * target_neurons.Agent.dt
            if title is None:
                use_title = f"Step {step} at {t_end:.2f} s"
            else:
                use_title = f"{title} (step {step} at {t_end:.2f} s)"

        num_plots = len(weights)
        num_rows = int(np.max([1, np.floor(np.sqrt(num_plots / ratio))]))
        num_cols = int(np.ceil(num_plots / num_rows))
        axes = plot_util.init_rate_map_axes(
            num_plots=len(weights),
            num_cols=num_cols,
            **kwargs,
        )

        plot_2D_input_place_cell_weights(
            target_neurons,
            place_weights=weights,
            t_end=t_end,
            ax=axes,
            **kwargs,
        )

        for sub_ax in axes.ravel()[num_plots:]:
            sub_ax.axis("off")

        if use_title is not None:
            axes.ravel()[-1].figure.suptitle(use_title, y=y, fontsize=20)

        all_axes.append(axes)

    return all_axes


def plot_place_cell_inputs_over_time(
    target_neurons: "riab_neurons.Neurons",
    PCs_input_name: str = "CA3_PCs",
    filter_key: str | None = None,
    plot_position: bool = True,
    downsample_pos: int = 1,
    target_num_samples: int = 100,
    max_pos_sec: float = 20,
    num_cols: int = 10,
    **kwargs,
):
    """
    plot_place_cell_inputs()

    Plot the input place cell weights of a layer.

    Args:
    - target_neurons (riab_neurons.Neurons): Target neurons.
    - PCs_input_name (str, optional): Name of the input place cell layer. Default is
        "CA3_PCs".
    - filter_key (str, optional): Filter key to use. Default is None.
    - plot_position (bool, optional): Whether to plot the agent's position.
        Default is True.
    - downsample_pos (int, optional): Downsample the position plot. Default is 1.
    - target_num_samples (int, optional): Number of samples to plot. Default is 100.
    - max_pos_sec (float, optional): Maximum number of positions to plot, in seconds.
        Default is 2.
    - num_cols (int, optional): Number of columns in the plot. Default is 10.

    Keyword args:
    - **kwargs: Keyword arguments passed to plot_2D_input_place_cell_weights().

    Returns:
    - axes (np.ndarray): Subplot with 1D input place cell weights plotted.
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

    plot_2D_input_place_cell_weights(
        target_neurons,
        place_weights=data,
        ax=axes,
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
    replot_env: bool = False,
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
    - replot_env (bool, optional): Whether to plot the environment if axis provided.
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
        sub_ax = NeuronLayer.Agent.Environment.plot_environment(autosave=False)
    elif replot_env:
        sub_ax = NeuronLayer.Agent.Environment.plot_environment(
            sub_ax=sub_ax, autosave=False
        )

    if sub_ax is None:
        raise RuntimeError("sub_ax is None.")

    if NeuronLayer.Agent.Environment.dimensionality == "2D":
        reshape = NeuronLayer.Agent.Environment.discrete_coords.shape[:2]
    else:
        reshape = [-1]

    rate_maps = np.asarray(rate_maps)[chosen_neurons].reshape(N_neurons, *reshape)

    if method == "mean":
        rate_map = np.mean(rate_maps, axis=0)
    elif method == "median":
        rate_map = np.median(rate_maps, axis=0)
    elif method == "max":
        rate_map = np.max(rate_maps, axis=0)
    elif method == "min":
        rate_map = np.min(rate_maps, axis=0)
    elif method == "sum":
        rate_map = np.sum(rate_maps, axis=0)
    else:
        raise ValueError(f"method {method} not recognized.")

    # PLOT 2D
    if NeuronLayer.Agent.Environment.dimensionality == "2D":
        ex = NeuronLayer.Agent.Environment.extent
        im = sub_ax.imshow(rate_map, extent=ex, zorder=0, cmap="inferno")

        if colorbar == True:
            cax = sub_ax.figure.append_axes("right", size="5%", pad=0.05)  # type: ignore[has-method]
            cbar = plt.colorbar(im, cax=cax)
            vmin, vmax = rate_map.min(), rate_map.max()
            lim_v = vmax if vmax > -vmin else vmin
            cbar.set_ticks([0, lim_v])
            cbar.set_ticklabels([0, round(lim_v, 1)])
            cbar.outline.set_visible(False)

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

        sub_ax.set_xlabel("Position / m")
        sub_ax.set_ylabel("Neurons")

    fig = sub_ax.figure
    plot_util.save_figure(fig, f"{NeuronLayer.name}_overlayed_rate_maps", save=autosave)  # type: ignore[attr-defined]

    return sub_ax


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
    - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

    Returns:
    - sub_ax (plt.Axes): Subplot with timeseries of the layer plotted.
    """

    t = np.asarray(NeuronLayer.history["t"])
    startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)

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
    if imshow == False:
        firingrates = rate_timeseries.T
        _, sub_ax = rutils.mountain_plot(
            X=t / 60,
            NbyX=firingrates,
            color=color,  # type: ignore[assignment]
            xlabel="Time / min",
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
                time_when_spiked = t[spike_data[:, i]] / 60
                h = (i + 1 - 0.1) * np.ones_like(time_when_spiked)
                sub_ax.scatter(
                    time_when_spiked,
                    h,
                    color=(NeuronLayer.color or "C1"),  # type: ignore[attr-defined]
                    alpha=0.5,
                    s=5,
                    linewidth=0,
                )

        xmin = t[0] / 60 if was_ax else min(t[0] / 60, sub_ax.get_xlim()[0])  # type: ignore[operator]
        xmax = t[-1] / 60 if was_ax else max(t[-1] / 60, sub_ax.get_xlim()[1])  # type: ignore[operator]
        sub_ax.set_xlim(xmin, xmax)
        sub_ax.set_xticks([xmin, xmax])
        sub_ax.set_xticklabels([round(xmin, 2), round(xmax, 2)])
        if xlim is not None:
            sub_ax.set_xlim(right=xlim / 60)  # type: ignore[operator]
            sub_ax.set_xticks([round(t_start / 60, 2), round(xlim / 60, 2)])  # type: ignore[operator]
            sub_ax.set_xticklabels([round(t_start / 60, 2), round(xlim / 60, 2)])  # type: ignore[operator]

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
        sub_ax.set_xlabel("Time / min")
        sub_ax.set_xticks([t_start, t_end])
        sub_ax.set_xticklabels([round(t_start / 60, 2), round(t_end / 60, 2)])  # type: ignore[operator]
        sub_ax.set_yticks([])
        sub_ax.set_ylabel("Neurons")

    fig = sub_ax.figure
    plot_util.save_figure(fig, f"{NeuronLayer.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

    return sub_ax
