from collections.abc import Callable
import copy
import warnings

from matplotlib import pyplot as plt
import numpy as np

from ratinabox.Agent import Agent
from ratinabox.Neurons import Neurons as riabNeurons  # type: ignore[import]
from ratinabox.Neurons import PlaceCells as riabPlaceCells  # type: ignore[import]
from ratinabox.Neurons import GridCells as riabGridCells  # type: ignore[import]
from ratinabox.Neurons import ObjectVectorCells as riabObjectVectorCells  # type: ignore[import]
from ratinabox.Neurons import FeedForwardLayer as riabFeedForwardLayer  # type: ignore[import]
from ratinabox.contribs import ValueNeuron as riabValueNeuron
from ratinabox import utils as rutils  # type: ignore[import]

from predhpc import plot_fcts
from predhpc.util import signal_util, plot_util, ext_util


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Use kwarg `optimise_plot_for_single_neuron = True`",
)


class NeuronsMixin(ext_util.ParamsManagerMixin):
    """
    NeuronsMixin()

    Adds the following methods to a ratinabox.Neurons.Neurons object:
    • self.get_chosen_neurons()
    • self.get_plotting_times()
    • self.get_min_max_firingrates()
    • self.get_oscillation_df()
    • self.log_oscillation_stats()
    • self.get_binned_rates()
    • self.plot_oscillations()
    • self.plot_binned_rates()
    • self.plot_rate_map()
    • self.plot_rate_timeseries()
    • self.plot_rate_correlations()

    See NeuronsMixin for additional properties and methods.
    """

    Agent: Agent
    history: dict
    super: riabNeurons
    return_list_of_neurons: Callable

    # Default parameters
    default_params = dict()  # type: dict[str, Any]

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def get_chosen_neurons(self, chosen_neurons="all"):
        """
         self.get_chosen_neurons()

         Obtain the indices of the chosen neurons.

         Args:
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
             Default is "all".

         Returns:
         - chosen_neurons (1D np.ndarray): Indices of the chosen neurons.
        """

        chosen_neurons = np.asarray(self.return_list_of_neurons(chosen_neurons))

        return chosen_neurons

    def get_plotting_times(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        raise_error: bool = True,
    ):
        """
        self.get_plotting_times()

        Obtain the times to plot.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.
        - raise_error (bool, optional): Whether to raise an error if the start and end
            times are not in the history. Default is True.

        Returns:
        - t (1D np.ndarray): Times to plot.
        - startid (int): Index of the start time point.
        - endid (int): Index of the end time point.
        """

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(
            t, t_start=t_start, t_end=t_end, raise_error=raise_error
        )
        t = t[startid : endid + 1]

        return t, startid, endid

    def get_min_max_firingrates(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str | int | list | np.ndarray = "all",
    ):
        """
        self.get_min_max_firingrates()

        Obtain the minimum and maximum firing rates of the layer.

        Args:
        - t_start (float, optional): Start time for obtaining firingrate min and max.
            Default is None.
        - t_end (float, optional): Stop time for obtaining firingrate min and max.
            Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".

        Returns:
        - min_firingrate (float): Minimum firing rate.
        - max_firingrate (float): Maximum firing rate.
        """

        _, startid, endid = self.get_plotting_times(t_start, t_end)
        firingrates = np.asarray(self.history["firingrate"])[startid : endid + 1]

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)

        min_firingrate = np.min(firingrates[:, chosen_neurons])
        max_firingrate = np.max(firingrates[:, chosen_neurons])

        return min_firingrate, max_firingrate

    def get_history_ratemap(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str | int | list | np.ndarray = "all",
        bin_size: float | None = None,
        nan_zero_bins: bool = True,
    ):
        """
        self.get_history_ratemap()

        Obtain a rate map based on firingrate history.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - bin_size (float, optional): Bin size for the rate map. Default is None.
        - nan_zero_bins (bool, optional): Whether to set zero bins to NaN.
            Default is True.

        Returns:
        - rate_maps (2D np.ndarray): Rate maps for the chosen neurons. Unvisited bins
            are set to NaN.
        - centers (1D np.ndarray): center positions for the bins.
        """

        _, startid, endid = self.get_plotting_times(t_start, t_end)
        firingrates = np.asarray(self.history["firingrate"])[startid : endid + 1]
        pos = np.asarray(self.Agent.history["pos"])[startid : endid + 1]

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)

        if bin_size is None:
            bin_size = 0.05 if self.Agent.Environment.D == 2 else 0.1
        extent = self.Agent.Environment.extent

        if self.Agent.Environment.D == 1:
            pos = pos[:, 0]

        rate_maps = list()
        for chosen_neuron in chosen_neurons:
            outputs = rutils.bin_data_for_histogramming(
                data=pos,
                extent=self.Agent.Environment.extent,
                dx=bin_size,
                weights=firingrates[:, chosen_neuron],
                norm_by_bincount=True,
                return_zero_bins=True,
            )

            rate_map = outputs[0]
            if nan_zero_bins:
                rate_map[outputs[-1]] = np.nan
            rate_maps.append(rate_map)

        rate_maps = np.asarray(rate_maps)

        xedges = np.arange(extent[0], extent[1] + bin_size, bin_size)
        centers = (xedges[1:] + xedges[:-1]) / 2
        if self.Agent.Environment.D == 2:
            yedges = np.arange(extent[2], extent[3] + bin_size, bin_size)
            ycenters = (yedges[1:] + yedges[:-1]) / 2
            centers = np.asarray([centers, ycenters])

        return rate_maps, centers

    def get_firingrate_CC_matrix(
        self, num_periods: int = 8, plot: bool = False, sub_ax: plt.Axes | None = None
    ):
        """
        self.get_firingrate_CC_matrix()

        Obtain the firing rate cross-correlation matrix across periods.

        Args:
        - num_periods (int, optional): Number of periods to assess. Default is 8.
        - plot (bool, optional): Whether to plot the firing rate cross-correlation
            matrix. Default is False.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Returns:
        - CC_matrix (2D np.ndarray): Firing rate cross-correlation matrix across periods.
        if plot:
        - sub_ax (plt.Axes): Subplot with firing rate cross-correlation matrix plotted.
        """

        outputs = ext_util.assess_firingrate_CC_across_periods(
            self.history["firingrate"],
            num_periods=num_periods,
            plot=plot,
            sub_ax=sub_ax,
        )

        if plot:
            CC_matrix, sub_ax = outputs
            return CC_matrix, sub_ax
        else:
            CC_matrix = outputs
            return CC_matrix

    def get_oscillation_df(
        self,
        chosen_neurons="all",
        window=5,
        amp_thr=0.1,
        t_start=None,
        t_end=None,
    ):
        """
        self.get_oscillation_df()

        Obtain a DataFrame of single frame oscillations in the firing rates of the
        chosen neurons. Useful for debugging network oscillation.

        Args:
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - window (int, optional): Window size for identifying oscillations.
            Default is 5.
        - amp_thr (float, optional): Threshold for amplitude of firing rate
            oscillation mean and median. Default is 0.1.

        Returns:
        - oscillation_df (pd.DataFrame): DataFrame of oscillations in the firing rates.
        """

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)

        firingrates = np.asarray(self.history["firingrate"])[:, chosen_neurons]

        oscillation_df = ext_util.get_oscillation_df(
            firingrates, window=window, amp_thr=amp_thr
        )

        neuron_sub_idx = oscillation_df["neuron_sub_idx"].to_numpy()
        neuron_idx = chosen_neurons[neuron_sub_idx.astype(int)]
        oscillation_df["neuron_idx"] = neuron_idx

        if t_start is not None or t_end is not None:
            _, startid, endid = self.get_plotting_times(t_start, t_end)

            if t_start is not None:
                oscillation_df = oscillation_df.loc[
                    oscillation_df["stop_frame"] > startid
                ]

                oscillation_df.loc[
                    oscillation_df["start_frame"] < startid, "start_frame"
                ] = startid

            if t_end is not None:
                oscillation_df = oscillation_df.loc[
                    oscillation_df["start_frame"] < endid
                ]

                oscillation_df.loc[
                    oscillation_df["stop_frame"] > endid, "stop_frame"
                ] = endid

        oscillation_df["start_time"] = oscillation_df["start_frame"] * self.Agent.dt
        oscillation_df["stop_time"] = oscillation_df["stop_frame"] * self.Agent.dt

        return oscillation_df

    def log_oscillation_stats(self, chosen_neurons="all", **kwargs):
        """
        self.log_oscillation_stats()

        Log single frame oscillation statistics of the layer. Useful for debugging
        network oscillation.

        Args:
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons for
            which to log oscillation stats. Default is "all".

        Keyword args:
        - **kwargs: Keyword arguments passed to self.get_oscillation_df().
        """

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)
        oscillation_df = self.get_oscillation_df(chosen_neurons, **kwargs)

        num_events, prop_frs = list(), list()
        num_neurons = 0
        for neuron_idx in chosen_neurons:
            sub_df = oscillation_df[oscillation_df["neuron_idx"] == neuron_idx]
            if len(sub_df):
                num_neurons += 1
                num_events.append(len(sub_df))
                prop_frs.append(sub_df["num_frames"].sum())

        log_str = f"Oscillations found in {num_neurons}/{len(chosen_neurons)} neurons."

        if num_neurons:
            mean_num_events = np.mean(num_events)
            mean_prop_frs = int(np.around(np.mean(prop_frs)))
            log_str = f"{log_str}\n    ({mean_num_events:.2f} events per neuron)"
            log_str = f"{log_str}\n    ({mean_prop_frs} frames per neuron)"

        print(log_str)

    def get_binned_rates(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        num_bins: int = 100,
        part_run: float = 0.2,
        merge: bool = True,
        chosen_neurons: str | int | list | np.ndarray = "all",
        vel_sign_smooth: int = 5,
    ):
        """
        self.get_binned_rates()

        Obtain the firing rates of the layer, binned by position.

        Args:
        - t_start (float, optional): Time at which to start plotting data.
            Default is None.
        - t_end (float, optional): Time at which to stop plotting data.
            Default is None.
        - num_bins (int, optional): Number of bins to use for binning the firing rates.
            Default is 100.
        - part_run (float, optional): Proportion of the run to use for binning the
            firing rates. Default is 0.2.
        - merge (bool, optional): Whether to merge the firing rates of the neurons.
            Default is True.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - vel_sign_smooth (int, optional): Smoothing window for detecting velocity sign
            change. Default is 5.

        Returns:
        - binned_rate_means (1D np.ndarray): Mean binned firing rates.
        - occupancy (1D np.ndarray): Occupancy of the bins.
        """

        if self.Agent.Environment.dimensionality != "1D":
            raise ValueError(
                "Rate colormap plotting is only supported for 1D environments."
            )

        t, startid, endid = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)

        rate = np.asarray(self.history["firingrate"])[
            startid : endid + 1, chosen_neurons
        ]
        rel_pos = (
            np.asarray(self.Agent.history["pos"])[startid : endid + 1, 0]
            / self.Agent.Environment.scale
        )
        vel = np.asarray(self.Agent.history["vel"])[startid : endid + 1, 0]

        binned_rate_means, occupancy = signal_util.get_binned_rates(
            rate,
            rel_pos,
            vel=vel,
            num_bins=num_bins,
            part_run=part_run,
            merge=merge,
            vel_sign_smooth=vel_sign_smooth,  # higher value for detecting velocity sign change
        )

        return binned_rate_means, occupancy

    def plot_oscillations(
        self,
        chosen_neurons="all",
        t_start=None,
        t_end=None,
        plot_type="full",
        aligned=True,
        pad_prop=0.1,
        order_by="neuron_idx",
        reverse=False,
        max_num=1000,
        sharey=True,
        color=None,
        in_min=True,
        ax=None,
        **kwargs,
    ):
        """
        self.plot_oscillations()

        Plot the oscillations in the firing rates of the layer. Useful for debugging
        network oscillation.

        Args:
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - t_start (float, optional): Time at which to start plotting data.
            Default is None.
        - t_end (float, optional): Time at which to stop plotting data.
            Default is None.
        - plot_type (str, optional): Type of plot to make. Options are "full",
            "limited" or "individual". Default is "full".
        - aligned (bool, optional): Whether to align the oscillations, if plot type
            is "limited". Default is True.
        - pad_prop (float, optional): Proportion of the firing rate range to pad the
            oscillation plot, if plot type is "limited". Default is 0.1.
        - order_by (str, optional): Column to order the individual plots by, if plot
            type is "individual". Default is "neuron_idx".
        - reverse (bool, optional): Whether to reverse the order of the individual
            plots, if plot type is "individual". Default is False.
        - max_num (int, optional): Maximum number of individual plots to make, if plot
            type is "individual". Default is 1000.
        - sharey (bool, optional): Whether to share the y-axis across individual plots.
            Default is True.
        - color (str, optional): Color to plot the oscillations. Default is None.
        - in_min (bool, optional): Whether to plot time in minutes instead of seconds.
            Default is True.
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots
            (if plot_type is "individual") to plot on. If None, a new subplot or
            array of subplots is created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to self.get_oscillation_df().

        Returns:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots
            (if plot_type is "individual") with oscillations plotted.
        """

        t, startid, endid = self.get_plotting_times(t_start, t_end)
        if in_min:
            t = t / 60
            t_start = t_start / 60 if t_start is not None else None
            t_end = t_end / 60 if t_end is not None else None

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)
        firingrates = np.asarray(self.history["firingrate"])[
            startid : endid + 1, chosen_neurons
        ]

        oscillation_df = self.get_oscillation_df(
            chosen_neurons=chosen_neurons, t_start=t_start, t_end=t_end, **kwargs
        )
        if in_min:
            oscillation_df["start_time"] = oscillation_df["start_time"] / 60
            oscillation_df["stop_time"] = oscillation_df["stop_time"] / 60

        oscillation_df["start_frame"] -= startid
        oscillation_df["stop_frame"] -= startid

        if plot_type == "full":
            sub_ax = plot_fcts.plot_with_marked_oscillations(
                oscillation_df,
                firingrates,
                t=t,
                sub_ax=ax,
                color=color or self.color,
            )
        elif plot_type == "limited":
            sub_ax = plot_fcts.plot_oscillations(
                oscillation_df,
                firingrates,
                aligned=aligned,
                pad_prop=pad_prop,
                sub_ax=ax,
                color=color or self.color,
            )
        elif plot_type == "individual":
            sub_ax = plot_fcts.plot_oscillation_events(
                oscillation_df,
                firingrates,
                order_by=order_by,
                reverse=reverse,
                max_num=max_num,
                num_cols=15,
                sharey=sharey,
                axes=ax,
                color=color or self.color,
            )
        else:
            raise ValueError(
                f"Invalid plot_type: {plot_type}. Must be 'full', 'limited' or 'individual'."
            )

        return sub_ax

    def plot_binned_rates(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        ax: plt.Axes | np.ndarray | None = None,
        num_bins: int = 100,
        part_run: float = 0.2,
        merge: bool = True,
        vel_sign_smooth: int = 5,
        chosen_neurons: str | int | list | np.ndarray = "all",
        plot_occ: bool = True,
        shared_range: bool = False,
        vmin: float = 0,
        vmax: float | None = None,
        mark_runs: bool = False,
        plot_colorbars: bool = True,
        cbar_aspect: int = 12,
        cbar_label: str = "Firing rate",
        autosave: bool | None = None,
    ) -> plt.Axes | np.ndarray:
        """
        self.plot_binned_rates()

        Plot the firing rates of the layer, binned by position (for 1D environments only).

        Args:
        - t_start (float, optional): Time at which to start plotting data.
            Default is None.
        - t_end (float, optional): Time at which to stop plotting data.
            Default is None.
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.
        - num_bins (int, optional): Number of bins to use for binning the firing rates.
            Default is 100.
        - part_run (float, optional): Proportion of the run to use for binning the
            firing rates. Default is 0.2.
        - merge (bool, optional): Whether to merge the firing rates of the neurons.
            Default is True.
        - vel_sign_smooth (int, optional): Smoothing window for detecting velocity sign
            change. Default is 5.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - plot_occ (bool, optional): Whether to plot the occupancy. Default is True.
        - shared_range (bool, optional): Whether to use a shared range for the colormap.
            Default is False.
        - vmin (float, optional): Minimum value for the colormap. Default is 0.
        - vmax (float, optional): Maximum value for the colormap. Default is None.
        - mark_runs (bool, optional): Whether to mark runs in the plot. Default is False.
        - plot_colorbars (bool, optional): Whether to plot colorbars. Default is True.
        - cbar_aspect (int, optional): Aspect ratio of the colorbar. Default is 12.
        - cbar_label (str, optional): Label for the colorbar. Default is "Firing rate".
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is
           2D).
        """

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)

        binned_rate_means, occupancy = self.get_binned_rates(
            t_start=t_start,
            t_end=t_end,
            num_bins=num_bins,
            part_run=part_run,
            merge=merge,
            chosen_neurons=chosen_neurons,
            vel_sign_smooth=vel_sign_smooth,
        )

        if not plot_occ:
            occupancy = None

        ax = plot_util.plot_binned_rates(
            binned_rate_means,
            occupancy=occupancy,
            ax=ax,
            shared_range=shared_range,
            vmin=vmin,
            vmax=vmax,
            mark_runs=mark_runs,
            plot_colorbars=plot_colorbars,
            cbar_aspect=cbar_aspect,
            cbar_label=cbar_label,
        )

        for i in chosen_neurons:
            np.asarray(ax).ravel()[i].set_title(f"Neuron {i}")

        if autosave:
            fig = np.asarray(ax).ravel()[0].figure
            plot_util.save_figure(fig, f"{self.name}_binned_rates", save=autosave)

        return ax

    def plot_rate_map(self, ax=None, **kwargs):
        """
        self.plot_rate_map()

        Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to super().plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        kwargs = plot_util.organize_fig_ax_kwargs(ax=ax, return_env_fig=True, **kwargs)

        _, ax_out = super().plot_rate_map(**kwargs)

        if ax is None:
            ax = ax_out

        return ax

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str | int | list | np.ndarray = "all",
        sub_ax: plt.Axes | None = None,
        adjust_xlim: bool = True,
        in_min: bool = True,
        imshow: bool = False,
        norm_by: str | None = None,
        lw: float = 1.0,
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        self.plot_rate_timeseries()

        Plot the firing rate timeseries of the layer.

        Args:
        - t_start (float, optional): Time at which to start plotting data.
            Default is None.
        - t_end (float, optional): Time at which to stop plotting data.
            Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - adjust_xlim (bool, optional): Whether to adjust the x limits to the start
            and stop times. Default is True.
        - in_min (bool, optional): Whether to plot time in minutes instead of seconds.
            Default is True.
        - norm_by (str, optional): Normalization method for the firing rates. If "none",
            parameters are chosen so no normalization is applied. Default is None.
        - lw (float, optional): Line width for the firing rate timeseries. Default is 1.0.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to super().plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate timeseries plotted.
        """

        if "linewidth" in kwargs.keys():
            lw = kwargs.pop("linewidth")

        if not in_min:
            raise NotImplementedError("Plotting in seconds is not implemented.")

        kwargs = plot_util.organize_fig_ax_kwargs(
            sub_ax=sub_ax, return_env_fig=True, **kwargs
        )

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        adjust_ylim = False
        if not imshow and norm_by == "none":
            adjust_ylim = True
            kwargs["norm_by"] = 1
            kwargs["overlap"] = 1
            kwargs["global_shift"] = -1
        else:
            kwargs["norm_by"] = norm_by

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            chosen_neurons=chosen_neurons,
            linewidth=lw,
            imshow=imshow,
            autosave=False,
            **kwargs,
        )

        if adjust_ylim:
            chosen_neurons = self.get_chosen_neurons(chosen_neurons)
            min_fr, max_fr = self.get_min_max_firingrates(
                t_start=t_start, t_end=t_end, chosen_neurons=chosen_neurons
            )
            ymin = min(0, min_fr / kwargs["norm_by"])
            ymax = max_fr / kwargs["norm_by"] + len(chosen_neurons) - 1
            sub_ax.set_ylim(ymin, ymax)
            plot_util.pad_axis(sub_ax, pad_prop=0.2, axis="y")

        xlabel = "Time (min)" if in_min else "Time (s)"
        sub_ax.set_xlabel(xlabel)

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_rate_correlations(
        self,
        t_start: float | None = 0,
        t_end: float | None = None,
        autosave: bool | None = None,
        **kwargs,
    ):
        """
        self.plot_rate_correlations()

        Plot rate correlations between neurons.

        Args:
        - t_start (float, optional): Time at which to start plotting data.
            Default is 0.
        - t_end (float, optional): Time at which to stop plotting data.
            Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to plot_util.plot_rate_correlations().
            Default is dict().

        Returns:
        - sub_ax (plt.Axes): Subplot with rate correlations plotted.
        """

        _, startid, endid = self.get_plotting_times(t_start, t_end)

        sub_ax = plot_util.plot_rate_correlations(
            firingrates=self.history["firingrate"][startid : endid + 1], **kwargs
        )

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_rate_correlations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class Neurons(NeuronsMixin, riabNeurons):
    """
    Neurons()

    Class extending ratinabox.Neurons.Neurons to add some functionalities to the
    plotting functions.

    See ratinabox.Neurons.Neurons for default parameters, and
    NeuronsMixin for additional properties and methods.
    """

    default_params = riabNeurons.get_all_default_params()  # type: dict[str, Any]

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent, params=dict()):
        """
        Neurons(Agent)

        Initialise a neuron layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, params=params)


class PlaceCells(NeuronsMixin, riabPlaceCells):
    """
    PlaceCells()

    Class extending ratinabox.Neurons.PlaceCells to add some functionalities to the
    plotting functions.

    See ratinabox.Neurons.PlaceCells for default parameters, and
    NeuronsMixin for additional properties and methods.

    List of methods (in addition ratinabox.Neurons.PlaceCells methods):
        • self.plot_place_cell_locations()
        • self.shuffle_place_cell_locations
    """

    default_params = riabPlaceCells.get_all_default_params()  # type: dict[str, Any]

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent, params=dict()):
        """
        PlaceCells(Agent)

        Initialise a place cell layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self._handle_place_cell_center_param()
        params = copy.copy(self.params)

        super().__init__(Agent, params=params)

        self._current_sorter = np.arange(self.n)
        self.shuffle_sorters = list()
        self.shuffle_times = list()

    @property
    def place_cell_center_type(self):
        """
        self.place_cell_center_type

        Points to self.place_cell_centre_type.
        """

        return self.place_cell_centre_type

    @property
    def place_cell_centers(self):
        """
        self.place_cell_centers

        Points to self.place_cell_centres.
        """

        return self.place_cell_centres

    def _handle_place_cell_center_param(self):
        """
        self._handle_place_cell_center_param

        Stores the original description of the place cell center param if provided as
        a string. Also makes spelling flexible to both 'center' and 'centre'.
        """

        if "place_cell_centers" in self.params.keys():  # takes precedence
            self.params["place_cell_centres"] = self.params.pop("place_cell_centers")

        if isinstance(self.params["place_cell_centres"], str):
            self.place_cell_centre_type = self.params["place_cell_centres"]
        else:
            self.place_cell_centre_type = "specified"

    def get_place_field_FWHM(self, average=True):
        """
        self.get_place_field_FWHM()

        Obtain the full width at half maximum of the place fields.

        Args:
        - average (bool, optional): Whether to return the average FWHM. Default is True.

        Returns:
        - FWHM (float or 1D np.ndarray): Full width at half maximum of the place fields.
        """

        if self.description != "gaussian":
            raise ValueError("FWHM is only available for gaussian place fields.")

        FWHM = self.place_cell_widths * np.sqrt(8 * np.log(2))

        if average:
            FWHM = np.mean(FWHM)

        return FWHM

    def get_place_field_sigma_in_steps(self, average=True):
        """
        self.get_place_field_sigma_in_steps()

        Obtain the standard deviation of the place fields in steps.

        Args:
        - average (bool, optional): Whether to return the average sigma in steps.
            Default is True.

        Returns:
        - sigma_in_steps (float or 1D np.ndarray): Standard deviation of the place
            fields in steps.
        """

        mean_speed = self.Agent.speed_mean
        if self.Agent.Environment.dimensionality == "2D" and self.Agent.speed_std != 0:
            mean_speed = rutils.get_rayleigh_mean(mean_speed)

        sigma_in_steps = ext_util.get_sigma_in_steps(
            sigma=self.place_cell_widths, dt=self.Agent.dt, mean_speed=mean_speed
        )

        if average:
            sigma_in_steps = np.mean(sigma_in_steps)

        return sigma_in_steps

    def shuffle_place_cell_locations(
        self, randst=None, shuffle_sorter=None, record=True
    ):
        """
        self.shuffle_place_cell_locations()

        Shuffle the place cell locations. Always applied on original (not latest)
        place cell order.

        Args:
        - randst (int, optional): Random seed. Default is None.
        - shuffle_sorter (1D np.ndarray, optional): Shuffle sorter. Default is None.
        - record (bool, optional): Whether to record the shuffle. Default is True.

        Returns:
        - shuffle_sorter (1D np.ndarray): Shuffle sorter used.
        """

        if shuffle_sorter is None:
            randst = np.random.RandomState(randst)
            shuffle_sorter = np.arange(self.n)
            randst.shuffle(shuffle_sorter)

        if len(shuffle_sorter) != self.n:
            raise ValueError(
                "Length of shuffle_sorter must be equal to the number of place cells."
            )

        use_sorter = np.argsort(self._current_sorter)[shuffle_sorter]

        self.place_cell_centers[:] = self.place_cell_centers[use_sorter]

        self._current_sorter = shuffle_sorter

        if record:
            self.shuffle_sorters.append(shuffle_sorter)
            self.shuffle_times.append(self.Agent.t)

        return shuffle_sorter

    def plot_place_cell_locations(
        self,
        sub_ax=None,
        s=15,
        marker="x",
        alpha=0.8,
        chosen_neurons="all",
        plot_env=True,
        autosave=None,
        **kwargs,
    ):
        """
        self.plot_place_cell_locations()

        Plot place cell locations for the layer. Taken from
        ratinabox.PlaceCells.plot_place_cell_locations().

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - s (int, optional): Size of the markers. Default is 15.
        - marker (str, optional): Marker style for the place cell locations.
            Default is "x".
        - alpha (float, optional): Alpha transparency of the markers. Default is 0.8.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - plot_env (bool, optional): Whether to plot the environment if a subplot is
            provided. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to Environment.plot_environment().

        Returns:
        - sub_ax (plt.Axes): Subplot with place cell locations plotted.
        """

        fig = None
        if sub_ax is not None:
            fig = sub_ax.figure

        if plot_env or sub_ax is None:
            sub_ax = self.Agent.Environment.plot_environment(
                sub_ax=sub_ax, alpha=0.6, autosave=False, **kwargs
            )

        place_cell_centers = self.place_cell_centers
        if not isinstance(chosen_neurons, str) or chosen_neurons != "all":
            chosen_neurons = self.get_chosen_neurons(chosen_neurons)
            place_cell_centers = place_cell_centers[chosen_neurons]

        x = place_cell_centers[:, 0]
        if self.Agent.Environment.dimensionality == "1D":
            y = np.zeros_like(x)
        elif self.Agent.Environment.dimensionality == "2D":
            y = place_cell_centers[:, 1]

        sub_ax.scatter(x, y, c=self.color, alpha=alpha, marker=marker, s=s, zorder=2)

        plot_util.save_figure(fig, f"{self.name}_place_cell_locations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class GridCells(NeuronsMixin, riabGridCells):
    """
    GridCells()

    Class extending ratinabox.Neurons.GridCells to add some functionalities to the
    plotting functions.

    See ratinabox.Neurons.GridCells for default parameters, and
    NeuronsMixin for additional properties and methods.
    """

    default_params = riabGridCells.get_all_default_params()  # type: dict[str, Any]

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent, params=dict()):
        """
        GridCells(Agent)

        Initialise a grid cell layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, params=params)


class ObjectVectorCells(NeuronsMixin, riabObjectVectorCells):
    """
    ObjectVectorCells()

    Class extending ratinabox.Neurons.ObjectVectorCells to add some functionalities to
    the plotting functions.

    See ratinabox.Neurons.ObjectVectorCells for default parameters, and
    NeuronsMixin for additional properties and methods.
    """

    default_params = (
        riabObjectVectorCells.get_all_default_params()
    )  # type: dict[str, Any]

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent, params=dict()):
        """
        ObjectVectorCells(Agent)

        Initialise an object vector cell layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, params=params)


class FeedForwardLayer(NeuronsMixin, riabFeedForwardLayer):
    """
    FeedForwardLayer()

    Class extending ratinabox.Neurons.FeedForwardLayer to add some functionalities to
    the plotting functions.

    See ratinabox.Neurons.FeedForwardLayer for default parameters, and
    NeuronsMixin for additional properties and methods.
    """

    default_params = (
        riabFeedForwardLayer.get_all_default_params()
    )  # type: dict[str, Any]

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent, params=dict()):
        """
        FeedForwardLayer(Agent)

        Initialise a feedforward layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, params=params)


class ValueNeuron(NeuronsMixin, riabValueNeuron):
    """
    ValueNeuron()

    Class extending ratinabox.contribs.ValueNeuron to add some functionalities to
    the plotting functions.

    See ratinabox.contribs.ValueNeuron for default parameters, and
    NeuronsMixin for additional properties and methods.
    """

    default_params = riabValueNeuron.get_all_default_params()  # type: dict[str, Any]
    default_params["name"] = "ValueNeuron"

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent, params=dict()):
        """
        ValueNeuron(Agent)

        Initialise a value neuron layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, params=params)
