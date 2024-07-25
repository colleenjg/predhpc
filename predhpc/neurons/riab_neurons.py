from matplotlib import pyplot as plt
import numpy as np

from ratinabox.Neurons import Neurons as riabNeurons  # type: ignore[import]
from ratinabox.Neurons import PlaceCells as riabPlaceCells  # type: ignore[import]
from ratinabox.Neurons import GridCells as riabGridCells  # type: ignore[import]
from ratinabox.Neurons import ObjectVectorCells as riabObjectVectorCells  # type: ignore[import]
from ratinabox.Neurons import FeedForwardLayer as riabFeedForwardLayer  # type: ignore[import]
from ratinabox.contribs import ValueNeuron as riabValueNeuron

from predhpc import plot_util, util


class Neurons(riabNeurons):
    """
    Neurons()

    Class extending ratinabox.Neurons.Neurons to add some functionalities to the
    plotting functions.

    List of methods (in addition ratinabox.Neurons.Neurons methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
    """

    def get_plotting_times(
        self, t_start: float | None = None, t_end: float | None = None
    ):
        """
        self.get_plotting_times()

        Obtain the times to plot.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.

        Returns:
        - t (1D np.ndarray): Times to plot.
        - startid (int): Index of the start time point.
        - endid (int): Index of the end time point.
        """

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t = t[startid : endid + 1]

        return t, startid, endid

    def plot_rate_map(self, ax=None, **kwargs):
        """
        self.plot_rate_map()

        Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Neurons.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax_out = super().plot_rate_map(ax=ax, **kwargs)

        if ax is not None:
            ax = ax_out

        return ax

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        adjust_xlim: bool = True,
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
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - adjust_xlim (bool, optional): Whether to adjust the x limits to the start
            and stop times. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            ax=sub_ax,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class PlaceCells(riabPlaceCells):
    """
    PlaceCells()

    Class extending ratinabox.Neurons.PlaceCells to add some functionalities to the
    plotting functions.

    List of methods (in addition ratinabox.Neurons.PlaceCells methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_place_cell_locations()
    """

    def get_plotting_times(
        self, t_start: float | None = None, t_end: float | None = None
    ):
        """
        self.get_plotting_times()

        Obtain the times to plot.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.

        Returns:
        - t (1D np.ndarray): Times to plot.
        - startid (int): Index of the start time point.
        - endid (int): Index of the end time point.
        """

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t = t[startid : endid + 1]

        return t, startid, endid

    def plot_rate_map(self, ax=None, **kwargs):
        """
        self.plot_rate_map()

        Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.PlaceCells.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax_out = super().plot_rate_map(ax=ax, **kwargs)

        if ax is not None:
            ax = ax_out

        return ax

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        adjust_xlim: bool = True,
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
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - adjust_xlim (bool, optional): Whether to adjust the x limits to the start
            and stop times. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            ax=sub_ax,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_place_cell_locations(self, sub_ax=None, **kwargs):
        """
        self.plot_place_cell_locations()

        Plot place cell locations for the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.PlaceCells.plot_place_cell_locations().

        Returns:
        - sub_ax (plt.Axes): Subplot with place cell locations plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        _, sub_ax = super().plot_place_cell_locations(**kwargs)

        return sub_ax


class GridCells(riabGridCells):
    """
    GridCells()

    Class extending ratinabox.Neurons.GridCells to add some functionalities to the
    plotting functions.

    List of methods (in addition ratinabox.Neurons.GridCells methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
    """

    def get_plotting_times(
        self, t_start: float | None = None, t_end: float | None = None
    ):
        """
        self.get_plotting_times()

        Obtain the times to plot.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.

        Returns:
        - t (1D np.ndarray): Times to plot.
        - startid (int): Index of the start time point.
        - endid (int): Index of the end time point.
        """

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t = t[startid : endid + 1]

        return t, startid, endid

    def plot_rate_map(self, ax=None, **kwargs):
        """
        self.plot_rate_map()

        Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.GridCells.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax_out = super().plot_rate_map(ax=ax, **kwargs)

        if ax is not None:
            ax = ax_out

        return ax

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        adjust_xlim: bool = True,
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
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - adjust_xlim (bool, optional): Whether to adjust the x limits to the start
            and stop times. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            ax=sub_ax,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class ObjectVectorCells(riabObjectVectorCells):
    """
    ObjectVectorCells()

    Class extending ratinabox.Neurons.ObjectVectorCells to add some functionalities to
    the plotting functions.

    List of methods (in addition ratinabox.Neurons.ObjectVectorCells methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
    """

    def get_plotting_times(
        self, t_start: float | None = None, t_end: float | None = None
    ):
        """
        self.get_plotting_times()

        Obtain the times to plot.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.

        Returns:
        - t (1D np.ndarray): Times to plot.
        - startid (int): Index of the start time point.
        - endid (int): Index of the end time point.
        """

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t = t[startid : endid + 1]

        return t, startid, endid

    def plot_rate_map(self, ax=None, **kwargs):
        """
        self.plot_rate_map()

        Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.ObjectVectorCells.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax_out = super().plot_rate_map(ax=ax, **kwargs)

        if ax is not None:
            ax = ax_out

        return ax

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        adjust_xlim: bool = True,
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
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - adjust_xlim (bool, optional): Whether to adjust the x limits to the start
            and stop times. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            ax=sub_ax,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class FeedForwardLayer(riabFeedForwardLayer):
    """
    FeedForwardLayer()

    Class extending ratinabox.Neurons.FeedForwardLayer to add some functionalities to
    the plotting functions.

    List of methods (in addition ratinabox.Neurons.FeedForwardLayer methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
    """

    def get_plotting_times(
        self, t_start: float | None = None, t_end: float | None = None
    ):
        """
        self.get_plotting_times()

        Obtain the times to plot.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.

        Returns:
        - t (1D np.ndarray): Times to plot.
        - startid (int): Index of the start time point.
        - endid (int): Index of the end time point.
        """

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t = t[startid : endid + 1]

        return t, startid, endid

    def plot_rate_map(self, ax=None, **kwargs):
        """
        self.plot_rate_map()

        Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.FeedForwardLayer.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax_out = super().plot_rate_map(ax=ax, **kwargs)

        if ax is not None:
            ax = ax_out

        return ax

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        adjust_xlim: bool = True,
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
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - adjust_xlim (bool, optional): Whether to adjust the x limits to the start
            and stop times. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            ax=sub_ax,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class ValueNeuron(riabValueNeuron):
    """
    ValueNeuron()

    Class extending ratinabox.contribs.ValueNeuron to add some functionalities to
    the plotting functions.

    List of methods (in addition ratinabox.contribs.ValueNeuron methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
    """

    def get_plotting_times(
        self, t_start: float | None = None, t_end: float | None = None
    ):
        """
        self.get_plotting_times()

        Obtain the times to plot.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.

        Returns:
        - t (1D np.ndarray): Times to plot.
        - startid (int): Index of the start time point.
        - endid (int): Index of the end time point.
        """

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t = t[startid : endid + 1]

        return t, startid, endid

    def plot_rate_map(self, ax=None, **kwargs):
        """
        self.plot_rate_map()

        Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to
            ratinabox.contribs.ValueNeurons.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax_out = super().plot_rate_map(ax=ax, **kwargs)

        if ax is not None:
            ax = ax_out

        return ax

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        adjust_xlim: bool = True,
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
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - adjust_xlim (bool, optional): Whether to adjust the x limits to the start
            and stop times. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            ax=sub_ax,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax
