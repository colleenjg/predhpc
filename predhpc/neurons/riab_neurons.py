import copy
import warnings

from matplotlib import pyplot as plt
import numpy as np

from ratinabox.Neurons import Neurons as riabNeurons  # type: ignore[import]
from ratinabox.Neurons import PlaceCells as riabPlaceCells  # type: ignore[import]
from ratinabox.Neurons import GridCells as riabGridCells  # type: ignore[import]
from ratinabox.Neurons import ObjectVectorCells as riabObjectVectorCells  # type: ignore[import]
from ratinabox.Neurons import FeedForwardLayer as riabFeedForwardLayer  # type: ignore[import]
from ratinabox.contribs import ValueNeuron as riabValueNeuron

from predhpc import plot_util, util


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Use kwarg `optimise_plot_for_single_neuron = True`",
)


class Neurons(riabNeurons, util.ParamsManagerMixin):
    """
    Neurons()

    Class extending ratinabox.Neurons.Neurons to add some functionalities to the
    plotting functions.

    See ratinabox.Neurons.Neurons for default parameters.

    List of methods (in addition ratinabox.Neurons.Neurons methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_rate_correlations()
    """

    default_params = dict()  # type: dict[str, Any]

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

        kwargs = plot_util.organize_fig_ax_kwargs(ax=ax, return_env_fig=True, **kwargs)

        _, ax_out = super().plot_rate_map(**kwargs)

        if ax is None:
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

        kwargs = plot_util.organize_fig_ax_kwargs(
            sub_ax=sub_ax, return_env_fig=True, **kwargs
        )

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

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
            firingrates=self.history["firingrate"][startid:endid], **kwargs
        )

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_rate_correlations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class PlaceCells(riabPlaceCells, util.ParamsManagerMixin):
    """
    PlaceCells()

    Class extending ratinabox.Neurons.PlaceCells to add some functionalities to the
    plotting functions.

    See ratinabox.Neurons.PlaceCells for default parameters.

    List of methods (in addition ratinabox.Neurons.PlaceCells methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_place_cell_locations()
        • self.plot_rate_correlations()
    """

    default_params = dict()  # type: dict[str, Any]

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

        super().__init__(Agent, params=params)

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

        kwargs = plot_util.organize_fig_ax_kwargs(ax=ax, return_env_fig=True, **kwargs)

        _, ax_out = super().plot_rate_map(**kwargs)

        if ax is None:
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

        kwargs = plot_util.organize_fig_ax_kwargs(
            sub_ax=sub_ax, return_env_fig=True, **kwargs
        )

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_place_cell_locations(self, sub_ax=None, autosave=None, **kwargs):
        """
        self.plot_place_cell_locations()

        Plot place cell locations for the layer. Taken from
        ratinabox.PlaceCells.plot_place_cell_locations().

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
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

        sub_ax = self.Agent.Environment.plot_environment(
            sub_ax=sub_ax, autosave=False, **kwargs
        )

        place_cell_centres = self.place_cell_centres

        x = place_cell_centres[:, 0]
        if self.Agent.Environment.dimensionality == "1D":
            y = np.zeros_like(x)
        elif self.Agent.Environment.dimensionality == "2D":
            y = place_cell_centres[:, 1]

        sub_ax.scatter(x, y, c="C1", marker="x", s=15, zorder=2)

        util.save_figure(fig, f"{self.name}_place_cell_locations", save=autosave)  # type: ignore[attr-defined]

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
            firingrates=self.history["firingrate"][startid:endid], **kwargs
        )

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_rate_correlations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class GridCells(riabGridCells, util.ParamsManagerMixin):
    """
    GridCells()

    Class extending ratinabox.Neurons.GridCells to add some functionalities to the
    plotting functions.

    See ratinabox.Neurons.GridCells for default parameters.

    List of methods (in addition ratinabox.Neurons.GridCells methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_rate_correlations()
    """

    default_params = dict()  # type: dict[str, Any]

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

        kwargs = plot_util.organize_fig_ax_kwargs(ax=ax, return_env_fig=True, **kwargs)

        _, ax_out = super().plot_rate_map(**kwargs)

        if ax is None:
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

        kwargs = plot_util.organize_fig_ax_kwargs(
            sub_ax=sub_ax, return_env_fig=True, **kwargs
        )

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

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
            firingrates=self.history["firingrate"][startid:endid], **kwargs
        )

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_rate_correlations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class ObjectVectorCells(riabObjectVectorCells, util.ParamsManagerMixin):
    """
    ObjectVectorCells()

    Class extending ratinabox.Neurons.ObjectVectorCells to add some functionalities to
    the plotting functions.

    See ratinabox.Neurons.ObjectVectorCells for default parameters.

    List of methods (in addition ratinabox.Neurons.ObjectVectorCells methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_rate_correlations()
    """

    default_params = dict()  # type: dict[str, Any]

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

        kwargs = plot_util.organize_fig_ax_kwargs(ax=ax, return_env_fig=True, **kwargs)

        _, ax_out = super().plot_rate_map(**kwargs)

        if ax is None:
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

        kwargs = plot_util.organize_fig_ax_kwargs(
            sub_ax=sub_ax, return_env_fig=True, **kwargs
        )

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

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
            firingrates=self.history["firingrate"][startid:endid], **kwargs
        )

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_rate_correlations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class FeedForwardLayer(riabFeedForwardLayer, util.ParamsManagerMixin):
    """
    FeedForwardLayer()

    Class extending ratinabox.Neurons.FeedForwardLayer to add some functionalities to
    the plotting functions.

    See ratinabox.Neurons.FeedForwardLayer for default parameters.

    List of methods (in addition ratinabox.Neurons.FeedForwardLayer methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_rate_correlations()
    """

    default_params = dict()  # type: dict[str, Any]

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

        kwargs = plot_util.organize_fig_ax_kwargs(ax=ax, return_env_fig=True, **kwargs)

        _, ax_out = super().plot_rate_map(**kwargs)

        if ax is None:
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

        kwargs = plot_util.organize_fig_ax_kwargs(
            sub_ax=sub_ax, return_env_fig=True, **kwargs
        )

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

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
            firingrates=self.history["firingrate"][startid:endid], **kwargs
        )

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_rate_correlations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class ValueNeuron(riabValueNeuron, util.ParamsManagerMixin):
    """
    ValueNeuron()

    Class extending ratinabox.contribs.ValueNeuron to add some functionalities to
    the plotting functions.

    See ratinabox.contribs.ValueNeuron for default parameters.

    List of methods (in addition ratinabox.contribs.ValueNeuron methods):
        • self.get_plotting_times()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_rate_correlations()
    """

    default_params = dict()  # type: dict[str, Any]

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

        kwargs = plot_util.organize_fig_ax_kwargs(ax=ax, return_env_fig=True, **kwargs)

        _, ax_out = super().plot_rate_map(**kwargs)

        if ax is None:
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

        kwargs = plot_util.organize_fig_ax_kwargs(
            sub_ax=sub_ax, return_env_fig=True, **kwargs
        )

        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        _, sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            sub_ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            sub_ax.set_xticks(xticks)
            sub_ax.set_xticklabels(xticks)

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

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
            firingrates=self.history["firingrate"][startid:endid], **kwargs
        )

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_rate_correlations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax
