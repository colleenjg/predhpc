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

from predhpc import plot_fcts
from predhpc.util import plot_util, ext_util


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
    • self.get_oscillation_df()
    • self.log_oscillation_stats()
    • self.plot_oscillations()
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

        neuron_idx = oscillation_df["neuron_idx"].to_numpy()
        neuron_num = chosen_neurons[neuron_idx.astype(int)]
        oscillation_df["neuron_num"] = neuron_num

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
        for neuron_num in chosen_neurons:
            sub_df = oscillation_df[oscillation_df["neuron_num"] == neuron_num]
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

    def plot_oscillations(
        self,
        chosen_neurons="all",
        t_start=None,
        t_end=None,
        plot_type="full",
        aligned=True,
        pad_prop=0.1,
        order_by="neuron_num",
        reverse=False,
        max_num=1000,
        sharey=True,
        color=None,
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
            type is "individual". Default is "neuron_num".
        - reverse (bool, optional): Whether to reverse the order of the individual
            plots, if plot type is "individual". Default is False.
        - max_num (int, optional): Maximum number of individual plots to make, if plot
            type is "individual". Default is 1000.
        - sharey (bool, optional): Whether to share the y-axis across individual plots.
            Default is True.
        - color (str, optional): Color to plot the oscillations. Default is None.
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

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)
        firingrates = np.asarray(self.history["firingrate"])[
            startid : endid + 1, chosen_neurons
        ]

        oscillation_df = self.get_oscillation_df(
            chosen_neurons=chosen_neurons, t_start=t_start, t_end=t_end, **kwargs
        )

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
        - **kwargs: Keyword arguments passed to super().plot_rate_timeseries().

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
        plot_util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

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
        - **kwargs: Keyword arguments passed to super().plot_rate_timeseries().

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
            firingrates=self.history["firingrate"][startid:endid], **kwargs
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

        super().__init__(Agent, params=params)

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
