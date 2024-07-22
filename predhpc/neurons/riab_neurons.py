from matplotlib import pyplot as plt
import numpy as np

from ratinabox.Neurons import Neurons as riabNeurons  # type: ignore[import]
from ratinabox.Neurons import PlaceCells as riabPlaceCells  # type: ignore[import]
from ratinabox.Neurons import GridCells as riabGridCells  # type: ignore[import]
from ratinabox.Neurons import ObjectVectorCells as riabObjectVectorCells  # type: ignore[import]
from ratinabox.Neurons import FeedForwardLayer as riabFeedForwardLayer  # type: ignore[import]
from ratinabox.contribs import ValueNeuron as riabValueNeuron


class Neurons(riabNeurons):

    def plot_rate_map(self, ax=None, **kwargs):
        """Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.Neurons.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax = super().plot_rate_map(ax=ax, **kwargs)

        return ax

    def plot_rate_timeseries(self, sub_ax=None, **kwargs):
        """Plot rate timeseries for the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.Neurons.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        _, sub_ax = super().plot_rate_timeseries(**kwargs)

        return sub_ax


class PlaceCells(riabPlaceCells):

    def plot_rate_map(self, ax=None, **kwargs):
        """Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.PlaceCells.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax = super().plot_rate_map(ax=ax, **kwargs)

        return ax

    def plot_rate_timeseries(self, sub_ax=None, **kwargs):
        """Plot rate timeseries for the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.PlaceCells.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        _, sub_ax = super().plot_rate_timeseries(**kwargs)

        return sub_ax

    def plot_place_cell_locations(self, sub_ax=None, **kwargs):
        """Plot place cell locations for the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.PlaceCells.plot_place_cell_locations().

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

    def plot_rate_map(self, ax=None, **kwargs):
        """Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.GridCells.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax = super().plot_rate_map(ax=ax, **kwargs)

        return ax

    def plot_rate_timeseries(self, sub_ax=None, **kwargs):
        """Plot rate timeseries for the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.GridCells.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        _, sub_ax = super().plot_rate_timeseries(**kwargs)

        return sub_ax


class ObjectVectorCells(riabObjectVectorCells):

    def plot_rate_map(self, ax=None, **kwargs):
        """Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.ObjectVectorCells.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax = super().plot_rate_map(ax=ax, **kwargs)

        return ax

    def plot_rate_timeseries(self, sub_ax=None, **kwargs):
        """Plot rate timeseries for the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.ObjectVectorCells.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        _, sub_ax = super().plot_rate_timeseries(**kwargs)

        return sub_ax


class FeedForwardLayer(riabFeedForwardLayer):

    def plot_rate_map(self, ax=None, **kwargs):
        """Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.FeedForwardLayer.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax = super().plot_rate_map(ax=ax, **kwargs)

        return ax

    def plot_rate_timeseries(self, sub_ax=None, **kwargs):
        """Plot rate timeseries for the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.FeedForwardLayer.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        _, sub_ax = super().plot_rate_timeseries(**kwargs)

        return sub_ax


class ValueNeuron(riabValueNeuron):

    def plot_rate_map(self, ax=None, **kwargs):
        """Plot the rate map of the layer.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.contribs.ValueNeurons.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        if ax is not None:
            kwargs["fig"] = np.asarray(ax).ravel()[0].figure

        _, ax = super().plot_rate_map(ax=ax, **kwargs)

        return ax

    def plot_rate_timeseries(self, sub_ax=None, **kwargs):
        """Plot rate timeseries for the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments for ratinabox.contribs.ValueNeurons.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with timeseries plotted.
        """

        if sub_ax is not None:
            kwargs["ax"] = sub_ax

        if "ax" in kwargs.keys():
            kwargs["fig"] = kwargs["ax"].figure

        _, sub_ax = super().plot_rate_timeseries(**kwargs)

        return sub_ax
