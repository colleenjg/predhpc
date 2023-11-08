import copy
from typing import TYPE_CHECKING, Any, Sequence

import warnings

from matplotlib import pyplot as plt
from matplotlib import figure as mpl_figure
import numpy as np

from predhpc import util, plot_util
from predhpc.neurons import learning_neurons

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


class TwoCompLayer:
    """This trained class defines a population of neurons with two compartments.
    This class is a subclass of Neurons() and inherits its properties/plotting
    functions.

    Must be initialised with an Agent, and a "params" dictionary, including input
    layers.

    List of functions:
        • get_state()
        • add_input()
        • fit()
        • update()
        • plot_rate_map()
        • plot_loss()

    """

    default_params = {
        "n": 2,
        "name": "TwoCompLayer",
        "soma_input_layers": [],
        "dend_input_layers": [],
        "soma_to_dend_weight": 0.5,
        "dend_to_soma_weight": 0.5,
        "dend_first": False,
        "soma_color": "C0",
        "dend_color": "C1",
        "inhibit_dend": True,
        "inhibit_color": "k",
        "inhibit_weight": 0.5,  # multiplied by -1 identity matrix
        "inhibit_activation_params": util.get_standard_sigmoid_params(center_0=False),
        "inhibit_input_filter_tau": 0.1,
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(
        self,
        Agent: "ratinabox.Agent",
        params: dict[str, Any] = dict(),
    ):
        """Initialise RegressionLayer(), takes as input a parameter
        dictionary. Any values not provided by the params dictionary are
        taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to dict().
        """

        self.Agent = Agent

        self.organize_params(params)
        self.create_compartments()

        self.set_learn(soma=True, dend=True, inhibit=False)

    def organize_params(self, params: dict[str, Any]):
        self.soma_params = {"name": "soma"}
        self.dend_params = dict()

        all_params = copy.deepcopy(self.default_params)  # type: ignore[name-defined]
        all_params.update(params)

        for key, value in self.fixed_params.items():
            if key in params.keys() and value != params[key]:
                raise ValueError(
                    f"'{key}' parameter should not be passed, unless it is set to "
                    f"'{value}'."
                )
            params[key] = value

        local_attributes = [
            "soma_to_dend_weight",
            "dend_to_soma_weight",
            "dend_first",
        ]

        for key, value in all_params.items():
            if key in self.ignored_param_keys:
                warnings.warn(
                    f"'{key}' should not be provided for {cls.__name__}. "  # type: ignore[name-defined]
                    "Will be ignored."
                )

            elif key in local_attributes or key.startswith("inhibit_"):
                setattr(self, key, value)

            elif key == "name":
                self.name = value
                if "dend_name" not in all_params.keys():
                    self.dend_params["name"] = f"{value}_dend"
                if "soma_name" not in all_params.keys():
                    self.soma_params["name"] = f"{value}_soma"

            elif key == "n":
                self.n = value
                self.soma_params["n"] = value
                self.dend_params["n"] = value

            elif key.startswith("soma_") or key.startswith("dend_"):
                for compartment, comp_dict in [
                    ("soma", self.soma_params),
                    ("dend", self.dend_params),
                ]:
                    lead_str = f"{compartment}_"
                    if key.startswith(lead_str):
                        base_key = key.replace(lead_str, "")
                        if base_key == "n":
                            raise ValueError(
                                f"Cannot set 'n' for {compartment} layer only."
                            )
                        comp_dict[key.replace(lead_str, "")] = value

            else:
                self.soma_params[key] = value
                self.dend_params[key] = value

    def create_compartments(self):
        self.SomaCompartment = learning_neurons.NMDALayer(self.Agent, self.soma_params)
        self.DendriteCompartment = learning_neurons.NMDALayer(
            self.Agent, self.dend_params
        )

        if self.SomaCompartment.n != self.n or self.DendriteCompartment.n != self.n:  # type: ignore[attr-defined]
            raise ValueError(
                f"The two compartment layers must have same number of units ({self.n})."
            )

        dend_to_soma_weight = np.eye(self.n) * self.dend_to_soma_weight  # type: ignore[attr-defined]
        self.SomaCompartment.add_input(self.DendriteCompartment, w=dend_to_soma_weight)
        self.SomaCompartment.add_input_layers_with_no_learning(self.DendriteCompartment.name)  # type: ignore[attr-defined]

        soma_to_dend_weight = np.eye(self.n) * self.soma_to_dend_weight  # type: ignore[attr-defined]
        self.DendriteCompartment.add_input(self.SomaCompartment, w=soma_to_dend_weight)
        self.DendriteCompartment.add_input_layers_with_no_learning(self.SomaCompartment.name)  # type: ignore[attr-defined]

        if self.inhibit_dend:  # type: ignore[attr-defined]
            inhibit_params = {
                "name": "SomaInhibitionOfDendrites",
                "n": self.n,
                "activation_params": self.inhibit_activation_params,  # type: ignore[attr-defined]
                "color": self.inhibit_color,  # type: ignore[attr-defined]
                "input_filter_tau": self.inhibit_input_filter_tau,  # type: ignore[attr-defined]
            }

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "No input layers", UserWarning)
                self.DendriteInhibition = learning_neurons.HebbianLayer(
                    self.Agent, params=inhibit_params
                )

            soma_input = np.eye(self.n) * self.inhibit_weight  # type: ignore[attr-defined]
            self.DendriteInhibition.add_input(self.SomaCompartment, w=soma_input)

            dend_inhibition = np.eye(self.n) * -1
            self.DendriteCompartment.add_input(
                self.DendriteInhibition, w=dend_inhibition
            )

            self.DendriteCompartment.add_input_layers_with_no_learning(self.DendriteInhibition.name)  # type: ignore[attr-defined]

    def set_learn(self, learn=None, soma=None, dend=None, inhibit=None):
        if learn is not None:
            soma = learn if soma is None else soma
            dend = learn if dend is None else dend
            inhibit = learn if inhibit is None else inhibit

        self.SomaCompartment.set_learn(soma)
        self.DendriteCompartment.set_learn(dend)
        if self.inhibit_dend:  # type: ignore[attr-defined]
            self.DendriteInhibition.set_learn(inhibit)

    def set_btsp_learn(self, learn=None, soma=None, dend=None):
        if learn is not None:
            soma = learn if soma is None else soma
            dend = learn if dend is None else dend

        self.SomaCompartment.set_btsp_learn(soma)
        self.DendriteCompartment.set_btsp_learn(dend)

    def update(
        self,
        # btsp_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list(),
        dend_first: bool | None = None,
    ):
        if self.inhibit_dend:  # type: ignore[attr-defined]
            self.DendriteInhibition.update()

        if dend_first is None:
            dend_first = self.dend_first  # type: ignore[attr-defined]

        if dend_first:
            self.DendriteCompartment.update()
            self.SomaCompartment.update()
        else:
            self.SomaCompartment.update()
            self.DendriteCompartment.update()

        return

    def plot_rate_map(
        self,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        compartment: str | None = None,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]]:
        """Plot the rate map of the soma and dendritic layers, overlayed, ensuring no
        more than 20 columns are plotted.

        Args:
            fig (mpl_figure.Figure, optional): Figure object. Defaults to None.
            ax (plt.Axes, optional): Axes object. Defaults to None.
            compartment (str, optional): Which compartment to plot, if environment is
                2D ("soma", "dend" or "both"). Defaults to None
                (i.e., "soma" if environment is 2D, and "both" otherwise).
            no_legend (bool, optional): Whether to remove the legend. Defaults to False.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        if compartment is None:
            if self.Agent.Environment.dimensionality == "1D":
                compartment = "all"
            else:
                compartment = "soma"

        if self.Agent.Environment.dimensionality == "2D" and compartment == "all":
            warnings.warn(
                "Plotting rate maps for all compartments in a 2D environment will "
                "result in only the soma compartment appearing."
            )

        if compartment not in ["soma", "dend", "inhibit", "all"]:
            raise ValueError(
                f"compartment must be 'soma', 'dend', 'inhibit' or 'all', not '{compartment}'."
            )

        if compartment == "inhibit" and not self.inhibit_dend:  # type: ignore[attr-defined]
            raise ValueError(
                "Cannot plot inhibition rate maps, as inhibition is not enabled."
            )

        if self.inhibit_dend and compartment in ["all", "inhibit"]:
            fig, ax = self.DendriteInhibition.plot_rate_map(
                fig=fig,
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

        if compartment in ["all", "dend"]:
            self.DendriteCompartment.plot_rate_map(
                fig=fig,
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

        if compartment in ["all", "soma"]:
            fig, ax = self.SomaCompartment.plot_rate_map(
                fig=fig,
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

        if not no_legend and self.Agent.Environment.dimensionality == "1D":
            sub_ax = ax
            if compartment in ["all", "soma"]:
                sub_ax.plot([], [], color=self.SomaCompartment.color, label="soma")
            if compartment in ["all", "dend"]:
                sub_ax.plot([], [], color=self.DendriteCompartment.color, label="dend")
            if self.inhibit_dend and compartment in ["all", "inhibit"]:
                sub_ax.plot([], [], color=self.DendriteInhibition.color, label="inhib.")
            sub_ax.legend(loc="lower right")

        util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

        return fig, ax

    def plot_rate_timeseries(
        self,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        soma_color: str | None = None,
        dend_color: str | None = None,
        inhibit_color: str | None = None,
        autosave: bool | None = None,
        separate_axes: bool = False,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]]:
        """Plot a timeseries of the firing rate of the soma and dendritic layers of a
        neuron, overlayed.

        Args:
            fig (mpl_figure.Figure, optional): Figure object. Defaults to None.
            axes (plt.Axes, optional): Axes object. Defaults to None.
            soma_color (str, optional): Color for soma compartment. Defaults to None.
            dend_color (str, optional): Color for dendrite compartment. Defaults to None.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        if separate_axes:
            num_rows = 2 + self.inhibit_dend  # type: ignore[attr-defined]
            if ax is None:
                fig, ax = plt.subplots(
                    num_rows, 1, figsize=[6, 1.2 * num_rows], sharex=True, sharey=True
                )
            elif ax.shape != (num_rows,):
                raise ValueError(
                    f"ax must be a 1D array of length {num_rows}, not {ax.shape}."
                )

        soma_color = soma_color or self.SomaCompartment.color
        sub_ax = ax[0] if separate_axes else ax
        fig, sub_ax = self.SomaCompartment.plot_rate_timeseries(
            fig=fig,
            ax=sub_ax,
            color=soma_color,
            autosave=False,
            **kwargs,
        )
        ax = ax if separate_axes else sub_ax

        dend_color = dend_color or self.DendriteCompartment.color
        sub_ax = ax[1] if separate_axes else ax
        self.DendriteCompartment.plot_rate_timeseries(
            fig=fig,
            ax=sub_ax,
            color=dend_color,
            autosave=False,
            **kwargs,
        )

        if self.inhibit_dend:
            inhibit_color = inhibit_color or self.DendriteInhibition.color
            sub_ax = ax[2] if separate_axes else ax
            self.DendriteInhibition.plot_rate_timeseries(
                fig=fig,
                ax=sub_ax,
                color=inhibit_color,
                autosave=False,
                **kwargs,
            )

        if separate_axes:
            titles = ["Soma compartment", "Dendrite compartment", "Interneuron"]
            for s, sub_ax in enumerate(ax):
                sub_ax.set_title(titles[s])
                plot_util.add_target_reset_points(self.Agent, self, sub_ax)
                if s != len(ax) - 1:
                    sub_ax.set_xlabel("")
        else:
            ax.plot([], [], color=soma_color, label="soma")
            ax.plot([], [], color=dend_color, label="dend")
            if self.inhibit_dend:
                ax.plot([], [], color=inhibit_color, label="inhib.")
            ax.legend()

        util.save_figure(fig, f"{self.name}_firingrate", save=autosave)  # type: ignore[attr-defined]

        return fig, ax

    def plot_rate_maps_across_learning(
        self,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        compartment: str | None = None,
        no_legend: bool = False,
        title: str | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]]:
        """Plot the rate map of the soma and dendritic layers, overlayed, ensuring no
        more than 20 columns are plotted.

        Args:
            fig (mpl_figure.Figure, optional): Figure object. Defaults to None.
            ax (plt.Axes, optional): Axes object. Defaults to None.
            compartment (str, optional): Which compartment to plot, if environment is
                2D ("soma", "dend", "inhibit" or "both"). Defaults to None
                (i.e., "soma" if environment is 2D, and "both" otherwise).
            no_legend (bool, optional): Whether to remove the legend. Defaults to False.
            title (str, optional): Title for the figure. Defaults to None.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Keyword Args:
            kwargs: Keyword arguments to pass to plot_rate_maps_across_learning.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        if compartment is None:
            if self.Agent.Environment.dimensionality == "1D":
                compartment = "all"
            else:
                compartment = "soma"

        if self.Agent.Environment.dimensionality == "2D" and compartment == "both":
            warnings.warn(
                "Plotting rate maps across learning for both compartments in a 2D "
                "environment will result in only the soma compartment appearing."
            )

        if compartment not in ["soma", "dend", "inhibit", "all"]:
            raise ValueError(
                f"compartment must be 'soma', 'dend', 'inhibit' or 'all', not '{compartment}'."
            )

        if compartment == "inhibit" and not self.inhibit_dend:  # type: ignore[attr-defined]
            raise ValueError(
                "Cannot plot inhibition rate maps, as inhibition is not enabled."
            )

        if self.inhibit_dend and compartment in ["all", "inhibit"]:
            fig, ax = self.DendriteInhibition.plot_rate_maps_across_learning(
                fig=fig,
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

        if compartment in ["all", "dend"]:
            self.DendriteCompartment.plot_rate_maps_across_learning(
                fig=fig,
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

        if compartment in ["all", "soma"]:
            fig, ax = self.SomaCompartment.plot_rate_maps_across_learning(
                fig=fig,
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

        if not no_legend and self.Agent.Environment.dimensionality == "1D":
            sub_ax = ax

            if compartment in ["all", "soma"]:
                sub_ax.plot([], [], color=self.SomaCompartment.color, label="soma")
            if compartment in ["all", "dend"]:
                sub_ax.plot([], [], color=self.DendriteCompartment.color, label="dend")
            if self.inhibit_dend and compartment in ["all", "inhibit"]:
                sub_ax.plot([], [], color=self.DendriteInhibition.color, label="inhib.")
            sub_ax.legend(loc="lower right")

        if title is None:
            if compartment == "both":
                title_start = "Rate maps"
            elif compartment == "soma":
                title_start = "Soma rate maps"
            elif compartment == "inhibit":
                title_start = "Inhibition rate maps"
            else:
                title_start = "Dendrite rate maps"

            title = f"{title_start} across learning"

        fig.suptitle(title, y=0.90)

        util.save_figure(fig, f"{self.name}_rate_maps_across_learning", save=autosave)  # type: ignore[attr-defined]

        return fig, ax
