import copy
from typing import TYPE_CHECKING, Any, Sequence

import warnings

from matplotlib import pyplot as plt
import numpy as np

from predhpc import util, plot_util, params_util
from predhpc.neurons import learning_neurons

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


class TwoCompLayer(object):
    """
    TwoCompLayer()

    This neuron layer class defines a population of neurons with two compartments
    (somatic and dendritic), each of which is an NMDALayer. An additional HebbianLayer
    dendritic inhibition compartment can optionally be included

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 2,
        "name": "TwoCompLayer",
        "soma_input_layers": [],
        "dend_input_layers": [],
        "soma_to_dend_weight": 0.2,
        "dend_to_soma_weight": 1.0,
        "dend_first": True,
        "soma_color": "C0",
        "dend_color": "C1",
        "inhibit_dend": True,
        "inhibit_color": "k",
        "inhibit_weight": 3.0,  # multiplied by -1 identity matrix
        "inhibit_activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_FUNCTION,
        "inhibit_input_filter_tau": 3,
        "inhibit_input_trend_tau": None,
    }

    No property attributes.

    List of methods:
        • self.set_learn()
        • self.set_BTSP_learn()
        • self.update()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_rate_maps_across_learning()
    """

    default_params = {
        "n": 2,
        "name": "TwoCompLayer",
        "soma_input_layers": [],
        "dend_input_layers": [],
        "soma_to_dend_weight": 0.2,
        "dend_to_soma_weight": 1.0,
        "dend_first": True,
        "soma_color": "C0",
        "dend_color": "C1",
        "inhibit_dend": True,
        "inhibit_color": "k",
        "inhibit_weight": 3.0,  # multiplied by -1 identity matrix
        "inhibit_activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_FUNCTION,
        "inhibit_input_filter_tau": 3,
        "inhibit_input_trend_tau": None,
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(
        self,
        Agent: "ratinabox.Agent",
        params: dict[str, Any] = dict(),
    ):
        """
        TwoCompLayer(Agent)

        Initialise a two compartment layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Two compartment layer parameters. Default is dict().
        """

        self.Agent = Agent

        self._organize_params(params)
        self._create_compartments()

        self.set_learn(soma=True, dend=True, inhibit=False)

    def _organize_params(self, params: dict[str, Any]):
        """
        self._organize_params(params)

        Organise the parameters passed to the TwoCompLayer class, passing them to each
        compartment as appropriate.

        Attributes:
        - dend_params (dict): Parameters for the dendrite compartment.
        - name (str): Name of the layer.
        - soma_params (dict): Parameters for the soma compartment.

        Args:
        - params (dict): Parameters passed to the TwoCompLayer class.
        """

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

    def _create_compartments(self):
        """
        self._create_compartments()

        Create the soma and dendrite compartments, and connect them to each other for
        each neuron.

        If applicable, an inhibitory compartment is also created for each neuron and
        connected to the neuron's somatic and dendritic compartments.

        Inter-compartment connections:
            Soma <--> Dendrite
            if self.inhibit_dend:
                Soma -->* Dendritic inhibition --> Dendrite
            *: learning possible

        Attributes:
        - DendriteCompartment (learning_neurons.NMDALayer): Dendrite compartment.
        - DendriteInhibition (learning_neurons.HebbianLayer): Inhibitory compartment.
        - SomaCompartment (learning_neurons.NMDALayer): Soma compartment.
        """

        self.SomaCompartment = learning_neurons.NMDALayer(self.Agent, self.soma_params)
        self.DendriteCompartment = learning_neurons.NMDALayer(
            self.Agent, self.dend_params
        )

        if self.SomaCompartment.n != self.n or self.DendriteCompartment.n != self.n:  # type: ignore[attr-defined]
            raise ValueError(
                f"The two compartment layers must have same number of units ({self.n})."
            )

        dend_to_soma_weight = np.eye(self.n) * self.dend_to_soma_weight  # type: ignore[attr-defined]
        self.SomaCompartment.add_input_layers_with_no_learning(self.DendriteCompartment.name)  # type: ignore[attr-defined]
        self.SomaCompartment.add_input(
            self.DendriteCompartment,
            w=dend_to_soma_weight,
            recurrent=not (self.dend_first),
        )

        soma_to_dend_weight = np.eye(self.n) * self.soma_to_dend_weight  # type: ignore[attr-defined]
        self.DendriteCompartment.add_input_layers_with_no_learning(
            self.SomaCompartment.name  # type: ignore[attr-defined]
        )
        self.DendriteCompartment.add_input(
            self.SomaCompartment, w=soma_to_dend_weight, recurrent=self.dend_first
        )

        if self.inhibit_dend:  # type: ignore[attr-defined]
            inhibit_params = {
                "name": "SomaInhibitionOfDendrites",
                "n": self.n,
                "activation_function": self.inhibit_activation_function,  # type: ignore[attr-defined]
                "color": self.inhibit_color,  # type: ignore[attr-defined]
                "input_filter_tau": self.inhibit_input_filter_tau,  # type: ignore[attr-defined]
                "input_trend_tau": self.inhibit_input_trend_tau,  # type: ignore[attr-defined]
            }

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "No input layers", UserWarning)
                self.DendriteInhibition = learning_neurons.HebbianLayer(
                    self.Agent, params=inhibit_params
                )

            soma_input = np.eye(self.n) * self.inhibit_weight  # type: ignore[attr-defined]
            self.DendriteInhibition.add_input(self.SomaCompartment, w=soma_input)

            dend_inhibition = np.eye(self.n) * -1
            self.DendriteCompartment.add_input_layers_with_no_learning(
                self.DendriteInhibition.name  # type: ignore[attr-defined]
            )
            self.DendriteCompartment.add_input(
                self.DendriteInhibition, w=dend_inhibition, recurrent=self.dend_first
            )

    def set_learn(self, learn=None, soma=None, dend=None, inhibit=None):
        """
        self.set_learn()

        Set, for each compartment, whether it should learn during self.update() calls.
        Only affects input weights that are learnable.

        Args:
        - learn (bool, optional): Whether to learn learnable weights into all
            compartments.  Default is None.
        - soma (bool, optional): Whether to learn learnable weights into the
            soma compartment.  Default is None.
        - dend (bool, optional): Whether to learn learnable weights into the
            dendrite compartment. Default is None.
        """

        if learn is not None:
            soma = learn if soma is None else soma
            dend = learn if dend is None else dend
            inhibit = learn if inhibit is None else inhibit

        self.SomaCompartment.set_learn(soma)
        self.DendriteCompartment.set_learn(dend)
        if self.inhibit_dend:  # type: ignore[attr-defined]
            self.DendriteInhibition.set_learn(inhibit)

    def set_BTSP_learn(self, learn=None, soma=None, dend=None):
        """
        self.set_BTSP_learn()

        Set whether the soma and dendrite compartments should learn using BTSP during
        self.update() calls. Only affects input weights that are learnable.

        Args:
        - learn (bool, optional): Whether to learn learnable weights into both
            compartments. Default is None
        - soma (bool, optional): Whether to learn learnable weights into the
            soma compartment.  Default is None.
        - dend (bool, optional): Whether to learn learnable weights into the
            dendrite compartment. Default is None.
        """

        if learn is not None:
            soma = learn if soma is None else soma
            dend = learn if dend is None else dend

        self.SomaCompartment.set_BTSP_learn(soma)
        self.DendriteCompartment.set_BTSP_learn(dend)

    def update(self, dend_first: bool | None = None):
        """
        self.update()

        Update the somatic and dendritic compartments of the two compartment layer. If
        there is an dendrite inhibition compartment, it is also updated.

        Update order:
            1. Dendrite inhibition compartment (if applicable)
            if dend_first:
                2. Dendrite compartment
                3. Soma compartment
            otherwise:
                2. Soma compartment
                3. Dendrite compartment

        Args:
        - dend_first (bool, optional): Whether to update the dendrite compartment
            before the soma compartment. If None, the attribute is used.
            Default is None.
        """

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
        ax: plt.Axes | np.ndarray | None = None,
        compartment: str | None = None,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | plt.Axes:
        """
        self.plot_rate_map()

        Plot the rate map of the specified compartments, overlayed, with one subplot
        per two-compartment neuron.

        Args:
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.
        - compartment (str, optional): Which compartment to plot, if environment is
            2D ("soma", "dend" or "both"). Default is None
            (i.e., "soma" if environment is 2D, and "both" otherwise).
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to NMDALayer.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots with rate maps
            plotted (one per plotted ROI).
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
            ax_out = self.DendriteInhibition.plot_rate_map(
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

            if ax is None:
                ax = ax_out

        if compartment in ["all", "dend"]:
            ax_out = self.DendriteCompartment.plot_rate_map(
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

            if ax is None:
                ax = ax_out

        if compartment in ["all", "soma"]:
            ax_out = self.SomaCompartment.plot_rate_map(
                ax=ax,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )

            if ax is None:
                ax = ax_out

        if not no_legend and self.Agent.Environment.dimensionality == "1D":
            sub_ax = ax
            if compartment in ["all", "soma"]:
                sub_ax.plot([], [], color=self.SomaCompartment.color, label="soma")
            if compartment in ["all", "dend"]:
                sub_ax.plot([], [], color=self.DendriteCompartment.color, label="dend")
            if self.inhibit_dend and compartment in ["all", "inhibit"]:
                sub_ax.plot([], [], color=self.DendriteInhibition.color, label="inhib.")
            sub_ax.legend(loc="lower right")

        fig = np.asarray(ax).ravel()[0].figure
        util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

        return ax

    def plot_rate_maps_across_learning(
        self,
        axes: plt.Axes | np.ndarray | None = None,
        compartment: str | None = None,
        no_legend: bool = False,
        title: str | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
        """
        self.plot_rate_maps_across_learning()

        Plot the rate map of the specified compartments, overlayed, with one subplot
        per two-compartment neuron and for each stage of learning.

        Args:
        - axes (2D np.ndarray, optional): Array of subplots to plot on with shape
            (number of ROIs, num_maps) or v.v.. If None, a new subplot array is created.
            Default is None.
        - compartment (str, optional): Which compartment to plot, if environment is
            2D ("soma", "dend", "inhibit" or "both"). Default is None
            (i.e., "soma" if environment is 2D, and "both" otherwise).
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - title (str, optional): Title for the figure. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to
            NMDALayer.plot_rate_maps_across_learning().

        Raises:
        - ValueError: If compartment is not "soma", "dend", "inhibit" or "all".
        - ValueError: If compartment is "inhibit", but self.inhibit_dend is False.

        Returns:
        - axes (2D np.ndarray): Array of subplots. If input axes was None,
            shape is 2D (number of ROIs, num_maps) or v.v. if only one ROI.
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
            axes_out = self.DendriteInhibition.plot_rate_maps_across_learning(
                axes=axes,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )
            if axes is None:
                axes = axes_out

        if compartment in ["all", "dend"]:
            axes_out = self.DendriteCompartment.plot_rate_maps_across_learning(
                axes=axes,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )
            if axes is None:
                axes = axes_out

        if compartment in ["all", "soma"]:
            axes_out = self.SomaCompartment.plot_rate_maps_across_learning(
                axes=axes,
                autosave=False,
                no_legend=no_legend,
                **kwargs,
            )
            if axes is None:
                axes = axes_out

        if not no_legend and self.Agent.Environment.dimensionality == "1D":
            sub_ax = np.asarray(axes).ravel()[0]

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

        fig = np.asarray(axes).ravel()[0].figure
        fig.suptitle(title, y=0.90)

        util.save_figure(fig, f"{self.name}_rate_maps_across_learning", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_rate_timeseries(
        self,
        ax: plt.Axes | np.ndarray | None = None,
        soma_color: str | None = None,
        dend_color: str | None = None,
        inhibit_color: str | None = None,
        separate_axes: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
        """
        self.plot_rate_timeseries()

        Plot a timeseries of the firing rate of the specified compartments, either
        overlayed or split across subplots.

        Args:
        - ax (1D np.ndarray or plt.Axes, optional): Subplot or 1D array of subplots
            if separate_axes (one per compartment). Default is None.
        - soma_color (str, optional): Color for soma compartment. Default is None.
        - dend_color (str, optional): Color for dendrite compartment. Default is None.
        - inhibit_color (str, optional): Color for inhibitory compartment.
            Default is None.
        - separate_axes (bool, optional): Whether to plot each compartment on a
            separate subplot. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to NMDALayer.plot_rate_timeseries().

        Returns:
        - ax (2D np.ndarray or plt.Axes): Subplot or 1D array of subplots
            if separate_axes (one per compartment).
        """

        if separate_axes:
            num_rows = 2 + self.inhibit_dend  # type: ignore[attr-defined]
            if ax is None:
                _, ax = plt.subplots(
                    num_rows,
                    1,
                    figsize=[6, 1.2 * num_rows],
                    sharex=True,
                    sharey=True,
                    squeeze=False,
                )

            else:
                ax_shape = np.asarray(ax).shape
                if not (ax_shape == (num_rows,) or ax_shape == (num_rows, 1)):
                    raise ValueError(
                        f"ax must have shape ({num_rows}, ) or ({num_rows}, 1)."
                    )

            ax1D = np.asarray(ax).ravel()
        else:
            sub_ax = ax

        soma_color = soma_color or self.SomaCompartment.color
        soma_ax = ax1D[0] if separate_axes else sub_ax
        soma_ax = self.SomaCompartment.plot_rate_timeseries(
            sub_ax=soma_ax,
            color=soma_color,
            autosave=False,
            **kwargs,
        )

        dend_color = dend_color or self.DendriteCompartment.color
        dend_ax = ax1D[1] if separate_axes else sub_ax
        self.DendriteCompartment.plot_rate_timeseries(
            sub_ax=dend_ax,
            color=dend_color,
            autosave=False,
            **kwargs,
        )

        if self.inhibit_dend:
            inhibit_color = inhibit_color or self.DendriteInhibition.color
            inh_ax = ax1D[2] if separate_axes else sub_ax
            self.DendriteInhibition.plot_rate_timeseries(
                sub_ax=inh_ax,
                color=inhibit_color,
                autosave=False,
                **kwargs,
            )

        if separate_axes:
            titles = ["Soma compartment", "Dendrite compartment", "Interneuron"]
            for s, sub_ax in enumerate(ax1D):
                sub_ax.set_title(titles[s])
                plot_util.mark_target_and_reset_points(self.Agent, self, sub_ax=sub_ax)
                if s != len(ax1D) - 1:
                    sub_ax.set_xlabel("")
        else:
            soma_ax.plot([], [], color=soma_color, label="soma")
            soma_ax.plot([], [], color=dend_color, label="dend")
            if self.inhibit_dend:
                soma_ax.plot([], [], color=inhibit_color, label="inhib.")
            soma_ax.legend()

        fig = np.asarray(ax).ravel()[0].figure
        util.save_figure(fig, f"{self.name}_firingrate", save=autosave)  # type: ignore[attr-defined]

        return ax
