import copy
from typing import TYPE_CHECKING, Any, Sequence

import warnings

from matplotlib import pyplot as plt
import numpy as np

from predhpc import plot_fcts
from predhpc.neurons import learning_neurons, riab_neurons
from predhpc.util import gen_util, plot_util, params_util

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
        "inhibit_activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_PARAMS,
        "inhibit_input_filter_tau": 3,
        "inhibit_input_trend_tau": None,
        "mutual_inhibition_weight": None,
        "lateral_tau": 0.3,
    }

    No property attributes.

    List of methods:
        • self.set_learn()
        • self.set_BTSP_learn()
        • self.get_place_cell_centre_of_main_dendrite_input()
        • self.get_vectors_to_place_cell_centre_of_main_dendrite_input()
        • self.get_distances_to_place_cell_centre_of_main_dendrite_input()
        • self.get_closest_steps_to_target()
        • self.match_closest_to_target_steps_to_BTSP_steps()
        • self.update()
        • self.add_compartment_legend()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_rate_maps_across_learning()
        • self.plot_distances_to_target()
        • self.plot_distances_to_targets()
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
        "inhibit_activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_PARAMS,
        "inhibit_input_filter_tau": 3,
        "inhibit_input_trend_tau": None,
        "mutual_inhibition_weight": None,
        "lateral_tau": 0.1,
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
            "mutual_inhibition_weight",
            "lateral_tau",
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

        if self.mutual_inhibition_weight is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "No input layers", UserWarning)

                lateral_params = {
                    "name": "LateralInhibition",
                    "n": self.n,
                    "activation_function": self.SomaCompartment.activation_function,
                    "color": "gray",
                    "input_filter_tau": self.lateral_tau,
                }

                self.LateralInhibition = learning_neurons.SmoothFeedForwardLayer(
                    self.Agent, params=lateral_params
                )

            self.LateralInhibition.add_input(self.SomaCompartment, w=np.eye(self.n))
            mutual_inhibition = (np.eye(self.n) - 1) * self.mutual_inhibition_weight
            self.SomaCompartment.add_input(self.LateralInhibition, w=mutual_inhibition)
            self.SomaCompartment.add_input_layers_with_no_learning(
                self.LateralInhibition.name
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

    def get_compartments(
        self,
        compartment: str = "all",
        incl_lateral: bool = False,
    ):
        """
        self.get_compartments()

        - compartment (str, optional): Which compartments to retrieve
            ("soma", "dend", "both", "inhibit", "all"). Default is "all".

        Returns:
        - compartments (list): List of compartments.
        """

        if compartment not in ["soma", "dend", "both", "inhibit", "all"]:
            raise ValueError(
                "compartment must be 'soma', 'dend', 'both', 'inhibit' or 'all', "
                f"not '{compartment}'."
            )

        compartments = list()
        if compartment in ["soma", "both", "all"]:
            compartments.append(self.SomaCompartment)
        if compartment in ["dend", "both", "all"]:
            compartments.append(self.DendriteCompartment)
        if compartment in ["inhibit", "all"]:
            if self.inhibit_dend:
                compartments.append(self.DendriteInhibition)
            elif compartment == "inhibit":  # type: ignore[attr-defined]
                raise ValueError(
                    "Cannot retrieve inhibition compartment, as inhibition is not enabled."
                )
        if incl_lateral:
            if self.mutual_inhibition_weight is None:
                raise ValueError(
                    "Cannot retrieve lateral inhibition compartment, as mutual "
                    "inhibition is not enabled."
                )
            else:
                compartments.append(self.LateralInhibition)

        return compartments

    def get_min_max_firingrates(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        compartment: str = "all",
        incl_lateral: bool = False,
    ):
        """
        self.get_min_max_firingrates()

        Obtain the minimum and maximum firing rates of the layer.

        Args:
        - t_start (float, optional): Start time for obtaining firingrate min and max.
            Default is None.
        - t_end (float, optional): Stop time for obtaining firingrate min and max.
            Default is None.
        - compartment (str, optional): Which compartment to plot, if environment is
            2D ("soma", "dend", "both", "inhibit", "all"). Default is "all".

        Returns:
        - min_firingrate (float): Minimum firing rate.
        - max_firingrate (float): Maximum firing rate.
        """

        min_firingrate = np.inf
        max_firingrate = -np.inf

        compartments = self.get_compartments(compartment, incl_lateral=incl_lateral)
        for comp in compartments:
            min_rate, max_rate = comp.get_min_max_firingrates(
                t_start=t_start, t_end=t_end
            )
            min_firingrate = min(min_firingrate, min_rate)
            max_firingrate = max(max_firingrate, max_rate)

        return min_firingrate, max_firingrate

    def get_place_cell_centre_of_main_dendrite_input(
        self, neuron_num: int = 0, src_name: str = "Obj"
    ):
        """
        self.get_place_cell_centre_of_main_dendrite_input()

        Get the place cell centre input location for the dendrite of a specified neuron.

        Args:
        - neuron_num (int, optional): Neuron number. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".

        Returns:
        - place_cell_centre (1D np.ndarray): Main dendrite input place cell centre
            location.
        """

        if src_name not in self.DendriteCompartment.inputs.keys():
            raise ValueError(f"No '{src_name}' input to dendrite.")

        input_dict = self.DendriteCompartment.inputs[src_name]

        if not isinstance(input_dict["layer"], riab_neurons.PlaceCells):
            raise ValueError(f"Input layer '{src_name}' is not a PlaceCells layer.")

        if neuron_num > self.n:
            raise ValueError(
                f"Neuron number {neuron_num} is greater than the number of neurons "
                "in the layer."
            )

        input_idx = np.argmax(input_dict["w"][:, neuron_num])
        place_cell_centre = input_dict["layer"].place_cell_centres[input_idx]

        return place_cell_centre

    def get_vectors_to_place_cell_centre_of_main_dendrite_input(
        self,
        neuron_num: int = 0,
        src_name: str = "Obj",
        polar: bool = False,
        radians: bool = False,
    ):
        """
        self.get_vectors_to_place_cell_centre_of_main_dendrite_input()

        Get the vectors from the agent's current position to the place cell centre
        input location for the dendrite of a specified neuron.

        Args:
        - neuron_num (int, optional): Neuron number. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - polar (bool, optional): Whether to return vectors in polar coordinates.
            Default is False.
        - radians (bool, optional): If True and polar is True, return angles in radians.
            Default is False.

        Returns:
        - vectors (2D np.ndarray): Vectors from agent's position to dendrite input
            place cell centre.
        """

        place_cell_centre = self.get_place_cell_centre_of_main_dendrite_input(
            neuron_num=neuron_num, src_name=src_name
        )
        pos = np.asarray(self.Agent.history["pos"])

        vectors = gen_util.get_vectors_to_target(
            pos, target=place_cell_centre, polar=polar, radians=radians
        )

        return vectors

    def get_distances_to_place_cell_centre_of_main_dendrite_input(
        self, neuron_num: int = 0, src_name: str = "Obj"
    ):
        """
        self.get_distances_to_place_cell_centre_of_main_dendrite_input()

        Get the distances from the agent's current position to the place cell centre
        input location for the dendrite of a specified neuron.

        Args:
        - neuron_num (int, optional): Neuron number. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".

        Returns:
        - distances (1D np.ndarray): Distances from agent's position to dendrite input
            place cell centre.
        """

        vectors = self.get_vectors_to_place_cell_centre_of_main_dendrite_input(
            neuron_num, src_name
        )

        distances = np.linalg.norm(vectors, ord=2, axis=1)

        return distances

    def get_closest_steps_to_target(
        self,
        neuron_num=0,
        target_src_name="Obj",
        min_dist=0.2,
        min_steps_btw=20,
        log=False,
    ):
        """
        self.get_closest_steps_to_target()

        Get the steps where the agent is closest to the target specified by the place
        cell centre of the main input to the neuron's dendrite.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - neuron_num (int, optional): Neuron number. Default is 0.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.2.
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 20.
        - log (bool, optional): Whether to print the number of closest steps identified.
            Default is False.

        Returns:
        - closest_steps (1D np.ndarray): Steps identified as locally closest to the target.
        """

        distances = self.get_distances_to_place_cell_centre_of_main_dendrite_input(
            neuron_num, src_name=target_src_name
        )
        closest_steps = gen_util.get_minima_indices(
            distances, minimum=min_dist, min_pts_btw=min_steps_btw
        )

        if log:
            print(
                f"{len(closest_steps)}/{len(distances)} steps identified as locally "
                "closest to the target."
            )

        return closest_steps

    def match_closest_to_target_steps_to_BTSP_steps(
        self,
        target_src_name="Obj",
        neuron_num=0,
        max_step_dist=40,
        min_dist=0.2,
        t_start=None,
        t_end=None,
    ):
        """
        self.match_closest_to_target_steps_to_BTSP_steps()

        Match the steps closest to the target to the BTSP steps of the specified neuron.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - neuron_num (int, optional): Neuron number. Default is 0.
        - max_step_dist (int, optional): Maximum distance between steps to be considered
            a match. Default is 40.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.2.
        - t_start (int, optional): Start time for matching. Default is None.
        - t_end (int, optional): End time for matching. Default is None.

        Returns:
        - steps_dict (dict): Dictionary of steps closest to the target, matched to
            BTSP steps, with keys:
                "steps_before": closest steps occurring before neuron's first BTSP step
                "steps_near_BTSP": closest steps occurring near BTSP steps
                "steps_of_nearest_BTSP": index of nearest BTSP step for each step_near_BTSP value
                "steps_other": closest steps after first BTSP, but not near a BTSP step
                "all_BTSP_steps": all BTSP steps, whether close to target or not
        """

        if neuron_num >= self.n:
            raise ValueError(
                f"Neuron number ({neuron_num}) must be smaller than number of "
                f"neurons ({self.n})."
            )

        _, start, end = self.SomaCompartment.get_plotting_times(
            t_start=t_start, t_end=t_end
        )

        BTSP_steps = self.SomaCompartment.get_BTSP_step_dict()[neuron_num]
        closest_steps = self.get_closest_steps_to_target(
            neuron_num, target_src_name=target_src_name, min_dist=min_dist
        )

        keys = [
            "steps_before",
            "steps_near_BTSP",
            "steps_of_nearest_BTSP",
            "steps_other",
            "other_BTSP_steps",
        ]
        steps_dict = {key: list() for key in keys}
        if len(BTSP_steps):
            for step in closest_steps:
                if step < start or step >= end:
                    continue
                diff = np.absolute(BTSP_steps - step)
                if diff.min() < max_step_dist:
                    key = "steps_near_BTSP"
                    steps_dict["steps_of_nearest_BTSP"].append(
                        BTSP_steps[np.argmin(diff)]
                    )
                elif step < min(BTSP_steps):
                    key = "steps_before"
                else:
                    key = "steps_other"
                steps_dict[key].append(step)

            steps_dict["other_BTSP_steps"] = [
                step
                for step in BTSP_steps
                if step not in steps_dict["steps_of_nearest_BTSP"]
            ]
        else:
            steps_dict["steps_other"] = closest_steps

        return steps_dict

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
            if self.mutual_inhibition_weight is not None:
                self.LateralInhibition.update()
        else:
            self.SomaCompartment.update()
            self.DendriteCompartment.update()
            if self.mutual_inhibition_weight is not None:
                self.LateralInhibition.update()

        return

    def add_compartment_legend(
        self,
        sub_ax,
        compartment="all",
        plot_lateral=False,
        loc="best",
        soma_color=None,
        dend_color=None,
        inhibit_color=None,
        lateral_color=None,
    ):
        """
        self.add_compartment_legend()

        Add a legend to a plot with the colors of the specified compartments.

        Args:
        - sub_ax (plt.Axes): Subplot to add the legend to.
        - compartment (str, optional): Which compartments to include in the legend
            ("soma", "dend", "inhibit", "all"). Default is "all".
        - plot_lateral (bool, optional): Whether to include the lateral inhibition
            compartment in the legend. Default is False.
        - soma_color (str, optional): Color for soma compartment. Default is None.
        - dend_color (str, optional): Color for dendrite compartment. Default is None.
        - inhibit_color (str, optional): Color for inhibitory compartment.
            Default is None.
        - lateral_color (str, optional): Color for lateral inhibition compartment.
            Default is None.
        """

        if compartment not in ["soma", "dend", "both", "inhibit", "all"]:
            raise ValueError(
                "compartment must be 'soma', 'dend', 'both', 'inhibit' or 'all', not "
                f"'{compartment}'."
            )

        if compartment in ["all", "soma"]:
            color = soma_color or self.SomaCompartment.color
            sub_ax.plot([], [], color=color, label="soma")
        if compartment in ["all", "dend"]:
            color = dend_color or self.DendriteCompartment.color
            sub_ax.plot([], [], color=color, label="dend")
        if self.inhibit_dend and compartment in ["all", "inhibit"]:
            color = inhibit_color or self.DendriteInhibition.color
            sub_ax.plot([], [], color=color, label="inhib.")
        if plot_lateral and self.mutual_inhibition_weight is not None:
            color = lateral_color or self.LateralInhibition.color
            sub_ax.plot([], [], color=color, label="lat. inh.")

        sub_ax.legend(loc=loc)

    def plot_rate_map(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        ax: plt.Axes | np.ndarray | None = None,
        compartment: str | None = None,
        norm_by: str | None = None,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | plt.Axes:
        """
        self.plot_rate_map()

        Plot the rate map of the specified compartments, overlayed, with one subplot
        per two-compartment neuron.

        Args:
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time. Default is None.
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.
        - compartment (str, optional): Which compartment to plot, if environment is
            2D ("soma", "dend", "both", "inhibit", "all"). Default is None
            (i.e., "soma" if environment is 2D, and "both" otherwise).
        - norm_by (str, optional): Normalisation method for rate maps.
            Default is "shared_fr_max".
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

        if norm_by is None and compartment in ["both", "all"]:
            norm_by = "shared_fr_max"

        if norm_by == "shared_fr_max":
            kwargs["norm_by"] = self.get_min_max_firingrates(
                t_start=t_start, t_end=t_end, compartment=compartment
            )[1]
        elif norm_by is not None:
            kwargs["norm_by"] = norm_by

        if self.Agent.Environment.dimensionality == "2D" and compartment == "all":
            warnings.warn(
                "Plotting rate maps for all compartments in a 2D environment will "
                "result in only the soma compartment appearing."
            )

        for comp in self.get_compartments(compartment)[::-1]:
            ax_out = comp.plot_rate_map(
                t_start=t_start,
                t_end=t_end,
                ax=ax,
                no_legend=True,
                autosave=autosave,
                **kwargs,
            )
            ax = ax or ax_out

        if not no_legend and self.Agent.Environment.dimensionality == "1D":
            self.add_compartment_legend(ax, compartment=compartment, loc="lower right")

        fig = np.asarray(ax).ravel()[0].figure
        plot_util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

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

        for comp in self.get_compartments(compartment)[::-1]:
            axes_out = comp.plot_rate_maps_across_learning(
                axes=axes,
                autosave=False,
                no_legend=True,
                **kwargs,
            )
            axes = axes or axes_out

        if not no_legend and self.Agent.Environment.dimensionality == "1D":
            sub_ax = np.asarray(axes).ravel()[0]
            self.add_compartment_legend(
                sub_ax, compartment=compartment, loc="lower right"
            )

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

        y = 0.9 if self.Agent.Environment.dimensionality == 1 else 0.97
        fig.suptitle(title, y=y)

        plot_util.save_figure(fig, f"{self.name}_rate_maps_across_learning", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        ax: plt.Axes | np.ndarray | None = None,
        soma_color: str | None = None,
        dend_color: str | None = None,
        inhibit_color: str | None = None,
        lateral_color: str | None = None,
        separate_axes: bool = False,
        plot_lateral: bool = False,
        norm_by: str | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
        """
        self.plot_rate_timeseries()

        Plot a timeseries of the firing rate of the specified compartments, either
        overlayed or split across subplots.

        Args:
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time. Default is None.
        - ax (1D np.ndarray or plt.Axes, optional): Subplot or 1D array of subplots
            if separate_axes (one per compartment). Default is None.
        - soma_color (str, optional): Color for soma compartment. Default is None.
        - dend_color (str, optional): Color for dendrite compartment. Default is None.
        - inhibit_color (str, optional): Color for inhibitory compartment.
            Default is None.
        - lateral_color (str, optional): Color for lateral inhibition compartment.
            Default is None.
        - separate_axes (bool, optional): Whether to plot each compartment on a
            separate subplot. Default is False.
        - plot_lateral (bool, optional): Whether to plot the lateral inhibition layer,
            if it exists. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to NMDALayer.plot_rate_timeseries().

        Returns:
        - ax (2D np.ndarray or plt.Axes): Subplot or 1D array of subplots
            if separate_axes (one per compartment).
        """

        compartments = self.get_compartments("all", incl_lateral=plot_lateral)

        if separate_axes:
            num_rows = len(compartments)
            norm_by = norm_by or "max"
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
            if "sub_ax" in kwargs.keys() and ax is None:
                ax = kwargs.pop("sub_ax")
            sub_ax = ax
            norm_by = norm_by or "shared_max"

        if norm_by == "shared_max":
            norm_by = self.get_min_max_firingrates(
                t_start=t_start, t_end=t_end, incl_lateral=plot_lateral
            )[1]

        colors = [soma_color, dend_color]
        separate_titles = ["Soma compartment", "Dendrite compartment"]
        if self.inhibit_dend:  # type: ignore[attr-defined]
            colors.append(inhibit_color)
            separate_titles.append("Inhibitory interneuron")
        if plot_lateral and self.mutual_inhibition_weight is not None:
            colors.append(lateral_color)
            separate_titles.append("Lateral inhibitor")

        if len(compartments) != len(colors):
            raise NotImplementedError(
                "Number of compartments does not match number of colors."
            )

        for c, comp in enumerate(compartments):
            color = colors[c] or comp.color
            use_sub_ax = ax1D[c] if separate_axes else sub_ax
            sub_ax_out = comp.plot_rate_timeseries(
                t_start=t_start,
                t_end=t_end,
                sub_ax=use_sub_ax,
                color=color,
                norm_by=norm_by,
                autosave=False,
                **kwargs,
            )
            if not separate_axes:
                sub_ax = sub_ax or sub_ax_out

        if separate_axes:
            for s, sub_ax in enumerate(ax1D):
                sub_ax.set_title(separate_titles[s])
                plot_fcts.mark_target_and_reset_points(self.Agent, self, sub_ax=sub_ax)
                if s != len(ax1D) - 1:
                    sub_ax.set_xlabel("")
            fig = np.asarray(ax).ravel()[0].figure
        else:
            self.add_compartment_legend(
                sub_ax,
                compartment="all",
                plot_lateral=plot_lateral,
                soma_color=soma_color,
                dend_color=dend_color,
                inhibit_color=inhibit_color,
                lateral_color=lateral_color,
            )
            fig = sub_ax.figure

        plot_util.save_figure(fig, f"{self.name}_firingrate", save=autosave)  # type: ignore[attr-defined]

        return ax

    def plot_distances_to_target(
        self,
        neuron_num=0,
        target_src_name="Obj",
        sub_ax=None,
        mark_soma_BTSP=True,
        mark_teleport=True,
        mark_closest=True,
        min_dist=0.2,
        min_steps_btw=20,
        log_num_closest=False,
    ):
        """
        self.plot_distances_to_target()

        Plot the distances from the agent's current position to the place cell centre
        of the main input to the neuron's dendrite, over time.

        Args:
        - neuron_num (int, optional): Neuron number. Default is 0.
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - mark_soma_BTSP (bool, optional): Whether to mark the soma compartment BTSP
            points. Default is True.
        - mark_teleport (bool, optional): Whether to mark the teleport points.
            Default is True.
        - mark_closest (bool, optional): Whether to mark the closest points to the
            target. Default is True.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.2.
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 20.
        - log_num_closest (bool, optional): Whether to print the number of closest
            steps identified. Default is False.

        Returns:
        - sub_ax (plt.Axes): Subplot with distances plotted.
        """

        time_min = np.asarray(self.Agent.history["t"]) / 60
        distances = self.get_distances_to_place_cell_centre_of_main_dendrite_input(
            neuron_num, src_name=target_src_name
        )

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=[8, 1.3])

        sub_ax.plot(time_min, distances)

        if mark_soma_BTSP:
            self.SomaCompartment.add_BTSP_markers_to_plots(
                sub_ax, chosen_neurons=[neuron_num], timeseries=True
            )
            plot_util.pad_axis(sub_ax, end="high")

        if mark_teleport:
            plot_util.pad_axis(sub_ax, end="high", pad_prop=0.1)
            self.Agent.add_teleportation_markers_to_plots(sub_ax, timeseries=True)

            legend = sub_ax.get_legend()
            if legend is not None:
                sub_ax.legend(loc="upper right", fontsize=5)

        if mark_closest or log_num_closest:
            closest_steps = self.get_closest_steps_to_target(
                neuron_num=neuron_num,
                target_src_name=target_src_name,
                min_dist=min_dist,
                min_steps_btw=min_steps_btw,
                log=log_num_closest,
            )
            closest_steps = gen_util.get_minima_indices(
                distances, minimum=min_dist, min_pts_btw=min_steps_btw
            )

            if mark_closest and len(closest_steps):
                plot_util.pad_axis(sub_ax, axis="y", end="both", pad_prop=0.2)
                sub_ax.plot(
                    time_min[closest_steps],
                    np.zeros_like(closest_steps),
                    lw=0,
                    marker="o",
                    ms=2,
                    color=self.SomaCompartment.color,
                )

        sub_ax.spines[["right", "top"]].set_visible(False)
        sub_ax.set_ylabel("Dist. to target")
        sub_ax.set_xlabel("Time / min.")

        return sub_ax

    def plot_distances_to_targets(
        self,
        target_src_name="Obj",
        num_neurons="all",
        mark_soma_BTSP=True,
        mark_teleport=True,
        mark_closest=True,
        min_dist=0.2,
        min_steps_btw=20,
        axes=None,
        num_cols=2,
        sharey=True,
        log_num_closest=False,
    ):
        """
        self.plot_distances_to_targets()

        Plot the distances from the agent's current position to the place cell centre
        of the main input to the neuron's dendrite, over time.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - axes (2D np.ndarray): Array of subplots to plot on (one per neuron).
            Default is None.
        - neuron_num (int, optional): Neuron number. Default is 0.
        - mark_soma_BTSP (bool, optional): Whether to mark the soma compartment BTSP
            points. Default is True.
        - mark_teleport (bool, optional): Whether to mark the teleport points.
            Default is True.
        - mark_closest (bool, optional): Whether to mark the closest points to the
            target. Default is True.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.2.
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 20.
        - num_cols (int, optional): Number of columns in the subplot array.
            Default is 2.
        - sharey (bool, optional): Whether to share the y-axis across subplots.
            Default is True.
        - log_num_closest (bool, optional): Whether to print the number of closest
            steps identified. Default is False.

        Returns:
        - axes (2D np.ndarray): Array of subplots with distances plotted.
        """

        if num_neurons == "all":
            num_neurons = self.n
        elif num_neurons > self.n:
            raise ValueError(
                f"num_neurons ({num_neurons}) must be less than or equal to the number "
                "of neurons in the layer."
            )

        if axes is None:
            num_cols = min(num_neurons, num_cols)
            num_rows = int(np.ceil(num_neurons / num_cols))
            _, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=[8, 0.8 * num_rows],
                sharex=True,
                squeeze=False,
            )
        elif len(axes.ravel()) != num_neurons:
            raise ValueError(
                f"Number of subplots ({len(axes.ravel())}) must match number of "
                f"neurons specified ({num_neurons})."
            )

        ax2D = axes.reshape(len(axes), -1)

        i = 0
        for r, ax_row in enumerate(ax2D):
            for c, sub_ax in enumerate(ax_row):
                if i < num_neurons:
                    use_mark_teleport = False
                    if r == 0:
                        use_mark_teleport = mark_teleport

                    self.plot_distances_to_target(
                        neuron_num=i,
                        target_src_name=target_src_name,
                        sub_ax=sub_ax,
                        mark_soma_BTSP=mark_soma_BTSP,
                        mark_teleport=use_mark_teleport,
                        mark_closest=mark_closest,
                        min_dist=min_dist,
                        min_steps_btw=min_steps_btw,
                        log_num_closest=log_num_closest,
                    )
                    if use_mark_teleport and c != len(ax2D[0]) - 1:
                        legend = sub_ax.get_legend()
                        if legend is not None:
                            legend.remove()
                else:
                    sub_ax.spines[["left", "bottom"]].set_visible(False)
                    sub_ax.set_xticks([])
                    sub_ax.set_yticks([])

                if c == 0 and r == len(ax2D) // 2:
                    sub_ax.set_ylabel("Distance to target")
                else:
                    sub_ax.set_ylabel("")
                if r == len(ax2D) - 1:
                    sub_ax.set_xlabel("Time / min.")
                else:
                    sub_ax.set_xlabel("")

                i += 1

        if sharey:
            y_lims = np.asarray([sub_ax.get_ylim() for sub_ax in axes.ravel()]).T
            y_lims = [np.min(y_lims[0]), np.max(y_lims[1])]
            for sub_ax in axes.ravel():
                sub_ax.set_ylim(y_lims)

        neuron_str = "all" if num_neurons == self.n else f"first {num_neurons}"

        y = 0.885 + (0.005 * ax2D.shape[1])
        sub_ax.figure.suptitle(f"Distance to target for {neuron_str} neurons.", y=y)

        return axes

    def plot_neuron_properties_at_BTSP_and_closest_to_target_steps(
        self,
        neuron_num=0,
        target_src_name="Obj",
        t_start=None,
        t_end=None,
        axes=None,
        k=5,
        legend=True,
    ):
        """
        self.plot_neuron_properties_at_BTSP_and_closest_to_target_steps()

        Plot properties (step number, firing rate, velocity and angle near target)
        at BTSP and closest to target steps for a neuron.

        Args:
        - neuron_num (int, optional): Neuron number. Default is 0.
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - t_start (int, optional): Start time for plotting. Default is None.
        - t_end (int, optional): End time for plotting. Default is None.
        - axes (2D np.ndarray, optional): Array of subplots to plot on. Default is None.
        - k (bool, optional): Number of points across which to smooth the firing rate,
        velocity and angle data backward, before the target point. If None, no
        smoothing is done. Default is 5.
        - legend (bool, optional): Whether to include a legend in the plots.
            Default is True.

        Returns:
        - axes (2D np.ndarray): Array of subplots with properties plotted.
        """

        if axes is None:
            _, axes = plt.subplots(1, 4, figsize=[8, 2], sharey=True, squeeze=False)
        elif len(axes.ravel()) != 4:
            raise ValueError("axes must have length 4.")

        distances = self.get_distances_to_place_cell_centre_of_main_dendrite_input(
            neuron_num, src_name=target_src_name
        )
        firingrates = np.asarray(self.SomaCompartment.history["firingrate"]).T[
            neuron_num
        ]

        velocities = np.sqrt(np.sum(np.asarray(self.Agent.history["vel"]) ** 2, axis=1))
        angles = self.get_vectors_to_place_cell_centre_of_main_dendrite_input(
            neuron_num, src_name=target_src_name, polar=True
        )[:, 1]

        steps_dict = self.match_closest_to_target_steps_to_BTSP_steps(
            target_src_name="Obj", neuron_num=neuron_num, t_start=t_start, t_end=t_end
        )

        for i, (x_data_type, x_data, sub_ax) in enumerate(
            zip(
                ["Step", "Firing rate", "Velocity near target", "Angle near target"],
                [None, firingrates, velocities, angles],
                axes.ravel(),
            )
        ):
            y_data_type = "Distance from target" if i == 0 else ""

            if k is not None and x_data_type != "Step":
                smoothed = np.convolve(x_data, np.full(k, 1 / k), mode="valid")
                for j in range(len(x_data) - len(smoothed)):
                    smoothed = np.insert(smoothed, j, x_data[: j + 1].mean())
                x_data = smoothed

            incl_legend = i == (len(axes.ravel()) - 1) and legend
            plot_fcts.plot_property_at_BTSP_and_closest_to_target_steps(
                steps_dict,
                y_data=distances,
                x_data=x_data,
                x_data_type=x_data_type,
                y_data_type=y_data_type,
                sub_ax=sub_ax,
                legend=incl_legend,
            )

            if x_data_type == "Step":
                for t in [t_start, t_end]:
                    if t is not None:
                        step = t / self.Agent.dt
                        sub_ax.axvline(step, color="k", ls="dashed", lw=1)

        sub_ax.figure.suptitle(
            f"Properties near BTSP and closest to target steps (#{neuron_num})"
        )

        return axes

    def plot_properties_at_BTSP_and_closest_to_target_steps(
        self,
        target_src_name="Obj",
        sort_by_num_BTSP=False,
        t_start=None,
        t_end=None,
        k=5,
        legend=True,
    ):
        """
        self.plot_properties_at_BTSP_and_closest_to_target_steps()

        Plot properties at BTSP and closest to target steps for all neurons.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - sort_by_num_BTSP (bool, optional): If True, neurons are sorted by number of
            BTSP events. Default is False.
        - t_start (int, optional): Start time for plotting. Default is None.
        - t_end (int, optional): End time for plotting. Default is None.
        - k (bool, optional): Number of points across which to smooth the firing rate,
        velocity and angle data backward, before the target point. If None, no
        smoothing is done. Default is 5.
        - legend (bool, optional): Whether to include a legend in the plots.
            Default is True.

        Returns:
        - axes (2D np.ndarray): Array of subplots with properties plotted.
        """

        n_row = self.n
        n_col = 4
        figsize = (n_col * 2, n_row * 1.5)
        fig, axes = plt.subplots(
            n_row, n_col, figsize=figsize, sharex=False, sharey="row", squeeze=False
        )

        sorter = np.arange(self.n)
        if sort_by_num_BTSP:
            num_BTSP = [
                len(self.SomaCompartment.get_BTSP_step_dict()[i]) for i in sorter
            ]
            sorter = np.argsort(num_BTSP)

        for i, neuron_num in enumerate(sorter):
            self.plot_neuron_properties_at_BTSP_and_closest_to_target_steps(
                neuron_num=neuron_num,
                target_src_name=target_src_name,
                t_start=t_start,
                t_end=t_end,
                axes=axes[i],
                k=k,
                legend=legend,
            )
            title = f"Neuron {neuron_num}"
            if sort_by_num_BTSP:
                title = f"{title} ({num_BTSP[neuron_num]} BTSP events)"
            axes[i, 0].set_title(title)

        fig.suptitle("Properties near BTSP and closest to target steps", y=0.885)

        # adjust x limits to match within each column
        for i in range(axes.shape[1]):
            x_lims = np.asarray([sub_ax.get_xlim() for sub_ax in axes[:, i]]).T
            x_lims = [np.min(x_lims[0]), np.max(x_lims[1])]
            for r, sub_ax in enumerate(axes[:, i]):
                sub_ax.set_xlim(*x_lims)
                if r != self.n - 1:
                    sub_ax.set_xlabel("")
                    sub_ax.xaxis.set_tick_params(labelbottom=False)

        # adjust y label
        for sub_ax in axes[:, 0]:
            sub_ax.set_ylabel("Dist. from target")

        return axes
