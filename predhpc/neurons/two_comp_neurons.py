import copy
from typing import TYPE_CHECKING, Any, Sequence

import warnings

from matplotlib import pyplot as plt
import numpy as np

from predhpc import plot_fcts
from predhpc.neurons import learning_neurons, riab_neurons
from predhpc.util import gen_util, trig_util, plot_util, params_util

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


class TwoCompLayer(object):
    """
    TwoCompLayer()

    This neuron layer class defines a population of neurons with two compartments
    (proximal and distal), each of which is an NMDALayer. An additional HebbianLayer
    distal inhibition compartment can optionally be included

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 2,
        "name": "TwoCompLayer",
        "proximal_input_layers": list(),
        "distal_input_layers": list(),
        "proximal_to_distal_weight": 0.2,
        "distal_to_proximal_weight": 1.0,
        "distal_first": True,
        "proximal_color": "C0",
        "distal_color": "C1",
        "inhibitory_distal": True,
        "inhibitory_color": "k",
        "inhibitory_weight": 3.0,  # multiplied by -1 identity matrix
        "inhibitory_activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_PARAMS,
        "inhibitory_input_filter_tau": 3,
        "inhibitory_input_trend_tau": None,
        "lateral_inhibition_weight": None,
        "lateral_tau": 0.1,
    }

    List of methods:
        • self.set_learn()
        • self.set_BTSP_learn()
        • self.get_compartments()
        • self.get_min_max_firingrates()
        • self.get_main_distal_input_layer()
        • self.get_index_of_main_distal_input()
        • self.get_place_cell_center_of_main_distal_input()
        • self.get_vectors_to_place_cell_center_of_main_distal_input()
        • self.get_distances_to_place_cell_center_of_main_distal_input()
        • self.get_target_visits()
        • self.get_closest_steps_to_target()
        • self.get_nbr_visits_per_target()
        • self.match_closest_steps_to_BTSP_events()
        • self.update()
        • self.add_compartment_legend()
        • self.plot_rate_map()
        • self.plot_rate_maps_across_learning()
        • self.plot_rate_timeseries()
        • plot_binned_rates()
        • self.plot_distance_to_distal_target()
        • self.plot_distance_to_distal_targets()
        • plot_neuron_properties_at_BTSP_and_closest_to_target_steps()
        • plot_properties_at_BTSP_and_closest_to_target_steps()
        • plot_BTSP_counts_vs_target_visits()
    """

    default_params = {
        "n": 2,
        "name": "TwoCompLayer",
        "proximal_input_layers": list(),
        "distal_input_layers": list(),
        "proximal_to_distal_weight": 0.2,
        "distal_to_proximal_weight": 1.0,
        "distal_first": True,
        "proximal_color": "C0",
        "distal_color": "C1",
        "inhibitory_distal": True,
        "inhibitory_color": "k",
        "inhibitory_weight": 3.0,  # multiplied by -1 identity matrix
        "inhibitory_activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_PARAMS,
        "inhibitory_input_filter_tau": 3,
        "inhibitory_input_trend_tau": None,
        "lateral_inhibition_weight": None,
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
        self.Environment = Agent.Environment

        self._organize_params(params)
        self._create_compartments()

        self.set_learn(proximal=True, distal=True, inhibitory=False)

    def _organize_params(self, params: dict[str, Any]):
        """
        self._organize_params(params)

        Organise the parameters passed to the TwoCompLayer class, passing them to each
        compartment as appropriate.

        Attributes:
        - distal_params (dict): Parameters for the distal compartment.
        - name (str): Name of the layer.
        - proximal_params (dict): Parameters for the proximal compartment.

        Args:
        - params (dict): Parameters passed to the TwoCompLayer class.
        """

        self.proximal_params = {"name": "proximal"}
        self.distal_params = dict()

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
            "proximal_to_distal_weight",
            "distal_to_proximal_weight",
            "distal_first",
            "lateral_inhibition_weight",
            "lateral_tau",
        ]

        for key, value in all_params.items():
            if key in self.ignored_param_keys:
                warnings.warn(
                    f"'{key}' should not be provided for {cls.__name__}. "  # type: ignore[name-defined]
                    "Will be ignored."
                )

            elif key in local_attributes or key.startswith("inhibitory_"):
                setattr(self, key, value)

            elif key == "name":
                self.name = value
                if "distal_name" not in all_params.keys():
                    self.distal_params["name"] = f"{value}_distal"
                if "proximal_name" not in all_params.keys():
                    self.proximal_params["name"] = f"{value}_proximal"

            elif key == "n":
                self.n = value
                self.proximal_params["n"] = value
                self.distal_params["n"] = value

            elif key.startswith("proximal_") or key.startswith("distal_"):
                for compartment, comp_dict in [
                    ("proximal", self.proximal_params),
                    ("distal", self.distal_params),
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
                self.proximal_params[key] = value
                self.distal_params[key] = value

    def _create_compartments(self):
        """
        self._create_compartments()

        Create the proximal and distal compartments, and connect them to each other for
        each neuron.

        If applicable, an inhibitory compartment is also created for each neuron and
        connected to the neuron's proximal and distal compartments.

        Inter-compartment connections:
            Proximal <--> Distal
            if self.inhibitory_distal:
                Proximal -->* Distal inhibition --> Distal
            *: learning possible

        Attributes:
        - DistalCompartment (learning_neurons.NMDALayer): Distal compartment.
        - DistalInhibition (learning_neurons.HebbianLayer): Inhibitory compartment.
        - ProximalCompartment (learning_neurons.NMDALayer): Proximal compartment.
        """

        self.ProximalCompartment = learning_neurons.NMDALayer(
            self.Agent, self.proximal_params
        )
        self.DistalCompartment = learning_neurons.NMDALayer(
            self.Agent, self.distal_params
        )

        if self.ProximalCompartment.n != self.n or self.DistalCompartment.n != self.n:  # type: ignore[attr-defined]
            raise ValueError(
                f"The two compartment layers must have same number of units ({self.n})."
            )

        distal_to_proximal_weight = np.eye(self.n) * self.distal_to_proximal_weight  # type: ignore[attr-defined]
        self.ProximalCompartment.add_input_layers_with_no_learning(self.DistalCompartment.name)  # type: ignore[attr-defined]
        self.ProximalCompartment.add_input(
            self.DistalCompartment,
            w=distal_to_proximal_weight,
            recurrent=not (self.distal_first),
        )

        proximal_to_distal_weight = np.eye(self.n) * self.proximal_to_distal_weight  # type: ignore[attr-defined]
        self.DistalCompartment.add_input_layers_with_no_learning(
            self.ProximalCompartment.name  # type: ignore[attr-defined]
        )
        self.DistalCompartment.add_input(
            self.ProximalCompartment,
            w=proximal_to_distal_weight,
            recurrent=self.distal_first,
        )

        if self.inhibitory_distal:  # type: ignore[attr-defined]
            inhibitory_params = {
                "name": "ProximalInhibitionOfDistal",
                "n": self.n,
                "activation_function": self.inhibitory_activation_function,  # type: ignore[attr-defined]
                "color": self.inhibitory_color,  # type: ignore[attr-defined]
                "input_filter_tau": self.inhibitory_input_filter_tau,  # type: ignore[attr-defined]
                "input_trend_tau": self.inhibitory_input_trend_tau,  # type: ignore[attr-defined]
            }

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "No input layers", UserWarning)
                self.DistalInhibition = learning_neurons.HebbianLayer(
                    self.Agent, params=inhibitory_params
                )

            proximal_input = np.eye(self.n) * self.inhibitory_weight  # type: ignore[attr-defined]
            self.DistalInhibition.add_input(self.ProximalCompartment, w=proximal_input)

            distal_inhibition = np.eye(self.n) * -1
            self.DistalCompartment.add_input_layers_with_no_learning(
                self.DistalInhibition.name  # type: ignore[attr-defined]
            )
            self.DistalCompartment.add_input(
                self.DistalInhibition, w=distal_inhibition, recurrent=self.distal_first
            )

        if self.lateral_inhibition_weight is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "No input layers", UserWarning)

                lateral_params = {
                    "name": "LateralInhibition",
                    "n": self.n,
                    "activation_function": self.ProximalCompartment.activation_function,
                    "color": "gray",
                    "input_filter_tau": self.lateral_tau,
                }

                self.LateralInhibition = learning_neurons.SmoothFeedForwardLayer(
                    self.Agent, params=lateral_params
                )

            self.LateralInhibition.add_input(self.ProximalCompartment, w=np.eye(self.n))
            lateral_inhibition = (np.eye(self.n) - 1) * self.lateral_inhibition_weight
            self.ProximalCompartment.add_input(
                self.LateralInhibition, w=lateral_inhibition
            )
            self.ProximalCompartment.add_input_layers_with_no_learning(
                self.LateralInhibition.name
            )

    def set_learn(self, learn=None, proximal=None, distal=None, inhibitory=None):
        """
        self.set_learn()

        Set, for each compartment, whether it should learn during self.update() calls.
        Only affects input weights that are learnable.

        Args:
        - learn (bool, optional): Whether to learn learnable weights into all
            compartments.  Default is None.
        - proximal (bool, optional): Whether to learn learnable weights into the
            proximal compartment.  Default is None.
        - distal (bool, optional): Whether to learn learnable weights into the
            distal compartment. Default is None.
        - inhibitory (bool, optional): Whether to learn learnable weights into the
            inhibitory compartment. Default is None.
        """

        if learn is not None:
            proximal = learn if proximal is None else proximal
            distal = learn if distal is None else distal
            inhibitory = learn if inhibitory is None else inhibitory

        self.ProximalCompartment.set_learn(proximal)
        self.DistalCompartment.set_learn(distal)
        if self.inhibitory_distal:  # type: ignore[attr-defined]
            self.DistalInhibition.set_learn(inhibitory)

    def set_BTSP_learn(self, learn=None, proximal=None, distal=None):
        """
        self.set_BTSP_learn()

        Set whether the proximal and distal compartments should learn using BTSP during
        self.update() calls. Only affects input weights that are learnable.

        Args:
        - learn (bool, optional): Whether to learn learnable weights into both
            compartments. Default is None
        - proximal (bool, optional): Whether to learn learnable weights into the
            proximal compartment.  Default is None.
        - distal (bool, optional): Whether to learn learnable weights into the
            distal compartment. Default is None.
        """

        if learn is not None:
            proximal = learn if proximal is None else proximal
            distal = learn if distal is None else distal

        self.ProximalCompartment.set_BTSP_learn(proximal)
        self.DistalCompartment.set_BTSP_learn(distal)

    def get_compartments(
        self,
        compartment: str = "all",
        incl_lateral: bool = False,
    ):
        """
        self.get_compartments()

        - compartment (str, optional): Which compartments to retrieve
            ("proximal", "distal", "both", "inhibitory", "all"). Default is "all".

        Returns:
        - compartments (list): List of compartments.
        """

        if compartment not in ["proximal", "distal", "both", "inhibitory", "all"]:
            raise ValueError(
                "compartment must be 'proximal', 'distal', 'both', 'inhibitory' or 'all', "
                f"not '{compartment}'."
            )

        compartments = list()
        if compartment in ["proximal", "both", "all"]:
            compartments.append(self.ProximalCompartment)
        if compartment in ["distal", "both", "all"]:
            compartments.append(self.DistalCompartment)
        if compartment in ["inhibitory", "all"]:
            if self.inhibitory_distal:
                compartments.append(self.DistalInhibition)
            elif compartment == "inhibitory":  # type: ignore[attr-defined]
                raise ValueError(
                    "Cannot retrieve inhibition compartment, as inhibition is not enabled."
                )
        if incl_lateral:
            if self.lateral_inhibition_weight is None:
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
        chosen_neurons: str | int | list | np.ndarray = "all",
        compartment: str = "all",
        incl_lateral: bool = False,
    ):
        """
        self.get_min_max_firingrates()

        Obtain the minimum and maximum firing rates of the layer.

        Args:
        - t_start (float, optional): Start time for obtaining firingrate min and max.
            Default is None.
        - t_end (float, optional): End time for obtaining firingrate min and max.
            Default is None.
        - chosen_neurons (str, int, list or np.ndarray, optional): Neurons to consider
            for min and max firing rates. Default is "all".
        - compartment (str, optional): Which compartment to obtain max for
            ("proximal", "distal", "both", "inhibitory", "all"). Default is "all".
        - incl_lateral (bool, optional): Whether to include the lateral inhibition
            compartment. Default is False.

        Returns:
        - min_firingrate (float): Minimum firing rate.
        - max_firingrate (float): Maximum firing rate.
        """

        min_firingrate = np.inf
        max_firingrate = -np.inf

        compartments = self.get_compartments(compartment, incl_lateral=incl_lateral)
        for comp in compartments:
            min_rate, max_rate = comp.get_min_max_firingrates(
                t_start=t_start, t_end=t_end, chosen_neurons=chosen_neurons
            )
            min_firingrate = min(min_firingrate, min_rate)
            max_firingrate = max(max_firingrate, max_rate)

        return min_firingrate, max_firingrate

    def get_main_distal_input_layer(
        self, src_name: str = "Obj", return_dict: bool = False
    ):
        """
        self.get_main_distal_input_layer()

        Get the main input layer to the distal compartment.

        Args:
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - return_dict (bool, optional): Whether to return the full input dict instead
            of only the layer. Default is False.

        Returns:
        if return_dict:
        - input_dict (dict): Full input dictionary for the main distal input layer.
        else:
        - main_distal_input_layer (riab_neurons.PlaceCells): Main distal input layer.
        """

        if src_name not in self.DistalCompartment.inputs.keys():
            raise ValueError(f"No '{src_name}' input to distal compartment.")

        main_distal_input_dict = self.DistalCompartment.inputs[src_name]

        if not gen_util.attribute_type_checker(
            main_distal_input_dict["layer"], "PlaceCells"
        ):
            raise ValueError(f"Input layer '{src_name}' is not a PlaceCells layer.")

        if return_dict:
            return main_distal_input_dict

        else:
            main_distal_input_layer = main_distal_input_dict["layer"]

            return main_distal_input_layer

    def get_index_of_main_distal_input(
        self, neuron_idx: int = 0, src_name: str = "Obj"
    ):
        """
        self.get_index_of_main_distal_input()

        Get the index of the main input to the distal compartment of a specified neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".

        Returns:
        - input_idx (int): Index of main distal input.
        """

        main_distal_input_dict = self.get_main_distal_input_layer(
            src_name=src_name, return_dict=True
        )

        if neuron_idx > self.n:
            raise ValueError(
                f"Neuron index {neuron_idx} is greater than the number of neurons "
                "in the layer."
            )

        input_idx = np.argmax(main_distal_input_dict["w"][:, neuron_idx])

        return input_idx

    def get_place_cell_center_of_main_distal_input(
        self, neuron_idx: int = 0, src_name: str = "Obj"
    ):
        """
        self.get_place_cell_center_of_main_distal_input()

        Get the place cell center input location for the distal compartment of a
        specified neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".

        Returns:
        - place_cell_center (1D np.ndarray): Main distal input place cell center
            location.
        """

        main_distal_input_layer = self.get_main_distal_input_layer(src_name=src_name)

        input_idx = self.get_index_of_main_distal_input(
            neuron_idx=neuron_idx, src_name=src_name
        )

        place_cell_center = main_distal_input_layer.place_cell_centers[input_idx]

        return place_cell_center

    def get_vectors_to_place_cell_center_of_main_distal_input(
        self,
        neuron_idx: int = 0,
        src_name: str = "Obj",
        polar: bool = False,
        radians: bool = False,
    ):
        """
        self.get_vectors_to_place_cell_center_of_main_distal_input()

        Get the vectors from the agent's current position to the place cell center
        input location for the distal compartment of a specified neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - polar (bool, optional): Whether to return vectors in polar coordinates.
            Default is False.
        - radians (bool, optional): If True and polar is True, return angles in radians.
            Default is False.

        Returns:
        - vectors (2D np.ndarray): Vectors from agent's position to distal input
            place cell center.
        """

        place_cell_center = self.get_place_cell_center_of_main_distal_input(
            neuron_idx=neuron_idx, src_name=src_name
        )
        pos = np.asarray(self.Agent.history["pos"])

        vectors = trig_util.get_vectors_to_target(
            pos, target=place_cell_center, polar=polar, radians=radians
        )

        return vectors

    def get_distances_to_place_cell_center_of_main_distal_input(
        self, neuron_idx: int = 0, src_name: str = "Obj"
    ):
        """
        self.get_distances_to_place_cell_center_of_main_distal_input()

        Get the distances from the agent's current position to the place cell center
        input location for the distal compartment of a specified neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".

        Returns:
        - distances (1D np.ndarray): Distances from agent's position to distal input
            place cell center.
        """

        vectors = self.get_vectors_to_place_cell_center_of_main_distal_input(
            neuron_idx, src_name
        )

        distances = np.linalg.norm(vectors, ord=2, axis=1)

        return distances

    def get_target_visits(
        self,
        neuron_idx: int = 0,
        target_src_name: str = "Obj",
        min_steps_btw=30,
        min_dist=0.05,
    ):
        """
        self.get_target_visits()

        Get the indices of the steps where the agent is closest to the target specified
        by the place cell center of the main input to the neuron's distal compartment.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 30.
        - min_dist (float, optional): Minimum distance to be considered a visit.
            Default is 0.05.

        Returns:
        - visit_indices (1D np.ndarray): Indices of the steps where the agent is
            closest to the target.
        """

        distances = self.get_distances_to_place_cell_center_of_main_distal_input(
            neuron_idx=neuron_idx, src_name=target_src_name
        )

        visit_indices = gen_util.get_minima_indices(
            distances, min_pts_btw=min_steps_btw, minimum=min_dist
        )

        return visit_indices

    def get_closest_steps_to_target(
        self,
        neuron_idx=0,
        target_src_name="Obj",
        min_dist=0.1,
        min_steps_btw=20,
        log=False,
    ):
        """
        self.get_closest_steps_to_target()

        Get the steps where the agent is closest to the target specified by the place
        cell center of the main input to the neuron's distal compartment.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.1.
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 20.
        - log (bool, optional): Whether to print the number of closest steps identified.
            Default is False.

        Returns:
        - closest_steps (1D np.ndarray): Steps identified as locally closest to the
            target.
        """

        distances = self.get_distances_to_place_cell_center_of_main_distal_input(
            neuron_idx, src_name=target_src_name
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

    def get_nbr_visits_per_target(
        self,
        target_src_name: str = "Obj",
        min_steps_btw=30,
        min_dist=0.1,
        t_start=None,
        t_end=None,
    ):
        """
        self.get_nbr_visits_per_target()

        Get the number of visits to the target specified by the place cell center of
        the main input to the neuron's distal compartment for each neuron in the layer.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 30.
        - min_dist (float, optional): Minimum distance to be considered a visit.
            Default is 0.1.
        - t_start (int, optional): Start time for obtaining number of visits.
            Default is None.
        - t_end (int, optional): End time for obtaining number of visits.
            Default is None.

        Returns:
        - nbr_visits_per_BTSP_target (1D np.ndarray): Number of visits to the target
            specified by the place cell center of the main input to the neuron's
            distal compartment for each neuron in the layer.
        """

        _, startid, endid = self.ProximalCompartment.get_plotting_times(
            t_start=t_start, t_end=t_end
        )

        nbr_visits_per_BTSP_target = np.zeros(self.n, dtype=int)
        for neuron_idx in range(self.n):
            visit_indices = self.get_target_visits(
                neuron_idx=neuron_idx,
                target_src_name=target_src_name,
                min_steps_btw=min_steps_btw,
                min_dist=min_dist,
            )

            if len(visit_indices):
                visit_indices = visit_indices[
                    (visit_indices >= startid) & (visit_indices <= endid)
                ]
                nbr_visits_per_BTSP_target[neuron_idx] = len(visit_indices)

        return nbr_visits_per_BTSP_target

    def match_closest_to_target_steps_to_BTSP_steps(
        self,
        target_src_name="Obj",
        neuron_idx=0,
        max_step_dist=40,
        min_dist=0.1,
        t_start=None,
        t_end=None,
    ):
        """
        self.match_closest_to_target_steps_to_BTSP_steps()

        Match the steps closest to the target to the BTSP steps of the specified neuron.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - max_step_dist (int, optional): Maximum distance between steps to be considered
            a match. Default is 40.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.1.
        - t_start (int, optional): Start time for matching. Default is None.
        - t_end (int, optional): End time for matching. Default is None.

        Returns:
        - steps_dict (dict): Dictionary of steps closest to the target, matched to
            each BTSP step, with keys:
                "steps_before": steps close to target before neuron's first BTSP step
                "steps_near_BTSP": steps close to target nearest each BTSP step
                "steps_of_nearest_BTSP": BTSP steps closest to each step close to target
                "steps_other": other steps close to target after first BTSP
                "other_BTSP_steps": all other BTSP steps, whether close to target or not
        """

        if neuron_idx >= self.n:
            raise ValueError(
                f"Neuron index ({neuron_idx}) must be smaller than number of "
                f"neurons ({self.n})."
            )

        _, start, end = self.ProximalCompartment.get_plotting_times(
            t_start=t_start, t_end=t_end
        )

        BTSP_steps = self.ProximalCompartment.get_BTSP_step_dict()[neuron_idx]
        closest_steps = self.get_closest_steps_to_target(
            neuron_idx, target_src_name=target_src_name, min_dist=min_dist
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

    def update(self, distal_first: bool | None = None):
        """
        self.update()

        Update the proximal and distal compartments of the two compartment layer. If
        there is an distal inhibition compartment, it is also updated.

        Update order:
            1. Distal inhibition compartment (if applicable)
            if distal_first:
                2. Distal compartment
                3. Proximal compartment
            otherwise:
                2. Proximal compartment
                3. Distal compartment

        Args:
        - distal_first (bool, optional): Whether to update the distal compartment
            before the proximal compartment. If None, the attribute is used.
            Default is None.
        """

        if self.inhibitory_distal:  # type: ignore[attr-defined]
            self.DistalInhibition.update()

        if distal_first is None:
            distal_first = self.distal_first  # type: ignore[attr-defined]

        if distal_first:
            self.DistalCompartment.update()
            self.ProximalCompartment.update()
            if self.lateral_inhibition_weight is not None:
                self.LateralInhibition.update()
        else:
            self.ProximalCompartment.update()
            self.DistalCompartment.update()
            if self.lateral_inhibition_weight is not None:
                self.LateralInhibition.update()

        return

    def add_compartment_legend(
        self,
        sub_ax,
        compartment="all",
        plot_lateral=False,
        loc="best",
        proximal_color=None,
        distal_color=None,
        inhibitory_color=None,
        lateral_color=None,
        lw=1.0,
        label=None,
        **kwargs,
    ):
        """
        self.add_compartment_legend()

        Add a legend to a plot with the colors of the specified compartments.

        Args:
        - sub_ax (plt.Axes): Subplot to add the legend to.
        - compartment (str, optional): Which compartments to include in the legend
            ("proximal", "distal", "inhibitory", "all"). Default is "all".
        - plot_lateral (bool, optional): Whether to include the lateral inhibition
            compartment in the legend. Default is False.
        - proximal_color (str, optional): Color for proximal compartment. Default is None.
        - distal_color (str, optional): Color for distal compartment. Default is None.
        - inhibitory_color (str, optional): Color for inhibitory compartment.
            Default is None.
        - lateral_color (str, optional): Color for lateral inhibition compartment.
            Default is None.
        - lw (float, optional): Line width for the timeseries. Default is 1.0.
        - label (str, optional): Label for the legend. If None, compartment names are
            used. Default is None.

        Keyword args:
        - **kwargs: Additional keyword arguments passed to plt.legend().
        """

        if compartment not in ["proximal", "distal", "both", "inhibitory", "all"]:
            raise ValueError(
                "compartment must be 'proximal', 'distal', 'both', 'inhibitory' or "
                f"'all', not '{compartment}'."
            )

        if compartment in ["both", "all"] and label is not None:
            raise ValueError(
                "Cannot specify label when plotting more than one compartment."
            )

        if compartment in ["all", "proximal"]:
            use_label = label or "proximal"
            color = proximal_color or self.ProximalCompartment.color
            sub_ax.plot(list(), list(), color=color, lw=lw, label=use_label)
        if compartment in ["all", "distal"]:
            use_label = label or "distal"
            color = distal_color or self.DistalCompartment.color
            sub_ax.plot(list(), list(), color=color, lw=lw, label=use_label)
        if self.inhibitory_distal and compartment in ["all", "inhibitory"]:
            use_label = label or "inhibitory"
            color = inhibitory_color or self.DistalInhibition.color
            sub_ax.plot(list(), list(), color=color, lw=lw, label=use_label)
        if plot_lateral and self.lateral_inhibition_weight is not None:
            use_label = label or "lat. inh."
            color = lateral_color or self.LateralInhibition.color
            sub_ax.plot(list(), list(), color=color, lw=lw, label=use_label)

        sub_ax.legend(loc=loc, **kwargs)

    def plot_rate_map(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str | int | list | np.ndarray = "all",
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
        - chosen_neurons (str, int, list or np.ndarray, optional): Neurons to plot.
            Default is "all" (i.e., all neurons in the layer).
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.
        - compartment (str, optional): Which compartment to plot, if environment is
            2D ("proximal", "distal", "both", "inhibitory", "all"). Default is None
            (i.e., "proximal" if environment is 2D, and "both" otherwise).
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
            if self.Environment.D == 1:
                compartment = "all"
            else:
                compartment = "proximal"

        if norm_by is None and compartment in ["both", "all"]:
            norm_by = "shared_fr_max"

        if norm_by == "shared_fr_max":
            kwargs["norm_by"] = self.get_min_max_firingrates(
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                compartment=compartment,
            )[1]
        elif norm_by is not None:
            kwargs["norm_by"] = norm_by

        if self.Environment.D == 2 and compartment == "all":
            warnings.warn(
                "Plotting rate maps for all compartments in a 2D environment will "
                "result in only the proximal compartment appearing."
            )

        for comp in self.get_compartments(compartment)[::-1]:
            ax_out = comp.plot_rate_map(
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                ax=ax,
                no_legend=True,
                autosave=autosave,
                **kwargs,
            )
            ax = ax or ax_out

        if not no_legend and self.Environment.D == 1:
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
            2D ("proximal", "distal", "inhibitory" or "both"). Default is None
            (i.e., "proximal" if environment is 2D, and "both" otherwise).
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - title (str, optional): Title for the figure. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to
            NMDALayer.plot_rate_maps_across_learning().

        Raises:
        - ValueError: If compartment is not "proximal", "distal", "inhibitory" or "all".
        - ValueError: If compartment is "inhibitory", but self.inhibitory_distal is False.

        Returns:
        - axes (2D np.ndarray): Array of subplots. If input axes was None,
            shape is 2D (number of ROIs, num_maps) or v.v. if only one ROI.
        """

        if compartment is None:
            if self.Environment.D == 1:
                compartment = "all"
            else:
                compartment = "proximal"

        if self.Environment.D == 2 and compartment == "both":
            warnings.warn(
                "Plotting rate maps across learning for both compartments in a 2D "
                "environment will result in only the proximal compartment appearing."
            )

        for comp in self.get_compartments(compartment)[::-1]:
            axes_out = comp.plot_rate_maps_across_learning(
                axes=axes,
                autosave=False,
                no_legend=True,
                **kwargs,
            )
            axes = axes or axes_out

        if not no_legend and self.Environment.D == 1:
            sub_ax = np.asarray(axes).ravel()[0]
            self.add_compartment_legend(
                sub_ax, compartment=compartment, loc="lower right"
            )

        if title is None:
            if compartment == "both":
                title_start = "Rate maps"
            elif compartment == "proximal":
                title_start = "Proximal rate maps"
            elif compartment == "inhibitory":
                title_start = "Inhibition rate maps"
            else:
                title_start = "Distal rate maps"

            title = f"{title_start} across learning"

        fig = np.asarray(axes).ravel()[0].figure

        y = 0.9 if self.Environment.D == 1 else 0.97
        fig.suptitle(title, y=y)

        plot_util.save_figure(fig, f"{self.name}_rate_maps_across_learning", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str | int | list | np.ndarray = "all",
        ax: plt.Axes | np.ndarray | None = None,
        proximal_color: str | None = None,
        distal_color: str | None = None,
        inhibitory_color: str | None = None,
        lateral_color: str | None = None,
        separate_axes: bool = False,
        plot_lateral: bool = False,
        single_x_axis: bool = True,
        norm_by: str | None = None,
        in_min: bool = True,
        lw: float = 1.0,
        omit_target_reset: bool = False,
        no_legend: bool = False,
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
        - chosen_neurons (str, int, list or np.ndarray, optional): Neurons to plot.
            Default is "all".
        - ax (1D np.ndarray or plt.Axes, optional): Subplot or 1D array of subplots
            if separate_axes (one per compartment). Default is None.
        - proximal_color (str, optional): Color for proximal compartment. Default is None.
        - distal_color (str, optional): Color for distal compartment. Default is None.
        - inhibitory_color (str, optional): Color for inhibitory compartment.
            Default is None.
        - lateral_color (str, optional): Color for lateral inhibition compartment.
            Default is None.
        - separate_axes (bool, optional): Whether to plot each compartment on a
            separate subplot. Default is False.
        - plot_lateral (bool, optional): Whether to plot the lateral inhibition layer,
            if it exists. Default is False.
        - single_x_axis (bool, optional): Whether to plot x axis spine and ticks only
            for the last subplot, instead of all of them. Default is True.
        - norm_by (str, optional): Normalisation method for rate maps. Default is None.
        - in_min (bool, optional): Whether to plot the time in minutes instead of
            seconds. Default is True.
        - lw (float, optional): Line width for the timeseries. Default is 1.0.
        - omit_target_reset (bool, optional): Whether to omit marking the target and
            reset points on the plot. Default is False.
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to NMDALayer.plot_rate_timeseries().

        Returns:
        - ax (2D np.ndarray or plt.Axes): Subplot or 1D array of subplots
            if separate_axes (one per compartment).
        """

        compartments = self.get_compartments("all", incl_lateral=plot_lateral)

        if "linewidth" in kwargs.keys():
            lw = kwargs.pop("linewidth")

        if "imshow" in kwargs.keys() and kwargs["imshow"]:
            raise NotImplementedError("'imshow' option is not implemented.")

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
                        f"ax must have shape ({num_rows}, ) or ({num_rows}, 1), but "
                        f"found {ax.shape}."
                    )

            ax1D = np.asarray(ax).ravel()
        else:
            if "sub_ax" in kwargs.keys() and ax is None:
                ax = kwargs.pop("sub_ax")
            sub_ax = ax
            norm_by = norm_by or "shared_max"

        if norm_by == "shared_max":
            norm_by = self.get_min_max_firingrates(
                t_start=t_start,
                t_end=t_end,
                incl_lateral=plot_lateral,
                chosen_neurons=chosen_neurons,
            )[1]

        colors = [proximal_color, distal_color]
        separate_titles = ["Proximal compartment", "Distal compartment"]
        if self.inhibitory_distal:  # type: ignore[attr-defined]
            colors.append(inhibitory_color)
            separate_titles.append("Inhibitory interneuron")
        if plot_lateral and self.lateral_inhibition_weight is not None:
            colors.append(lateral_color)
            separate_titles.append("Lateral inhibition")

        if len(compartments) != len(colors):
            raise NotImplementedError(
                "Number of compartments does not match number of colors."
            )

        for c, comp in enumerate(compartments):
            if norm_by == "max_per":
                use_norm_by = comp.get_min_max_firingrates(
                    t_start=t_start, t_end=t_end, chosen_neurons=chosen_neurons
                )[1]
            else:
                use_norm_by = norm_by
            color = colors[c] or comp.color
            use_sub_ax = ax1D[c] if separate_axes else sub_ax
            sub_ax_out = comp.plot_rate_timeseries(
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                sub_ax=use_sub_ax,
                color=color,
                norm_by=use_norm_by,
                in_min=in_min,
                lw=lw,
                autosave=False,
                **kwargs,
            )
            if not separate_axes:
                sub_ax = sub_ax or sub_ax_out

        if separate_axes:
            if single_x_axis:
                plot_util.clear_bottom(ax1D[:-1])

            plural = "s" if self.n > 1 else ""
            for s, sub_ax in enumerate(ax1D):
                sub_ax.set_title(f"{separate_titles[s]}{plural}")
                if not omit_target_reset:
                    plot_fcts.mark_target_and_reset_points(self, sub_ax=sub_ax, lw=lw)
                if s != len(ax1D) - 1:
                    sub_ax.set_xlabel("")
            fig = np.asarray(ax).ravel()[0].figure
        else:
            if not no_legend:
                self.add_compartment_legend(
                    sub_ax,
                    compartment="all",
                    plot_lateral=plot_lateral,
                    proximal_color=proximal_color,
                    distal_color=distal_color,
                    inhibitory_color=inhibitory_color,
                    lateral_color=lateral_color,
                    lw=lw,
                )
            fig = sub_ax.figure

        plot_util.save_figure(fig, f"{self.name}_firingrate", save=autosave)  # type: ignore[attr-defined]

        return ax

    def plot_binned_rates(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        axes: np.ndarray | None = None,
        plot_lateral: bool = False,
        chosen_neurons: str | int | list | np.ndarray = "all",
        num_bins: int = 100,
        part_run: float = 0.2,
        merge: bool = True,
        plot_occ: bool = True,
        shared_range: bool = True,
        vmin: float = 0,
        vmax: float | None = None,
        mark_runs: bool = False,
        plot_colorbars: bool = True,
        cmap: str = "inferno",
        cbar_aspect: int = 12,
        cbar_label: str = "Firing rate",
        cbar_label_position: str = "left",
        autosave: bool | None = None,
    ) -> np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
        """
        self.plot_binned_rates()

        Plot the firing rates of the layer, binned by position
        (for 1D environments only).

        Args:
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time. Default is None.
        - axes (2D np.ndarray): 2D array of subplots
        - plot_lateral (bool, optional): Whether to plot the lateral inhibition layer,
            if it exists. Default is False.
        - chosen_neurons (str, int, list or np.ndarray, optional): Neurons to plot.
            Default is "all".
        - num_bins (int, optional): Number of bins to use for binning the firing rates.
            Default is 100.
        - part_run (float, optional): Proportion of the run to use for binning the
            firing rates. Default is 0.2.
        - merge (bool, optional): Whether to merge the firing rates of the neurons.
            Default is True.
        - plot_occ (bool, optional): Whether to plot the occupancy. Default is True.
        - shared_range (bool, optional): Whether to use a shared range for the
            colormap across all subplots. Default is True.
        - vmin (float, optional): Minimum value for the colormap. Default is 0.
        - vmax (float, optional): Maximum value for the colormap. Default is None.
        - mark_runs (bool, optional): Whether to mark the runs on the plot.
            Default is False.
        - plot_colorbars (bool, optional): Whether to plot colorbars. Default is True.
        - cmap (str, optional): Colormap to use for the firing rates. Default is "inferno".
        - cbar_aspect (int, optional): Aspect ratio of the colorbars. Default is 12.
        - cbar_label (str, optional): Label for the colorbars. Default is "Firing rate".
        - cbar_label_position (str, optional): Position of the colorbar label
            ("left" or "right"). Default is "left".
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Returns:
        - axes (2D np.ndarray): 2D array of subplots.
        """

        compartments = self.get_compartments("all", incl_lateral=plot_lateral)
        chosen_neurons = self.ProximalCompartment.get_chosen_neurons(chosen_neurons)

        num_rows = len(compartments)
        if axes is None:
            axes = plot_util.get_binned_rate_axes(
                num_neurons=len(chosen_neurons),
                num_rows=num_rows,
                plot_occ=plot_occ,
                plot_colorbars=plot_colorbars,
            )
        else:
            num_cols = len(chosen_neurons) + plot_occ
            ax_shape = np.asarray(axes).shape
            if ax_shape != (num_rows, num_cols):
                raise ValueError(
                    f"axes must have shape ({num_rows}, {num_cols}), not {ax_shape}."
                )
        titles = ["Proximal comp.", "Distal comp."]
        if self.inhibitory_distal:  # type: ignore[attr-defined]
            titles.append("Inhib. interneuron")
        if plot_lateral and self.lateral_inhibition_weight is not None:
            titles.append("Lateral inhib.")

        adjust_range = shared_range and (vmin is None or vmax is None)
        for c, comp in enumerate(compartments):
            comp.plot_binned_rates(
                t_start=t_start,
                t_end=t_end,
                ax=np.asarray(axes)[c],
                chosen_neurons=chosen_neurons,
                num_bins=num_bins,
                part_run=part_run,
                merge=merge,
                plot_occ=plot_occ,
                vmin=vmin,
                vmax=vmax,
                mark_runs=mark_runs,
                plot_colorbars=plot_colorbars,
                cmap=cmap,
                cbar_aspect=cbar_aspect,
                cbar_label=cbar_label,
                cbar_label_position=cbar_label_position,
                autosave=False,
            )
            if adjust_range:
                for sub_ax in np.asarray(axes)[c, : self.n]:
                    if sub_ax.images:
                        vmin = min(
                            np.inf if vmin is None else vmin,
                            sub_ax.images[0].get_array().min(),
                        )
                        vmax = max(
                            -np.inf if vmax is None else vmax,
                            sub_ax.images[0].get_array().max(),
                        )

            for i in chosen_neurons:
                np.asarray(axes)[c][i].set_title(f"{titles[c]} (#{i})")

        if adjust_range:
            last = len(chosen_neurons) - 1
            for sub_ax in np.asarray(axes)[:, : last + 1].ravel():
                if sub_ax.images:
                    sub_ax.images[0].set_clim(vmin, vmax)

            if plot_colorbars:
                num_skip = len(compartments) * (1 + int(plot_occ))
                caxes = sub_ax.figure.get_axes()[num_skip:]
                for i in range(0, len(caxes), 1 + int(plot_occ)):
                    cax = caxes[i]
                    cax.yaxis.set_label_position("left")  # corrects bug

        for ax1D in np.asarray(axes)[:-1]:
            for sub_ax in ax1D:
                sub_ax.set_xlabel("")

        fig = np.asarray(axes).ravel()[0].figure

        plot_util.save_figure(fig, f"{self.name}_binned_rates", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_distance_to_distal_target(
        self,
        neuron_idx=0,
        target_src_name="Obj",
        sub_ax=None,
        mark_proximal_BTSP=True,
        mark_teleport=True,
        mark_closest=True,
        min_dist=0.1,
        min_steps_btw=20,
        BTSP_prop_data="above",
        BTSP_s=plot_fcts.BTSP_S,
        base_s=8,
        in_min=True,
        autosave=None,
    ):
        """
        self.plot_distance_to_distal_target()

        Plot the distances from the agent's current position to the place cell center
        of the main input to the neuron's distal compartment, over time.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - mark_proximal_BTSP (bool, optional): Whether to mark the proximal compartment
            BTSP points. Default is True.
        - mark_teleport (bool, optional): Whether to mark the teleport points.
            Default is True.
        - mark_closest (bool, optional): Whether to mark the closest points to the
            target. Default is True.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.1.
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 20.
        - BTSP_prop_data (float ro str, optional): Proportional y-offset for the BTSP
            markers with respect to the distance data. If "above", markers are placed
            above the highest data point. Default is "above".
        - BTSP_s (float, optional): Marker size for BTSP points.
            Default is plot_fcts.BTSP_S.
        - base_s (float, optional): Base marker size for target and teleportation points.
            Default is 8.
        - in_min (bool, optional): Whether to plot the time in minutes instead of
            seconds. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with distances plotted.
        """

        t = self.Agent.get_t_history()
        if in_min:
            t = t / 60
        distances = self.get_distances_to_place_cell_center_of_main_distal_input(
            neuron_idx, src_name=target_src_name
        )

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=[8, 1.3])

        sub_ax.plot(t, distances)

        target_close_idxs = gen_util.get_minima_indices(
            distances, minimum=min_dist, min_pts_btw=min_steps_btw
        )

        if len(target_close_idxs) and mark_closest:
            plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.1, prop_high=0)
            plot_kwargs = plot_util.get_plot_marker_kwargs("target", base_s=base_s)
            sub_ax.scatter(
                t[target_close_idxs],
                np.zeros_like(target_close_idxs),
                alpha=0.7,
                **plot_kwargs,
            )

        if mark_teleport:
            if not hasattr(self.Agent, "teleportation_df"):
                mark_teleport = False
            elif len(self.Agent.teleportation_df) == 0:
                mark_teleport = False

        heights = [distances.max()]
        if mark_teleport:
            heights = self.Environment.get_marker_yvals(
                sub_ax,
                heights,
                object_type="teleport",
                base_s=base_s,
                above=above,
            )
            self.Agent.add_teleportation_markers_to_plots(
                sub_ax, timeseries=True, base_s=base_s, heights=heights
            )

            legend = sub_ax.get_legend()
            if legend is not None:
                sub_ax.legend(loc="upper right", fontsize=5)

        if mark_proximal_BTSP:
            above = 0.8 if mark_teleport else 1.5
            heights = plot_util.get_marker_yvals(
                sub_ax,
                data=heights,
                s=BTSP_s,
                prop_data=BTSP_prop_data,
                above=3,
            )
            self.ProximalCompartment.add_BTSP_markers_to_plots(
                sub_ax,
                chosen_neurons=[neuron_idx],
                timeseries=True,
                s=BTSP_s,
                heights=heights,
            )

        sub_ax.spines[["right", "top"]].set_visible(False)
        sub_ax.set_ylabel("Dist. to target")

        xlabel = "Time (min)" if in_min else "Time (s)"
        sub_ax.set_xlabel(xlabel)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_distances_to_target", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_distances_to_distal_targets(
        self,
        target_src_name="Obj",
        num_neurons="all",
        mark_proximal_BTSP=True,
        mark_teleport=True,
        mark_closest=True,
        min_dist=0.1,
        min_steps_btw=20,
        axes=None,
        num_cols=2,
        sharey=True,
        in_min=True,
        autosave=None,
    ):
        """
        self.plot_distances_to_distal_targets()

        Plot the distances from the agent's current position to the place cell center
        of the main input to the neuron's distal compartment, over time.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - num_neurons (str or int, optional): Number of neurons whose distal target
            distances should be plotted. Default is "all".
        - mark_proximal_BTSP (bool, optional): Whether to mark the proximal compartment
            BTSP points. Default is True.
        - mark_teleport (bool, optional): Whether to mark the teleport points.
            Default is True.
        - mark_closest (bool, optional): Whether to mark the closest points to the
            target. Default is True.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.1.
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 20.
        - axes (2D np.ndarray): Array of subplots to plot on (one per neuron).
            Default is None.
        - num_cols (int, optional): Number of columns in the subplot array.
            Default is 2.
        - sharey (bool, optional): Whether to share the y-axis across subplots.
            Default is True.
        - in_min (bool, optional): Whether to plot the time in minutes instead of
            seconds. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

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

                    self.plot_distance_to_distal_target(
                        neuron_idx=i,
                        target_src_name=target_src_name,
                        sub_ax=sub_ax,
                        mark_proximal_BTSP=mark_proximal_BTSP,
                        mark_teleport=use_mark_teleport,
                        mark_closest=mark_closest,
                        min_dist=min_dist,
                        min_steps_btw=min_steps_btw,
                        in_min=in_min,
                        autosave=False,
                    )
                    if use_mark_teleport and c != len(ax2D[0]) - 1:
                        legend = sub_ax.get_legend()
                        if legend is not None:
                            legend.remove()
                else:
                    sub_ax.spines[["left", "bottom"]].set_visible(False)
                    sub_ax.set_xticks(list())
                    sub_ax.set_yticks(list())

                if c == 0 and r == len(ax2D) // 2:
                    sub_ax.set_ylabel("Distance to target")
                else:
                    sub_ax.set_ylabel("")
                if r == len(ax2D) - 1:
                    xlabel = "Time (min)" if in_min else "Time (s)"
                    sub_ax.set_xlabel(xlabel)
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
        fig = sub_ax.figure
        fig.suptitle(f"Distance to target for {neuron_str} neurons.", y=y)

        plot_util.save_figure(fig, f"{self.name}_distances_to_targets", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_neuron_properties_at_BTSP_and_closest_to_target_steps(
        self,
        neuron_idx=0,
        target_src_name="Obj",
        t_start=None,
        t_end=None,
        axes=None,
        k=5,
        no_legend=False,
        autosave=None,
    ):
        """
        self.plot_neuron_properties_at_BTSP_and_closest_to_target_steps()

        Plot properties (step number, firing rate, velocity near and angle from target)
        at BTSP and closest to target steps for a neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - t_start (int, optional): Start time for plotting. Default is None.
        - t_end (int, optional): End time for plotting. Default is None.
        - axes (2D np.ndarray, optional): Array of subplots to plot on. Default is None.
        - k (bool, optional): Number of points across which to smooth the firing rate,
        velocity and angle data backward, before the target point. If None, no
        smoothing is done. Default is 5.
        - no_legend (bool, optional): Whether to skip adding a legend to the last plot.
            Default is False.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Returns:
        - axes (2D np.ndarray): Array of subplots with properties plotted.
        """

        if axes is None:
            _, axes = plt.subplots(1, 4, figsize=[8, 2], sharey=True, squeeze=False)
        elif len(axes.ravel()) != 4:
            raise ValueError("axes must have length 4.")

        distances = self.get_distances_to_place_cell_center_of_main_distal_input(
            neuron_idx, src_name=target_src_name
        )
        firingrates = np.asarray(self.ProximalCompartment.history["firingrate"]).T[
            neuron_idx
        ]

        velocities = np.sqrt(np.sum(np.asarray(self.Agent.history["vel"]) ** 2, axis=1))
        angles = self.get_vectors_to_place_cell_center_of_main_distal_input(
            neuron_idx, src_name=target_src_name, polar=True
        )[:, 1]

        steps_dict = self.match_closest_to_target_steps_to_BTSP_steps(
            target_src_name="Obj", neuron_idx=neuron_idx, t_start=t_start, t_end=t_end
        )

        for i, (x_data_type, x_data, sub_ax) in enumerate(
            zip(
                ["Step", "Firing rate", "Velocity (m/s)", "Angle from target (deg)"],
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

            sub_ax_no_legend = True
            if i == (len(axes.ravel()) - 1):
                sub_ax_no_legend = no_legend

            plot_fcts.plot_property_at_BTSP_and_closest_to_target_steps(
                steps_dict,
                y_data=distances,
                x_data=x_data,
                x_data_type=x_data_type,
                y_data_type=y_data_type,
                sub_ax=sub_ax,
                no_legend=sub_ax_no_legend,
            )

            if x_data_type == "Step":
                for t in [t_start, t_end]:
                    if t is not None:
                        step = t / self.Agent.dt
                        sub_ax.axvline(step, color="k", ls="dashed", lw=1)

        fig = sub_ax.figure
        fig.suptitle(f"Properties at steps near target (#{neuron_idx})")

        plot_util.save_figure(fig, f"{self.name}_neuron_properties_at_BTSP", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_properties_at_BTSP_and_closest_to_target_steps(
        self,
        target_src_name="Obj",
        sort_by_num_BTSP=False,
        t_start=None,
        t_end=None,
        k=5,
        no_legend=False,
        autosave=None,
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
        - no_legend (bool, optional): Whether to skip adding a legend to the plots.
            Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

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
                len(self.ProximalCompartment.get_BTSP_step_dict()[i]) for i in sorter
            ]
            sorter = np.argsort(num_BTSP)

        for i, neuron_idx in enumerate(sorter):
            self.plot_neuron_properties_at_BTSP_and_closest_to_target_steps(
                neuron_idx=neuron_idx,
                target_src_name=target_src_name,
                t_start=t_start,
                t_end=t_end,
                axes=axes[i],
                k=k,
                no_legend=no_legend,
                autosave=False,
            )
            neuron_label = f"Neuron {neuron_idx}"
            labelpad = 15
            if sort_by_num_BTSP:
                n = num_BTSP[neuron_idx]
                plural = "" if n == 1 else "s"
                neuron_label = f"{neuron_label}\n({n} BTSP event{plural})"
                labelpad = 20

            # on the right side
            axes[i, -1].set_ylabel(neuron_label, rotation=270, labelpad=labelpad)
            axes[i, -1].yaxis.set_label_position("right")

        fig.suptitle("Properties at steps near target", y=0.885)

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

        plot_util.save_figure(sub_ax.figure, f"{self.name}_properties_at_BTSP", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_BTSP_counts_vs_target_visits(
        self,
        target_src_name="Obj",
        min_steps_btw=30,
        min_dist=0.1,
        applied_only=False,
        t_start=None,
        t_end=None,
        max_spread=0.3,
        spread_bin_width_prop=0.1,
        hline=1,
        xmin=0,
        alpha=0.5,
        sub_ax=None,
        autosave=None,
    ):
        """
        self.plot_BTSP_counts_vs_target_visits()

        Plot the number BTSP events vs visits for each neuron's targets.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - min_steps_btw (int, optional): Minimum number of steps between BTSP events.
            Default is 30.
        - min_dist (float, optional): Minimum distance to be considered a visit.
            Default is 0.1.
        - applied_only (bool, optional): Whether to only include neurons with
            applied BTSP events. Default is False.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - max_spread (float, optional): Max spread to apply to duplicate data over y
            axis. Default is 0.3.
        - spread_bin_width_prop (float, optional): Proportion of x axis range to bin
            together when spreading data. Default is 0.1.
        - hline (float, optional): Y value at which to draw a horizontal line.
            Default is 1.
        - xmin (float, optional): Minimum x value to display. Default is 0.
        - ymin (float, optional): Minimum y value to display. Default is 0.
        - alpha (float, optional): Alpha value for scatter plot points. Default is 0.5.
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with BTSP frequency plotted.
        """

        nbr_visits_per_target = self.get_nbr_visits_per_target(
            target_src_name=target_src_name,
            min_steps_btw=min_steps_btw,
            min_dist=min_dist,
            t_start=t_start,
            t_end=t_end,
        )

        BTSP_counts = self.ProximalCompartment.get_BTSP_counts(
            applied_only=applied_only, t_start=t_start, t_end=t_end
        )

        if max_spread:
            spread_bin_width = spread_bin_width_prop * (
                nbr_visits_per_target.max() - nbr_visits_per_target.min()
            )
            BTSP_counts = gen_util.spread_data(
                nbr_visits_per_target,
                BTSP_counts,
                max_spread=max_spread,
                spread_bin_width=spread_bin_width,
            )

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(6, 3))

        if hline is not None:
            sub_ax.axhline(hline, color="k", ls="dashed", lw=1, alpha=0.3)
        sub_ax.scatter(
            nbr_visits_per_target,
            BTSP_counts,
            color=self.ProximalCompartment.color,
            alpha=alpha,
            s=10,
        )

        plot_util.pad_axis(sub_ax, axis="x", pad_prop=0.1)
        plot_util.pad_axis(sub_ax, axis="y")
        if xmin is None:
            xticks = sub_ax.get_xticks()
            if len(xticks) > 1:
                xmin, xmax = sub_ax.get_xlim()
                xmin = min(min(xticks), xmin)
                xmax = max(max(xticks), xmax)
                sub_ax.set_xlim(xmin, xmax)
        else:
            sub_ax.set_xlim(xmin, None)
        if sub_ax.get_ylim()[0] > 0:
            sub_ax.set_ylim(0, None)

        yticks = np.arange(BTSP_counts.max() + 1)
        if len(yticks) > 5:
            yticks = yticks[:: int(len(yticks) / 5) + 1]
        sub_ax.set_yticks(yticks)

        sub_ax.spines[["right", "top"]].set_visible(False)
        sub_ax.set_xlabel("Number of target visits")
        sub_ax.set_ylabel("Number of BTSP events")
        sub_ax.set_title("BTSP target visits vs events", y=1.05)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_BTSP_counts_vs_visits", save=autosave)  # type: ignore[attr-defined]

        return sub_ax
