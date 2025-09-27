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
    (somatic and apical), each of which is an NMDALayer. An additional HebbianLayer
    apical inhibition compartment can optionally be included

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 2,
        "name": "TwoCompLayer",
        "somatic_input_layers": [],
        "apical_input_layers": [],
        "somatic_to_apical_weight": 0.2,
        "apical_to_somatic_weight": 1.0,
        "apical_first": True,
        "somatic_color": "C0",
        "apical_color": "C1",
        "inhibitory_apical": True,
        "inhibitory_color": "k",
        "inhibitory_weight": 3.0,  # multiplied by -1 identity matrix
        "inhibitory_activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_PARAMS,
        "inhibitory_input_filter_tau": 3,
        "inhibitory_input_trend_tau": None,
        "lateral_inhibition_weight": None,
        "lateral_tau": 0.3,
    }

    No property attributes.

    List of methods:
        • self.set_learn()
        • self.set_BTSP_learn()
        • self.get_place_cell_center_of_main_apical_input()
        • self.get_vectors_to_place_cell_center_of_main_apical_input()
        • self.get_distances_to_place_cell_center_of_main_apical_input()
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
        "somatic_input_layers": [],
        "apical_input_layers": [],
        "somatic_to_apical_weight": 0.2,
        "apical_to_somatic_weight": 1.0,
        "apical_first": True,
        "somatic_color": "C0",
        "apical_color": "C1",
        "inhibitory_apical": True,
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

        self._organize_params(params)
        self._create_compartments()

        self.set_learn(somatic=True, apical=True, inhibitory=False)

    def _organize_params(self, params: dict[str, Any]):
        """
        self._organize_params(params)

        Organise the parameters passed to the TwoCompLayer class, passing them to each
        compartment as appropriate.

        Attributes:
        - apical_params (dict): Parameters for the apical compartment.
        - name (str): Name of the layer.
        - somatic_params (dict): Parameters for the somatic compartment.

        Args:
        - params (dict): Parameters passed to the TwoCompLayer class.
        """

        self.somatic_params = {"name": "somatic"}
        self.apical_params = dict()

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
            "somatic_to_apical_weight",
            "apical_to_somatic_weight",
            "apical_first",
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
                if "apical_name" not in all_params.keys():
                    self.apical_params["name"] = f"{value}_apical"
                if "somatic_name" not in all_params.keys():
                    self.somatic_params["name"] = f"{value}_somatic"

            elif key == "n":
                self.n = value
                self.somatic_params["n"] = value
                self.apical_params["n"] = value

            elif key.startswith("somatic_") or key.startswith("apical_"):
                for compartment, comp_dict in [
                    ("somatic", self.somatic_params),
                    ("apical", self.apical_params),
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
                self.somatic_params[key] = value
                self.apical_params[key] = value

    def _create_compartments(self):
        """
        self._create_compartments()

        Create the somatic and apical compartments, and connect them to each other for
        each neuron.

        If applicable, an inhibitory compartment is also created for each neuron and
        connected to the neuron's somatic and apical compartments.

        Inter-compartment connections:
            Somatic <--> Apical
            if self.inhibitory_apical:
                Somatic -->* Apical inhibition --> Apical
            *: learning possible

        Attributes:
        - ApicalCompartment (learning_neurons.NMDALayer): Apical compartment.
        - ApicalInhibition (learning_neurons.HebbianLayer): Inhibitory compartment.
        - SomaticCompartment (learning_neurons.NMDALayer): Somatic compartment.
        """

        self.SomaticCompartment = learning_neurons.NMDALayer(
            self.Agent, self.somatic_params
        )
        self.ApicalCompartment = learning_neurons.NMDALayer(
            self.Agent, self.apical_params
        )

        if self.SomaticCompartment.n != self.n or self.ApicalCompartment.n != self.n:  # type: ignore[attr-defined]
            raise ValueError(
                f"The two compartment layers must have same number of units ({self.n})."
            )

        apical_to_somatic_weight = np.eye(self.n) * self.apical_to_somatic_weight  # type: ignore[attr-defined]
        self.SomaticCompartment.add_input_layers_with_no_learning(self.ApicalCompartment.name)  # type: ignore[attr-defined]
        self.SomaticCompartment.add_input(
            self.ApicalCompartment,
            w=apical_to_somatic_weight,
            recurrent=not (self.apical_first),
        )

        somatic_to_apical_weight = np.eye(self.n) * self.somatic_to_apical_weight  # type: ignore[attr-defined]
        self.ApicalCompartment.add_input_layers_with_no_learning(
            self.SomaticCompartment.name  # type: ignore[attr-defined]
        )
        self.ApicalCompartment.add_input(
            self.SomaticCompartment,
            w=somatic_to_apical_weight,
            recurrent=self.apical_first,
        )

        if self.inhibitory_apical:  # type: ignore[attr-defined]
            inhibitory_params = {
                "name": "SomaticInhibitionOfApical",
                "n": self.n,
                "activation_function": self.inhibitory_activation_function,  # type: ignore[attr-defined]
                "color": self.inhibitory_color,  # type: ignore[attr-defined]
                "input_filter_tau": self.inhibitory_input_filter_tau,  # type: ignore[attr-defined]
                "input_trend_tau": self.inhibitory_input_trend_tau,  # type: ignore[attr-defined]
            }

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "No input layers", UserWarning)
                self.ApicalInhibition = learning_neurons.HebbianLayer(
                    self.Agent, params=inhibitory_params
                )

            somatic_input = np.eye(self.n) * self.inhibitory_weight  # type: ignore[attr-defined]
            self.ApicalInhibition.add_input(self.SomaticCompartment, w=somatic_input)

            apical_inhibition = np.eye(self.n) * -1
            self.ApicalCompartment.add_input_layers_with_no_learning(
                self.ApicalInhibition.name  # type: ignore[attr-defined]
            )
            self.ApicalCompartment.add_input(
                self.ApicalInhibition, w=apical_inhibition, recurrent=self.apical_first
            )

        if self.lateral_inhibition_weight is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "No input layers", UserWarning)

                lateral_params = {
                    "name": "LateralInhibition",
                    "n": self.n,
                    "activation_function": self.SomaticCompartment.activation_function,
                    "color": "gray",
                    "input_filter_tau": self.lateral_tau,
                }

                self.LateralInhibition = learning_neurons.SmoothFeedForwardLayer(
                    self.Agent, params=lateral_params
                )

            self.LateralInhibition.add_input(self.SomaticCompartment, w=np.eye(self.n))
            lateral_inhibition = (np.eye(self.n) - 1) * self.lateral_inhibition_weight
            self.SomaticCompartment.add_input(
                self.LateralInhibition, w=lateral_inhibition
            )
            self.SomaticCompartment.add_input_layers_with_no_learning(
                self.LateralInhibition.name
            )

    def set_learn(self, learn=None, somatic=None, apical=None, inhibitory=None):
        """
        self.set_learn()

        Set, for each compartment, whether it should learn during self.update() calls.
        Only affects input weights that are learnable.

        Args:
        - learn (bool, optional): Whether to learn learnable weights into all
            compartments.  Default is None.
        - somatic (bool, optional): Whether to learn learnable weights into the
            somatic compartment.  Default is None.
        - apical (bool, optional): Whether to learn learnable weights into the
            apical compartment. Default is None.
        - inhibitory (bool, optional): Whether to learn learnable weights into the
            inhibitory compartment. Default is None.
        """

        if learn is not None:
            somatic = learn if somatic is None else somatic
            apical = learn if apical is None else apical
            inhibitory = learn if inhibitory is None else inhibitory

        self.SomaticCompartment.set_learn(somatic)
        self.ApicalCompartment.set_learn(apical)
        if self.inhibitory_apical:  # type: ignore[attr-defined]
            self.ApicalInhibition.set_learn(inhibitory)

    def set_BTSP_learn(self, learn=None, somatic=None, apical=None):
        """
        self.set_BTSP_learn()

        Set whether the somatic and apical compartments should learn using BTSP during
        self.update() calls. Only affects input weights that are learnable.

        Args:
        - learn (bool, optional): Whether to learn learnable weights into both
            compartments. Default is None
        - somatic (bool, optional): Whether to learn learnable weights into the
            somatic compartment.  Default is None.
        - apical (bool, optional): Whether to learn learnable weights into the
            apical compartment. Default is None.
        """

        if learn is not None:
            somatic = learn if somatic is None else somatic
            apical = learn if apical is None else apical

        self.SomaticCompartment.set_BTSP_learn(somatic)
        self.ApicalCompartment.set_BTSP_learn(apical)

    def get_compartments(
        self,
        compartment: str = "all",
        incl_lateral: bool = False,
    ):
        """
        self.get_compartments()

        - compartment (str, optional): Which compartments to retrieve
            ("somatic", "apical", "both", "inhibitory", "all"). Default is "all".

        Returns:
        - compartments (list): List of compartments.
        """

        if compartment not in ["somatic", "apical", "both", "inhibitory", "all"]:
            raise ValueError(
                "compartment must be 'somatic', 'apical', 'both', 'inhibitory' or 'all', "
                f"not '{compartment}'."
            )

        compartments = list()
        if compartment in ["somatic", "both", "all"]:
            compartments.append(self.SomaticCompartment)
        if compartment in ["apical", "both", "all"]:
            compartments.append(self.ApicalCompartment)
        if compartment in ["inhibitory", "all"]:
            if self.inhibitory_apical:
                compartments.append(self.ApicalInhibition)
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
        - t_end (float, optional): Stop time for obtaining firingrate min and max.
            Default is None.
        - chosen_neurons (str, int, list or np.ndarray, optional): Neurons to consider
            for min and max firing rates. Default is "all".
        - compartment (str, optional): Which compartment to obtain max for
            ("somatic", "apical", "both", "inhibitory", "all"). Default is "all".
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

    def get_index_of_main_apical_input(
        self, neuron_idx: int = 0, src_name: str = "Obj"
    ):
        """
        self.get_index_of_main_apical_input()

        Get the index of the main input to the apical compartment of a specified neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".

        Returns:
        - input_idx (int): Index of main apical input.
        """

        if src_name not in self.ApicalCompartment.inputs.keys():
            raise ValueError(f"No '{src_name}' input to apical compartment.")

        input_dict = self.ApicalCompartment.inputs[src_name]

        if not isinstance(input_dict["layer"], riab_neurons.PlaceCells):
            raise ValueError(f"Input layer '{src_name}' is not a PlaceCells layer.")

        if neuron_idx > self.n:
            raise ValueError(
                f"Neuron index {neuron_idx} is greater than the number of neurons "
                "in the layer."
            )

        input_idx = np.argmax(input_dict["w"][:, neuron_idx])

        return input_idx

    def get_place_cell_center_of_main_apical_input(
        self, neuron_idx: int = 0, src_name: str = "Obj"
    ):
        """
        self.get_place_cell_center_of_main_apical_input()

        Get the place cell center input location for the apical compartment of a
        specified neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".

        Returns:
        - place_cell_center (1D np.ndarray): Main apical input place cell center
            location.
        """

        input_idx = self.get_index_of_main_apical_input(
            neuron_idx=neuron_idx, src_name=src_name
        )

        input_layer = self.ApicalCompartment.inputs[src_name]["layer"]

        place_cell_center = input_layer.place_cell_centers[input_idx]

        return place_cell_center

    def get_vectors_to_place_cell_center_of_main_apical_input(
        self,
        neuron_idx: int = 0,
        src_name: str = "Obj",
        polar: bool = False,
        radians: bool = False,
    ):
        """
        self.get_vectors_to_place_cell_center_of_main_apical_input()

        Get the vectors from the agent's current position to the place cell center
        input location for the apical compartment of a specified neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - polar (bool, optional): Whether to return vectors in polar coordinates.
            Default is False.
        - radians (bool, optional): If True and polar is True, return angles in radians.
            Default is False.

        Returns:
        - vectors (2D np.ndarray): Vectors from agent's position to apical input
            place cell center.
        """

        place_cell_center = self.get_place_cell_center_of_main_apical_input(
            neuron_idx=neuron_idx, src_name=src_name
        )
        pos = np.asarray(self.Agent.history["pos"])

        vectors = trig_util.get_vectors_to_target(
            pos, target=place_cell_center, polar=polar, radians=radians
        )

        return vectors

    def get_distances_to_place_cell_center_of_main_apical_input(
        self, neuron_idx: int = 0, src_name: str = "Obj"
    ):
        """
        self.get_distances_to_place_cell_center_of_main_apical_input()

        Get the distances from the agent's current position to the place cell center
        input location for the apical compartment of a specified neuron.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".

        Returns:
        - distances (1D np.ndarray): Distances from agent's position to apical input
            place cell center.
        """

        vectors = self.get_vectors_to_place_cell_center_of_main_apical_input(
            neuron_idx, src_name
        )

        distances = np.linalg.norm(vectors, ord=2, axis=1)

        return distances

    def get_target_visits(
        self,
        neuron_idx: int = 0,
        target_src_name: str = "Obj",
        min_pts_btw=30,
        min_dist=0.05,
    ):
        """
        self.get_target_visits()

        Get the indices of the steps where the agent is closest to the target specified
        by the place cell center of the main input to the neuron's apical compartment.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - min_pts_btw (int, optional): Minimum number of steps between closest steps.
            Default is 30.
        - min_dist (float, optional): Minimum distance to be considered a visit.
            Default is 0.05.

        Returns:
        - visit_indices (1D np.ndarray): Indices of the steps where the agent is
            closest to the target.
        """

        distances = self.get_distances_to_place_cell_center_of_main_apical_input(
            neuron_idx=neuron_idx, src_name=target_src_name
        )

        visit_indices = gen_util.get_minima_indices(
            distances, min_pts_btw=min_pts_btw, minimum=min_dist
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
        cell center of the main input to the neuron's apical compartment.

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

        distances = self.get_distances_to_place_cell_center_of_main_apical_input(
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
        min_pts_btw=30,
        min_dist=0.1,
        t_start=None,
        t_end=None,
    ):
        """
        self.get_nbr_visits_per_target()

        Get the number of visits to the target specified by the place cell center of
        the main input to the neuron's apical compartment for each neuron in the layer.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - min_pts_btw (int, optional): Minimum number of steps between closest steps.
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
            apical compartment for each neuron in the layer.
        """

        _, startid, endid = self.SomaticCompartment.get_plotting_times(
            t_start=t_start, t_end=t_end
        )

        nbr_visits_per_BTSP_target = np.zeros(self.n, dtype=int)
        for neuron_idx in range(self.n):
            visit_indices = self.get_target_visits(
                neuron_idx=neuron_idx,
                target_src_name=target_src_name,
                min_pts_btw=min_pts_btw,
                min_dist=min_dist,
            )

            if len(visit_indices):
                visit_indices = visit_indices[
                    (visit_indices >= startid) & (visit_indices < endid)
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
            BTSP steps, with keys:
                "steps_before": closest steps occurring before neuron's first BTSP step
                "steps_near_BTSP": closest steps occurring near BTSP steps
                "steps_of_nearest_BTSP": index of nearest BTSP step for each step_near_BTSP value
                "steps_other": closest steps after first BTSP, but not near a BTSP step
                "all_BTSP_steps": all BTSP steps, whether close to target or not
        """

        if neuron_idx >= self.n:
            raise ValueError(
                f"Neuron index ({neuron_idx}) must be smaller than number of "
                f"neurons ({self.n})."
            )

        _, start, end = self.SomaticCompartment.get_plotting_times(
            t_start=t_start, t_end=t_end
        )

        BTSP_steps = self.SomaticCompartment.get_BTSP_step_dict()[neuron_idx]
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

    def update(self, apical_first: bool | None = None):
        """
        self.update()

        Update the somatic and apical compartments of the two compartment layer. If
        there is an apical inhibition compartment, it is also updated.

        Update order:
            1. Apical inhibition compartment (if applicable)
            if apical_first:
                2. Apical compartment
                3. Somatic compartment
            otherwise:
                2. Somatic compartment
                3. Apical compartment

        Args:
        - apical_first (bool, optional): Whether to update the apical compartment
            before the somatic compartment. If None, the attribute is used.
            Default is None.
        """

        if self.inhibitory_apical:  # type: ignore[attr-defined]
            self.ApicalInhibition.update()

        if apical_first is None:
            apical_first = self.apical_first  # type: ignore[attr-defined]

        if apical_first:
            self.ApicalCompartment.update()
            self.SomaticCompartment.update()
            if self.lateral_inhibition_weight is not None:
                self.LateralInhibition.update()
        else:
            self.SomaticCompartment.update()
            self.ApicalCompartment.update()
            if self.lateral_inhibition_weight is not None:
                self.LateralInhibition.update()

        return

    def add_compartment_legend(
        self,
        sub_ax,
        compartment="all",
        plot_lateral=False,
        loc="best",
        somatic_color=None,
        apical_color=None,
        inhibitory_color=None,
        lateral_color=None,
        lw=1.0,
        **kwargs,
    ):
        """
        self.add_compartment_legend()

        Add a legend to a plot with the colors of the specified compartments.

        Args:
        - sub_ax (plt.Axes): Subplot to add the legend to.
        - compartment (str, optional): Which compartments to include in the legend
            ("somatic", "apical", "inhibitory", "all"). Default is "all".
        - plot_lateral (bool, optional): Whether to include the lateral inhibition
            compartment in the legend. Default is False.
        - somatic_color (str, optional): Color for somatic compartment. Default is None.
        - apical_color (str, optional): Color for apical compartment. Default is None.
        - inhibitory_color (str, optional): Color for inhibitory compartment.
            Default is None.
        - lateral_color (str, optional): Color for lateral inhibition compartment.
            Default is None.
        - lw (float, optional): Line width for the timeseries. Default is 1.0.

        Keyword args:
        - **kwargs: Additional keyword arguments passed to plt.legend().
        """

        if compartment not in ["somatic", "apical", "both", "inhibitory", "all"]:
            raise ValueError(
                "compartment must be 'somatic', 'apical', 'both', 'inhibitory' or "
                f"'all', not '{compartment}'."
            )

        if compartment in ["all", "somatic"]:
            color = somatic_color or self.SomaticCompartment.color
            sub_ax.plot([], [], color=color, lw=lw, label="somatic")
        if compartment in ["all", "apical"]:
            color = apical_color or self.ApicalCompartment.color
            sub_ax.plot([], [], color=color, lw=lw, label="apical")
        if self.inhibitory_apical and compartment in ["all", "inhibitory"]:
            color = inhibitory_color or self.ApicalInhibition.color
            sub_ax.plot([], [], color=color, lw=lw, label="inhibitory")
        if plot_lateral and self.lateral_inhibition_weight is not None:
            color = lateral_color or self.LateralInhibition.color
            sub_ax.plot([], [], color=color, lw=lw, label="lat. inh.")

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
            2D ("somatic", "apical", "both", "inhibitory", "all"). Default is None
            (i.e., "somatic" if environment is 2D, and "both" otherwise).
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
                compartment = "somatic"

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

        if self.Agent.Environment.dimensionality == "2D" and compartment == "all":
            warnings.warn(
                "Plotting rate maps for all compartments in a 2D environment will "
                "result in only the somatic compartment appearing."
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
            2D ("somatic", "apical", "inhibitory" or "both"). Default is None
            (i.e., "somatic" if environment is 2D, and "both" otherwise).
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - title (str, optional): Title for the figure. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to
            NMDALayer.plot_rate_maps_across_learning().

        Raises:
        - ValueError: If compartment is not "somatic", "apical", "inhibitory" or "all".
        - ValueError: If compartment is "inhibitory", but self.inhibitory_apical is False.

        Returns:
        - axes (2D np.ndarray): Array of subplots. If input axes was None,
            shape is 2D (number of ROIs, num_maps) or v.v. if only one ROI.
        """

        if compartment is None:
            if self.Agent.Environment.dimensionality == "1D":
                compartment = "all"
            else:
                compartment = "somatic"

        if self.Agent.Environment.dimensionality == "2D" and compartment == "both":
            warnings.warn(
                "Plotting rate maps across learning for both compartments in a 2D "
                "environment will result in only the somatic compartment appearing."
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
            elif compartment == "somatic":
                title_start = "Somatic rate maps"
            elif compartment == "inhibitory":
                title_start = "Inhibition rate maps"
            else:
                title_start = "Apical rate maps"

            title = f"{title_start} across learning"

        fig = np.asarray(axes).ravel()[0].figure

        y = 0.9 if self.Agent.Environment.dimensionality == 1 else 0.97
        fig.suptitle(title, y=y)

        plot_util.save_figure(fig, f"{self.name}_rate_maps_across_learning", save=autosave)  # type: ignore[attr-defined]

        return axes

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
        cbar_aspect: int = 12,
        cbar_label: str = "Firing rate",
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
        - cbar_aspect (int, optional): Aspect ratio of the colorbars. Default is 12.
        - cbar_label (str, optional): Label for the colorbars. Default is "Firing rate".
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Returns:
        - axes (2D np.ndarray): 2D array of subplots.
        """

        compartments = self.get_compartments("all", incl_lateral=plot_lateral)
        chosen_neurons = self.SomaticCompartment.get_chosen_neurons(chosen_neurons)

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
        titles = ["Somatic comp.", "Apical comp."]
        if self.inhibitory_apical:  # type: ignore[attr-defined]
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
                cbar_aspect=cbar_aspect,
                cbar_label=cbar_label,
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

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str | int | list | np.ndarray = "all",
        ax: plt.Axes | np.ndarray | None = None,
        somatic_color: str | None = None,
        apical_color: str | None = None,
        inhibitory_color: str | None = None,
        lateral_color: str | None = None,
        separate_axes: bool = False,
        plot_lateral: bool = False,
        single_x_axis: bool = True,
        norm_by: str | None = None,
        in_min: bool = True,
        lw: float = 1.0,
        omit_reset: bool = False,
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
        - somatic_color (str, optional): Color for somatic compartment. Default is None.
        - apical_color (str, optional): Color for apical compartment. Default is None.
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
        - omit_reset (bool, optional): Whether to omit resetting the points for
            marking target and reset points. Default is False.
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
                t_start=t_start,
                t_end=t_end,
                incl_lateral=plot_lateral,
                chosen_neurons=chosen_neurons,
            )[1]

        colors = [somatic_color, apical_color]
        separate_titles = ["Somatic compartment", "Apical compartment"]
        if self.inhibitory_apical:  # type: ignore[attr-defined]
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

            for s, sub_ax in enumerate(ax1D):
                sub_ax.set_title(separate_titles[s])
                plot_fcts.mark_target_and_reset_points(
                    self, sub_ax=sub_ax, lw=lw, omit_reset=omit_reset
                )
                if s != len(ax1D) - 1:
                    sub_ax.set_xlabel("")
            fig = np.asarray(ax).ravel()[0].figure
        else:
            if not no_legend:
                self.add_compartment_legend(
                    sub_ax,
                    compartment="all",
                    plot_lateral=plot_lateral,
                    somatic_color=somatic_color,
                    apical_color=apical_color,
                    inhibitory_color=inhibitory_color,
                    lateral_color=lateral_color,
                    lw=lw,
                )
            fig = sub_ax.figure

        plot_util.save_figure(fig, f"{self.name}_firingrate", save=autosave)  # type: ignore[attr-defined]

        return ax

    def plot_distances_to_target(
        self,
        neuron_idx=0,
        target_src_name="Obj",
        sub_ax=None,
        mark_somatic_BTSP=True,
        mark_teleport=True,
        mark_closest=True,
        min_dist=0.1,
        min_steps_btw=20,
        log_num_closest=False,
        in_min=True,
        autosave=None,
    ):
        """
        self.plot_distances_to_target()

        Plot the distances from the agent's current position to the place cell center
        of the main input to the neuron's apical compartment, over time.

        Args:
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - mark_somatic_BTSP (bool, optional): Whether to mark the somatic compartment
            BTSP points. Default is True.
        - mark_teleport (bool, optional): Whether to mark the teleport points.
            Default is True.
        - mark_closest (bool, optional): Whether to mark the closest points to the
            target. Default is True.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.1.
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 20.
        - log_num_closest (bool, optional): Whether to print the number of closest
            steps identified. Default is False.
        - in_min (bool, optional): Whether to plot the time in minutes instead of
            seconds. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with distances plotted.
        """

        t = np.asarray(self.Agent.history["t"])
        if in_min:
            t = t / 60
        distances = self.get_distances_to_place_cell_center_of_main_apical_input(
            neuron_idx, src_name=target_src_name
        )

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=[8, 1.3])

        sub_ax.plot(t, distances)

        if mark_somatic_BTSP:
            self.SomaticCompartment.add_BTSP_markers_to_plots(
                sub_ax, chosen_neurons=[neuron_idx], timeseries=True
            )
            plot_util.pad_axis(sub_ax, prop_high=1.0)

        if mark_teleport:
            plot_util.pad_axis(sub_ax, pad_prop=0.1, prop_high=1.0)
            self.Agent.add_teleportation_markers_to_plots(sub_ax, timeseries=True)

            legend = sub_ax.get_legend()
            if legend is not None:
                sub_ax.legend(loc="upper right", fontsize=5)

        if mark_closest or log_num_closest:
            closest_steps = self.get_closest_steps_to_target(
                neuron_idx=neuron_idx,
                target_src_name=target_src_name,
                min_dist=min_dist,
                min_steps_btw=min_steps_btw,
                log=log_num_closest,
            )
            closest_steps = gen_util.get_minima_indices(
                distances, minimum=min_dist, min_pts_btw=min_steps_btw
            )

            if mark_closest and len(closest_steps):
                plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.2)
                sub_ax.plot(
                    t[closest_steps],
                    np.zeros_like(closest_steps),
                    lw=0,
                    marker="o",
                    ms=2,
                    color=self.SomaticCompartment.color,
                )

        sub_ax.spines[["right", "top"]].set_visible(False)
        sub_ax.set_ylabel("Dist. to target")

        xlabel = "Time (min)" if in_min else "Time (s)"
        sub_ax.set_xlabel(xlabel)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_distances_to_target", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_distances_to_targets(
        self,
        target_src_name="Obj",
        num_neurons="all",
        mark_somatic_BTSP=True,
        mark_teleport=True,
        mark_closest=True,
        min_dist=0.1,
        min_steps_btw=20,
        axes=None,
        num_cols=2,
        sharey=True,
        log_num_closest=False,
        in_min=True,
        autosave=None,
    ):
        """
        self.plot_distances_to_targets()

        Plot the distances from the agent's current position to the place cell center
        of the main input to the neuron's apical compartment, over time.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - axes (2D np.ndarray): Array of subplots to plot on (one per neuron).
            Default is None.
        - neuron_idx (int, optional): Neuron index. Default is 0.
        - mark_somatic_BTSP (bool, optional): Whether to mark the somatic compartment
            BTSP points. Default is True.
        - mark_teleport (bool, optional): Whether to mark the teleport points.
            Default is True.
        - mark_closest (bool, optional): Whether to mark the closest points to the
            target. Default is True.
        - min_dist (float, optional): Minimum distance to be considered closest.
            Default is 0.1.
        - min_steps_btw (int, optional): Minimum number of steps between closest steps.
            Default is 20.
        - num_cols (int, optional): Number of columns in the subplot array.
            Default is 2.
        - sharey (bool, optional): Whether to share the y-axis across subplots.
            Default is True.
        - log_num_closest (bool, optional): Whether to print the number of closest
            steps identified. Default is False.
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

                    self.plot_distances_to_target(
                        neuron_idx=i,
                        target_src_name=target_src_name,
                        sub_ax=sub_ax,
                        mark_somatic_BTSP=mark_somatic_BTSP,
                        mark_teleport=use_mark_teleport,
                        mark_closest=mark_closest,
                        min_dist=min_dist,
                        min_steps_btw=min_steps_btw,
                        log_num_closest=log_num_closest,
                        in_min=in_min,
                        autosave=False,
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
        legend=True,
        autosave=None,
    ):
        """
        self.plot_neuron_properties_at_BTSP_and_closest_to_target_steps()

        Plot properties (step number, firing rate, velocity and angle near target)
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
        - legend (bool, optional): Whether to include a legend in the plots.
            Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Returns:
        - axes (2D np.ndarray): Array of subplots with properties plotted.
        """

        if axes is None:
            _, axes = plt.subplots(1, 4, figsize=[8, 2], sharey=True, squeeze=False)
        elif len(axes.ravel()) != 4:
            raise ValueError("axes must have length 4.")

        distances = self.get_distances_to_place_cell_center_of_main_apical_input(
            neuron_idx, src_name=target_src_name
        )
        firingrates = np.asarray(self.SomaticCompartment.history["firingrate"]).T[
            neuron_idx
        ]

        velocities = np.sqrt(np.sum(np.asarray(self.Agent.history["vel"]) ** 2, axis=1))
        angles = self.get_vectors_to_place_cell_center_of_main_apical_input(
            neuron_idx, src_name=target_src_name, polar=True
        )[:, 1]

        steps_dict = self.match_closest_to_target_steps_to_BTSP_steps(
            target_src_name="Obj", neuron_idx=neuron_idx, t_start=t_start, t_end=t_end
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

        fig = sub_ax.figure
        fig.suptitle(
            f"Properties near BTSP and closest to target steps (#{neuron_idx})"
        )

        plot_util.save_figure(fig, f"{self.name}_neuron_properties_at_BTSP", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_properties_at_BTSP_and_closest_to_target_steps(
        self,
        target_src_name="Obj",
        sort_by_num_BTSP=False,
        t_start=None,
        t_end=None,
        k=5,
        legend=True,
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
        - legend (bool, optional): Whether to include a legend in the plots.
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
                len(self.SomaticCompartment.get_BTSP_step_dict()[i]) for i in sorter
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
                legend=legend,
                autosave=False,
            )
            title = f"Neuron {neuron_idx}"
            if sort_by_num_BTSP:
                n = num_BTSP[neuron_idx]
                neuron_str = f"{n} BTSP event" if n == 1 else f"{n} BTSP events"
                title = f"{title} ({neuron_str})"
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

        plot_util.save_figure(sub_ax.figure, f"{self.name}_properties_at_BTSP", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_BTSP_counts_vs_target_visits(
        self,
        target_src_name="Obj",
        min_pts_btw=30,
        min_dist=0.1,
        applied_only=False,
        t_start=None,
        t_end=None,
        max_spread=0.1,
        sub_ax=None,
        autosave=None,
    ):
        """
        self.plot_BTSP_counts_vs_target_visits()

        Plot the number BTSP events vs visits for each neuron's targets.

        Args:
        - target_src_name (str, optional): Name of the input layer
            (must be a place cell-derived layer). Default is "Obj".
        - min_pts_btw (int, optional): Minimum number of points between BTSP events.
            Default is 30.
        - min_dist (float, optional): Minimum distance to be considered a visit.
            Default is 0.1.
        - applied_only (bool, optional): Whether to only include neurons with
            applied BTSP events. Default is False.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - max_spread (float, optional): Max spread to apply to duplicate data over y
            axis. Default is True.
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with BTSP frequency plotted.
        """

        nbr_visits_per_target = self.get_nbr_visits_per_target(
            target_src_name=target_src_name,
            min_pts_btw=min_pts_btw,
            min_dist=min_dist,
            t_start=t_start,
            t_end=t_end,
        )

        BTSP_counts = self.SomaticCompartment.get_BTSP_counts(
            applied_only=applied_only, t_start=t_start, t_end=t_end
        )

        if max_spread:
            BTSP_counts = gen_util.spread_data(
                nbr_visits_per_target, BTSP_counts, max_spread=max_spread
            )

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(6, 3))

        sub_ax.axhline(1, color="k", ls="dashed", lw=1, alpha=0.3)
        sub_ax.scatter(
            nbr_visits_per_target,
            BTSP_counts,
            color=self.SomaticCompartment.color,
            alpha=0.5,
            s=10,
        )

        plot_util.pad_axis(sub_ax, axis="x")
        plot_util.pad_axis(sub_ax, axis="y")
        if sub_ax.get_xlim()[0] > 0:
            sub_ax.set_xlim(0, None)
        if sub_ax.get_ylim()[0] > 0:
            sub_ax.set_ylim(0, None)

        sub_ax.spines[["right", "top"]].set_visible(False)
        sub_ax.set_xlabel("Number of target visits")
        sub_ax.set_ylabel("Number of BTSP events")
        sub_ax.set_title("BTSP target visits vs events", y=1.05)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_BTSP_counts_vs_visits", save=autosave)  # type: ignore[attr-defined]

        return sub_ax
