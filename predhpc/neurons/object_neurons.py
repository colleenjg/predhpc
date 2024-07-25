import copy
from typing import TYPE_CHECKING, Any
import warnings

from matplotlib import pyplot as plt
import numpy as np

from predhpc import env, plot_util, util
from predhpc.neurons import riab_neurons

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


class ObjectCells(riab_neurons.FeedForwardLayer):
    """
    ObjectCells()

    Class extending riab_neurons.FeedForwardLayer. Defines a population of neurons
    that respond to certain object types.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation.

    default_params = {
        "name": "ObjectCells",
        "description": "gaussian",  # input place cells
        "widths": 0.20,  # input place cells
        "wall_geometry": "line_of_sight",  # input place cells
        "min_fr": 0,
        "max_fr": 1,
    }

    List of properties (in addition to riab_neurons.FeedForwardLayer properties):
        • self.input_object_types
        • self.input_object_locations
        • self.object_types
        • self.place_cell_input_weights
        • self.neuron_type_dict

    List of methods (in addition to riab_neurons.FeedForwardLayer methods):
        • self.update()
        • self.log_num_neurons_per_object_name()
    """

    default_params = {
        "name": "ObjectCells",
        "description": "gaussian",  # input place cells
        "widths": 0.20,  # input place cells
        "wall_geometry": "line_of_sight",  # input place cells
        "min_fr": 0,
        "max_fr": 1,
    }

    ignored_param_keys = list()
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {"input_layers": list()}

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        ObjectCells(Agent)

        Initialise an object cell layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.
        - place_params (dict): Place cell parameter dictionary.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.Agent = Agent

        n = self._get_num_neurons()

        if "n" in params and params["n"] != n:
            raise ValueError(
                "Number of cells should not be passed as a parameter to "
                "ObjectCells. It is set automatically based on "
                "the environment."
            )

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        params = copy.deepcopy(params)

        # update object params
        self.params.update(params)
        self.params["n"] = n

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No input layers")
            super().__init__(Agent, self.params)

        # retrieve place cell keys and prepare place cell params
        self.place_params = dict()
        for key in ["description", "widths", "wall_geometry", "min_fr", "max_fr"]:
            self.place_params[key] = self.params[key]

        self._create_place_cell_layer()
        self._add_place_inputs()

    @property
    def input_object_types(self):
        """
        self.input_object_types

        Obtain all object types from the environment.

        Returns:
        - (1D np.ndarray): All object types.
        """

        if not hasattr(self, "_input_object_types"):
            self._input_object_types = self.Agent.Environment.objects["object_types"]
        return self._input_object_types

    @property
    def input_object_locations(self):
        """
        self.input_object_locations

        Obtain all object locations in the environment.

        Returns:
        - (2D np.ndarray): All object locations, with shape
            (object, location [x, ] or [x, y]).
        """

        if not hasattr(self, "_input_object_locations"):
            self._input_object_locations = self.Agent.Environment.objects["objects"]
        return self._input_object_locations

    @property
    def object_types(self):
        """
        self.object_types

        Obtain object types for each neuron in the layer.

        Returns:
        - (2D np.ndarray): Object type for each neuron in the layer.
        """

        if not hasattr(self, "_object_types"):
            if len(self.input_object_types) == 0:
                raise RuntimeError("No objects found in environment.")

            object_types = np.unique(self.input_object_types)
            np.random.shuffle(object_types)
            self._object_types = object_types
        return self._object_types

    @property
    def place_cell_input_weights(self):
        """
        self.place_cell_input_weights

        Obtain place cell input weights for each object cell.

        Returns:
        - (2D np.ndarray): Place cell input weights for each object cell,
            with shape (number of object neurons, number of input place cells).
        """

        if not hasattr(self, "_place_cell_input_weights"):
            place_cell_input_weights = np.zeros((self.n, self.PlaceCellInputs.n))

            for i, obj_type in enumerate(self.input_object_types):
                if obj_type not in self.object_types:
                    continue
                for j in np.where(np.asarray(self.object_types) == obj_type)[0]:
                    place_cell_input_weights[j, i] = 1
            self._place_cell_input_weights = place_cell_input_weights

        return self._place_cell_input_weights

    @property
    def neuron_type_dict(self):
        """
        self.neuron_type_dict

        Obtain the number of neurons per object name in the layer.

        Returns:
        - (dict): Number of neurons per object name in the layer, with keys and values:
            - object name (str): Number of neurons (int).
        """

        if not hasattr(self, "_neuron_type_dict"):
            object_types, counts = np.unique(self.object_types, return_counts=True)
            _neuron_type_dict = {obj_type: 0 for obj_type in self.input_object_types}
            for object_type, count in zip(object_types, counts):
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]
                if "teleport" in object_name:
                    object_name = "teleport"
                _neuron_type_dict[object_name] += count

            self._neuron_type_dict = _neuron_type_dict

        return self._neuron_type_dict

    def _get_num_neurons(self):
        """
        self._get_num_neurons()

        Obtain the number of neurons in the layer.

        Returns:
        - (int): Number of neurons in the layer.
        """

        num_neurons = len(self.object_types)
        return num_neurons

    def _create_place_cell_layer(self):
        """
        self._create_place_cell_layer()

        Create place cell layer, with one place cell per object location.

        Attributes:
        - PlaceCellInputs (riab_neurons.PlaceCells): Place cell layer.
        """

        self.place_params["place_cell_centres"] = self.input_object_locations
        self.PlaceCellInputs = riab_neurons.PlaceCells(
            self.Agent, params=self.place_params
        )

    def _add_place_inputs(self):
        """
        self._add_place_inputs()

        Add place cell layer as an input.
        """

        self.add_input(
            input_layer=self.PlaceCellInputs,
            w=self.place_cell_input_weights,
        )

    def update(self):
        """
        self.update()

        Update the object cell layer, after updating its place cell input layer.
        """

        self.PlaceCellInputs.update()
        super().update()

    def log_num_neurons_per_object_name(self):
        """
        self.log_num_neurons_per_object_name()

        Log the number of neurons per object name in the layer.
        """

        obj_strs = [
            f"{count} {obj_name} neurons"
            for obj_name, count in self._neuron_type_dict.items()
        ]

        sep = "\n    "
        log_str = f"Layer comprises:{sep}{sep.join(obj_strs)}"
        print(log_str)


class FixedObjectCells(ObjectCells):
    """
    FixedObjectCells()

    Class extending ObjectCells. Defines a population of neurons that respond to
    certain object types, with a certain number of neurons per object type.

    This class is only compatible with OpenField environments.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation.

    default_params = {
        "name": "FixedObjectCells",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
    }

    List of properties (in addition to riab_neurons.FeedForwardLayer properties):
        • self.object_types

    See ObjectCells for methods.
    """

    default_params = {
        "name": "FixedObjectCells",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        FixedObjectCells(Agent)

        Initialise a fixed object cell layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.
        - num_novel (int): Number of novel object cells.
        - num_reward (int): Number of reward object cells.
        - num_teleport (int): Number of teleport object cells

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """
        self.Agent = Agent

        if not isinstance(self.Agent.Environment, env.OpenField):
            raise ValueError("Environment must be an OpenField to use FixedObjectCells")

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.num_novel = self.params["num_novel"]
        self.num_reward = self.params["num_reward"]
        self.num_teleport = self.params["num_teleport"]

        super().__init__(Agent, self.params)

    @property
    def object_types(self):
        """
        self.object_types

        Obtain object types for each neuron in the layer.

        Returns:
        - (2D np.ndarray): Object type for each neuron in the layer.
        """

        if not hasattr(self, "_object_types"):
            if len(self.input_object_types) == 0:
                raise RuntimeError("No objects found in environment.")

            object_types = list()
            for object_type in np.unique(self.input_object_types):
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]

                if hasattr(self, f"num_{object_name}"):
                    num = getattr(self, f"num_{object_name}")
                    object_types.extend([object_type] * num)

            np.random.shuffle(object_types)

            self._object_types = object_types

        return self._object_types


class WeightedObjectCells(ObjectCells):
    """
    WeightedObjectCells()

    Class extending ObjectCells. Defines a population of neurons that respond to
    certain object types, with a random number of neurons per object type, determined
    using a weighted distribution.

    This class is only compatible with OpenField environments.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation.

    default_params = {
        "n": 10,
        "name": "WeightedObjectCells",
        "novel_weight": 1,  # altogether
        "reward_weight": 5,  # altogether
        "teleport_weight": 0,  # per teleportation object
        "allow_omit_object_types": False,
    }

    List of properties (in addition to riab_neurons.FeedForwardLayer properties):
        • self.object_types

    See ObjectCells for methods.
    """

    default_params = {
        "n": 10,
        "name": "WeightedObjectCells",
        "novel_weight": 1,  # altogether
        "reward_weight": 5,  # altogether
        "teleport_weight": 0,  # per teleportation object
        "allow_omit_object_types": False,
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        WeightedObjectCells(Agent)

        Initialise a weighted object cell layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.
        - place_params (dict): Place cell parameter dictionary.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """
        self.Agent = Agent

        if not isinstance(self.Agent.Environment, env.OpenField):
            raise ValueError(
                "Environment must be an OpenField to use WeightedObjectCells"
            )

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.n = self.params["n"]
        self.novel_weight = self.params["novel_weight"]
        self.reward_weight = self.params["reward_weight"]
        self.teleport_weight = self.params["teleport_weight"]

        super().__init__(Agent, self.params)

    @property
    def object_types(self):
        """
        self.object_types

        Obtain object types for each neuron in the layer.

        Returns:
        - (2D np.ndarray): Object type for each neuron in the layer.
        """

        if not hasattr(self, "_object_types"):
            if len(self.input_object_types) == 0:
                raise RuntimeError("No objects found in environment.")

            weight_dict = dict()
            for object_type in np.unique(self.input_object_types):
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]

                if not hasattr(self, f"{object_name}_weight"):
                    continue

                weight_dict[object_type] = getattr(self, f"{object_name}_weight")

            object_types = util.get_weighted_object_types(
                weight_dict, self.n, self.allow_omit_object_types
            )

            self._object_types = object_types

        return self._object_types

    def _get_num_neurons(self):
        """
        self._get_num_neurons()

        Obtain the number of neurons in the layer.

        Returns:
        - (int): Number of neurons in the layer.
        """

        return self.n


class FixedObjectVectorCells(riab_neurons.ObjectVectorCells):
    """
    FixedObjectVectorCells()

    Class extending riab_neurons.ObjectVectorCells. Defines a population of neurons
    that respond to certain object types, at a particular angle, with a fixed number of
    neurons per object type.

    The reference frame can be allocentric or egocentric. In the latter case the tuning
    angle is relative to the heading direction of the agent.

    This class is only compatible with OpenField environments.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation.

    default_params = {
        "name": "FixedObjectVectorCells",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
        "reference_frame": "egocentric",
    }

    List of properties (in addition to riab_neurons.ObjectVectorCells properties):
        • self.input_object_types
        • self.input_object_locations
        • self.object_types
        • self.neuron_type_dict

    List of methods (in addition to riab_neurons.ObjectVectorCells methods):
        • self.set_tuning_types()
        • self.log_num_neurons_per_object_name()
    """

    default_params = {
        "name": "FixedObjectVectorCells",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
        "reference_frame": "egocentric",
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        FixedObjectVectorCells(Agent)

        Initialise a fixed object vector cell layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().

        """

        self.Agent = Agent

        if not isinstance(self.Agent.Environment, env.OpenField):
            raise ValueError("Environment must be an OpenField to use FixedObjectCells")

        if "n" in params and params["n"] != n:
            raise ValueError(
                "Number of cells should not be passed as a parameter to "
                "ObjectCells. It is set automatically based on "
                "the environment."
            )

        self.num_novel = self.params["num_novel"]
        self.num_reward = self.params["num_reward"]
        self.num_teleport = self.params["num_teleport"]

        n = self._get_num_neurons()

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.params["n"] = n

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No input layers")
            super().__init__(Agent, self.params)

    @property
    def input_object_types(self):
        """
        self.input_object_types

        Obtain all object types from the environment.

        Returns:
        - (1D np.ndarray): All object types.
        """

        if not hasattr(self, "_input_object_types"):
            self._input_object_types = self.Agent.Environment.objects["object_types"]
        return self._input_object_types

    @property
    def input_object_locations(self):
        """
        self.input_object_locations

        Obtain all object locations in the environment.

        Returns:
        - (2D np.ndarray): All object locations (object, location [x, y]).
        """

        if not hasattr(self, "_input_object_locations"):
            self._input_object_locations = self.Agent.Environment.objects["objects"]
        return self._input_object_locations

    @property
    def object_types(self):
        """
        self.object_types

        Obtain object types for each neuron in the layer.

        Returns:
        - (2D np.ndarray): Object type for each neuron in the layer.
        """

        if not hasattr(self, "_object_types"):
            if len(self.input_object_types) == 0:
                raise RuntimeError("No objects found in environment.")

            object_types = list()
            for object_type in np.unique(self.input_object_types):
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]

                if hasattr(self, f"num_{object_name}"):
                    num = getattr(self, f"num_{object_name}")
                    object_types.extend([object_type] * num)

            np.random.shuffle(object_types)

            self._object_types = object_types

        return self._object_types

    @property
    def neuron_type_dict(self):
        """
        self.neuron_type_dict

        Obtain the number of neurons per object name in the layer.

        Returns:
        - (dict): Number of neurons per object name in the layer, with keys and values:
            - object name (str): Number of neurons (int).
        """

        if not hasattr(self, "_neuron_type_dict"):
            object_types, counts = np.unique(self.object_types, return_counts=True)
            _neuron_type_dict = {obj_type: 0 for obj_type in self.input_object_types}
            for object_type, count in zip(object_types, counts):
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]
                if "teleport" in object_name:
                    object_name = "teleport"
                _neuron_type_dict[object_name] += count

            self._neuron_type_dict = _neuron_type_dict

        return self._neuron_type_dict

    def _get_num_neurons(self):
        """
        self._get_num_neurons()

        Obtain the number of neurons in the layer.

        Returns:
        - (int): Number of neurons in the layer.
        """

        return self.n

    def set_tuning_types(self):
        """
        self.set_tuning_types()

        Set the preferred object types for each ObjectVectorCell (OVC).

        This method is called during OVC initialisation.

        Attributes:
        - tuning_types (list): Preferred object type for each OVC.
        """

        tuning_types = list()
        for object_type in np.unique(self.object_types):
            object_name = self.Agent.Environment.object_type_num_to_name_dict[
                object_type
            ]
            if object_name == "novel":
                tuning_types.extend([object_type] * self.num_novel)  # type: ignore[attr-defined]
            elif object_name == "reward":
                tuning_types.extend([object_type] * self.num_reward)  # type: ignore[attr-defined]
            elif "teleport" in object_name:
                tuning_types.extend([object_type] * self.num_teleport)  # type: ignore[attr-defined]

        np.random.shuffle(tuning_types)

        self.tuning_types = tuning_types

    def log_num_neurons_per_object_name(self):
        """
        self.log_num_neurons_per_object_name()

        Log the number of neurons per object name in the layer.
        """

        obj_strs = [
            f"{count} {obj_name} neurons"
            for obj_name, count in self._neuron_type_dict.items()
        ]

        sep = "\n    "
        log_str = f"Layer comprises:{sep}{sep.join(obj_strs)}"
        print(log_str)


class WeightedObjectVectorCells(riab_neurons.ObjectVectorCells):
    """
    WeightedObjectVectorCells()

    Class extending riab_neurons.ObjectVectorCells. Defines a population of neurons
    that respond to certain object types, at a particular angle, with a fixed number of
    neurons per object type.

    The reference frame can be allocentric or egocentric. In the latter case the tuning
    angle is relative to the heading direction of the agent.

    This class is only compatible with OpenField environments.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation.

    default_params = {
        "n": 10,
        "name": "WeightedObjectVectorCells",
        "novel_weight": 1,  # altogether
        "reward_weight": 5,  # altogether
        "teleport_weight": 0,  # per teleportation object
        "reference_frame": "egocentric",
        "allow_omit_object_types": False,
    }

    List of properties (in addition to riab_neurons.ObjectVectorCells properties):
        • self.input_object_types
        • self.input_object_locations
        • self.object_types
        • self.neuron_type_dict

    List of methods (in addition to riab_neurons.ObjectVectorCells methods):
        • self.set_tuning_types()
        • self.log_num_neurons_per_object_name()
    """

    default_params = {
        "n": 10,
        "name": "WeightedObjectVectorCells",
        "novel_weight": 1,  # altogether
        "reward_weight": 5,  # altogether
        "teleport_weight": 0,  # per teleportation object
        "reference_frame": "egocentric",
        "allow_omit_object_types": False,
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):

        self.Agent = Agent

        if not isinstance(self.Agent.Environment, env.OpenField):
            raise ValueError(
                "Environment must be an OpenField to use WeightedObjectCells"
            )

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.n = self.params["n"]
        self.novel_weight = self.params["novel_weight"]
        self.reward_weight = self.params["reward_weight"]
        self.teleport_weight = self.params["teleport_weight"]

        super().__init__(Agent, self.params)

    @property
    def input_object_types(self):
        """
        self.input_object_types

        Obtain all object types from the environment.

        Returns:
        - (1D np.ndarray): All object types.
        """

        if not hasattr(self, "_input_object_types"):
            self._input_object_types = self.Agent.Environment.objects["object_types"]
        return self._input_object_types

    @property
    def input_object_locations(self):
        """
        self.input_object_locations

        Obtain all object locations in the environment.

        Returns:
        - (2D np.ndarray): All object locations (object, location [x, y]).
        """

        if not hasattr(self, "_input_object_locations"):
            self._input_object_locations = self.Agent.Environment.objects["objects"]
        return self._input_object_locations

    @property
    def object_types(self):
        """
        self.object_types

        Obtain object types for each neuron in the layer.

        Returns:
        - (2D np.ndarray): Object type for each neuron in the layer.
        """

        if not hasattr(self, "_object_types"):
            if len(self.input_object_types) == 0:
                raise RuntimeError("No objects found in environment.")

            weight_dict = dict()
            for object_type in np.unique(self.input_object_types):
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]

                if not hasattr(self, f"{object_name}_weight"):
                    continue

                weight_dict[object_type] = getattr(self, f"{object_name}_weight")

            object_types = util.get_weighted_object_types(
                weight_dict, self.n, self.allow_omit_object_types
            )

            self._object_types = object_types

        return self._object_types

    @property
    def neuron_type_dict(self):
        """
        self.neuron_type_dict

        Obtain the number of neurons per object name in the layer.

        Returns:
        - (dict): Number of neurons per object name in the layer, with keys and values:
            - object name (str): Number of neurons (int).
        """

        if not hasattr(self, "_neuron_type_dict"):
            object_types, counts = np.unique(self.object_types, return_counts=True)
            _neuron_type_dict = {obj_type: 0 for obj_type in self.input_object_types}
            for object_type, count in zip(object_types, counts):
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]
                if "teleport" in object_name:
                    object_name = "teleport"
                _neuron_type_dict[object_name] += count

            self._neuron_type_dict = _neuron_type_dict

        return self._neuron_type_dict

    def _get_num_neurons(self):
        """
        self._get_num_neurons()

        Obtain the number of neurons in the layer.

        Returns:
        - (int): Number of neurons in the layer.
        """

        num_neurons = len(self.object_types)
        return num_neurons

    def set_tuning_types(self):
        """
        self.set_tuning_types()

        Set the preferred object types for each ObjectVectorCell (OVC).

        This method is called during OVC initialisation.

        Attributes:
        - tuning_types (list): Preferred object type for each OVC.
        """

        tuning_types = list()
        for object_type in np.unique(self.object_types):
            object_name = self.Agent.Environment.object_type_num_to_name_dict[
                object_type
            ]
            if object_name == "novel":
                tuning_types.extend([object_type] * self.num_novel)  # type: ignore[attr-defined]
            elif object_name == "reward":
                tuning_types.extend([object_type] * self.num_reward)  # type: ignore[attr-defined]
            elif "teleport" in object_name:
                tuning_types.extend([object_type] * self.num_teleport)  # type: ignore[attr-defined]

        np.random.shuffle(tuning_types)

        self.tuning_types = tuning_types

    def log_num_neurons_per_object_name(self):
        """
        self.log_num_neurons_per_object_name()

        Log the number of neurons per object name in the layer.
        """

        obj_strs = [
            f"{count} {obj_name} neurons"
            for obj_name, count in self._neuron_type_dict.items()
        ]

        sep = "\n    "
        log_str = f"Layer comprises:{sep}{sep.join(obj_strs)}"
        print(log_str)
