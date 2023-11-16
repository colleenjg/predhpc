import copy
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
from ratinabox.Neurons import FeedForwardLayer, ObjectVectorCells, PlaceCells  # type: ignore[import]

from predhpc import env

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


class ObjectCells(FeedForwardLayer):
    """ """

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

        self._set_place_inputs()
        self._add_place_inputs()

    @property
    def input_object_types(self):
        if not hasattr(self, "_input_object_types"):
            self._input_object_types = self.Agent.Environment.objects["object_types"]
        return self._input_object_types

    @property
    def input_object_locations(self):
        if not hasattr(self, "_input_object_locations"):
            self._input_object_locations = self.Agent.Environment.objects["objects"]
        return self._input_object_locations

    @property
    def object_types(self):
        if not hasattr(self, "_object_types"):
            if len(self.input_object_types) == 0:
                raise RuntimeError("No objects found in environment.")

            object_types = np.unique(self.input_object_types)
            np.random.shuffle(object_types)
            self._object_types = object_types
        return self._object_types

    @property
    def place_cell_input_weights(self):
        if not hasattr(self, "_place_cell_input_weights"):
            place_cell_input_weights = np.zeros((self.n, self.PlaceCellInputs.n))

            for i, obj_type in enumerate(self.input_object_types):
                if obj_type not in self.object_types:
                    continue
                for j in np.where(np.asarray(self.object_types) == obj_type)[0]:
                    place_cell_input_weights[j, i] = 1
            self._place_cell_input_weights = place_cell_input_weights

        return self._place_cell_input_weights

    def _get_num_neurons(self):
        num_neurons = len(self.object_types)
        return num_neurons

    def _set_place_inputs(self):
        self.place_params["place_cell_centres"] = self.input_object_locations
        self.PlaceCellInputs = PlaceCells(self.Agent, params=self.place_params)

    def _add_place_inputs(self):
        self.add_input(
            input_layer=self.PlaceCellInputs,
            w=self.place_cell_input_weights,
        )

    def update(self):
        self.PlaceCellInputs.update()
        super().update()


class FixedObjectCells(ObjectCells):
    """Initialises FixedObjectCells(), takes as input a parameter dictionary.
    Any values not provided by the params dictionary are taken from a default
    dictionary below.

    See ObjectCells for details. FixedObjectCells are initialised with a
    fixed number of cells per object type. They are only compatible with OpenField
    environments.

    default_params = {
        "name": "FixedObjectCells",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
    }
    """

    default_params = {
        "name": "FixedObjectCells",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
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
        if not hasattr(self, "_object_types"):
            if len(self.input_object_types) == 0:
                raise RuntimeError("No objects found in environment.")

            object_types = list()
            num_novel_object_neurons = 0
            num_reward_object_neurons = 0
            num_teleport_object_neurons = 0
            for object_type in np.unique(self.input_object_types):
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]
                new_objects = list()

                if object_name == "novel":
                    new_objects = [object_type] * self.num_novel  # type: ignore[attr-defined]
                    num_novel_object_neurons += len(new_objects)
                elif object_name == "reward":
                    new_objects = [object_type] * self.num_reward  # type: ignore[attr-defined]
                    num_reward_object_neurons += len(new_objects)
                elif "teleport" in object_name:
                    new_objects = [object_type] * self.num_teleport  # type: ignore[attr-defined]
                    num_teleport_object_neurons += len(new_objects)

                object_types.extend(new_objects)  # type: ignore[attr-defined]

            if num_novel_object_neurons != self.num_novel_object_neurons:
                raise RuntimeError("Wrong number of novel object neurons identified.")
            if num_reward_object_neurons != self.num_reward_object_neurons:
                raise RuntimeError("Wrong number of reward object neurons identified.")
            if num_teleport_object_neurons != self.num_teleport_object_neurons:
                raise RuntimeError(
                    "Wrong number of teleport object neurons identified."
                )

            np.random.shuffle(object_types)

            self._object_types = object_types

        return self._object_types

    def _get_num_neurons(self):
        (
            num_novel_objects,
            num_reward_objects,
            num_teleport_objects,
        ) = self.Agent.Environment.get_number_object_types_split()
        self.num_novel_object_neurons = bool(num_novel_objects) * self.num_novel
        self.num_reward_object_neurons = bool(num_reward_objects) * self.num_reward
        self.num_teleport_object_neurons = num_teleport_objects * self.num_teleport
        n = int(
            self.num_novel_object_neurons
            + self.num_reward_object_neurons
            + self.num_teleport_object_neurons
        )

        return n


class WeightedObjectCells(ObjectCells):
    """Initialises WeightedObjectCells(), takes as input a parameter dictionary.
    Any values not provided by the params dictionary are taken from a default dictionary below.

    See ObjectCells for details. WeightedObjectCells are initialised with a random
    number of cells per object type, based on a weighted distribution per object type.
    They are only compatible with OpenField environments.

    default_params = {
        "n": 10, #each will be randomly assigned an object type, tuning angle and tuning distance
        "name": "ObjectVectorCell",

    }
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
        if not hasattr(self, "_object_types"):
            if len(self.input_object_types) == 0:
                raise RuntimeError("No objects found in environment.")

            unique_input_object_types = np.unique(self.input_object_types)

            object_weights = list()
            for object_type in unique_input_object_types:
                object_name = self.Agent.Environment.object_type_num_to_name_dict[
                    object_type
                ]
                if object_name == "novel":
                    object_weights.append(self.novel_weight)  # type: ignore[attr-defined]
                elif object_name == "reward":
                    object_weights.append(self.reward_weight)  # type: ignore[attr-defined]
                elif "teleport" in object_name:
                    object_weights.append(self.teleport_weight)  # type: ignore[attr-defined]

            object_weights = np.asarray(object_weights).astype(float)
            object_weights /= object_weights.sum()

            n = self.n  # type: ignore[attr-defined]
            rand_n = n
            base_object_types = list()
            if not self.allow_omit_object_types:  # type: ignore[attr-defined]
                base_object_types = np.unique(
                    [
                        o
                        for o, p in zip(unique_input_object_types, object_weights)
                        if p > 0
                    ]
                )
                if n < len(base_object_types):
                    raise RuntimeError(
                        "Not enough cells to represent all object types. Must increase n "
                        "or set allow_omit_object_types to True, when initializing "
                        "the WeightedObjectCells instance."
                    )
                rand_n = n - len(base_object_types)

            object_types = np.random.choice(
                unique_input_object_types,
                replace=True,
                size=(rand_n,),
                p=object_weights,
            )

            object_types = np.concatenate((object_types, base_object_types))
            np.random.shuffle(object_types)

            self._object_types = object_types

        return self._object_types

    def _get_num_neurons(self):
        return self.n


class FixedObjectVectorCells(ObjectVectorCells):
    """Initialises FixedObjectVectorCells(), takes as input a parameter dictionary.
    Any values not provided by the params dictionary are taken from a default
    dictionary below.

    See ObjectVectorCells for details. FixedObjectVectorCells are initialised with a
    fixed number of cells per object type. They are only compatible with OpenField
    environments.

    Reference frame can be allocentric or egocentric. In the latter case the tuning
    angle is relative to the heading direction of the agent.

    default_params = {
        "name": "FixedObjectVectorCells",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
        "reference_frame": "egocentric",
    }
    """

    default_params = {
        "name": "FixedObjectVectorCells",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
        "reference_frame": "egocentric",
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        self.Agent = Agent

        if not isinstance(self.Agent.Environment, env.OpenField):
            raise ValueError(
                "Environment must be an OpenField to use FixedObjectVectorCells"
            )

        (
            num_novel,
            num_reward,
            num_teleport,
        ) = self.Agent.Environment.get_number_object_types_split()
        n = int(bool(num_novel)) + int(bool(num_reward)) + num_teleport

        if "n" in params and params["n"] != n:
            raise ValueError(
                "Number of cells should not be passed as a parameter to "
                "WeightedObjectVectorCells. It is set automatically based on "
                "the environment."
            )

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)
        self.params["n"] = n

        super().__init__(Agent, self.params)

    def set_tuning_types(self):
        """Sets the preferred object types for each OVC.

        This is called automatically when the OVCs are initialised.
        """

        self.object_types = self.Agent.Environment.objects["object_types"]

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


class WeightedObjectVectorCells(ObjectVectorCells):
    """Initialises WeightedObjectVectorCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

    See ObjectVectorCells for details. WeightedObjectVectorCells are initialised with a random number of cells per object type, based on a weighted distribution per object type.
    They are only compatible with OpenField environments.

    Reference frame can be allocentric or egocentric. In the latter case the tuning angle is relative to the heading direction of the agent.

    default_params = {
        "n": 10, #each will be randomly assigned an object type, tuning angle and tuning distance
        "name": "ObjectVectorCell",

    }
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
                "Environment must be an OpenField to use WeightedObjectVectorCells"
            )

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

    def set_tuning_types(self):
        """Sets the preferred object types for each OVC.

        This is called automatically when the OVCs are initialised.
        """

        self.object_types = self.Agent.Environment.objects["object_types"]
        object_types = np.unique(self.object_types)

        object_weights = list()
        for object_type in object_types:
            object_name = self.Agent.Environment.object_type_num_to_name_dict[
                object_type
            ]
            if object_name == "novel":
                object_weights.append(self.novel_weight)  # type: ignore[attr-defined]
            elif object_name == "reward":
                object_weights.append(self.reward_weight)  # type: ignore[attr-defined]
            elif "teleport" in object_name:
                object_weights.append(self.teleport_weight)  # type: ignore[attr-defined]

        object_weights = np.asarray(object_weights).astype(float)
        object_weights /= object_weights.sum()

        n = self.n  # type: ignore[attr-defined]
        rand_n = n
        base_tuning_types = list()
        if not self.allow_omit_object_types:  # type: ignore[attr-defined]
            base_tuning_types = np.unique(
                [o for o, p in zip(object_types, object_weights) if p > 0]
            )
            if n < len(base_tuning_types):
                raise RuntimeError(
                    "Not enough cells to represent all object types. Must increase n "
                    "or set allow_omit_object_types to True, when initializing "
                    "the WeightedObjectVectorCells instance."
                )
            rand_n = n - len(base_tuning_types)

        tuning_types = np.random.choice(
            object_types, replace=True, size=(rand_n,), p=object_weights
        )

        tuning_types = np.concatenate((tuning_types, base_tuning_types))
        np.random.shuffle(tuning_types)

        self.tuning_types = tuning_types
