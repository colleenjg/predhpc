import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from ratinabox.Neurons import ObjectVectorCells  # type: ignore[import]

from predhpc import env

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


class FixedObjectVectorCells(ObjectVectorCells):
    """Initialises FixedObjectVectorCells(), takes as input a parameter dictionary.
    Any values not provided by the params dictionary are taken from a default
    dictionary below.

    See ObjectVectorCells for details. FixedObjectVectorCells are initialised with a
    fixed number of cells per object type. They are only compatible with ExploreBox
    environments.

    Reference frame can be allocentric or egocentric. In the latter case the tuning
    angle is relative to the heading direction of the agent.

    default_params = {
        "name": "WeightedObjectVectorCell",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
        "reference_frame": "egocentric",
    }
    """

    default_params = {
        "name": "WeightedObjectVectorCell",
        "num_novel": 1,  # altogether
        "num_reward": 5,  # altogether
        "num_teleport": 1,  # per teleportation object
        "reference_frame": "egocentric",
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        self.Agent = Agent

        if not isinstance(self.Agent.Environment, env.ExploreBox):
            raise ValueError(
                "Environment must be an ExploreBox to use WeightedObjectVectorCells"
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
        params["n"] = n

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

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
                tuning_types.append([object_type] * self.num_novel)  # type: ignore[attr-defined]
            elif object_name == "reward":
                tuning_types.append([object_type] * self.num_reward)  # type: ignore[attr-defined]
            elif "teleport" in object_name:
                tuning_types.append([object_type] * self.num_teleport)  # type: ignore[attr-defined]

        np.random.shuffle(tuning_types)

        self.tuning_types = tuning_types


class WeightedObjectVectorCells(ObjectVectorCells):
    """Initialises WeightedObjectVectorCells(), takes as input a parameter dictionary. Any values not provided by the params dictionary are taken from a default dictionary below.

    See ObjectVectorCells for details. WeightedObjectVectorCells are initialised with a random number of cells per object type, based on a weighted distribution per object type.
    They are only compatible with ExploreBox environments.

    Reference frame can be allocentric or egocentric. In the latter case the tuning angle is relative to the heading direction of the agent.

    default_params = {
        "n": 10, #each will be randomly assigned an object type, tuning angle and tuning distance
        "name": "ObjectVectorCell",

    }
    """

    default_params = {
        "n": 10,
        "name": "WeightedObjectVectorCell",
        "novel_weight": 1,  # altogether
        "reward_weight": 5,  # altogether
        "teleport_weight": 0,  # per teleportation object
        "reference_frame": "egocentric",
        "allow_omit_object_types": False,
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        self.Agent = Agent

        if not isinstance(self.Agent.Environment, env.ExploreBox):
            raise ValueError(
                "Environment must be an ExploreBox to use WeightedObjectVectorCells"
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
