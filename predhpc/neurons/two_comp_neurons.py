import copy
from typing import TYPE_CHECKING, Any
import warnings

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
        "inhibit_dend": True,
        "inhibit_color": None,
        "inhibit_weight": 0.5,  # multiplied by -1 identity matrix
        "inhibit_activation_params": learning_neurons.STANDARD_SIGMOID_PARAMS,
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
        self.SomaCompartment = learning_neurons.BTSPLayer(self.Agent, self.soma_params)
        self.DendriteCompartment = learning_neurons.BTSPLayer(
            self.Agent, self.dend_params
        )

        if self.SomaCompartment.n != self.n or self.DendriteCompartment.n != self.n:  # type: ignore[attr-defined]
            raise ValueError(
                f"The two compartment layers must have same number of units ({self.n})."
            )

        dend_to_soma_weight = np.eye(self.n) * self.dend_to_soma_weight  # type: ignore[attr-defined]
        self.SomaCompartment.add_input(self.DendriteCompartment, w=dend_to_soma_weight)
        self.SomaCompartment.input_layers_with_no_learning = (
            self.DendriteCompartment.name  # type: ignore[attr-defined]
        )

        soma_to_dend_weight = np.eye(self.n) * self.soma_to_dend_weight  # type: ignore[attr-defined]
        self.DendriteCompartment.add_input(self.SomaCompartment, w=soma_to_dend_weight)
        self.DendriteCompartment.input_layers_with_no_learning = (
            self.SomaCompartment.name  # type: ignore[attr-defined]
        )

        if self.inhibit_dend:  # type: ignore[attr-defined]
            inhibit_params = {
                "name": "SomaInhibitionOfDendrites",
                "n": self.n,
                "activation_params": self.inhibit_activation_params,  # type: ignore[attr-defined]
                "color": self.inhibit_color,  # type: ignore[attr-defined]
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

    def set_learn(self, soma=True, dend=True, inhibit=False):
        if soma:
            self.SomaCompartment.set_learn()
        if dend:
            self.DendriteCompartment.set_learn()
        if inhibit and self.inhibit_dend:  # type: ignore[attr-defined]
            self.DendriteInhibition.set_learn()

    def set_btsp_learn(self, soma=True, dend=True):
        if soma:
            self.SomaCompartment.set_btsp_learn()
        if dend:
            self.DendriteCompartment.set_btsp_learn()

    def set_freeze(self, soma=True, dend=True, inhibit=False):
        if soma:
            self.SomaCompartment.set_freeze()
        if dend:
            self.DendriteCompartment.set_freeze()
        if inhibit and self.inhibit_dend:  # type: ignore[attr-defined]
            self.DendriteInhibition.set_freeze()

    def set_btsp_freeze(self, soma=True, dend=True):
        if soma:
            self.SomaCompartment.set_btsp_freeze()
        if dend:
            self.DendriteCompartment.set_btsp_freeze()

    def update(
        self,
        btsp_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list(),
        dend_first: bool | None = None,
    ):
        if self.inhibit_dend:  # type: ignore[attr-defined]
            self.DendriteInhibition.update()

        if dend_first is None:
            dend_first = self.dend_first  # type: ignore[attr-defined]

        if dend_first:
            self.DendriteCompartment.update(btsp_targets)
            self.SomaCompartment.update(btsp_targets)
        else:
            self.SomaCompartment.update(btsp_targets)
            self.DendriteCompartment.update(btsp_targets)

        return
