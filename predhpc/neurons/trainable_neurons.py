import copy
from typing import TYPE_CHECKING, Any

from matplotlib import pyplot as plt  # type: ignore[import]
import numpy as np
from sklearn.linear_model import Ridge  # type: ignore[import]
from sklearn.multioutput import MultiOutputRegressor  # type: ignore[import]

from predhpc.neurons import riab_neurons, learning_neurons
from predhpc.util import ext_util, plot_util

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]

    try:
        import torch as typing_torch
    except ModuleNotFoundError:
        # hacky workaround, in case torch is not installed
        import numpy as typing_torch  # type: ignore[no-redef]


class RegressionLayer(learning_neurons.LearnLayer):
    """
    RegressionLayer()

    Class extending learning_neurons.LearnLayer, where weights are fitted through
    regression.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 2,
        "name": "RegressionLayer",
        "alpha": 1e2,
        "solver": "auto",
        "fit_intercept": False,
        "max_iter": 1000,
    }

    List of properties (in addition to learning_neurons.LearnLayer properties):
        • self.trainable_biases

    List of methods (in addition to learning_neurons.LearnLayer methods):
        • self.init_model()
        • self.check_Xs_y()
        • self.get_Xs_y()
        • self.fit()
        • self.plot_loss()
        • self.plot_histogram()
    """

    default_params = {
        "n": 2,
        "name": "RegressionLayer",
        "alpha": 1e2,
        "solver": "auto",
        "fit_intercept": False,
        "max_iter": 1000,
        "activation_function": {"activation": "linear"},  # overwrite parent default
    }

    ignored_param_keys = [
        "lr",
    ]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "activation_function": {"activation": "linear"},  # keep regression output
        "biases": None,
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        RegressionLayer(Agent)

        Initialise a regression layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        self.set_learn(False)  # learning handled manually

        self.init_model()

        return

    @property
    def trainable_biases(self) -> bool:
        """
        self.trainable_biases

        Check whether there are trainable biases in the layer.

        Returns:
        - (bool): Whether there are trainable biases in the layer.
        """

        if not hasattr(self, "_trainable_biases"):
            self._trainable_biases = self.fit_intercept  # type: ignore[attr-defined]
        return self._trainable_biases

    def init_model(self):
        """
        self.init_model()

        Initialise the ridge regression model.

        Attributes:
        - self.model (sklearn.linear_model.Ridge): Ridge regression model.
        """

        self.model = Ridge(
            alpha=self.alpha,  # type: ignore[attr-defined]
            solver=self.solver,  # type: ignore[attr-defined]
            max_iter=self.max_iter,  # type: ignore[attr-defined]
            fit_intercept=self.fit_intercept,  # type: ignore[attr-defined]
        )

        if self.n > 1:  # type: ignore[attr-defined]
            self.model = MultiOutputRegressor(self.model)

    def check_Xs_y(
        self,
        Xs: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
        y: np.ndarray[tuple[int], np.dtype[np.float64]],
    ):
        """
        self.check_Xs_y(Xs, y)

        Check that input and output data are of the correct shapes for one another and
        for the layer's neurons and inputs.

        Args:
        - Xs (list): List of 2D input data arrays, each
            with shape (items, layer inputs).
        - y (2D np.ndarray): Output data array (items, outputs).

        Raises:
        - ValueError: If the size of the second dimension of the output data array does
            not match the number of neurons in the layer.
        - ValueError: If the number of input layers does not match the number of input
            data arrays.
        - ValueError: If the number of items in any input data arrays does not match
            the number of items in the output data array.
        - ValueError: If the size of the second dimension of any input data arrays does
            not match the size of its corresponding input layer.
        """

        num_items, num_outputs = y.shape

        if num_outputs != self.n:
            raise ValueError(
                f"Size of the second dimension of y ({num_outputs}) does not match "
                f"number of neurons in the layer ({self.n})."
            )

        if len(Xs) != len(self.inputs):
            raise ValueError(
                f"Number of input layers ({len(self.inputs)}) does not match "
                f"number of elements in Xs ({np.asarray(Xs).shape[1]})."
            )

        for i, (name, input_layer) in enumerate(self.inputs.items()):
            num_items_in, num_in = Xs[i].shape
            if num_items_in != num_items:
                raise ValueError(
                    f"Number of items in the {i}th Xs array ({num_items_in}) does "
                    f"not match number of items in the output data ({num_items})."
                )

            num_weights_in = input_layer["w"].shape[1]
            if num_in != num_weights_in:
                raise ValueError(
                    f"Size of input layer {name} ({num_weights_in}) does not match "
                    f"size of the second dimension of the {i}th Xs array ({num_in})."
                )

    def get_Xs_y(self):
        """
        self.get_Xs_y()

        Get the input and output data for the layer, as the recorded firing rates of
        the input layers, and the recorded positions of the agent.

        Raises:
        - ValueError: If the number of coordinates specifying the agent's position
            does not match the number of neurons in the layer.
        - ValueError: If the number of steps recorded for any input layer does not
            match the number of steps recorded for the agent.

        Returns:
        - Xs (list): List of 2D input data arrays, each
            with shape (steps, layer inputs).
        - y (2D np.ndarray): Output data array (steps, outputs).
        """

        y = np.asarray(self.Agent.history["pos"])

        num_steps, num_coords = y.shape
        if num_coords != self.n:
            raise ValueError(
                f"Cannot infer targets, as the size of the second dimension of the "
                f"agent's position data ({num_coords}) does not match number of "
                f"neurons in the layer ({self.n})."
            )

        Xs = list()
        for i, input_layer in enumerate(self.inputs.values()):
            firingrates = np.asarray(input_layer["layer"].history["firingrate"])
            if len(firingrates) != num_steps:
                raise ValueError(
                    f"Number of steps recorded for the input layer {i} "
                    f"({len(firingrates)}) does not match number of steps recorded "
                    f"for the agent ({num_steps})."
                )
            Xs.append(firingrates)

        if not len(Xs):
            raise ValueError("No input layers provided to the layer.")

        return Xs, y

    def fit(
        self,
        Xs: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] | None = None,
        y: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    ):
        """
        self.fit(Xs, y)

        Fits the input layer weights, and optionally biases, to the data.

        Attributes:
        - last_fit_step (int): Step at which the layer was most recently fitted.

        Args:
        - Xs (list): List of 2D input data arrays, each with shape (steps, neurons).
        - y (2D np.ndarray): Output data array.
        """

        if (Xs is None) != (y is None):
            raise ValueError("Xs and y must both be provided or both be None.")

        if Xs is None:
            Xs, y = self.get_Xs_y()
        else:
            self.check_Xs_y(Xs, y)

        X = np.concatenate(Xs, axis=1)

        self.model.fit(X, y)

        if hasattr(self.model, "estimators_"):
            estimators = self.model.estimators_  # type: ignore[attr-defined]
        else:
            estimators = [self.model]  # type: ignore[attr-defined]

        st = 0
        for input_layer in self.inputs.values():
            num_in = input_layer["w"].shape[1]
            for e, estimator in enumerate(estimators):
                coef = estimator.coef_  # type: ignore[attr-defined]
                input_layer["w"][e] = coef[st : st + num_in]
            st += num_in

        if self.trainable_biases:
            for e, estimator in enumerate(estimators):
                self.biases[e] = estimator.intercept_  # type: ignore[attr-defined]

        self.last_fit_step = self.num_steps_total

        return

    def plot_loss(self, **kwargs) -> plt.Axes:  # type: ignore[override]
        """
        self.plot_loss()

        Plot the loss of the layer over time.

        Keyword args:
        - **kwargs: Keyword arguments passed to LearnLayer.plot_loss().

        Returns:
        - sub_ax (plt.Axes): Subplot with loss plotted.

        Raises:
        - ValueError: If the layer was not trained with targets.
        """

        if self.last_fit_step is not None:
            test_p = 1 - self.last_fit_step / self.num_steps_total
            if "test_p" in kwargs:
                raise ValueError("'test_p' is overridden if the layer has been fitted.")
            kwargs["test_p"] = test_p

        sub_ax = super().plot_loss(**kwargs)

        return sub_ax

    def plot_histogram(  # type: ignore[override]
        self, autosave: bool | None = None, **kwargs
    ) -> plt.Axes:
        """
        self.plot_histogram()

        Plot a histogram of the firing rates of the layer.  If the layer has been
        fitted, the histogram is plotted starting from the most recent fitting step.

        Args:
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to to LearnLayer.plot_histogram().

        Returns:
        - sub_ax (plt.Axes): Subplot withhistogram plotted.
        """

        fit_str = " (unfitted)"
        if self.last_fit_step is not None:
            if "t_start" in kwargs.keys():
                raise ValueError(
                    "'t_start' is overridden if the layer has been fitted."
                )
            elif self.last_fit_step == self.num_steps_total:
                raise ValueError("Layer has not been updated since last fit step.")

            kwargs["t_start"] = self.history["t"][self.last_fit_step]
            fit_str = " (fitted)"

        sub_ax = super().plot_histogram(autosave=False, **kwargs)

        sub_ax.set_xlabel(f"Firing rate{fit_str}")

        fig = sub_ax.get_figure()
        plot_util.save_figure(fig, f"{self.name}_firing_rate_histogram", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class TorchLayer(learning_neurons.LearnLayer):
    """
    TorchLayer()

    Class extending learning_neurons.LearnLayer, where weights are fitted through
    backpropagation.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 2,
        "name": "TorchLayer",
        "biases": None,
        "sigmoid": True,  # use instead of "activation_params"
        "use_targets": False,
    }

    List of properties (in addition to learning_neurons.LearnLayer properties):
        • self.layer

    List of methods (in addition to learning_neurons.LearnLayer methods):
        • self.update_weights()
        • self.add_input()
        • self.plot_loss()
        • self.plot_histogram()
    """

    default_params = {
        "n": 2,
        "name": "TorchLayer",
        "biases": None,
        "sigmoid": True,  # use instead of "activation_params"
        "use_targets": False,
    }

    ignored_param_keys = [
        "activation_params",
        "lr",
        "init_weights_zero",
        "w_init_scale",
    ]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        TorchLayer(Agent)

        Initialise a torch layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.
        - train_steps (list): Steps at which the layer was trained.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        self.set_learn(False)  # learning handled manually

        self.train_steps = list()  # type: list[int]

        return

    @property
    def layer(self) -> "typing_torch.nn.Sequential":
        """
        self.layer

        Obtain the torch layer for the neuron layer.

        Returns:
        - (torch.nn.Sequential): Torch layer for the neuron layer.
        """

        if not hasattr(self, "_layer"):
            if len(self.inputs) == 0:
                raise ValueError("No input layers provided.")
            n_in = sum([X["w"].shape[1] for X in self.inputs.values()])

            import torch

            n_out = self.n  # type: ignore[attr-defined]
            self._layer = torch.nn.Sequential(
                torch.nn.Linear(n_in, n_out, bias=self.trainable_biases), self.activ()
            )

            self.update_weights()  # match weights between torch and neuron layer

        return self._layer

    def add_fixed_params(self, params: dict[str, Any] = dict()) -> dict[str, Any]:
        """
        self.add_fixed_params()

        Set fixed parameters.

        Args:
        - params (dict, optional): Environment parameters. Default is dict().

        Returns:
        - params (dict): Neuron layer parameters with fixed parameters added.
        """

        params = super().add_fixed_params(params)

        import torch

        params = copy.copy(
            params
        )  # avoid deep copy to preserve reference to input layers

        if params["sigmoid"]:
            params["activation_function"] = ext_util.get_standard_sigmoid_params()
            self.activ = torch.nn.Sigmoid  # type: ignore[assignment]
        else:
            params["activation_function"] = {"activation": "relu"}
            self.activ = torch.nn.ReLU  # type: ignore[assignment]

        return params

    def update_weights(self):
        """
        self.update_weights()

        Update the weights and biases of the neuron layer to match the torch layer
        weights and biases. Records steps as a layer training step.

        Attributes:
        - train_steps (list): Steps at which the layer was trained.
        """

        if not hasattr(self, "_layer"):
            raise RuntimeError("Layer not initialised.")

        st = 0
        for input_layer in self.inputs.values():
            n_in = len(input_layer["w"])
            input_layer["w"][:] = self.layer[0].weight[st : st + n_in].detach().numpy()  # type: ignore[has-method]
            st += n_in

        if self.trainable_biases:
            self.biases[:] = self.layer[0].bias.detach().numpy()  # type: ignore[callable]

        self.train_steps.append(self.num_steps_total)

        return

    def add_input(self, input_layer: riab_neurons.Neurons, **kwargs):
        """
        self.add_input(input_layer)

        Add an input layer to the TorchLayer.

        Args:
        - input_layer (riab_neurons.Neurons): Input layer to add.

        Keyword args:
        - **kwargs: Keyword arguments passed to LearnLayer.add_input().

        Raises:
        - RuntimeError: If the neuron layer's torch layer has already been initialised.
        """

        if hasattr(self, "_layer"):
            raise RuntimeError(
                "Cannot add input layer after self.layer has been initialised."
            )

        super().add_input(input_layer, **kwargs)

    def plot_loss(self, **kwargs) -> plt.Axes:  # type: ignore[override]
        """
        self.plot_loss()

        Plot the loss of the layer over time.

        Keyword args:
        - **kwargs: Keyword arguments passed to LearnLayer.plot_loss().

        Returns:
        - sub_ax (plt.Axes): Subplot with loss plotted.

        Raises:
        - ValueError: If the layer was not trained with targets.
        """

        if len(self.train_steps) != 0:
            test_p = 1 - self.train_steps[-1] / self.num_steps_total
            if "test_p" in kwargs:
                raise ValueError("'test_p' is overridden if the layer has been fitted.")
            kwargs["test_p"] = test_p

        sub_ax = super().plot_loss(**kwargs)

        return sub_ax

    def plot_histogram(  # type: ignore[override]
        self, autosave: bool | None = None, **kwargs
    ) -> plt.Axes:
        """
        self.plot_histogram()

        Plot a histogram of the layer's firingrates. If the layer has been trained, the
        histogram is plotted starting from the most recent training step.

        Args:
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to LearnLayer.plot_histogram().

        Returns:
        - sub_ax (plt.Axes): Subplot with histogram plotted.
        """

        train_str = " (untrained)"
        if len(self.train_steps) != 0:
            if "t_start" in kwargs.keys():
                raise ValueError(
                    "'t_start' is overridden if the layer has been trained."
                )
            elif self.train_steps[-1] == self.num_steps_total:
                raise ValueError("Layer has not been updated since last training step.")

            kwargs["t_start"] = self.history["t"][self.train_steps[-1]]
            train_str = " (trained)"

        sub_ax = super().plot_histogram(autosave=False, **kwargs)

        sub_ax.set_xlabel(f"Firing rate{train_str}")

        fig = sub_ax.get_figure()
        plot_util.save_figure(fig, f"{self.name}_firing_rate_histogram", save=autosave)  # type: ignore[attr-defined]

        return sub_ax
