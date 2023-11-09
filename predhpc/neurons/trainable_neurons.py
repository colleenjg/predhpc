import copy
from typing import TYPE_CHECKING, Any

from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import figure as mpl_figure
import numpy as np
from sklearn.linear_model import Ridge  # type: ignore[import]
from sklearn.multioutput import MultiOutputRegressor  # type: ignore[import]
from ratinabox.Neurons import Neurons  # type: ignore[import]

from predhpc import util
from predhpc.neurons import learning_neurons

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]

    try:
        import torch as typing_torch
    except ModuleNotFoundError:
        # hacky workaround, in case torch is not installed
        import numpy as typing_torch  # type: ignore[no-redef]


class RegressionLayer(learning_neurons.LearnLayer):
    """This trained class defines a population of neurons that fit their weights
    through regression.
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
        "name": "RegressionLayer",
        "alpha": 1e2,
        "solver": "auto",
        "fit_intercept": False,
        "max_iter": 1000,
    }

    ignored_param_keys = [
        "lr",
    ]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "activation_params": {"activation": "linear"},  # keep regression output exactly
        "biases": None,
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """Initialise RegressionLayer(), takes as input a parameter
        dictionary. Any values not provided by the params dictionary are
        taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        self.init_model()

        return

    @property
    def trainable_biases(self) -> bool:
        if not hasattr(self, "_trainable_biases"):
            self._trainable_biases = self.fit_intercept  # type: ignore[attr-defined]
        return self._trainable_biases

    def init_model(self):
        """Initialise the model."""

        alpha, solver = self.alpha, self.solver  # type: ignore[attr-defined]
        max_iter, fit_intercept = self.max_iter, self.fit_intercept  # type: ignore[attr-defined]

        self.model = Ridge(
            alpha=alpha,
            solver=solver,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
        )

        if self.n > 1:  # type: ignore[attr-defined]
            self.model = MultiOutputRegressor(self.model)

    def fit(
        self,
        Xs: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
        y: np.ndarray[tuple[int], np.dtype[np.float64]],
    ):
        """Fits the layer weights, and optionally biases, to the data."""

        X_len = [len(X) for X in Xs]
        X = np.concatenate(Xs, axis=1)

        self.model.fit(X, y)

        if hasattr(self.model, "estimators_"):
            estimators = self.model.estimators_  # type: ignore[attr-defined]
        else:
            estimators = [self.model]  # type: ignore[attr-defined]

        st = 0
        for i, input_layer in enumerate(self.inputs.values()):
            for e, estimator in enumerate(estimators):
                coef = estimator.coef_  # type: ignore[attr-defined]
                input_layer["w"][e] = coef[st : st + X_len[i]]
            st += X_len[i]

        if self.trainable_biases:
            for e, estimator in enumerate(estimators):
                self.biases[e] = estimator.intercept_  # type: ignore[attr-defined]

        self.last_fit_step = self.num_steps_total

        return

    def plot_loss(self, **loss_kwargs) -> tuple[mpl_figure.Figure, plt.Axes]:  # type: ignore[override]
        """
        Plot the loss of the layer over time.

        Kwargs:
            loss_kwargs: Keyword arguments passed to super().plot_loss().

        Returns:
            fig, ax: Figure and axis of the plot.

        Raises:
            ValueError: If the layer was not trained with targets.

        Example:
            >>> fig, ax = layer.plot_loss() # plot the loss of the layer
        """

        test_p = None
        if self.last_fit_step is not None:
            test_p = 1 - self.last_fit_step / self.num_steps_total

        loss_kwargs["test_p"] = test_p

        fig, ax = super().plot_loss(**loss_kwargs)

        return fig, ax

    def plot_histogram(  # type: ignore[override]
        self, autosave: bool | None = None, **loss_kwargs
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the firing rate histogram of the layer.

        Args:
            autosave (bool, optional): Whether to save the figure. Defaults to None.

        Keyword Args:
            loss_kwargs: Keyword arguments passed to super().plot_histogram().

        Returns:
            fig, ax: Figure and axis of the plot.
        """

        fit_str = " (unfitted)"
        if self.last_fit_step is not None:
            loss_kwargs["t_start"] = self.history["t"][self.last_fit_step]
            fit_str = " (fitted)"

        fig, ax = super().plot_histogram(autosave=False, **loss_kwargs)

        ax.set_xlabel(f"Firing rate{fit_str}")

        util.save_figure(fig, f"{self.name}_firing_rate_histogram", save=autosave)  # type: ignore[attr-defined]

        return fig, ax


class TorchLayer(learning_neurons.LearnLayer):
    """This trained class defines a population of neurons that fit their weights
    through backprop, using pytorch.
    This class is a subclass of Neurons() and inherits its properties/plotting
    functions.

    Must be initialised with an Agent, and a "params" dictionary, including input
    layers.

    List of functions:
        • get_state()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()
    """

    default_params = {
        "n": 2,
        "name": "TorchLayer",
        "biases": None,
        "sigmoid": True,
        "use_targets": False,
    }

    ignored_param_keys = [
        "lr",
        "init_weights_zero",
        "w_init_scale",
    ]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """Initialise TorchLayer(), takes as input a parameter
        dictionary. Any values not provided by the params dictionary are
        taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        self.train_steps = list()  # type: list[int]

        return

    def add_fixed_params(self, params: dict[str, Any] = dict()) -> dict[str, Any]:
        """Sets fixed parameters."""

        params = super().add_fixed_params(params)

        import torch

        params = copy.copy(
            params
        )  # avoid deep copy to preserve reference to input layers

        # ignore activation params, even if they are passed
        if params["sigmoid"]:
            params["activation_params"] = {"activation": "relu"}
            self.activ = torch.nn.ReLU  # type: ignore[assignment]
        else:
            params["activation_params"] = util.get_standard_sigmoid_params()
            self.activ = torch.nn.Sigmoid  # type: ignore[assignment]

        return params

    def update_weights(self):
        """Update the weights of the layer."""

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

    @property
    def layer(self) -> "typing_torch.nn.Sequential":
        """Retrieves Torch layer."""

        if not hasattr(self, "_layer"):
            if len(self.inputs) == 0:
                raise ValueError("No input layers provided.")
            n_in = sum([X["w"].shape[1] for X in self.inputs.values()])

            import torch

            n_out = self.n  # type: ignore[attr-defined]
            self._layer = torch.nn.Sequential(
                torch.nn.Linear(n_in, n_out, bias=self.trainable_biases), self.activ()
            )

            self.update_weights()

        return self._layer

    def add_input(self, input_layer: Neurons, **kwargs):
        """Add an input layer to the TorchLayer.

        Args:
            input_layer (_type_): _description_
        """

        if hasattr(self, "_layer"):
            raise AttributeError(
                "Cannot add input layer after self.layer has been initialised."
            )

        super().add_input(input_layer, **kwargs)

    def plot_loss(self, **loss_kwargs) -> tuple[mpl_figure.Figure, plt.Axes]:  # type: ignore[override]
        """
        Plot the loss of the layer over time.

        Kwargs:
            loss_kwargs: Keyword arguments passed to super().plot_loss().

        Returns:
            fig, ax: Figure and axis of the plot.

        Raises:
            ValueError: If the layer was not trained with targets.

        Example:
            >>> fig, ax = layer.plot_loss() # plot the loss of the layer
        """

        test_p = None
        if len(self.train_steps) != 0:
            test_p = 1 - self.train_steps[-1] / self.num_steps_total

        loss_kwargs["test_p"] = test_p

        fig, ax = super().plot_loss(**loss_kwargs)

        return fig, ax

    def plot_histogram(  # type: ignore[override]
        self, autosave: bool | None = None, **loss_kwargs
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the firing rate histogram of the layer.

        Args:
            autosave (bool, optional): Whether to save the figure. Defaults to None.

        Keyword Args:
            loss_kwargs: Keyword arguments passed to super().plot_histogram().

        Returns:
            fig, ax: Figure and axis of the plot.
        """

        train_str = " (untrained)"
        if len(self.train_steps) != 0:
            loss_kwargs["t_start"] = self.history["t"][self.train_steps[-1]]
            train_str = " (trained)"

        fig, ax = super().plot_histogram(autosave=False, **loss_kwargs)

        ax.set_xlabel(f"Firing rate{train_str}")

        util.save_figure(fig, f"{self.name}_firing_rate_histogram", save=autosave)  # type: ignore[attr-defined]

        return fig, ax
