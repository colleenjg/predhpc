import copy
from typing import TYPE_CHECKING, Any, Sequence

from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import markers
import numpy as np

from ratinabox import utils as rutils  # type: ignore[import]

from predhpc import util, plot_util, params_util
from predhpc.neurons import riab_neurons

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


class SmoothFeedForwardLayer(riab_neurons.FeedForwardLayer):
    """
    SmoothFeedForwardLayer()

    Class extending riab_neurons.FeedForwardLayer. Defines a population of neurons
    that receive feedforward input that is smoothed.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 10,
        "activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_FUNCTION,
        "name": "SmoothFeedForwardLayer",
        "input_filter_tau": 0.1,  # in sec
        "input_trend_tau": None,  # in sec
    }

    See riab_neurons.FeedForwardLayer for properties.

    List of methods (in addition to riab_neurons.FeedForwardLayer methods):
        • self.add_input()
        • self.get_filter_tau()
        • self.save_to_history()
        • self.get_state()
        • self.update_filtered_inputs()
        • self.update()
        • self.plot_activation_function()
        • self.plot_firingrate_distribution()
        • self.plot_filtered()
    """

    default_params = {
        "n": 10,
        "activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_FUNCTION,
        "name": "SmoothFeedForwardLayer",
        "input_filter_tau": 0.1,  # in sec
        "input_trend_tau": None,  # in sec
    }

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        SmoothFeedForwardLayer(Agent)

        Initialise a feedforward layer with smoothed inputs.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.
        - activation_params (dict): Activation function parameters.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.activation_params = self.params[
            "activation_function"
        ]  # store activation parameters
        super().__init__(Agent, self.params)

        return

    def add_input(self, input_layer: riab_neurons.Neurons, **kwargs):
        """
        self.add_input(input_layer)

        Add an input layer.

        Also add variables tracking filtered inputs and trends for this input, and
        history keys to store these values across steps.

        Args:
        - input_layer (riab_neurons.Neurons): Neuron layer to add as input.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.add_input().
        """

        super().add_input(input_layer, **kwargs)

        if self.input_filter_tau or self.input_trend_tau:  # type: ignore[attr-defined]
            name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]
            self.inputs[name_in]["filtered_inputs"] = np.full(n_in, np.nan)
            self.inputs[name_in]["filtered_trends"] = np.zeros(n_in)

    def get_filter_tau(self, filter_tau: float | None = None) -> float:
        """
        self.get_filter_tau()

        Obtain the exponential filter time constant parameter.

        Args:
        - filter_tau (float, optional): Filter time constant. If None, the agent's
            step size is used. Default is None.

        Raises:
        - ValueError: If the filter time constant is smaller than the Agent time step.

        Returns:
        - filter_tau (float): Filter time constant.
        """

        if filter_tau is None:
            filter_tau = float(self.Agent.dt)

        elif filter_tau < self.Agent.dt:
            raise ValueError(
                f"'filter_tau' ({filter_tau}) cannot be smaller than "
                f"self.Agent.dt ({self.Agent.dt})."
            )

        return filter_tau

    def save_to_history(self):
        """
        self.save_to_history()

        Save the current state of the layer to the history, including the filtered
        inputs and trends for input layers.
        """

        super().save_to_history()

        if self.input_filter_tau or self.input_trend_tau:  # type: ignore[attr-defined]
            for name, input_layer in self.inputs.items():
                for key in ["inputs", "trends"]:
                    self.history[f"filtered_{key}"][name].append(
                        input_layer[f"filtered_{key}"].tolist()
                    )

    def get_state(self, evaluate_at="last", max_recurrence=None, **kwargs):
        """
        self.get_state()

        Obtain the firing rate of the layer. Adapted from FeedForward.get_state().

        Args:
        - evaluate_at (str, optional). Default is 'last'.
        - max_recurrence: The maximum number of time get_state() recursively calls
            recurrent inputs (prevents infinite recursion error). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForward.get_state().

        Returns:
           (1D np.ndarray): Array of firing rates
        """

        if evaluate_at == "last":
            V = np.zeros(self.n)
        elif evaluate_at == "all":
            V = np.zeros(
                (self.n, self.Agent.Environment.flattened_discrete_coords.shape[0])
            )
        else:
            V = np.zeros((self.n, kwargs["pos"].shape[0]))

        for inputlayer in self.inputs.values():
            pass_max_recurrence = max_recurrence
            if max_recurrence is not None and inputlayer["recurrent"]:
                if max_recurrence <= 0:
                    continue
                pass_max_recurrence = max_recurrence - 1
            w = inputlayer["w"]
            if evaluate_at == "last":
                if self.input_filter_tau or self.input_trend_tau:
                    I = inputlayer["filtered_inputs"]
                    if not np.isfinite(I).all():
                        I = inputlayer["layer"].firingrate
                else:
                    I = inputlayer["layer"].firingrate
            else:  # recursive call
                I = inputlayer["layer"].get_state(
                    evaluate_at=evaluate_at,
                    max_recurrence=pass_max_recurrence,
                    **kwargs,
                )

            inputlayer["I"] = I
            V += np.matmul(w, I)

        biases = self.biases
        if biases.shape != V.shape:
            biases = biases.reshape((-1, 1))
        V += biases

        firingrate = self.activation_function(V, deriv=False)
        # saves current copy of activation derivative at firing rate (useful for learning rules)
        if (
            evaluate_at == "last"
        ):  # save copy of the firing rate through the dervative of the activation function
            self.firingrate_prime = self.activation_function(V, deriv=True)

        return firingrate

    def update_filtered_inputs(
        self,
        filter_tau: float | None = None,
        trend_tau: float | None = None,
        filter_key: str = "filtered_inputs",
    ):
        """
        self.update_filtered_inputs()

        Update the filtered inputs for the layer.

        Args:
        - filter_tau (float, optional): Filter time constant. Default is None.
        - trend_tau (float, optional): Trend time constant. Default is None.
        - filter_key (str, optional): Key of the filter to update.
            Default is "filtered_inputs".
        """

        filter_tau = self.get_filter_tau(filter_tau)
        effective_filter_tau = filter_tau / self.Agent.dt
        filter_alpha = 1 - np.exp(-1 / effective_filter_tau)

        trend_key = filter_key.replace("_inputs", "_trends")
        trend_tau = self.get_filter_tau(trend_tau)
        effective_trend_tau = trend_tau / self.Agent.dt
        trend_alpha = 1 - np.exp(-1 / effective_trend_tau)

        for input_layer in self.inputs.values():
            I_t1 = input_layer["layer"].firingrate
            X_t = input_layer[filter_key]
            if not np.isfinite(X_t).all():
                X_t = I_t1
            T_t = input_layer[trend_key]

            input_layer[filter_key] = filter_alpha * I_t1 + (1 - filter_alpha) * (
                X_t + T_t
            )
            if trend_tau > self.Agent.dt:
                input_layer[trend_key] = (
                    trend_alpha * (input_layer[filter_key] - X_t)
                    + (1 - trend_alpha) * T_t
                )

        return

    def update(self):
        """
        self.update()

        Update the layer, and filtered inputs.
        """

        if self.input_filter_tau or self.input_trend_tau:
            self.update_filtered_inputs(
                self.input_filter_tau,
                self.input_trend_tau,
                filter_key="filtered_inputs",
            )
        super().update()

    def plot_activation_function(self, min_input_fr=-15, max_input_fr=15, sub_ax=None):
        """
        self.plot_activation_function()

        Plot the activation function of the layer.

        Args:
        - min_input_fr (int, optional): Minimum input firing rate to plot from.
            Default is -15.
        - max_input_fr (int, optional): Maximum input firing rate to plot to.
            Default is 15.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with activation function plotted.
        """

        sub_ax = plot_util.plot_activation_function(
            self.activation_function,
            min_input_fr=min_input_fr,
            max_input_fr=max_input_fr,
            sub_ax=sub_ax,
            color=self.color,
        )

        sub_ax.set_title("Activation function")

        return sub_ax

    def plot_firingrate_distribution(self, sub_ax=None, bins=50):
        """
        self.plot_firingrate_distribution()

        Plot the firing rate distribution of the layer as a histogram.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - bins (int, optional): Number of histogram bins. Default is 50.

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate distribution plotted.
        """

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(4, 2))

        firingrates = np.asarray(self.history["firingrate"]).ravel()
        sub_ax.hist(firingrates, bins=bins, color=self.color, alpha=0.6, density=True)

        sub_ax.axvline(0, color="k", lw=1, ls="dashed")

        sub_ax.set_xlabel("Firing rate")
        sub_ax.set_ylabel("Density")
        sub_ax.set_title("Firing rate distribution")

        sub_ax.spines[["right", "top"]].set_visible(False)

        return sub_ax

    def plot_filtered(
        self,
        input_layer_name: str | None = None,
        filter_key: str = "filtered_inputs",
        t_start: float | None = None,
        t_end: float | None = None,
        title: str | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        sub_ax: plt.Axes | None = None,
        autosave: bool | None = None,
    ) -> tuple[plt.Axes, np.ndarray[tuple[int], np.dtype[np.float64]]]:
        """
        self.plot_filtered()

        Plot the filtered inputs or firingrates of the layer.

        Args:
        - input_layer_name (str, optional): Name of the input layer to plot. If None,
            the layer itself is used.
        - filter_key (str, optional): Key of the filtered inputs to plot.
            Default is "filtered_inputs".
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - title (str, optional): Title of the plot. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Raises:
        - ValueError: If the input layer is not found.

        Returns:
        - sub_ax (plt.Axes): Subplot with filtered inputs or firingrates plotted.
        - t (1D np.ndarray): Times plotted.
        """

        t, startid, endid = self.get_plotting_times(t_start, t_end)

        if input_layer_name is None:
            layer = self
        else:
            if input_layer_name not in self.inputs.keys():
                raise ValueError(
                    f"Input layer '{input_layer_name}' not found. Available input layers: "
                    f"{self.inputs.keys()}."
                )
            layer = self.inputs[input_layer_name]["layer"]

        chosen_neurons = np.asarray(
            layer.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type, attr-defined]
        )

        unfiltered = np.asarray(layer.history["firingrate"])[  # type: ignore[attr-defined]
            startid : endid + 1, chosen_neurons
        ]

        if filter_key not in self.history.keys():
            raise ValueError(f"Filter key '{filter_key}' not found.")

        if input_layer_name is None:
            filtered = np.asarray(self.history[filter_key])[
                startid : endid + 1, chosen_neurons
            ]
        else:
            filtered = np.asarray(self.history[filter_key][input_layer_name])[
                startid : endid + 1, chosen_neurons
            ]

        height = 0.6 * (unfiltered.max() - unfiltered.min())
        shifts = np.arange(unfiltered.shape[1]).reshape(1, -1) * height

        if sub_ax is None:
            n = unfiltered.shape[1]
            height = max([1, min(n / 12.0 + 5 / 3, 8)])
            _, sub_ax = plt.subplots(figsize=[6, height])

        color = layer.color  # type: ignore[attr-defined]

        sub_ax.plot(t, unfiltered + shifts, ls=(0, (1, 1)), color=color, alpha=1.0)
        sub_ax.plot(t, filtered + shifts, alpha=0.8, color=color)
        for i, shift in enumerate(shifts.T):
            sub_ax.fill_between(
                t, shift, filtered[:, i] + shift, color=color, alpha=0.4, lw=0
            )

        sub_ax.spines[["top", "right"]].set_visible(False)
        sub_ax.set_ylabel("Firing rate")
        sub_ax.set_xlabel("Time (s)")

        if title is None:
            title = filter_key.replace("_", " ").capitalize().replace("btsp", "BTSP")
        sub_ax.set_title(title)

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_{filter_key}", save=autosave)  # type: ignore[attr-defined]

        return sub_ax, t


class LearnLayer(SmoothFeedForwardLayer):
    """
    LearnLayer()

    Class extending SmoothFeedForwardLayer. Defines a population of neurons with
    weights that can be updated through learning.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 10,
        "name": "LearnLayer",
        "lr": 1e-4,  # learning rate
        "biases": None,
        "use_targets": False,
        "init_weights_zero": False,  # whether to initialise weights to 0
        "w_init_loc": 0,  # mean of the initial weights
        "w_init_scale": 1,  # scale of the initial weights
        "use_targets": False,  # whether to use targets
        "input_filter_tau": None,  # rise time constant
        "input_trend_tau": None,  # decay time constant
    }

    List of properties (in addition to SmoothFeedForwardLayer properties).
        • self.trainable_biases
        • self.target
        • self.learn
        • self.input_layers_with_no_learning

    List of methods (in addition to SmoothFeedForwardLayer methods):
        • self.set_learn()
        • self.add_input_layers_with_no_learning()
        • self.add_input()
        • self.save_to_history()
        • self.update_weights()
        • self.update()
        • self.plot_rate_map()
        • self.plot_rate_maps_across_learning()
        • self.plot_loss()
        • self.plot_histogram()
    """

    default_params = {
        "n": 10,
        "name": "LearnLayer",
        "lr": 1e-4,  # learning rate
        "biases": None,
        "use_targets": False,
        "init_weights_zero": False,  # whether to initialise weights to 0
        "w_init_loc": 0,  # mean of the initial weights
        "w_init_scale": 1,  # scale of the initial weights
        "use_targets": False,  # whether to use targets
        "input_filter_tau": None,  # rise time constant
        "input_trend_tau": None,  # decay time constant
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        LearnLayer(Agent)

        Initialise a layer that can learn weight updates. MSE with target is added to
        history.

        Attributes:
        - num_steps_total (int): Total number of update steps for the layer.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        self.set_learn(True)

        self.history["target_mse"] = list()
        self.num_steps_total = 0

        return

    @property
    def trainable_biases(self) -> bool:
        """
        self.trainable_biases

        Whether the biases of the layer are trainable.

        Returns:
        - (bool): Whether the biases are trainable.
        """

        if not hasattr(self, "_trainable_biases"):
            self._trainable_biases = self.params["biases"] is not None
        return self._trainable_biases

    @property
    def target(self) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """
        self.target

        Agent target, if it exists.

        Returns:
        - (1D np.ndarray or None): Agent's target position [x, y] or None.
        """

        if self.use_targets:  # type: ignore[attr-defined]
            return self.Agent.pos
        else:
            return None

    @property
    def learn(self) -> bool:
        """
        self.learn

        Whether this layer learns during self.update() calls. Only reflects input
        weights that are learnable.

        Returns:
        - (bool): Whether the layer learns during self.update() calls.
        """

        return self._learn

    @property
    def input_layers_with_no_learning(self) -> list[str]:
        """Obtain a list of input layer names that are not learning.

        Returns:
        - (list): List of input layer names that are not learning.
        """
        if hasattr(self, "_input_layers_with_no_learning"):
            return self._input_layers_with_no_learning
        else:
            return list()

    def set_learn(self, learn=None):
        """
        self.set_learn()

        Set whether this layer learns during self.update() calls. Only affects input
        weights that are learnable.

        Args:
        - learn (bool, optional): Whether the layer should learn during self.update()
            calls. If None, the current setting remains unchanged. Default is None.
        """

        if learn is None:
            pass
        else:
            self._learn = learn

    def add_input_layers_with_no_learning(self, input_layers=list()) -> None:
        """
        self.add_input_layers_with_no_learning()

        Add name of input layers that are not learning.

        Args:
        - input_layers (str or list): Name of the input layer(s) to add to list of
            layers with no learning. Default is list().
        """

        if not hasattr(self, "_input_layers_with_no_learning"):
            self._input_layers_with_no_learning = list()
        if not isinstance(input_layers, list):
            input_layers = [input_layers]
        self._input_layers_with_no_learning.extend(input_layers)

    def add_input(self, input_layer: riab_neurons.Neurons, **kwargs):
        """
        self.add_input()

        Add an input layer.

        Args:
        - input_layer (riab_neurons.Neurons): Neuron layer to add as input

        Keyword args:
        - **kwargs: Keyword arguments passed to SmoothFeedForward.add_input().
        """

        n_in, n_out = input_layer.n, self.n  # type: ignore[attr-defined]

        if "w" not in kwargs.keys() or kwargs["w"] is None:
            if self.init_weights_zero:  # type: ignore[attr-defined]
                kwargs["w"] = np.zeros((n_out, n_in))
            elif self.w_init_loc != 0:  # type: ignore[attr-defined]
                kwargs["w"] = np.random.normal(
                    loc=self.w_init_loc,  # type: ignore[attr-defined]
                    scale=self.w_init_scale / np.sqrt(n_in),  # type: ignore[attr-defined]
                    size=(n_out, n_in),
                )

        super().add_input(input_layer, w_init_scale=self.w_init_scale, **kwargs)  # type: ignore[attr-defined]

        name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]
        self.inputs[name_in]["filtered_inputs_for_learning"] = np.full(n_in, np.nan)
        self.inputs[name_in]["filtered_trends_for_learning"] = np.zeros(n_in)

        name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]
        for key in ["inputs", "trends"]:
            if f"filtered_{key}_for_learning" not in self.history.keys():
                self.history[f"filtered_{key}_for_learning"] = dict()
            self.history[f"filtered_{key}_for_learning"][name_in] = list()

    def save_to_history(self):
        """
        self.save_to_history()

        Save the current state of the layer to the history, including the MSE loss
        with the target.
        """

        super().save_to_history()

        if self.target is None:
            self.history["target_mse"].append(np.nan)
        else:
            target_mse = np.mean((self.target - self.firingrate) ** 2)
            self.history["target_mse"].append(target_mse)

    def update_weights(self):
        """
        self.update_weights()

        Update weights. (Method should be overwritten in subclasses.)
        """

        return

    def update(self, **kwargs):
        """
        self.update()

        Update the layer, i.e. calculate the new firing rates and update the
        weights and biases, if applicable.

        Keyword args:
        - **kwargs: Keyword arguments passed to self.update_weights().

        Attributes:
        - num_steps_total (int): Total number of update steps for the layer.
        """

        super().update()

        if self.learn:
            self.update_weights(**kwargs)

        self.num_steps_total += 1

        return

    def plot_rate_map(
        self,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        shape: tuple | None = None,
        target_num_col: int = 15,
        no_legend: bool = False,
        ax: np.ndarray | plt.Axes | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes | np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
        """
        self.plot_rate_map()

        Plot the rate map of the layer, ensuring no more than 20 columns are plotted.

        See FeedForwardLayer.plot_rate_map() for more information.

        Args:
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - shape (tuple, optional): Shape of the plot. Default is None.
        - target_num_col (int, optional): Aimed number of columns. Default is 15.
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot on
           (one per plotted neuron, if environment is 2D). Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots with rate map
            plotted.
        """

        if shape is None:
            n = len(self.return_list_of_neurons(chosen_neurons=chosen_neurons))  # type: ignore[arg-type]
            shape = plot_util.get_plot_shape(n, target_num_col=target_num_col)

        kwargs["chosen_neurons"] = chosen_neurons
        kwargs["shape"] = shape[::-1]

        ax = super().plot_rate_map(autosave=False, ax=ax, **kwargs)

        if no_legend:
            for sub_ax in np.asarray(ax).ravel():
                if sub_ax.get_legend() is not None:
                    sub_ax.get_legend().remove()

        fig = np.asarray(ax).ravel()[0].figure
        util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

        return ax

    def plot_rate_maps_across_learning(
        self,
        num_maps: int = 3,
        prop_each: float = 0.4,
        normalize_together: bool = True,
        title: str | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        axes: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | None = None,
        autosave: bool | None = None,
        **kwargs,
    ):
        """
        self.plot_rate_maps_across_learning()

        Plot the rate maps of the layer across learning.

        Args:
        - num_maps (int, optional): Number of maps to plot. Default is 3.
        - prop_each (float, optional): Proportion of the learning period to plot
            for each map. Default is 0.4.
        - normalize_together (bool, optional): Whether to normalize the maps
            together. Default is True.
        - title (str, optional): Title of the plot. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - axes (2D np.ndarray, optional): Array of subplots with shape
           (number of ROIs, num_maps or v.v.). If None, a new array is created.
            Default is None.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to LearnLayer.plot_rate_map().

        Returns:
        - axes (2D np.ndarray): Array of subplots. If input axes was None,
            shape is 2D with shape (number of ROIs, num_maps or v.v. if only one ROI).
        """

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type, attr-defined]

        # initialise axes
        if axes is None:
            if len(chosen_neurons) == 1:
                ncols = num_maps
                nrows = len(chosen_neurons)
                row = "chosen_neurons"
            else:
                ncols = len(chosen_neurons)
                nrows = num_maps
                row = "num_maps"
            _, axes = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                figsize=(ncols * 3, nrows * 3),
                squeeze=False,
            )
        else:
            axes = axes.reshape(len(axes), -1)
            nrows, ncols = axes.shape
            if len(chosen_neurons) >= num_maps:
                row = "chosen_neurons" if nrows >= ncols else ncols
            else:
                row = "num_maps" if nrows >= ncols else ncols

            rows_needed = len(chosen_neurons) if row == "chosen_neurons" else num_maps
            cols_needed = num_maps if row == "chosen_neurons" else len(chosen_neurons)

            if len(axes) < len(rows_needed):
                row_str = "neurons" if row == "chosen_neurons" else "maps"
                raise ValueError(
                    f"Insufficient number of rows for number of {row_str}."
                )

            if len(axes) < len(cols_needed):
                col_str = "maps" if row == "chosen_neurons" else "neurons"
                raise ValueError(
                    f"Insufficient number of columns for number of {col_str}."
                )

        t = self.history["t"]
        n_pts = int(prop_each * len(t))
        start_pts = [int(st) for st in np.linspace(0, len(t) - n_pts, num_maps)]

        subplots = []
        for i, start in enumerate(start_pts):
            map_axes = axes[i] if row == "num_maps" else axes[:, i]
            t_start = t[start]
            stop = min([len(t) - 1, start + n_pts])
            t_end = t[stop]
            map_axes[0].set_title(f"From {t_start / 60:.2f} to {t_end / 60:.2f} min.")

            self.plot_rate_map(
                ax=map_axes,
                t_start=t_start,
                t_end=t_end,
                method="history",
                colorbar=False,
                chosen_neurons=chosen_neurons,
                autosave=False,
                **kwargs,
            )

            subplots.append(map_axes.ravel()[0])

        if normalize_together:
            plot_util.normalize_cmaps(subplots, shrink=0.7)

        fig = np.asarray(axes).ravel()[0].figure

        if title is None:
            title = "Rate maps across learning"

        fig.suptitle(title, y=0.90)

        util.save_figure(fig, f"{self.name}_rate_maps_across_learning", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_loss(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        color: str | None = None,
        alpha: float = 0.7,
        xlim: tuple[float, float] | None = None,
        k_prop_to_loss_length: float = 0.15,
        k_max: int = 10000,
        autosave: bool | None = None,
        **loss_kwargs,
    ) -> plt.Axes:
        """
        self.plot_loss()

        Plot the loss of the layer over time.

        Args:
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - color (str, optional): Color of the plot. Default is None.
        - alpha (float, optional): Alpha of the plot. Default is 0.7.
        - xlim (tuple, optional): x limits of the plot. Default is None.
        - k_prop_to_loss_length (float, optional): Smoothing factor, proportional to
            clength of loss array. Default is 0.15.
        - k_max (int, optional): Maximum smoothing factor. Default is 10000.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **loss_**kwargs: Keyword arguments passed to plot_util.plot_loss().

        Returns:
        - sub_ax (plt.Axes): Subplot with loss plotted.
        """

        reset_times = None
        if hasattr(self.Agent, "trajectory_df"):
            reset_times = self.Agent.get_reset_times()  # type: ignore[attr-defined]

        if color is None:
            color = self.color  # type: ignore[attr-defined]

        sub_ax = plot_util.plot_loss(
            self.history["t"],
            self.history["target_mse"],
            mark_ts=reset_times,
            t_start=t_start,
            t_end=t_end,
            sub_ax=sub_ax,
            color=color,
            alpha=alpha,
            xlim=xlim,
            k_prop_to_loss_length=k_prop_to_loss_length,
            k_max=k_max,
            **loss_kwargs,
        )

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_loss", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_histogram(
        self,
        sub_ax: plt.Axes | None = None,
        color: str | None = None,
        alpha: float = 0.7,
        t_start: float | None = None,
        xlabel: str | None = None,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """
        self.plot_histogram()

        Plot a histogram of the firing rates of the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - color (str, optional): Color of the plot. Default is None.
        - alpha (float, optional): Alpha of the plot. Default is 0.7.
        - t_start (float, optional): Start timepoint of the plot. Default is None.
        - xlabel (str, optional): x label of the plot. Default is None.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot withhistogram plotted.
        """

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(8, 3))

        _, startid, _ = self.get_plotting_times(t_start=t_start)
        firingrates = np.asarray(self.history["firingrate"])[startid:]

        if color is None:
            color = str(self.color)  # type: ignore[attr-defined]
        if firingrates.shape[1] > 1:
            color = [color for _ in range(firingrates.shape[1])]  # type: ignore[assignment]

        sub_ax.hist(firingrates, color=color, alpha=alpha)

        if xlabel is None:
            xlabel = "Firing rate"
        sub_ax.set_xlabel(xlabel)
        sub_ax.set_ylabel("Count")
        sub_ax.spines[["top", "right"]].set_visible(False)

        fig = sub_ax.figure

        util.save_figure(fig, f"{self.name}_firing_rate_histogram", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class HebbianLayer(LearnLayer):
    """
    HebbianLayer()

    Class extending LearnLayer. Defines a population of neurons that tune their
    weights through Hebbian learning.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 10,
        "name": "HebbianLayer",
        "lr": 1e-4,  # learning rate
        "biases": None,
        "normalize_weights_divisively": False,
        "apply_Ojas_rule": False,
        "regularization_alpha": None,
        "use_targets": False,
        "init_weights_zero": False,  # whether to initialise weights to 0
        "w_init_scale": 1,  # scale of the initial weights
        "learning_filter_tau": None,
        "learning_trend_tau": None,
        "p": 1,  # power for normalization, if used
    }

    See LearnLayer for properties.

    List of methods (in addition to LearnLayer methods):
        • self.add_input()
        • self.get_regularization_alpha()
        • self.save_to_history()
        • self.update_weights()
        • self.update()
    """

    default_params = {
        "n": 10,
        "name": "HebbianLayer",
        "lr": 1e-4,  # learning rate
        "biases": None,
        "normalize_weights_divisively": False,
        "apply_Ojas_rule": False,
        "regularization_alpha": None,
        "use_targets": False,
        "init_weights_zero": False,  # whether to initialise weights to 0
        "w_init_scale": 1,  # scale of the initial weights
        "learning_filter_tau": None,
        "learning_trend_tau": None,
        "p": 1,  # power for normalization, if used
    }

    ignored_param_keys = list()
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Agent: "ratinabox.Agent", params: dict = dict()):
        """
        HebbianLayer(Agent)

        Initialise a layer that learns weight updates via Hebbian learning.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        if self.apply_Ojas_rule and self.normalize_weights_divisively:  # type: ignore[attr-defined]
            raise ValueError("Can only set 'oja' or 'norm' to True, not both.")

        return

    def add_input(self, input_layer: riab_neurons.Neurons, **kwargs):
        """
        self.add_input(input_layer)

        Add an input layer.

        Args:
        - input_layer (riab_neurons.Neurons): Neuron layer to add as input layer

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.add_input().
        """

        super().add_input(input_layer, **kwargs)

        if self.normalize_weights_divisively:
            name_in = input_layer.name  # type: ignore[attr-defined]
            self.update_weights(
                filter_key="I", lr=0
            )  # normalize weights only (no update)
            self.inputs[name_in]["w_init"] = copy.deepcopy(self.inputs[name_in]["w"])

    def get_regularization_alpha(self, alpha=None):
        """
        self.get_regularization_alpha()

        Obtain the regularization factor for the Hebbian update.

        Args:
        - alpha (float): Regularization factor. If None, the regularization attribute
            or a default value are used. Default is None.

        Returns:
        - alpha (float): Regularization factor.
        """

        if self.apply_Ojas_rule:  # type: ignore[attr-defined]
            if alpha is None:
                alpha = self.regularization_alpha or 0.1
        elif self.normalize_weights_divisively:  # type: ignore[attr-defined]
            if alpha is None:
                alpha = self.regularization_alpha or 1.0
        else:
            alpha = None

        return alpha

    def save_to_history(self):
        """
        self.save_to_history()

        Save the current state of the layer to the history including the filtered
        input and trend data used for learning.
        """

        super().save_to_history()

        for name, input_layer in self.inputs.items():
            for key in ["inputs", "trends"]:
                self.history[f"filtered_{key}_for_learning"][name].append(
                    input_layer[f"filtered_{key}_for_learning"].tolist()
                )

    def update_weights(
        self,
        filter_key: str = "filtered_inputs_for_learning",
        O: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        lr: np.ndarray[tuple[int], np.dtype[np.float64]] | float | None = None,
        calculate_only: bool = False,
    ) -> (
        tuple[
            list[np.ndarray[tuple[int], np.dtype[np.float64]]],
            np.ndarray[tuple[int], np.dtype[np.float64]] | None,
        ]
        | None
    ):
        """
        self.update_weights()

        Update the weights of the layer.

        Args:
        - filter_key (str, optional): Key of the input to use for a weight update.
            Default is "filtered_inputs_for_learning".
        - O (1D np.ndarray, optional): Output values to use instead of targets or layer
            firingrates. Default is None.
        - lr (1D np.ndarray or float, optional): Learning rate, optionally per layer
            neuron. Default is None.
        - calculate_only (bool, optional): If True, the update is only calculated and
            returned, but not applied. Default is False.

        Returns:
        - if calculate_only:
            ws_delta (list[2D np.ndarray]): List of weight updates, each
                with shape (O, I_i)
            b_delta (1D np.ndarray or None): Bias updates, if applicable.
        """

        if O is None:
            if self.use_targets:  # type: ignore[attr-defined]
                O = np.asarray(self.target).astype(np.float64)
            else:
                O = np.asarray(self.firingrate).astype(np.float64)

        Is = [
            input_layer[filter_key]
            for name, input_layer in self.inputs.items()
            if name not in self.input_layers_with_no_learning
        ]

        ws = [
            input_layer["w"]
            for name, input_layer in self.inputs.items()
            if name not in self.input_layers_with_no_learning
        ]
        b = self.biases if self.trainable_biases else None
        if lr is None:
            lr = self.lr  # type: ignore[attr-defined]

        if calculate_only:
            ws_pre = ws
            ws = copy.deepcopy(ws)
            if self.trainable_biases:
                b_pre = b
                b = copy.deepcopy(b)

        alpha = self.get_regularization_alpha()
        if self.apply_Ojas_rule and not calculate_only:  # type: ignore[attr-defined]
            util.perform_Oja_update_(Is, ws, O, lr=lr, b=b, alpha=alpha)
        elif self.normalize_weights_divisively and not calculate_only:  # type: ignore[attr-defined]
            util.perform_divisively_normalized_Hebbian_update_(
                Is, ws, O, lr=lr, b=b, p=self.p, alpha=alpha
            )
        else:
            util.perform_Hebbian_update_(Is, ws, O, lr=lr, b=b)

        if calculate_only:
            ws_delta = [ws[i] - ws_pre[i] for i in range(len(ws))]
            if self.trainable_biases:
                b_delta = np.asarray(b) - np.asarray(b_pre)
            else:
                b_delta = None
            return ws_delta, b_delta

        else:
            return None

    def update(self):
        """Update the layer, i.e. calculate the new firing rates and update the
        weights and biases, if applicable."""

        self.update_filtered_inputs(
            self.learning_filter_tau,  # type: ignore[attr-defined]
            self.learning_trend_tau,  # type: ignore[attr-defined]
            filter_key="filtered_inputs_for_learning",
        )

        super().update(filter_key="filtered_inputs_for_learning")


class BTSPLayer(HebbianLayer):
    """
    BTSPLayer

    Class extending HebbianLayer. Defines a population of neurons that tune their
    weights through Hebbian learning with BTSP.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 10,
        "name": "BTSPLayer",
        "BTSP_filter_tau": 4,
        "BTSP_trend_tau": None,
        "post_BTSP_filter_tau": "half",
        "post_BTSP_trend_tau": None,
        "BTSP_lr_fact": 300,
        "single_BTSP": False,
        "BTSP_distance_prop": None,  # None to remove constraint
    }

    List of properties (in addition to HebbianLayer properties):
        • self.BTSP_learn

    List of methods (in addition to HebbianLayer methods):
        • self.set_BTSP_learn()
        • self.add_input()
        • self.save_to_history()
        • self.log_num_steps_to_apply_BTSP()
        • self.update_BTSP_buffer()
        • self.reset_BTSP_buffer()
        • self.compute_BTSP_update()
        • self.check_apply_update_for_BTSP()
        • self.update_for_BTSP()
        • self.update()
        • self.plot_filtered_for_BTSP()
        • self.plot_BTSP_ramp()
        • self.add_BTSP_markers_to_plots()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
    """

    default_params = {
        "n": 10,
        "name": "BTSPLayer",
        "BTSP_filter_tau": 4,
        "BTSP_trend_tau": None,
        "post_BTSP_filter_tau": "half",
        "post_BTSP_trend_tau": None,
        "BTSP_lr_fact": 300,
        "single_BTSP": False,
        "BTSP_distance_prop": None,  # None to remove constraint
    }

    ignored_param_keys = list()
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        BTSPLayer(Agent)

        Initialise a layer that can learn weight updates using BTSP. BTSP variables are
        added to the history.

        Attributes:
        - BTSP_buffer (dict): Buffer for BTSP updates, with keys and values:
            - num_steps (1D np.ndarray): Number of steps to apply BTSP.
            - pre_ws_delta (None or list): Weight updates before BTSP.
                List of weight updates, each with shape (O, I_i)
            - pre_b_delta (1D np.ndarray): Bias updates before BTSP.
            - ws_delta (None or list): Weight updates after BTSP.
                List of weight updates, each with shape (O, I_i)
        - last_BTSP_pos (list): Last position at which a BTSP event occurred.
        - last_BTSP_step (int): Last step at which a BTSP event occurred.
        - num_BTSP_to_date (int): Number of BTSP events to date.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        self.history["BTSP_events"] = list()
        self.history["BTSP_targets"] = list()
        self.history["num_steps_to_apply_BTSP"] = list()

        self._init_post_BTSP_filters()

        self.num_BTSP_to_date = np.zeros(self.n)  # type: ignore[attr-defined]
        # type: ignore[attr-defined]
        self.last_BTSP_step = np.full(self.n, np.nan)  # type: ignore[attr-defined]
        self.last_BTSP_pos = [None for _ in range(self.n)]  # type: ignore[attr-defined]

        self.BTSP_buffer = {
            "num_steps": np.zeros(self.n),
            "pre_ws_delta": None,
            "pre_b_delta": np.zeros(self.n),
            "ws_delta": None,
            "b_delta": np.zeros(self.n),
        }

        if self.use_targets:  # type: ignore[attr-defined]
            raise ValueError("BTSPLayer does not support targets.")

        self.set_BTSP_learn(True)

        return

    @property
    def BTSP_learn(self) -> bool:
        """
        self.BTSP_learn

        Whether this layer undergoes BTSP learning during self.update() calls. Only
        reflects input weights that are learnable.

        Returns:
        - (bool): BTSP learning state.
        """

        return self._BTSP_learn

    def _init_post_BTSP_filters(self):
        """
        self._init_post_BTSP_filters()

        Initialise the filter parameters used to compute learning updates for
        inputs activated after a BTSP (post BTSP kernel).

        Also adds post BTSP filter keys to the history dictionary.

        Attributes:
        - filtered_post_BTSP_activity (1D np.ndarray): Filtered post BTSP activity.
        - filtered_post_BTSP_activity_trend (1D np.ndarray): Filtered post BTSP
            activity trend.
        - post_BTSP_exp_AUC (float): Exponential AUC of the post BTSP filter.
        - post_BTSP_filter_tau (float): Time constant of the post BTSP filter.
        - post_BTSP_trend_tau (float): Time constant of the post BTSP trend.
        - pre_BTSP_exp_AUC (float): Exponential AUC of the pre BTSP filter.
        """

        if isinstance(self.post_BTSP_filter_tau, str):
            if self.post_BTSP_filter_tau == "half":
                div = 2
            elif self.post_BTSP_filter_tau == "third":
                div = 3
            elif self.post_BTSP_filter_tau == "equal":
                div = 1
            else:
                raise ValueError(
                    f"Invalid post_BTSP_filter_tau value: {self.post_BTSP_filter_tau}"
                )
            self.post_BTSP_filter_tau = self.BTSP_filter_tau / div  # type: ignore[attr-defined]

        # start at 0
        self.filtered_post_BTSP_activity = np.zeros(self.n)  # type: ignore[attr-defined]
        self.filtered_post_BTSP_activity_trend = np.zeros(self.n)  # type: ignore[attr-defined]

        self.pre_BTSP_exp_AUC = util.get_exponential_AUC(
            self.BTSP_filter_tau, self.BTSP_trend_tau, dt=self.Agent.dt  # type: ignore[attr-defined]
        )

        self.post_BTSP_exp_AUC = util.get_exponential_AUC(
            self.post_BTSP_filter_tau, self.post_BTSP_trend_tau, dt=self.Agent.dt  # type: ignore[attr-defined]
        )

        for key_str in ["activity", "activity_trend"]:
            key = f"filtered_post_BTSP_{key_str}"
            if key not in self.history.keys():
                self.history[key] = list()

    def set_BTSP_learn(self, learn=None):
        """
        self.set_BTSP_learn()

        Set the layer to learn using BTSP during self.update() calls. Only affects 
        input weights that are learnable.

        Args:
        - learn (bool, optional): Whether the layer should learn using BTSP. If None,
            the current setting remains unchanged. Default is None.
        """

        if learn is None:
            pass
        else:
            self._BTSP_learn = learn
        return

    def add_input(self, input_layer: riab_neurons.Neurons, **kwargs):
        """
        self.add_input(input_layer)

        Adds an input layer, and initialises variables storing BTSP filtered inputs,
        as well as corresponding history keys.

        Args:
        - input_layer (riab_neurons.Neurons): Neuron layer to add as input layer

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.add_input().
        """
        super().add_input(input_layer, **kwargs)

        name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]

        self.inputs[name_in]["filtered_inputs_for_BTSP"] = np.full(n_in, np.nan)
        self.inputs[name_in]["filtered_trends_for_BTSP"] = np.zeros(n_in)

        for key in ["inputs", "trends"]:
            if f"filtered_{key}_for_BTSP" not in self.history.keys():
                self.history[f"filtered_{key}_for_BTSP"] = dict()
            self.history[f"filtered_{key}_for_BTSP"][name_in] = list()

    def save_to_history(self):
        """
        self.save_to_history()

        Save the current state of the layer to the history, including the filtered
        input and trend data used for learning
        """

        super().save_to_history()

        for name in ["activity", "activity_trend"]:
            key = f"filtered_post_BTSP_{name}"
            self.history[key].append(getattr(self, key).tolist())

        for name, input_layer in self.inputs.items():
            for key in ["inputs", "trends"]:
                self.history[f"filtered_{key}_for_BTSP"][name].append(
                    input_layer[f"filtered_{key}_for_BTSP"].tolist()
                )

    def log_num_steps_to_apply_BTSP(self, last=False):
        """
        self.log_num_steps_to_apply_BTSP()

        Log the number of steps required to apply BTSP.

        Args:
        - last (bool, optional): Whether to log only the last recorded BTSP event
            applied. Default is False.
        """

        if len(self.history["num_steps_to_apply_BTSP"]) == 0:
            log_str = "No BTSP events applied."

        elif last:
            num_steps = self.history["num_steps_to_apply_BTSP"][-1]
            seconds = num_steps * self.Agent.dt
            log_str = (
                f"Last BTSP event was applied after {num_steps} "
                f"({seconds:.2f} sec.)."
            )

        else:
            num_steps = np.asarray(self.history["num_steps_to_apply_BTSP"])
            seconds = num_steps * self.Agent.dt
            all_steps = ", ".join([str(num_step) for num_step in num_steps])

            log_str = (
                f"BTSP events applied after {num_steps.mean():.2f} steps "
                f"({seconds.mean():.2f} sec.) (num steps: {all_steps})."
            )

        print(log_str)

    def update_BTSP_buffer(self, ws_delta, b_delta=None, pre=False):
        """
        self.update_BTSP_buffer(ws_delta)

        Update the BTSP buffer.

        Args:
        - ws_delta (list): List of weight updates, each with shape (O, I_i).
        - b_delta (1D np.ndarray, optional): Bias updates, if applicable.
            Default is None.
        - pre (bool, optional): Whether the update is pre or post BTSP.
            Default is False.
        """

        pre_str = "pre_" if pre else ""

        if self.BTSP_buffer[f"{pre_str}ws_delta"] is None:
            self.BTSP_buffer[f"{pre_str}ws_delta"] = ws_delta
        else:
            for w in range(len(ws_delta)):
                self.BTSP_buffer[f"{pre_str}ws_delta"][w] += ws_delta[w]

        if self.trainable_biases:
            self.BTSP_buffer[f"{pre_str}b_delta"] += b_delta

        updated = self.BTSP_buffer["num_steps"].astype(bool)
        self.BTSP_buffer["num_steps"][updated] = (
            self.BTSP_buffer["num_steps"][updated] + 1
        )

    def reset_BTSP_buffer(self, i=0):
        """
        self.reset_BTSP_buffer()

        Reset the BTSP buffer for a specific neuron.

        Args:
        - i (int, optional): Index of the neuron to reset. Default is 0.
        """

        ws = [
            input_layer["w"]
            for name, input_layer in self.inputs.items()
            if name not in self.input_layers_with_no_learning
        ]

        for w in range(len(ws)):
            self.BTSP_buffer["pre_ws_delta"][w][i] = 0
            self.BTSP_buffer["ws_delta"][w][i] = 0

        if self.trainable_biases:
            self.BTSP_buffer["pre_b_delta"][i] = 0
            self.BTSP_buffer["b_delta"][i] = 0

        self.BTSP_buffer["num_steps"][i] = 0

    def compute_BTSP_update(self, i=0, w=0, bias=False):
        """
        self.compute_BTSP_update()

        Compute the BTSP update for a specific neuron and weight.

        Args:
        - i (int, optional): Index of the neuron. Default is 0.
        - w (int, optional): Index of the weight. Default is 0.
        - bias (bool, optional): Whether the update is for a bias instead of a weight.
            Default is False.

        Returns:
        - BTSP_update (float or 1D np.ndarray): Computed BTSP update.
        """

        if bias:
            BTSP_update = (
                self.BTSP_buffer["pre_b_delta"][i]
                + self.BTSP_buffer["b_delta"][i] * self.post_BTSP_factor  # type: ignore[attr-defined]
            ) / 2

        else:
            BTSP_update = (
                self.BTSP_buffer["pre_ws_delta"][w][i] * self.pre_BTSP_exp_AUC
                + self.BTSP_buffer["ws_delta"][w][i]
            ) / (self.pre_BTSP_exp_AUC + self.post_BTSP_exp_AUC)

        return BTSP_update

    def check_apply_update_for_BTSP(self):
        """
        self.check_apply_update_for_BTSP()

        Checks whether BTSP updates should be applied. If so, computes and applies the
        updates.
        """

        apply_BTSP = self.BTSP_buffer[
            "num_steps"
        ] * ~self.filtered_post_BTSP_activity.astype(bool)

        if not apply_BTSP.any():
            return

        ws = [
            input_layer["w"]
            for name, input_layer in self.inputs.items()
            if name not in self.input_layers_with_no_learning
        ]
        b = self.biases if self.trainable_biases else None

        for i in np.where(apply_BTSP)[0]:
            for w in range(len(ws)):
                w_update = self.compute_BTSP_update(i=i, w=w)
                ws[w][i] += w_update

            if self.trainable_biases:
                b_update = self.compute_BTSP_update(i=i, bias=True)
                b[i] += b_update

            self.history["num_steps_to_apply_BTSP"].append(
                int(self.BTSP_buffer["num_steps"][i])
            )

            self.reset_BTSP_buffer(i=i)

        if self.apply_Ojas_rule:  # type: ignore[attr-defined]
            raise NotImplementedError("Oja's rule not implemented for BTSP.")
        elif self.normalize_weights_divisively:  # type: ignore[attr-defined]
            self.update_weights(
                filter_key="I", lr=0
            )  # normalize weights only (no learning update)

    def get_BTSP_targets(
        self,
        BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] | None = list(),
    ):
        """
        self.get_BTSP_targets()

        Obtain the BTSP targets.

        Args:
        - BTSP_targets (list or 1D np.ndarray, optional): BTSP targets
            (neuron indices). Default is list().

        Returns:
        - BTSP_targets (1D np.ndarray): BTSP targets (neuron indices).
        """

        if BTSP_targets is None:
            BTSP_targets = list()

        BTSP_targets = np.asarray(BTSP_targets)

        if len(BTSP_targets) and BTSP_targets.max() >= self.n:
            raise ValueError("BTSP target neuron index out of range.")

        return BTSP_targets

    def update_from_BTSP_targets(
        self,
        BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list(),
    ):
        """
        self.update_from_BTSP_targets()

        Update the layer using BTSP targets, if provided.

        Args:
        - BTSP_targets (list or 1D np.ndarray, optional): List of BTSP targets.
            Default is list().
        """

        BTSP_targets = self.get_BTSP_targets(BTSP_targets)

        if not self.BTSP_learn or len(BTSP_targets) == 0:
            return

        # cannot trigger a new BTSP event while one is on-going for the same neuron
        BTSP_targets = np.asarray(
            [targ for targ in BTSP_targets if not self.BTSP_buffer["num_steps"][targ]]
        )

        if self.single_BTSP:  # type: ignore[attr-defined]
            keep_BTSP_targets = np.asarray(
                [targ for targ in BTSP_targets if self.num_BTSP_to_date[targ] == 0]
            )

        elif self.BTSP_distance_prop is not None:  # type: ignore[attr-defined]
            keep_BTSP_targets = list()
            for targ in BTSP_targets:
                closest_recent_BTSP_event = self.last_BTSP_pos[targ]
                if closest_recent_BTSP_event is None:
                    keep_BTSP_targets.append(targ)
                else:
                    dist = np.sqrt(
                        np.sum(
                            (self.Agent.pos - closest_recent_BTSP_event) ** 2,
                            axis=-1,
                        )
                    )
                    if dist > self.BTSP_distance_prop * self.Agent.dt:
                        keep_BTSP_targets.append(targ)

            keep_BTSP_targets = np.asarray(keep_BTSP_targets)

        else:
            keep_BTSP_targets = np.asarray(BTSP_targets)

        if len(keep_BTSP_targets) == 0:
            return

        BTSP_targets = keep_BTSP_targets

        lr = np.zeros(self.n)  # type: ignore[attr-defined]
        lr[np.asarray(BTSP_targets)] = self.BTSP_lr_fact * self.lr  # type: ignore[attr-defined]

        self.num_BTSP_to_date[np.asarray(BTSP_targets)] += +1
        self.last_BTSP_step[np.asarray(BTSP_targets)] = self.num_steps_total - 1
        for targ in BTSP_targets:
            self.last_BTSP_pos[targ] = self.Agent.pos

        ws_delta, b_delta = self.update_weights(
            filter_key="filtered_inputs_for_BTSP", lr=lr, calculate_only=True
        )
        self.update_BTSP_buffer(ws_delta, b_delta, pre=True)
        self.BTSP_buffer["num_steps"][BTSP_targets] = 1

        self.history["BTSP_events"].append(
            self.num_steps_total - 1
        )  # recorded after update
        self.history["BTSP_targets"].append(BTSP_targets)

        self.filtered_post_BTSP_activity[BTSP_targets] = self.firingrate[BTSP_targets]

        return

    def update_for_BTSP(
        self, BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list()
    ):
        """
        self.update_for_BTSP()

        Update the layer using BTSP, optionally using BTSP targets.

        Args:
        - BTSP_targets (list or 1D np.ndarray, optional): List of BTSP targets.
            Default is list().
        """

        self.update_filtered_inputs(
            self.BTSP_filter_tau,  # type: ignore[attr-defined]
            self.BTSP_trend_tau,  # type: ignore[attr-defined]
            filter_key="filtered_inputs_for_BTSP",
        )

        if self.BTSP_buffer["num_steps"].any():
            # compute post BTSP update
            ws_delta, b_delta = self.update_weights(
                filter_key="I",  # where most recent, unfiltered input is stored
                O=self.filtered_post_BTSP_activity,
                lr=self.BTSP_lr_fact * self.lr,
                calculate_only=True,
            )
            self.update_BTSP_buffer(ws_delta, b_delta, pre=False)

        # main BTSP update
        self.update_from_BTSP_targets(BTSP_targets)

        # calculate filtered signal for post BTSP
        X_t1, T_t1 = util.get_filtered_signal(
            np.zeros_like(self.filtered_post_BTSP_activity),
            X_t=self.filtered_post_BTSP_activity,
            T_t=self.filtered_post_BTSP_activity_trend,
            filter_tau=self.post_BTSP_filter_tau,
            trend_tau=self.post_BTSP_trend_tau,
            dt=self.Agent.dt,
            atol=1e-6,
        )

        self.filtered_post_BTSP_activity = X_t1
        self.filtered_post_BTSP_activity_trend = T_t1

        self.check_apply_update_for_BTSP()

    def update(
        self, BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list()
    ):
        """
        self.update()

        Update the layer, i.e. calculate the new firing rates and update the
        weights and biases, if applicable.

        Args:
        - BTSP_targets (list or 1D np.ndarray, optional): List of BTSP targets to use in
            BTSP update. Default is list().
        """

        super().update()

        self.update_for_BTSP(BTSP_targets)

        return

    def plot_filtered_for_BTSP(
        self,
        input_layer_name: str | None = None,
        t_start: float | None = None,
        title: str | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        self.plot_filtered_for_BTSP()

        Plot the filtered inputs of the layer.

        Args:
        - input_layer_name (str, optional): Name of the input layer to plot.
        - t_start (float, optional): Start time of the plot. Default is None.
        - title (str, optional): Title of the plot. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to HebbianLayer.plot_filtered().

        Returns:
        - sub_ax (plt.Axes): Subplot with filtered inputs or firingrates for BTSP
            plotted.
        """

        if input_layer_name is None:
            layer = self
            filter_key = "filtered_post_BTSP_activity"
            title = title or "Firing rate filtered post BTSP"
        else:
            if input_layer_name not in self.inputs.keys():
                raise ValueError(
                    f"Input layer '{input_layer_name}' not found. Available input "
                    f"layers: {self.inputs.keys()}."
                )
            layer = self.inputs[input_layer_name]["layer"]
            filter_key = "filtered_inputs_for_BTSP"
            title = title or "Inputs filtered for BTSP"

        chosen_neurons = np.asarray(
            layer.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type, attr-defined]
        )

        sub_ax, t = super().plot_filtered(
            input_layer_name=input_layer_name,
            filter_key=filter_key,
            t_start=t_start,
            title=title,
            chosen_neurons=chosen_neurons,
            autosave=False,
            **kwargs,
        )

        _, startid, _ = self.get_plotting_times(t_start=t_start)

        BTSP_events = np.asarray(self.history["BTSP_events"]) - startid
        BTSP_targets = self.history["BTSP_targets"]
        BTSP_mask = (BTSP_events >= 0) & (BTSP_events < len(t))

        flat_BTSP_events = list()  # type: list[int]
        flat_BTSP_targets = list()  # type: list[int]
        for BTSP_idx in np.where(BTSP_mask)[0]:
            ev = BTSP_events[BTSP_idx]
            targs = BTSP_targets[BTSP_idx]
            flat_BTSP_events.extend([ev for _ in range(len(targs))])
            flat_BTSP_targets.extend(targs)

        n = len(chosen_neurons)
        shift = np.diff(sub_ax.get_ylim())[0] / (n + 0.4)
        heights = (np.arange(n + 1)[1:] - 0.35) * shift

        plot_BTSP_events, plot_BTSP_targets, plot_heights = list(), list(), list()
        for ev, targ in zip(flat_BTSP_events, flat_BTSP_targets):
            if targ in chosen_neurons:
                idx = chosen_neurons.tolist().index(targ)
                plot_BTSP_events.append(ev)
                plot_BTSP_targets.append(targ)
                plot_heights.append(heights[idx])

        if len(plot_BTSP_events):
            sub_ax.scatter(
                t[np.asarray(plot_BTSP_events)],
                plot_heights,
                color=(self.color or "k"),
                alpha=0.8,
                marker=markers.MarkerStyle("x"),
                s=10,
            )

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_filtered_inputs_for_BTSP", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_BTSP_ramp(self, axes=None, plot_events=True, autosave=None):
        """
        self.plot_BTSP_ramp()

        Plot the BTSP ramp variables of the layer.

        Args:
        - axes (1 or 2D np.ndarray): Array of subplots to plot on,
            with shape (2, 1) or (2, ). Default is None.
        - plot_events (bool, optional): Whether to plot BTSP event markers.
            Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

        Returns:
        - axes (1 or 2D np.ndarray): Array of subplots with BTSP ramp variables plotted.
            If input axes was None, shape is 2D (2, 1).
        """

        if axes is None:
            _, axes = plt.subplots(2, 1, figsize=[9, 5], sharex=True, squeeze=False)
        elif axes.shape != (2, 1) and axes.shape != (2,):
            raise ValueError("axes must have shape (2, 1) or (2, ).")

        if self.n > 1:
            raise NotImplementedError(
                "Plotting BTSP ramp only implemented for 1 neuron."
            )

        t = np.asarray(self.history["t"])
        BTSP_ramp = np.asarray(self.history["BTSP_ramp"])[:, 0]  # 1st neuron only

        ax1D = np.asarray(axes).ravel()
        ax1D[0].plot(t, BTSP_ramp, lw=1.2, color=self.color)
        ax1D[0].fill_between(
            t,
            np.zeros(len(t)),
            BTSP_ramp,
            lw=0,
            alpha=0.2,
            color=self.color,
        )
        ax1D[0].set_ylabel("Prop. of BTSP\nthreshold reached")
        ax1D[0].spines[["top", "right"]].set_visible(False)
        ax1D[0].axhline(1, ls="dashed")

        ax1D[1].plot(t, self.history["firingrate"], lw=1.2, color=self.color)
        ax1D[1].axhline(self.BTSP_induction_threshold, ls="dashed", color=self.color)
        ax1D[1].set_xlabel("Time (s)")
        ax1D[1].set_ylabel("Firing rates")
        ax1D[1].spines[["top", "right"]].set_visible(False)

        i = 0
        num_steps_for_plateau = int(np.ceil(self.BTSP_plateau_length / self.Agent.dt))
        labels = ["BTSP event", "insufficient"]
        while i < len(BTSP_ramp):
            next_BTSP_possible = np.where(BTSP_ramp[i : i + num_steps_for_plateau] > 0)[
                0
            ]
            if len(next_BTSP_possible):
                next_BTSP_possible = next_BTSP_possible[0] + i
                BTSP_slice = slice(
                    next_BTSP_possible, next_BTSP_possible + num_steps_for_plateau + 1
                )
                i = next_BTSP_possible + num_steps_for_plateau + 1
                if (BTSP_ramp[BTSP_slice] >= 1).any():
                    alpha, l = 0.5, 0
                    i += 1
                else:
                    alpha, l = 0.2, 1

                start = max(0, next_BTSP_possible - 1)
                end = min(len(t) - 1, next_BTSP_possible + num_steps_for_plateau)
                ax1D[1].axvspan(
                    t[start],
                    t[end],
                    alpha=alpha,
                    lw=0,
                    label=labels[l],
                    color=self.color,
                )

                labels[l] = None
            else:
                i += num_steps_for_plateau

        if plot_events:
            BTSP_events = np.asarray(self.history["BTSP_events"])
            if len(BTSP_events):
                y = 1
                ax1D[0].scatter(
                    t[np.asarray(BTSP_events)],
                    np.full(len(BTSP_events), y),
                    color=(self.color or "k"),
                    alpha=0.8,
                    marker=markers.MarkerStyle("x"),
                    s=10,
                )
        if BTSP_ramp.max() < 0:
            ax1D[1].legend()

        fig = ax1D[0].figure

        util.save_figure(fig, f"{self.name}_BTSP_ramp", save=autosave)  # type: ignore[attr-defined]

        return axes

    def add_BTSP_markers_to_plots(
        self,
        ax: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | plt.Axes,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        timeseries: bool = False,
        color: str | None = None,
    ):
        """
        self.add_BTSP_markers_to_plots()

        Adds BTSP markers to timeseries or rate map plots.

        Args:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots. Single subplot
            if plotting timeseries or 1D rate map. Otherwise, an array of subplots.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - timeseries (bool, optional): Whether the plot is timeseries (map expected,
            otherwise). Default is False.
        - color (str, optional): Color of the BTSP markers. Default is None.
        """

        if color is None:
            color = self.color or "C1"

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type]
        num_neurons = len(chosen_neurons)

        t, startid, _ = self.get_plotting_times(t_start=t_start, t_end=t_end)

        BTSP_events = np.asarray(self.history["BTSP_events"]) - startid
        BTSP_targets = self.history["BTSP_targets"]

        ax = np.asarray(ax).ravel()
        if timeseries or self.Agent.Environment.dimensionality == "1D":
            if len(ax) != 1:
                raise ValueError(
                    "Only one axis expected for timeseries or 1D rate map."
                )
            sub_ax = ax

        for event, targets in zip(BTSP_events, BTSP_targets):
            for target in targets:
                if event >= len(t):
                    continue
                elif event < 0:
                    alpha = 0.6
                else:
                    alpha = 1.0

                if target not in chosen_neurons:
                    continue

                i = chosen_neurons.index(target)

                if timeseries:
                    if event < 0:
                        continue
                    x_pos = t[event] / 60
                    line_sep = (sub_ax.get_ylim()[1] - 1) / num_neurons
                    y_pos = 1 + line_sep * i + line_sep * 0.7
                    pos = [x_pos, y_pos]
                else:
                    pos = self.Agent.history["pos"][event + startid]
                    if self.Agent.Environment.dimensionality == "1D":
                        line_sep = (sub_ax.get_ylim()[1] - 1) / num_neurons
                        y_pos = 1 + line_sep * i + line_sep * 0.7
                        pos = pos + [y_pos]
                    else:
                        sub_ax = ax[i]

                sub_ax.scatter(
                    *pos,
                    color=color,
                    alpha=alpha,
                    marker=markers.MarkerStyle("x"),
                    s=10,
                )

    def plot_rate_map(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        mark_BTSP: bool = True,
        ax: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes | np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]:
        """
        self.plot_rate_map()

        Plot the rate map of the layer, ensuring no more than 20 columns are plotted.

        See SmoothFeedForwardLayer.plot_rate_map() for more information.

        Args:
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - mark_BTSP (bool, optional): Whether to include BTSP markers
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots. Single subplot
            if plotting timeseries or 1D rate map. Otherwise, an array of subplots.
        - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to FeedForwardLayer.plot_rate_map().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots with rate maps
            plotted.
        """

        ax_out = super().plot_rate_map(
            t_start=t_start,
            t_end=t_end,
            chosen_neurons=chosen_neurons,
            ax=ax,
            autosave=False,
            **kwargs,
        )

        if ax is None:
            ax = ax_out

        if mark_BTSP:
            self.add_BTSP_markers_to_plots(
                ax=ax,
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                timeseries=False,
            )

        fig = ax[0].figure
        util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

        return ax

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        xlim: tuple[float, float] | None = None,
        color: str | None = None,
        mark_BTSP: bool = True,
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        self.plot_rate_timeseries()

        Plot the rate timeseries of the layer.

        See SmoothFeedForwardLayer.plot_rate_timeseries() for more information.

        Args:
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time. Default is None.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - xlim (tuple[float, float], optional): The x limits of the plot. Default is None.
        - color (str, optional): The color of the plot. Default is None.
        - mark_BTSP (bool, optional): Whether to include BTSP markers
        - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to SmoothFeedForwardLayer.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot withtimeseries plotted.
        """

        sub_ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            sub_ax=sub_ax,
            chosen_neurons=chosen_neurons,
            xlim=xlim,
            color=color,
            autosave=False,
            **kwargs,
        )

        if mark_BTSP:
            self.add_BTSP_markers_to_plots(
                ax=sub_ax,
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                timeseries=True,
                color=color,
            )

        if xlim is not None:
            sub_ax.set_xlim(xlim)

        fig = sub_ax.figure
        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class NMDACurrent(object):
    """
    NMDACurrent()

    Class implementing an NMDA current to a layer of neurons

    Must be initialised with an NMDA input layer that the current will apply to.

    List of properties:
        • self.activation_params
        • self.activation_function
        • self.binding_params
        • self.binding_function

    List of methods:
        • self.get_decay()
        • self.get_state()
        • self.update()
        • self.plot_function()
        • self.plot_timeseries()
    """

    def __init__(
        self,
        InputLayer,
        name="NMDACurrent",
        NMDA_activation_decay_tau=0.1,  # seconds
        NMDA_desensitization_decay_tau=0.3,  # seconds
        NMDA_activation_threshold=0.8,  # firing rate
        max_current=3.0,
        start_desensitization=0,
        test_activation=0.3,
        color="C5",
        save_history=True,
    ):
        """
        NMDACurrent(InputLayer)

        Initialise an NMDA current object, with a history.

        Attributes (in addition to the arguments below):
        - self.n (int): Number of neurons in the input layer.
        - self.firingrate (1D np.ndarray): Firing rates of the neurons.
        - self.NMDA_receptor_binding (1D np.ndarray): NMDA receptor binding levels
            for each neuron.
        - self.NMDA_receptor_activation (1D np.ndarray): NMDA receptor activation
            levels for each neuron.
        - self.NMDA_receptor_desensitization (1D np.ndarray): NMDA receptor
            desensitization levels for each neuron.

        Args:
        - InputLayer (NMDALayer): Input layer.
        - name (str, optional): Name of the NMDA current. Default is "NMDACurrent".
        - NMDA_activation_decay_tau (float, optional): Time constant for the decay of
            NMDA receptor activation. Default is 0.1.
        - NMDA_desensitization_decay_tau (float, optional): Time constant for the
            decay of NMDA receptor desensitization. Default is 0.3.
        - NMDA_activation_threshold (float, optional): Threshold for NMDA receptor
            activation. Default is 0.8.
        - max_current (float, optional): Maximum current. Default is 3.0.
        - start_desensitization (float, optional): Initial desensitization level.
            Default is 0.
        - test_activation (float, optional): Activation level for testing.
            Default is 0.3.
        - color (str, optional): Color of the NMDA current. Default is "C5".
        - save_history (bool, optional): Whether to save the history.
            Default is True.

        Raises:
        - TypeError: If the input layer is not of type NMDALayer.
        """

        self.name = name
        self.InputLayer = InputLayer
        self.Agent = self.InputLayer.Agent
        self.n = self.InputLayer.n
        self.firingrate = np.zeros(self.InputLayer.n)  # type: ignore[attr-defined]

        if not isinstance(self.InputLayer, NMDALayer):
            raise TypeError(
                f"InputLayer must be of type NMDALayer, not {type(self.InputLayer)}"
            )

        self.NMDA_activation_decay_tau = NMDA_activation_decay_tau
        self.NMDA_desensitization_decay_tau = NMDA_desensitization_decay_tau
        self.NMDA_activation_threshold = NMDA_activation_threshold
        self.max_current = max_current
        self.start_desensitization = start_desensitization
        self.test_activation = test_activation
        self.color = color

        self.NMDA_receptor_binding = np.zeros(self.n)
        self.NMDA_receptor_activation = np.zeros(self.n)
        self.NMDA_receptor_desensitization = (
            np.ones(self.n) * self.start_desensitization
        )

        self.save_history = save_history

        if save_history:
            self.history = dict()
            self.history["t"] = list()
            self.history["current"] = list()
            self.history["receptor_binding"] = list()
            self.history["receptor_activation"] = list()
            self.history["receptor_desensitization"] = list()

    @property
    def activation_params(self):
        """
        self.activation_params

        Obtain the parameters for the activation function of the NMDA current.

        Returns:
        - (dict): Activation parameters, with keys and values:
            - "activation" (str): Type of activation function.
            - "min_fr" (float): Minimum firing rate.
            - "max_fr" (float): Maximum firing rate.
            - "width_x" (float): Width of the sigmoid.
            - "mid_x" (float): Midpoint of the sigmoid.
        """

        self._activation_params = {
            "activation": "sigmoid",
            "min_fr": 0.0,
            "max_fr": 1.0,
            "width_x": 0.5,
            "mid_x": self.NMDA_activation_threshold,
        }

        return self._activation_params

    @property
    def activation_function(self):
        """
        self.activation_function

        Obtain the activation function of the NMDA current.

        Returns:
        - (function): Activation function.
        """

        return lambda x, deriv=False: rutils.activate(
            x, deriv=deriv, other_args=self.activation_params
        )

    @property
    def binding_params(self):
        """
        self.binding_params

        Obtain the parameters of the NMDA current's NMDA receptor binding function.

        Returns:
        - (dict): NMDA receptor binding parameters with keys and values:
            - "activation" (str): Type of activation function.
            - "min_fr" (float): Minimum firing rate.
            - "max_fr" (float): Maximum firing rate.
            - "width_x" (float): Width of the sigmoid.
        """

        self._binding_params = {
            "activation": "sigmoid",
            "min_fr": 0.0,
            "max_fr": 1.0,
            "width_x": 20,
        }
        self._binding_params["mid_x"] = self._binding_params["width_x"] / 1.5

        return self._binding_params

    @property
    def binding_function(self):
        """
        self.binding_function

        Obtain NMDA receptor binding function for the NMDA current.

        Returns:
        - (function): NMDA receptor binding function.
        """
        return lambda x, deriv=False: rutils.activate(
            x, deriv=deriv, other_args=self.binding_params
        )

    def get_decay(self, decay_type: str = "activation", dt: float | None = None):
        """
        self.get_decay()

        Obtain the decay factor for the NMDA receptor activation or desensitization.

        Args:
        - decay_type (str, optional): Type of decay
           ("activation" or "desensitization"). Default is "activation".
        - dt (float, optional): Time step. If None, agent time step is used.
            Default is None.

        Returns:
        - decay (float): Decay factor.
        """

        if decay_type == "activation":
            tau = self.NMDA_activation_decay_tau
        elif decay_type == "desensitization":
            tau = self.NMDA_desensitization_decay_tau
        else:
            raise ValueError(f"Unknown decay type {decay_type}")

        if dt is None:
            dt = float(self.Agent.dt)

        decay = np.exp(-dt / tau)

        return decay

    def get_state(
        self,
        evaluate_at="last",
        return_all: bool = False,
        dt: float | None = None,
        **catchall_kwargs,
    ):
        """
        self.get_state()

        Obtain the state of the NMDA current.

        Args:
        - evaluate_at (str, optional): Whether to evaluate the state at the last time
            step or the current time step. Default is "last".
        - return_all (bool, optional): Whether to return all state variables.
            Default is False.
        - dt (float, optional): Time step. Default is None.

        Keyword args:
        - **catchall_kwargs: Ignored keyword arguments (hacky!)

        Returns:
        - current (1D np.ndarray): State of the NMDA current for each neuron.

        and, if return_all:
        - receptor_binding (1D np.ndarray): Receptor binding for each neuron.
        - receptor_activation (1D np.ndarray): Receptor activation for each neuron.
        - receptor_desensitization (1D np.ndarray): Receptor desensitization for each
            neuron.
        """

        receptor_binding = self.binding_function(
            self.InputLayer.get_incoming_firingrates(),
        )  # rethink how to measure receptor binding (considering weights / biases)

        if evaluate_at == "last":
            # biexponential decay: loss of activation and desensitization
            receptor_activation = self.NMDA_receptor_activation * self.get_decay(
                "activation", dt=dt
            )
            new_desensitization = self.NMDA_receptor_activation - receptor_activation
            receptor_desensitization = (
                self.NMDA_receptor_desensitization
                * self.get_decay("desensitization", dt=dt)
                + new_desensitization
            )

            # compute proportion of sites that can be bound with a potential to be
            # activated (neither currently active, nor inactive)
            activatable_sites = (
                np.ones_like(receptor_desensitization)
                - receptor_activation
                - receptor_desensitization
            )
            effective_binding = activatable_sites * receptor_binding

            # compute additional activation, based on effective binding and level of
            # depolarization of the neuron
            neuron_activity_gate = self.activation_function(self.InputLayer.firingrate)

            receptor_activation += effective_binding * neuron_activity_gate
        else:
            receptor_activation = np.minimum(
                receptor_binding, self.test_activation
            )  # avoid saturation when evaluating rate maps

            if evaluate_at == "all":
                pos_dim = self.Agent.Environment.flattened_discrete_coords.shape[0]
            else:
                pos_dim = kwargs["pos"].shape[0]

            receptor_activation = np.repeat(receptor_activation, pos_dim).reshape(
                -1, pos_dim
            )

        current = receptor_activation * self.max_current  # type: ignore[operator]

        if evaluate_at == "last" and return_all:
            return (
                current,
                receptor_binding,
                receptor_activation,
                receptor_desensitization,  # type: ignore[unbound]
            )
        else:
            return current

    def update(self):
        """
        self.update()

        Update the NMDA current, and history.

        Attributes:
        - self.firingrate (1D np.ndarray): Firing rates of the neurons.
        - self.NMDA_receptor_binding (1D np.ndarray): NMDA receptor binding levels
            for each neuron.
        - self.NMDA_receptor_activation (1D np.ndarray): NMDA receptor activation
            levels for each neuron.
        - self.NMDA_receptor_desensitization (1D np.ndarray): NMDA receptor
            desensitization levels for each neuron.
        """

        current, receptor_binding, receptor_activation, receptor_desensitization = (
            self.get_state(evaluate_at="last", return_all=True)
        )

        self.firingrate = current
        self.NMDA_receptor_binding = receptor_binding
        self.NMDA_receptor_activation = receptor_activation
        self.NMDA_receptor_desensitization = receptor_desensitization

        if self.save_history:
            self.history["t"].append(self.Agent.t)
            self.history["current"].append(current.tolist())
            self.history["receptor_binding"].append(receptor_binding.tolist())
            self.history["receptor_activation"].append(receptor_activation.tolist())
            self.history["receptor_desensitization"].append(
                receptor_desensitization.tolist()
            )

    def plot_function(
        self,
        param_type="binding",
        min_input_fr=-15,
        max_input_fr=15,
        sub_ax=None,
    ):
        """
        self.plot_function()

        Plot the NMDA receptor binding or activation function.

        Args:
        - param_type (str, optional): Type of function to plot. Default is "binding".
        - min_input_fr (int, optional): Minimum input firing rate to plot from.
            Default is -15.
        - max_input_fr (int, optional): Maximum input firing rate to plot from.
            Default is 15.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with binding or activation function plotted.
        """

        if param_type == "binding":
            function = self.binding_function
        elif param_type == "activation":
            function = self.activation_function
        else:
            raise ValueError(f"Unknown param type {param_type}")

        sub_ax = plot_util.plot_activation_function(
            function,
            min_input_fr=min_input_fr,
            max_input_fr=max_input_fr,
            sub_ax=sub_ax,
            color=self.color,
        )

        sub_ax.set_title(f"{param_type.capitalize()} function")

        return sub_ax

    def plot_timeseries(
        self,
        t_start: float | None = None,
        title: str | None = None,
        datatypes: list[str] | str = "current",
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        ax: plt.Axes | np.ndarray[plt.Axes, np.dtype[np.int64]] | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> np.ndarray[plt.Axes, np.dtype[np.int64]]:
        """
        self.plot_timeseries()

        Plot the current, and optionally the receptor binding, activation and
        densensitization time series.

        Args:
        - input_layer_name (str): Name of the input layer to plot. Default is None.
        - t_start (float, optional): Start time of the plot. Default is None.
        - title (str, optional): Title of the plot. Default is None.
        - datatypes (list or str, optional): Type of data to plot. Default is "current".
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - ax (plt.Axes or np.ndarray): Subplot of array of subplots to plot on
           (one per datatype). Default is None.
        - autosave (bool, optional): Whether to save the plot. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to plot_util.plot_timeseries().

        Returns:
        - ax (plt.Axes or np.ndarray): Subplot or array of subplots with timeseries
            plotted (one for each datatype).
        """

        if title is None:
            title = "NMDA current traces"

        all_datatypes = [
            "receptor_binding",
            "receptor_desensitization",
            "current",
            "receptor_activation",
        ]
        if isinstance(datatypes, str):
            if datatypes == "all":
                datatypes = all_datatypes
            else:
                datatypes = [datatypes]

        chosen_neurons = np.asarray(
            self.InputLayer.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type, attr-defined]
        )

        if ax is None:
            n = len(chosen_neurons)
            height = max([1, min(n / 12.0 + 5 / 3, 8)]) * len(datatypes)
            _, ax = plt.subplots(
                len(datatypes),
                1,
                sharex=True,
                sharey=False,
                figsize=[6, height],
                squeeze=False,
            )  # type: ignore[assignment]

        axis1D = np.asarray(ax).ravel()

        if len(axis1D) != len(datatypes):
            raise ValueError(
                f"Number of subplots ({len(axis1D)}) does not match number of datatypes "
                f"to plot ({len(datatypes)})."
            )

        for d, datatype in enumerate(datatypes):
            if datatype not in all_datatypes:
                raise ValueError(
                    f"Datatype {datatype} not recognized. "
                    f"Must be one of {all_datatypes}."
                )
            plot_util.plot_timeseries(
                self,  # type: ignore[assignment]
                t_start=t_start,
                sub_ax=axis1D[d],
                trace_name=datatype,
                chosen_neurons=chosen_neurons,
                autosave=False,
                **kwargs,
            )

            axis1D[d].set_ylabel(datatype.capitalize().replace("_", " "))
            if d != len(datatypes) - 1:
                axis1D[d].set_xlabel("")

        fig = axis1D[0].figure
        fig.suptitle(title, y=0.90)

        util.save_figure(fig, f"{self.name}_NMDA_current_traces", save=autosave)  # type: ignore[attr-defined]

        return ax


class NMDALayer(BTSPLayer):
    """
    NMDALayer()

    Class extending BTSPLayer. Defines a population of neurons that tune their
    weights through Hebbian learning with BTSP and NMDA receptors.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 10,
        "name": "NMDALayer",
        "NMDA_activation_threshold": 2,
        "BTSP_induction_threshold": 8,
        "BTSP_plateau_length": 0.12,  # seconds
    }

    See BTSPLayer for properties.

    List of methods (in addition to BTSPLayer methods):
        • self.save_to_history()
        • self.get_incoming_firingrates()
        • self.get_BTSP_targets()
        • self.update()
        • self.plot_BTSP_ramp()
    """

    default_params = {
        "n": 10,
        "name": "NMDALayer",
        "NMDA_activation_threshold": 2,
        "BTSP_induction_threshold": 8,
        "BTSP_plateau_length": 0.12,  # seconds
    }

    ignored_param_keys = list()
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """
        NMDALayer(Agent)

        Initialise HebbianLayer(), takes as input a parameter
        dictionary. Any values not provided by the params dictionary are
        taken from a default dictionary below.

        Initialise a layer that can learn weight updates using BTSP and NMDA receptors.
        BTSP ramp is added to the history.

        Attributes:
        - BTSP_ramp (1D np.ndarray): BTSP ramp of the layer.

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict()."""

        self.Agent = Agent

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        self._check_BTSP_plateau_length()
        self._add_NMDA_current()

        self.BTSP_ramp = np.zeros(self.n).astype(float)  # type: ignore[attr-defined]
        self.history["BTSP_ramp"] = list()

    def _check_BTSP_plateau_length(self):
        """
        self._check_BTSP_plateau_length()

        Check that the BTSP plateau length is a multiple of the Agent's time step.

        Raises:
        - ValueError: If the BTSP plateau length is not a multiple of the Agent's
            time step.
        """

        num_steps = self.BTSP_plateau_length / self.Agent.dt
        if not np.isclose(num_steps, int(num_steps)):
            low_plateau_length = np.floor(num_steps) * self.Agent.dt
            high_plateau_length = np.ceil(num_steps) * self.Agent.dt
            raise ValueError(
                f"BTSP_plateau_length must be a multiple of the Agent's dt ({self.Agent.dt}). "
                f"Try {low_plateau_length} or {high_plateau_length}."
            )

    def _add_NMDA_current(self):
        """
        self._add_NMDA_current()

        Add the NMDA current as a non-learning input to the layer.

        Returns:
        - NMDACurrent (NMDACurrent): NMDA current object.
        """

        self.NMDACurrent = NMDACurrent(
            self,
            name="NMDACurrent",
            NMDA_activation_threshold=self.NMDA_activation_threshold,  # type: ignore[attr-defined]
            color=self.color,  # type: ignore[attr-defined]
            save_history=self.save_history,  # type: ignore[attr-defined]
        )

        self.add_input_layers_with_no_learning(self.NMDACurrent.name)
        self.add_input(self.NMDACurrent, w=np.eye(self.n), recurrent=True)  # type: ignore[attr-defined]

        return self.NMDACurrent

    def save_to_history(self):
        """
        self.save_to_history()

        Save the current state of the layer to the history, including the BTSP ramp.
        """

        super().save_to_history()

        self.history["BTSP_ramp"].append(self.BTSP_ramp.tolist())

        return

    def get_incoming_firingrates(self, evaluate_at="last", **kwargs):
        """
        self.get_incoming_firingrates()

        Obtain the firing rates coming into each of the layer's neurons. By default
        this method uses the last saved firing rates from its input layers.

        Alternatively 'evaluate_at' and 'kwargs' can be used to retrieve a different
        state from the input layers.

        NOTE: This means that pre-synaptic weights is ignored, and all input neurons
        are equipotent. May have to be rethought?

        Args:
        - evaluate_at (str, optional). Default is 'last'.

        Keyword args:
        - **kwargs: Keyword arguments passed to each input layer's get_state() method.

        Returns:
        - V (1D np.ndarray): Sum of input firing rates reaching each layer neuron.
        """

        n = int(self.n)  # type: ignore[attr-defined]

        if evaluate_at == "last":
            V = np.zeros(n)
        elif evaluate_at == "all":
            V = np.zeros((n, self.Agent.Environment.flattened_discrete_coords.shape[0]))
        else:
            V = np.zeros((n, kwargs["pos"].shape[0]))

        for name, inputlayer in self.inputs.items():
            if name == self.NMDACurrent.name:
                continue
            w_ones = np.ones_like(inputlayer["w"])
            if evaluate_at == "last":
                I = inputlayer["layer"].firingrate
            else:
                I = inputlayer["layer"].get_state(evaluate_at, **kwargs)
            inputlayer["I"] = I
            V += np.matmul(w_ones, I)

        return V

    def get_BTSP_targets(
        self,
        BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] | None = list(),
    ):
        """
        self.get_BTSP_targets()

        Obtain the BTSP targets for the layer.

        Args:
        - BTSP_targets (list or 1D np.ndarray, optional): BTSP targets
            (neuron indices). Default is list().

        Returns:
        - BTSP_targets (1D np.ndarray): BTSP targets (neuron indices).
        """

        above_threshold = self.firingrate > self.BTSP_induction_threshold  # type: ignore[attr-defined]
        self.BTSP_ramp[~above_threshold] = 0
        self.BTSP_ramp[above_threshold] += self.Agent.dt / self.BTSP_plateau_length  # type: ignore[attr-defined]

        BTSP_targets = np.where(np.isclose(self.BTSP_ramp, 1))[
            0
        ]  # only once per plateau

        return BTSP_targets

    def update(self):
        """
        self.update()

        Update the layer, starting with the input NMDA current.
        """

        self.NMDACurrent.update()
        super().update()

    def plot_BTSP_ramp(self, axes=None, autosave=None):
        """
        self.plot_BTSP_ramp()

        Plot the BTSP ramp of the layer.

        Args:
        - axes (1 or 2D np.ndarray): Array of subplots to plot on,
            with shape (3, 1) or (3, ). Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

        Returns:
        - axes (1 or 2D np.ndarray): Array of subplots with BTSP ramp variables plotted.
            If input axes was None, shape is 2D (3, 1).
        """

        if axes is None:
            _, axes = plt.subplots(3, 1, figsize=[9, 7], sharex=True, squeeze=False)
        elif axes.shape != (3, 1) and axes.shape != (3,):
            raise ValueError("axes must have shape (3, 1) or (3, ).")

        super().plot_BTSP_ramp(axes=axes[:2], autosave=False)

        ax1D = np.asarray(axes).ravel()
        ax1D[2].plot(
            self.history["t"],
            self.inputs["NMDACurrent"]["layer"].history["current"],
            lw=1.2,
            color=self.color,
        )
        ax1D[1].set_xlabel("")
        ax1D[2].set_xlabel("Time (s)")
        ax1D[2].set_ylabel("CA1 NMDA current")
        ax1D[2].spines[["top", "right"]].set_visible(False)

        fig = ax1D[0].figure
        util.save_figure(fig, f"{self.name}_BTSP_ramp", save=autosave)  # type: ignore[attr-defined]

        return axes
