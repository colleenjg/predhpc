import copy
from typing import TYPE_CHECKING, Any, Sequence
import warnings

from matplotlib import pyplot as plt
from matplotlib import markers as mpl_markers
from matplotlib.colors import ListedColormap
import numpy as np

from ratinabox import utils as rutils

from predhpc import plot_fcts
from predhpc.neurons import riab_neurons
from predhpc.util import (
    ext_util,
    gen_util,
    signal_util,
    plot_util,
    params_util,
    learn_util,
    hyper_util,
)

if TYPE_CHECKING:
    import ratinabox


class SmoothFeedForwardLayer(riab_neurons.FeedForwardLayer):
    """
    SmoothFeedForwardLayer()

    Class extending riab_neurons.FeedForwardLayer. Defines a population of neurons
    that receive feedforward input that is smoothed.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 10,
        "activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_PARAMS,
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
        • self.update()
        • self.plot_activation_function()
        • self.plot_firingrate_distribution()
        • self.plot_filtered()
    """

    default_params = {
        "n": 10,
        "activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_PARAMS,
        "name": "SmoothFeedForwardLayer",
        "input_filter_tau": 0.1,  # in sec
        "input_trend_tau": None,  # in sec
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

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
            self.inputs[name_in]["filtered_inputs"] = np.zeros(n_in)
            self.inputs[name_in]["filtered_trends"] = np.zeros(n_in)

            for key in ["inputs", "trends"]:
                if f"filtered_{key}" not in self.history.keys():
                    self.history[f"filtered_{key}"] = dict()
                self.history[f"filtered_{key}"][name_in] = list()

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
                inputlayer["I"] = I
            else:  # recursive call
                I = inputlayer["layer"].get_state(
                    evaluate_at=evaluate_at,
                    max_recurrence=pass_max_recurrence,
                    **kwargs,
                )

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

    def _update_filtered_inputs(
        self,
        filter_tau: float | None = None,
        trend_tau: float | None = None,
        filter_key: str = "filtered_inputs",
        peak_hold: bool = False,
    ):
        """
        self._update_filtered_inputs()

        Update the filtered inputs for the layer.

        Args:
        - filter_tau (float, optional): Filter time constant. Default is None.
        - trend_tau (float, optional): Trend time constant. Default is None.
        - filter_key (str, optional): Key of the filter to update.
            Default is "filtered_inputs".
        - peak_hold (bool, optional): Whether to use peak hold. Default is False.
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
            if peak_hold:
                input_layer[filter_key] = np.maximum(input_layer[filter_key], I_t1)
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
            self._update_filtered_inputs(
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

    def plot_firingrate_distribution(
        self, sub_ax=None, bins=50, t_start=None, t_end=None
    ):
        """
        self.plot_firingrate_distribution()

        Plot the firing rate distribution of the layer as a histogram.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - bins (int, optional): Number of histogram bins. Default is 50.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with firing rate distribution plotted.
        """

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(4, 2))

        _, startid, endid = self.get_plotting_times(t_start, t_end)

        firingrates = np.asarray(self.history["firingrate"])[
            startid : endid + 1
        ].ravel()
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
        rasterize_shading: bool = False,
        plot_unfiltered: bool = True,
        in_min: bool = True,
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
        - rasterize_shading (bool, optional): Whether to rasterize the shading,
            reducing the size of exported vector files. Default is False.
        - plot_unfiltered (bool, optional): Whether to plot the unfiltered signal.
            Default is True.
        - in_min (bool, optional): Whether to plot the time in minutes instead of
            seconds. Default is True.
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
        if in_min:
            t = t / 60.0

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

        range_val = filtered.max() - filtered.min()
        if plot_unfiltered:
            range_val = max(range_val, (unfiltered.max() - unfiltered.min()))

        height = 0.6 * range_val
        shifts = np.arange(filtered.shape[1]).reshape(1, -1) * height

        if sub_ax is None:
            n = unfiltered.shape[1]
            height = max([1, min(n / 12.0 + 5 / 3, 8)])
            _, sub_ax = plt.subplots(figsize=[6, height])

        color = layer.color  # type: ignore[attr-defined]

        if plot_unfiltered:
            sub_ax.plot(t, unfiltered + shifts, ls=(0, (1, 1)), color=color, alpha=1.0)
        sub_ax.plot(t, filtered + shifts, alpha=0.8, color=color)

        for i, shift in enumerate(shifts.T):
            sub_ax.fill_between(
                t,
                shift,
                filtered[:, i] + shift,
                color=color,
                alpha=0.4,
                lw=0,
                rasterized=rasterize_shading,  # svg too big, othersize
            )

        sub_ax.spines[["top", "right"]].set_visible(False)
        sub_ax.set_ylabel("Firing rate")

        xlabel = "Time (min)" if in_min else "Time (s)"
        sub_ax.set_xlabel(xlabel)

        if title is None:
            title = filter_key.replace("_", " ").capitalize().replace("btsp", "BTSP")
        sub_ax.set_title(title)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_{filter_key}", save=autosave)  # type: ignore[attr-defined]

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

        self.history["target_MSE"] = list()
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

    @property
    def input_layers_with_learning(self) -> list[str]:
        """Obtain a list of input layer names that are learning.

        Returns:
        - (list): List of input layer names that are learning.
        """
        input_layers_with_learning = [
            input_layer_name
            for input_layer_name in self.inputs.keys()
            if input_layer_name not in self.input_layers_with_no_learning
        ]
        return input_layers_with_learning

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
        self.inputs[name_in]["filtered_inputs_for_learning"] = np.zeros(n_in)
        self.inputs[name_in]["filtered_trends_for_learning"] = np.zeros(n_in)

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
            self.history["target_MSE"].append(np.nan)
        else:
            target_MSE = np.mean((self.target - self.firingrate) ** 2)
            self.history["target_MSE"].append(target_MSE)

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

        with ext_util.TemporarilyChangeAttribute(self, "save_history", False):
            super().update()

        if self.learn:
            self.update_weights(**kwargs)

        if self.save_history is True:
            self.save_to_history()

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

        ax_out = super().plot_rate_map(autosave=False, ax=ax, **kwargs)

        if ax is None:
            ax = ax_out

        if no_legend:
            for sub_ax in np.asarray(ax).ravel():
                if sub_ax.get_legend() is not None:
                    sub_ax.get_legend().remove()

        fig = np.asarray(ax).ravel()[0].figure
        plot_util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

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
            end = min([len(t) - 1, start + n_pts])
            t_end = t[end]
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
            cbar = plot_util.normalize_cmaps(subplots, shrink=0.7)
            cbar.set_label("Firing rate (Hz)")

        fig = np.asarray(axes).ravel()[0].figure

        if title is None:
            title = "Rate maps across learning"

        y = 0.9 if self.Agent.Environment.dimensionality == 1 else 0.97
        fig.suptitle(title, y=y)

        plot_util.save_figure(fig, f"{self.name}_rate_maps_across_learning", save=autosave)  # type: ignore[attr-defined]

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
        in_min: bool = True,
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
        - in_min (bool, optional): Whether to plot the time in minutes instead of
            seconds. Default is True.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **loss_**kwargs: Keyword arguments passed to plot_fcts.plot_loss().

        Returns:
        - sub_ax (plt.Axes): Subplot with loss plotted.
        """

        reset_times = None
        if hasattr(self.Agent, "trajectory_df"):
            reset_times = self.Agent.get_reset_times()  # type: ignore[attr-defined]

        if color is None:
            color = self.color  # type: ignore[attr-defined]

        sub_ax = plot_fcts.plot_loss(
            self.history["t"],
            self.history["target_MSE"],
            mark_ts=reset_times,
            t_start=t_start,
            t_end=t_end,
            sub_ax=sub_ax,
            color=color,
            alpha=alpha,
            xlim=xlim,
            k_prop_to_loss_length=k_prop_to_loss_length,
            k_max=k_max,
            in_min=in_min,
            **loss_kwargs,
        )

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_loss", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_histogram(
        self,
        sub_ax: plt.Axes | None = None,
        color: str | None = None,
        alpha: float = 0.7,
        t_start: float | None = None,
        t_end: float | None = None,
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
        - t_end (float, optional): End timepoint of the plot. Default is None.
        - xlabel (str, optional): x label of the plot. Default is None.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot withhistogram plotted.
        """

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(8, 3))

        _, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)
        firingrates = np.asarray(self.history["firingrate"])[startid : endid + 1]

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

        plot_util.save_figure(fig, f"{self.name}_firing_rate_histogram", save=autosave)  # type: ignore[attr-defined]

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
        • self.get_normalization_value_dict()
        • self.get_normalization_values()
        • self.log_max_normalization_value()
        • self.get_learning_kernel_kwargs()
        • self.get_learning_kernel()
        • self.save_to_history()
        • self.update_weights()
        • self.update()
        • self.plot_learning_kernel()
        • self.plot_normalization_values()
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

        if self.trainable_biases:
            if self.normalize_weights_divisively or self.apply_Ojas_rule:
                self.b_norm = None
                self.history["b_norm"] = list()
            if self.normalize_weights_divisively:
                self.b_norm_inits = list()

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

        name_in = input_layer.name  # type: ignore[attr-defined]
        if self.normalize_weights_divisively:
            self.update_weights(filter_key="I", lr=0)  # normalize (no update)
            self.inputs[name_in]["w_init"] = copy.deepcopy(self.inputs[name_in]["w"])
            if "w_norm" in self.inputs[name_in].keys():
                self.inputs[name_in]["w_init_norm"] = self.inputs[name_in]["w_norm"]
                if self.trainable_biases:
                    self.b_norm_inits.append(self.b_norm)

        if self.normalize_weights_divisively or self.apply_Ojas_rule:
            if "w_norm" not in self.history.keys():
                self.history["w_norm"] = dict()
            self.history["w_norm"][name_in] = list()

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

    def get_normalization_value_dict(self):
        """
        self.get_normalization_value_dict()

        Obtain the normalization values for the weights of the layer.

        Returns:
        - norm_values (dict): Dictionary of normalization values.
        """

        if not (self.apply_Ojas_rule or self.normalize_weights_divisively):  # type: ignore[attr-defined]
            raise RuntimeError("Normalization not applied.")

        norm_values = dict()
        for name_in, input_layer in self.inputs.items():
            if "w_norm" in input_layer.keys() or "w_init_norm" in input_layer.keys():
                if name_in not in norm_values.keys():
                    norm_values[name_in] = dict()
                if "w_init_norm" in input_layer.keys():
                    norm_values[name_in]["w_init_norm"] = input_layer["w_init_norm"]
                if "w_norm" in input_layer.keys():
                    outputs = list(
                        zip(
                            *[
                                (i, val)
                                for i, val in enumerate(self.history["w_norm"][name_in])
                                if val is not None
                            ]
                        )
                    )
                    if len(outputs):
                        steps, w_norm_vals = outputs
                        norm_values[name_in]["w_norm_steps"] = list(steps)
                        norm_values[name_in]["w_norms"] = list(w_norm_vals)

        if self.trainable_biases:
            if hasattr(self, "b_init_norms"):
                norm_values["b_init_norms"] = self.b_init_norms
            if hasattr(self, "b_norm"):
                outputs = list(
                    zip(
                        *[
                            (i, val)
                            for i, val in enumerate(self.history["b_norm"])
                            if val is not None
                        ]
                    )
                )
                if len(outputs):
                    steps, b_norm_vals = outputs
                    norm_values["b_norm_steps"] = list(steps)
                    norm_values[name_in]["b_norms"] = list(b_norm_vals)

        return norm_values

    def get_normalization_values(
        self,
        input_layer: str,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        bias_norms: bool = False,
    ):
        """
        self.get_normalization_values()

        Obtain the normalization values for the weights of the layer.

        Args:
        - input_layer (str): Name of the input layer.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - bias_norms (bool, optional): Whether to plot the bias normalization values.
            Default is False.

        Returns:
        - steps (1D np.ndarray): Steps of the normalization values.
        - norm_values (2D np.ndarray): Normalization values (steps x neurons).
        """

        norm_dict = self.get_normalization_value_dict()

        key_str = "b" if bias_norms else "w"

        norm_str = f"{key_str}_norms"
        init_str = f"{key_str}_init_norm"
        steps_str = f"{key_str}_norm_steps"

        if input_layer not in norm_dict.keys():
            raise KeyError(f"Input layer {input_layer} not found.")

        norm_dict = norm_dict[input_layer]
        if init_str not in norm_dict.keys():
            data_type = "weight" if not bias_norms else "bias"
            raise KeyError(f"No {data_type} normalization values found.")

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type]

        t, startid, stopid = self.get_plotting_times(
            t_start=t_start, t_end=t_end, raise_error=False
        )

        steps = [0]
        norm_values = [norm_dict[init_str]]
        if norm_str in norm_dict.keys():
            steps = np.asarray(steps + norm_dict[steps_str])
            norm_values = np.asarray(norm_values + norm_dict[norm_str])
        else:
            steps = np.asarray(steps)
            norm_values = np.asarray(norm_values)

        keep = np.where((steps >= startid) * (steps < stopid))[0]
        steps = steps[keep]
        norm_values = norm_values[keep]

        t = t[steps - startid]

        if bias_norms or self.normalize_weights_divisively:
            norm_values = norm_values[:, chosen_neurons]

        return steps, norm_values

    def log_max_normalization_value(self, input_layer, **kwargs):
        """
        self.log_max_normalization_value()

        Log the maximum normalization value of the weights and biases, if applicable.
        """

        _, norm_values = self.get_normalization_values(input_layer, **kwargs)

        if len(norm_values) == 0:
            log_str = "No weight normalization values recorded."

        else:
            max_norm = np.max(norm_values)

            if self.normalize_weights_divisively and max_norm <= 1:
                log_str = (
                    "No weight normalization applied "
                    f"(max value of {max_norm:.4f} <= 1)."
                )
            else:
                log_str = f"Max. weight normalization value applied: {max_norm:.4f}."

        print(log_str)

    def get_learning_kernel_kwargs(self):
        """
        self.get_learning_kernel_kwargs()

        Obtain the keyword arguments for the learning kernel.

        Returns
        - learning_kernel_kwargs (dict): Keyword arguments for the learning kernel.
        """

        learning_kernel_kwargs = {
            "pre_filter_tau": self.learning_filter_tau,  # type: ignore[attr-defined]
            "pre_trend_tau": self.learning_trend_tau,  # type: ignore[attr-defined]
            "post_filter_tau": self.learning_filter_tau,  # type: ignore[attr-defined]
            "post_trend_tau": self.learning_trend_tau,  # type: ignore[attr-defined]
        }

        return learning_kernel_kwargs

    def get_learning_kernel(self, **kwargs):
        """
        self.get_learning_kernel()

        Obtain the learning kernel.

        Keyword args:
        - **kwargs: Keyword arguments passed to plot_util.plot_pre_post_exponential().

        Returns
        - sub_ax (plt.Axes): Subplot with BTSP kernel plotted.
        """

        sub_ax, _, _ = signal_util.get_pre_post_exponential(
            dt=self.Agent.dt,
            **self.get_learning_kernel_kwargs(),
            **kwargs,
        )

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_learning_kernel", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

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
            if self.normalize_weights_divisively or self.apply_Ojas_rule:
                w_norm = input_layer["w_norm"]
                if w_norm is not None:
                    w_norm = w_norm.tolist()
                self.history["w_norm"][name].append(w_norm)

        if self.trainable_biases:
            if self.normalize_weights_divisively or self.apply_Ojas_rule:
                self.history["b_norm"].append(self.b_norm)

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

        input_layers_with_learning = self.input_layers_with_learning
        Is = [
            self.inputs[name_in][filter_key] for name_in in input_layers_with_learning
        ]

        ws = [self.inputs[name_in]["w"] for name_in in input_layers_with_learning]
        b = self.biases if self.trainable_biases else None
        if lr is None:
            lr = self.lr  # type: ignore[attr-defined]

        if calculate_only:
            ws_pre = ws
            ws = copy.deepcopy(ws)  # disconnected from the original weights
            if self.trainable_biases:
                b_pre = b
                b = copy.deepcopy(b)  # disconnected from the original biases

        alpha = self.get_regularization_alpha()

        if calculate_only:  # do not use Ojas rule or divisive normalization
            apply_Ojas_rule = False
            normalize_weights_divisively = False
        else:
            apply_Ojas_rule = self.apply_Ojas_rule
            normalize_weights_divisively = self.normalize_weights_divisively

        norm_values = learn_util.perform_update_(
            Is,
            ws,
            O,
            lr=lr,
            b=b,
            normalize_weights_divisively=normalize_weights_divisively,
            p=self.p,
            alpha=alpha,
            apply_Ojas_rule=apply_Ojas_rule,
        )

        if calculate_only:
            ws_delta = [ws[i] - ws_pre[i] for i in range(len(ws))]
            if self.trainable_biases:
                b_delta = np.asarray(b) - np.asarray(b_pre)
            else:
                b_delta = None
            return ws_delta, b_delta

        else:
            if self.normalize_weights_divisively or self.apply_Ojas_rule:
                w_norm, b_norm = norm_values
                if self.trainable_biases:
                    self.b_norm = b_norm

                for i, name_in in enumerate(input_layers_with_learning):
                    val = w_norm[i] if self.apply_Ojas_rule else w_norm
                    self.inputs[name_in]["w_norm"] = val

            return None

    def update(self):
        """Update the layer, i.e. calculate the new firing rates and update the
        weights and biases, if applicable."""

        if self.normalize_weights_divisively or self.apply_Ojas_rule:
            for input_layer in self.inputs.values():
                input_layer["w_norm"] = None
            if self.trainable_biases:
                self.b_norm = None

        self._update_filtered_inputs(
            self.learning_filter_tau,  # type: ignore[attr-defined]
            self.learning_trend_tau,  # type: ignore[attr-defined]
            filter_key="filtered_inputs_for_learning",
        )

        super().update(filter_key="filtered_inputs_for_learning")

    def plot_learning_kernel(self, autosave: bool | None = None, **kwargs):
        """
        self.plot_learning_kernel()

        Plot the learning kernel.

        Args:
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to plot_util.plot_pre_post_exponential().

        Returns
        - sub_ax (plt.Axes): Subplot with BTSP kernel plotted.
        """

        sub_ax, _ = plot_util.plot_pre_post_exponential(
            dt=self.Agent.dt,
            **self.get_learning_kernel_kwargs(),
            **kwargs,
        )

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_learning_kernel", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_normalization_values_over_time(
        self,
        input_layer: str,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: int | str = "all",
        bias_norms: bool = False,
        in_min: bool = True,
        lw: float = 1,
        autosave: bool | None = None,
        sub_ax: plt.Axes | None = None,
    ):
        """
        self.plot_normalization_values_over_time()

        Plot the normalization values for the weights of the layer over time.

        Args:
        - input_layer (str): Name of the input layer.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - chosen_neurons (int, str, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - bias_norms (bool, optional): Whether to plot the bias normalization values.
            Default is False.
        - in_min (bool, optional): Whether to plot the time in minutes. Default is True.
        - lw (float, optional): Line width of the plot. Default is 1.
        - autosave (bool, optional): Whether to save the figure. Default is None.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with normalization values plotted.
        """

        use_chosen_neurons = chosen_neurons
        if isinstance(chosen_neurons, str) and chosen_neurons == "max":
            use_chosen_neurons = "all"

        use_chosen_neurons = self.return_list_of_neurons(chosen_neurons=use_chosen_neurons)  # type: ignore[arg-type]

        steps, norm_values = self.get_normalization_values(
            input_layer=input_layer,
            t_start=t_start,
            t_end=t_end,
            chosen_neurons=use_chosen_neurons,
            bias_norms=bias_norms,
        )

        full_t = self.get_plotting_times(t_start=t_start, t_end=t_end)[0]
        if in_min:
            full_t = full_t / 60

        if isinstance(chosen_neurons, str) and chosen_neurons == "max":
            norm_values = norm_values.max(axis=1, keepdims=True)  # max

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(6, 3))

        if len(steps):
            t = np.asarray(self.history["t"])[steps]
            if in_min:
                t = t / 60

            color = self.inputs[input_layer]["layer"].color

            for i, data in enumerate(norm_values.T):
                keep_idx = np.concatenate([[0], np.where(np.diff(data) > 0)[0] + 1])
                if lw > 0 and len(keep_idx) > 1:
                    sub_ax.plot(
                        t[keep_idx],
                        data[keep_idx],
                        color=color,
                        lw=lw,
                        alpha=0.3,
                    )
                abv_1 = data[keep_idx] > 1
                for i, alpha in enumerate([0.3, 0.8]):
                    abv_bool = abv_1 if i == 1 else ~abv_1
                    if abv_bool.sum():
                        sub_ax.scatter(
                            t[keep_idx][abv_bool],
                            data[keep_idx][abv_bool],
                            color=color,
                            marker="d",
                            s=10,
                            alpha=alpha,
                        )

            if len(norm_values):
                abv_1 = np.where(norm_values.max(axis=1) > 1)[0]
                if len(abv_1):
                    sub_ax.axvline(
                        t[abv_1[0]],
                        color="k",
                        lw=1,
                        ls="--",
                        alpha=0.3,
                        zorder=-2,
                    )

        xmin, xmax = sub_ax.get_xlim()
        if xmin > full_t[0]:
            sub_ax.set_xlim(left=full_t[0])
        if xmax < full_t[-1]:
            sub_ax.set_xlim(right=full_t[-1])

        plot_util.pad_axis(sub_ax, axis="x", pad_prop=0.05)
        plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.5, prop_high=1.0)

        sub_ax.axhline(1, color="k", lw=1, ls="--", alpha=0.3)

        if sub_ax.get_ylim()[0] > 0:
            sub_ax.set_ylim(bottom=0)

        if sub_ax.get_ylim()[1] < 1.05:
            sub_ax.set_ylim(top=1.05)

        time_str = "Time (min)" if in_min else "Time (s)"
        sub_ax.set_xlabel(time_str)

        norm_str = "Oja" if self.apply_Ojas_rule else "Divisive"
        sub_ax.set_ylabel(f"{norm_str} normalization value")

        norm_str = "Max norm." if chosen_neurons == "max" else "Norm."
        sub_ax.set_title(
            f"{norm_str} factor calculated for weights from {input_layer}", y=1.05
        )

        sub_ax.spines[["top", "right"]].set_visible(False)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_norm_over_time", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_normalization_value_hist(
        self,
        input_layer: str,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        bias_norms: bool = False,
        above_one: bool = False,
        max_per_step: bool = False,
        bins: int = 50,
        autosave: bool | None = None,
        sub_ax: plt.Axes | None = None,
    ):
        """
        self.plot_normalization_values()

        Plot the normalization values for the weights of the layer.

        Args:
        - input_layer (str): Name of the input layer.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - bias_norms (bool, optional): Whether to plot the bias normalization values.
            Default is False.
        - above_one (bool, optional): Whether to plot only values above 1. Default is
            False.
        - max_per_step (bool, optional): Whether to plot the maximum value per step.
            Default is False.
        - bins (int, optional): Number of bins for the histogram. Default is 50.
        - autosave (bool, optional): Whether to save the figure. Default is None.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with normalization histogram plotted.

        """

        _, norm_values = self.get_normalization_values(
            input_layer=input_layer,
            t_start=t_start,
            t_end=t_end,
            chosen_neurons=chosen_neurons,
            bias_norms=bias_norms,
        )

        if max_per_step:
            norm_values = np.max(norm_values, axis=1)
        else:
            norm_values = norm_values.flatten()

        if above_one:
            norm_values = norm_values[norm_values > 1]

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(8, 3))

        if len(norm_values):
            sub_ax.hist(
                norm_values, bins=bins, density=False, color=self.color, alpha=0.8
            )
            sub_ax.set_xlim(left=0)

        if not above_one:
            sub_ax.axvline(1, color="k", lw=1, ls="--", alpha=0.3)

        norm_str = "Oja" if self.apply_Ojas_rule else "Divisive"
        sub_ax.set_xlabel(f"{norm_str} normalization value")

        sub_ax.set_ylabel("Frequency")

        sub_ax.set_title(
            f"Normalization calculated for weights from {input_layer}", y=1.05
        )

        sub_ax.spines[["top", "right"]].set_visible(False)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_norm_hist", save=autosave)  # type: ignore[attr-defined]

        return sub_ax


class BTSPLayer(HebbianLayer):
    """
    BTSPLayer()

    Class extending HebbianLayer. Defines a population of neurons that tune their
    weights through Hebbian learning with BTSP.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation, with, for example, input layers.

    default_params = {
        "n": 10,
        "name": "BTSPLayer",
        "pre_BTSP_filter_tau_pos": 4,
        "post_BTSP_filter_tau_pos": "half",
        "pre_BTSP_neg_weight": 0,
        "pre_BTSP_filter_tau_neg": None,
        "post_BTSP_filter_tau_neg": None,
        "post_BTSP_neg_weight": 0,
        "BTSP_lr": 1,
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
        • self.get_BTSP_kernel_kwargs()
        • self.get_BTSP_kernel()
        • self.get_estimated_BTSP_roots()
        • self.get_BTSP_step_dict()
        • self.get_BTSP_steps()
        • self.get_BTSP_info()
        • self.get_BTSP_counts()
        • self.get_BTSP_ramp_peaks()
        • self.get_BTSP_ramp_peak_dict()
        • self.update()
        • self.plot_BTSP_kernel()
        • self.plot_filtered_for_BTSP()
        • self.plot_BTSP_frequency()
        • self.plot_BTSP_step_histogram()
        • self.plot_BTSP_ramp_histogram()
        • self.plot_BTSP_responses()
        • self.plot_BTSP_locations()
        • self.plot_BTSP_ramp()
        • self.plot_BTSP_ramp_factors()
        • self.add_BTSP_markers_to_plots()
        • self.plot_rate_map()
        • self.plot_rate_timeseries()
        • self.plot_normalization_values_over_time()

    """

    default_params = {
        "n": 10,
        "name": "BTSPLayer",
        "pre_BTSP_filter_tau_pos": 1,
        "pre_BTSP_trend_tau_pos": None,
        "post_BTSP_filter_tau_pos": "half",
        "post_BTSP_trend_tau_pos": None,
        "pre_BTSP_neg_weight": 0,
        "pre_BTSP_filter_tau_neg": 0.5,
        "pre_BTSP_trend_tau_neg": None,
        "post_BTSP_filter_tau_neg": "half",
        "post_BTSP_trend_tau_neg": None,
        "post_BTSP_neg_weight": 0,
        "BTSP_lr": 1,
        "single_BTSP": False,
        "BTSP_distance_prop": None,  # None to remove constraint
        "log_BTSP": False,
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
            - b_delta (1D np.ndarray): Bias updates.
            - num_steps (1D np.ndarray): Number of steps to apply BTSP.
            - ws_delta (None or list): BTSP weight updates.
                List of weight updates, each with shape (O, I_i)
        - last_BTSP_pos (list): Last position at which a BTSP event occurred.
        - last_BTSP_step (int): Last step at which a BTSP event occurred.
        - num_BTSP (int): Number of BTSP events recorded.

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

        self.num_BTSP = np.zeros(self.n)  # type: ignore[attr-defined]
        # type: ignore[attr-defined]
        self.last_BTSP_step = np.full(self.n, np.nan)  # type: ignore[attr-defined]
        self.last_BTSP_pos = [None for _ in range(self.n)]  # type: ignore[attr-defined]

        self.BTSP_buffer = {
            "num_steps": np.zeros(self.n),
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

    @property
    def pre_BTSP_norm_adj_neg_weight(self) -> float:
        """
        self.pre_BTSP_norm_adj_neg_weight

        The adjusted negative weight for the pre BTSP filter when applied to normalized
        positive and negative exponentials.
        """

        self._pre_BTSP_neg_weight = signal_util.get_adj_neg_weight(
            neg_weight=self.pre_BTSP_neg_weight,
            filter_tau_pos=self.pre_BTSP_filter_tau_pos,
            filter_tau_neg=self.pre_BTSP_filter_tau_neg,
        )

        return self._pre_BTSP_neg_weight

    @property
    def post_BTSP_norm_adj_neg_weight(self) -> float:
        """
        self.post_BTSP_norm_adj_neg_weight

        The adjusted negative weight for the post BTSP filter when applied to normalized
        positive and negative exponentials.
        """

        self._post_BTSP_neg_weight = signal_util.get_adj_neg_weight(
            neg_weight=self.post_BTSP_neg_weight,
            filter_tau_pos=self.post_BTSP_filter_tau_pos,
            filter_tau_neg=self.post_BTSP_filter_tau_neg,
        )

        return self._post_BTSP_neg_weight

    def _init_post_BTSP_filters(self):
        """
        self._init_post_BTSP_filters()

        Initialise the filter parameters used to compute learning updates for
        inputs activated after a BTSP (post BTSP kernel).

        Also adds post BTSP filter keys to the history dictionary.

        Attributes:
        - BTSP_integral (float): Integral of the absolute BTSP filter given self.dt.
        - filtered_BTSP_trace (1D np.ndarray): Filtered post BTSP activity.
        - filtered_BTSP_trace_neg (1D np.ndarray): Negatively filtered post
            BTSP activity.
        - filtered_BTSP_trace_pos (1D np.ndarray): Positively filtered post
            BTSP activity.
        - filtered_BTSP_trace_trend_neg (1D np.ndarray): Negatively filtered
            post BTSP activity trend.
        - filtered_BTSP_trace_trend_pos (1D np.ndarray): Positively filtered
            post BTSP activity trend.
        - post_BTSP_filter_tau_neg (float): Time constant of the post BTSP negative
            filter.
        - post_BTSP_filter_tau_pos (float): Time constant of the post BTSP positive
            filter.
        - post_BTSP_trend_tau_neg (float): Time constant of the post BTSP negative
            filter trend.
        - post_BTSP_trend_tau_pos (float): Time constant of the post BTSP positive
            filter trend.
        """

        if self.post_BTSP_neg_weight is None:
            self.post_BTSP_neg_weight = self.pre_BTSP_neg_weight

        for suffix in ["_pos", "_neg"]:

            post_BTSP_filter_tau = signal_util.get_relative_filter_tau(
                getattr(self, f"post_BTSP_filter_tau{suffix}"),
                base_filter_tau=getattr(self, f"pre_BTSP_filter_tau{suffix}"),
            )
            setattr(self, f"post_BTSP_filter_tau{suffix}", post_BTSP_filter_tau)

            # start at 0
            setattr(self, f"filtered_BTSP_trace{suffix}", np.zeros(self.n))
            setattr(self, f"filtered_BTSP_trace_trend{suffix}", np.zeros(self.n))

            for name in ["", "_trend"]:
                key = f"filtered_BTSP_trace{name}{suffix}"
                if key not in self.history.keys():
                    self.history[key] = list()

        self.filtered_BTSP_trace = np.zeros(self.n)
        key = "filtered_BTSP_trace"
        if key not in self.history.keys():
            self.history[key] = list()

        self.BTSP_integral = np.sum(np.absolute(self.get_BTSP_kernel()[0]))

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

        self.inputs[name_in]["filtered_inputs_for_BTSP"] = np.zeros(n_in)
        key = "filtered_inputs_for_BTSP"
        if key not in self.history.keys():
            self.history[key] = dict()
        self.history["filtered_inputs_for_BTSP"][name_in] = list()

        for suffix in ["_pos", "_neg"]:
            self.inputs[name_in][f"filtered_inputs_for_BTSP{suffix}"] = np.zeros(n_in)
            self.inputs[name_in][f"filtered_trends_for_BTSP{suffix}"] = np.zeros(n_in)

            for key in ["inputs", "trends"]:
                if f"filtered_{key}_for_BTSP{suffix}" not in self.history.keys():
                    self.history[f"filtered_{key}_for_BTSP{suffix}"] = dict()
                self.history[f"filtered_{key}_for_BTSP{suffix}"][name_in] = list()

    def save_to_history(self):
        """
        self.save_to_history()

        Save the current state of the layer to the history, including the filtered
        input and trend data used for learning
        """

        super().save_to_history()

        for name, input_layer in self.inputs.items():
            self.history["filtered_inputs_for_BTSP"][name].append(
                input_layer["filtered_inputs_for_BTSP"].tolist()
            )

        self.history["filtered_BTSP_trace"].append(self.filtered_BTSP_trace.tolist())

        for suffix in ["_pos", "_neg"]:
            for name, input_layer in self.inputs.items():
                for key in ["inputs", "trends"]:
                    self.history[f"filtered_{key}_for_BTSP{suffix}"][name].append(
                        input_layer[f"filtered_{key}_for_BTSP{suffix}"].tolist()
                    )

            for name in ["", "_trend"]:
                key = f"filtered_BTSP_trace{name}{suffix}"
                self.history[key].append(getattr(self, key).tolist())

    def log_num_steps_to_apply_BTSP(self, last=False, stats_only=False):
        """
        self.log_num_steps_to_apply_BTSP()

        Log the number of steps required to apply BTSP.

        Args:
        - last (bool, optional): Whether to log only the last recorded BTSP event
            applied. Default is False.
        - stats_only (bool, optional): If not last, whether to log only the statistics
            of the BTSP events. Default is False.
        """

        if len(self.history["num_steps_to_apply_BTSP"]) == 0:
            log_str = "No BTSP events applied."

        elif last:
            num_steps = self.history["num_steps_to_apply_BTSP"][-1]
            seconds = num_steps * self.Agent.dt
            log_str = (
                f"Last BTSP event was applied after {num_steps} steps "
                f"({seconds:.2f} sec.)."
            )

        else:
            num_steps = np.asarray(self.history["num_steps_to_apply_BTSP"])
            seconds = num_steps * self.Agent.dt
            all_steps = ", ".join([str(num_step) for num_step in num_steps])

            log_str = (
                f"BTSP events applied after {num_steps.mean():.2f} steps "
                f"({seconds.mean():.2f} sec.)"
            )

            if not stats_only:
                log_str = f"{log_str} (num steps: {all_steps})."

        print(log_str)

    def log_BTSP_applied_then_triggered_interval(
        self, shortest=False, stats_only=False, t_start=None, t_end=None
    ):
        """
        self.log_BTSP_applied_then_triggered_interval()

        Log the number of steps between when BTSP is applied and next triggered in
        the same neuron. Time constraints apply to BTSP trigger and application steps.

        Args:
        - shortest (bool, optional): Whether to log only the shortest interval
            identified. Default is False.
        - stats_only (bool, optional): If not shortest, whether to log only the
            statistics of the intervals. Default is False.
        - t_start (float, optional): Start time for including BTSP events.
            Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.
        """

        trigger_steps = self.get_BTSP_step_dict(
            apply_step=False, t_start=t_start, t_end=t_end
        )
        apply_steps = self.get_BTSP_step_dict(
            apply_step=True, t_start=t_start, t_end=t_end
        )
        all_diffs = list()
        for i, neuron_trigger_steps in trigger_steps.items():
            if len(neuron_trigger_steps) < 2:
                continue
            neuron_apply_steps = apply_steps[i]
            diff = (
                np.asarray(neuron_trigger_steps)[1 : len(neuron_apply_steps) + 1]
                - np.asarray(neuron_apply_steps)[:-1]
            )
            all_diffs.extend(list(diff))

        if len(all_diffs) == 0:
            log_str = "No neuron had at least two BTSP events triggered."

        elif shortest:
            num_steps = min(all_diffs)
            seconds = num_steps * self.Agent.dt
            log_str = (
                f"Shortest interval between applying and subsequently triggering a "
                f"BTSP event in the same neuron was {num_steps} steps "
                f"({seconds:.2f} sec.)."
            )

        else:
            num_steps = np.asarray(all_diffs)
            seconds = num_steps * self.Agent.dt
            all_steps = ", ".join([str(num_step) for num_step in num_steps])

            log_str = (
                "Average interval between applying and subsequently triggering a "
                f"BTSP event in the same neuron was {num_steps.mean():.2f} steps "
                f"({seconds.mean():.2f} sec.)"
            )

            if not stats_only:
                log_str = f"{log_str} (num steps: {all_steps})."

        print(log_str)

    def get_BTSP_kernel_kwargs(self):
        """
        self.get_BTSP_kernel_kwargs()

        Obtain the keyword arguments for the BTSP kernel.

        Returns
        - BTSP_kernel_kwargs (dict): Keyword arguments for the BTSP kernel.
        """

        BTSP_kernel_kwargs = {
            "pre_filter_tau_pos": self.pre_BTSP_filter_tau_pos,
            "pre_trend_tau_pos": self.pre_BTSP_trend_tau_pos,
            "pre_filter_tau_neg": self.pre_BTSP_filter_tau_neg,
            "pre_trend_tau_neg": self.pre_BTSP_trend_tau_neg,
            "pre_neg_weight": self.pre_BTSP_neg_weight,
            "post_filter_tau_pos": self.post_BTSP_filter_tau_pos,
            "post_trend_tau_pos": self.post_BTSP_trend_tau_pos,
            "post_filter_tau_neg": self.post_BTSP_filter_tau_neg,
            "post_trend_tau_neg": self.post_BTSP_trend_tau_neg,
            "post_neg_weight": self.post_BTSP_neg_weight,
        }

        return BTSP_kernel_kwargs

    def get_BTSP_kernel(self, t_pre=None, t_post=None, **kwargs):
        """
        self.get_BTSP_kernel()

        Obtain the BTSP kernel.

        Keyword args:
        - **kwargs: Keyword arguments passed to signal_util.get_summed_exp().

        Returns
        - BTSP_kernel (1D np.ndarray): BTSP kernel.
        - align_pt (int): Alignment point of the pre and post components of the BTSP
            kernel.
        """

        BTSP_kernel, align_pt = signal_util.get_summed_exp(
            dt=self.Agent.dt,
            **self.get_BTSP_kernel_kwargs(),
            **kwargs,
        )

        if t_pre is not None:
            t_pre = int(np.round(t_pre / self.Agent.dt))
            if t_pre < 0:
                raise ValueError("t_pre must be non-negative.")
            if t_pre > len(BTSP_kernel) - align_pt - 1:
                raise ValueError(
                    f"The BTSP kernel is too short for a t_pre value of {t_pre}."
                )
            BTSP_kernel = BTSP_kernel[align_pt - t_pre :]
            align_pt = t_pre

        if t_post is not None:
            t_post = int(np.round(t_post / self.Agent.dt))
            if t_post < 0:
                raise ValueError("t_post must be non-negative.")
            if t_post > align_pt:
                raise ValueError(
                    f"The BTSP kernel is too short for a t_post value of {t_post}."
                )
            BTSP_kernel = BTSP_kernel[: align_pt + t_post + 1]

        return BTSP_kernel, align_pt

    def get_estimated_BTSP_roots(self, near_zero=1e-4, **kwargs):
        """
        self.get_estimated_BTSP_roots()

        Obtain the estimated roots of the BTSP kernel.

        Keyword args:
        - **kwargs: Keyword arguments passed to signal_util.get_summed_exp().

        Returns
        - root_dict (dict): Estimated roots of the BTSP kernel. np.inf is returned if
            no roots were found.
        """

        BTSP_kernel, align_pt = self.get_BTSP_kernel(**kwargs)
        root_dict = hyper_util.get_root_dict(
            BTSP_kernel, align_pt, dt=self.Agent.dt, near_zero=near_zero
        )

        return root_dict

    def get_BTSP_step_dict(self, apply_step=False, t_start=None, t_end=None):
        """
        self.get_BTSP_step_dict()

        Get a dictionary of BTSP events for each neuron.

        Args:
        - apply_step (bool, optional): Whether to return the step at which the BTSP
            event was applied. Default is False.
        - t_start (float, optional): Start time for including BTSP events.
            Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.

        Returns:
        - BTSP_step_dict (dict): Dictionary of BTSP events for each neuron.
        """

        _, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)

        BTSP_step_dict = {neuron_idx: list() for neuron_idx in range(self.n)}

        for i, (step, targets) in enumerate(
            zip(self.history["BTSP_events"], self.history["BTSP_targets"])
        ):
            if apply_step:
                if len(self.history["num_steps_to_apply_BTSP"]) > i:
                    step += self.history["num_steps_to_apply_BTSP"][i]
                else:
                    step = np.nan

            if np.isfinite(step) and (step < startid or step >= endid):
                continue

            for target in targets:
                BTSP_step_dict[target].append(step)

        return BTSP_step_dict

    def get_BTSP_steps(
        self,
        applied_only=False,
        apply_step=False,
        applied_also=False,
        t_start=None,
        t_end=None,
        chosen_neurons="all",
    ):
        """
        self.get_BTSP_steps()

        Get the steps at which BTSP updates were triggered. Time constraints are
        applied to the BTSP trigger time if apply_step is False, and to the BTSP
        application time if apply_step.

        If applied_also is True, BTSP events with application steps outside of the time
        constraints are still included.

        Args:
        - applied_only (bool, optional): Whether to return only applied BTSP events.
            Default is False.
        - apply_step (bool, optional): Whether to return the step at which the BTSP
            event was applied. Default is False.
        - applied_also (bool, optional): Whether to also return steps where BTSP was
            applied. Default is False.
        - t_start (float, optional): Start time for including BTSP events.
            Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional):
            Neurons to consider for BTSP steps. Default is "all".

        Returns:
        - all_steps (1D np.ndarray): Steps at which BTSP events occurred.
        if applied_also:
        - applied_steps (list, optional): Steps at which BTSP events were
            applied, if applied_also is True. np.nan if not applied.
        """

        if applied_also and apply_step:
            raise ValueError("Cannot set both applied_also and apply_step to True.")

        if applied_also or (applied_only and not apply_step):
            _, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)
            BTSP_step_dict = self.get_BTSP_step_dict(apply_step=False)
            BTSP_apply_step_dict = self.get_BTSP_step_dict(apply_step=True)

        else:
            BTSP_step_dict = self.get_BTSP_step_dict(
                apply_step=apply_step, t_start=t_start, t_end=t_end
            )

        if chosen_neurons != "all":
            chosen_neurons = self.get_chosen_neurons(chosen_neurons)
            BTSP_step_dict = {
                target: steps
                for target, steps in BTSP_step_dict.items()
                if target in chosen_neurons
            }

        all_steps = list()
        applied_steps = list()
        for target, steps in BTSP_step_dict.items():
            for i, step in enumerate(steps):
                if np.isnan(step) or step in all_steps:
                    continue
                if applied_also or (applied_only and not apply_step):
                    if step < startid or step >= endid:
                        continue
                    if i >= len(BTSP_apply_step_dict[target]):
                        step_applied = np.nan
                    elif np.isnan(BTSP_apply_step_dict[target][i]):
                        step_applied = np.nan
                    else:
                        step_applied = BTSP_apply_step_dict[target][i]

                    if applied_only and (
                        np.isnan(step_applied) or step_applied >= endid
                    ):
                        continue

                    applied_steps.append(step_applied)
                all_steps.append(step)

        all_steps = np.asarray(all_steps)

        if applied_also:
            applied_steps = np.asarray(applied_steps)
            return all_steps, applied_steps
        else:
            all_steps = np.sort(all_steps)
            return all_steps

    def get_BTSP_info(self, BTSP_idx=0, neuron_idx=0, t_start=None, t_end=None):
        """
        self.get_BTSP_info()

        Get information about a specific BTSP event.

        Args:
        - BTSP_idx (int, optional): Index of the BTSP event to get information about.
            Default is 0.
        - neuron_idx (int, optional): Index of the neuron to get information about.
            Default is 0.
        - t_start (float, optional): Start time for including BTSP events.
            Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.

        Returns:
        - BTSP_info (dict): Dictionary with information about the BTSP event.
            Keys:
                - step (int): Step at which the BTSP event occurred.
                - time (float): Time at which the BTSP event occurred.
                - position (1D np.ndarray): Position at which the BTSP event occurred.
                - applied_step (int): Step at which the BTSP event was applied.
                - applied_time (float): Time at which the BTSP event was applied.
        """

        BTSP_steps, apply_steps = self.get_BTSP_steps(
            chosen_neurons=[neuron_idx], t_start=t_start, t_end=t_end, applied_also=True
        )

        BTSP_info = {
            "step": np.nan,
            "time": np.nan,
            "position": None,
            "applied_step": np.nan,
            "applied_time": np.nan,
        }

        if len(BTSP_steps) > BTSP_idx:
            BTSP_info["step"] = BTSP_steps[BTSP_idx]
            BTSP_info["time"] = self.history["t"][BTSP_info["step"]]
            BTSP_info["position"] = self.Agent.history["pos"][BTSP_info["step"]]

            if len(apply_steps) > BTSP_idx and np.isfinite(apply_steps[BTSP_idx]):
                BTSP_info["applied_step"] = apply_steps[BTSP_idx]
                BTSP_info["applied_time"] = self.history["t"][BTSP_info["applied_step"]]

        return BTSP_info

    def get_BTSP_counts(self, applied_only=False, t_start=None, t_end=None):
        """
        self.get_BTSP_counts()

        Get the number of BTSP events per neuron. Time constraints are applied to the
        BTSP trigger time and application time if applied_only is True.

        Args:
        - applied_only (bool, optional): Whether to count only BTSP events applied
            within the same time constraints. Default is False.
        - t_start (float, optional): Start time for including BTSP events.
            Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.

        Returns:
        - counts (1D np.ndarray): Number of BTSP events for each neuron.
        """

        counts = np.zeros(self.n)
        for neuron_idx in range(self.n):
            counts[neuron_idx] = len(
                self.get_BTSP_steps(
                    chosen_neurons=[neuron_idx],
                    t_start=t_start,
                    t_end=t_end,
                    applied_only=applied_only,
                )
            )

        return counts

    def get_estimated_num_steps_pre_post_BTSP(self, as_time=False):
        """
        self.get_estimated_num_steps_pre_post_BTSP()

        Get the estimated number of steps used in the calculation of a BTSP event
        before and after. The mean number of steps post is used, and the number of
        steps pre is estimated based on the BTSP kernel roots.

        Args:
        - as_time (bool, optional): Whether to return the number of steps as time
            instead. Default is False.

        Returns:
        - pre (int): Estimated number of steps or time before BTSP event.
        - post (int): Estimated number of steps or time after BTSP event.
        """

        num_steps_to_apply_BTSP = self.history["num_steps_to_apply_BTSP"]
        if len(num_steps_to_apply_BTSP) == 0:
            raise NotImplementedError(
                "Cannot estimate number of steps as no BTSP events have occurred."
            )

        post = int(np.mean(num_steps_to_apply_BTSP))

        root_dict = self.get_estimated_BTSP_roots()
        if isinstance(root_dict, dict):
            time_pre = root_dict["pre_outer"]
            time_post = root_dict["post_outer"]
            pre = int(np.around(np.absolute(post / time_post * time_pre)))

        else:
            raise NotImplementedError(
                "Cannot estimate number of steps pre as the kernel root estimate did "
                "not find roots."
            )

        if as_time:
            pre = pre * self.Agent.dt
            post = post * self.Agent.dt

        return pre, post

    def get_BTSP_ramp_peaks(self, t_start=None, t_end=None):
        """
        self.get_BTSP_ramp_peaks()

        Get the BTSP ramp peak for each neuron.

        Args:
        - t_start (float, optional): Start time for including BTSP events.
            Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.

        Returns:
        - BTSP_ramp_peaks (1D np.ndarray): BTSP ramp peak for each neuron.
        """

        _, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)

        BTSP_ramp = np.asarray(self.history["BTSP_ramp"])

        BTSP_ramp_peaks = BTSP_ramp[startid:endid].max(axis=0)

        return BTSP_ramp_peaks

    def get_BTSP_ramp_peak_dict(self, t_start=None, t_end=None):
        """
        self.get_BTSP_ramp_peak_dict()

        Get the peak values of the BTSP ramp.

        Args:
        - t_start (float, optional): Start time for including BTSP events.
            Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.

        Returns:
        - ramp_peak_dict (dict): Dictionary with peak values and steps for each neuron.
        """

        _, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)

        BTSP_ramp = np.asarray(self.history["BTSP_ramp"])

        ramp_peak_dict = {
            "peaks": dict(),
            "steps": dict(),
        }
        for n, neuron_ramp in enumerate(BTSP_ramp.T):
            peaks = gen_util.get_minima_indices(-neuron_ramp, minimum=-1e-5)
            if len(peaks):
                peak_vals = neuron_ramp[peaks]
            else:
                peak_vals = np.asarray(list())

            include_mask = (peaks >= startid) * (peaks <= endid)

            ramp_peak_dict["peaks"][n] = peak_vals[include_mask]
            ramp_peak_dict["steps"][n] = peaks[include_mask]

        return ramp_peak_dict

    def get_nbr_BTSP_position_bins(self, t_start=None, t_end=None, bins=21):
        """
        self.get_nbr_BTSP_position_bins()

        Get the number of position bins in which BTSP events occurred for each neuron.

        Args:
        - t_start (float, optional): Start time for including BTSP events.
            Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.

        Returns:
        - nbr_BTSP_position_bins (1D np.ndarray): Number of position bins with BTSP
            events for each neuron.
        """

        if self.Agent.Environment.D != 1:
            raise NotImplementedError(
                "get_nbr_BTSP_position_bins() is only implemented for 1D environments."
            )

        BTSP_step_dict = self.get_BTSP_step_dict(t_start=t_start, t_end=t_end)

        nbr_BTSP_position_bins = np.zeros(self.n)

        bins = np.linspace(0, self.Agent.Environment.scale, bins)
        pos = np.asarray(self.Agent.history["pos"])[:, 0]  # 1D
        for i, steps in BTSP_step_dict.items():
            if len(steps) == 0:
                continue

            all_BTSP_positions = pos[steps]
            pos_counts, _ = np.histogram(all_BTSP_positions, bins=bins)
            nbr_BTSP_position_bins[i] = len(np.where(pos_counts)[0])

        return nbr_BTSP_position_bins

    def get_speed_around_BTSP(
        self, pre=4, post=2, linear=True, directional=False, cm=True
    ):
        """
        self.get_speed_around_BTSP()

        Obtain the speed around BTSP events.

        Args:
        - pre (int, optional): Number of steps before BTSP event. Default is 4.
        - post (int, optional): Number of steps after BTSP event. Default is 2.
        - linear (bool, optional): Whether to use linear speed. Default is True.
        - directional (bool, optional): Whether to use directional speed. Default is
            False.
        - cm (bool, optional): Whether to return speed in cm/s instead of m/s.
            Default is True.

        Returns:
        - time (1D np.ndarray): Time around BTSP events.
        - speed_arr (3D np.ndarray): Speed around BTSP events, with shape BTSP steps,
            time steps, speed dimensions.
        """

        steps = np.unique(self.get_BTSP_steps())

        rel_indices = plot_util.get_time_indices(pre=pre, post=post, dt=self.Agent.dt)
        time = np.linspace(-pre, post, len(rel_indices))

        speed = self.Agent.get_speed(linear=linear, directional=directional, cm=cm)

        speed_arr = np.full((len(steps), len(rel_indices), speed.shape[1]), np.nan)
        for i, step in enumerate(steps):
            indices = step + rel_indices
            mask = (indices >= 0) * (indices < len(speed))
            arr_indices = np.arange(len(indices))[mask]
            speed_arr[i, arr_indices] = speed[indices[mask]]

        return time, speed_arr

    def get_position_around_BTSP(self, pre=6, post=3, cm=True):
        """
        self.get_position_around_BTSP()

        Obtain the position around BTSP events.

        Args:
        - pre (int, optional): Number of steps before BTSP event. Default is 6.
        - post (int, optional): Number of steps after BTSP event. Default is 3.
        - cm (bool, optional): Whether to return position in cm instead of m. Default
            is True.

        Returns:
        - time (1D np.ndarray): Time around BTSP events.
        - position_arr (3D np.ndarray): Position around BTSP events, with shape
            BTSP steps, time steps, position dimensions.
        """

        steps = np.unique(self.get_BTSP_steps())
        position = np.asarray(self.Agent.history["pos"])
        if cm:
            position *= 100

        rel_indices = plot_util.get_time_indices(pre=pre, post=post, dt=self.Agent.dt)
        time = np.linspace(-pre, post, len(rel_indices))

        position_arr = np.full(
            (len(steps), len(rel_indices), position.shape[1]), np.nan
        )
        for i, step in enumerate(steps):
            indices = step + rel_indices
            mask = (indices >= 0) * (indices < len(position))
            arr_indices = np.arange(len(indices))[mask]
            position_arr[i, arr_indices] = position[indices[mask]]

        return time, position_arr

    def get_BTSP_response_dict(
        self, pre=1, post=2, chosen_neurons="all", t_start=None, t_end=None
    ):
        """
        self.get_BTSP_response_dict()

        Get the BTSP responses of the layer.

        Args:
        - pre (float, optional): Time before BTSP events to include. Default is 1.
        - post (float, optional): Time after BTSP events to include. Default is 2.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to include.
            Default is "all".
        - t_start (float, optional): Start time for inclusion. Default is None.
        - t_end (float, optional): End time for inclusion. Default is None.

        Returns:
        - time (1D np.ndarray): Time vector for the BTSP responses.
        - BTSP_response_dict (dict): Dictionary of BTSP responses for each neuron.
        """

        BTSP_step_dict = self.get_BTSP_step_dict(t_start=t_start, t_end=t_end)
        firingrates = np.asarray(self.history["firingrate"])

        relative_indices = plot_util.get_time_indices(
            pre=pre, post=post, dt=self.Agent.dt
        )
        time = np.linspace(-pre, post, len(relative_indices))
        num_steps_total = len(self.history["t"])

        chosen_neurons = self.get_chosen_neurons(chosen_neurons)

        BTSP_response_dict = dict()
        for neuron_idx in chosen_neurons:
            steps = BTSP_step_dict[neuron_idx]
            if len(steps) == 0:
                BTSP_response_dict[neuron_idx] = None

            responses = np.full((len(steps), len(relative_indices)), np.nan)
            for j, step in enumerate(steps):
                indices = relative_indices + step
                mask = (indices >= 0) * (indices < num_steps_total)
                responses[j, mask] = firingrates[indices[mask], neuron_idx]
            BTSP_response_dict[neuron_idx] = responses

        return time, BTSP_response_dict

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

    def filter_BTSP_targets(
        self,
        BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] | None = list(),
    ):
        """
        self.filter_BTSP_targets()

        Filter out any BTSP targets that are not applicable for the current layer.

        Returns:
        - BTSP_targets (1D np.ndarray): Applicable BTSP targets (neuron
            indices).
        """

        if not self.BTSP_learn or len(BTSP_targets) == 0:
            BTSP_targets = np.asarray([])

        else:
            # cannot trigger a new BTSP event while one is on-going for the same neuron
            BTSP_targets = np.asarray(
                [
                    targ
                    for targ in BTSP_targets
                    if not self.BTSP_buffer["num_steps"][targ]
                ]
            )

            if self.single_BTSP:  # type: ignore[attr-defined]
                keep_BTSP_targets = np.asarray(
                    [targ for targ in BTSP_targets if self.num_BTSP[targ] == 0]
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

            BTSP_targets = keep_BTSP_targets

        return BTSP_targets

    def _record_BTSP_events(
        self,
        BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list(),
    ):
        """
        self._record_BTSP_events()

        Update the layer with BTSP targets, if applicable.

        Args:
        - BTSP_targets (list or 1D np.ndarray, optional): List of BTSP targets.
            Default is list().
        """

        if len(BTSP_targets) == 0:
            return

        self.num_BTSP[np.asarray(BTSP_targets)] += +1
        self.last_BTSP_step[np.asarray(BTSP_targets)] = self.num_steps_total - 1
        for targ in BTSP_targets:
            self.last_BTSP_pos[targ] = self.Agent.pos

        self.BTSP_buffer["num_steps"][BTSP_targets] = 1

        self.history["BTSP_events"].append(
            self.num_steps_total - 1
        )  # recorded after update
        self.history["BTSP_targets"].append(BTSP_targets)

        if self.log_BTSP:
            step_time_str = self.Agent.get_step_and_time(as_str=True)
            neuron_str = ""
            if self.n > 1:
                neuron_str = "neuron" if len(BTSP_targets) == 1 else "neurons"
                neurons = ", ".join([str(i) for i in BTSP_targets])
                neuron_str = f" for {neuron_str} {neurons}"
            print(f"BTSP event recorded{neuron_str} at {step_time_str}.")

        return BTSP_targets

    def _update_BTSP_filters(self, BTSP_targets=list()):
        """
        self._update_BTSP_filters()

        Update the BTSP filters.

        Args:
        - BTSP_targets (1D np.ndarray): BTSP targets (neuron indices). Default is list().
        """

        # update inputs filtered for BTSP
        for suffix in ["_pos", "_neg"]:
            self._update_filtered_inputs(
                getattr(self, f"pre_BTSP_filter_tau{suffix}"),
                getattr(self, f"pre_BTSP_trend_tau{suffix}"),
                filter_key=f"filtered_inputs_for_BTSP{suffix}",
            )

        for _, input_layer in self.inputs.items():
            input_layer["filtered_inputs_for_BTSP"], _ = (
                signal_util.get_exp_weighted_sum(
                    input_layer["filtered_inputs_for_BTSP_pos"],
                    input_layer["filtered_inputs_for_BTSP_neg"],
                    neg_weight=self.pre_BTSP_neg_weight,
                )
            )

        # update BTSP trace
        for suffix in ["_pos", "_neg"]:
            update_value = np.zeros(self.n)
            if len(BTSP_targets):
                update_value[np.asarray(BTSP_targets)] = 1

            filtered_BTSP_trace = getattr(self, f"filtered_BTSP_trace{suffix}")
            filtered_BTSP_trace_trend = getattr(
                self, f"filtered_BTSP_trace_trend{suffix}"
            )

            X_t1, T_t1 = signal_util.get_exp_filtered_signal(
                update_value,
                X_t=filtered_BTSP_trace,
                T_t=filtered_BTSP_trace_trend,
                filter_tau=getattr(self, f"post_BTSP_filter_tau{suffix}"),
                trend_tau=getattr(self, f"post_BTSP_trend_tau{suffix}"),
                dt=self.Agent.dt,
                atol=1e-8,
            )

            setattr(self, f"filtered_BTSP_trace{suffix}", X_t1)
            setattr(self, f"filtered_BTSP_trace_trend{suffix}", T_t1)

        self.filtered_BTSP_trace, _ = signal_util.get_exp_weighted_sum(
            self.filtered_BTSP_trace_pos,
            self.filtered_BTSP_trace_neg,
            neg_weight=self.post_BTSP_neg_weight,
        )

    def _update_BTSP_buffer(self, increment_steps=True):
        """
        self._update_BTSP_buffer(ws_delta)

        Update the BTSP buffer.

        Args:
        - increment_steps (bool, optional): Whether to increment the number of steps
            in the buffer. Default is True.
        """

        BTSP_ongoing = self.BTSP_buffer["num_steps"].astype(bool)

        if not BTSP_ongoing.any():
            return

        lr = np.zeros(self.n)
        lr[BTSP_ongoing] = self.BTSP_lr
        ws_delta, b_delta = self.update_weights(
            filter_key="filtered_inputs_for_BTSP",
            O=self.filtered_BTSP_trace,
            lr=lr,
            calculate_only=True,
        )

        if self.BTSP_buffer["ws_delta"] is None:
            self.BTSP_buffer["ws_delta"] = ws_delta
        else:
            for w in range(len(ws_delta)):
                self.BTSP_buffer["ws_delta"][w] += ws_delta[w]

        if self.trainable_biases:
            self.BTSP_buffer["b_delta"] += b_delta

        if increment_steps:
            self.BTSP_buffer["num_steps"][BTSP_ongoing] = (
                self.BTSP_buffer["num_steps"][BTSP_ongoing] + 1
            )

    def _reset_BTSP_buffer(self, i=0):
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
            self.BTSP_buffer["ws_delta"][w][i] = 0

        if self.trainable_biases:
            self.BTSP_buffer["b_delta"][i] = 0

        self.BTSP_buffer["num_steps"][i] = 0

    def _apply_BTSP_if_applicable(self, num_steps_near_zero=10):
        """
        self._apply_BTSP_if_applicable()

        Checks whether BTSP updates should be applied. If so, computes and applies the
        updates. Weights or biases are set to 0 if the update would take them lower.

        Args:
        - num_steps_near_zero (int, optional): Number of steps near zero required to
            apply BTSP. Ensures that a transient 0 value in the filtered signal does
            not prematurely trigger an update. Default is 10.
        """

        self.BTSP_applied = np.zeros(self.n, dtype=bool)

        if len(self.history["filtered_BTSP_trace"]) < num_steps_near_zero:
            return

        signal = np.concatenate(
            [
                self.history["filtered_BTSP_trace"][-num_steps_near_zero + 1 :],
                self.filtered_BTSP_trace.reshape(1, -1),
            ],
            axis=0,
        )

        apply_BTSP = self.BTSP_buffer["num_steps"].astype(bool) * np.all(
            signal == 0, axis=0
        )

        if not apply_BTSP.any():
            return

        ws = [
            input_layer["w"]
            for name, input_layer in self.inputs.items()
            if name not in self.input_layers_with_no_learning
        ]
        b = self.biases if self.trainable_biases else None

        self.last_ws_delta = copy.deepcopy(self.BTSP_buffer["ws_delta"])

        for i in np.where(apply_BTSP)[0]:
            for w in range(len(ws)):
                w_update = self.BTSP_buffer["ws_delta"][w][i] / self.BTSP_integral
                ws[w][i] += w_update
                below_zero = np.where(ws[w][i] < 0)[0]
                if len(below_zero):
                    ws[w][i][below_zero] = 0

            if self.trainable_biases:
                b_update = self.BTSP_buffer["b_delta"][i] / self.BTSP_integral
                b[i] += b_update
                if b[i] < 0:
                    b[i] = 0

            self.history["num_steps_to_apply_BTSP"].append(
                int(self.BTSP_buffer["num_steps"][i])
            )

            self._reset_BTSP_buffer(i=i)
            self.BTSP_applied[i] = True

        if self.apply_Ojas_rule:  # type: ignore[attr-defined]
            raise NotImplementedError("Oja's rule not implemented for BTSP.")

        elif self.normalize_weights_divisively:  # type: ignore[attr-defined]
            self.update_weights(
                filter_key="I", lr=0
            )  # normalize weights only (no learning update)

        if self.log_BTSP and self.BTSP_applied.sum():
            step_time_str = self.Agent.get_step_and_time(as_str=True)
            neuron_str = ""
            if self.n > 1:
                applied_idxs = np.where(self.BTSP_applied)[0]
                neuron_str = "neuron" if len(applied_idxs) == 1 else "neurons"
                neurons = ", ".join([str(i) for i in applied_idxs])
                neuron_str = f" to {neuron_str} {neurons}"
            print(f"BTSP update applied{neuron_str} at {step_time_str}.")

    def _update_for_BTSP(
        self, BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list()
    ):
        """
        self._update_for_BTSP()

        Update all BTSP components for the layer.

        Args:
        - BTSP_targets (list or 1D np.ndarray, optional): List of BTSP targets.
            Default is list().
        """

        BTSP_targets = self.get_BTSP_targets(BTSP_targets)
        BTSP_targets = self.filter_BTSP_targets(BTSP_targets)
        if len(BTSP_targets) > 0:
            self._record_BTSP_events(BTSP_targets)

        self._update_BTSP_filters(BTSP_targets)
        self._update_BTSP_buffer()
        self._apply_BTSP_if_applicable()

        return

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

        with ext_util.TemporarilyChangeAttribute(self, "save_history", False):
            super().update()

        self._update_for_BTSP(BTSP_targets)

        if self.save_history is True:
            self.save_to_history()

        return

    def get_BTSP_kernel_based_cmap(
        self,
        t_pre=14,
        t_post=10,
        gray_level=0.5,
        use_alpha=True,
        min_alpha=0,
        shared_max=True,
        for_colorbar=False,
    ):
        """
        self.get_BTSP_kernel_based_cmap()

        Obtain a colormap that tracks the strength and valence of the BTSP kernel
        within a specific time range around the BTSP event.

        Args:
        - t_pre (float, optional): Time before the BTSP event to consider in
            constructing the colormap. Default is 14.
        - t_post (float, optional): Time after the BTSP event to consider in
            constructing the colormap. Default is 10.
        - gray_level (float, optional): Gray level to use for zero kernel value.
            Higher values are closer to black. Default is 0.5.
        - use_alpha (bool, optional): Whether to use alpha channel to indicate
            strength instead of color strength. Default is True.
        - min_alpha (float, optional): Minimum alpha value to use if use_alpha is
            True. Default is 0.
        - shared_max (bool, optional): Whether to use a shared maximum value
            across positive and negative values. Default is True.
        - for_colorbar (bool, optional): Whether the colormap is intended for use
            in a colorbar. If True, the colormap will reflect the strength of the
            kernel. Default is False.

        Returns:
        - cmap (ListedColormap): Colormap based on the BTSP kernel.
        - vmin (float): Minimum value of the colormap.
        - vmax (float): Maximum value of the colormap.
        """

        gray_RGB = 1 - gray_level

        kernel, _ = self.get_BTSP_kernel(t_pre=t_pre, t_post=t_post)
        if use_alpha:
            full_value = 0
            num_dim = 4
        else:
            full_value = gray_RGB
            num_dim = 3

        if shared_max:
            vmax = np.absolute(kernel).max()
            vmin = -vmax
        else:
            vmax = kernel.max()
            vmin = kernel.min()

        if for_colorbar:
            kernel = np.linspace(vmin, vmax, 1000)

        colors = np.full((len(kernel), num_dim), full_value).astype(float)

        neg_mask = kernel < 0
        if np.any(neg_mask):
            neg_scaled = np.absolute(kernel[neg_mask] / vmax)
            if use_alpha:
                colors[neg_mask, 2] = 1.0  # B
                colors[neg_mask, 3] = neg_scaled
            else:
                colors[neg_mask, 0] = (gray_RGB) * (1 - neg_scaled)  # R
                colors[neg_mask, 1] = gray_RGB * (1 - neg_scaled)  # G
                colors[neg_mask, 2] = gray_RGB + (1 - gray_RGB) * neg_scaled  # B

        pos_mask = kernel > 0
        if np.any(pos_mask):
            pos_scaled = np.absolute(kernel[pos_mask] / vmin)
            if use_alpha:
                colors[pos_mask, 0] = 1.0  # R
                colors[pos_mask, 3] = pos_scaled
            else:
                colors[pos_mask, 0] = gray_RGB + (1 - gray_RGB) * pos_scaled  # R
                colors[pos_mask, 1] = gray_RGB * (1 - pos_scaled)  # G
                colors[pos_mask, 2] = gray_RGB * (1 - pos_scaled)  # B

        if use_alpha:
            colors[:, 3] = np.maximum(colors[:, 3], min_alpha)

        cmap = ListedColormap(colors, name="BTSP_kernel_cmap", N=len(kernel))

        return cmap, vmin, vmax

    def plot_BTSP_kernel(self, autosave: bool | None = None, **kwargs):
        """
        self.plot_BTSP_kernel()

        Plot the BTSP kernel.

        Args:
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to plot_util.plot_summed_exp_kernel().

        Returns
        - sub_ax (plt.Axes): Subplot with BTSP kernel plotted.
        """

        sub_ax, _, _ = plot_util.plot_summed_exp_kernel(
            dt=self.Agent.dt,
            **self.get_BTSP_kernel_kwargs(),
            **kwargs,
        )

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_BTSP_kernel", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_filtered_for_BTSP(
        self,
        input_layer_name: str | None = None,
        t_start: float | None = None,
        component: str | None = None,
        title: str | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        in_min: bool = True,
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
        - in_min (bool, optional): Whether to plot the time in minutes instead of
            seconds. Default is True.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to HebbianLayer.plot_filtered() and
        self.plot_rate_timeseries().

        Returns:
        - sub_ax (plt.Axes): Subplot with filtered inputs or firingrates for BTSP
            plotted.
        """

        if input_layer_name is None:
            layer = self
            filter_key = "filtered_BTSP_trace"
            title = title or "Post BTSP filter"
            plot_unfiltered = False  # the filter is not on the firing rate
        else:
            if input_layer_name not in self.inputs.keys():
                raise ValueError(
                    f"Input layer '{input_layer_name}' not found. Available input "
                    f"layers: {self.inputs.keys()}."
                )
            layer = self.inputs[input_layer_name]["layer"]
            filter_key = "filtered_inputs_for_BTSP"
            title = title or "Inputs filtered for BTSP"
            plot_unfiltered = True

        if component is not None:
            if component not in ["pos", "neg"]:
                raise ValueError(
                    f"Component '{component}' not recognized. Use 'pos' or 'neg'."
                )
            filter_key = f"{filter_key}_{component}"
            title = f"{title} ({component})"

        chosen_neurons = np.asarray(
            layer.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type, attr-defined]
        )

        sub_ax, t = super().plot_filtered(
            input_layer_name=input_layer_name,
            filter_key=filter_key,
            t_start=t_start,
            chosen_neurons=chosen_neurons,
            title=title,
            plot_unfiltered=plot_unfiltered,
            in_min=in_min,
            autosave=False,
            **kwargs,
        )

        sub_ax.set_ylabel("BTSP signal")

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
                marker=mpl_markers.MarkerStyle("x"),
                s=10,
            )

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_filtered_inputs_for_BTSP", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_BTSP_frequency(
        self,
        sub_ax=None,
        width=0.2,
        t_start=None,
        t_end=None,
        applied_only=False,
        autosave=None,
    ):
        """
        self.plot_BTSP_frequency()

        Plot the frequency of BTSP events for each neuron.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - width (float, optional): Width of the bars. Default is 0.3.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - applied_only (bool, optional): Whether to plot only applied BTSP events.
            Default is False.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with BTSP frequency plotted.
        """

        counts = self.get_BTSP_counts(t_start=t_start, t_end=t_end)
        applied_counts = self.get_BTSP_counts(
            applied_only=True, t_start=t_start, t_end=t_end
        )

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(5, 2))

        bins = np.arange(counts.max() + 3)
        for i, alpha in enumerate([0.8, 0.4]):
            data = applied_counts if i == 0 else counts
            if applied_only:
                label = None
                if i == 1:
                    continue
            else:
                label = "applied BTSP events" if i == 0 else "all BTSP events"
            bin_counts, bins = np.histogram(data, bins=bins)
            centers = bins[:-1] - width / 2 if i == 0 else bins[:-1] + width / 2

            sub_ax.bar(
                centers,
                bin_counts,
                align="center",
                width=width,
                color=self.color,
                alpha=alpha,
                label=label,
            )

            sub_ax.axvline(
                data.mean(),
                color="k",
                ls="dashed",
                label=f"mean={data.mean():.1f}",
                alpha=alpha,
            )

        sub_ax.spines[["right", "top"]].set_visible(False)
        sub_ax.set_ylabel("Number of neurons")
        sub_ax.set_xlabel("Number of BTSP events")
        sub_ax.set_title("Frequency of BTSP events", y=1.1)
        sub_ax.legend()

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_BTSP_frequency", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_BTSP_step_histogram(
        self,
        sub_ax=None,
        nbins=40,
        interval=False,
        t_start=None,
        t_end=None,
        autosave=None,
    ):
        """
        self.plot_BTSP_step_histogram()

        Plot the histogram of when BTSP events occur across neurons.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - nbins (int, optional): Number of bins for the histogram. Default is 40.
        - interval (bool, optional): Whether to plot the step intervals instead of step
            numbers.
        - t_start (float, optional): Start time for including BTSP events. Default is
            None.
        - t_end (float, optional): End time for including BTSP events. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with BTSP frequency plotted.
        """

        steps = self.get_BTSP_steps(t_start=t_start, t_end=t_end)
        applied_steps = self.get_BTSP_steps(
            applied_only=True, t_start=t_start, t_end=t_end
        )
        unapplied_steps = steps[~np.isin(steps, applied_steps)]

        if interval:
            steps = np.diff(steps)
            applied_steps = np.diff(applied_steps)
            if len(applied_steps) and len(unapplied_steps):
                unapplied_steps = np.insert(unapplied_steps, 0, applied_steps[-1])
            unapplied_steps = np.diff(unapplied_steps)
            xlabel = "Step interval"
        else:
            xlabel = "Step number"

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(5, 2))

        max_bin = (np.max(steps) // nbins + 1) * nbins
        bins = np.linspace(0, max_bin, nbins)

        for i, alpha in enumerate([0.8, 0.3]):
            plot_steps = applied_steps if i == 0 else unapplied_steps
            if not len(plot_steps):
                continue
            sub_ax.hist(
                plot_steps, bins=bins, density=False, color=self.color, alpha=alpha
            )

        mean_step = np.mean(steps)
        mean_time = mean_step * self.Agent.dt
        in_min = True if mean_time > 60 * 3 else False
        mean_time_str = f"{mean_time / 60:.2f} min" if in_min else f"{mean_time:.2f} s"

        sub_ax.axvline(
            mean_step,
            color="k",
            ls="dashed",
            label=f"mean={int(mean_step)} steps / {mean_time_str}",
        )

        sub_ax.spines[["right"]].set_visible(False)
        sub_ax.set_ylabel("BTSP events")
        sub_ax.set_xlabel(xlabel)
        plot_util.pad_axis(sub_ax, axis="y", prop_high=1.0)

        twin_sub_ax = sub_ax.twiny()

        twin_xlabel = "Time (min)" if in_min else "Time (s)"
        twin_sub_ax.set_xlabel(twin_xlabel)
        x_min, x_max = sub_ax.get_xlim()
        twin_sub_ax.set_xlim(x_min * self.Agent.dt, x_max * self.Agent.dt)
        twin_sub_ax.spines[["right"]].set_visible(False)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_BTSP_frequency", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_BTSP_ramp_peak_histogram(
        self, sub_ax=None, nbins=40, t_start=None, t_end=None, autosave=None
    ):
        """
        self.plot_BTSP_ramp_peak_histogram()

        Plot the histogram of BTSP ramp peaks.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - nbins (int, optional): Maximum number of bins for the histogram. Default is 40.
        - t_start (float, optional): Start time for including BTSP events. Default is None.
        - t_end (float, optional): End time for including BTSP events. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with BTSP ramp peaks plotted.
        """

        peak_dict = self.get_BTSP_ramp_peak_dict(t_start=t_start, t_end=t_end)
        peaks = np.concatenate(list(peak_dict["peaks"].values()))

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(5, 2))

        max_bin = int(peaks.max()) + 1
        bins = np.linspace(0, max_bin, nbins)

        for i, alpha in enumerate([0.3, 0.8]):
            peaks_to_plot = peaks
            label = None
            if i == 1:
                peaks_to_plot = peaks[peaks >= 1]
                perc = len(peaks_to_plot) / len(peaks) * 100
                label = f"{perc:.2f}% reached thr."

            if len(peaks_to_plot):
                sub_ax.hist(
                    peaks_to_plot,
                    bins=bins,
                    density=False,
                    color=self.color,
                    label=label,
                    alpha=alpha,
                )

        mean_peak = np.mean(peaks)
        sub_ax.axvline(mean_peak, color="k", ls="dashed", label=f"mean={mean_peak:.2f}")

        sub_ax.spines[["right", "top"]].set_visible(False)
        sub_ax.set_ylabel("Number of peaks")
        sub_ax.set_xlabel("BTSP ramp peaks")
        sub_ax.legend()
        plot_util.pad_axis(sub_ax, axis="y", prop_high=1.0)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_BTSP_ramp_peaks", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_BTSP_stats_summary(
        self, axes=None, t_start=None, t_end=None, autosave=None
    ):
        """
        self.plot_BTSP_stats_summary()

        Plot six BTSP statistics: responses, locations, and frequency, ramp peak,
        step and step interval histograms.

        Args:
        - axes (plt.Axes, optional): Axes to plot on. If provided, must have shape
            (3, 2). Default is None.

        Returns:
        - axes (plt.Axes): Axes with BTSP statistics plotted.
        """

        if axes is None:
            fig, axes = plt.subplots(3, 2, figsize=[8, 8], gridspec_kw={"hspace": 0.4})
        elif axes.shape != (3, 2):
            raise ValueError("If passed, axes must have shape (3, 2).")

        kwargs = {"t_start": t_start, "t_end": t_end, "autosave": False}

        fig = axes.ravel()[0].figure
        fig.suptitle("BTSP statistics", y=1.05)

        self.plot_BTSP_responses(split=False, fill=False, ax=axes[0, 0], **kwargs)
        self.Agent.Environment.plot_environment(sub_ax=axes[0, 1], alpha=0.6)
        self.plot_BTSP_locations(sub_ax=axes[0, 1], **kwargs)

        self.plot_BTSP_frequency(sub_ax=axes[1, 0], **kwargs)
        self.plot_BTSP_ramp_peak_histogram(sub_ax=axes[1, 1], **kwargs)

        for i, interval in enumerate([False, True]):
            self.plot_BTSP_step_histogram(
                sub_ax=axes[2, i], interval=interval, **kwargs
            )

        plot_util.save_figure(fig, f"{self.name}_BTSP_stats_summary", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_around_BTSP(
        self,
        pre=4,
        post=2,
        linear=True,
        directional=False,
        datatype="speed",
        cm=True,
        color=None,
        cumsum=False,
        sub_ax=None,
        autosave=None,
    ):
        """
        self.plot_around_BTSP()

        Plot the speed or position around BTSP events.

        Args:
        - pre (int, optional): Number of steps before BTSP event. Default is 4.
        - post (int, optional): Number of steps after BTSP event. Default is 2.
        - linear (bool, optional): Whether to plot the linear speed. Default is True.
        - directional (bool, optional): Whether to plot the directional speed.
            Default is False.
        - cm (bool, optional): Whether to plot the speed in cm/s. Default is True.
        - datatype (str, optional): Type of data to plot ("speed" or "position").
            Default is "speed".
        - color (str, optional): Color of the plot. If None, the layer color is used.
            Default is None.
        - cumsum (bool, optional): Whether to plot the cumulative sum of the speed.
            Default is False.
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with speed around BTSP events plotted.
        """

        if datatype == "speed":
            time, data_arr = self.get_speed_around_BTSP(
                pre=pre, post=post, linear=linear, directional=directional, cm=cm
            )
            data_label = self.Agent.get_speed_label(
                linear=linear, directional=directional, incl_unit=False
            )
            unit_str = "cm/s" if cm else "m/s"

        elif datatype == "position":
            time, data_arr = self.get_position_around_BTSP(pre=pre, post=post, cm=cm)
            data_label = "Position"
            unit_str = "m" if cm else "cm"

        time_incr = np.mean(np.diff(time))  # to rescale the cumulative sum over seconds

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(6, 1.5))

        if len(data_arr) == 1:
            alphas = [0.9]
        elif len(data_arr) == 2:
            alphas = [0.5, 0.9]
        elif len(data_arr) > 2:
            alphas = np.linspace(0.3, 0.9, data_arr.shape[0])

        if data_arr.shape[1] == 2:
            labels = ["x", "y"]
            lws = [1.0, 0.6]
        else:
            labels = [None, None]
            lws = [None, None]

        color = color or self.color
        lws = [None] if data_arr.shape[-1] == 1 else []
        for i, data in enumerate(data_arr):
            for j, dim_data in enumerate(data.T):
                mask = ~np.isnan(dim_data)
                dim_data = dim_data[mask]
                if cumsum:
                    dim_data = np.cumsum(dim_data, axis=0)
                    BTSP_pt = np.argmin(np.absolute(time[mask]))
                    dim_data -= dim_data[BTSP_pt]
                    dim_data *= time_incr

                label = labels[j] if i == 0 else None
                sub_ax.plot(
                    time[mask],
                    dim_data,
                    color=color,
                    alpha=alphas[i],
                    lw=lws[j],
                    label=label,
                )

        if data_arr.shape[1] == 2:
            sub_ax.legend()

        save_label = f"{data_label.lower().replace(' ', '_')}_around_BTSP"
        if cumsum:
            sub_ax.axhline(0, color="k", ls="dashed")
            data_label = f"Cumul. {data_label[0].lower()}{data_label[1:]}"
            save_label = f"cumul_{datatype}"

        sub_ax.set_ylabel(f"{data_label} ({unit_str})")

        sub_ax.set_xlabel("Time (s)")
        sub_ax.set_title(f"{data_label} around BTSP events", y=1.1)

        sub_ax.axvline(0, color="k", ls="dashed")
        sub_ax.spines[["right", "top"]].set_visible(False)
        plot_util.pad_axis(sub_ax, axis="y")

        plot_util.save_figure(sub_ax.figure, f"{self.name}_{save_label}", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_position_around_BTSP(
        self,
        pre=4,
        post=2,
        color=None,
        cumsum=False,
        sub_ax=None,
    ):
        """
        self.plot_position_around_BTSP()

        Plot the position around BTSP events.

        Args:
        - pre (int, optional): Number of steps before BTSP event. Default is 4.
        - post (int, optional): Number of steps after BTSP event. Default is 2.
        - color (str, optional): Color of the plot. If None, the layer color is used.
            Default is None.
        - cumsum (bool, optional): Whether to plot the cumulative sum of the speed.
            Default is False.
        - sub_ax (plt.Axes, optional): Subplot to plot on. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with speed around BTSP events plotted.
        """

        time, position_arr = self.get_position_around_BTSP(pre=pre, post=post)
        BTSP_pt = np.argmin(np.absolute(time))

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(6, 2))

        if len(position_arr) == 1:
            alphas = [0.9]
        elif len(position_arr) == 2:
            alphas = [0.5, 0.9]
        elif len(position_arr) > 2:
            alphas = np.linspace(0.3, 0.9, position_arr.shape[0])

        if position_arr.shape[1] == 2:
            labels = ["x", "y"]
            lws = [1.0, 0.6]
        else:
            labels = [None, None]
            lws = [None, None]

        color = color or self.color
        lws = [None] if position_arr.shape[-1] == 1 else []
        for i, position in enumerate(position_arr):
            if cumsum:
                speed = np.cumsum(speed, axis=0)
                speed -= speed[BTSP_pt]

            for j, dim_speed in enumerate(speed.T):
                label = labels[j] if i == 0 else None
                sub_ax.plot(
                    time,
                    dim_speed,
                    color=color,
                    alpha=alphas[i],
                    lw=lws[j],
                    label=label,
                )

        if speed_arr.shape[1] == 2:
            sub_ax.legend()

        label = self.Agent.get_speed_label(
            linear=linear, directional=directional, incl_unit=False
        )
        if cumsum:
            ylabel = f"Cumul. {label[0].lower()}{label[1:]}"
        unit = "cm" if cm else "m"

        sub_ax.set_ylabel(f"{ylabel} ({unit}/s)")
        sub_ax.set_xlabel("Time (s)")
        sub_ax.set_title(f"{ylabel} around BTSP events", y=1.1)

        sub_ax.axvline(0, color="k", ls="dashed")
        sub_ax.spines[["right", "top"]].set_visible(False)

        plot_util.save_figure(sub_ax.figure, f"{self.name}_speed_around_BTSP", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_BTSP_responses(
        self,
        pre=1,
        post=2,
        num_cols=10,
        ax=None,
        split=True,
        fill=True,
        t_start=None,
        t_end=None,
        chosen_neurons="all",
        autosave=None,
    ):
        """
        self.plot_BTSP_responses()

        Plot the BTSP responses of the layer.

        Args:
        - pre (float, optional): Time before BTSP event to plot. Default is 1.
        - post (float, optional): Time after BTSP event to plot. Default is 2.
        - num_cols (int, optional): Number of columns for the subplot grid. Default is 10.
        - ax (plt.Axes or 2D np.ndarray, optional): Subplot to plot on. Default is None.
        - split (bool, optional): Whether to split the plot into subplots for each neuron.
            Default is True.
        - fill (bool, optional): Whether to fill the area under the BTSP responses.
            Default is True.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - ax (plt.Axes or 2D np.ndarray): Subplot with BTSP responses plotted.
        """

        time, BTSP_response_dict = self.get_BTSP_response_dict(
            pre=pre,
            post=post,
            chosen_neurons=chosen_neurons,
            t_start=t_start,
            t_end=t_end,
        )
        chosen_neurons = self.get_chosen_neurons(chosen_neurons)

        if ax is None:
            if split:
                n = len(chosen_neurons)
                num_cols = min(n, num_cols)
                num_rows = int(np.ceil(n / num_cols))
                fig, ax = plt.subplots(
                    num_rows,
                    num_cols,
                    figsize=(num_cols, num_rows),
                    sharex=True,
                    sharey=True,
                    squeeze=False,
                )
            else:
                fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        elif split:
            if len(ax.shape) != 2:
                raise ValueError("If split is True and 'ax' is passed, must be 2D.")
            if len(ax.ravel()) < len(chosen_neurons):
                raise ValueError("Not enough subplots in 'ax' for the chosen neurons.")
        elif not split and not isinstance(ax, plt.Axes):
            raise ValueError("If not split and 'ax' is passed, must a subplot.")

        if split:
            base_lw = 0.75
            base_alpha = 0.2
        else:
            base_lw = 0.5
            base_alpha = 0.1

        legend_kwargs = {
            "handletextpad": -0.1,
            "handlelength": 0,
            "loc": "upper right",
            "fontsize": 5,
        }

        all_responses = list()
        num_neurons_with_BTSP = 0
        for i, neuron_idx in enumerate(BTSP_response_dict.keys()):
            sub_ax = ax.ravel()[i] if split else ax

            if split or i == 0:
                sub_ax.axvline(0, color="k", ls="dashed")
                sub_ax.spines[["right", "top"]].set_visible(False)

            if BTSP_response_dict[neuron_idx] is None:
                sub_ax.plot([], [], label=f"#{neuron_idx} ({num_events})")
                sub_ax.legend(**legend_kwargs)
                continue

            num_neurons_with_BTSP += 1
            all_responses.append(BTSP_response_dict[neuron_idx])
            if fill:
                for response in BTSP_response_dict[neuron_idx]:
                    sub_ax.fill_between(
                        time,
                        np.zeros_like(response),
                        response,
                        color=self.color,
                        alpha=base_alpha,
                        lw=0,
                    )

            sub_ax.plot(
                time,
                BTSP_response_dict[neuron_idx].T,
                color=self.color,
                alpha=base_alpha * 2,
                lw=base_lw,
                ls="dashed",
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message="Mean of empty slice"
                )
                mean_response = np.nanmean(BTSP_response_dict[neuron_idx], axis=0)

            sub_ax.plot(
                time,
                mean_response,
                color=self.color,
                alpha=base_alpha * 4,
                lw=base_lw * 2,
            )

            if split:
                num_events = len(BTSP_response_dict[neuron_idx])
                sub_ax.plot([], [], label=f"#{neuron_idx} ({num_events})")
                sub_ax.legend(**legend_kwargs)

        if split and len(chosen_neurons) < len(ax.ravel()):
            for sub_ax in ax.ravel()[len(chosen_neurons) :]:
                sub_ax.axis("off")

        plot_util.pad_axis(sub_ax, axis="y", prop_high=1.0)

        if split:
            fig.suptitle("BTSP responses", y=0.96)
            for sub_ax in ax[:, 0]:
                sub_ax.set_ylabel("Firing rate")
            for sub_ax in ax[-1]:
                sub_ax.set_xlabel("Time (s)")
        else:
            color = "white" if fill else self.color
            all_responses = np.concatenate(all_responses)
            ax.plot(
                time, np.nanmean(all_responses, axis=0), color=color, alpha=0.8, lw=3
            )
            ax.set_ylabel("Firing rate")
            ax.set_xlabel("Time (s)")
            ax.set_title(
                f"BTSP responses ({len(all_responses)} from "
                f"{num_neurons_with_BTSP}/{len(chosen_neurons)} neurons)"
            )

        fig = np.asarray(ax).ravel()[0].figure
        plot_util.save_figure(fig, f"{self.name}_BTSP_responses", save=autosave)  # type: ignore[attr-defined]

        return ax

    def plot_BTSP_locations(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        sub_ax: plt.Axes | None = None,
        color: str | None = None,
        autosave: bool | None = None,
    ):
        """
        self.plot_BTSP_locations()

        Plot the locations of BTSP events.

        Args:
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time. Default is None.
        - chosen_neurons (str, int, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created with environment plotted. Default is None.
        - color (str, optional): Color of the BTSP markers. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with BTSP locations plotted.
        """

        if color is None:
            color = self.color or "C1"

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type]
        num_neurons = len(chosen_neurons)

        t, startid, _ = self.get_plotting_times(t_start=t_start, t_end=t_end)

        BTSP_step_dict = self.get_BTSP_step_dict(t_end=t_end)

        min_not_yet = self.num_steps_total - self.BTSP_buffer["num_steps"].max()

        if sub_ax is None:
            sub_ax = self.Agent.Environment.plot_environment(alpha=0.6)

        if self.Agent.Environment.dimensionality == "1D":
            ymax = sub_ax.get_ylim()[1]

        for target, steps in BTSP_step_dict.items():
            for step in steps:
                if step < startid:
                    alpha = 0.6  # happened before
                elif step >= min_not_yet:
                    alpha = 0.3  # not yet applied
                else:
                    alpha = 1.0

                if target not in chosen_neurons:
                    continue

                i = chosen_neurons.index(target)

                pos = self.Agent.history["pos"][step]
                if self.Agent.Environment.dimensionality == "1D":
                    line_sep = (ymax - 1) / num_neurons
                    y_pos = 1 + line_sep * i + line_sep * 0.7
                    pos = pos + [y_pos]

                sub_ax.scatter(
                    *pos,
                    color=color,
                    alpha=alpha,
                    marker=mpl_markers.MarkerStyle("x"),
                    s=10,
                )

        plot_util.save_figure(sub_ax.figure, f"{self.name}_BTSP_locations", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def add_BTSP_markers_to_plots(
        self,
        ax: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | plt.Axes,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        timeseries: bool = False,
        in_min: bool = True,
        in_steps: bool = False,
        color: str | None = None,
        s: int = 10,
        marker: str | tuple = "x",
        prop_y: float = 0.95,
        lw: float = 1.0,
        zorder: int = 5,
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
        - in_min (bool, optional): Whether to plot time in minutes if timeseries is
            True. Default is True.
        - in_steps (bool, optional): Whether to plot time in steps if timeseries is
            True. If True, takes precedence over in_min. Default is False.
        - color (str, optional): Color of the BTSP markers. Default is None.
        - s (int, optional): Size of the BTSP markers. Default is 10.
        - marker (str, optional): Marker style for the BTSP markers. Default is "x".
        - prop_y (float, optional): Proportional y-offset for the markers if timeseries
            is True. Default is 0.95.
        - lw (float, optional): Line width of the markers. Default is 1.
        - zorder (int, optional): Z-order of the markers. Default is 5.
        """

        if color is None:
            color = self.color or "C1"

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type]
        num_neurons = len(chosen_neurons)

        t, startid, _ = self.get_plotting_times(
            t_start=t_start, t_end=t_end, raise_error=False
        )
        if len(t) == 0:
            return

        ax1D = np.asarray(ax).ravel()
        if timeseries or self.Agent.Environment.dimensionality == "1D":
            if len(ax1D) != 1:
                raise ValueError(
                    "Only one axis expected for timeseries or 1D rate map."
                )
            sub_ax = ax1D[0]
            ymax = sub_ax.get_ylim()[1]

        min_not_yet = self.num_steps_total - self.BTSP_buffer["num_steps"].max()

        BTSP_step_dict = self.get_BTSP_step_dict(t_end=t_end)
        for target, steps in BTSP_step_dict.items():
            for step in steps:
                if step < startid:
                    alpha = 0.7  # happened before
                elif step >= min_not_yet:
                    alpha = 0.5  # not yet applied
                else:
                    alpha = 1.0

                if target not in chosen_neurons:
                    continue

                i = chosen_neurons.index(target)

                if timeseries:
                    if step < startid:
                        continue
                    if in_steps:
                        x_pos = step
                    else:
                        x_pos = t[step - startid]
                        if in_min:
                            x_pos = x_pos / 60
                    line_sep = (ymax - 1) / num_neurons
                    y_pos = 1 + line_sep * i + line_sep * prop_y
                    pos = [x_pos, y_pos]
                else:
                    pos = self.Agent.history["pos"][step]
                    if self.Agent.Environment.dimensionality == "1D":
                        line_sep = (ymax - 1) / num_neurons
                        y_pos = 1 + line_sep * i + line_sep * prop_y
                        pos = pos + [y_pos]
                    else:
                        sub_ax = ax1D[i]

                sub_ax.scatter(
                    *pos,
                    color=color,
                    alpha=alpha,
                    marker=marker,
                    s=s,
                    lw=lw,
                    zorder=zorder,
                )

    def plot_BTSP_ramp(
        self,
        sub_ax=None,
        mark_BTSP=True,
        neuron_idx=0,
        t_start=None,
        t_end=None,
        in_min=True,
        mark_threshold=True,
        marker="x",
        s=10,
        lw=1.2,
        autosave=None,
    ):
        """
        self.plot_BTSP_ramp()

        Plot the BTSP ramp of the layer.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - mark_BTSP (bool, optional): Whether to plot BTSP event markers.
            Default is True.
        - neuron_idx (int, optional): Index of the neuron to plot. Default is 0.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - in_min (bool, optional): Whether to plot the time in minutes. Default is True.
        - mark_threshold (bool, optional): Whether to mark the BTSP threshold.
            Default is True.
        - marker (str, optional): Marker style for BTSP event markers. Default is "x".
        - s (int, optional): Marker size for BTSP event markers. Default is 10.
        - lw (float, optional): Line width for the BTSP ramp plot. Default is 1.2.
        - autosave (bool, optional): Whether to autosave the figure. If None, the global
            autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with BTSP ramp plotted.
        """

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=[7, 1.5])

        if neuron_idx >= self.n:
            raise ValueError(f"Neuron index {neuron_idx} not in range.")

        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)
        if in_min:
            t = t / 60

        BTSP_ramp = np.asarray(self.history["BTSP_ramp"])[
            startid : endid + 1, neuron_idx
        ]

        sub_ax.plot(t, BTSP_ramp, lw=lw, color=self.color)
        sub_ax.fill_between(
            t,
            np.zeros(len(t)),
            BTSP_ramp,
            lw=0,
            alpha=0.2,
            color=self.color,
        )
        sub_ax.set_ylabel("Prop. of BTSP\nthreshold reached")
        sub_ax.spines[["top", "right"]].set_visible(False)
        if mark_threshold:
            sub_ax.axhline(1, ls="dotted", lw=lw, color=self.color)
        plot_util.pad_axis(sub_ax, axis="y", prop_high=1.0)

        xlabel = "Time (min)" if in_min else "Time (s)"
        sub_ax.set_xlabel(xlabel)

        if mark_BTSP:
            self.add_BTSP_markers_to_plots(
                ax=sub_ax,
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=[neuron_idx],
                timeseries=True,
                marker=marker,
                s=s,
                lw=lw,
                color=(self.color or "k"),
                prop_y=1.0,
            )

        fig = sub_ax.figure

        plot_util.save_figure(fig, f"{self.name}_BTSP_ramp", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_BTSP_ramp_factors(
        self,
        axes=None,
        mark_BTSP=True,
        neuron_idx=0,
        t_start=None,
        t_end=None,
        in_min=True,
        autosave=None,
    ):
        """
        self.plot_BTSP_ramp_factors()

        Plot the BTSP ramp factors of the layer.

        Args:
        - axes (1 or 2D np.ndarray): Array of subplots to plot on,
            with shape (2, 1) or (2, ). Default is None.
        - mark_BTSP (bool, optional): Whether to plot BTSP event markers.
            Default is True.
        - neuron_idx (int, optional): Index of the neuron to plot. Default is 0.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - in_min (bool, optional): Whether to plot the time in minutes. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

        Returns:
        - axes (1 or 2D np.ndarray): Array of subplots with BTSP ramp variables plotted.
            If input axes was None, shape is 2D (2, 1).
        """

        if axes is None:
            _, axes = plt.subplots(2, 1, figsize=[7, 3], sharex=True, squeeze=False)
        elif axes.shape != (2, 1) and axes.shape != (2,):
            raise ValueError("axes must have shape (2, 1) or (2, ).")

        if neuron_idx >= self.n:
            raise ValueError(f"Neuron index {neuron_idx} not in range.")

        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)
        if in_min:
            t = t / 60

        BTSP_ramp = np.asarray(self.history["BTSP_ramp"])[
            startid : endid + 1, neuron_idx
        ]
        firingrate = np.asarray(self.history["firingrate"])[
            startid : endid + 1, neuron_idx
        ]

        ax1D = np.asarray(axes).ravel()
        self.plot_BTSP_ramp(ax1D[0])
        ax1D[0].set_xlabel("")

        ax1D[1].plot(t, firingrate, lw=1.2, color=self.color)
        ax1D[1].axhline(self.BTSP_induction_threshold, ls="dashed", color=self.color)

        xlabel = "Time (min)" if in_min else "Time (s)"
        ax1D[1].set_xlabel(xlabel)
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

        if BTSP_ramp.max() < 0:
            ax1D[1].legend()

        fig = ax1D[0].figure

        plot_util.save_figure(fig, f"{self.name}_BTSP_ramp_factors", save=autosave)  # type: ignore[attr-defined]

        return axes

    def plot_rate_map(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: (
            str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]]
        ) = "all",
        mark_BTSP: bool = True,
        BTSP_s: int = 10,
        BTSP_lw: float = 1.0,
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
        - mark_BTSP (bool, optional): Whether to include BTSP markers. Default is True.
        - BTSP_s (int, optional): Size of the BTSP markers. Default is 10.
        - BTSP_lw (float, optional): Line width of the BTSP markers. Default is 1.0.
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
                s=BTSP_s,
                lw=BTSP_lw,
            )

        fig = np.asarray(ax).ravel()[0].figure
        plot_util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

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
        BTSP_s: int = 10,
        BTSP_lw: float = 1.0,
        BTSP_marker: str | tuple = "x",
        in_min: bool = True,
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
        - mark_BTSP (bool, optional): Whether to include BTSP markers. Default is True.
        - BTSP_s (int, optional): Size of the BTSP markers. Default is 10.
        - BTSP_lw (float, optional): Line width of the BTSP markers. Default is 1.0.
        - BTSP_marker (str, optional): Marker style for the BTSP markers. Default is "x".
        - in_min (bool, optional): Whether to plot time in minutes. Default is True.
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
            in_min=in_min,
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
                in_min=in_min,
                s=BTSP_s,
                lw=BTSP_lw,
                marker=BTSP_marker,
            )

        if xlim is not None:
            sub_ax.set_xlim(xlim)

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return sub_ax

    def plot_normalization_values_over_time(
        self,
        input_layer: str,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: int | str = "all",
        mark_BTSP: bool = True,
        bias_norms: bool = False,
        in_min: bool = True,
        lw: float = 1,
        autosave: bool | None = None,
        sub_ax: plt.Axes | None = None,
    ):
        """
        self.plot_normalization_values_over_time()

        Plot the normalization values for the weights of the layer over time.

        Args:
        - input_layer (str): Name of the input layer.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - chosen_neurons (int, str, list or 1D np.ndarray, optional): Neurons to plot.
            Default is "all".
        - mark_BTSP (bool, optional): Whether to mark the BTSP. Default is True.
        - bias_norms (bool, optional): Whether to plot the bias normalization values.
            Default is False.
        - in_min (bool, optional): Whether to plot the time in minutes. Default is True.
        - lw (float, optional): Line width of the plot. Default is 1.
        - autosave (bool, optional): Whether to save the figure. Default is None.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with normalization values plotted.
        """

        sub_ax = super().plot_normalization_values_over_time(
            input_layer=input_layer,
            t_start=t_start,
            t_end=t_end,
            chosen_neurons=chosen_neurons,
            bias_norms=bias_norms,
            in_min=in_min,
            lw=lw,
            autosave=False,
            sub_ax=sub_ax,
        )

        if mark_BTSP:
            pre_ymin, pre_ymax = sub_ax.get_ylim()
            sub_ax.autoscale(axis="y")
            plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.5, prop_high=1.0)

            self.add_BTSP_markers_to_plots(
                ax=sub_ax,
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                timeseries=True,
                in_min=in_min,
                lw=lw,
            )
            plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.05, prop_high=1.0)

            ymin, ymax = sub_ax.get_ylim()
            sub_ax.set_ylim(min(pre_ymin, ymin), max(pre_ymax, ymax))

        fig = sub_ax.figure
        plot_util.save_figure(fig, f"{self.name}_norm_over_time", save=autosave)  # type: ignore[attr-defined]

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

        Obtain the decay time constant for the NMDA receptor activation or
        desensitization.

        Args:
        - decay_type (str, optional): Type of decay
           ("activation" or "desensitization"). Default is "activation".
        - dt (float, optional): Time step. If None, agent time step is used.
            Default is None.

        Returns:
        - decay (float): Decay time constant.
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
        **kwargs,
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
        - **kwargs: Mostly ignored keyword arguments (hacky!)

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
        - **kwargs: Keyword arguments passed to plot_fcts.plot_timeseries().

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
            height = max([1, min(n / 12.0 + 1, 8)]) * len(datatypes)
            _, ax = plt.subplots(
                len(datatypes),
                1,
                sharex=True,
                sharey=False,
                figsize=[6, height],
                squeeze=False,
            )  # type: ignore[assignment]

        ax1D = np.asarray(ax).ravel()

        if len(ax1D) != len(datatypes):
            raise ValueError(
                f"Number of subplots ({len(ax1D)}) does not match number of datatypes "
                f"to plot ({len(datatypes)})."
            )

        for d, datatype in enumerate(datatypes):
            if datatype not in all_datatypes:
                raise ValueError(
                    f"Datatype {datatype} not recognized. "
                    f"Must be one of {all_datatypes}."
                )
            plot_fcts.plot_timeseries(
                self,  # type: ignore[assignment]
                t_start=t_start,
                sub_ax=ax1D[d],
                trace_name=datatype,
                chosen_neurons=chosen_neurons,
                autosave=False,
                **kwargs,
            )

            ax1D[d].set_ylabel(datatype.capitalize().replace("_", "\n"))
            if d != len(datatypes) - 1:
                ax1D[d].set_xlabel("")

        fig = ax1D[0].figure

        y = 0.9 if self.Agent.Environment.dimensionality == 1 else 0.97
        fig.suptitle(title, y=y)

        plot_util.save_figure(fig, f"{self.name}_NMDA_current_traces", save=autosave)  # type: ignore[attr-defined]

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
        • self.plot_BTSP_ramp_factors()
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

    def plot_BTSP_ramp_factors(
        self,
        axes=None,
        t_start=None,
        t_end=None,
        neuron_idx=0,
        in_min=True,
        autosave=None,
        **kwargs,
    ):
        """
        self.plot_BTSP_ramp_factors()

        Plot the BTSP ramp factors of the layer.

        Args:
        - axes (1 or 2D np.ndarray): Array of subplots to plot on,
            with shape (3, 1) or (3, ). Default is None.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time of the plot. Default is None.
        - neuron_idx (int, optional): Index of the neuron to plot. Default is 0.
        - in_min (bool, optional): Whether to plot time in minutes. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the global
        autosave setting for ratinabox is used. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to plot_fcts.plot_BTSP_ramp_factors().

        Returns:
        - axes (1 or 2D np.ndarray): Array of subplots with BTSP ramp variables plotted.
            If input axes was None, shape is 2D (3, 1).
        """

        if axes is None:
            _, axes = plt.subplots(3, 1, figsize=[7, 4], sharex=True, squeeze=False)
        elif axes.shape != (3, 1) and axes.shape != (3,):
            raise ValueError("axes must have shape (3, 1) or (3, ).")

        super().plot_BTSP_ramp_factors(
            t_start=t_start,
            t_end=t_end,
            axes=axes[:2],
            in_min=in_min,
            neuron_idx=neuron_idx,
            autosave=False,
            **kwargs,
        )

        t, startid, endid = self.get_plotting_times(t_start, t_end)
        current = np.asarray(self.inputs["NMDACurrent"]["layer"].history["current"])[
            startid : endid + 1, neuron_idx
        ]

        if in_min:
            t = t / 60

        ax1D = np.asarray(axes).ravel()
        ax1D[2].plot(
            t,
            current,
            lw=1.2,
            color=self.color,
        )
        ax1D[1].set_xlabel("")

        xlabel = "Time (min)" if in_min else "Time (s)"
        ax1D[2].set_xlabel(xlabel)
        ax1D[2].set_ylabel("Pyr. NMDA current")
        ax1D[2].spines[["top", "right"]].set_visible(False)

        fig = ax1D[0].figure
        plot_util.save_figure(fig, f"{self.name}_BTSP_ramp", save=autosave)  # type: ignore[attr-defined]

        return axes
