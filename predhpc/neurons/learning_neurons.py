import copy
from typing import TYPE_CHECKING, Any, Sequence

from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import markers
from matplotlib import figure as mpl_figure
import numpy as np

from ratinabox.Neurons import Neurons, FeedForwardLayer  # type: ignore[import]
from ratinabox import utils as rutils  # type: ignore[import]

from predhpc import util, plot_util, params_util

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


class SmoothFeedForwardLayer(FeedForwardLayer, util.ParamsManagerMixin):
    """This class defines a population of neurons that receive feedforward input that
    is smoothed.
    This class is a subclass of FeedForwardLayer() and inherits its properties/plotting functions.

    Must be initialised with an Agent, and a "params" dictionary, including input
    layers.

    List of functions:
        • get_state()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()
    """

    default_params = {
        "n": 10,
        "activation_function": params_util.LINEAR_SIGMOID_ACTIVATION_FUNCTION,
        "name": "SmoothFeedForwardLayer",
        "input_filter_tau": 0.1,  # in sec
        "input_trend_tau": None,  # in sec
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """Initialise HebbianLayer(), takes as input a parameter
        dictionary. Any values not provided by the params dictionary are
        taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.activation_params = self.params[
            "activation_function"
        ]  # store activation parameters
        super().__init__(Agent, self.params)

        return

    def get_filter_tau(self, filter_tau: float | None = None) -> float:
        """Returns an exponential filter time constant parameter." """

        if filter_tau is None:
            filter_tau = float(self.Agent.dt)

        elif filter_tau < self.Agent.dt:
            raise ValueError(
                f"'filter_tau' ({filter_tau}) cannot be smaller than "
                f"self.Agent.dt ({self.Agent.dt})."
            )

        return filter_tau

    def plot_activation_function(
        self, min_input_fr=-15, max_input_fr=15, fig=None, ax=None
    ):
        fig, ax = plot_util.plot_activation_function(
            self.activation_function,
            min_input_fr=min_input_fr,
            max_input_fr=max_input_fr,
            fig=fig,
            ax=ax,
            color=self.color,
        )

        ax.set_title("Activation function")

        return fig, ax

    def plot_firingrate_distribution(self, fig=None, ax=None, bins=50):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))

        firingrates = np.asarray(self.history["firingrate"]).reshape(-1)
        ax.hist(firingrates, bins=bins, color=self.color, alpha=0.6, density=True)

        ax.axvline(0, color="k", lw=1, ls="dashed")

        ax.set_xlabel("Firing rate")
        ax.set_ylabel("Density")
        ax.set_title("Firing rate distribution")

        ax.spines[["right", "top"]].set_visible(False)

        return fig, ax

    def add_input(self, input_layer: Neurons, **kwargs):
        super().add_input(input_layer, **kwargs)

        if self.input_filter_tau or self.input_trend_tau:  # type: ignore[attr-defined]
            name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]
            self.inputs[name_in]["filtered_inputs"] = np.full(n_in, np.nan)
            self.inputs[name_in]["filtered_trends"] = np.zeros(n_in)

            for key in ["inputs", "trends"]:
                if f"filtered_{key}_for_learning" not in self.history.keys():
                    self.history[f"filtered_{key}"] = dict()
                self.history[f"filtered_{key}"][name_in] = list()

    def save_to_history(self):
        """Save the current state of the layer to the history, including the
        loss, if applicable.
        """

        super().save_to_history()

        if self.input_filter_tau:  # type: ignore[attr-defined]
            for name, input_layer in self.inputs.items():
                for key in ["inputs", "trends"]:
                    self.history[f"filtered_{key}"][name].append(
                        input_layer[f"filtered_{key}"].tolist()
                    )

    def update_filtered_inputs(
        self,
        filter_tau: float | None = None,
        trend_tau: float | None = None,
        filter_key: str = "filtered_inputs",
    ):
        """Update the filtered inputs of the layer."""

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

    def get_state(self, evaluate_at="last", max_recurrence=None, **kwargs):
        """Taken from FeedForward.get_state()

        Args:
            evaluate_at (str, optional). Defaults to 'last'.
            max_recurrence: The maximum number of time get_state() recursively calls
                recurrent inputs (prevents infinite recursion error). Default is None.

        Returns:
            firingrate: array of firing rates
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
            else:  # kick can down the road let input layer decide how to evaluate the firingrate. this is core to feedforward layer as this recursive call will backprop through the upstraem layers until it reaches a "core" (e.g. place cells) layer which will then evaluate the firingrate.
                I = inputlayer["layer"].get_state(
                    evaluate_at=evaluate_at,
                    max_recurrence=pass_max_recurrence,
                    **kwargs,
                )

            inputlayer["I_temp"] = I
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

    def get_plotting_times(
        self, t_start: float | None = None, t_end: float | None = None
    ):
        """Get the times to plot.

        Args:
            t_start (float, optional): Start time. Defaults to None.
            t_end (float, optional): End time. Defaults to None.

        Returns:
            t: Times to plot.
            startid: Index of the start time.
            endid: Index of the end time.
        """

        t = np.array(self.history["t"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t = t[startid : endid + 1]

        return t, startid, endid

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        adjust_xlim: bool = True,
        autosave: bool | None = None,
        **kwargs,
    ):
        t, _, _ = self.get_plotting_times(t_start, t_end)
        t_start = t[0]
        t_end = t[-1]

        fig, ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            autosave=False,
            **kwargs,
        )

        if adjust_xlim:
            xlim = np.asarray([t_start, t_end]) / 60
            ax.set_xlim(*xlim)

            xticks = np.around(xlim, 2)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)

        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return fig, ax

    def update(self):
        """Update the layer, and filtered inputs"""

        if self.input_filter_tau or self.input_trend_tau:
            self.update_filtered_inputs(
                self.input_filter_tau,
                self.input_trend_tau,
                filter_key="filtered_inputs",
            )
        super().update()

    def plot_filtered(
        self,
        input_layer_name: str,
        filter_key: str = "filtered_inputs",
        t_start: float | None = None,
        t_end: float | None = None,
        title: str | None = None,
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        autosave: bool | None = None,
    ) -> tuple[
        mpl_figure.Figure,
        plt.Axes,
        np.ndarray[tuple[int], np.dtype[np.float64]],
    ]:
        """Plot the filtered inputs of the layer.

        Args:
            input_layer_name (str): Name of the input layer to plot.
            filter_key (str, optional): Key of the filtered inputs to plot.
                Defaults to "filtered_inputs_for_learning".
            t_start (float, optional): Start time of the plot. Defaults to None.
            t_end (float, optional): End time of the plot. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            chosen_neurons (str or array, optional): Neurons to plot. Defaults to "all".
            fig (mpl_figure.Figure, optional): Figure to plot on.
                Defaults to None.
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            autosave (bool, optional): Whether to save the figure. Defaults to None.

        Raises:
            ValueError: If the input layer is not found.

        Returns:
            fig, ax: Figure and axes of the plot.
        """

        t, startid, endid = self.get_plotting_times(t_start, t_end)

        if input_layer_name not in self.inputs.keys():
            raise ValueError(
                f"Input layer '{input_layer_name}' not found. Available input layers: "
                f"{self.inputs.keys()}."
            )

        if filter_key not in self.history.keys():
            raise ValueError(f"Filter key '{filter_key}' not found.")

        input_layer = self.inputs[input_layer_name]["layer"]

        chosen_neurons = np.asarray(
            input_layer.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type, attr-defined]
        )

        unfiltered = np.asarray(input_layer.history["firingrate"])[  # type: ignore[attr-defined]
            startid : endid + 1, chosen_neurons
        ]
        filtered = np.asarray(self.history[filter_key][input_layer_name])[
            startid : endid + 1, chosen_neurons
        ]

        height = 0.6 * (unfiltered.max() - unfiltered.min())
        shifts = np.arange(unfiltered.shape[1]).reshape(1, -1) * height

        if ax is None:
            n = unfiltered.shape[1]
            height = max([1, min(n / 12.0 + 5 / 3, 8)])
            fig, ax = plt.subplots(figsize=[6, height])

        color = input_layer.color  # type: ignore[attr-defined]

        ax.plot(t, unfiltered + shifts, ls=(0, (1, 1)), color=color, alpha=1.0)
        ax.plot(t, filtered + shifts, alpha=0.8, color=color)
        for i, shift in enumerate(shifts.T):
            ax.fill_between(
                t, shift, filtered[:, i] + shift, color=color, alpha=0.4, lw=0
            )

        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Firing rate")
        ax.set_xlabel("Time (s)")

        if title is None:
            title = filter_key.replace("_", " ").capitalize()
        ax.set_title(title)

        if fig is None:
            fig = ax.figure

        util.save_figure(fig, f"{self.name}_{filter_key}", save=autosave)  # type: ignore[attr-defined]

        return fig, ax, t


class LearnLayer(SmoothFeedForwardLayer, util.ParamsManagerMixin):
    """This trained class defines a population of neurons that tune their
    activity through Hebbian learning.
    This class is a subclass of SmoothFeedForwardLayer() and inherits its properties/plotting functions.

    Must be initialised with an Agent, and a "params" dictionary, including input
    layers.

    List of functions:
        • get_state()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()
    """

    default_params = {
        "n": 10,
        "name": "LearnLayer",
        "lr": 1e-4,  # learning rate
        "biases": None,
        "use_targets": False,
        "init_weights_zero": False,  # whether to initialize weights to 0
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
        """Initialise HebbianLayer(), takes as input a parameter
        dictionary. Any values not provided by the params dictionary are
        taken from a default dictionary below.

        Args:
            params (dict, optional). Defaults to dict().
        """

        self.Agent = Agent

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(Agent, self.params)

        self.history["target_mse"] = list()
        self.num_steps_total = 0

        return

    @property
    def trainable_biases(self) -> bool:
        if not hasattr(self, "_trainable_biases"):
            self._trainable_biases = self.params["biases"] is not None
        return self._trainable_biases

    @property
    def target(self) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Calculate the target firing rate of the layer.

        Returns:
            array: Target position [x, y] or None.
        """

        if self.use_targets:  # type: ignore[attr-defined]
            return self.Agent.pos
        else:
            return None

    def add_input(self, input_layer: Neurons, **kwargs):
        """Add an input layer to the HebbianLayer.

        Args:
            input_layer (_type_): _description_
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

    def update(self):
        """Update the layer, i.e. calculate the new firing rates and update the weights
        and biases, if applicable."""

        super().update()
        self.num_steps_total += 1

    def save_to_history(self):
        """Save the current state of the layer to the history, including the
        loss, if applicable.
        """

        super().save_to_history()

        if self.target is not None:
            target_mse = np.mean((self.target - self.firingrate) ** 2)
            self.history["target_mse"].append(target_mse)

    def plot_rate_map(
        self,
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        shape: tuple | None = None,
        target_num_col: int = 15,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]]:
        """Plot the rate map of the layer, ensuring no more than 20 columns
        are plotted.

        See FeedForwardLayer.plot_rate_map() for more information.

        Args:
            chosen_neurons (list, optional): List of neurons to plot. Defaults to "all".
            shape (tuple, optional): Shape of the plot. Defaults to None.
            target_num_col (int, optional): Aimed number of columns. Defaults to 15.
            no_legend (bool, optional): Whether to remove the legend. Defaults to False.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        if shape is None:
            n = len(self.return_list_of_neurons(chosen_neurons=chosen_neurons))  # type: ignore[arg-type]
            shape = plot_util.get_plot_shape(n, target_num_col=target_num_col)

        kwargs["chosen_neurons"] = chosen_neurons
        kwargs["shape"] = shape

        fig, axes = super().plot_rate_map(autosave=False, **kwargs)

        if no_legend:
            for ax in np.asarray(axes).reshape(-1):
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

        util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

        return fig, axes

    def plot_rate_maps_across_learning(
        self,
        num_maps: int = 3,
        prop_each: float = 0.4,
        normalize_together: bool = True,
        title: str | None = None,
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        fig: mpl_figure.Figure | None = None,
        axes: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | None = None,
        autosave: bool | None = None,
        **kwargs,
    ):
        """Plot the rate maps of the layer across learning.

        Args:
            num_maps (int, optional): Number of maps to plot. Defaults to 3.
            prop_each (float, optional): Proportion of the learning period to plot
                for each map. Defaults to 0.4.
            normalize_together (bool, optional): Whether to normalize the maps
                together. Defaults to True.
            title (str, optional): Title of the plot. Defaults to None.
            chosen_neurons (str or array, optional): Neurons to plot. Defaults to "all".
            fig (mpl_figure.Figure, optional): Figure object. Defaults to None.
            axes (np.ndarray, optional): Axes object. Defaults to None.
            autosave (bool, optional): Whether to save the figure. Defaults to None.

        Keyword Args:
            **kwargs: Keyword arguments for the plot_rate_map() function.

        Returns:
            fig, axes: Figure and axes objects.
        """

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type, attr-defined]

        # initialize axes
        if fig is None or axes is None:
            if len(chosen_neurons) == 1:
                ncols = num_maps
                nrows = len(chosen_neurons)
                row = "chosen_neurons"
            else:
                ncols = len(chosen_neurons)
                nrows = num_maps
                row = "num_maps"
            fig, axes = plt.subplots(
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
                fig=fig,
                ax=map_axes,
                t_start=t_start,
                t_end=t_end,
                method="history",
                colorbar=False,
                chosen_neurons=chosen_neurons,
                autosave=False,
            )

            subplots.append(map_axes.reshape(-1)[0])

        if normalize_together:
            plot_util.normalize_cmaps(subplots, shrink=0.7)

        if title is None:
            title = "Rate maps across learning"
        fig.suptitle(title, y=0.90)

        util.save_figure(fig, f"{self.name}_rate_maps_across_learning", save=autosave)  # type: ignore[attr-defined]

        return fig, axes

    def plot_loss(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        color: str | None = None,
        alpha: float = 0.7,
        xlim: tuple[float, float] | None = None,
        k_prop_to_loss_length: float = 0.15,
        k_max: int = 10000,
        autosave: bool | None = None,
        **loss_kwargs,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """
        Plot the loss of the layer over time.

        Args:
            t_start (float, optional): Start time of the plot. Defaults to None.
            t_end (float, optional): End time of the plot. Defaults to None.
            fig (mpl_figure.Figure, optional): Figure to plot on.
                Defaults to None.
            ax (plt.Axes, optional): Axis to plot on.
                Defaults to None.
            color (str, optional): Color of the plot. Defaults to None.
            alpha (float, optional): Alpha of the plot. Defaults to 0.7.
            xlim (tuple, optional): x limits of the plot. Defaults to None.
            k_prop_to_loss_length (float, optional): Smoothing factor, proportional to
                clength of loss array. Defaults to 0.15.
            k_max (int, optional): Maximum smoothing factor. Defaults to 10000.

        Keyword Args:
            **loss_kwargs: Keyword arguments for the plot_loss() function.

        Returns:
            fig, ax: Figure and axis of the plot.

        Raises:
            ValueError: If the layer was not trained with targets.
        """

        reset_times = None
        if hasattr(self.Agent, "trajectory_df"):
            reset_times = self.Agent.get_reset_times()  # type: ignore[attr-defined]

        if color is None:
            color = self.color  # type: ignore[attr-defined]

        fig, ax = plot_util.plot_loss(
            self.history["t"],
            self.history["target_mse"],
            mark_ts=reset_times,
            t_start=t_start,
            t_end=t_end,
            fig=fig,
            ax=ax,
            color=color,
            alpha=alpha,
            xlim=xlim,
            k_prop_to_loss_length=k_prop_to_loss_length,
            k_max=k_max,
            **loss_kwargs,
        )

        util.save_figure(fig, f"{self.name}_loss", save=autosave)  # type: ignore[attr-defined]

        return fig, ax

    def plot_histogram(
        self,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        color: str | None = None,
        alpha: float = 0.7,
        t_start: float | None = None,
        xlabel: str | None = None,
        autosave: bool | None = None,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the firing rate histogram of the layer.

        Args:
            fig (mpl_figure.Figure, optional): Figure to plot on. Defaults to None.
            ax (plt.Axes, optional): Axis to plot on. Defaults to None.
            color (str, optional): Color of the plot. Defaults to None.
            alpha (float, optional): Alpha of the plot. Defaults to 0.7.
            t_start (float, optional): Start timepoint of the plot. Defaults to None.
            xlabel (str, optional): x label of the plot. Defaults to None.
            autosave (bool, optional): Whether to save the figure. Defaults to None.

        Returns:
            fig, ax: Figure and axis of the plot.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3))

        _, startid, _ = self.get_plotting_times(t_start=t_start)
        firingrates = np.asarray(self.history["firingrate"])[startid:]

        if color is None:
            color = str(self.color)  # type: ignore[attr-defined]
        if firingrates.shape[1] > 1:
            color = [color for _ in range(firingrates.shape[1])]  # type: ignore[assignment]

        ax.hist(firingrates, color=color, alpha=alpha)

        if xlabel is None:
            xlabel = "Firing rate"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

        if fig is None:
            fig = ax.figure

        util.save_figure(fig, f"{self.name}_firing_rate_histogram", save=autosave)  # type: ignore[attr-defined]

        return fig, ax


class HebbianLayer(LearnLayer):
    """This trained class defines a population of neurons that tune their activity through Hebbian learning.
    This class is a subclass of Neurons() and inherits its properties/plotting functions.

    Must be initialised with an Agent, and a "params" dictionary, including input layers.

    List of functions:
        • get_state()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()

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
        "init_weights_zero": False,  # whether to initialize weights to 0
        "w_init_scale": 1,  # scale of the initial weights
        "learning_filter_tau": None,
        "learning_trend_tau": None,
        "p": 2,  # power for normalization, if used
    }

    ignored_param_keys = list()
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Agent: "ratinabox.Agent", params: dict = dict()):
        """Initialise HebbianLayer(), takes as input a parameter
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

        self.set_learn(True)

        if self.apply_Ojas_rule and self.normalize_weights_divisively:  # type: ignore[attr-defined]
            raise ValueError("Can only set 'oja' or 'norm' to True, not both.")

        return

    @property
    def learn(self) -> bool:
        """Property for the learn attribute.

        Returns:
            bool: Whether the layer is learning or not.
        """

        return self._learn

    def set_learn(self, learn=None):
        """Set the layer to learn."""

        if learn is None:
            pass
        else:
            self._learn = learn

    @property
    def input_layers_with_no_learning(self) -> list[str]:
        """Returns a list of input layer names that are not learning."""
        if hasattr(self, "_input_layers_with_no_learning"):
            return self._input_layers_with_no_learning
        else:
            return list()

    def add_input_layers_with_no_learning(self, input_layers) -> None:
        if not hasattr(self, "_input_layers_with_no_learning"):
            self._input_layers_with_no_learning = list()
        if not isinstance(input_layers, list):
            input_layers = [input_layers]
        self._input_layers_with_no_learning.extend(input_layers)

    def add_input(self, input_layer: Neurons, **kwargs):
        super().add_input(input_layer, **kwargs)

        name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]
        self.inputs[name_in]["filtered_inputs_for_learning"] = np.full(n_in, np.nan)
        self.inputs[name_in]["filtered_trends_for_learning"] = np.zeros(n_in)

        for key in ["inputs", "trends"]:
            if f"filtered_{key}_for_learning" not in self.history.keys():
                self.history[f"filtered_{key}_for_learning"] = dict()
            self.history[f"filtered_{key}_for_learning"][name_in] = list()

        if self.normalize_weights_divisively:
            self.update_weights(
                filter_key="I", lr=0
            )  # normalize weights only (no update)
            self.inputs[name_in]["w_init"] = copy.deepcopy(self.inputs[name_in]["w"])

    def save_to_history(self):
        """Save the current state of the layer to the history, including the
        loss, if applicable.
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
    ):
        """Update the weights of the layer.

        Args:
            filter_key (str, optional): Key of the input to use for the update.
                Defaults to "filtered_inputs_for_learning".
        """

        if O is None:
            if self.use_targets:  # type: ignore[attr-defined]
                O = np.asarray(self.target).astype(np.float64)
            else:
                # IF FILTERING INPUT, ALSO FILTER OUTPUT? POST DIRECTION?
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

        if self.apply_Ojas_rule:  # type: ignore[attr-defined]
            alpha = self.regularization_alpha or 0.1
            util.perform_Oja_update_(Is, ws, O, lr=lr, b=b, alpha=alpha)
        elif self.normalize_weights_divisively:  # type: ignore[attr-defined]
            alpha = self.regularization_alpha or 1.0
            util.perform_divisively_normalized_Hebbian_update_(
                Is, ws, O, lr=lr, b=b, p=self.p, alpha=alpha
            )
        else:
            util.perform_Hebbian_update_(Is, ws, O, lr=lr, b=b)

    def update(self):
        """Update the layer, i.e. calculate the new firing rates and update the
        weights and biases, if applicable."""

        self.update_filtered_inputs(
            self.learning_filter_tau,
            self.learning_trend_tau,
            filter_key="filtered_inputs_for_learning",
        )  # type: ignore[attr-defined]

        super().update()

        if self.learn:
            self.update_weights(filter_key="filtered_inputs_for_learning")

        return


class BTSPLayer(HebbianLayer):
    """This trained class defines a population of neurons that tune their activity
    through Hebbian learning with BTSP.
    This class is a subclass of Neurons() and inherits its properties/plotting
    functions.

    Must be initialised with an Agent, and a "params" dictionary, including input
    layers.

    List of functions:
        • get_state()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()
    """

    default_params = {
        "n": 10,
        "name": "BTSPLayer",
        "BTSP_filter_tau": 4,
        "BTSP_trend_tau": None,
        "BTSP_lr_fact": 200,
        "single_BTSP": False,
        "BTSP_distance_prop": 10,  # None to remove constraint
    }

    ignored_param_keys = list()
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Agent: "ratinabox.Agent", params: dict[str, Any] = dict()):
        """Initialise HebbianLayer(), takes as input a parameter
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

        self.history["BTSP_events"] = list()
        self.history["BTSP_targets"] = list()

        self.num_BTSP_to_date = np.zeros(self.n)  # type: ignore[attr-defined]
        self.last_BTSP_step = np.full(self.n, np.nan)  # type: ignore[attr-defined]
        self.last_BTSP_pos = [None for _ in range(self.n)]  # type: ignore[attr-defined]

        if self.use_targets:  # type: ignore[attr-defined]
            raise ValueError("BTSPLayer does not support targets.")

        self.set_BTSP_learn(True)

        return

    @property
    def BTSP_learn(self) -> bool:
        """Returns the BTSP learning state of the layer.

        Returns:
            bool: BTSP learning state.
        """

        return self._BTSP_learn

        return

    def set_BTSP_learn(self, learn=None):
        """Set the layer to learn using BTSP."""

        if learn is None:
            pass
        else:
            self._BTSP_learn = learn
        return

    def add_input(self, input_layer: Neurons, **kwargs):
        """Adds an input layer to the layer by calling super().add_input(), and
        initialises the BTSP filtered inputs."""
        super().add_input(input_layer, **kwargs)

        name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]

        self.inputs[name_in]["filtered_inputs_for_BTSP"] = np.full(n_in, np.nan)
        self.inputs[name_in]["filtered_trends_for_BTSP"] = np.zeros(n_in)

        for key in ["inputs", "trends"]:
            if f"filtered_{key}_for_BTSP" not in self.history.keys():
                self.history[f"filtered_{key}_for_BTSP"] = dict()
            self.history[f"filtered_{key}_for_BTSP"][name_in] = list()

    def save_to_history(self):
        """Save the current state of the layer to the history, including the
        loss, if applicable.
        """

        super().save_to_history()

        for name, input_layer in self.inputs.items():
            for key in ["inputs", "trends"]:
                self.history[f"filtered_{key}_for_BTSP"][name].append(
                    input_layer[f"filtered_{key}_for_BTSP"].tolist()
                )

    def update(
        self, BTSP_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list()
    ):
        """Update the layer, i.e. calculate the new firing rates and update the
        weights and biases, if applicable.

        Args:
            BTSP_targets (list or 1D array, optional): List of BTSP targets.
                Defaults to [].
        """

        self.update_filtered_inputs(
            self.learning_filter_tau,
            self.learning_trend_tau,
            filter_key="filtered_inputs_for_learning",
        )
        self.update_filtered_inputs(
            self.BTSP_filter_tau,
            self.BTSP_trend_tau,
            filter_key="filtered_inputs_for_BTSP",
        )

        super().update()

        if self.BTSP_learn and BTSP_targets is not None and len(BTSP_targets):
            if self.single_BTSP:  # type: ignore[attr-defined]
                keep_BTSP_targets = np.asarray(
                    [targ for targ in BTSP_targets if self.num_BTSP_to_date[targ] == 0]
                )

            elif self.BTSP_distance_prop is not None:
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
            self.update_weights(filter_key="filtered_inputs_for_BTSP", lr=lr)
            self.history["BTSP_events"].append(
                self.num_steps_total - 1
            )  # recorded after update
            self.history["BTSP_targets"].append(BTSP_targets)

        return

    def plot_filtered_for_BTSP(
        self,
        input_layer_name: str,
        t_start: float | None = None,
        title: str | None = None,
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the filtered inputs of the layer.

        Args:
            input_layer_name (str): Name of the input layer to plot.
            t_start (float, optional): Start time of the plot. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            chosen_neurons (str or array, optional): Neurons to plot. Defaults to "all".
            **kwargs: Keyword arguments passed to the plot function.

        Returns:
            fig, ax: Figure and axes of the plot.
        """

        if title is None:
            title = "Inputs filtered for BTSP"

        input_layer = self.inputs[input_layer_name]["layer"]
        chosen_neurons = np.asarray(
            input_layer.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type, attr-defined]
        )

        fig, ax, t = super().plot_filtered(
            input_layer_name,
            filter_key="filtered_inputs_for_BTSP",
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
        shift = np.diff(ax.get_ylim())[0] / (n + 0.4)
        heights = (np.arange(n + 1)[1:] - 0.35) * shift

        plot_BTSP_events, plot_BTSP_targets, plot_heights = list(), list(), list()
        for ev, targ in zip(flat_BTSP_events, flat_BTSP_targets):
            if targ in chosen_neurons:
                idx = chosen_neurons.tolist().index(targ)
                plot_BTSP_events.append(ev)
                plot_BTSP_targets.append(targ)
                plot_heights.append(heights[idx])

        if len(plot_BTSP_events):
            ax.scatter(
                t[np.asarray(plot_BTSP_events)],
                plot_heights,
                color=(self.color or "k"),
                alpha=0.8,
                marker=markers.MarkerStyle("x"),
                s=10,
            )

        util.save_figure(fig, f"{self.name}_filtered_inputs_for_BTSP", save=autosave)  # type: ignore[attr-defined]

        return fig, ax

    def _add_BTSP_to_mountain_plots(
        self,
        axes: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | plt.Axes,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        timeseries: bool = False,
        color: str | None = None,
    ):
        """Plot the rate map of the layer, ensuring no more than 20 columns
        are plotted.

        See FeedForwardLayer.plot_rate_map() for more information.

        Args:
            fig (mpl_figure.Figure): Figure object.
            axes (np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | plt.Axes): Axes object.
            t_start (float, optional): Start time of the plot. Defaults to None.
            t_end (float, optional): End time. Defaults to None.
            chosen_neurons (list, optional): List of neurons to plot. Defaults to "all".
            timeseries (bool, optional): Whether the plot is timeseries (map expected, otherwise).
            color (str, optional): Color of the BTSP markers. Defaults to None.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        if color is None:
            color = self.color or "C1"

        chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type]
        num_neurons = len(chosen_neurons)

        t, startid, _ = self.get_plotting_times(t_start=t_start, t_end=t_end)

        BTSP_events = np.asarray(self.history["BTSP_events"]) - startid
        BTSP_targets = self.history["BTSP_targets"]

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
                    sub_ax = axes
                    x_pos = t[event] / 60
                    line_sep = (sub_ax.get_ylim()[1] - 1) / num_neurons
                    y_pos = 1 + line_sep * i + line_sep * 0.7
                    pos = [x_pos, y_pos]
                else:
                    pos = self.Agent.history["pos"][event + startid]
                    if self.Agent.Environment.dimensionality == "1D":
                        sub_ax = axes
                        line_sep = (sub_ax.get_ylim()[1] - 1) / num_neurons
                        y_pos = 1 + line_sep * i + line_sep * 0.7
                        pos = pos + [y_pos]
                    else:
                        sub_ax = axes.ravel()[i]

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
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        mark_BTSP: bool = True,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]]:
        """Plot the rate map of the layer, ensuring no more than 20 columns
        are plotted.

        See FeedForwardLayer.plot_rate_map() for more information.

        Args:
            t_start (float, optional): Start time of the plot. Defaults to None.
            t_end (float, optional): End time. Defaults to None.
            chosen_neurons (list, optional): List of neurons to plot. Defaults to "all".
            mark_BTSP (bool, optional): Whether to include BTSP markers
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        fig, axes = super().plot_rate_map(
            t_start=t_start,
            t_end=t_end,
            chosen_neurons=chosen_neurons,
            autosave=False,
            **kwargs,
        )

        if mark_BTSP:
            self._add_BTSP_to_mountain_plots(
                axes,
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                timeseries=False,
            )

        util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

        return fig, axes

    def plot_rate_timeseries(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        xlim: tuple[float, float] | None = None,
        color: str | None = None,
        mark_BTSP: bool = True,
        autosave=None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]]]:
        """Plot the rate timeseries of the layer.

        See Neurons.plot_rate_timeseries() for more information.

        Args:
            t_start (float, optional): Start time of the plot. Defaults to None.
            t_end (float, optional): End time. Defaults to None.
            chosen_neurons (list, optional): List of neurons to plot. Defaults to "all".
            xlim (tuple[float, float], optional): The x limits of the plot. Defaults to None.
            color (str, optional): The color of the plot. Defaults to None.
            mark_BTSP (bool, optional): Whether to include BTSP markers
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        fig, ax = super().plot_rate_timeseries(
            t_start=t_start,
            t_end=t_end,
            chosen_neurons=chosen_neurons,
            xlim=xlim,
            color=color,
            autosave=False,
            **kwargs,
        )

        if xlim is not None:
            xlim = ax.get_xlim()

        if mark_BTSP:
            self._add_BTSP_to_mountain_plots(
                ax,
                t_start=t_start,
                t_end=t_end,
                chosen_neurons=chosen_neurons,
                timeseries=True,
                color=color,
            )

        if xlim is not None:
            ax.set_xlim(xlim)

        util.save_figure(fig, f"{self.name}_timeseries", save=autosave)  # type: ignore[attr-defined]

        return fig, ax

    def plot_BTSP_ramp(self, fig=None, ax=None, plot_events=True, autosave=None):
        """Plot the BTSP ramp of the layer.

        Args:
            fig (mpl_figure.Figure): Figure object. Defaults to None.
            ax (plt.Axes): Axes object. Defaults to None.
            plot_events (bool, optional): Whether to plot BTSP event markers. Defaults to True.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=[9, 5], sharex=True)
        elif ax.shape != (2,):
            raise ValueError("ax must have shape (2,).")

        if self.n > 1:
            raise NotImplementedError(
                "Plotting BTSP ramp only implemented for 1 neuron."
            )

        t = np.asarray(self.history["t"])
        BTSP_ramp = np.asarray(self.history["BTSP_ramp"])[:, 0]  # 1st neuron only

        ax[0].plot(t, BTSP_ramp, lw=1.2, color=self.color)
        ax[0].fill_between(
            t,
            np.zeros(len(t)),
            BTSP_ramp,
            lw=0,
            alpha=0.2,
            color=self.color,
        )
        ax[0].set_ylabel("Prop. of BTSP\nthreshold reached")
        ax[0].spines[["top", "right"]].set_visible(False)
        ax[0].axhline(1, ls="dashed")

        ax[1].plot(t, self.history["firingrate"], lw=1.2, color=self.color)
        ax[1].axhline(self.BTSP_induction_threshold, ls="dashed", color=self.color)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Firingrates")
        ax[1].spines[["top", "right"]].set_visible(False)

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
                ax[1].axvspan(
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
                ax[0].scatter(
                    t[np.asarray(BTSP_events)],
                    np.full(len(BTSP_events), y),
                    color=(self.color or "k"),
                    alpha=0.8,
                    marker=markers.MarkerStyle("x"),
                    s=10,
                )
        if BTSP_ramp.max() < 0:
            ax[1].legend()

        if fig is None:
            fig = ax[0].figure

        util.save_figure(fig, f"{self.name}_BTSP_ramp", save=autosave)  # type: ignore[attr-defined]

        return fig, ax


class NMDACurrent:
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
        return lambda x, deriv=False: rutils.activate(
            x, deriv=deriv, other_args=self.activation_params
        )

    @property
    def binding_params(self):
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
        return lambda x, deriv=False: rutils.activate(
            x, deriv=deriv, other_args=self.binding_params
        )

    def plot_function(
        self, param_type="binding", min_input_fr=-15, max_input_fr=15, fig=None, ax=None
    ):
        if param_type == "binding":
            function = self.binding_function
        elif param_type == "activation":
            function = self.activation_function
        else:
            raise ValueError(f"Unknown param type {param_type}")

        fig, ax = plot_util.plot_activation_function(
            function,
            min_input_fr=min_input_fr,
            max_input_fr=max_input_fr,
            fig=fig,
            ax=ax,
            color=self.color,
        )

        ax.set_title(f"{param_type.capitalize()} function")

        return fig, ax

    def get_decay(self, decay_type: str = "activation", dt: float | None = None):
        """Get the decay factor for the NMDA receptor activation or desensitization.

        Args:
            decay_type (str, optional): Type of decay. Defaults to "activation".
            dt (float, optional): Time step. Defaults to None.

        Returns:
            float: Decay factor.
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
        """Get the state of the NMDA current.

        Args:
            evaluate_at (str, optional): Whether to evaluate the state at the last time
                step or the current time step. Defaults to "last".
            return_all (bool, optional): Whether to return all state variables.
                Defaults to False.
            dt (float, optional): Time step. Defaults to None.
            **kwargs: Keyword arguments ignored

        Returns:
            np.ndarray: State of the NMDA current.
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
        """Update the NMDA current."""

        (
            current,
            receptor_binding,
            receptor_activation,
            receptor_desensitization,
        ) = self.get_state(evaluate_at="last", return_all=True)

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

    def plot_timeseries(
        self,
        t_start: float | None = None,
        title: str | None = None,
        datatypes: list[str] | str = "current",
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        fig: mpl_figure.Figure | None = None,
        axes: plt.Axes | np.ndarray[plt.Axes, np.dtype[np.int64]] | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, np.ndarray[plt.Axes, np.dtype[np.int64]]]:
        """Plot the current, and optionally the receptor binding, activation and
        densensitization time series.

        Args:
            input_layer_name (str): Name of the input layer to plot.
            t_start (float, optional): Start time of the plot. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            datatypes (list[str] | str, optional): Type of data to plot. Defaults to "current".
            chosen_neurons (str | int | list[int] | np.ndarray[tuple[int], np.dtype[np.int64]], optional):
                Neurons to plot. Defaults to "all".
            fig (mpl_figure.Figure, optional): Figure to plot on. Defaults to None.
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            autosave (bool, optional): Whether to save the plot. Defaults to None.
            **kwargs: Keyword arguments passed to the plot function.

        Returns:
            fig, axes: Figure and axes of the plot.
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

        if fig is None or axes is None:
            n = len(chosen_neurons)
            height = max([1, min(n / 12.0 + 5 / 3, 8)]) * len(datatypes)
            fig, axes = plt.subplots(
                len(datatypes), 1, sharex=True, sharey=False, figsize=[6, height]
            )  # type: ignore[assignment]

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(-1, 1)

        if len(axes) != len(datatypes):
            raise ValueError(
                f"Number of axes ({len(axes)}) does not match number of datatypes "
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
                fig=fig,
                ax=axes[d, 0],
                trace_name=datatype,
                chosen_neurons=chosen_neurons,
                autosave=False,
                **kwargs,
            )

            axes[d, 0].set_ylabel(datatype.capitalize().replace("_", " "))
            if d != len(datatypes) - 1:
                axes[d, 0].set_xlabel("")

        fig.suptitle(title, y=0.90)

        util.save_figure(fig, f"{self.name}_NMDA_current_traces", save=autosave)  # type: ignore[attr-defined]

        return fig, axes


class NMDALayer(BTSPLayer):
    """This trained class defines a population of neurons that tune their activity
    through Hebbian learning with BTSP and NMDA receptors.
    This class is a subclass of Neurons() and inherits its properties/plotting
    functions.

    Must be initialised with an Agent, and a "params" dictionary, including input
    layers.

    List of functions:
        • get_state()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()
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
        """Initialise HebbianLayer(), takes as input a parameter
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

        self._check_BTSP_plateau_length()
        self._add_NMDA_current()

        self.BTSP_ramp = np.zeros(self.n).astype(float)  # type: ignore[attr-defined]
        self.history["BTSP_ramp"] = list()

    def _check_BTSP_plateau_length(self):
        num_steps = self.BTSP_plateau_length / self.Agent.dt
        if not np.isclose(num_steps, int(num_steps)):
            low_plateau_length = np.floor(num_steps) * self.Agent.dt
            high_plateau_length = np.ceil(num_steps) * self.Agent.dt
            raise ValueError(
                f"BTSP_plateau_length must be a multiple of the Agent's dt ({self.Agent.dt}). "
                f"Try {low_plateau_length} or {high_plateau_length}."
            )

    def _add_NMDA_current(self):
        """Set the NMDA intermediate layer."""

        self.NMDACurrent = NMDACurrent(
            self,
            name="NMDACurrent",
            NMDA_activation_threshold=self.NMDA_activation_threshold,  # type: ignore[attr-defined]
            color=self.color,  # type: ignore[attr-defined]
            save_history=self.save_history,  # type: ignore[attr-defined]
        )

        self.add_input(self.NMDACurrent, w=np.eye(self.n), recurrent=True)  # type: ignore[attr-defined]
        self.add_input_layers_with_no_learning(self.NMDACurrent.name)

        return self.NMDACurrent

    def save_to_history(self):
        """Save the current state of the layer to the history, including the
        loss, if applicable.
        """

        super().save_to_history()

        self.history["BTSP_ramp"].append(self.BTSP_ramp.tolist())

        return

    def get_incoming_firingrates(self, evaluate_at="last", **kwargs):
        """Returns the firing rates coming into each neuron. By default this layer uses
        the last saved firingrate from its input layers. Alternatively evaluate_at and
        kwargs can be set to be anything else which will just be passed to the input
        layer for evaluation.

        Once the firing rate of the input layers is established these are multiplied by
        a normalized weight matrix of 1s.

        NOTE: This means that pre-synaptic plasticity is ignored, and all input neurons
        are equipotent. May have to be rethought.

        Args:
            evaluate_at (str, optional). Defaults to 'last'.
        Returns:
            firingrate: array of firing rates
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
            inputlayer["I_temp"] = I
            V += np.matmul(w_ones, I)

        return V

    def update(self):
        self.update_filtered_inputs(
            self.learning_filter_tau,
            self.learning_trend_tau,
            filter_key="filtered_inputs_for_learning",
        )
        self.update_filtered_inputs(
            self.BTSP_filter_tau,
            self.BTSP_trend_tau,
            filter_key="filtered_inputs_for_BTSP",
        )

        self.NMDACurrent.update()
        super().update()

        above_threshold = self.firingrate > self.BTSP_induction_threshold  # type: ignore[attr-defined]
        self.BTSP_ramp[~above_threshold] = 0
        self.BTSP_ramp[above_threshold] += self.Agent.dt / self.BTSP_plateau_length  # type: ignore[attr-defined]

        BTSP_targets = np.where(np.isclose(self.BTSP_ramp, 1))[
            0
        ]  # only once per plateau

        if self.BTSP_learn and len(BTSP_targets):
            if self.single_BTSP:
                keep_BTSP_targets = np.asarray(
                    [targ for targ in BTSP_targets if self.num_BTSP_to_date[targ] == 0]
                )

            else:
                keep_BTSP_targets = list()
                for targ in BTSP_targets:
                    closest_recent_BTSP_event = self.last_BTSP_pos[targ]
                    if (
                        self.BTSP_distance_prop is None
                        or closest_recent_BTSP_event is None
                    ):
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

            if len(keep_BTSP_targets) == 0:
                return
            BTSP_targets = keep_BTSP_targets

            lr = np.full(self.n, self.lr)  # type: ignore[attr-defined]
            lr[np.asarray(BTSP_targets)] *= self.BTSP_lr_fact  # type: ignore[attr-defined]

            self.num_BTSP_to_date[np.asarray(BTSP_targets)] += +1
            self.last_BTSP_step[np.asarray(BTSP_targets)] = self.num_steps_total - 1
            for targ in BTSP_targets:
                self.last_BTSP_pos[targ] = self.Agent.pos

            self.update_weights(filter_key="filtered_inputs_for_BTSP", lr=lr)
            self.history["BTSP_events"].append(
                self.num_steps_total - 1
            )  # recorded after update
            self.history["BTSP_targets"].append(BTSP_targets.tolist())

        return

    def plot_BTSP_ramp(self, fig=None, ax=None, autosave=None):
        """Plot the BTSP ramp of the layer.

        Args:
            fig (mpl_figure.Figure): Figure object. Defaults to None.
            ax (plt.Axes): Axes object. Defaults to None.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            mpl_figure.Figure: Figure object.
            plt.Axes: Axes object.
        """

        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=[9, 7], sharex=True)
        elif ax.shape != (3,):
            raise ValueError("ax must have shape (3,).")

        super().plot_BTSP_ramp(fig=fig, ax=ax[:2], autosave=False)

        ax[2].plot(
            self.history["t"],
            self.inputs["NMDACurrent"]["layer"].history["current"],
            lw=1.2,
            color=self.color,
        )
        ax[1].set_xlabel("")
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("CA1 NMDA current")
        ax[2].spines[["top", "right"]].set_visible(False)

        if fig is None:
            fig = ax[0].figure

        util.save_figure(fig, f"{self.name}_BTSP_ramp", save=autosave)  # type: ignore[attr-defined]

        return fig, ax
