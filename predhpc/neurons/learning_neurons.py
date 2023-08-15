import copy
from typing import TYPE_CHECKING, Any, Sequence
import warnings

from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import markers
from matplotlib import figure as mpl_figure
import numpy as np

from ratinabox.Neurons import Neurons, FeedForwardLayer  # type: ignore[import]
from ratinabox import utils as rutils  # type: ignore[import]

from predhpc import util, plot_util

if TYPE_CHECKING:
    import ratinabox  # type: ignore[import]


BETA_1_WIDTH_X = 2 * np.log(19)  # so that the sigmoid beta value is 1

STANDARD_SIGMOID_PARAMS = {
    "activation": "sigmoid",
    "min_fr": 0.0,
    "max_fr": 1.0,
    "width_x": BETA_1_WIDTH_X,
    "mid_x": 0,
}


class LearnLayer(FeedForwardLayer, util.ParamsManagerMixin):
    """This trained class defines a population of neurons that tune their
    activity through Hebbian learning.
    This class is a subclass of Neurons() and inherits its properties/plotting functions.

    Must be initialised with an Agent, and a "params" dictionary, including input
    layers.

    List of functions:
        • get_state()
        • set_freeze()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()
    """

    default_params = {
        "n": 10,
        "activation_params": {"activation": "sigmoid"},
        "name": "LearnLayer",
        "lr": 1e-4,  # learning rate
        "biases": None,
        "use_targets": False,
        "init_weights_zero": False,  # whether to initialize weights to 0
        "w_init_loc": 0,  # mean of the initial weights
        "w_init_scale": 1,  # scale of the initial weights
        "use_targets": False,  # whether to use targets
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
        fig: mpl_figure.Figure | None = None,
        axes: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | None = None,
        autosave: bool | None = None,
    ):
        """Plot the rate maps of the layer across learning.

        Args:
            num_maps (int, optional): Number of maps to plot. Defaults to 3.
            prop_each (float, optional): Proportion of the learning period to plot
                for each map. Defaults to 0.4.
            normalize_together (bool, optional): Whether to normalize the maps
                together. Defaults to True.
            title (str, optional): Title of the plot. Defaults to None.
            fig (mpl_figure.Figure, optional): Figure object. Defaults to None.
            axes (np.ndarray, optional): Axes object. Defaults to None.
            autosave (bool, optional): Whether to save the figure. Defaults to None.

        Returns:
            fig, axes: Figure and axes objects.
        """

        if fig is None or axes is None:
            fig, axes = plt.subplots(ncols=num_maps, figsize=(num_maps * 3, 3))

        flat_axes = np.asarray(axes).reshape(-1)

        if len(flat_axes) != num_maps:
            raise ValueError(
                f"Number of axes ({len(flat_axes)}) does not match number of "
                f"maps ({num_maps})."
            )

        t = self.history["t"]
        n_pts = int(prop_each * len(t))
        start_pts = [int(st) for st in np.linspace(0, len(t) - n_pts, num_maps)]

        subplots = []
        for i, start in enumerate(start_pts):
            t_start = t[start]
            stop = min([len(t) - 1, start + n_pts])
            t_end = t[stop]
            flat_axes[i].set_title(f"From {t_start / 60:.2f} to {t_end / 60:.2f} min.")

            _, map_axes = self.plot_rate_map(
                fig=fig,
                ax=flat_axes[i],
                t_start=t_start,
                t_end=t_end,
                method="history",
                colorbar=False,
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

        if fig is None or ax is None:
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

        util.save_figure(fig, f"{self.name}_firing_rate_histogram", save=autosave)  # type: ignore[attr-defined]

        return fig, ax


class HebbianLayer(LearnLayer):
    """This trained class defines a population of neurons that tune their activity through Hebbian learning.
    This class is a subclass of Neurons() and inherits its properties/plotting functions.

    Must be initialised with an Agent, and a "params" dictionary, including input layers.

    List of functions:
        • get_state()
        • set_freeze()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()

    """

    default_params = {
        "n": 10,
        "activation_params": {"activation": "sigmoid"},
        "name": "HebbianLayer",
        "lr": 1e-4,  # learning rate
        "biases": None,
        "normalize_weights": False,
        "apply_Ojas_rule": False,
        "use_targets": False,
        "init_weights_zero": False,  # whether to initialize weights to 0
        "w_init_scale": 1,  # scale of the initial weights
        "filter_tau": None,
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

        self.set_learn()

        if self.apply_Ojas_rule and self.normalize_weights:  # type: ignore[attr-defined]
            raise ValueError("Can only set 'oja' or 'norm' to True, not both.")

        return

    @property
    def learn(self) -> bool:
        """Property for the learn attribute.

        Returns:
            bool: Whether the layer is learning or not.
        """

        return self._learn

    def set_freeze(self):
        """Set the layer to not learn."""

        self._learn = False
        return

    def set_learn(self):
        """Set the layer to learn."""

        self._learn = True
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

    @property
    def input_layers_with_no_learning(self) -> list[str]:
        """Returns a list of input layer names that are not learning."""
        if hasattr(self, "_input_layers_with_no_learning"):
            return self._input_layers_with_no_learning
        else:
            return list()

    @input_layers_with_no_learning.setter
    def input_layers_with_no_learning(self, input_layers) -> None:
        self._input_layers_with_no_learning = input_layers

    def add_input(self, input_layer: Neurons, **kwargs):
        super().add_input(input_layer, **kwargs)

        name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]
        self.inputs[name_in]["filtered_inputs"] = np.zeros(n_in)

        if "filtered_inputs" not in self.history.keys():
            self.history["filtered_inputs"] = dict()
        self.history["filtered_inputs"][name_in] = list()

    def save_to_history(self):
        """Save the current state of the layer to the history, including the
        loss, if applicable.
        """

        super().save_to_history()

        for name, input_layer in self.inputs.items():
            self.history["filtered_inputs"][name].append(
                input_layer["filtered_inputs"].tolist()
            )

    def update_filtered_inputs(
        self, filter_tau: float | None = None, filter_key: str = "filtered_inputs"
    ):
        """Update the filtered inputs of the layer."""

        filter_tau = self.get_filter_tau(filter_tau)
        effective_tau = filter_tau / self.Agent.dt

        for input_layer in self.inputs.values():
            X_t = input_layer[filter_key]
            I_t1 = input_layer["layer"].firingrate
            d_filt = (I_t1 - X_t) / effective_tau
            input_layer[filter_key] += d_filt

        return

    def update_weights(
        self,
        filter_key: str = "filtered_inputs",
        O: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    ):
        """Update the weights of the layer.

        Args:
            filter_key (str, optional): Key of the input to use for the update.
                Defaults to "filtered_inputs".
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
        lr = self.lr  # type: ignore[attr-defined]

        if self.apply_Ojas_rule:  # type: ignore[attr-defined]
            util.perform_Oja_update_(Is, ws, O, lr=lr, b=b)
        elif self.normalize_weights:  # type: ignore[attr-defined]
            util.perform_normalized_Hebbian_update_(Is, ws, O, lr=lr, b=b)
        else:
            util.perform_Hebbian_update_(Is, ws, O, lr=lr, b=b)

    def update(self):
        """Update the layer, i.e. calculate the new firing rates and update the
        weights and biases, if applicable."""

        self.update_filtered_inputs(self.filter_tau, filter_key="filtered_inputs")  # type: ignore[attr-defined]
        super().update()

        if self.learn:
            self.update_weights(filter_key="filtered_inputs")

        return

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
                Defaults to "filtered_inputs".
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

        if fig is None or ax is None:
            n = unfiltered.shape[1]
            height = max([1, min(n / 12.0 + 5 / 3, 8)])
            fig, ax = plt.subplots(figsize=[6, height])

        color = self.color  # type: ignore[attr-defined]

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
            title = "Filtered inputs"
        ax.set_title(title)

        util.save_figure(fig, f"{self.name}_filtered_inputs", save=autosave)  # type: ignore[attr-defined]

        return fig, ax, t


class BTSPLayer(HebbianLayer):
    """This trained class defines a population of neurons that tune their activity
    through Hebbian learning with BTSP.
    This class is a subclass of Neurons() and inherits its properties/plotting
    functions.

    Must be initialised with an Agent, and a "params" dictionary, including input
    layers.

    List of functions:
        • get_state()
        • set_freeze()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()
    """

    default_params = {
        "n": 10,
        "name": "BTSPLayer",
        "btsp_tau": 3,
        "btsp_fr": 1,
        "btsp_single": False,
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

        self.history["btsp_events"] = list()
        self.history["btsp_targets"] = list()

        self.btsp_to_date = np.zeros(self.n)

        if self.use_targets:  # type: ignore[attr-defined]
            raise ValueError("BTSPLayer does not support targets.")

        self.set_btsp_learn()

        return

    @property
    def btsp_learn(self) -> bool:
        """Returns the BTSP learning state of the layer.

        Returns:
            bool: BTSP learning state.
        """

        return self._btsp_learn

    def set_btsp_freeze(self):
        """Set the layer to not learn using BTSP."""

        self._btsp_learn = False
        return

    def set_btsp_learn(self):
        """Set the layer to learn using BTSP."""

        self._btsp_learn = True
        return

    def add_input(self, input_layer: Neurons, **kwargs):
        """Adds an input layer to the layer by calling super().add_input(), and
        initialises the BTSP filtered inputs."""
        super().add_input(input_layer, **kwargs)

        name_in, n_in = input_layer.name, input_layer.n  # type: ignore[attr-defined]
        self.inputs[name_in]["btsp_filtered_inputs"] = np.zeros(n_in)

        if "btsp_filtered_inputs" not in self.history.keys():
            self.history["btsp_filtered_inputs"] = dict()
        self.history["btsp_filtered_inputs"][name_in] = list()

    def save_to_history(self):
        """Save the current state of the layer to the history, including the
        loss, if applicable.
        """

        super().save_to_history()

        for name, input_layer in self.inputs.items():
            self.history["btsp_filtered_inputs"][name].append(
                input_layer["btsp_filtered_inputs"].tolist()
            )

    def update(
        self, btsp_targets: list | np.ndarray[tuple[int], np.dtype[np.int64]] = list()
    ):
        """Update the layer, i.e. calculate the new firing rates and update the
        weights and biases, if applicable.

        Args:
            btsp_targets (list or 1D array, optional): List of BTSP targets.
                Defaults to [].
        """

        filter_tau, btsp_tau = self.filter_tau, self.btsp_tau  # type: ignore[attr-defined]
        self.update_filtered_inputs(filter_tau, filter_key="filtered_inputs")
        self.update_filtered_inputs(btsp_tau, filter_key="btsp_filtered_inputs")

        super().update()

        if self.btsp_learn and btsp_targets is not None and len(btsp_targets):
            if self.btsp_single:
                keep_btsp_targets = np.asarray(
                    [targ for targ in btsp_targets if self.btsp_to_date[targ] == 0]
                )
                if len(keep_btsp_targets) == 0:
                    return
                btsp_targets = keep_btsp_targets

            n, btsp_fr = self.n, self.btsp_fr  # type: ignore[attr-defined]
            O = np.zeros(n)
            O[np.asarray(btsp_targets)] = btsp_fr

            self.btsp_to_date[np.asarray(btsp_targets)] += +1

            self.update_weights(filter_key="btsp_filtered_inputs", O=O)
            self.history["btsp_events"].append(
                self.num_steps_total - 1
            )  # recorded after update
            self.history["btsp_targets"].append(btsp_targets)

        return

    def plot_btsp_filtered(
        self,
        input_layer_name: str,
        t_start: float | None = None,
        title: str | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the filtered inputs of the layer.

        Args:
            input_layer_name (str): Name of the input layer to plot.
            t_start (float, optional): Start time of the plot. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            **kwargs: Keyword arguments passed to the plot function.

        Returns:
            fig, ax: Figure and axes of the plot.
        """

        if title is None:
            title = "BTSP filtered inputs"

        fig, ax, t = super().plot_filtered(
            input_layer_name,
            filter_key="btsp_filtered_inputs",
            t_start=t_start,
            title=title,
            autosave=False,
            **kwargs,
        )

        _, startid, _ = self.get_plotting_times(t_start=t_start)

        btsp_events = np.asarray(self.history["btsp_events"]) - startid
        btsp_targets = np.asarray(self.history["btsp_targets"])
        btsp_mask = (btsp_events >= 0) & (btsp_events < len(t))

        miny, maxy = ax.get_ylim()

        flat_btsp_events = list()  # type: list[int]
        flat_btsp_targets = list()  # type: list[int]
        for ev, targ in zip(btsp_events[btsp_mask], btsp_targets[btsp_mask]):
            flat_btsp_targets.extend(targ)
            flat_btsp_events.extend([ev for _ in range(len(targ))])

        flat_btsp_events_arr = np.array(flat_btsp_events)
        flat_btsp_targets_arr = np.array(flat_btsp_targets)  # type: ignore[arg-type]

        height_diff = 0.01 * (maxy - miny)
        heights = maxy - height_diff * (flat_btsp_targets_arr + 1)

        if len(flat_btsp_events_arr):
            ax.scatter(
                t[flat_btsp_events_arr],
                heights,
                color="k",
                alpha=0.8,
                marker=markers.MarkerStyle("x"),
                s=10,
            )

        util.save_figure(fig, f"{self.name}_btsp_filtered_inputs", save=autosave)  # type: ignore[attr-defined]

        return fig, ax

    def plot_rate_map(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        chosen_neurons: str
        | int
        | list[int]
        | np.ndarray[tuple[int], np.dtype[np.int64]] = "all",
        mark_btsp: bool = True,
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
            mark_btsp (bool, optional): Whether to include BTSP markers
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

        if mark_btsp:
            chosen_neurons = self.return_list_of_neurons(chosen_neurons=chosen_neurons)  # type: ignore[arg-type]
            t, startid, _ = self.get_plotting_times(t_start=t_start, t_end=t_end)

            btsp_events = np.asarray(self.history["btsp_events"]) - startid
            btsp_targets = np.asarray(self.history["btsp_targets"])

            for event, target in zip(btsp_events, btsp_targets):
                if event >= len(t):
                    continue
                elif event < 0:
                    alpha = 0.4
                else:
                    alpha = 0.8

                if target not in chosen_neurons:
                    continue

                i = chosen_neurons.index(target)

                pos = self.Agent.history["pos"][event + startid]
                if self.Agent.Environment.dimensionality == "1D":
                    sub_ax = axes
                    pos = np.asarray(pos + [sub_ax.get_ylim()[0]])
                else:
                    sub_ax = axes.ravel()[
                        i
                    ]  ### WILL THIS WORK? ARE THERE MULTIPLE AXES?
                sub_ax.scatter(
                    *pos,
                    color="k",
                    alpha=alpha,
                    marker=markers.MarkerStyle("x"),
                    s=10,
                )

        util.save_figure(fig, f"{self.name}_ratemaps", save=autosave)  # type: ignore[attr-defined]

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
        • set_freeze()
        • set_learn()
        • add_input()
        • update()
        • plot_rate_map()
        • plot_loss()
    """

    default_params = {
        "n": 10,
        "name": "NMDALayer",
        "NMDA_activation_threshold": 0.8,
        "BTSP_induction_threshold": 0.8,
        "BTSP_plateau_length": 0.1,  # seconds
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

        self._add_NMDA_current()

        self.ramp_to_btsp = np.zeros(self.n).astype(float)  # type: ignore[attr-defined]

        return

    def _add_NMDA_current(self):
        """Set the NMDA intermediate layer."""

        self.NMDACurrent = NMDACurrent(
            self,
            name="NMDACurrent",
            NMDA_activation_threshold=self.NMDA_activation_threshold,  # type: ignore[attr-defined]
            color=self.color,  # type: ignore[attr-defined]
            save_history=self.save_history,  # type: ignore[attr-defined]
        )

        self.add_input(self.NMDACurrent, w=np.eye(self.n))  # type: ignore[attr-defined]

        return self.NMDACurrent

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
            else:  # kick can down the road let input layer decide how to evaluate the firingrate. this is core to feedforward layer as this recursive call will backprop through the upstraem layers until it reaches a "core" (e.g. place cells) layer which will then evaluate the firingrate.
                I = inputlayer["layer"].get_state(evaluate_at, **kwargs)
            inputlayer["I_temp"] = I
            V += np.matmul(w_ones, I)

        return V

    def update(self):
        filter_tau, btsp_tau = self.filter_tau, self.btsp_tau  # type: ignore[attr-defined]
        self.update_filtered_inputs(filter_tau, filter_key="filtered_inputs")
        self.update_filtered_inputs(btsp_tau, filter_key="btsp_filtered_inputs")

        self.NMDACurrent.update()
        super().update()

        above_threshold = self.firingrate > self.BTSP_induction_threshold  # type: ignore[attr-defined]
        self.ramp_to_btsp[~above_threshold] = 0
        self.ramp_to_btsp[above_threshold] += self.Agent.dt / self.BTSP_plateau_length  # type: ignore[attr-defined]

        btsp_targets = np.where(self.ramp_to_btsp >= 1)[0]
        if self.btsp_learn and len(btsp_targets):
            if self.btsp_single:
                keep_btsp_targets = np.asarray(
                    [targ for targ in btsp_targets if self.btsp_to_date[targ] == 0]
                )
                if len(keep_btsp_targets) == 0:
                    return
                btsp_targets = keep_btsp_targets

            n, btsp_fr = self.n, self.btsp_fr  # type: ignore[attr-defined]
            O = np.zeros(n)
            O[np.asarray(btsp_targets)] = btsp_fr

            self.btsp_to_date[np.asarray(btsp_targets)] += +1

            self.update_weights(filter_key="btsp_filtered_inputs")
            self.history["btsp_events"].append(
                self.num_steps_total - 1
            )  # recorded after update
            self.history["btsp_targets"].append(btsp_targets)

        return


class NMDACurrent:
    def __init__(
        self,
        InputLayer,
        name="NMDACurrent",
        NMDA_activation_decay_tau=0.1,  # seconds
        NMDA_desensitization_decay_tau=0.3,  # seconds
        NMDA_activation_threshold=0.8,  # firing rate
        max_current=3.0,
        start_desensitization=0.1,
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
        self.color = color

        self.NMDA_receptor_binding = np.zeros(self.n)
        self.NMDA_receptor_activation = np.zeros(self.n)
        self.NMDA_receptor_desensitization = (
            np.zeros(self.n) * self.start_desensitization
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
    def binding_params(self):
        self._binding_params = {
            "activation": "sigmoid",
            "min_fr": 0.0,
            "max_fr": 1.0,
            "width_x": 20,
        }
        self._binding_params["mid_x"] = self._binding_params["width_x"] / 1.5

        return self._binding_params

    def plot_function(
        self, param_type="binding", min_fr=-10, max_fr=10, fig=None, ax=None
    ):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 2))

        if param_type == "binding":
            params = self.binding_params
        elif param_type == "activation":
            params = self.activation_params
        else:
            raise ValueError(f"Unknown param type {param_type}")

        x = np.linspace(min_fr, max_fr, 1000)
        y = rutils.activate(x, other_args=params)

        ax.plot(x, y, color=self.color, lw=1.5)
        ax.axhline(0, color="k", lw=1, ls="dashed")
        ax.axvline(0, color="k", lw=1, ls="dashed")

        ax.set_xlabel("Input firing rate")
        ax.set_ylabel("Output value")
        ax.set_title(f"{param_type.capitalize()} function")

        ax.spines[["right", "top"]].set_visible(False)

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
        self, evaluate_at="last", return_all: bool = False, dt: float | None = None
    ):
        """Get the state of the NMDA current.

        Args:
            evaluate_at (str, optional): Whether to evaluate the state at the last time
                step or the current time step. Defaults to "last".
            return_all (bool, optional): Whether to return all state variables.
                Defaults to False.
            dt (float, optional): Time step. Defaults to None.

        Returns:
            np.ndarray: State of the NMDA current.
        """

        receptor_binding = rutils.activate(
            self.InputLayer.get_incoming_firingrates(), other_args=self.binding_params
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
            neuron_activity_gate = rutils.activate(
                self.InputLayer.firingrate, other_args=self.activation_params
            )

            receptor_activation += effective_binding * neuron_activity_gate
        else:
            receptor_activation = receptor_binding

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
            "current",
            "receptor_binding",
            "receptor_activation",
            "receptor_desensitization",
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
