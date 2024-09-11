import copy

import numpy as np

from predhpc.neurons import riab_neurons


class SimpleValueNeuron(riab_neurons.PlaceCells):
    """
    SimpleValueNeuron()

    Class extending riab_neurons.PlaceCells. Defines a simple value neuron layer.

    Must be initialised with an Agent. A parameters dictionary can also be passed at
    initialisation.

    default_params = {
        "n": 1,
        "peak": [0.5, 0.5],
        "widths": 0.3,
        "description": "gaussian",
        "wall_geometry": "geodesic",
        "max_fr": 10,
    }

    See riab_neurons.PlaceCells for properties.

    List of methods (in addition to riab_neurons.FeedForwardLayer methods):
        • self.get_local_gradient()
        • self.plot_local_gradient()
    """

    default_params = {
        "n": 1,
        "peak": [0.5, 0.5],
        "widths": 0.3,
        "description": "gaussian",
        "wall_geometry": "geodesic",
        "max_fr": 10,
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Agent, params=dict()):
        """
        SimpleValueNeuron(Agent)

        Initialise a simple value neuron layer.

        Attributes:
        - Agent (agent.ResetableAgent): Associated agent.
        - peak (2D np.ndarray): Peak location, with shape (number of neurons, 2).

        Args:
        - Agent (agent.ResetableAgent): Associated agent.
        - params (dict, optional): Neuron layer parameters. Default is dict().
        """

        self.Agent = Agent
        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.params["place_cell_centres"] = np.asarray(self.params["peak"]).reshape(
            -1, 2
        )

        super().__init__(Agent, self.params)

        self.peak = np.asarray(self.params["peak"]).reshape(-1, 2)

    def get_local_gradient(
        self, evaluate_at="agent", abs_shift=1e-3, p=2, thresh_gradV=None, pos=None
    ):
        """
        self.get_local_gradient()

        Get the local gradient of the value function at a given position.

        Args:
        - evaluate_at (str, optional): Where to evaluate the gradient.
            Default is "agent".
        - abs_shift (float, optional): Shift to use to compute local gradient.
            Default is 1e-3.
        - p (float, optional): Power to use for gradient scaling. Default is 2.
        - thresh_gradV (float, optional): Threshold for returning a gradient norm.
            Default is None.
        - pos (1D np.ndarray, optional): Position at which to evaluate the gradient,
            required if and only if evaluate_at is not "agent" or "all".
            Default is None.

        Returns:
        - gradV (np.ndarray): Local gradient of the value function.
        """

        if evaluate_at == "agent":
            if pos is not None:
                raise RuntimeError("If `evaluate_at` is `agent`, `pos` must be None.")
            pos = self.Agent.pos
        elif evaluate_at == "all":
            raise NotImplementedError("Cannot get gradient for full environment.")
        elif pos is None:
            raise RuntimeError("If `evaluate_at` is not `agent`, `pos` cannot be None.")

        pos = np.asarray(pos).reshape(2)

        abs_shift = np.absolute(abs_shift)

        V = self.get_state(evaluate_at="pos", pos=pos)[0]

        if V <= 0.05 * self.max_fr:
            return None
        else:
            V_dxs, V_dys = list(), list()
            for x_shift in [-abs_shift, abs_shift]:
                V_dxs.append(
                    self.get_state(
                        evaluate_at="pos", pos=pos + np.asarray([x_shift, 0])
                    )[0][0]
                )
            for y_shift in [-abs_shift, abs_shift]:
                V_dys.append(
                    self.get_state(
                        evaluate_at="pos", pos=pos + np.asarray([0, y_shift])
                    )[0][0]
                )

            gradV = np.asarray([V_dxs[1] - V_dxs[0], V_dys[1] - V_dys[0]]) / 2
            norm = np.linalg.norm(gradV)

            if np.isclose(norm, 0):
                gradV *= 0
            else:
                prog_norm = ((self.max_fr - V) / self.max_fr) ** p
                gradV = gradV / norm * prog_norm

            end_norm = np.sqrt(np.sum(gradV**2))

            if thresh_gradV is not None:
                end_norm = np.sqrt(np.sum(gradV**2))
                if end_norm < thresh_gradV:
                    return None

            return gradV

    def plot_local_gradient(self, evaluate_at="agent", pos=None, ax=None, **kwargs):
        """
        self.plot_local_gradient()

        Plot the local gradient of the value function at a given position.

        Args:
        - evaluate_at (str, optional): Where to evaluate the gradient.
            Default is "agent".
        - pos (1D np.ndarray, optional): Position at which to evaluate the gradient,
            required if and only if evaluate_at is not "agent" or "all".
            Default is None.
        - ax (np.ndarray or plt.Axes, optional): Subplot or array of subplots to plot
            on (one per plotted ROI, if environment is 2D). Default is None.

        Keyword args:
        - **kwargs: Keyword arguments to pass to self.get_local_gradient().

        Returns:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots
           (one per plotted ROI, if environment is 2D).
        """

        gradV = self.get_local_gradient(evaluate_at="pos", pos=pos, **kwargs)

        if evaluate_at == "agent":
            pos = self.Agent.pos

        pos = np.asarray(pos).reshape(2)

        xs = [pos[0], pos[0] + gradV[0]]
        ys = [pos[1], pos[1] + gradV[1]]

        ax_out = self.plot_rate_map(ax=ax, no_legend=True)

        if ax is None:
            ax = ax_out

        for sub_ax in np.asarray(ax).ravel():
            sub_ax.scatter(*pos, color="red", marker=".", s=20, zorder=12)
            sub_ax.plot(
                xs,
                ys,
                color="red",
                ls="dotted",
                zorder=12,
            )

        return ax
