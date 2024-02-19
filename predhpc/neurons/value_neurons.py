import copy

import numpy as np

from ratinabox import PlaceCells


class SimpleValueNeuron(PlaceCells):

    default_params = {
        "n": 1,
        "peak": [0.5, 0.5],
        "widths": 0.3,
        "description": "gaussian",
        "wall_geometry": "geodesic",
        "max_fr": 10,
    }

    def __init__(self, Agent, params=dict()):

        self.Agent = Agent
        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.params["place_cell_centres"] = np.asarray(self.params["peak"]).reshape(
            -1, 2
        )

        super().__init__(Agent, self.params)

        self.peak = np.asarray(self.params["peak"]).reshape(
            -1, 2
        )

    def get_local_gradient(self, evaluate_at="agent", abs_shift=1e-3, p=2, thresh_gradV=None, **kwargs):
        if evaluate_at == "agent":
            pos = self.Agent.pos
        elif evaluate_at == "all":
            raise NotImplementedError("Cannot get gradient for full environment.")
        else:
            if "pos" not in kwargs.keys():
                raise RuntimeError(
                    "If `evaluate_at` is not `agent`, must provide a position."
                )
            pos = kwargs["pos"]

        pos = np.asarray(pos).reshape(2)

        abs_shift = np.absolute(abs_shift)

        V = self.get_state(evaluate_at="pos", pos=pos)[0]

        if V <= 0.05 * self.max_fr:
            return None
        else:
            V_dxs, V_dys = list(), list()
            for x_shift in [-abs_shift, abs_shift]:
                V_dxs.append(
                    self.get_state(evaluate_at="pos", pos=pos + np.array([x_shift, 0]))[
                        0
                    ][0]
                )
            for y_shift in [-abs_shift, abs_shift]:
                V_dys.append(
                    self.get_state(evaluate_at="pos", pos=pos + np.array([0, y_shift]))[
                        0
                    ][0]
                )

            gradV = np.array([V_dxs[1] - V_dxs[0], V_dys[1] - V_dys[0]]) / 2
            norm = np.linalg.norm(gradV)

            if np.isclose(norm, 0):
                gradV *= 0
            else:
                prog_norm = ((self.max_fr - V) / self.max_fr) ** p
                gradV = gradV / norm * prog_norm

            end_norm = np.sqrt(np.sum(gradV ** 2))

            if thresh_gradV is not None:
                end_norm = np.sqrt(np.sum(gradV ** 2))
                if end_norm < thresh_gradV:
                    return None

            return gradV

    def plot_local_gradient(
        self, evaluate_at="agent", pos=None, fig=None, ax=None, **kwargs
    ):

        if evaluate_at == "agent":
            pos = self.Agent.pos
        elif evaluate_at == "all":
            raise NotImplementedError("Cannot get gradient for full environment.")
        elif pos is None:
            raise RuntimeError(
                "If `evaluate_at` is not `agent`, must provide a position."
            )
        pos = np.array(pos).reshape(2)

        gradV = self.get_local_gradient(evaluate_at="pos", pos=pos, **kwargs)

        xs = [pos[0], pos[0] + gradV[0]]
        ys = [pos[1], pos[1] + gradV[1]]

        fig, ax = self.plot_rate_map(fig=fig, ax=ax, no_legend=True)

        ax[0].scatter(*pos, color="red", marker=".", s=20, zorder=12)
        ax[0].plot(
            xs,
            ys,
            color="red",
            ls="dotted",
            zorder=12,
        )

        return fig, ax
