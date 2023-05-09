import copy
import warnings
import pprint

from ratinabox import Environment
from ratinabox import utils as rutils

from predhpc import util



def get_T_shape_boundaries(width_prop=0.2, scale_x=1, scale_y=1, prop_x=None, 
                           prop_y=None):
    """Get boundaries for a T-shape environment.
    
    Args:
        width_prop (float): proportion of the width of the environment that the
            T-shape arms should take up.
        scale_x (float): scale of the environment in the x-direction.
        scale_y (float): scale of the environment in the y-direction.
        prop_x (float): proportion of the width of the environment that the
            T-shape should take up in the x-direction. If None, defaults to
            width_prop.
        prop_y (float): proportion of the width of the environment that the
            T-shape should take up in the y-direction. If None, defaults to
            width_prop.

    Returns:
        boundaries (list): list of lists of the form [[x1, y1], [x2, y2], ...]
    """

    prop_x = width_prop if prop_x is None else prop_x
    prop_y = width_prop if prop_y is None else prop_y

    for prop, dim in [(prop_x, "x"), (prop_y, "y")]:
        if prop >= 1:
            raise ValueError(
                f"{dim} proportion must be strictly smaller than 1, "
                f"but found {prop}."
                )

    # add diff and width
    left_t_x = 0.5 - prop_x / 2
    top_t_y = 1 - prop_y

    left_boundaries = [
        [0, 1],
        [0, top_t_y],
        [left_t_x, top_t_y],
        [left_t_x, 0]
    ]

    right_boundaries = [
        [1 - x, y] for x, y in left_boundaries
    ]

    boundaries_unscaled = left_boundaries + right_boundaries[::-1]
    boundaries = [[x * scale_x, y * scale_y] for x, y in boundaries_unscaled]

    return boundaries


class TEnv(Environment, util.ParamsMixin):
    """T-shaped environment.   
    """

    default_params = {
        "prop_width": 0.2,
        "scale_x": 1,
        "scale_y": 1,
        "prop_x": None,
        "prop_y": None,
    }
    
    ignored_param_keys = ["boundary", "scale", "aspect"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "holes": [],  # no holes
    }

    def __init__(self, params=dict()):
    
        self.check_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)     
        self.params.update(params)

        self.set_fixed_params()
                
        super().__init__(self.params)

        self.prop_x = self.prop_width if self.prop_x is None else self.prop_x
        self.prop_y = self.prop_width if self.prop_y is None else self.prop_y


    def set_fixed_params(self):
        """Sets fixed parameters.
        """
        all_fixed_params = self.get_all_fixed_params()
        for key, value in all_fixed_params.items():
            self.params[key] = value

        self.params["boundary"] = get_T_shape_boundaries(
            width_prop=self.params["prop_width"],
            scale_x=self.params["scale_x"],
            scale_y=self.params["scale_y"],
            prop_x=self.params["prop_x"],
            prop_y=self.params["prop_y"],
            )


    @property
    def branch_y(self):
        if not hasattr(self, "_branch_y"):
            self._branch_y = (1 - self.prop_y) * self.scale_y
        return self._branch_y


    @property
    def left_T_end(self):
        if not hasattr(self, "_left_T_end"):
            x_dim = self.prop_x / 2 * self.scale_x
            y_dim = (1 - self.prop_y / 2) * self.scale_y
            self._left_T_end = [x_dim, y_dim]
        return self._left_T_end

    @property
    def right_T_end(self):
        if not hasattr(self, "_right_T_end"):
            x_dim = (1 - self.prop_x / 2) * self.scale_x
            y_dim = (1 - self.prop_y / 2) * self.scale_y
            self._right_T_end = [x_dim, y_dim]
        return self._right_T_end

    @property
    def T_ends(self):
        """Get the coordinates of the ends of the T-shape arms.
        """

        return [self.left_T_end, self.right_T_end]


    @property
    def T_start(self):
        """Get the coordinates of the start of the T-shape.
        """

        if not hasattr(self, "_T_start"):
            x_dim = 0.5 * self.scale_x
            y_dim = (self.prop_y / 2) * self.scale_y
            self._T_start = [x_dim, y_dim]

        return self._T_start


    @property
    def T_split(self):
        """Get the coordinates of the split of the T branches.
        """

        if not hasattr(self, "_T_split"):
            x_dim = 0.5 * self.scale_x
            y_dim = (1 - self.prop_y / 2) * self.scale_y
            self._T_split = [x_dim, y_dim]

        return self._T_split



    def plot_environment(self, fig=None, ax=None, **kwargs):
        """Plot the environment.
        """

        fig, ax = super().plot_environment(fig=fig, ax=ax, **kwargs)

        ax.scatter(*self.T_start, marker="o", color="blue", s=20, zorder=5, label="start")
        ax.scatter(*self.left_T_end, marker="x", color="red", s=20, zorder=5, label="reset")
        ax.scatter(*self.right_T_end, marker="x", color="red", s=20, zorder=5)
        ax.legend(loc="lower right", frameon=False)

        return fig, ax

