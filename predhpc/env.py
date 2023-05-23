import copy

from ratinabox import Environment

from predhpc import util






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

        self.params["boundary"] = util.get_T_shape_boundaries(
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

