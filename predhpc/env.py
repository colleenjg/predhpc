import copy
import itertools
from typing import Any, TYPE_CHECKING
import warnings

import numpy as np
from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import figure as mpl_figure  # type: ignore[import]
from matplotlib import colormaps as mpl_cmap  # type: ignore[import]
import pandas as pd  # type: ignore[import]

from ratinabox import Environment as riabEnv  # type: ignore[import]

from predhpc.util import gen_util, trig_util, params_util, plot_util, ext_util


class EnvironmentWarning(UserWarning):
    """Class is for environment-related user warnings."""

    pass


warnings.simplefilter("once", EnvironmentWarning)


class Environment(riabEnv, ext_util.ParamsManagerMixin):
    """
    Environment()

    Class ratinabox environment.

    A parameters dictionary can be passed at initialisation.

    See ratinabox.Environment for default parameters.

    See ratinabox.Environment for properties.

    List of methods (in addition to ratinabox.Environment methods):
        • self.get_environment_figsize()
        • self.plot_environment()
    """

    default_params = dict()  # type: dict[str, Any]

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, params=dict()):
        """
        Environment()

        Initialise an environment.

        Args:
        - params (dict, optional): Environment parameters. Default is dict().
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(params=params)

    def get_environment_figsize(self, size_fact=1.0):
        """
        self.get_environment_figsize()

        Obtain the figure size for a 2D environment.

        Args:
        - size_fact (float, optional): Size factor by which to expand the environment
            figure size in each dimension. Default is 1.0.

        Raises:
        - RuntimeError: If the environment dimensionality is not 2D.

        Returns:
        - figsize (tuple): Figure size for the environment.
        """

        if self.dimensionality != "2D":
            raise RuntimeError(
                "Environment dimensionality must be 2D to get the environment figsize."
            )

        extent = self.extent
        x_base = extent[1] - extent[0]
        y_base = extent[3] - extent[2]
        figsize = (size_fact * x_base, size_fact * y_base)

        return figsize

    def plot_environment(
        self,
        fig=None,
        ax=None,
        return_env_fig=False,
        s=10,
        alpha=0.8,
        plot_objects=True,
        **kwargs,
    ):
        """
        self.plot_environment()

        Plot the environment.

        Args:
        - fig (mpl_figure.Figure, optional): Figure with subplot to plot on. If None,
            a new figure is created. Kept for compatibility and inferred if missing.
            Default is None.
        - ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is created.
            Default is None.
        - return_env_fig (bool, optional): Whether to return the figure
            (for compatibility). Default is False.
        - s (float, optional): Size for the plotted objects. Default is 10.
        - alpha (float, optional): Alpha value for the plotted objects. Default is 0.8.
        - plot_objects (bool, optional): Whether to plot the objects. Default is True.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Environment.plot_environment().

        Returns:
        if return_fig:
        - fig (mpl_figure.Figure): Figure with environment plotted

        - sub_ax (plt.Axes): Subplot with environment plotted.
        """

        kwargs = plot_util.organize_fig_ax_kwargs(fig=fig, ax=ax, **kwargs)

        fig, sub_ax = super().plot_environment(plot_objects=False, **kwargs)

        if plot_objects:
            object_cmap = mpl_cmap[self.object_colormap]
            for i, object in enumerate(self.objects["objects"]):
                object_color = object_cmap(
                    self.objects["object_types"][i] / (self.n_object_types - 1 + 1e-8)
                )
                sub_ax.scatter(
                    object[0],
                    0,
                    facecolor=[0, 0, 0, 0],
                    edgecolors=object_color,
                    s=s,
                    zorder=2,
                    marker="o",
                    alpha=alpha,
                )

        if return_env_fig:
            return fig, sub_ax
        else:
            return sub_ax


class TEnv(Environment):
    """
    TEnv()

    Class extending environment to a T-shape structure.

    A parameters dictionary can be passed at initialisation.

    default_params = {
        "prop_env": 0.2,  # T-shape arms and stem width (prop of env dims)
        "scale_x": 1,  # env width
        "scale_y": 1,  # env height
        "stem_width_as_prop_of_x": None,  # T-shape stem width (prop of env width)
        "arm_height_as_prop_of_y": None,  # T-shape arms width (prop of env height)
    }

    List of properties (in addition to Environment properties):
        • self.stem_left
        • self.stem_right
        • self.branch_y
        • self.left_T_end
        • self.right_T_end
        • self.T_ends
        • self.T_start
        • self.T_split
        • self.top_arms_prop_of_area

    List of methods (in addition to Environment methods):
        • self.add_fixed_params()
        • self.get_scale_x()
        • self.get_scale_y()
        • self.get_T_extents()
        • self.sample_positions()
        • self.plot_environment()
    """

    default_params = {
        "prop_env": 0.2,  # T-shape arms and stem width (prop of env dims)
        "scale_x": 1,  # env width
        "scale_y": 1,  # env height
        "stem_width_as_prop_of_x": None,  # T-shape stem width (prop of env width)
        "arm_height_as_prop_of_y": None,  # T-shape arms width (prop of env height)
    }

    ignored_param_keys = ["boundary", "scale", "aspect"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "holes": [],  # no holes
    }

    def __init__(self, params: dict[str, Any] = dict()):
        """
        TEnv()

        Initialise the T-shaped environment.

        Attributes:
        - arm_height_as_prop_of_y (float): T-shape arms width (prop of env height)
        - stem_width_as_prop_of_x (float): T-shape stem width (prop of env width)

        Args:
        - params (dict, optional): Environment parameters. Default is dict().
        """

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.params["boundary"] = ext_util.get_T_shape_env_boundaries(
            prop_env=self.params["prop_env"],
            scale_x=self.params["scale_x"],
            scale_y=self.params["scale_y"],
            stem_width_as_prop_of_x=self.params["stem_width_as_prop_of_x"],
            arm_height_as_prop_of_y=self.params["arm_height_as_prop_of_y"],
        )

        super().__init__(self.params)

        self.scale = (self.scale_x + self.scale_y) / 2  # type: ignore[attr-defined]

        prop_env = self.prop_env  # type: ignore[attr-defined]
        if self.stem_width_as_prop_of_x is None:  # type: ignore[attr-defined,has-type]
            self.stem_width_as_prop_of_x = prop_env

        if self.arm_height_as_prop_of_y is None:  # type: ignore[attr-defined,has-type]
            self.arm_height_as_prop_of_y = prop_env

    @property
    def stem_left(self) -> float:
        """
        self.stem_left

        Obtain the left x-coordinate of the T-shape stem.

        Returns:
        - (float): Left x-coordinate of the T-shape stem.
        """

        if not hasattr(self, "_stem_left"):
            self._stem_left = (
                0.5 - self.stem_width_as_prop_of_x / 2
            ) * self.get_scale_x()
        return self._stem_left

    @property
    def stem_right(self) -> float:
        """
        self.stem_right

        Obtain the right x-coordinate of the T-shape stem.

        Returns:
        - (float): Right x-coordinate of the T-shape stem.
        """

        if not hasattr(self, "_stem_right"):
            self._stem_right = (
                0.5 + self.stem_width_as_prop_of_x / 2
            ) * self.get_scale_x()
        return self._stem_right

    @property
    def branch_y(self) -> float:
        """
        self.branch_y

        Obtain the y-coordinate of the branch of the T-shape.

        Returns:
        - (float): Y-coordinate of the branch of the T-shape.
        """

        if not hasattr(self, "_branch_y"):
            self._branch_y = (1 - self.arm_height_as_prop_of_y) * self.get_scale_y()
        return self._branch_y

    @property
    def left_T_end(self) -> list[float]:
        """
        self.left_T_end

        Obtain the coordinates of the end of the left arm of the T-shape.

        Returns:
        - (list): Coordinates of the end of the left arm of the T-shape.
        """

        if not hasattr(self, "_left_T_end"):
            x_dim = self.stem_width_as_prop_of_x / 2 * self.get_scale_x()
            y_dim = (1 - self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._left_T_end = [x_dim, y_dim]
        return self._left_T_end

    @property
    def right_T_end(self) -> list[float]:
        """
        self.right_T_end

        Obtain the coordinates of the end of the right arm of the T-shape.

        Returns:
        - (list): Coordinates of the end of the right arm of the T-shape.
        """

        if not hasattr(self, "_right_T_end"):
            x_dim = (1 - self.stem_width_as_prop_of_x / 2) * self.get_scale_x()
            y_dim = (1 - self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._right_T_end = [x_dim, y_dim]
        return self._right_T_end

    @property
    def T_ends(self) -> list[list[float]]:
        """
        self.T_ends

        Obtain the coordinates of the ends of the T-shape arms.

        Returns:
        - (list): Coordinates of the ends of the T-shape arms [left, right].
        """

        return [self.left_T_end, self.right_T_end]

    @property
    def T_start(self) -> list[float]:
        """
        self.T_start

        Obtain the coordinates of the start of the T-shape.

        Returns:
        - (list): Coordinates of the start of the T-shaped environment (base of T).
        """

        if not hasattr(self, "_T_start"):
            x_dim = 0.5 * self.get_scale_x()
            y_dim = (self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._T_start = [x_dim, y_dim]

        return self._T_start

    @property
    def T_split(self) -> list[float]:
        """
        self.T_split

        Obtain the coordinates of the split of the T branches.

        Returns:
        - (list): Coordinates of the location where the T branches split from the trunk.
        """

        if not hasattr(self, "_T_split"):
            x_dim = 0.5 * self.get_scale_x()
            y_dim = (1 - self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._T_split = [x_dim, y_dim]

        return self._T_split

    @property
    def T_split_top(self) -> list[float]:
        """
        self.T_split_top

        Obtain the coordinates at the top of the split of the T branches.

        Returns:
        - (list): Top coordinates above location where the T branches split from the trunk.
        """

        if not hasattr(self, "_T_split_top"):
            x_dim = 0.5 * self.get_scale_x()
            y_dim = self.get_scale_y()
            self._T_split_top = [x_dim, y_dim]

        return self._T_split_top

    @property
    def top_arms_prop_of_area(self):
        """
        self.top_arms_prop_of_area

        Obtain the proportion of the T-shape area that is occupied by the top arms.

        Returns:
        - (float): Proportion of the T-shape area that is occupied by the top arms.
        """

        bottom_stem_area = self.get_area(area="bottom")

        top_arms_area = self.get_area(area="top")

        top_arms_prop_of_area = top_arms_area / (top_arms_area + bottom_stem_area)

        return top_arms_prop_of_area

    def get_scale_x(self):
        """
        self.get_scale_x()

        Obtain the x-scale of the environament in the x direction.

        Returns:
        - (float): x-scale of the environment in the x direction.
        """

        return self.scale_x  # type: ignore[attr-defined]

    def get_scale_y(self):
        """
        self.get_scale_y()

        Obtain the y-scale of the environament in the y direction.

        Returns:
        - (float): y-scale of the environment in the y direction.
        """

        return self.scale_y  # type: ignore[attr-defined]

    def get_area(self, area="both"):
        """
        self.get_area()

        Obtain the area of the T-shaped environment.

        Args:
        - area (str, optional): Area to get extent for ("top", "bottom" or "both").
            Default is "both".

        Returns:
        - total_area (float): Total area of the T-shaped environment.
        """

        total_area = 0
        if area in ["top", "both"]:
            top_arms_height = self.arm_height_as_prop_of_y * self.get_scale_y()
            top_arms_width = self.get_scale_x()
            top_arms_area = top_arms_height * top_arms_width
            total_area += top_arms_area
        if area in ["bottom", "both"]:
            bottom_stem_height = (1 - self.arm_height_as_prop_of_y) * self.get_scale_y()
            bottom_stem_width = self.stem_width_as_prop_of_x * self.get_scale_x()
            bottom_stem_area = bottom_stem_height * bottom_stem_width
            total_area += bottom_stem_area
        if area not in ["top", "bottom", "both"]:
            raise ValueError(f"Unknown area: {area}")

        return total_area

    def get_T_extents(self, area="both"):
        """
        self.get_T_extents()

        Obtain the extent of the T-shaped environment in the x and y directions.

        Args:
        - area (str, optional): Area to get extent for. Default is "both".

        Returns:
        - extent_x (list): Extent of the T-shaped environment in the x direction.
        - extent_y (list): Extent of the T-shaped environment in the y direction.
        """

        if area == "top":
            extent_x = [0, self.get_scale_x()]
            extent_y = [
                self.get_scale_y() * (1 - self.arm_height_as_prop_of_y),
                self.get_scale_y(),
            ]
        elif area == "bottom":
            extent_x = [
                (0.5 - self.stem_width_as_prop_of_x / 2) * self.get_scale_x(),
                (0.5 + self.stem_width_as_prop_of_x / 2) * self.get_scale_x(),
            ]
            extent_y = [0, self.get_scale_y() * (1 - self.arm_height_as_prop_of_y)]
        else:
            raise ValueError(f"Unknown area: {area}")

        return extent_x, extent_y

    def sample_positions(self, n=10, method="uniform_jitter", area="both"):
        """
        self.sample_positions()

        Sample n positions across the T-shaped environment.

        Args:
        - n (int): Number of positions to sample across the T-shaped environment.
            Default is 10.
        - method (str): Method for sampling the positions
            (e.g., "uniform", "uniform_jittered" or "random").
            Default is "uniform_jitter".
        - area (str): Area to sample from ("top", "bottom", or "both").
            Default is "both".

        Returns:
        - positions (2D np.ndarray): Positions with shape (n, 2).
        """

        n_top, n_bottom = ext_util.get_num_samples_top_bottom_T_arms(
            n, area=area, top_arms_prop_of_area=self.top_arms_prop_of_area
        )

        positions = list()
        adjusted_bottom_upper_limit = None  # for uniform sampling
        for area in ["top", "bottom"]:
            n = n_top if area == "top" else n_bottom
            if n == 0:
                continue

            extent_x, extent_y = self.get_T_extents(area=area)

            area_positions, adjusted_bottom_upper_limit = ext_util.sample_from_T_areas(
                n,
                extent_x=extent_x,
                extent_y=extent_y,
                method=method,
                top=(area == "top"),
                adjusted_bottom_upper_limit=adjusted_bottom_upper_limit,
            )
            positions.append(area_positions)

        positions = np.vstack(positions)

        return positions

    def plot_environment(
        self,
        fig: mpl_figure.Figure | None = None,
        sub_ax: plt.Axes | None = None,
        return_env_fig: bool = False,
        no_legend: bool = False,
        base_s: float = 15,
        alpha: float = 0.8,
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        self.plot_environment()

        Plot the environment.

        Args:
        - fig (mpl_figure.Figure, optional): Figure with subplot to plot on. If None,
            a new figure is created. Kept for compatibility and inferred if missing.
            Default is None.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created.
        - return_env_fig (bool, optional): Whether to return the figure
            (for compatibility). Default is False.
        - no_legend (bool, optional): Whether to plot the legend. Default is False.
        - base_s (float, optional): Base size for the plotted objects. Default is 15.
        - alpha (float, optional): Alpha value for the plotted objects. Default is 0.8.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Environment.plot_environment().

        Returns:
        if return_fig:
        - fig (mpl_figure.Figure): Figure with environment plotted

        - sub_ax (plt.Axes): Subplot with environment plotted.
        """

        kwargs = plot_util.organize_fig_ax_kwargs(fig=fig, sub_ax=sub_ax, **kwargs)

        sub_ax = super().plot_environment(autosave=False, plot_objects=False, **kwargs)

        start_kwargs = plot_util.get_plot_marker_kwargs("start", base_s=base_s)
        sub_ax.scatter(
            *self.T_start, zorder=5, label="start", alpha=alpha, **start_kwargs
        )

        if len(self.objects):
            target_kwargs = plot_util.get_plot_marker_kwargs("target", base_s=base_s)
            for object_coords in self.objects["objects"]:
                sub_ax.scatter(
                    *object_coords,
                    zorder=5,
                    label="target",
                    alpha=alpha,
                    **target_kwargs,
                )

        reset_kwargs = plot_util.get_plot_marker_kwargs("reset", base_s=base_s)
        for T_end in [self.left_T_end, self.right_T_end]:
            sub_ax.scatter(*T_end, zorder=5, label="reset", alpha=alpha, **reset_kwargs)

        sub_ax.legend(loc="lower right", frameon=False)

        legend = sub_ax.get_legend()
        if no_legend and legend is not None:
            legend.remove()

        fig = sub_ax.figure
        plot_util.save_figure(fig, "Environment", save=autosave)

        if return_env_fig:
            return fig, sub_ax
        else:
            return sub_ax


class OpenField(Environment):
    """
    OpenField()

    Class extending environment to an open field structure.

    A parameters dictionary can be passed at initialisation.

    default_params = {
        "init_random_reward_obj": 1,
        "init_random_novel_obj": 5,
        "init_random_walls": 5,
        "init_random_teleport_pairs": 2,
        "wall_lengths": [0.1, 0.2],
        "init_reward_objs": list(),
        "init_novel_objs": list(),
        "init_teleport_pairs": list(),
        "min_dist": 0.1,  # between objects (walls is half)
        "init_seed": None,
    }

    List of properties (in addition to Environment properties):
        • self.object_df_columns
        • self.object_df
        • self.object_type_num_to_name_dict
        • self.object_type_name_to_num_dict
        • self.teleport_pairs_dict

    List of methods (in addition to Environment methods):
        • self.get_object_type_num_to_plot_params_dict
        • self.get_area()
        • self.check_if_walls_ends_too_close()
        • self.get_teleport_coords()
        • self.get_teleport_pair_orientation()
        • self.get_number_for_each_object_type()
        • self.get_dist_from_coords_to_closest_object()
        • self.get_dist_from_coords_to_closest_wall()
        • self.sample_coords()
        • self.sample_wall_end()
        • self.add_object()
        • self.add_reward_objects()
        • self.add_novel_objects()
        • self.add_teleport_pairs()
        • self.add_walls()
        • self.get_teleport_plotting_marker()
        • self.plot_environment()
    """

    default_params = {
        "init_random_reward_obj": 1,
        "init_random_novel_obj": 5,
        "init_random_walls": 5,
        "init_random_teleport_pairs": 2,
        "wall_lengths": [0.1, 0.2],
        "init_reward_obj": list(),
        "init_novel_obj": list(),
        "init_teleport_pairs": list(),
        "min_dist": 0.1,  # between objects (walls is half)
        "init_seed": None,
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "holes": [],  # no holes,
        "boundary": None,
    }

    def __init__(self, params: dict[str, Any] = dict()):
        """
        OpenField()

        Initialise an open field environment.

        Attributes:
        - num_teleport_pairs (num): Number of teleportation pairs in the environment.
        - rng (np.random.RandomState): Random number generator.

        Args:
        - params (dict, optional): Environment parameters. Default is dict().
        """

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(self.params)

        if min(self.wall_lengths) <= 0:  # type: ignore[attr-defined]
            raise ValueError("Wall lengths must be positive.")

        if self.init_seed is None:  # type: ignore[attr-defined]
            self.rng = np.random
        else:
            self.rng = np.random.RandomState(self.init_seed)  # type: ignore[attr-defined,assignment]

        self.num_teleport_pairs = 0
        self.add_reward_objects(coords=self.init_reward_obj)
        self.add_novel_objects(coords=self.init_novel_obj)
        self.add_teleport_pairs(coord_pairs=self.init_teleport_pairs)

        self.add_reward_objects(self.init_random_reward_obj)  # type: ignore[attr-defined]
        self.add_novel_objects(self.init_random_novel_obj)  # type: ignore[attr-defined]
        self.add_teleport_pairs(self.init_random_teleport_pairs)  # type: ignore[attr-defined]
        self.add_walls(self.init_random_walls)  # type: ignore[attr-defined]

    @property
    def object_df_columns(self):
        """
        self.object_df_columns

        Obtain the columns for the object dataframe.

        Returns:
        - (list): Columns for the object dataframe.
        """

        if not hasattr(self, "_object_df_columns"):
            self._object_df_columns = [
                "object_type_num",
                "object_type_name",
                "idx_within_type",
                "position_x",
                "position_y",
                "teleport_pair_num",
                "teleport_direction",
            ]

        return self._object_df_columns

    @property
    def object_df(self):
        """
        self.object_df

        Obtain the object dataframe.

        Returns:
        - (pd.DataFrame): Object dataframe.
        """

        if not hasattr(self, "_object_df"):
            object_df = pd.DataFrame(columns=self.object_df_columns)
            self._object_df = object_df

        return self._object_df

    def _get_object_type_num(self, object_type_name="reward") -> int:
        """
        self._get_object_type_num()

        Obtain the type number for the provided object name. If the name is not in the
        existing dictionary num to name dictionary, it will be added.

        Returns:
        - object_type_num (int): Object type number.
        """

        if object_type_name in self.object_type_name_to_num_dict.keys():
            object_type_num = self.object_type_name_to_num_dict[object_type_name]

        else:
            object_type_nums = list(self.object_type_num_to_name_dict.keys())
            if len(object_type_nums):
                object_type_num = np.max(object_type_nums) + 1
            else:
                object_type_num = 0
            self._object_type_num_to_name_dict[object_type_num] = object_type_name

        return object_type_num

    @property
    def object_type_num_to_name_dict(self) -> dict[int, str]:
        """
        self.object_type_num_to_name_dict

        Dictionary for getting object type name from object type number.

        Returns:
        - object_type_num_to_name_dict (dict): Dictionary with keys and values:
            - object type number (int): object type name (str)
        """

        if not hasattr(self, "_object_type_num_to_name_dict"):
            self._object_type_num_to_name_dict = dict()

        return self._object_type_num_to_name_dict

    @property
    def object_type_name_to_num_dict(self) -> dict[str, int]:
        """
        self.object_type_name_to_num_dict

        Dictionary for getting object type number from object type name.

        Returns:
        - object_type_name_to_num_dict (dict): Dictionary with keys and values:
            - object type name (str): object type number (int)
        """

        object_type_name_to_num_dict = {
            val: key for key, val in self.object_type_num_to_name_dict.items()
        }

        return object_type_name_to_num_dict

    @property
    def teleport_pairs_dict(self) -> dict[int, dict[str, tuple[int, list[float]]]]:
        """
        self.teleport_pairs_dict

        Obtain dictionary of teleportation pairs (directional).

        Returns:
        - teleport_pairs_dict (dict): Teleportation pairs dictionary with keys and
            values (dict):
            - teleport pair number (int): Teleportation pair dictionary with
                keys and values (dict):
                - "in": object type number and coordinates for teleport in (tuple)
                - "out": object type number and coordinates for teleport out (tuple)
        """

        if not hasattr(self, "_teleport_pairs_dict"):
            teleport_pairs_dict = dict()
            for name, object_type in self.object_type_name_to_num_dict.items():
                if name.startswith("teleport_") and "in" in name:
                    object_type_in = object_type
                    teleport_pair = int(
                        name.replace("teleport_", "").replace("_in", "")
                    )
                    out_key = f"teleport_{teleport_pair}_out"
                    if out_key not in self.object_type_name_to_num_dict.keys():
                        raise RuntimeError(
                            f"Teleport in {teleport_pair} does not have 'out' pair."
                        )
                    object_type_out = self.object_type_name_to_num_dict[out_key]

                    coords = list()
                    for object_type in [object_type_in, object_type_out]:
                        object_idxs = np.where(
                            self.objects["object_types"] == object_type
                        )[0]
                        if len(object_idxs) != 1:
                            raise RuntimeError(
                                f"Expected teleport in {teleport_pair} to correspond "
                                f"to exactly one object, but found {len(object_idxs)}."
                            )
                        coords.append(self.objects["objects"][object_idxs[0]])

                    teleport_pairs_dict[teleport_pair] = {
                        "in": (object_type_in, coords[0]),
                        "out": (object_type_out, coords[1]),
                    }

            self._teleport_pairs_dict = teleport_pairs_dict

        return self._teleport_pairs_dict

    def get_object_type_num_to_plot_params_dict(
        self, s=20
    ) -> dict[int, dict[str, Any]]:
        """
        self.get_object_type_num_to_plot_params_dict()

        Obtains plotting parameters from object type number.

        Args:
        - s (int, optional): Size for the object. Default is 20.

        Returns:
        - object_type_num_to_plot_params_dict (dict): Dictionary with keys and values:
            - object type number (int): dictionary with keys and values (dict):
                - "name" (str): object type name
                - "marker" (str): marker for the object
                - "color" (str): color for the object
                - "s" (int): size for the object
                - "zorder" (int): zorder for the object
        """

        teleport_nums = [
            val.replace("teleport_", "").replace("in_", "")
            for val in self.object_type_num_to_name_dict.values()
            if val.startswith("teleport") and "_in" in val
        ]
        teleport_vals = np.linspace(0.5, 1, len(teleport_nums))
        teleport_colors = plt.get_cmap("Oranges")(teleport_vals)  # type: ignore[callable]

        object_type_num_to_plot_params_dict = dict()
        for num, name in self.object_type_num_to_name_dict.items():
            if name == "reward":
                object_type_num_to_plot_params_dict[num] = {
                    "name": name,
                    "marker": "o",
                    "color": params_util.TARGET_COLOR,
                    "s": s,
                    "zorder": 5,
                }
            elif name == "novel":
                object_type_num_to_plot_params_dict[num] = {
                    "name": name,
                    "marker": "o",
                    "color": params_util.NOVEL_COLOR,
                    "s": s,
                    "zorder": 5,
                }
            elif name.startswith("teleport"):
                direc = "in" if "_in" in name else "out"
                teleport_num = int(
                    name.replace("teleport_", "").replace(f"_{direc}", "")
                )
                color = teleport_colors[teleport_num]
                object_type_num_to_plot_params_dict[num] = {
                    "name": name,
                    "marker": self.get_teleport_plotting_marker(
                        teleport_num, direction=direc
                    ),
                    "color": color,
                    "s": s,
                    "zorder": 5,
                }
            else:
                raise ValueError(f"Unknown object type name: {name}")

        return object_type_num_to_plot_params_dict

    def get_area(self):
        """
        self.get_area()

        Obtain the area of the open field environment.

        Returns:
        - area (float): Area of the open field environment.
        """

        if len(self.boundary) != 4:
            raise NotImplementedError(
                "get_area() method expects exactly four boundary walls."
            )

        boundaries = np.asarray(self.boundary)
        distances = np.diff(np.append(boundaries, boundaries[0:1], axis=0), axis=0)

        if (distances[:2] != -distances[3:]).all():
            raise NotImplementedError("get_area() method expects a parallelogram.")

        side_lengths = np.linalg.norm(distances[:2], ord=2, axis=1)
        angle = np.deg2rad(
            trig_util.get_angle_between_vectors(distances[0], distances[1])
        )
        short_idx = np.argmin(side_lengths)
        height = np.sin(angle) * side_lengths[short_idx]

        area = side_lengths[1 - short_idx] * height

        if len(self.holes):
            raise NotImplementedError(
                "get_area() method does not yet take holes into account."
            )

        return area

    def check_if_walls_ends_too_close(
        self,
        new_wall_coords: (
            np.ndarray[tuple[int, int], np.dtype[np.float64]] | list[list[float]]
        ),
        min_dist=None,
    ) -> bool:
        """
        self.check_if_walls_ends_too_close(new_wall_coords)

        Checks whether a new wall's ends is too close to the ends of existing
        walls.

        Specifically checks whether an end of the new wall intersects at less
        than 45 degrees near the end of an existing wall, forming an V shape
        with small ends sticking out. If so, returns True, else False.

        Does NOT check whether the new wall overlaps exactly with an existing
        wall, or intersects near the middle of either wall.

        Args:
        - new_wall_coords (list or 2D np.ndarray): Coordinates of new wall
            [[x1, y1], [x2, y2]].
        - min_dist (float, optional): Minimum distance between walls. Default is None.

        Returns:
        - walls_ends_too_close (bool): True if the ends of a new wall are too close
            to an existing wall. False otherwise.
        """

        if len(self.walls) == 0:
            return False

        if min_dist is None:
            min_dist = float(self.min_dist)  # type: ignore[attr-defined]

        new_wall = np.asarray(new_wall_coords)

        walls_ends_too_close = False
        for wall in self.walls:
            # get angle between two vectors
            angle = trig_util.get_angle_between_vectors(
                np.diff(new_wall_coords, axis=0)[0], np.diff(wall, axis=0)[0]
            )

            if angle > 45:
                continue

            # if angle is less than 45 degrees, check any ends of the walls are too
            # close together
            distances, coords = list(), []
            for c1, c2 in itertools.product([0, 1], [0, 1]):
                coords.append([c1, c2])
                distances.append(np.linalg.norm(wall[c1] - new_wall[c2], ord=2))

            order = np.argsort(distances)

            if distances[order[0]] < min_dist:
                # farther must be at least as far as if the walls
                # intersected only at their ends (no intersection)
                farthest = distances[order[-1]]
                c1, c2 = coords[order[-1]]

                end1 = wall[c1] - wall[1 - c1]
                end2 = new_wall[c2] - new_wall[1 - c2]
                exp_dist = np.linalg.norm(end1 - end2, ord=2)

                if farthest < exp_dist:
                    walls_ends_too_close = True
                    break

        return walls_ends_too_close

    def get_object_locations(self, object_type_name="reward"):
        """
        self.get_object_locations()

        Obtain the locations of objects of a given type in the environment.

        Args:
        - object_type_name (str, optional): Name of the object type to get locations
            for (e.g., "reward", "novel"). Default is "reward".

        Raises:
        - ValueError: If no objects of the specified type are found.

        Returns:
        - object_locations (2D np.ndarray): Locations of the objects of the specified
            type with shape (n_objects, 2), where n_objects is the number of objects of
            the specified type.
        """

        sub_df = self.object_df.loc[
            self.object_df["object_type_name"] == object_type_name
        ]
        if len(sub_df) == 0:
            raise ValueError(f"No objects of type {object_type_name} found.")

        pos_x = sub_df["position_x"].to_numpy()
        pos_y = sub_df["position_y"].to_numpy()

        object_locations = np.vstack((pos_x, pos_y)).T

        return object_locations

    def get_teleport_coords(self, teleport_pair_num: int = 1, direction="in"):
        """
        self.get_teleport_coords()

        Obtain the teleport coordinates for the given teleportation pair.

        Args:
        - teleport_pair_num (int, optional): The teleportation pair to get the
            coordinates for. Default is 1.
        - direction (str, optional): The direction to get the coordinates for.
            Default is "in".

        Raises:
        - ValueError: If the teleportation pair number is not found.
        - ValueError: If the direction is not recognized.

        Returns:
        - teleport_coords (1D np.ndarray): The teleport coordinates.
        """

        if teleport_pair_num not in self.teleport_pairs_dict.keys():
            raise ValueError(f"Teleportation pair {teleport_pair_num} not found.")

        if direction not in ["in", "out"]:
            raise ValueError(f"Direction {direction} not recognized.")

        teleport_coords = self.teleport_pairs_dict[teleport_pair_num][direction][1]

        return teleport_coords

    def get_teleport_pair_orientation(self, teleport_pair_num: int = 1) -> str:
        """
        self.get_teleport_pair_orientation()

        Obtain the orientation of a teleportation pair.

        Args:
        - teleport_pair_num (int, optional): Teleportation pair number. Default is 1.

        Returns:
        - orientation (str): Orientation of the teleportation pair.
        """

        if teleport_pair_num % 2 == 0:
            orientation = "horizontal"
        else:
            orientation = "vertical"

        return orientation

    def get_number_for_each_object_type(self) -> tuple[int, int, int]:
        """
        self.get_number_for_each_object_type()

        Obtain the number of objects for each object type.

        Returns:
        - num_novel (int): Number of novel objects
        - num_reward (int): Number of reward objects
        - num_teleport (int): Number of teleport objects
        """

        num_novel, num_reward, num_teleport = 0, 0, 0
        for object_type in self.objects["object_types"]:  # type: ignore[attr-defined]
            object_name = self.object_type_num_to_name_dict[object_type]
            if object_name == "novel":
                num_novel += 1
            elif object_name == "reward":
                num_reward += 1
            elif "teleport" in object_name:
                num_teleport += 1

        if num_teleport % 2:
            raise RuntimeError("Number of teleportation pairs should be even.")

        return num_novel, num_reward, num_teleport

    def get_dist_from_coords_to_closest_object(
        self, coords: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        """
        self.get_dist_from_coords_to_closest_object(coords)

        Obtain the distance from a set of coordinates to the closest object in the
        environment.

        Args:
        - coords (1D np.ndarray): Coordinates [x, y] to get distance for.

        Returns:
        - closest_dist (float): Distance to closest object.
        """

        if len(self.objects["objects"]) == 0:
            return np.inf

        closest_distances = list()
        for object_coords in self.objects["objects"]:
            closest_distance = np.linalg.norm(coords - np.asarray(object_coords), ord=2)
            closest_distances.append(closest_distance)

        closest_dist = float(np.min(closest_distances))  # type: ignore[assignment]

        return closest_dist

    def get_dist_from_coords_to_closest_wall(
        self, coords: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> float:
        """
        self.get_dist_from_coords_to_closest_wall(coords)

        Obtain the distance from a set of coordinates to the closest wall.

        Args:
        - coords (1D np.ndarray): Coordinates [x, y] to get distance for.

        Returns:
        - closest_dist (float): Closest distance to wall.
        """

        if len(self.walls) == 0:
            return np.inf

        # returns points (1) x vectors x coords
        closest_dist = float(
            np.min(
                trig_util.shortest_distances_from_points_to_lines(coords, self.walls)
            )
        )

        return closest_dist

    def sample_coords(
        self, min_dist: float | None = None, max_attempts: int = 1000
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        self.sample_coords()

        Sample coordinates situated at least min_dist from the closest object
        (at least half min_dist for walls).

        Args:
        - min_dist (float, optional): Minimum distance to closest object or
            wall. Default is None.
        - max_attempts (int, optional): Maximum number of attempts to make to sample
            valid coordinates. Default is 1000.

        Raises:
        - ValueError: if could not sample valid coordinates after max_attempts
            attempts.

        Returns:
        - coords (1D np.ndarray): Sampled coordinates [x, y].
        """

        if min_dist is None:
            min_dist = float(self.min_dist)  # type: ignore[attr-defined]

        i = 0
        while True:
            x = self.rng.uniform(self.extent[0], self.extent[1])
            y = self.rng.uniform(self.extent[2], self.extent[3])

            coords = np.asarray([x, y])

            # check distance to objects, then walls
            if self.get_dist_from_coords_to_closest_object(coords) >= min_dist:
                if self.get_dist_from_coords_to_closest_wall(coords) >= min_dist / 2:
                    break
            if i > max_attempts:
                raise ValueError(
                    "Could not sample valid coordinates situated at least "
                    f"{min_dist} from the closest objects (or half for walls)."
                )
            i += 1

        return coords

    def sample_wall_end(
        self,
        start_coords: np.ndarray[tuple[int], np.dtype[np.float64]],
        min_dist: float | None = None,
        raise_error: bool = False,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """
        self.sample_wall_end(start_coords)

        Sample valid coordinates for the end of a wall given the start coordinates.

        Args:
        - start_coords (1D np.ndarray): Start of wall.
        - min_dist (float, optional): Minimum distance to closest object.
            Default is None.
        - raise_error (bool, optional): Whether to raise an error if could not sample
            valid end coordinates. Default is False.

        Returns:
        - end_coords (1D np.ndarray): Sampled end of wall coordinates [x, y].
            None if could not sample valid end coordinates, but raise_error is False.
        """

        if not self.check_if_position_is_in_environment(start_coords):
            return None

        if min_dist is None:
            min_dist = float(self.min_dist / 2)  # type: ignore[attr-defined]

        # sample wall length
        wall_length = self.rng.uniform(*self.wall_lengths)  # type: ignore[attr-defined]

        # sample orientation + direction, then cycle through if needed, before
        # abandoning each time check that the wall's max distance from another
        # objects is reasonable.
        wall_orientations = ["x", "y"]
        wall_directions = [-1, 1]
        wall_ori_direcs = list(itertools.product(wall_orientations, wall_directions))

        shuffle_order = np.arange(len(wall_ori_direcs))
        self.rng.shuffle(shuffle_order)
        wall_ori_direcs = [wall_ori_direcs[i] for i in shuffle_order]

        end_coords = None
        for wall_ori, wall_direc in wall_ori_direcs:
            c = 0 if wall_ori == "x" else 1
            end_coords = copy.deepcopy(np.asarray(start_coords))
            end_coords[c] += wall_length * wall_direc

            # check that end_coords are within bounds
            if not self.check_if_position_is_in_environment(end_coords):
                end_coords = None

            # check that end_coords are far enough from objects, if there are any
            if end_coords is not None and len(self.objects["objects"]) != 0:
                closest_dist = np.min(
                    trig_util.shortest_distances_from_points_to_lines(
                        self.objects["objects"], [start_coords, end_coords]
                    )
                )

                if closest_dist < min_dist:
                    end_coords = None

            if end_coords is not None:
                break

        if end_coords is None and raise_error:
            raise ValueError(
                "Could not sample valid end coordinates for wall given start "
                f"coordinates: {start_coords}."
            )

        return end_coords

    def add_object(
        self,
        object: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        object_type: str | int = "new",
    ):
        """
        self.add_object(object)

        Add an object to the environment, including to object dataframe.

        Args:
        - object (1D np.ndarray, optional): Object coordinates [x, y].
            If None, object coordinates are sampled. Default is None.
        - object_type_name (str or int, optional): Object type. Default is "new".
        """

        if object is None:
            object = self.sample_coords()

        if object_type == "new" or isinstance(object_type, int):
            raise RuntimeError(
                "For the OpenField, object type should be a string, but not 'new'."
            )

        object_type_num = self._get_object_type_num(object_type_name=object_type)
        super().add_object(object, type=object_type_num)  # type: ignore[arg-type]

        # add to object dataframe
        sub_df = self.object_df[self.object_df["object_type_num"] == object_type]
        if len(sub_df) == 0:
            idx_within_type = 0
        else:
            idx_within_type = sub_df["idx_within_type"].max() + 1

        new_object = {
            "object_type_num": object_type_num,
            "object_type_name": object_type,
            "idx_within_type": idx_within_type,
            "position_x": object[0],
            "position_y": object[1],
        }

        if "teleport" in object_type:
            _, teleport_pair_num, teleport_direction = object_type.split("_")
            new_object["teleport_pair_num"] = int(teleport_pair_num)
            new_object["teleport_direction"] = teleport_direction

        self.object_df.loc[len(self.object_df)] = new_object  # type: ignore[attr-defined]

    def add_reward_objects(
        self,
        num: int = 1,
        coords: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    ):
        """
        self.add_reward_objects()

        Add reward objects to environment.

        Args:
        - num (int, optional): Number of reward objects to add. Default is 1.
        - coords (1D np.ndarray, optional): Coordinates [x, y] to add reward objects at.
            If None, coordinates are sampled. Default is None.
        """

        if coords is not None:
            num = len(coords)

        for n in range(num):
            if coords is None:
                coord = self.sample_coords()
            else:
                coord = np.asarray(coords[n], dtype=np.float64).reshape(2)
                self.check_if_position_is_in_environment(coord)
            self.add_object(coord, object_type="reward")

    def add_novel_objects(
        self,
        num: int = 1,
        coords: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    ):
        """
        self.add_novel_objects()

        Add novel objects to environment.

        Args:
        - num (int, optional): Number of novel objects to add. Default is 1.
        - coords (1D np.ndarray, optional): Coordinates [x, y] to add novel objects at.
            If None, coordinates are sampled. Default is None.
        """

        if coords is not None:
            num = len(coords)

        for n in range(num):
            if coords is None:
                coord = self.sample_coords()
            else:
                coord = np.asarray(coords[n], dtype=np.float64).reshape(2)
                self.check_if_position_is_in_environment(coord)

            self.add_object(coord, object_type="novel")

    def add_teleport_pairs(self, num: int = 1, coord_pairs=None):
        """
        self.add_teleport_pairs()

        Add teleportation pairs (directional).

        Args:
        - num (int, optional): Number of teleportation pairs to add. Default is 1.
        - coord_pairs (list, optional): List of coordinate pairs [[x1, y1], [x2, y2]]
            for teleportation pairs. If None, coordinates are sampled. Default is None.
        """

        if coord_pairs is not None:
            num = len(coord_pairs)

        def format_teleport_pair(coord_pair):
            """
            format_teleport_pair(coord_pair)

            Format and check teleportation pair coordinates.

            Args:
            - coord_pair (list): Coordinate pair [[x1, y1], [x2, y2]].

            Returns:
            - coord_pair (list): Formatted coordinate pair
            """

            try:
                coords_in, coords_out = coord_pair
            except ValueError as err:
                if "values to unpack" in str(err):
                    raise ValueError(
                        "Expected two coordinates per teleportation pair, "
                        f"but got {len(coords)}."
                    )
                elif "unpack non-iterable" in str(err):
                    raise ValueError(
                        f"Each coordinate pair must be an iterable of length 2."
                    )
            coord_pair = [coords_in, coords_out]
            for c in range(2):
                coord_pair[c] = np.asarray(coord_pair[c], dtype=np.float64).reshape(2)
                self.check_if_position_is_in_environment(coord_pair[c])

            return coord_pair

        for n in range(num):
            i = self.num_teleport_pairs
            self.num_teleport_pairs += 1
            if coord_pairs is not None:
                coord_pair = format_teleport_pair(coord_pairs[n])
            for o, direction in enumerate(["in", "out"]):
                if coord_pairs is None:
                    coords = self.sample_coords()
                else:
                    coords = coord_pair[o]
                self.add_object(coords, object_type=f"teleport_{i}_{direction}")

    def add_walls(self, num: int = 1, max_attempts: int = 1000):
        """
        self.add_walls()

        Add walls to environment.

        Checks that walls are not too close to objects and that they do not
        overlap too much with one another.

        Does NOT check whether a new wall creates a hole.

        Args:
        - num (int, optional): Number of walls to add. Default is 1.
        - max_attempts (int, optional): Maximum number of attempts to sample
            valid wall start and end coordinates. Default is 1000.

        Raises:
        - ValueError: If could not sample valid wall start and end coordinates
            after max_attempts attempts.
        """

        if num > 0:
            warnings.warn(
                "add_walls() does not check whether a new wall will create a hole "
                "in the environment. Be sure to check environment visually.",
                category=EnvironmentWarning,
            )

        for _ in range(num):
            i = 0
            while True:
                start_coords = self.sample_coords()
                end_coords = self.sample_wall_end(start_coords)
                if end_coords is not None:
                    # check that wall ends are not too close to another
                    if self.check_if_walls_ends_too_close(
                        np.asarray([start_coords, end_coords])
                    ):
                        end_coords = None

                if end_coords is not None:
                    self.add_wall([start_coords, end_coords])
                    break
                if i > max_attempts:
                    raise ValueError(
                        "Could not sample valid wall start and end coordinates."
                    )
                i += 1

    def get_teleport_plotting_marker(
        self, teleport_pair_num: int = 1, direction: str = "in"
    ) -> str:
        """
        self.get_teleport_plotting_marker()

        Obtain the marker for a teleportation port.

        Args:
        - teleport_pair_num (int, optional): Teleportation pair number. Default is 1.
        - direction (str, optional): Direction of the teleportation port.
            Default is "in".

        Returns:
        - marker (str): Marker for the teleportation port.
        """

        orientation = self.get_teleport_pair_orientation(teleport_pair_num)

        if orientation == "vertical":
            marker = "v" if direction == "in" else "^"
        else:
            marker = ">" if direction == "in" else "<"

        return marker

    def plot_environment(
        self,
        fig: mpl_figure.Figure | None = None,
        sub_ax: plt.Axes | None = None,
        plot_objects: bool = True,
        skip_object_types: list = list(),
        size_fact: float = 2.5,
        alpha: float = 0.8,
        s: float = 20,
        no_legend: bool = False,
        return_env_fig: bool = False,
        scale_loc: None | tuple = None,
        scale_length: float = 0.2,
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        self.plot_environment()

        Plot the environment.

        Args:
        - fig (mpl_figure.Figure, optional): Figure with subplot to plot on. If None,
            a new figure is created. Kept for compatibility and inferred if missing.
            Default is None.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created.
        - plot_objects (bool, optional): Whether to plot objects in environment.
            Default is True.
        - skip_object_types (list, optional): List of object types to skip plotting.
            Default is an empty list.
        - size_fact (float, optional): Factor to multiply environment width by to
            determine figure size. Default is 2.5.
        - alpha (float, optional): Alpha value for plotting objects. Default is 0.8.
        - s (float, optional): Size of objects. Default is 20.
        - no_legend (bool, optional): Whether to remove legend from plot.
            Default is False.
        - return_env_fig (bool, optional): Whether to return the figure
            (for compatibility). Default is False.
        - scale_loc (None or tuple, optional): Location at which to add a scale bar to
            the plot. If None, no scale bar is added. Default is None.
        - scale_length (float, optional): Length of the scale bar in meters. Default
            is 0.2.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Keyword Args:
        - **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
        if return_env_fig:
        - fig (mpl_figure.Figure): Figure with environment plotted.

        - sub_ax (plt.Axes): Subplot with environment plotted.
        """

        kwargs = plot_util.organize_fig_ax_kwargs(fig=fig, sub_ax=sub_ax, **kwargs)

        if "ax" not in kwargs.keys():
            env_width, env_height = self.get_environment_figsize(size_fact=size_fact)
            if plot_objects:
                env_width *= 1.5  # for legend and labels
            kwargs["fig"], kwargs["ax"] = plt.subplots(figsize=(env_width, env_height))

        sub_ax = super().plot_environment(autosave=False, plot_objects=False, **kwargs)

        if plot_objects:
            object_type_num_to_plot_params_dict = (
                self.get_object_type_num_to_plot_params_dict(s=s)
            )
            labelled = list()
            for coords, object_type in zip(
                self.objects["objects"], self.objects["object_types"]
            ):
                name = object_type_num_to_plot_params_dict[object_type].pop("name")
                skip = any([skip_type in name for skip_type in skip_object_types])
                if skip:
                    continue
                label = None if name in labelled else name
                labelled.append(name)

                sub_ax.scatter(
                    *coords,
                    **object_type_num_to_plot_params_dict[object_type],
                    alpha=alpha,
                    label=label,
                )
                object_type_num_to_plot_params_dict[object_type]["name"] = name

            sub_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        legend = sub_ax.get_legend()
        if no_legend and legend is not None:
            legend.remove()

        if scale_loc is not None:
            y_shift = gen_util.get_proportion_edges(sub_ax.get_ylim(), 0.09)
            xs = [scale_loc[0] - scale_length / 2, scale_loc[0] + scale_length / 2]
            ys = [scale_loc[1], scale_loc[1]]
            sub_ax.plot(xs, ys, color="black", lw=2, zorder=10, alpha=0.8)
            # add text right below
            sub_ax.text(
                scale_loc[0],
                scale_loc[1] - y_shift,
                f"{scale_length} m",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                alpha=0.8,
            )

        fig = sub_ax.figure
        plot_util.save_figure(fig, "Environment", save=autosave)

        if return_env_fig:
            return fig, sub_ax
        else:
            return sub_ax
