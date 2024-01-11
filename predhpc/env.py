import copy
import itertools
from typing import Any
import warnings

import numpy as np
from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import markers
from matplotlib import figure as mpl_figure
import pandas as pd  # type: ignore[import]

from ratinabox import Environment  # type: ignore[import]

from predhpc import util


class EnvironmentWarning(UserWarning):
    pass


warnings.simplefilter("once", EnvironmentWarning)


class TEnv(Environment, util.ParamsManagerMixin):
    """T-shaped environment."""

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
        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        self.params["boundary"] = util.get_T_shape_env_boundaries(
            prop_env=self.params["prop_env"],
            scale_x=self.params["scale_x"],
            scale_y=self.params["scale_y"],
            stem_width_as_prop_of_x=self.params["stem_width_as_prop_of_x"],
            arm_height_as_prop_of_y=self.params["arm_height_as_prop_of_y"],
        )

        super().__init__(self.params)

        prop_env = self.prop_env  # type: ignore[attr-defined]
        if self.stem_width_as_prop_of_x is None:  # type: ignore[attr-defined,has-type]
            self.stem_width_as_prop_of_x = prop_env

        if self.arm_height_as_prop_of_y is None:  # type: ignore[attr-defined,has-type]
            self.arm_height_as_prop_of_y = prop_env

    def add_fixed_params(self, params: dict[str, Any] = dict()) -> dict[str, Any]:
        """Sets fixed parameters."""
        all_fixed_params = self.get_all_fixed_params()

        params = copy.copy(
            params
        )  # avoid deep copy to preserve reference to input layers
        for key, value in all_fixed_params.items():
            if key in params.keys() and value != params[key]:
                raise ValueError(
                    f"'{key}' parameter should not be passed, unless it is set to "
                    f"'{value}'."
                )
            params[key] = value

        return params

    def get_scale_x(self):
        """Get the x-scale of the environament in the x direction."""

        return self.scale_x  # type: ignore[attr-defined]

    def get_scale_y(self):
        """Get the y-scale of the environament in the x direction."""

        return self.scale_y  # type: ignore[attr-defined]

    @property
    def branch_y(self) -> float:
        if not hasattr(self, "_branch_y"):
            self._branch_y = (1 - self.arm_height_as_prop_of_y) * self.get_scale_y()
        return self._branch_y

    @property
    def left_T_end(self) -> list[float]:
        if not hasattr(self, "_left_T_end"):
            x_dim = self.stem_width_as_prop_of_x / 2 * self.get_scale_x()
            y_dim = (1 - self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._left_T_end = [x_dim, y_dim]
        return self._left_T_end

    @property
    def right_T_end(self) -> list[float]:
        if not hasattr(self, "_right_T_end"):
            x_dim = (1 - self.stem_width_as_prop_of_x / 2) * self.get_scale_x()
            y_dim = (1 - self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._right_T_end = [x_dim, y_dim]
        return self._right_T_end

    @property
    def T_ends(self) -> list[list[float]]:
        """Get the coordinates of the ends of the T-shape arms."""

        return [self.left_T_end, self.right_T_end]

    @property
    def T_start(self) -> list[float]:
        """Get the coordinates of the start of the T-shape."""

        if not hasattr(self, "_T_start"):
            x_dim = 0.5 * self.get_scale_x()
            y_dim = (self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._T_start = [x_dim, y_dim]

        return self._T_start

    @property
    def T_split(self) -> list[float]:
        """Get the coordinates of the split of the T branches."""

        if not hasattr(self, "_T_split"):
            x_dim = 0.5 * self.get_scale_x()
            y_dim = (1 - self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._T_split = [x_dim, y_dim]

        return self._T_split

    @property
    def top_arms_prop_of_area(self):
        bottom_stem_height = (1 - self.arm_height_as_prop_of_y) * self.get_scale_y()
        bottom_stem_width = self.stem_width_as_prop_of_x * self.get_scale_x()
        bottom_stem_area = bottom_stem_height * bottom_stem_width

        top_arms_height = self.arm_height_as_prop_of_y * self.get_scale_y()
        top_arms_width = self.get_scale_x()
        top_arms_area = top_arms_height * top_arms_width

        top_arms_prop_of_area = top_arms_area / (top_arms_area + bottom_stem_area)

        return top_arms_prop_of_area

    def sample_positions(self, n=10, method="uniform_jitter", area="both"):
        """Scatters 'n' locations across the environment.
        Args:
            n (int): number of features
            method: "uniform", "uniform_jittered" or "random" for how points are distributed

        Returns:
            array: (n x dimensionality) of positions
        """

        # sample from each area separately (top arms or bottom stem)
        if area == "both":
            n_top = int(np.around(self.top_arms_prop_of_area * n))
            n_bottom = int(n - n_top)
        elif area == "top":
            n_top = n
            n_bottom = 0
        elif area == "bottom":
            n_top = 0
            n_bottom = n
        else:
            raise ValueError(f"Unknown area: {area}")

        positions = list()
        adjusted_bottom_upper_limit = None  # for uniform sampling
        for area in ["top", "bottom"]:
            if area == "top":
                n = n_top
                extent_x = [0, self.get_scale_x()]
                extent_y = [
                    self.get_scale_y() * (1 - self.arm_height_as_prop_of_y),
                    self.get_scale_y(),
                ]
            elif area == "bottom":
                n = n_bottom
                extent_x = [
                    (0.5 - self.stem_width_as_prop_of_x / 2) * self.get_scale_x(),
                    (0.5 + self.stem_width_as_prop_of_x / 2) * self.get_scale_x(),
                ]
                extent_y = [0, self.get_scale_y() * (1 - self.arm_height_as_prop_of_y)]

            if method == "random":
                area_positions = np.zeros((n, 2))
                area_positions[:, 0] = np.random.uniform(*extent_x, size=n)
                area_positions[:, 1] = np.random.uniform(*extent_y, size=n)
            elif method[:7] == "uniform":
                area_size = (extent_x[1] - extent_x[0]) * (extent_y[1] - extent_y[0])

                delta = np.sqrt(area_size / n)

                if area == "top":
                    num_y_vals = min(1, int((extent_y[1] - extent_y[0]) // delta))
                    num_x_vals = int(n // num_y_vals)
                    delta_y = delta
                    delta_x = (extent_x[1] - extent_x[0]) / num_x_vals

                elif area == "bottom":
                    num_x_vals = min(1, int((extent_x[1] - extent_x[0]) // delta))
                    num_y_vals = int(n // num_x_vals)
                    delta_x = delta

                    if adjusted_bottom_upper_limit is not None:
                        delta_y = (adjusted_bottom_upper_limit - extent_y[0]) / (
                            num_y_vals + 0.5
                        )
                        extent_y[1] = adjusted_bottom_upper_limit - delta_y / 2
                    else:
                        delta_y = (extent_y[1] - extent_y[0]) / num_y_vals

                if num_x_vals < 2:
                    x = np.array([extent_x[0] + (extent_x[1] - extent_x[0]) / 2])
                else:
                    x = np.linspace(
                        extent_x[0] + delta_x / 2, extent_x[1] - delta_x / 2, num_x_vals
                    )

                if num_y_vals < 2:
                    y = np.array([extent_y[0] + (extent_y[1] - extent_y[0]) / 2])
                else:
                    y = np.linspace(
                        extent_y[0] + delta_y / 2, extent_y[1] - delta_y / 2, num_y_vals
                    )

                if area == "top":
                    adjusted_bottom_upper_limit = y[0]

                area_positions = np.array(np.meshgrid(x, y)).reshape(2, -1).T
                n_uniformly_distributed = area_positions.shape[0]
                if "jitter" in method:
                    delta_x = x[0] - extent_x[0]
                    area_positions[:, 0] += np.random.uniform(
                        -0.45 * delta_x, 0.45 * delta_x, n
                    )
                    delta_y = y[0] - extent_y[0]
                    area_positions[:, 1] += np.random.uniform(
                        -0.45 * delta_y, 0.45 * delta_y, n
                    )
                n_remaining = n - n_uniformly_distributed
                if n_remaining > 0:
                    positions_remaining = self.sample_positions(
                        n=n_remaining, method="random", area=area
                    )
                    area_positions = np.vstack((area_positions, positions_remaining))

            positions.append(area_positions)

        positions = np.vstack(positions)

        return positions

    def plot_environment(
        self,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the environment."""

        fig, ax = super().plot_environment(
            fig=fig, ax=ax, autosave=False, plot_objects=False, **kwargs
        )

        if ax is None:
            raise RuntimeError("ax is None.")

        ax.scatter(
            *self.T_start,
            marker=markers.MarkerStyle("^"),
            color="gold",
            s=20,
            zorder=5,
            label="start",
        )

        if len(self.objects):
            for object_coords in self.objects["objects"]:
                ax.scatter(
                    *object_coords,
                    marker=markers.MarkerStyle("o"),
                    color="blue",
                    s=18,
                    zorder=5,
                    label="target",
                )

        ax.scatter(
            *self.left_T_end,
            marker=markers.MarkerStyle("x"),
            color="red",
            s=18,
            zorder=5,
            label="reset",
        )
        ax.scatter(
            *self.right_T_end,
            marker=markers.MarkerStyle("x"),
            color="red",
            s=18,
            zorder=5,
        )
        ax.legend(loc="lower right", frameon=False)

        if fig is None:
            fig = ax.figure

        util.save_figure(fig, "Environment", save=autosave)

        return fig, ax


class OpenField(Environment, util.ParamsManagerMixin):
    """Open field environment to explore."""

    default_params = {
        "init_random_reward_obj": 1,
        "init_random_novel_obj": 5,
        "init_random_walls": 5,
        "init_random_teleport_pairs": 2,
        "wall_lengths": [0.1, 0.2],
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
        """Initialize the environment."""

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(self.params)

        if min(self.wall_lengths) <= 0:  # type: ignore[attr-defined]
            raise ValueError("Wall lengths must be positive.")

        self.num_teleport_pairs = 0

        if self.init_seed is None:  # type: ignore[attr-defined]
            self.rng = np.random
        else:
            self.rng = np.random.RandomState(self.init_seed)  # type: ignore[attr-defined,assignment]

        self.add_reward_objects(self.init_random_reward_obj)  # type: ignore[attr-defined]
        self.add_novel_objects(self.init_random_novel_obj)  # type: ignore[attr-defined]
        self.add_teleport_pairs(self.init_random_teleport_pairs)  # type: ignore[attr-defined]
        self.add_walls(self.init_random_walls)  # type: ignore[attr-defined]

    @property
    def object_df_columns(self):
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
        if not hasattr(self, "_object_df"):
            object_df = pd.DataFrame(columns=self.object_df_columns)
            self._object_df = object_df

        return self._object_df

    def get_new_teleport_pair_object_type_nums(
        self, first: int | None = None
    ) -> dict[str, int]:
        """Get object type numbers for a new teleport pair.

        Args:
            first (int): First object type number to use. If None, use the next
                available number. Defaults to None.

        Returns:
            object_type_nums (dict): Dictionary of object type numbers for the
                teleport pair.
        """

        if first is None:
            first = np.max(list(self.object_type_num_to_name_dict.keys())) + 1

        first = int(first)  # type: ignore[assignment]

        object_type_nums = {
            "in": first,
            "out": first + 1,
        }

        return object_type_nums

    def reset_object_type_dicts(self):
        """Reset the object type dictionaries."""

        dict_attr_names = [
            "_object_type_num_to_name_dict",
            "_type_num_to_plot_params_dict",
            "_teleport_pairs_dict",
        ]

        for dict_attr_name in dict_attr_names:
            if hasattr(self, dict_attr_name):
                delattr(self, dict_attr_name)

    @property
    def object_type_num_to_name_dict(self) -> dict[int, str]:
        """Dictionary for getting object type name from number."""

        if not hasattr(self, "_object_type_num_to_name_dict"):
            object_type_num_to_name_dict = {
                0: "reward",
                1: "novel",
            }

            for n in range(self.num_teleport_pairs):
                object_type_nums = self.get_new_teleport_pair_object_type_nums(
                    first=np.max(list(object_type_num_to_name_dict.keys())) + 1
                )
                for direction, i in object_type_nums.items():
                    object_type_num_to_name_dict[i] = f"teleport_{n}_{direction}"
            self._object_type_num_to_name_dict = object_type_num_to_name_dict

        return self._object_type_num_to_name_dict

    @property
    def type_name_to_num_dict(self) -> dict[str, int]:
        """Dictionary for getting object type number from name."""

        object_type_name_to_num_dict = {
            val: key for key, val in self.object_type_num_to_name_dict.items()
        }

        return object_type_name_to_num_dict

    @property
    def type_num_to_plot_params_dict(self) -> dict[int, dict[str, Any]]:
        """Dictionary for getting object type number from name."""

        if not hasattr(self, "_type_num_to_plot_params_dict"):
            teleport_nums = [
                val.replace("teleport_", "").replace("in_", "")
                for val in self.object_type_num_to_name_dict.values()
                if val.startswith("teleport") and "_in" in val
            ]
            teleport_vals = np.linspace(0.5, 1, len(teleport_nums))
            teleport_colors = plt.get_cmap("Oranges")(teleport_vals)  # type: ignore[callable]

            type_num_to_plot_params_dict = dict()
            for num, name in self.object_type_num_to_name_dict.items():
                if name == "reward":
                    type_num_to_plot_params_dict[num] = {
                        "name": name,
                        "marker": "o",
                        "color": "blue",
                        "s": 20,
                        "zorder": 5,
                    }
                elif name == "novel":
                    type_num_to_plot_params_dict[num] = {
                        "name": name,
                        "marker": "o",
                        "color": "green",
                        "s": 20,
                        "zorder": 5,
                    }
                elif name.startswith("teleport"):
                    direc = "in" if "_in" in name else "out"
                    teleport_num = int(
                        name.replace("teleport_", "").replace(f"_{direc}", "")
                    )
                    color = teleport_colors[teleport_num]
                    type_num_to_plot_params_dict[num] = {
                        "name": name,
                        "marker": self.get_teleport_pair_marker(
                            teleport_num, direction=direc
                        ),
                        "color": color,
                        "s": 20,
                        "zorder": 5,
                    }
                else:
                    raise ValueError(f"Unknown object type name: {name}")

            self._type_num_to_plot_params_dict = type_num_to_plot_params_dict

        return self._type_num_to_plot_params_dict

    def get_teleport_coords(self, teleport_pair_num, direction="in"):
        """Get the teleport coordinates for the given teleport pair.

        Args:
            teleport_pair_num (int): The teleport pair to get the coordinates for.
            direction (str, optional): The direction to get the coordinates for.
                Defaults to "in".

        Returns:
            np.ndarray: The teleport coordinates.
        """

        teleport_coords = self.teleport_pairs_dict[teleport_pair_num][direction][1]

        return teleport_coords

    def get_teleport_pair_orientation(self, teleport_pair_num: int = 1) -> str:
        """Get the orientation of a teleport pair.

        Args:
            teleport_pair_num (int): teleport pair number.

        Returns:
            str: orientation of the teleport pair.
        """

        if teleport_pair_num % 2 == 0:
            orientation = "vertical"
        else:
            orientation = "horizontal"

        return orientation

    def get_number_object_types_split(self) -> tuple[int, int, int]:
        """Get the number of each object type.

        Returns:
            tuple: number of novel, reward, and teleport objects.
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
            raise RuntimeError("Number of teleport pairs should be even.")

        return num_novel, num_reward, num_teleport

    def get_teleport_pair_marker(
        self, teleport_pair_num: int = 1, direction: str = "in"
    ) -> str:
        """Get the orientation of a teleport pair.

        Args:
            teleport_pair_num (int): teleport pair number.

        Returns:
            str: orientation of the teleport pair.
        """

        orientation = self.get_teleport_pair_orientation(teleport_pair_num)

        if orientation == "vertical":
            marker = "v" if direction == "in" else "^"
        else:
            marker = "<" if direction == "in" else ">"

        return marker

    def add_fixed_params(self, params: dict[str, Any] = dict()) -> dict[str, Any]:
        """Sets fixed parameters."""

        all_fixed_params = self.get_all_fixed_params()

        params = copy.copy(
            params
        )  # avoid deep copy to preserve reference to input layers
        for key, value in all_fixed_params.items():
            if key in params.keys() and value != params[key]:
                raise ValueError(
                    f"'{key}' parameter should not be passed, unless it is set to "
                    f"'{value}'."
                )

            params[key] = value

        return params

    def get_dist_from_coords_to_closest_object(
        self, coords: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        """Get the distance from a set of coordinates to the closest objects.

        Args:
            coords (np.ndarray): coordinates to get distance from.

        Returns:
            float: closest distance.
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
        """Get the distance from a set of coordinates to the closest wall.

        Args:
            coords (np.ndarray): coordinates to get distance from.

        Returns:
            float: closest distance.
        """

        if len(self.walls) == 0:
            return np.inf

        # returns points (1) x vectors x coords
        closest_dist = float(
            np.min(util.shortest_distances_from_points_to_lines(coords, self.walls))
        )

        return closest_dist

    def sample_coords(
        self, min_dist: float | None = None, max_attempts: int = 1000
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Sample coordinates situated at least min_dist from the closest
        object (half for walls).

        Args:
            min_dist (float, optional): minimum distance to closest object or
                wall. Defaults to None.
            max_attempts (int, optional): maximum number of attempts to sample
                valid coordinates. Defaults to 1000.

        Raises:
            ValueError: if could not sample valid coordinates after max_attempts
                attempts.

        Returns:
            coords (1d array): sampled coordinates [x, y].
        """

        if min_dist is None:
            min_dist = float(self.min_dist)  # type: ignore[attr-defined]

        i = 0
        while True:
            x = self.rng.uniform(self.extent[0], self.extent[1])
            y = self.rng.uniform(self.extent[2], self.extent[3])

            coords = np.array([x, y])

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
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Sample valid coordinates for the end of a wall given the start coordinates.

        Args:
            start_coords (1d array): start of wall.
            min_dist (float, optional): minimum distance to closest object.
                Defaults to None.

        Returns:
            end_coords (1d array): sampled end of wall coordinates [x, y].
                Returns None if could not sample valid end coordinates.
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
            end_coords = np.array(start_coords)  # new array
            end_coords[c] += wall_length * wall_direc

            # check that end_coords are within bounds
            if not self.check_if_position_is_in_environment(end_coords):
                end_coords = None

            # check that end_coords are far enough from objects, if there are any
            if end_coords is not None and len(self.objects["objects"]) != 0:
                closest_dist = np.min(
                    util.shortest_distances_from_points_to_lines(
                        self.objects["objects"], [start_coords, end_coords]
                    )
                )

                if closest_dist < min_dist:
                    end_coords = None

            if end_coords is not None:
                break

        return end_coords

    def add_object(
        self,
        object: np.ndarray[tuple[int], np.dtype[np.float64]],
        object_type: str | int = "new",
    ):
        super().add_object(object, type=object_type)  # type: ignore[arg-type]

        # add to object dataframe
        sub_df = self.object_df[self.object_df["object_type_num"] == object_type]
        if len(sub_df) == 0:
            idx_within_type = 0
        else:
            idx_within_type = sub_df["idx_within_type"].max() + 1

        object_type_name = self.object_type_num_to_name_dict[int(object_type)]

        new_object = {
            "object_type_num": object_type,
            "object_type_name": object_type_name,
            "idx_within_type": idx_within_type,
            "position_x": object[0],
            "position_y": object[1],
        }

        if "teleport" in object_type_name:
            _, teleport_pair_num, teleport_direction = object_type_name.split("_")
            new_object["teleport_pair_num"] = int(teleport_pair_num)
            new_object["teleport_direction"] = teleport_direction

        self.object_df.loc[len(self.object_df)] = new_object  # type: ignore[attr-defined]

    def add_reward_objects(self, num: int = 1, coords=None):
        """Add reward objects.

        Args:
            num (int, optional): number of reward objects to add. Defaults to 1.
        """

        reward_type = self.type_name_to_num_dict["reward"]

        if coords is not None:
            num = len(coords)

        for n in range(num):
            if coords is None:
                coord = self.sample_coords()
            else:
                coord = np.asarray(coords[n], dtype=np.float64).reshape(2)
                self.check_if_position_is_in_environment(coord)
            self.add_object(coord, object_type=reward_type)

        if num > 0:
            self.reset_object_type_dicts()

    def add_novel_objects(self, num: int = 1, coords=None):
        """Add novel objects.

        Args:
            num (int, optional): number of novel objects to add. Defaults to 1.
        """

        novel_type = self.type_name_to_num_dict["novel"]

        if coords is not None:
            num = len(coords)

        for n in range(num):
            if coords is None:
                coord = self.sample_coords()
            else:
                coord = np.asarray(coords[n], dtype=np.float64).reshape(2)
                self.check_if_position_is_in_environment(coord)

            self.add_object(coord, object_type=novel_type)

        if num > 0:
            self.reset_object_type_dicts()

    def add_teleport_pairs(self, num: int = 1, coord_pairs=None):
        """Add teleport pairs (directional).

        Args:
            num (int, optional): number of teleport pairs to add. Defaults to 1.
        """

        if coord_pairs is not None:
            num = len(coord_pairs)

        def format_teleport_pair(coord_pair):
            try:
                coords_in, coords_out = coord_pair
            except ValueError as err:
                if "values to unpack" in str(err):
                    raise ValueError(
                        f"Expected two coordinates per teleport pair, but got {len(coords)}."
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
            object_type_nums = self.get_new_teleport_pair_object_type_nums()
            self.num_teleport_pairs += 1
            self.reset_object_type_dicts()  # within loop, so that teleport pair object types are not reused
            if coord_pairs is not None:
                coord_pair = format_teleport_pair(coord_pairs[n])
            for o, object_type_num in enumerate(object_type_nums.values()):
                if coord_pairs is None:
                    coords = self.sample_coords()
                else:
                    coords = coord_pair[o]
                self.add_object(coords, object_type=object_type_num)

    @property
    def teleport_pairs_dict(self) -> dict[int, dict[str, tuple[int, list[float]]]]:
        """Returns dictionary of teleport pairs (directional)."""

        if not hasattr(self, "_teleport_pairs_dict"):
            teleport_pairs_dict = dict()
            for name, object_type in self.type_name_to_num_dict.items():
                if name.startswith("teleport_") and "in" in name:
                    object_type_in = object_type
                    teleport_pair = int(
                        name.replace("teleport_", "").replace("_in", "")
                    )
                    out_key = f"teleport_{teleport_pair}_out"
                    if out_key not in self.type_name_to_num_dict.keys():
                        raise RuntimeError(
                            f"Teleport in {teleport_pair} does not have 'out' pair."
                        )
                    object_type_out = self.type_name_to_num_dict[out_key]

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

    def check_if_walls_ends_too_close(
        self,
        new_wall_coords: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        | list[list[float]],
        min_dist=None,
    ) -> bool:
        """
        Checks whether a new wall's ends is too close to the ends of existing
        walls.

        Specifically checks whether an end of the new wall intersects at less
        than 45 degrees near the end of an existing wall, forming an V shape
        with small ends sticking out. If so, returns True, else False.

        Does NOT check whether the new wall overlaps exactly with an existing
        wall, or intersects near the middle of either wall.

        Args:
            new_wall_coords (list or 2D array): coordinates of new wall,
                with dims [[x1, y1], [x2, y2]]
            min_dist (float, optional): minimum distance between walls.
                Defaults to None.

        Returns:
            bool: True if the ends of a new wall are too close to an existing wall,
                else False.
        """

        if len(self.walls) == 0:
            return False

        if min_dist is None:
            min_dist = float(self.min_dist)  # type: ignore[attr-defined]

        new_wall = np.asarray(new_wall_coords)

        for wall in self.walls:
            # get angle between two vectors
            angle = util.get_angle_between_vectors(
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
                    return True

        return False

    def add_walls(self, num: int = 1, max_attempts: int = 1000):
        """Add walls.

        Checks that walls are not too close to objects and that they do not
        overlap too much with one another.

        Does NOT check whether new wall creates a hole.

        Args:
            num (int, optional): number of walls to add. Defaults to 1.
            max_attempts (int, optional): maximum number of attempts to sample
                valid wall start and end coordinates. Defaults to 1000.

        Raises:
            ValueError: if could not sample valid wall start and end coordinates
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

    def plot_environment(
        self,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        plot_objects: bool = True,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the environment.

        Args:
            fig (matplotlib figure, optional): figure to plot on. Defaults to None.
            ax (matplotlib axis, optional): axis to plot on. Defaults to None.
            plot_objects (bool, optional): whether to plot objects. Defaults to True.
            no_legend (bool, optional): whether to remove legend. Defaults to False.
            autosave (bool, optional): whether to save the plot. Defaults to None.

        Returns:
            fig (matplotlib figure): figure with environment plotted.
            ax (matplotlib axis): axis with environment plotted.
        """

        if ax is None:
            env_width = self.extent[1] - self.extent[0]
            add_x = 0
            if plot_objects:
                add_x = 3 * env_width  # for legend and labels
            fig, ax = plt.subplots(
                figsize=(3 * env_width + add_x, 3 * (self.extent[3] - self.extent[2]))
            )

        fig, ax = super().plot_environment(
            fig=fig, ax=ax, autosave=False, plot_objects=False, **kwargs
        )

        if ax is None:
            raise RuntimeError("ax is None.")

        if plot_objects:
            type_num_to_plot_params_dict = copy.deepcopy(
                self.type_num_to_plot_params_dict
            )
            for coords, object_type in zip(
                self.objects["objects"], self.objects["object_types"]
            ):
                label = None
                if "name" in type_num_to_plot_params_dict[object_type].keys():
                    label = type_num_to_plot_params_dict[object_type].pop("name")
                ax.scatter(
                    *coords,
                    **type_num_to_plot_params_dict[object_type],
                    label=label,
                )

            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        legend = ax.get_legend()
        if no_legend and legend is not None:
            legend.remove()

        if fig is None:
            fig = ax.figure

        util.save_figure(fig, "Environment", save=autosave)

        return fig, ax
