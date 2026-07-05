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
from ratinabox import MOUNTAIN_PLOT_WIDTH_MM, FIGURE_INCH_PER_ENVIRONMENT_METRE

from predhpc.util import gen_util, trig_util, plot_util, ext_util


class EnvironmentWarning(UserWarning):
    """Class is for environment-related user warnings."""

    pass


warnings.simplefilter("once", EnvironmentWarning)


class Environment(riabEnv, ext_util.ParamsManagerMixin):
    """
    Environment()

    Class extending the ratinabox environment.

    default_params = {
        "dft_object_type_name": "object",
    }


    List of attributes (in addition to ratinabox.Environment attributes):
        • self.object_type_name

    List of methods (in addition to ratinabox.Environment methods):
        • self.format_position()
        • self.get_environment_figsize()
        • self.get_object_label()
        • self.add_object_to_plot()
        • self.plot_environment()
    """

    default_params = {
        "dft_object_type_name": "object",
    }

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

    @property
    def object_type_name(self):
        """
        self.object_type_name

        Obtain the name of the object.
        """

        self._object_type_name = self.dft_object_type_name

        return self._object_type_name

    def format_position(
        self,
        position: (
            np.ndarray[tuple[int], np.dtype[np.float64]] | list[float] | None
        ) = None,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """
        self.format_position()

        Obtain formatted position. If input position is None, None is returned.

        Args:
        - position (1D np.ndarray or list, optional): Position to format.
            Default is None.

        Raises:
        - ValueError: If position is not within the environment extent.

        Returns:
        - position (1D np.ndarray or None): Formatted position. If input position is
            None, None is returned.
        """

        if position is not None:
            position = np.asarray(position).ravel()
            if len(position) != self.D:
                raise ValueError(f"Positions must comprise exactly {self.D} value(s).")
            position = position.reshape(self.D)

            # [min, max] or [left, right, bottom, top]
            extent = self.extent

            if self.D == 1:
                if position < extent[0] or position > extent[1]:
                    raise ValueError(
                        "Position must be within the environment extent: " f"{extent}."
                    )
            elif self.D == 2:
                if position[0] < extent[0] or position[0] > extent[1]:
                    raise ValueError(
                        "First position value must be within the "
                        f"environment extent: {extent[:2]}."
                    )
                if position[1] < extent[2] or position[1] > extent[3]:
                    raise ValueError(
                        "Second position value must be within the "
                        f"environment extent: {extent[2:]}."
                    )

            else:
                raise ValueError(
                    "Expected environment dimensionality to be 1 or 2. "
                    f"Got {self.D}."
                )

        return position

    def get_environment_figsize(self, size_factor=1.0):
        """
        self.get_environment_figsize()

        Obtain the figure size for a 2D environment.

        Args:
        - size_factor (float, optional): Size factor by which to expand the environment
            figure size in each dimension. Default is 1.0.

        Raises:
        - RuntimeError: If the environment dimensionality is not 2D.

        Returns:
        - figsize (tuple): Figure size for the environment.
        """

        extent = self.extent

        if self.D == 1:
            base_width = MOUNTAIN_PLOT_WIDTH_MM / 25 * (extent[1] - extent[0])
            base_height = 1
        else:
            base_width = FIGURE_INCH_PER_ENVIRONMENT_METRE * (extent[1] - extent[0])
            base_height = FIGURE_INCH_PER_ENVIRONMENT_METRE * (extent[3] - extent[2])

        figsize = (size_factor * base_width, size_factor * base_height)

        return figsize

    def get_object_label(self, name="target"):
        """
        self.get_object_label()

        Obtain the label for an object for plotting.

        Args:
        - name (str, optional): Name of the object. Default is "target".

        Returns:
        - label (str): Label for the object for plotting.
        """

        label = name.replace("_", " ")

        return label

    def add_object_to_plot(self, sub_ax, s=10, alpha=0.8, zorder=2):
        """
        self.add_object_to_plot()

        Method to add an object to the plot. To be implemented by child classes.
        """

        object_cmap = mpl_cmap[self.object_colormap]
        for i, coords in enumerate(self.objects["objects"]):
            object_color = object_cmap(
                self.objects["object_types"][i] / (self.n_object_types - 1 + 1e-8)
            )
            y = 0 if self.D == 1 else coords[1]
            sub_ax.scatter(
                coords[0],
                y,
                facecolor=[0, 0, 0, 0],
                edgecolors=object_color,
                s=s,
                zorder=zorder,
                marker="o",
                alpha=alpha,
            )

    def plot_environment(
        self,
        fig=None,
        ax=None,
        return_env_fig=False,
        s=10,
        alpha=0.8,
        zorder=2,
        title=None,
        size_factor=1.0,
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
        - zorder (int, optional): Z-order for the plotted objects. Default is 2.
        - title (str, optional): Title for the plot. Default is None.
        - size_factor (float, optional): Size factor by which to expand the environment
            figure size in each dimension, if creating new subplot. Default is 1.0.
        - plot_objects (bool, optional): Whether to plot the objects. Default is True.
        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Environment.plot_environment().

        Returns:
        if return_fig:
        - fig (mpl_figure.Figure): Figure with environment plotted

        - sub_ax (plt.Axes): Subplot with environment plotted.
        """

        kwargs = plot_util.organize_fig_ax_kwargs(fig=fig, ax=ax, **kwargs)

        if "ax" not in kwargs.keys():
            figsize = self.get_environment_figsize(size_factor)
            kwargs["fig"], kwargs["ax"] = plt.subplots(figsize=figsize)

        fig, sub_ax = super().plot_environment(plot_objects=False, **kwargs)

        if plot_objects:
            self.add_object_to_plot(sub_ax, s=s, alpha=alpha, zorder=zorder)

        if title is not None:
            sub_ax.set_title(title)

        if return_env_fig:
            return fig, sub_ax
        else:
            return sub_ax


class LinearResetEnv(Environment):
    """
    LinearResetEnv()

    Ratinabox linear reset environment. Accomodates only a single object, which
    can be moved to different positions in the environment. The start and reset
    positions are fixed.

    A parameters dictionary can be passed at initialisation.

    default_params = {
        "scale": 6,
        "boundary_conditions": "periodic",  # solid vs periodic
        "start_prop": 0.005,  # start position from left, as proportion of environment length
        "reset_prop": 0.005,  # reset position from right, as proportion of environment length
        "init_env_object_prop": 0.6,
        "dft_object_type_name": "landmark",
    }

    List of attributes (in addition to Environment attributes):
        • self.start_position
        • self.reset_position
        • self.env_object

    List of methods (in addition to Environment methods):
        • self.set_env_object()
        • self.init_objects()
        • self.add_object()
        • self.get_objects_to_plot()
        • self.add_start_and_reset_to_plot()
        • self.add_objects_to_plot()
        • self.plot_environment()
    """

    default_params = {
        "scale": 6,
        "boundary_conditions": "periodic",  # solid vs periodic
        "start_prop": 0.005,  # start position from left, as proportion of environment length
        "reset_prop": 0.005,  # reset position from right, as proportion of environment length
        "init_env_object_prop": 0.6,
        "dft_object_type_name": "landmark",
    }

    ignored_param_keys = ["boundary", "aspect"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "1D",  # 1D or 2D environment
        "holes": list(),  # no holes
        "objects": list(),  # no objects
    }

    def __init__(self, params=dict()):
        """
        LinearResetEnv()

        Initialise a linear reset environment.

        Args:
        - params (dict, optional): Environment parameters. Default is dict().
        """

        for problem_keys in ["start_position", "env_object", "reset_position"]:
            if problem_keys in params:
                raise ValueError(
                    f"Cannot specify '{problem_keys}' in params. "
                    "Use 'start_prop', 'init_env_object_prop', and 'reset_prop' instead."
                )

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        super().__init__(params=params)

        self.init_objects()

    @property
    def start_position(self) -> float:
        """
        self.start_position

        Obtain the start position in the linear reset environment.

        Returns:
        - (float): Start marker position in the environment.
        """

        if not hasattr(self, "_start_position"):
            if self.start_prop < 0 or self.start_prop > 1:
                raise ValueError("self.start_prop must be between 0 and 1.")

            start_position = self.start_prop * self.scale
            self._start_position = self.format_position(start_position)

        return self._start_position

    @property
    def reset_position(self) -> float:
        """
        self.reset_position

        Obtain the reset position in the linear reset environment.

        Returns:
        - (float): Reset position in the environment.
        """

        if not hasattr(self, "_reset_position"):
            if self.reset_prop < 0 or self.reset_prop > 1:
                raise ValueError("self.reset_prop must be between 0 and 1.")

            reset_position = (1 - self.reset_prop) * self.scale
            self._reset_position = self.format_position(reset_position)

        return self._reset_position

    @property
    def env_object(self) -> float:
        """
        self.env_object

        Obtain the environment object in the linear reset environment.

        Returns:
        - (float): environment object in the environment.
        """

        if not hasattr(self, "_env_object"):
            self.set_env_object(env_object_prop=self.init_env_object_prop)

        return self._env_object

    def _update_objects(self):
        """
        self.update_objects()

        Update the objects dictionary in the linear reset environment based on the
        current environment object position. Excludes reset and start objects.

        If the object is removed, the coordinates are set to NaNs. This avoids problems
        when using with ObjectCells.

        Sets attributes:
        - self.objects: Dictionary containing the objects in the environment, with
            keys "objects" and "object_types".
        - self.n_object_types: Number of unique object types in the environment.
        """

        if len(self.objects["objects"]) == 1:
            if self.env_object is None:
                self.objects["objects"][0, :] = np.nan
            else:
                self.objects["objects"][0, :] = self.env_object[:]

        elif not len(self.objects["objects"]):
            self.objects["objects"] = np.asarray(self.env_object).reshape(1, -1)
            self.objects["object_types"] = (np.asarray([0]).reshape(1, 1),)

        else:
            raise RuntimeError(
                "Expected self.objects to contain either 0 or 1 objects, but found "
                f"{len(self.objects['objects'])}."
            )

        self.n_object_types = len(np.unique(self.objects["object_types"]))

    def set_env_object(
        self,
        env_object: float | None = None,
        env_object_prop: float | None = None,
    ):
        """
        self.set_env_object()

        Set the environment object position of the linear reset environment.

        Args:
        - env_object_prop (float, optional): Environment object position as a proportion
            of the environment length. Default is 0.6.
        - env_object (float): Environment object position in the environment.
            If provided, this is used instead of env_object_prop. Default is None.
        """

        if env_object is None:
            if env_object_prop is None:
                env_object = None
            elif env_object_prop < 0 or env_object_prop > 1:
                raise ValueError("env_object_prop must be between 0 and 1.")
            else:
                env_object = env_object_prop * self.scale
        else:
            if env_object_prop is not None:
                raise ValueError("Cannot specify both env_object and env_object_prop.")
            if env_object < 0 or env_object > self.scale:
                raise ValueError(
                    "env_object must be between 0 and environment scale "
                    f"{self.scale}, but got {env_object}."
                )

        env_object = self.format_position(env_object)
        if env_object is not None:
            env_object = env_object.copy()

        self._env_object = env_object

        self._update_objects()

    def init_objects(self):
        """
        self.init_objects()

        Initialise the objects in the environment. Sets the start and reset object
        properties, ensuring they are properly defined.
        """

        self.start_position
        self.reset_position
        self.env_object

    def add_object(self, *args, **kwargs):
        """
        Method to prevent user from trying to add objects to the environment.
        """

        raise NotImplementedError("Objects cannot be added to this environment.")

    def get_objects_to_plot(self) -> dict[str, list[float]]:
        """
        self.objects

        Obtain the objects to plot in the linear reset environment, as well as start
        and reset positions.

        Returns:
        - objects_to_plot (dict): Objects in the environment, with keys
            "start", "reset" and self.object_type_name.
        """

        objects_to_plot = {
            "start": self.start_position,
            "reset": self.reset_position,
            self.object_type_name: self.env_object,
        }

        return objects_to_plot

    def add_start_and_reset_to_plot(
        self,
        sub_ax: plt.Axes,
        base_s: float = 15,
        base_lw: float = 1.0,
        alpha: float = 0.8,
        zorder: int = 5,
        height: float = 0,
        no_legend: bool = False,
    ):
        """
        self.add_start_and_reset_to_plot()

        Adds the start and reset locations to a T-maze environment plot.

        Args:
        - sub_ax (plt.Axes): Subplot to plot on.
        - base_s (float, optional): Base size for the plotted objects. Default is 15.
        - base_lw (float, optional): Base linewidth for the plotted objects.
            Default is 1.
        - alpha (float, optional): Alpha value for the plotted objects. Default is 0.8.
        - zorder (int, optional): Z-order for the plotted objects. Default is 5.
        - height (float, optional): Height at which to plot the start and reset markers.
            Default is 0.
        - no_legend (bool, optional): Whether to skip plotting the legend.
            Default is False.
        """

        pos_dict = {"start": self.start_position, "reset": self.reset_position}

        for pos_name, pos in pos_dict.items():
            if pos is None:
                continue

            plot_kwargs = plot_util.get_plot_marker_kwargs(
                pos_name, base_s=base_s, base_lw=base_lw
            )
            sub_ax.scatter(
                pos,
                height,
                zorder=zorder,
                label=self.get_object_label(pos_name),
                alpha=alpha,
                **plot_kwargs,
            )

        if not no_legend:
            sub_ax.legend(ncol=2, frameon=False)

    def add_objects_to_plot(
        self,
        sub_ax: plt.Axes,
        base_s: float = 15,
        base_lw: float = 1.0,
        alpha: float = 1.0,
        zorder: int = 5,
        height: float = 0,
        incl_start_reset: bool = True,
        no_legend: bool = False,
    ):
        """
        self.add_objects_to_plot()

        Plot the objects in the linear reset environment. Also plots the start and
        reset locations.

        Args:
        - sub_ax (plt.Axes): Subplot to plot on.
        - base_s (float, optional): Base size for the plotted objects. Default is 15.
        - base_lw (float, optional): Base linewidth for the plotted objects.
            Default is 1.
        - alpha (float, optional): Alpha value for the plotted objects. Default is 1.0.
        - zorder (int, optional): Z-order for the plotted objects. Default is 5.
        - height (float, optional): Height at which to plot the start and reset markers.
            Default is 0.
        - incl_start_reset (bool, optional): If True, start and reset positions are
            plotted too. Default is True,
        - no_legend (bool, optional): Whether to skip plotting the legend.
            Default is False.
        """

        ncol = 0
        if incl_start_reset:
            self.add_start_and_reset_to_plot(
                sub_ax,
                base_s=base_s,
                base_lw=base_lw,
                alpha=alpha,
                zorder=zorder,
                no_legend=True,
            )
            ncol += 2

        if len(self.objects["objects"]):
            object_kwargs = plot_util.get_plot_marker_kwargs(
                self.object_type_name, base_s=base_s, base_lw=base_lw
            )
            for object_coords in self.objects["objects"]:
                if not np.isfinite(object_coords).all():
                    continue

                sub_ax.scatter(
                    object_coords,
                    height,
                    zorder=zorder,
                    label=self.get_object_label(self.object_type_name),
                    alpha=alpha,
                    **object_kwargs,
                )
                ncol += 1

        if not no_legend and ncol:
            sub_ax.legend(ncol=ncol, frameon=False)

    def plot_environment(
        self,
        fig: mpl_figure.Figure | None = None,
        sub_ax: plt.Axes | None = None,
        return_env_fig: bool = False,
        no_legend: bool = False,
        plot_objects: bool = True,
        minimalist: bool = False,
        size_factor: bool = 1.0,
        base_s: float = 15,
        base_lw: float = 1.0,
        alpha: float = 1.0,
        zorder: int = 5,
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
        - plot_objects (bool, optional): Whether to plot the objects, as well as start
            and reset positions. Default is True.
        - minimalist (bool, optional): Whether to create minimalist reset environment
            plot. Default is False.
        - size_factor (bool, optional): Size factor by which to expand the environment
            figure size in each dimension. Default is 1.0.
        - base_s (float, optional): Base size for the plotted objects. Default is 15.
        - base_lw (float, optional): Base linewidth for the plotted objects.
            Default is 1.
        - alpha (float, optional): Alpha value for the plotted objects. Default is 1.0.
        - zorder (int, optional): Z-order for the plotted objects. Default is 5.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Environment.plot_environment().

        Returns:
        if return_fig:
        - fig (mpl_figure.Figure): Figure with environment plotted

        - sub_ax (plt.Axes): Subplot with environment plotted.
        """

        kwargs = plot_util.organize_fig_ax_kwargs(fig=fig, sub_ax=sub_ax, **kwargs)

        if minimalist and "ax" not in kwargs.keys():
            figsize = (4 * size_factor, 0.7 * size_factor)
            kwargs["fig"], kwargs["ax"] = plt.subplots(figsize=figsize)

        sub_ax = super().plot_environment(
            autosave=False, plot_objects=False, size_factor=size_factor, **kwargs
        )

        if plot_objects:
            self.add_objects_to_plot(
                sub_ax,
                base_s=base_s,
                base_lw=base_lw,
                alpha=alpha,
                zorder=zorder,
                incl_start_reset=True,
                no_legend=no_legend,
            )

        if minimalist:
            edges = np.asarray([self.start_position, self.reset_position]).ravel()
            sub_ax.spines["bottom"].set_bounds(*sorted(list(edges)))
            plot_util.pad_axis(sub_ax, axis="x", pad_prop=0.05)
            sub_ax.set_xticks(list())
            sub_ax.set_xlabel("")
            sub_ax.spines["bottom"].set_linewidth(1.5)

        fig = sub_ax.figure
        plot_util.save_figure(fig, "Environment", save=autosave)

        if return_env_fig:
            return fig, sub_ax
        else:
            return sub_ax


class TEnv(Environment):
    """
    TEnv()

    Ratinabox T-shaped environment. Accomodates only a single environment object, which
    can be moved to different positions in the environment. The one start and two
    reset positions are fixed.

    A parameters dictionary can be passed at initialisation.

    default_params = {
        "width_prop_env": 0.2,  # T-shape arms and stem width (prop of env dims)
        "scale_x": 1,  # env width
        "scale_y": 1,  # env height
        "stem_width_as_prop_of_x": None,  # T-shape stem width (prop of env width)
        "arm_height_as_prop_of_y": None,  # T-shape arms width (prop of env height)
        "dft_object_type_name": "landmark",
    }

    List of attributes (in addition to Environment attributes):
        • self.stem_left
        • self.stem_right
        • self.branch_y
        • self.start_position
        • self.left_reset_position
        • self.right_reset_position
        • self.reset_position
        • self.T_split
        • self.T_split_top
        • self.top_arms_prop_of_area
        • self.env_object

    List of methods (in addition to Environment methods):
        • self.set_env_object()
        • self.init_objects()
        • self.add_object()
        • self.get_scale_x()
        • self.get_scale_y()
        • self.get_area()
        • self.get_T_extents()
        • self.sample_positions()
        • self.get_objects_to_plot()
        • self.add_start_and_reset_to_plot()
        • self.add_objects_to_plot()
        • self.plot_environment()
    """

    default_params = {
        "width_prop_env": 0.2,  # T-shape arms and stem width (prop of env dims)
        "scale_x": 1,  # env width
        "scale_y": 1,  # env height
        "stem_width_as_prop_of_x": None,  # T-shape stem width (prop of env width)
        "arm_height_as_prop_of_y": None,  # T-shape arms width (prop of env height)
        "dft_object_type_name": "landmark",
    }

    ignored_param_keys = ["boundary", "scale", "aspect"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "holes": list(),  # no holes
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
            width_prop_env=self.params["width_prop_env"],
            scale_x=self.params["scale_x"],
            scale_y=self.params["scale_y"],
            stem_width_as_prop_of_x=self.params["stem_width_as_prop_of_x"],
            arm_height_as_prop_of_y=self.params["arm_height_as_prop_of_y"],
        )

        super().__init__(self.params)

        self.scale = (self.scale_x + self.scale_y) / 2  # type: ignore[attr-defined]

        width_prop_env = self.width_prop_env  # type: ignore[attr-defined]
        if self.stem_width_as_prop_of_x is None:  # type: ignore[attr-defined,has-type]
            self.stem_width_as_prop_of_x = width_prop_env

        if self.arm_height_as_prop_of_y is None:  # type: ignore[attr-defined,has-type]
            self.arm_height_as_prop_of_y = width_prop_env

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
    def start_position(self) -> list[float]:
        """
        self.start_position

        Obtain the coordinates of the start position at the base of the T-shape.

        Returns:
        - (list): Coordinates of the start position at the base of the T-shape.
        """

        if not hasattr(self, "_start_position"):
            x_dim = 0.5 * self.get_scale_x()
            y_dim = (self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._start_position = [x_dim, y_dim]

        return self._start_position

    @property
    def left_reset_position(self) -> list[float]:
        """
        self.left_reset_position

        Obtain the coordinates of the reset position for the left arm of the T-shape.

        Returns:
        - (list): Coordinates of the reset position for the left arm of the T-shape.
        """

        if not hasattr(self, "_left_reset_position"):
            x_dim = self.stem_width_as_prop_of_x / 2 * self.get_scale_x()
            y_dim = (1 - self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._left_reset_position = [x_dim, y_dim]
        return self._left_reset_position

    @property
    def right_reset_position(self) -> list[float]:
        """
        self.right_reset_position

        Obtain the coordinates of the reset position for the right arm of the T-shape.

        Returns:
        - (list): Coordinates of the reset position for the right arm of the T-shape.
        """

        if not hasattr(self, "_right_reset_position"):
            x_dim = (1 - self.stem_width_as_prop_of_x / 2) * self.get_scale_x()
            y_dim = (1 - self.arm_height_as_prop_of_y / 2) * self.get_scale_y()
            self._right_reset_position = [x_dim, y_dim]
        return self._right_reset_position

    @property
    def reset_position(self) -> list[list[float]]:
        """
        self.reset_position

        Obtain the coordinates of the reset positions for the T-shape arms.

        Returns:
        - (list): Coordinates of the reset positions for the T-shape arms [left, right].
        """

        return [self.left_reset_position, self.right_reset_position]

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

    @property
    def env_object(self) -> float:
        """
        self.env_object

        Obtain the environment object in the linear reset environment.

        Returns:
        - (float): environment object in the environment.
        """

        if not hasattr(self, "_env_object"):
            self._env_object = None

        return self._env_object

    def _update_objects(self):
        """
        self.update_objects()

        Update the objects dictionary in the linear reset environment based on the
        current environment object position. Excludes reset and start objects.

        If the object is removed, the coordinates are set to NaNs. This avoids problems
        when using with ObjectCells.

        Sets attributes:
        - self.objects: Dictionary containing the objects in the environment, with
            keys "objects" and "object_types".
        - self.n_object_types: Number of unique object types in the environment.
        """

        if len(self.objects["objects"]) == 1:
            if self.env_object is None:
                self.objects["objects"][0, :] = np.nan
            else:
                self.objects["objects"][0, :] = self.env_object[:]

        elif not len(self.objects["objects"]):
            self.objects["objects"] = np.asarray(self.env_object).reshape(1, -1)
            self.objects["object_types"] = (np.asarray([0]).reshape(1, 1),)

        else:
            raise RuntimeError(
                "Expected self.objects to contain either 0 or 1 objects, but found "
                f"{len(self.objects['objects'])}."
            )

        self.n_object_types = len(np.unique(self.objects["object_types"]))

    def set_env_object(self, position: np.ndarray | None):
        """
        self.set_set_env_object()

        Set the environment object position of the T-maze.

        Args:
        - position (1D np.ndarray or None): Environment object position in the
            environment.
        """

        env_object = self.format_position(position)
        if env_object is not None:
            env_object = env_object.copy()

        self._env_object = env_object

        self._update_objects()

    def init_objects(self):
        """
        self.init_objects()

        Initialise the objects in the environment. Sets the start and reset object
        properties, ensuring they are properly defined.
        """

        self.start_position
        self.reset_position
        self.env_object

    def add_object(self, *args, **kwargs):
        """
        Method to prevent user from trying to add objects to the environment.
        """

        raise NotImplementedError("Objects cannot be added to this environment.")

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

    def get_objects_to_plot(self) -> dict[str, list[float]]:
        """
        self.objects

        Obtain the objects to plot in the linear reset environment.

        Returns:
        - objects_to_plot (dict): Objects in the environment, with keys
            "start", "reset" and self.object_type_name.
        """

        objects_to_plot = {
            "start": self.start_position,
            "reset": self.reset_position,
            self.object_type_name: self.env_object,
        }

        return objects_to_plot

    def add_start_and_reset_to_plot(
        self,
        sub_ax: plt.Axes,
        base_s: float = 15,
        base_lw: float = 1.0,
        alpha: float = 0.8,
        zorder: int = 5,
        no_legend: bool = False,
    ):
        """
        self.add_start_and_reset_to_plot()

        Adds the start and reset locations to a T-maze environment plot.

        Args:
        - sub_ax (plt.Axes): Subplot to plot on.
        - base_s (float, optional): Base size for the plotted objects. Default is 15.
        - base_lw (float, optional): Base linewidth for the plotted objects.
            Default is 1.
        - alpha (float, optional): Alpha value for the plotted objects. Default is 0.8.
        - zorder (int, optional): Z-order for the plotted objects. Default is 5.
        - no_legend (bool, optional): Whether to skip plotting the legend.
            Default is False.
        """

        start_kwargs = plot_util.get_plot_marker_kwargs(
            "start", base_s=base_s, base_lw=base_lw
        )
        sub_ax.scatter(
            *self.start_position,
            zorder=zorder,
            label="start",
            alpha=alpha,
            **start_kwargs,
        )

        reset_kwargs = plot_util.get_plot_marker_kwargs(
            "reset", base_s=base_s, base_lw=base_lw
        )
        for reset_position in [self.left_reset_position, self.right_reset_position]:
            sub_ax.scatter(
                *reset_position,
                zorder=zorder,
                label="reset",
                alpha=alpha,
                **reset_kwargs,
            )

            if not no_legend:
                sub_ax.legend(loc="lower right", frameon=False)

    def add_objects_to_plot(
        self,
        sub_ax: plt.Axes,
        base_s: float = 15,
        base_lw: float = 1.0,
        alpha: float = 0.8,
        zorder: int = 5,
        incl_start_reset: bool = True,
        no_legend: bool = False,
    ):
        """
        self.add_objects_to_plot()

        Adds objects to a T-maze environment plot.

        Args:
        - sub_ax (plt.Axes): Subplot to plot on.
        - base_s (float, optional): Base size for the plotted objects. Default is 15.
        - base_lw (float, optional): Base linewidth for the plotted objects.
            Default is 1.
        - alpha (float, optional): Alpha value for the plotted objects. Default is 0.8.
        - zorder (int, optional): Z-order for the plotted objects. Default is 5.
        - incl_start_reset (bool, optional): If True, start and reset positions are
            plotted too. Default is True,
        - no_legend (bool, optional): Whether to skip plotting the legend.
            Default is False.
        """

        plotted = False
        if incl_start_reset:
            self.add_start_and_reset_to_plot(
                sub_ax,
                base_s=base_s,
                base_lw=base_lw,
                alpha=alpha,
                zorder=zorder,
                no_legend=True,
            )
            plotted = True

        if len(self.objects["objects"]):
            object_kwargs = plot_util.get_plot_marker_kwargs(
                self.object_type_name, base_s=base_s, base_lw=base_lw
            )
            for object_coords in self.objects["objects"]:
                if not np.isfinite(object_coords).all():
                    continue

                sub_ax.scatter(
                    *object_coords,
                    zorder=zorder,
                    label=self.get_object_label(self.object_type_name),
                    alpha=alpha,
                    **object_kwargs,
                )
                plotted = True

        if not no_legend and plotted:
            sub_ax.legend(loc="lower right", frameon=False)

    def plot_environment(
        self,
        fig: mpl_figure.Figure | None = None,
        sub_ax: plt.Axes | None = None,
        return_env_fig: bool = False,
        no_legend: bool = False,
        plot_objects: bool = True,
        size_factor: bool = 1.0,
        base_s: float = 15,
        base_lw: float = 1.0,
        alpha: float = 0.8,
        zorder: int = 5,
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
        - plot_objects (bool, optional): Whether to plot the objects as well as start
            and reset positions. Default is True.
        - size_factor (bool, optional): Size factor by which to expand the environment
            figure size in each dimension. Default is 1.0.
        - base_s (float, optional): Base size for the plotted objects. Default is 15.
        - base_lw (float, optional): Base linewidth for the plotted objects.
            Default is 1.
        - alpha (float, optional): Alpha value for the plotted objects. Default is 0.8.
        - zorder (int, optional): Z-order for the plotted objects. Default is 5.
        - autosave (bool, optional): Whether to save the figure. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Environment.plot_environment().

        Returns:
        if return_fig:
        - fig (mpl_figure.Figure): Figure with environment plotted

        - sub_ax (plt.Axes): Subplot with environment plotted.
        """

        kwargs = plot_util.organize_fig_ax_kwargs(fig=fig, sub_ax=sub_ax, **kwargs)

        sub_ax = super().plot_environment(
            autosave=False, plot_objects=False, size_factor=size_factor, **kwargs
        )

        if plot_objects:
            self.add_objects_to_plot(
                sub_ax,
                base_s=base_s,
                base_lw=base_lw,
                alpha=alpha,
                zorder=zorder,
                incl_start_reset=True,
                no_legend=no_legend,
            )

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
        "init_random_objects": {"landmark": 5},
        "init_random_walls": 5,
        "init_random_teleport_pairs": 2,
        "wall_lengths": [0.1, 0.2],
        "init_objects": dict(),  # dict with object type keys, each with a list of coordinates
        "init_teleport_pairs": list(),
        "vertical_in_from_top": True,
        "horizontal_in_from_left": True,
        "min_dist": 0.1,  # between objects (from walls is half)
        "init_seed": None,
    }

    List of attributes (in addition to Environment attributes):
        • self.object_df_columns
        • self.object_df
        • self.object_type_num_to_name_dict
        • self.object_type_name_to_num_dict
        • self.teleport_pairs_dict

    List of methods (in addition to Environment methods):
        • self.get_teleport_params()
        • self.get_object_type_num_to_plot_params_dict()
        • self.get_marker_yvals()
        • self.get_area()
        • self.check_if_parallel_walls_too_close()
        • self.check_if_walls_end_too_close()
        • self.get_object_locations()
        • self.get_teleport_coords()
        • self.get_teleport_pair_orientation()
        • self.get_number_for_each_object_type()
        • self.get_dist_from_coords_to_closest_object()
        • self.get_dist_from_coords_to_closest_wall()
        • self.sample_coords()
        • self.distance_to_closest_parallel_wall()
        • self.sample_wall_end()
        • self.add_object()
        • self.add_objects()
        • self.add_teleport_pairs()
        • self.add_walls()
        • self.get_teleport_plotting_marker()
        • self.get_teleport_plot_label()
        • self.get_object_label()
        • self.add_objects_to_plot()
        • self.plot_environment()
    """

    default_params = {
        "init_random_objects": {"landmark": 5},
        "init_random_walls": 5,
        "init_random_teleport_pairs": 2,
        "wall_lengths": [0.1, 0.2],
        "init_objects": dict(),  # dict with object type keys, each with a list of coordinates
        "init_teleport_pairs": list(),
        "vertical_in_from_top": True,
        "horizontal_in_from_left": True,
        "min_dist": 0.1,  # between objects (from walls is half)
        "init_seed": None,
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "holes": list(),  # no holes,
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
        for object_type, coords in self.init_objects.items():
            self.add_objects(object_type, coords=coords)
        self.add_teleport_pairs(coord_pairs=self.init_teleport_pairs)

        for object_type, num in self.init_random_objects.items():
            self.add_objects(object_type, num=num)

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
                "teleport_pair_idx",
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

    def _get_object_type_num(self, object_type_name="landmark") -> int:
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

    def get_teleport_params(
        self,
        position_name: str | None = None,
        teleport_pair_idx: int | None = None,
        direction: str | None = None,
    ) -> dict[str, Any]:
        """
        self.get_teleport_params()

        Obtain the parameters for a given teleportation pair.

        Args:
        - teleport_pair_idx (int): Teleportation pair index.

        Returns:
        - teleport_params (dict): Dictionary with keys and values:
            - "teleport_pair_idx" (int): Teleportation pair index.
            - "direction" (str): Direction of the teleport pair ("in" or "out").
            - "orientation" (str): Orientation of the teleport pair ("horizontal"
                or "vertical").
            - "horizontal_in_from_left" (bool): Whether the teleport in is on the
                left or right of the teleport out.
            - "vertical_in_from_top" (bool): Whether the teleport in is above or
                below the teleport out.
        """

        if teleport_pair_idx is None:
            if position_name is None:
                raise ValueError(
                    "Either teleport_pair_idx or position_name must be provided."
                )
            direction, teleport_pair_idx = plot_util.get_teleport_direction_and_index(
                position_name
            )

        elif position_name is not None:
            raise ValueError(
                "Only one teleport_pair_idx or position_name should be provided, "
                "not both."
            )

        if direction is None:
            raise ValueError(
                "If teleport_pair_idx is provided, direction must also be provided."
            )

        teleport_params = {
            "teleport_pair_idx": teleport_pair_idx,
            "direction": direction,
            "orientation": self.get_teleport_pair_orientation(teleport_pair_idx),
            "horizontal_in_from_left": self.horizontal_in_from_left,
            "vertical_in_from_top": self.vertical_in_from_top,
        }

        return teleport_params

    def get_object_type_num_to_plot_params_dict(
        self, base_s=15, base_lw=1.0, zorder=5
    ) -> dict[int, dict[str, Any]]:
        """
        self.get_object_type_num_to_plot_params_dict()

        Obtains plotting parameters from object type number.

        Args:
        - base_s (int, optional): Base size for the object marker. Default is 15.
        - base_lw (float, optional): Base linewidth for the object marker.
            Default is 1.0.
        - zorder (int, optional): Z-order for the object marker. Default is 5.

        Returns:
        - object_type_num_to_plot_params_dict (dict): Dictionary with keys and values:
            - object type number (int): dictionary with keys and values (dict):
                - "name" (str): object type name
                - "marker" (str): marker for the object
                - "color" (str): color for the object
                - "s" (int): size for the object
                - "zorder" (int): zorder for the object
                - "lw" (float): linewidth for the object
        """

        object_type_num_to_plot_params_dict = dict()
        for object_num, position_name in self.object_type_num_to_name_dict.items():
            teleport_kwargs = dict()
            if "teleport" in position_name:
                teleport_kwargs = self.get_teleport_params(position_name)
                for key in ["teleport_pair_idx", "direction"]:
                    teleport_kwargs.pop(key)

            plot_kwargs = plot_util.get_plot_marker_kwargs(
                position_name, base_s=base_s, base_lw=base_lw, **teleport_kwargs
            )
            plot_kwargs["name"] = position_name
            plot_kwargs["zorder"] = zorder
            object_type_num_to_plot_params_dict[object_num] = plot_kwargs

        return object_type_num_to_plot_params_dict

    def get_marker_yvals(
        self, sub_ax, data=None, object_type="teleport", base_s=15, above=0.8
    ):
        """
        self.get_marker_yvals(data)

        Obtain the y-values to add markers of a given object type to a plot above the
        data.

        Args:
        - sub_ax (plt.Axes): Subplot based on which to identify y-values. If data is None,
            data is retrieved from the subplot lines.
        - data (2D np.ndarray, optional): Data groups for which to obtain the y-values
            (groups x datapoints). If None, sub_ax must be provided. Default is None.
        - data (pd.DataFrame): Dataframe with the data to plot.
        - object_type (str, optional): Object type to get y-values for.
            Default is "teleport".
        - base_s (float, optional): Base size for the object marker. Default is 15.
        - above (float, optional): Proportion of the data range to offset the markers by.
            Default is 0.8.

        Returns:
        - yvals (pd.Series): Y-values for adding markers of the given object type to a
            plot.
        """

        env_plot_params_dict = self.get_object_type_num_to_plot_params_dict(
            base_s=base_s
        )

        s = None
        for match in ["exact", "within"]:
            for num, param_dict in env_plot_params_dict.items():
                name = self.object_type_num_to_name_dict[num]
                if match == "exact" and name.lower() == object_type.lower():
                    s = param_dict["s"]
                    break

                if match == "within" and object_type in name.lower():
                    s = param_dict["s"]
                    break

        if s is None:
            raise ValueError(f"Unknown object type: {object_type}")

        yvals = plot_util.get_marker_yvals(
            sub_ax, data=data, s=s, prop_data="above", above=above
        )
        return yvals

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

    def check_if_parallel_walls_too_close(
        self,
        new_wall_coords: (
            np.ndarray[tuple[int, int], np.dtype[np.float64]] | list[list[float]]
        ),
        min_dist=None,
    ) -> bool:
        """
        self.check_if_parallel_walls_too_close(new_wall_coords)

        Checks whether a new wall is too close to existing walls that are parallel
        to it.

        Args:
        - new_wall_coords (list or 2D np.ndarray): Coordinates of new wall
            [[x1, y1], [x2, y2]].
        - min_dist (float, optional): Minimum distance between walls. Default is None.

        Returns:
        - parallel_walls_too_close (bool): True if the new wall is too close
            to an existing wall that is parallel to it. False otherwise.
        """

        if len(self.walls) == 0:
            return False

        if min_dist is None:
            min_dist = float(self.min_dist)  # type: ignore[attr-defined]

        new_wall = np.asarray(new_wall_coords)

        min_wall_dist = self.distance_to_closest_parallel_wall(new_wall)

        parallel_walls_too_close = False
        if min_wall_dist is not None and min_wall_dist < min_dist:
            parallel_walls_too_close = True

        return parallel_walls_too_close

    def check_if_walls_end_too_close(
        self,
        new_wall_coords: (
            np.ndarray[tuple[int, int], np.dtype[np.float64]] | list[list[float]]
        ),
        min_dist=None,
    ) -> bool:
        """
        self.check_if_walls_end_too_close(new_wall_coords)

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
            distances, coords = list(), list()
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

    def get_object_locations(self, object_type_name="landmark"):
        """
        self.get_object_locations()

        Obtain the locations of objects of a given type in the environment.

        Args:
        - object_type_name (str, optional): Name of the object type to get locations
            for. Default is "landmark".

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

    def get_teleport_coords(self, teleport_pair_idx: int = 1, direction="in"):
        """
        self.get_teleport_coords()

        Obtain the teleport coordinates for the given teleportation pair.

        Args:
        - teleport_pair_idx (int, optional): The teleportation pair to get the
            coordinates for. Default is 1.
        - direction (str, optional): The direction to get the coordinates for.
            Default is "in".

        Raises:
        - ValueError: If the Teleportation pair index is not found.
        - ValueError: If the direction is not recognized.

        Returns:
        - teleport_coords (1D np.ndarray): The teleport coordinates.
        """

        if teleport_pair_idx not in self.teleport_pairs_dict.keys():
            raise ValueError(f"Teleportation pair {teleport_pair_idx} not found.")

        if direction not in ["in", "out"]:
            raise ValueError(f"Direction {direction} not recognized.")

        teleport_coords = self.teleport_pairs_dict[teleport_pair_idx][direction][1]

        return teleport_coords

    def get_teleport_pair_orientation(self, teleport_pair_idx: int = 1) -> str:
        """
        self.get_teleport_pair_orientation()

        Obtain the orientation of a teleportation pair.

        Args:
        - teleport_pair_idx (int, optional): Teleportation pair index. Default is 1.

        Returns:
        - orientation (str): Orientation of the teleportation pair.
        """

        if teleport_pair_idx % 2 == 0:
            orientation = "horizontal"
        else:
            orientation = "vertical"

        return orientation

    def get_number_for_each_object_type(self) -> dict[str, int]:
        """
        self.get_number_for_each_object_type()

        Obtain the number of objects for each object type.

        Returns:
        - num_dict (dict): Dictionary with a number of objects for each object type.
        """

        num_dict = dict()
        for object_type in self.objects["object_types"]:  # type: ignore[attr-defined]
            object_type_name = self.object_type_num_to_name_dict[object_type]
            if "teleport" in object_type_name:
                object_type_name = "teleport"
            if object_type_name not in num_dict.keys():
                num_dict[object_type_name] = 0
            num_dict[object_type_name] += 1

        if "teleport" in num_dict and num_dict["teleport"] % 2 != 0:
            raise RuntimeError("Number of teleportation pairs should be even.")

        return num_dict

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

    def distance_to_closest_parallel_wall(self, wall_coords, angle_tol=10):
        """
        self.distance_to_closest_parallel_wall(wall_coords)

        Obtain the distance from a set of wall coordinates to the closest parallel wall.

        Args:
        - wall_coords (2D np.ndarray): Wall coordinates [[x1, y1], [x2, y2]].
        - angle_tol (float, optional): Angle tolerance (degrees) for considering
            walls to be parallel. Default is 10.

        Returns:
        - min_dist (float): Minimum distance to closest parallel wall.
        """

        min_dist = None
        for wall in self.walls:
            angle = trig_util.get_angle_between_vectors(
                np.diff(wall_coords, axis=1).T[0], np.diff(wall, axis=1).T[0]
            )
            if np.absolute(angle) >= angle_tol:
                continue

            dists1 = trig_util.shortest_distances_from_points_to_lines(
                wall, wall_coords
            )
            dists2 = trig_util.shortest_distances_from_points_to_lines(
                wall_coords, wall
            )

            dist = min(np.min(dists1), np.min(dists2))
            if min_dist is None or dist < min_dist:
                min_dist = dist
                if min_dist < 0.1:
                    import pdb

                    pdb.set_trace()

        return min_dist

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
        - min_dist (float, optional): Minimum distance to closest object. Double is
            applied between parallel walls. Default is None.
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
                wall_coords = np.asarray([start_coords, end_coords])
                closest_dist = np.min(
                    trig_util.shortest_distances_from_points_to_lines(
                        self.objects["objects"], wall_coords
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
            _, teleport_pair_idx, teleport_direction = object_type.split("_")
            new_object["teleport_pair_idx"] = int(teleport_pair_idx)
            new_object["teleport_direction"] = teleport_direction

        self.object_df.loc[len(self.object_df)] = new_object  # type: ignore[attr-defined]

    def add_objects(
        self,
        object_type: str | int = "landmark",
        num: int = 1,
        coords: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    ):
        """
        self.add_objects()

        Add objects of a specified type to the environment.

        Args:
        - object_type (str or int, optional): Type of object to add. Default is "landmark".
        - num (int, optional): Number of objects to add. Ignored if coords are specified.
            Default is 1.
        - coords (1D np.ndarray, optional): Coordinates [x, y] to add objects at.
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
            self.add_object(coord, object_type=object_type)

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
                    if self.check_if_parallel_walls_too_close(
                        np.asarray([start_coords, end_coords])
                    ):
                        end_coords = None

                if end_coords is not None:
                    if self.check_if_walls_end_too_close(
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
        self, teleport_pair_idx: int = 1, direction: str = "in"
    ) -> str:
        """
        self.get_teleport_plotting_marker()

        Obtain the marker for a teleportation port.

        Args:
        - teleport_pair_idx (int, optional): Teleportation pair index. Default is 1.
        - direction (str, optional): Direction of the teleportation port.
            Default is "in".

        Returns:
        - marker (str): Marker for the teleportation port.
        """

        orientation = self.get_teleport_pair_orientation(teleport_pair_idx)

        marker = plot_util.get_teleport_plotting_marker(
            direction,
            orientation,
            horizontal_in_from_left=self.horizontal_in_from_left,
            vertical_in_from_top=self.vertical_in_from_top,
        )

        return marker

    def get_teleport_plot_label(self, name="teleport_0_in"):
        """
        self.get_teleport_plot_label()

        Obtain the label for a teleportation port for plotting.

        Args:
        - name (str, optional): Name of the teleportation port.
            Default is "teleport_0_in".

        Returns:
        - label (str): Label for the teleportation port for plotting.
        """

        label = name.replace("_", " ")

        label = label.replace("in", "(in)").replace("out", "(out)")
        if self.num_teleport_pairs == 1:
            label = label.replace("teleport 0", "teleport")

        return label

    def get_object_label(self, name="target"):
        """
        self.get_object_label()

        Obtain the label for an object for plotting.

        Args:
        - name (str, optional): Name of the object. Default is "target".

        Returns:
        - label (str): Label for the object for plotting.
        """

        if "teleport" in name:
            label = self.get_teleport_plot_label(name)
        else:
            label = super().get_object_label(name)

        return label

    def add_objects_to_plot(
        self,
        sub_ax,
        skip_object_types: list = list(),
        alpha: float = 0.8,
        base_s: float = 15,
        base_lw: float = 1.0,
        zorder: int = 5,
        no_legend: bool = False,
        **kwargs,
    ):
        """
        self.add_objects_to_plot()

        Add objects to environment plot.

        Args:
        - sub_ax (plt.Axes): Subplot to add objects to.
        - skip_object_types (list, optional): List of object type names to skip
            plotting. Default is an empty list.
        - alpha (float, optional): Alpha value for plotting objects. Default is 0.8.
        - base_s (float, optional): Base size for object markers. Default is 15.
        - base_lw (float, optional): Base linewidth for object markers. Default is 1.0.
        - zorder (int | None, optional): Z-order for the plotted objects.
            If None, defaults are used. Default is 5.
        - no_legend (bool, optional): Whether to remove legend from plot.
            Default is False.

        Keyword Args:
        - **kwargs: Additional keyword arguments to pass to the legend function.
        """

        object_type_num_to_plot_params_dict = (
            self.get_object_type_num_to_plot_params_dict(
                base_s=base_s, base_lw=base_lw, zorder=zorder
            )
        )
        labelled = list()
        for coords, object_type in zip(
            self.objects["objects"], self.objects["object_types"]
        ):
            object_dict = object_type_num_to_plot_params_dict[object_type].copy()
            name = object_dict.pop("name")
            skip = any([skip_type in name for skip_type in skip_object_types])
            if skip:
                continue
            label = self.get_object_label(name)
            if label in labelled:
                label = None
            else:
                labelled.append(label)

            if zorder is not None:
                object_dict["zorder"] = zorder

            sub_ax.scatter(
                *coords,
                label=label,
                alpha=alpha,
                **object_dict,
            )

        if not no_legend:
            sub_ax.legend(
                loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, **kwargs
            )

    def plot_environment(
        self,
        fig: mpl_figure.Figure | None = None,
        sub_ax: plt.Axes | None = None,
        plot_objects: bool = True,
        skip_object_types: list = list(),
        size_factor: float = 2.5,
        alpha: float = 0.8,
        base_s: float = 15,
        base_lw: float = 1.0,
        zorder: int = 5,
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
        - size_factor (float, optional): Factor to multiply environment width by to
            determine figure size. Default is 2.5.
        - alpha (float, optional): Alpha value for plotting objects. Default is 0.8.
        - base_s (float, optional): Base size for object markers. Default is 15.
        - base_lw (float, optional): Base linewidth for object markers. Default is 1.0.
        - zorder (int | None, optional): Z-order for the plotted objects. If None,
            defaults are used. Default is 5.
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
            env_width, env_height = self.get_environment_figsize(
                size_factor=size_factor
            )
            if plot_objects:
                env_width *= 1.5  # for legend and labels
            kwargs["fig"], kwargs["ax"] = plt.subplots(figsize=(env_width, env_height))

        sub_ax = super().plot_environment(autosave=False, plot_objects=False, **kwargs)

        if plot_objects:
            self.add_objects_to_plot(
                sub_ax,
                skip_object_types=skip_object_types,
                base_s=base_s,
                base_lw=base_lw,
                zorder=zorder,
                alpha=alpha,
                no_legend=no_legend,
            )

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
