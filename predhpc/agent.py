from typing import Any, TYPE_CHECKING, Callable
import warnings

import copy
from matplotlib import pyplot as plt
from matplotlib import animation, markers
from matplotlib import colors as mpl_colors
from matplotlib import figure as mpl_figure
import numpy as np
import seaborn as sns
from ratinabox import Agent
from ratinabox import utils as rutils

from predhpc import env, util, plot_util

if TYPE_CHECKING:
    import ratinabox


class ResetableAgent(Agent):
    """Extend the agent so that is has an optimal maximum trajectory length after which
    it resets to a random location

    default_params = {
        "dt": 0.01,  # time step, in seconds
        "trajectory_length": None,  # int or iterable of ints
        "num_trajectories": None,  # number of trajectory lengths to sample
        "exp_factors": None,  # exponential factors for trajectory_length (inv. scale, rate, minimum). Defaults to None.
        "random_max": None,  # max value for randomizing trajectory_length
        "start_position": None,  # position to start trajectories from
        "reset_position": None,  # position to reset trajectories from
        "target_position": None,  # position to use as target
        "wait_between_targets": 30,  # number of steps to wait between target reaching
        "reset_reached_within_tolerance_prop_to_dt": 0.5,  # proportion of dt to use as reset tolerance
        "target_reached_within_tolerance_prop_to_dt": 0.5,  # proportion of dt to use as target tolerance
        "fixed_direction": False,  # keep same direction (1D environment only)
    }
    """

    default_params = {
        "dt": 0.01,  # time step, in seconds
        "trajectory_length": None,  # int or iterable of ints
        "num_trajectories": None,  # number of trajectory lengths to sample
        "exp_factors": None,  # exponential factors for trajectory_length (inv. scale, rate, minimum). Defaults to None.
        "random_max": None,  # max value for randomizing trajectory_length
        "start_position": None,  # position to start trajectories from
        "reset_position": None,  # position to reset trajectories from
        "target_position": None,  # position to use as target
        "wait_between_targets": 30,  # number of steps to wait between target reaching
        "reset_reached_within_tolerance_prop_to_dt": 0.5,  # proportion of dt to use as reset tolerance
        "target_reached_within_tolerance_prop_to_dt": 0.5,  # proportion of dt to use as target tolerance
        "fixed_direction": False,  # keep same direction (1D environment only)
    }

    def __init__(self, Env: "ratinabox.Environment", params: dict[str, Any] = dict()):
        """Initialise the agent.

        Args:
            Env (Environment): The environment in which the agent is placed.
            params (dict, optional): Parameters for the agent. Defaults to {}.

        Raises:
            ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        super().__init__(Env, self.params)

        if self.Environment.dimensionality == "2D":
            self.fixed_direction = False
        elif self.fixed_direction:
            if self.speed_mean < 0:  # type: ignore[reportGeneralTypeIssues]
                raise ValueError("Cannot have fixed direction with negative speed.")
            if self.Environment.boundary_conditions == "periodic":
                raise NotImplementedError(
                    "Fixed direction not implemented for periodic boundary conditions."
                )

        self.actual_trajectory_lengths = list()
        self.set_all_positions()
        self.set_trajectory_lengths()

    def set_all_positions(self):
        """Set all positions, checking that they are within the environment
        extent.
        """

        self.start_position = self.format_position(self.start_position)
        self.reset_position = self.format_position(self.reset_position)
        self.target_position = self.format_position(self.target_position)

        self.reached_reset_position = list()
        self.reached_target_position = list()

        if self.start_position is not None:
            self.set_position_and_velocity(position=self.start_position, velocity=0)
            self.must_fix_velocity_record = False
        self.steps_before_checking_for_target = 0

    def set_trajectory_lengths(self):
        """Set the trajectory lengths, either from the passed value, or by
        sampling from the exponential distribution.
        """

        if self.trajectory_length is not None:
            self.num_trajectories = None
            self.exp = None
            self.rand = None

        elif self.num_trajectories:
            self.trajectory_length = util.get_trajectory_lengths(
                num_trajectories=self.num_trajectories,
                exp_factors=self.exp_factors,  # type: ignore[reportGeneralTypeIssues]
                random_max=self.random_max,  # type: ignore[reportGeneralTypeIssues]
            )

        self.trajectory_lengths = None
        self.current_trajectory_length = 0
        if self.trajectory_length is not None:
            if not isinstance(self.trajectory_length, int):
                self.trajectory_length = np.maximum(self.trajectory_length, 1)
                if len(self.trajectory_length) == 0:
                    raise ValueError("If passing iterable, must have length > 0.")
                self.trajectory_lengths = self.trajectory_length
                self.trajectory_length = self.trajectory_lengths[0]

        self.num_steps_total = 0

    def format_position(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]]
        | list[float]
        | None = None,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Formats positions, if applicable, and returns them.

        Args:
            position (np.ndarray | list | None, optional): Position to format.
                Defaults to None.

        Raises:
            ValueError: If position is not within the environment extent.

        Returns:
            np.ndarray | None: Formatted position.
        """

        if position is not None:
            position = np.asarray(position).reshape(-1)
            if len(position) != self.Environment.D:
                raise ValueError(
                    f"Positions must comprise exactly {self.Environment.D} value(s)."
                )
            position = position.reshape(self.Environment.D)

            # [min, max] or [left, right, bottom, top]
            extent = self.Environment.extent

            if self.Environment.D == 1:
                if position < extent[0] or position > extent[1]:
                    raise ValueError(
                        "Position must be within the environment extent: " f"{extent}."
                    )
            elif self.Environment.D == 2:
                if position[0] < extent[0] or position[0] > extent[1]:
                    raise ValueError(
                        "First position dimension must be within the "
                        f"environment extent: {extent[:2]}."
                    )
                if position[1] < extent[2] or position[1] > extent[3]:
                    raise ValueError(
                        "Second position dimension must be within the "
                        f"environment extent: {extent[2:]}."
                    )

            else:
                raise ValueError(
                    "Expected environment dimensionality to be 1 or 2. "
                    f"Got {self.Environment.D}."
                )

        return position

    def fix_velocity_record(self, dt: float | None = None):
        """Fix values computed and recorded based on the velocity, if applicable."""

        if not hasattr(self, "prev_average_measured_speed"):
            return

        if dt is None:
            dt = float(self.dt)

        velocity = np.asarray(self.velocity).astype(np.float64)

        tau_speed = 10
        self.average_measured_speed = (
            self.prev_average_measured_speed
            + dt / tau_speed * (np.linalg.norm(velocity, ord=2))
        )

        self.save_velocity = velocity

        if self.save_history is True and len(self.history["vel"]):  # type: ignore[reportGeneralTypeIssues]
            self.history["vel"][-1] = self.save_velocity
            if self.Environment.dimensionality == "2D":
                rotational_velocity = float(self.rotational_velocity)
                self.history["rot_vel"][-1] = rotational_velocity

    def check_and_fix_velocity(
        self,
        prev_velocity: np.ndarray[tuple[int], np.dtype[np.float64]],
        dt: float | None = None,
    ):
        """Check if velocity is negative and fix if applicable."""

        if not (self.Environment.dimensionality == "1D" and self.fixed_direction):
            return

        if self.velocity >= 0:
            return

        if self.reset_position is not None and self.pos[0] > self.reset_position[0]:
            return

        if dt is None:
            dt = self.dt

        new_velocity = self.velocity
        speed_mean, speed_std = self.speed_mean, self.speed_std  # type: ignore[reportGeneralTypeIssues]
        for _ in range(10):
            if new_velocity < 0:  # resample velocity until it is positive
                new_velocity = prev_velocity + rutils.ornstein_uhlenbeck(
                    dt=dt,
                    x=prev_velocity,
                    drift=speed_mean,
                    noise_scale=speed_std,
                    coherence_time=self.speed_coherence_time,  # type: ignore[reportGeneralTypeIssues]
                )
            else:
                break

        if new_velocity < 0:
            new_velocity = prev_velocity * 0  # set to 0

        self.velocity = new_velocity
        self.fix_velocity_record(dt=dt)

    def set_position_and_velocity(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        velocity: float | None = None,
        rotational_velocity: float | None = 0.0,
        sample: bool = True,
    ):
        """Set the position and velocity of the agent.

        From Agent.__init__() in ratinabox/agent.py
        """

        # initialise starting positions and velocity

        speed_mean, speed_std = self.speed_mean, self.speed_std  # type: ignore[reportGeneralTypeIssues]
        if self.Environment.dimensionality == "2D":
            if position is None:
                self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            elif sample:
                self.pos = self.sample_within_tolerance(position)
            else:
                self.pos = position
            if velocity is None or len(np.asarray(velocity).reshape(-1)) == 1:
                direction = np.random.uniform(0, 2 * np.pi)
                velocity = speed_std
                velocity = velocity * np.array([np.cos(direction), np.sin(direction)])
            self.velocity = np.asarray(velocity).reshape(2)
            self.rotational_velocity = rotational_velocity

        elif self.Environment.dimensionality == "1D":
            if position is None:
                self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            elif sample:
                self.pos = self.sample_within_tolerance(position)
            else:
                self.pos = position
            if velocity is None:
                self.velocity = np.array([speed_mean]).reshape(1)
            else:
                self.velocity = np.array([velocity]).reshape(1)
            if self.Environment.boundary_conditions == "solid":
                if speed_mean != 0:
                    warnings.warn(
                        "solid 1D boundary conditions and non-zero speed mean."
                    )

        self.must_fix_velocity_record = True

        return

    def sample_within_tolerance(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]],
        sample_within_tolerance_prop_to_dt: float = 1,
        max_attempts: int = 100,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Sample a position within the tolerance of the given position.

        Args:
            position (np.ndarray): The position to sample around.
            sample_within_tolerance_prop_to_dt (float): The proportion of the tolerance to
                sample within. Defaults to None, in which case the agent's
                target_reached_within_tolerance_prop_to_dt is used.

        Returns:
            position (np.ndarray): The sampled position.
        """

        tolerance = self.dt * sample_within_tolerance_prop_to_dt

        new_position = None
        for _ in range(max_attempts):
            x_jitter = np.random.uniform(-tolerance, tolerance)

            if len(position) == 1:
                new_position = position + x_jitter
            elif len(position) == 2:
                y_max = np.sqrt(tolerance**2 - x_jitter**2)
                y_jitter = np.random.uniform(-y_max, y_max)
                new_position = position + np.asarray([x_jitter, y_jitter])
            else:
                raise NotImplementedError(
                    "Sampling within tolerance only implemented for 1D and 2D."
                )
            if self.Environment.check_if_position_is_in_environment(new_position):
                break

        if new_position is None:
            raise RuntimeError(
                f"Could not find a new position within tolerance proportion "
                f"{sample_within_tolerance_prop_to_dt} of {position}."
            )

        return new_position

    def reset(self):
        """Reset the agent to a random location."""

        self.set_position_and_velocity(position=self.start_position, velocity=0)

        self.actual_trajectory_lengths.append(self.current_trajectory_length)
        if self.trajectory_lengths is not None:
            i = len(self.actual_trajectory_lengths) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]

        self.current_trajectory_length = 0

        return

    def get_trajectory_lengths_to_date(self):
        """Return the trajectory lengths to date.

        Returns:
            list: Trajectory lengths to date.
        """
        traj_leng_to_date = self.actual_trajectory_lengths
        if self.current_trajectory_length > 0:
            traj_leng_to_date = self.actual_trajectory_lengths + [
                self.current_trajectory_length
            ]
        return traj_leng_to_date

    def log_trajectories_to_date(self):
        """Log the trajectory lengths to date."""
        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        print(
            f"Trajectory lengths ({len(traj_leng_to_date)}) to date (in steps): "
            f"{traj_leng_to_date}"
        )

    def log_trajectory_stats_to_date(self, log_as_time: bool = True):
        """Log the trajectory length statistics to date."""

        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        traj_length_unit = "steps"

        # get trajectory lengths in seconds
        if log_as_time:
            traj_leng_to_date = [leng * self.dt for leng in traj_leng_to_date]
            traj_length_unit = "sec"
            if np.mean(traj_leng_to_date) / 60 > 2:
                traj_leng_to_date = [leng / 60 for leng in traj_leng_to_date]
                traj_length_unit = "min"

        # get trajectory length statistics
        traj_leng_to_date_mean = np.mean(traj_leng_to_date)
        traj_leng_to_date_std = np.std(traj_leng_to_date)

        print(
            f"Trajectory lengths ({len(traj_leng_to_date)}) to date: "
            f"{traj_leng_to_date_mean:.2f} +/- {traj_leng_to_date_std:.2f} "
            f"{traj_length_unit} each"
        )

    def plot_trajectories_to_date(
        self, in_minutes: bool = True, autosave: bool | None = None
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the trajectory lengths to date.

        Args:
            in_minutes (bool, optional): Whether to plot in minutes. Defaults to True.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.
        """

        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        fig, ax = plot_util.plot_trajectory_lengths(
            dt=self.dt, trajectory_lengths=traj_leng_to_date, in_minutes=in_minutes
        )

        rutils.save_figure(fig, "trajectories_to_date", save=autosave)

        return fig, ax

    def get_reset_times(self):
        """Get the reset times.

        Returns:
            list: Reset times.

        Raises:
            ValueError: If agent does not have reset steps.
        """

        if hasattr(self, "actual_trajectory_lengths"):
            if len(self.actual_trajectory_lengths) == 0:
                reset_times = np.array([])
            else:
                reset_times = np.cumsum(self.actual_trajectory_lengths) * self.dt
        else:
            raise ValueError("Agent does not have reset steps.")

        return reset_times

    def check_if_position_reached(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        sample_within_tolerance_prop_to_dt: float = 0.5,
    ) -> bool:
        """Check if the agent has reached a position.

        Args:
            target_position (np.array): Target position.
            sample_within_tolerance_prop_to_dt (float): Tolerance proportion, wrt self.dt.

        Returns:
            bool: Whether the agent has reached the target position.
        """

        if position is not None:
            # calculate the distance between the current position and the reset position
            dist = np.linalg.norm(self.pos - position, ord=2)

            # check if the distance is less than the tolerance
            if dist < (self.dt * sample_within_tolerance_prop_to_dt):
                return True

        return False

    def check_if_reset_position_reached(self) -> bool:
        """Check if the agent has reached the reset position.

        Returns: Whether the agent has reached the reset position.
        """

        return self.check_if_position_reached(
            self.reset_position, self.reset_reached_within_tolerance_prop_to_dt  # type: ignore[reportGeneralTypeIssues]
        )

    def check_if_target_position_reached(self) -> bool:
        """Check if the agent has reached the target position.

        Returns: Whether the agent has reached the target position.
        """

        if self.target_position is None:
            return False

        if self.steps_before_checking_for_target > 0:
            self.steps_before_checking_for_target -= 1
            return False

        else:
            target_reached = self.check_if_position_reached(
                self.target_position, self.target_reached_within_tolerance_prop_to_dt  # type: ignore[reportGeneralTypeIssues]
            )
            if target_reached:
                self.steps_before_checking_for_target = self.wait_between_targets  # type: ignore[reportGeneralTypeIssues]
            return target_reached

    def check_if_trajectory_end_reached(self) -> bool:
        """Check if the agent has reached the end of its trajectory.

        Returns:
            bool: Whether the agent has reached the end of its trajectory.
        """

        self.reached_end = False
        if self.reset_position is not None and self.check_if_reset_position_reached():
            # record the time step at which the agent reached the reset position
            self.reached_end = True
            if len(self.reached_reset_position):
                if self.num_steps_total == self.reached_reset_position[-1]:
                    self.reached_end = False

            if self.reached_end:
                self.reached_reset_position.append(self.num_steps_total)

        if self.trajectory_length is not None:
            if self.current_trajectory_length >= self.trajectory_length:
                self.reached_end = True

        return self.reached_end

    def check_and_record_target_reached(self) -> bool:
        """Check if the agent has reached the target in its trajectory.

        Returns:
            bool: Whether the agent has reached the target in its trajectory.
        """

        self.reached_target = False
        if self.target_position is not None and self.check_if_target_position_reached():
            # record the time step at which the agent reached the target position
            self.reached_target = True
            if len(self.reached_target_position):
                if self.num_steps_total == self.reached_target_position[-1]:
                    self.reached_target = False

            if self.reached_target:
                self.reached_target_position.append(self.num_steps_total)

        return self.reached_target

    def update(self, dt: float | None = None, skip_checks: bool = False, **kwargs):
        """Update the agent, optionally with a new position and velocity.

        See Agent.update() in ratinabox/agent.py for kwargs.
        """

        if not skip_checks:
            self.check_and_record_target_reached()
            if self.check_if_trajectory_end_reached():
                self.reset()

        self.prev_average_measured_speed = self.average_measured_speed
        prev_velocity = self.velocity

        super().update(dt=dt, **kwargs)

        if self.Environment.dimensionality == "1D" and self.fixed_direction:
            self.check_and_fix_velocity(prev_velocity=prev_velocity, dt=dt)

        elif self.must_fix_velocity_record:
            self.velocity = prev_velocity
            self.fix_velocity_record(dt=dt)

        self.must_fix_velocity_record = False

        self.current_trajectory_length += 1
        self.num_steps_total += 1

    def plot_trajectory_resets(
        self,
        t_start: float = 0.0,
        t_end: float | None = None,
        framerate: int | float = 10,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        alpha: float = 0.6,
        color: str = "k",
        ms: int | float = 50,
        plot_targets: bool = True,
        autosave: bool | None = None,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plots the trajectory between t_start (seconds) and t_end (defaulting to the
        last time available)

        From Agent.plot_1D_trajectories() in ratinabox/agent.py. Modified to enable
        plotting of reset steps, and use of colormaps for trajectories.

        Args:
            t_start: start time in seconds
            t_end: end time in seconds (default = self.history["t"][-1])
            framerate: how many scatter points / per second of motion to display
            fig: matplotlib figure object
            ax: matplotlib axes object
            alpha: trajectory point opaqness
            color: trajectory point color
            ms: plot point size
            plot_targets: whether to plot the target
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            fig, ax
        """

        dt = self.dt
        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        if t_end == None:
            t_end = t[-1]
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end))) + 1
        skiprate = max(1, int((1 / framerate) / dt))

        t = t / 60  # minutes
        time = t[startid:endid][::skiprate]
        pos = pos[startid:endid][::skiprate]

        # get reset step indices
        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")
        elif len(time) == 0:
            raise RuntimeError("Duration too short. No time points to plot.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        if self.Environment.D == 1:
            min_y, max_y = self.Environment.extent
            diff = max_y - min_y
        elif self.Environment.D == 2:
            left, right, bottom, top = self.Environment.extent
            diff = max(right - left, top - bottom)
            min_y = min(left, bottom)
            max_y = max(right, top)
        else:
            raise RuntimeError("Only 1D and 2D environments are supported.")

        n_reached = len(self.reached_reset_position)
        n_targets = len(self.reached_target_position)
        alpha /= self.Environment.D
        alpha_pts = 0.9 / self.Environment.D
        for i in range(self.Environment.D):
            ax.scatter(
                time,
                pos,
                alpha=alpha,
                marker=markers.MarkerStyle("."),
                color=color,
                s=ms / 10,
            )

            if n_reached:
                if self.start_position is not None:
                    x_start = [
                        t[x]
                        for x in self.reached_reset_position
                        if x >= startid and x < endid
                    ]
                    y_start = [self.start_position[i]] * n_reached
                    ax.scatter(
                        x_start,
                        y_start,
                        marker=markers.MarkerStyle("."),
                        color="blue",
                        alpha=alpha_pts,
                        s=ms,
                    )

                if self.reset_position is not None:
                    x_reset = [
                        t[x - 1]
                        for x in self.reached_reset_position
                        if x >= startid and x < endid
                    ]
                    y_reset = [self.reset_position[i]] * n_reached
                    ax.scatter(
                        x_reset,
                        y_reset,
                        marker=markers.MarkerStyle("x"),
                        color="red",
                        alpha=alpha_pts,
                        s=ms / 3,
                    )

            if plot_targets and n_targets and self.target_position is not None:
                x_targ = [
                    t[x]
                    for x in self.reached_target_position
                    if x >= startid and x < endid
                ]
                y_targ = [self.target_position[i]] * n_targets
                ax.scatter(
                    x_targ,
                    y_targ,
                    marker=markers.MarkerStyle("d"),
                    color="gold",
                    alpha=alpha_pts,
                    s=ms / 5,
                )

        ax.set_xlabel("Time / min")
        ax.set_ylabel("Position / m")

        bottom = min_y - diff * 0.1
        top = max_y + diff * 0.1
        ax.set_ylim(bottom=bottom, top=top)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        rutils.save_figure(fig, "trajectory_resets", save=autosave)

        return fig, ax

    def plot_trajectory(
        self,
        t_start: float = 0.0,
        t_end: float | None = None,
        framerate: int | float = 10,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        decay_point_size: bool = False,
        plot_agent: bool = True,
        colormap: str | None | mpl_colors.Colormap = None,
        alpha: float = 0.7,
        xlim: float | None = None,
        background_color: str | None = None,
        plot_traj_ends: bool = True,
        target_alpha: float = 1.0,
        cmap_per: bool = False,
        scale_cmap_per: bool = False,
        ms_2D: int | float = 15,
        size_fact: float | None = None,
        autosave: bool | None = None,
        **kwargs,  # hacky catch-all...
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plots the trajectory between t_start (seconds) and t_end (defaulting to the last time available)

        From Agent.plot_trajectory() in ratinabox/agent.py. Modified to enable plotting of reset steps, and use of colormaps for trajectories.

        Args:
            t_start: start time in seconds
            t_end: end time in seconds (default = self.history["t"][-1])
            framerate: how many scatter points / per second of motion to display
            fig, ax: the fig, ax to plot on top of, optional, if not provided used self.Environment.plot_Environment().
              This can be used to plot trajectory on top of receptive fields etc.
            decay_point_size: decay trajectory point size over time (recent times = largest)
            plot_agent: dedicated point show agent current position
            colormap: colormap to use to plot trajectories
            alpha: plot point opaqness
            xlim: In 1D, forces the top (right) xlim to be a certain time (minutes) (useful if animating this function)
            background_color: color of the background if not matplotlib default, only for 1D (probably white)
            plot_traj_ends: plot a point at the end of each trajectory
            target_alpha: transparency with which to plot target position
            cmap_per: if True, the colormap is used to set the color for each time point. Otherwise, each trajectory has its own color.
            scale_cmap_per: if True, and cmap_per is True, the full range of the colormap is used for each trajectory, regardless of its length
            ms_2D: the size of the points in the 2D plot is set to this value.
            size_fact: if not None, the size of the points is multiplied by this value.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            fig, ax
        """

        dt = self.dt
        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        if t_end is None:
            t_end = float(t[-1])
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end))) + 1
        skiprate = max(1, int((1 / framerate) / dt))
        if self.Environment.dimensionality == "2D":
            trajectory = pos[startid:endid, :][::skiprate]
        elif self.Environment.dimensionality == "1D":
            trajectory = pos[startid:endid][::skiprate]
        else:
            raise RuntimeError(f"Environment dimensionality must be either 1D or 2D.")
        time = t[startid:endid][::skiprate]

        # get reset step indices
        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")
        elif len(time) == 0:
            raise RuntimeError("Duration too short. No time points to plot.")

        last_length = len(t) - sum(self.actual_trajectory_lengths)
        trajectory_lengths = self.actual_trajectory_lengths + [last_length]
        traj_idx = [np.full(steps, i) for i, steps in enumerate(trajectory_lengths)]
        if cmap_per:
            if scale_cmap_per:
                cmap_vals = [np.linspace(0, 1, steps) for steps in trajectory_lengths]
            else:
                cmap_vals = [np.arange(steps) for steps in trajectory_lengths]
        else:
            cmap_vals = traj_idx[:]
        cmap_vals = np.concatenate(cmap_vals).astype(float)
        cmap_vals = cmap_vals[startid:endid][::skiprate]
        cmap_min, cmap_max = cmap_vals.min(), cmap_vals.max()
        if cmap_min == cmap_max:
            cmap_vals[:] = 0.5  # mid point of the colormap
        else:
            cmap_vals = (cmap_vals - cmap_min) / (cmap_max - cmap_min)

        traj_idx = np.concatenate(traj_idx).astype(int)[startid:endid][::skiprate]

        if colormap is None:
            colormap = "crest"
        c = sns.color_palette(colormap, as_cmap=True)(cmap_vals)  # type: ignore[reportGeneralTypeIssues]
        ##############################

        if self.Environment.dimensionality == "2D":
            if size_fact is not None:
                extent = self.Environment.extent
                x_base = extent[1] - extent[0]
                y_base = extent[3] - extent[2]
                figsize = (size_fact * x_base, size_fact * y_base)
                fig, ax = plt.subplots(figsize=figsize)

            fig, ax = self.Environment.plot_environment(fig=fig, ax=ax)

            if ax is None:
                raise RuntimeError("ax is None.")

            if self.target_position is not None:
                ax.scatter(
                    *self.target_position,
                    marker=markers.MarkerStyle("d"),
                    color="gold",
                    s=20,
                    zorder=5,
                    edgecolors="darkgoldenrod",
                    linewidth=0.5,
                    alpha=target_alpha,
                )

            s = ms_2D * np.ones_like(time)
            if decay_point_size == True:
                s = ms_2D * np.exp((time - time[-1]) / 10)
                s[(time[-1] - time) > ms_2D] *= 0

            if plot_traj_ends == True and len(self.actual_trajectory_lengths):
                ends = np.where(np.diff(traj_idx) > 0)[0]
                ends = np.append(ends, len(trajectory) - 1)
                s[ends] = ms_2D * 2
                # set last colormap value to dark red
                c[ends] = mpl_colors.to_rgba("darkred")  # type: ignore[reportGeneralTypeIssues]

            if plot_agent == True:
                s[-1] = ms_2D * 2.75
                # set last colormap value to red
                c[-1] = mpl_colors.to_rgba("r")  # type: ignore[reportGeneralTypeIssues]

            ax.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                s=s,
                alpha=alpha,
                zorder=2,
                c=c,
                linewidth=0,
            )
        if self.Environment.dimensionality == "1D":
            if fig is None or ax is None:
                fig, ax = plt.subplots(figsize=(3, 1.5))
            ax.scatter(time / 60, trajectory, alpha=alpha, linewidth=0, c=c, s=5)
            ax.spines["left"].set_position(("data", t_start / 60))
            ax.set_xlabel("Time / min")
            ax.set_ylabel("Position / m")
            ax.set_xlim(t_start / 60, t_end / 60)
            if xlim is not None:
                ax.set_xlim(right=xlim)

            ax.set_ylim(bottom=0, top=self.Environment.extent[1])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xticks([t_start / 60, t_end / 60])
            ex = self.Environment.extent
            ax.set_yticks([ex[1]])
            if background_color is not None:
                ax.set_facecolor(background_color)
                fig.patch.set_facecolor(background_color)  # type: ignore[reportGeneralTypeIssues]

        rutils.save_figure(fig, "trajectory", save=autosave)

        return fig, ax

    def plot_trajectory_edges(
        self,
        t_start: float = 0.0,
        t_end: float | None = None,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        decay_point_size: bool = False,
        plot_agent: bool = True,
        colormap: str | None | mpl_colors.Colormap = None,
        alpha: float = 0.7,
        xlim: float | None = None,
        background_color: str | None = None,
        plot_starts: bool = True,
        plot_ends: bool = True,
        autosave: bool | None = None,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plots the trajectory starts and ends between t_start (seconds) and t_end
        (defaulting to the last time available)

        Args:
            t_start: start time in seconds
            t_end: end time in seconds (default = self.history["t"][-1])
            fig, ax: the fig, ax to plot on top of, optional, if not provided used
                self.Environment.plot_Environment(). This can be used to plot
                trajectory ends on top of receptive fields etc.
            decay_point_size: decay trajectory point size over time
                (recent times = largest)
            plot_agent: dedicated point show agent current position
            colormap: colormap to use to plot trajectories starts/ends
            alpha: plot point opaqness
            xlim: In 1D, forces the top (right) xlim to be a certain time (minutes)
                (useful if animating this function)
            background_color: color of the background if not matplotlib default,
                only for 1D (probably white)
            plot_starts: plot trajectory starts
            plot_ends: plot trajectory ends
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.

        Returns:
            fig, ax
        """

        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        if t_end is None:
            t_end = float(t[-1])
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end))) + 1

        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")

        if colormap is None:
            colormap = "crest"
        cmap = sns.color_palette(colormap, as_cmap=True)

        all_ends = np.cumsum(self.actual_trajectory_lengths)
        start_c, end_c = None, None
        traj_starts, traj_ends = None, None
        if plot_starts:
            traj_starts = np.insert(all_ends, 0, 0)
            start_c = cmap(np.linspace(0, 1, len(traj_starts)))  # type: ignore[reportGeneralTypeIssues]
        if plot_ends:
            traj_ends = np.append(all_ends - 1, len(t) - 1)
            end_c = cmap(np.linspace(0, 1, len(traj_ends)))  # type: ignore[reportGeneralTypeIssues]
        if not (plot_starts or plot_ends):
            raise ValueError(
                "At least one of 'plot_starts' or 'plot_ends' must be True."
            )

        for c, traj_idx, marker in [
            (start_c, traj_starts, "x"),
            (end_c, traj_ends, "o"),
        ]:
            if c is None or traj_idx is None:
                continue
            lw = 2 if marker == "x" else 0
            traj_idx = traj_idx[(traj_idx >= startid) & (traj_idx <= endid)]
            trajectory = pos[traj_idx]
            time = t[traj_idx]

            if len(time) == 0:
                raise RuntimeError("Duration too short. No trajectory points to plot.")

            if self.Environment.dimensionality == "2D":
                fig, ax = self.Environment.plot_environment(fig=fig, ax=ax)
                if ax is None:
                    raise RuntimeError("ax is None.")
                if self.target_position is not None:
                    ax.scatter(
                        *self.target_position,
                        marker=markers.MarkerStyle("d"),
                        color="gold",
                        s=20,
                        zorder=5,
                        edgecolors="black",
                        linewidth=0.5,
                    )

                s = 15 * np.ones_like(time)
                if decay_point_size == True:
                    s = 15 * np.exp((time - time[-1]) / 10)
                    s[(time[-1] - time) > 15] *= 0

                if plot_agent == True:
                    s[-1] = 40
                    # set last colormap value to red
                    c[-1] = mpl_colors.to_rgba("r")  # type: ignore[reportGeneralTypeIssues]

                ax.scatter(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    s=s,
                    alpha=alpha,
                    zorder=2,
                    c=c,
                    linewidth=lw,
                    marker=marker,
                )
            elif self.Environment.dimensionality == "1D":
                if fig is None or ax is None:
                    fig, ax = plt.subplots(figsize=(3, 1.5))
                ax.scatter(
                    time / 60,
                    trajectory,
                    alpha=alpha,
                    linewidth=lw,
                    c=c,
                    s=5,
                    marker=marker,
                )
                ax.spines["left"].set_position(("data", t_start / 60))
                ax.set_xlabel("Time / min")
                ax.set_ylabel("Position / m")
                ax.set_xlim(t_start / 60, t_end / 60)
                if xlim is not None:
                    ax.set_xlim(right=xlim)

                ax.set_ylim(bottom=0, top=self.Environment.extent[1])
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.set_xticks([t_start / 60, t_end / 60])
                ex = self.Environment.extent
                ax.set_yticks([ex[1]])
                if background_color is not None:
                    ax.set_facecolor(background_color)
                    fig.patch.set_facecolor(background_color)  # type: ignore[reportGeneralTypeIssues]

        rutils.save_figure(fig, "trajectory_edges", save=autosave)

        return fig, ax


class TAgent(ResetableAgent, util.ParamsManagerMixin):
    """Extend the reset agent so that it operates in a T maze"""

    default_params = {
        "target_arm": "left",
        "target_location_prop_to_arm": 0.75,  # proportion down arm at which to set target
        "left_arm_prop": 0.75,  # proportion of trajectories to target to left arm
    }

    ignored_param_keys = ["reset_position", "start_position", "target_position"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Env: env.TEnv, params: dict[str, Any] = dict()):
        """Initialise the agent.

        Args:
            params (dict, optional): Parameters for the agent. Defaults to {}.

        Raises:
            ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        if not isinstance(Env, env.TEnv):
            raise TypeError("Env must be a TEnv object.")

        self.set_fixed_params()

        super().__init__(Env, self.params)

        self.set_current_arm()

    @property
    def near_branch(self) -> bool:
        return self.pos[1] > (self.Environment.branch_y * 0.98)

    @property
    def at_branch(self) -> bool:
        return self.pos[1] > self.Environment.branch_y

    def get_direction(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the direction to the target."""

        if self.target == "left":
            target = self.Environment.left_T_end

        elif self.target == "right":
            target = self.Environment.right_T_end

        else:
            raise RuntimeError("Target must be 'left' or 'right'.")

        direction = np.asarray(target) - self.pos

        return direction

    def update(
        self,
        dt: float | None = None,
        speed_fact: int | float = 3,
        drift_to_random_strength_ratio: float = 0.7,
        **kwargs,
    ):
        """Update the agent, optionally with a new position and velocity.

        See Agent.update() in ratinabox/agent.py for kwargs.
        """

        self.check_and_record_target_reached()
        if self.check_if_trajectory_end_reached():
            self.reset()

        # calculate drift_velocity
        if self.near_branch:
            direction = self.get_direction()
        else:
            direction = self.Environment.T_split - self.pos
        drift_velocity = (
            speed_fact
            * self.speed_mean  # type: ignore[reportGeneralTypeIssues]
            * (direction / np.linalg.norm(direction, ord=2))
        )

        super().update(
            dt=dt,
            skip_checks=True,
            drift_velocity=drift_velocity,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
            **kwargs,
        )

    def set_current_arm(self):
        """Sets which arm the agent will navigate to, this run."""

        # randomly choose a current target arm
        arms = ["left", "right"]
        self.target = arms[np.random.rand() > self.left_arm_prop]  # type: ignore[reportGeneralTypeIssues]

        if not hasattr(self, "trajectory_targets"):
            self.trajectory_targets = list()
        self.trajectory_targets.append(self.target)

        return

    def reset(self):
        """Reset the agent to a random location."""

        super().reset()

        self.set_current_arm()

        return

    def set_all_positions(self):
        """
        Set all the positions for the agent.
        """

        self.start_position = self.format_position(self.Environment.T_start)

        # set reset positions
        self.left_reset_position = self.Environment.left_T_end
        self.right_reset_position = self.Environment.right_T_end
        self.reset_position = [
            self.format_position(reset_position)
            for reset_position in [self.left_reset_position, self.right_reset_position]
        ]
        self.reached_reset_position = list()

        # set target position
        target_arm = self.target_arm  # type: ignore[reportGeneralTypeIssues]
        target_location_prop_to_arm = self.target_location_prop_to_arm  # type: ignore[reportGeneralTypeIssues]
        if target_arm == "left":
            edge = self.Environment.left_T_end
        elif target_arm == "right":
            edge = self.Environment.right_T_end
        else:
            raise RuntimeError("Target must be 'left' or 'right'.")

        T_split = self.Environment.T_split
        self.target_position = [
            T_split[i] + (edge[i] - T_split[i]) * target_location_prop_to_arm
            for i in [0, 1]
        ]
        self.target_position = self.format_position(self.target_position)

        self.reached_target_position = list()
        self.steps_before_checking_for_target = 0

        # set initial position and velocity
        if self.start_position is not None:
            self.set_position_and_velocity(position=self.start_position, velocity=0)
            self.must_fix_velocity_record = False

    def check_if_reset_position_reached(self, position: str = "both") -> bool:
        """Check if the agent has reached either of the reset positions.

        Returns: Whether the agent has reached either of the reset positions.
        """

        # calculate the distance between the current position and the reset position
        if position == "both":
            distances = list()
            for reset_position in self.reset_position:  # type: ignore[reportGeneralTypeIssues]
                distances.append(np.linalg.norm(self.pos - reset_position, ord=2))
            distance = min(distances)
        elif position == "left":
            distance = np.linalg.norm(self.pos - self.left_reset_position, ord=2)
        elif position == "right":
            distance = np.linalg.norm(self.pos - self.right_reset_position, ord=2)
        else:
            raise ValueError("pos must be 'both', 'left', or 'right'.")

        # check if the distance is less than the tolerance
        if distance < (self.dt * self.reset_reached_within_tolerance_prop_to_dt):  # type: ignore[reportGeneralTypeIssues]
            return True

        return False

    def check_if_left_reset_position_reached(self) -> bool:
        return self.check_if_reset_position_reached(position="left")

    def check_if_right_reset_position_reached(self) -> bool:
        return self.check_if_reset_position_reached(position="right")


class BoxAgent(ResetableAgent, util.ParamsManagerMixin):
    """Extend the reset agent so that it operates in an exploration box"""

    default_params = {
        "reward_factor": 5,  # factor for setting a reward object as a target for a trajectory
        "no_target_factor": 1,  # factor for not setting any target for a trajectory
        "trajectory_length": 2000,  # int or iterable of ints
        "num_trajectories": 10,  # number of trajectory lengths to sample
        "wait_between_targets": 10,  # number of steps to wait between target reaching
        "target_reached_within_tolerance_prop_to_dt": 0.5,  # proportion of dt to use as target tolerance
        "num_random_walk_steps": 100,  # number of steps to random walk, if target is not in sight
        "always_log_teleportation": False,  # whether to log teleportation events when they occur
    }

    ignored_param_keys = [
        "reset_position",
        "start_position",
        "target_position",
        "reset_reached_within_tolerance_prop_to_dt",
        "fixed_direction",
    ]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Env: env.ExploreBox, params: dict[str, Any] = dict()):
        """Initialise the agent.

        Args:
            params (dict, optional): Parameters for the agent. Defaults to {}.

        Raises:
            ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.check_if_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)
        self.params.update(params)

        if not isinstance(Env, env.ExploreBox):
            raise TypeError("Env must be an ExploreBox object.")

        self.set_fixed_params()

        super().__init__(Env, self.params)

    def get_targets_and_probabilities(
        self,
        skip_object_position: np.ndarray[tuple[int], np.dtype[np.float64]]
        | None = None,
    ) -> tuple[
        list[np.ndarray[tuple[int], np.dtype[np.float64]]] | list[None],
        list[int],
        list[float],
        list[str],
    ]:
        """Get the targets and the probabilities to use when sampling from them."""

        reward_num = self.Environment.type_name_to_num_dict["reward"]
        novel_num = self.Environment.type_name_to_num_dict["novel"]

        objects = [None]
        obj_types = [-1]
        obj_weights = [self.no_target_factor]  # type: ignore[reportGeneralTypeIssues]
        obj_names = ["no target"]
        for obj, obj_type in zip(
            self.Environment.objects["objects"],
            self.Environment.objects["object_types"],
        ):
            if obj_type not in [reward_num, novel_num]:
                continue
            elif (
                skip_object_position is not None and (obj == skip_object_position).all()
            ):
                continue
            objects.append(obj)
            obj_types.append(obj_type)
            obj_weight = self.reward_factor if obj_type == reward_num else 1  # type: ignore[reportGeneralTypeIssues]
            obj_weights.append(obj_weight)
            obj_name = "reward" if obj_type == reward_num else "novel"
            obj_names.append(obj_name)

        div = sum(obj_weights)
        obj_weights = [obj_wei / div for obj_wei in obj_weights]

        return objects, obj_types, obj_weights, obj_names

    def set_current_target(self):
        """Set the current target. Resets the timer"""

        objects, obj_types, obj_weights, obj_names = self.get_targets_and_probabilities(
            self.target_position
        )
        target_idx = np.random.choice(len(objects), 1, p=np.asarray(obj_weights))[0]

        # sample an object to go toward (check if in FOV, 5 attempts, otherwise no target for x steps)
        self.target = obj_names[target_idx]
        self.target_position = objects[target_idx]
        self.target_type = obj_types[target_idx]

        if not hasattr(self, "trajectory_targets"):
            self.trajectory_targets = list()
        self.trajectory_targets.append(
            (self.target, self.target_position, self.target_type)
        )

        self.steps_before_checking_for_target = 0

    def check_if_target_is_in_sight(self) -> bool:
        """Check if the target is in sight.

        Returns:
            bool: Whether the target is in sight.
        """

        # check if the target is in the field of view
        dist = self.Environment.get_distances_between___accounting_for_environment(
            self.pos, self.target_position, wall_geometry="line_of_sight"
        )

        if dist == 1000:
            return False
        else:
            return True

    def set_random_walk(self):
        """Set the random walk."""

        if self.target_position is None or not self.check_if_target_is_in_sight():
            self.current_num_of_random_walk_steps = int(self.num_random_walk_steps)  # type: ignore[reportGeneralTypeIssues]
        else:
            self.current_num_of_random_walk_steps = 0

    def set_all_positions(self, first_setting: bool = True):
        """
        Set all the positions for the agent.

        Args:
            first_setting (bool, optional): Whether this is the first setting.
                Defaults to True.
        """

        # set initial position and velocity
        self.start_position = self.Environment.sample_coords()
        self.set_position_and_velocity(position=self.start_position, velocity=0)
        if first_setting:
            self.must_fix_velocity_record = False

        self.target_position = None
        self.set_current_target()

        self.steps_before_checking_for_target = 0
        self.set_random_walk()

        if first_setting:
            self.reached_target_position = list()
            self.teleported = list()
            self.teleport_pair_nums = list()
            self.start_positions = list()

        self.start_positions.append(self.start_position)

    def reset(self):
        """Reset the agent to a random location."""

        super().reset()

        self.set_all_positions(first_setting=False)

        if (
            len(self.reached_target_position) == 0
            or self.num_steps_total != self.reached_target_position[-1]
        ):
            self.reached_target_position.append(-1)

        return

    def log_teleportation(self, last=False):
        """Log the teleportation events.

        Args:
            last (bool, optional): Whether to log only the last teleportation event.
                Defaults to False.
        """

        if len(self.teleported) == 0:
            log_str = "No teleportation events."

        elif last:
            pair_num = self.teleport_pair_nums[-1]
            step_num = self.teleported[-1]
            if step_num == len(self.history["t"]):
                seconds = self.history["t"][-1] + self.dt
            else:
                seconds = self.history["t"][step_num]
            log_str = (
                f"Teleported through pair {pair_num} at step {step_num} "
                f"({seconds:.2f} sec.)"
            )

        else:
            teleport_str = "    \n".join(
                [
                    f"through pair {pair_num} at step {step} ({self.history['t'][step]:.2f} sec.)"
                    for step, pair_num in zip(self.teleported, self.teleport_pair_nums)
                ]
            )
            log_str = f"Teleportation events:\n    {teleport_str}"

        print(log_str)

    def sample_within_tolerance(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]],
        sample_within_tolerance_prop_to_dt: float | None = None,
        max_attempts: int = 100,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Sample a position within the tolerance of the given position.

        Args:
            position (np.ndarray): The position to sample around.
            sample_within_tolerance_prop_to_dt (float): The proportion of the tolerance to
                sample within. Defaults to None, in which case the agent's
                target_reached_within_tolerance_prop_to_dt is used.

        Returns:
            position (np.ndarray): The sampled position.
        """

        if len(position) != 2:
            raise ValueError(f"position must have length 2, but found {len(position)}.")

        if sample_within_tolerance_prop_to_dt is None:
            sample_within_tolerance_prop_to_dt = self.target_reached_within_tolerance_prop_to_dt  # type: ignore[reportGeneralTypeIssues]

        new_position = super().sample_within_tolerance(
            position,
            sample_within_tolerance_prop_to_dt=sample_within_tolerance_prop_to_dt,
            max_attempts=max_attempts,
        )

        return new_position

    def get_teleport_vector(
        self, teleport_pair_num: int = 0, direction: str = "in"
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the teleport vector for the given teleport pair.

        Args:
            teleport_pair_num (str): The teleport pair to get the vector for.

        Returns:
            np.ndarray: The teleport vector.
        """

        marker = self.Environment.get_teleport_pair_marker(
            teleport_pair_num, direction=direction
        )

        x, y = 0, 0
        if marker == "<":  # towards right
            x = 1
        elif marker == ">":  # towards left
            x = -1
        elif marker == "^":
            y = -1  # below
        elif marker == "v":
            y = 1  # above
        else:
            raise RuntimeError(f"Unrecognized marker {marker}.")

        teleport_vector = np.asarray([x, y])

        return teleport_vector

    def check_teleport_angles(
        self,
        teleport_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        teleport_coords: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        check_value: str | np.ndarray[tuple[int], np.dtype[np.float64]] = "position",
    ) -> bool:
        """Check if the agent is within range for the teleportation to activate.

        Args:
            teleport_vector (np.ndarray): The teleport vector.

        Returns:
            bool: Whether the agent is within the teleport vector.
        """

        velocity = False
        if isinstance(check_value, str):
            if check_value == "position":
                # in the right area wrt teleport location
                check_value = self.pos  # type: ignore[reportGeneralTypeIssues]
            elif check_value == "velocity":
                # heading towards teleport
                check_value = -self.velocity  # type: ignore[reportGeneralTypeIssues]
                velocity = True
            else:
                raise ValueError(f"Unrecognized check_value {check_value}.")

        if not velocity:
            if teleport_coords is None:
                raise ValueError(
                    "teleport_coords must be specified if check_value is not 'velocity'."
                )
            check_value = check_value - teleport_coords

        norm_teleport_vector = teleport_vector / np.linalg.norm(teleport_vector)
        norm_check = np.asarray(check_value).astype(float) / np.linalg.norm(check_value)

        if np.dot(norm_teleport_vector, norm_check) > 0.707:  # 45 degrees, either side
            return True
        else:
            return False

    def check_if_teleport_in_should_activate(self, teleport_pair_num: int = 0) -> bool:
        """Check if the agent is in the right situation for the teleportation to activate.

        Args:
            teleport_pair_num (int): The teleport pair to check.

        Returns:
            bool: Whether the agent should teleport.
        """

        teleport_coords = self.Environment.teleport_pairs_dict[teleport_pair_num]["in"][
            1
        ]

        tolerance_prop_to_dt = self.target_reached_within_tolerance_prop_to_dt  # type: ignore[reportGeneralTypeIssues]

        # check if close to teleport in
        near_teleport = self.check_if_position_reached(
            teleport_coords, tolerance_prop_to_dt
        )
        if not near_teleport:
            teleport_angles = False

        else:
            # check if agent is within 45 degrees, either side of the teleport in
            teleport_vector = self.get_teleport_vector(
                teleport_pair_num, direction="in"
            )
            teleport_angles = self.check_teleport_angles(
                teleport_vector, teleport_coords, check_value="position"
            )

        if not teleport_angles:
            heading_teleport = False

        else:
            # check if agent is heading towards teleport in
            heading_teleport = self.check_teleport_angles(
                teleport_vector, check_value="velocity"
            )

        teleport = heading_teleport

        return teleport

    def sample_teleport_out_position(
        self, teleport_pair_num: int = 0, max_attempts: int = 100
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Sample a position within the tolerance of the teleportation out coordinates.

        Args:
            teleport_pair_num (int): The teleport pair to sample from.
            max_attempts (int): The maximum number of attempts to make at sampling.

        Returns:
            np.ndarray: The sampled position.
        """

        teleport_coords = self.Environment.teleport_pairs_dict[teleport_pair_num][
            "out"
        ][1]
        teleport_vector = self.get_teleport_vector(teleport_pair_num, direction="out")

        tolerance_prop_to_dt = self.target_reached_within_tolerance_prop_to_dt  # type: ignore[reportGeneralTypeIssues]

        i = 0
        out_coords = None
        all_samp = []
        while out_coords is None:
            sampled_out_coords = self.sample_within_tolerance(
                teleport_coords, tolerance_prop_to_dt
            )
            for x in [1, -1]:
                for y in [1, -1]:
                    coords_diff = sampled_out_coords - teleport_coords
                    check_coords = teleport_coords + coords_diff * np.asarray([x, y])
                    all_samp.append(check_coords)
                    # position coordinates on the correct side of the teleport
                    in_range = self.check_teleport_angles(
                        teleport_vector, teleport_coords, check_value=check_coords
                    )
                    if in_range:
                        out_coords = check_coords
                        break
            if i > max_attempts:
                raise RuntimeError(
                    f"Could not find a suitable out teleportation coordinate for "
                    f"teleport pair {teleport_pair_num}."
                )
            i += 1

        return out_coords

    def get_teleport_out_position(
        self, teleport_pair_num: int = 0
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the teleport out position based on the teleport in position and the agent's current position.

        Args:
            teleport_pair_num (int, optional): The teleport pair to use. Defaults to 0.

        Returns:
            np.ndarray: The teleport out position.
        """

        # get the teleport input info
        teleport_in_coords = self.Environment.teleport_pairs_dict[teleport_pair_num][
            "in"
        ][1]
        teleport_in_vector = self.get_teleport_vector(teleport_pair_num, direction="in")

        # get the teleport output info
        teleport_out_coords = self.Environment.teleport_pairs_dict[teleport_pair_num][
            "out"
        ][1]
        teleport_out_vector = self.get_teleport_vector(
            teleport_pair_num, direction="out"
        )

        # get the output vector
        out_vector = util.rotate_to(
            in_vector=self.pos - teleport_in_coords,
            in_basis=teleport_in_vector,
            out_basis=teleport_out_vector,
        )

        out_coords = teleport_out_coords + out_vector

        if not self.Environment.check_if_position_is_in_environment(out_coords):
            raise RuntimeError(
                "Teleport out position is not in the environment. "
                "Teleport coordinates may be too close to a wall."
            )

        return out_coords

    def rotate_velocity(
        self, teleport_pair_num: int = 0
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Rotate the agent's velocity to match the teleportation.

        Args:
            teleport_pair_num (int, optional): The teleport pair to use. Defaults to 0.

        Returns:
            np.ndarray: The output velocity.
        """

        teleport_in_vector = self.get_teleport_vector(teleport_pair_num, direction="in")
        teleport_out_vector = self.get_teleport_vector(
            teleport_pair_num, direction="out"
        )

        out_velocity = util.rotate_to(
            in_vector=self.velocity,  # type: ignore[reportGeneralTypeIssues]
            in_basis=teleport_in_vector,
            out_basis=-teleport_out_vector,
        )

        return out_velocity

    def teleport_if_applicable(self, sample: bool = False) -> bool:
        """Check if the agent should teleport."""

        teleport = False
        for teleport_pair_num in self.Environment.teleport_pairs_dict.keys():
            teleport = self.check_if_teleport_in_should_activate(teleport_pair_num)

            if not teleport:
                continue

            # teleport (sampling near out teleport coords)
            if sample:
                out_coords = self.sample_teleport_out_position(teleport_pair_num)
            else:
                out_coords = self.get_teleport_out_position(teleport_pair_num)

            # rotate with respect to teleport in and out vectors
            out_velocity = self.rotate_velocity(teleport_pair_num)

            # retain rotational velocity
            rotational_velocity = self.rotational_velocity

            self.set_position_and_velocity(
                position=out_coords,
                velocity=out_velocity,
                rotational_velocity=rotational_velocity,
            )

            self.teleported.append(self.num_steps_total)
            self.teleport_pair_nums.append(teleport_pair_num)

            if self.always_log_teleportation:
                self.log_teleportation(last=True)

            break

        teleported = teleport

        return teleported

    def update(
        self,
        dt: float | None = None,
        speed_fact: int | float = 3,
        drift_to_random_strength_ratio: float = 0.7,
        **kwargs,
    ):
        """Update the agent, optionally with a new position and velocity.

        See Agent.update() in ratinabox/agent.py for kwargs.

        Args:
            dt (float): The time step to use.
            speed_fact (float): The speed factor.
            drift_to_random_strength_ratio (float): The ratio of the drift strength to
                the random walk strength.
            **kwargs: Keyword arguments for Agent.update().
        """

        target_reached = self.check_and_record_target_reached()

        if self.check_if_trajectory_end_reached():
            self.reset()
        elif target_reached:
            self.set_current_target()
            self.set_random_walk()

        self.teleport_if_applicable()

        if self.current_num_of_random_walk_steps == 0:
            if self.target_position is None:
                self.set_current_target()
            self.set_random_walk()

        # calculate drift_velocity
        if self.current_num_of_random_walk_steps > 0:
            drift_velocity = None
            self.current_num_of_random_walk_steps -= 1
        else:
            direction = np.asarray(self.target_position) - self.pos
            drift_velocity = (
                speed_fact
                * self.speed_mean  # type: ignore[reportGeneralTypeIssues]
                * (direction / np.linalg.norm(direction, ord=2))
            )

        super().update(
            dt=dt,
            skip_checks=True,
            drift_velocity=drift_velocity,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
            **kwargs,
        )

    def get_trajectory_nodes(
        self,
    ) -> tuple[
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
        np.ndarray[tuple[int], np.dtype[np.int64]],
        np.ndarray[tuple[int], np.dtype[np.int64]],
        np.ndarray[tuple[int, int], np.dtype[np.float64]],
    ]:
        """Get the trajectory nodes.

        Returns:
            nodes (1d array): Nodes that were reached.
            values (1d array): Values of the nodes that were reached.
                1: target, 0: start and -1: non target end
            steps (1d array): Number of steps at which each node was reached.
            unreached_targets (1d array): Targets that were not reached
                (corresponding to -1 values)
        """

        pos = np.asarray(self.history["pos"])

        targets = [
            target[1] for target in self.trajectory_targets if target[1] is not None
        ]
        targets = np.asarray(targets)
        traj_lengs = np.cumsum(self.get_trajectory_lengths_to_date())
        start_position = np.asarray([self.start_positions[0]])

        reached_target_position = np.asarray(self.reached_target_position)
        reached_targets_idxs = np.where(reached_target_position != -1)[0]
        reached_target_position = reached_target_position[reached_targets_idxs]

        reached_targets = np.zeros(len(targets)).astype(bool)
        reached_targets[reached_targets_idxs] = True
        unreached_targets = targets[~reached_targets]
        targets = targets[reached_targets]

        # get start and end nodes
        nodes = np.insert(targets, 0, start_position[0], axis=0)
        values = np.insert(np.ones(len(nodes) - 1), 0, 0)
        steps = np.insert(reached_target_position, 0, 0)
        for l, leng in enumerate(traj_lengs):
            if leng in reached_target_position:
                continue
            idx = np.where(leng < reached_target_position)[0]
            if len(idx):
                # add the end node
                nodes = np.insert(nodes, idx[0], pos[leng - 1 : leng], axis=0)
                values = np.insert(values, idx[0], -1)
                steps = np.insert(steps, idx[0], leng)
                # add the start node
                nodes = np.insert(
                    nodes, idx[0] + 1, start_position[l + 1 : l + 2], axis=0
                )
                values = np.insert(values, idx[0], 0)
                steps = np.insert(steps, idx[0], leng + 1)
            elif self.trajectory_targets[-1][0] != "no target":
                # add the end node
                nodes = np.append(nodes, pos[-2:-1], axis=0)
                values = np.append(values, -1)
                steps = np.append(steps, leng)

        if len(unreached_targets) != len(np.where(values == -1)[0]):
            raise RuntimeError("Wrong number of reset points found.")

        values = values.astype(np.int64)
        steps = steps.astype(np.int64)

        return nodes, values, steps, unreached_targets

    def plot_trajectory(
        self,
        t_end: bool | None = None,
        target_alpha: float = 0.7,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the trajectory.

        Args:
            target_alpha (float, optional): Alpha value of the targets.
            no_legend (bool, optional): Whether to remove the legend. Defaults to False.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            fig (mpl_figure.Figure): Figure.
            ax (plt.Axes): Axes.
        """
        fig, ax = super().plot_trajectory(
            t_end=t_end, target_alpha=target_alpha, autosave=False, **kwargs
        )

        if no_legend and ax.get_legend():
            ax.get_legend().remove()

        self.add_target(ax, t=t_end)

        rutils.save_figure(fig, "trajectory", save=autosave)

        return fig, ax

    def plot_trajectory_targets(
        self,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        alpha: float = 0.8,
        plot_env: bool = True,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the trajectory targets.

        Args:
            fig (mpl_figure.Figure, optional): Figure to plot on.
            ax (plt.Axes, optional): Axes to plot on.
            alpha (float, optional): Alpha value of the targets.
            plot_env (bool, optional): Whether to plot the environment.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            fig (mpl_figure.Figure): Figure with the plot.
            ax (plt.Axes): Axes with the plot.
        """

        if fig is None or ax is None or plot_env:
            fig, ax = self.Environment.plot_environment(fig=fig, ax=ax)

        if ax is None:
            raise RuntimeError("ax is None.")

        targets = [
            target[1] for target in self.trajectory_targets if target[1] is not None
        ]

        unique_targets, counts = list(), []
        for target in targets:
            present = [
                (target == unique_target).all() for unique_target in unique_targets
            ]
            if sum(present):
                counts[present.index(True)] += 1
            else:
                unique_targets.append(target)
                counts.append(1)

        for target, count in zip(unique_targets, counts):
            # write the number of times the target was visited
            ax.text(
                target[0],
                target[1],
                str(count),
                horizontalalignment="left",
                verticalalignment="bottom",
                color="white",
                fontsize=10,
                zorder=10,
                fontweight="bold",
            )

        if len(targets) == 0:
            return fig, ax

        nodes, values, steps, unreached_targets = self.get_trajectory_nodes()

        # get linewidths
        step_diff = np.diff(steps)
        lws = list()
        if len(step_diff) != 0:
            min_val, max_val = np.min(step_diff), np.max(step_diff)
            lws = np.around((step_diff - min_val) / (max_val - min_val), 1) + 1

        unreached = 0
        for n, node in enumerate(nodes[:-1]):
            ax.plot(
                [node[0], nodes[n + 1][0]],
                [node[1], nodes[n + 1][1]],
                color="black",
                linewidth=lws[n],
                alpha=alpha,
                zorder=1,
            )

            # add missed targets
            if values[n] == -1:
                ax.plot(
                    [nodes[n + 1][0], unreached_targets[unreached][0]],
                    [nodes[n + 1][1], unreached_targets[unreached][1]],
                    color="black",
                    ls="dashed",
                    alpha=alpha,
                    zorder=1,
                )
                unreached += 1

        if no_legend and ax.get_legend():
            ax.get_legend().remove()

        rutils.save_figure(fig, "trajectory_targets", save=autosave)

        return fig, ax

    def plot_trajectory_targets_over_time(
        self,
        t_start: float = 0.0,
        t_end: float | None = None,
        fig: mpl_figure.Figure | None = None,
        ax: plt.Axes | None = None,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> tuple[mpl_figure.Figure, plt.Axes]:
        """Plot the trajectory targets over time.

        Args:
            t_start (float, optional): Start time of the plot.
            t_end (float, optional): End time of the plot.
            fig (mpl_figure.Figure, optional): Figure to plot on.
            ax (plt.Axes, optional): Axes to plot on.
            no_legend (bool, optional): Whether to remove the legend. Defaults to False.
            autosave (bool, optional): Whether to autosave the figure. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            fig (mpl_figure.Figure): Figure with the plot.
            ax (plt.Axes): Axes with the plot.
        """

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 3))

        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        if t_end == None:
            t_end = t[-1]
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end))) + 1

        t = t[startid:endid]
        pos = pos[startid:endid]

        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")

        # plot reset points as vertical dashed lines
        reset_times = self.get_reset_times()
        for reset_time in reset_times:
            if reset_time >= t_start and reset_time <= t_end:
                ax.axvline(reset_time, color="black", ls="dashed", alpha=0.2, zorder=-1)

        # plot trajectory
        ax.plot(t, pos[:, 0], color="lightgray", label="X")
        ax.plot(t, pos[:, 1], color="darkgray", label="Y")

        ax.set_title("Position over time")
        ax.set_ylabel("Position")
        ax.set_xlabel("Time (s)")

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        # plot teleportation points as vertical dashed lines
        for step, pair in zip(self.teleported, self.teleport_pair_nums):
            if step < startid or step > endid:
                continue

            obj_type = self.Environment.teleport_pairs_dict[pair]["in"][0]
            color = self.Environment.type_num_to_plot_params_dict[obj_type]["color"]
            ax.axvline(t[step], color=color, ls="dashed", alpha=0.8, zorder=-1)

        # plot target objects
        targets, target_types = zip(
            *[
                (target[1], target[2])
                for target in self.trajectory_targets
                if target[1] is not None
            ]
        )

        # get the number of steps to the target
        num_steps = np.asarray(self.reached_target_position)
        if len(num_steps) < len(targets):
            num_steps = np.append(num_steps, -1)
        if len(num_steps) != len(targets):
            raise RuntimeError("Cannot match targets to number of steps to reach.")

        type_num_to_plot_params_dict = copy.deepcopy(
            self.Environment.type_num_to_plot_params_dict
        )
        r = 0
        reset_times = np.append(reset_times, t[-1])
        for target, target_type, num_step in zip(targets, target_types, num_steps):
            plot_params = type_num_to_plot_params_dict[target_type]
            label = None
            if "name" in plot_params:
                label = plot_params.pop("name")
                plot_params["markersize"] = plot_params.pop("s") / 8
            if num_step == -1:
                target_time = reset_times[r]
                alpha = 0.3
                r += 1
            else:
                target_time = t[num_step]
                alpha = 0.8
            if target_time < t_start or target_time > t_end:
                continue
            ax.plot(
                [target_time] * 2,
                target,
                **plot_params,
                label=label,
                lw=1.5,
                alpha=alpha,
            )

        ax.legend(
            loc="center left", fontsize="medium", bbox_to_anchor=(1, 0.5), frameon=False
        )

        if no_legend and ax.get_legend():
            ax.get_legend().remove()

        rutils.save_figure(fig, "trajectory_targets_over_time", save=autosave)

        return fig, ax

    def add_target(
        self,
        ax: plt.Axes,
        t: float | None = None,
    ):
        """
        Add the target for time t to the plot.

        Args:
            ax: The axis object.
            t: The current time step.
        """

        all_t = np.array(self.history["t"])
        if t is None:
            t = float(all_t[-1])
        endid = np.argmin(np.abs(all_t - (t))) + 1

        targets = [
            target[1] for target in self.trajectory_targets if target[1] is not None
        ]
        num_steps = np.asarray(self.reached_target_position)
        above = np.where(num_steps > endid)[0]
        target = targets[above[0]] if len(above) else None

        if target is None:
            return

        ax.scatter(
            *target,
            marker=markers.MarkerStyle("x"),
            s=60,
            zorder=4,
            color="red",
            label="target",
            lw=3,
            alpha=0.6,
        )

    def animate_trajectory(
        self, additional_plot_func: Callable | None = None, **kwargs
    ) -> animation.FuncAnimation:
        """
        Animate the trajectory of the agent.

        Args:
            additional_plot_func: A function that is called after each frame of the
                animation is plotted. It takes that takes in fig, ax, t and **kwargs
                and returns fig, ax.
            **kwargs: Additional keyword arguments passed to self.plot_trajectory().

        Returns:
           matplotlib.animation.FuncAnimation: The animation object.
        """

        def run_all_additional_plot_funcs(
            fig: mpl_figure.Figure,
            ax: plt.Axes,
            t: float | None = None,
            **kwargs,
        ) -> tuple[mpl_figure.Figure, plt.Axes]:
            """
            Run all additional plot functions.

            Args:
                fig: The figure object.
                ax: The axis object.
                t: The current time step.

            Returns:
                tuple[mpl_figure.Figure, plt.Axes]: The figure and axis objects.
            """

            if additional_plot_func is not None:
                fig, ax = additional_plot_func(fig=fig, ax=ax, t=t, **kwargs)

            self.add_target(ax, t=t)
            plot_util.remove_prev_handle_labels(ax)

            return fig, ax

        anim = super().animate_trajectory(
            additional_plot_func=run_all_additional_plot_funcs, **kwargs
        )

        return anim
