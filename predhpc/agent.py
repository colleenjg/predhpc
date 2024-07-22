from typing import Any, TYPE_CHECKING, Callable
import warnings

import copy
from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import animation, markers
from matplotlib import colors as mpl_colors
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from ratinabox import Agent as riabAgent  # type: ignore[import]
from ratinabox import utils as rutils

from predhpc import env, util, plot_util


class ResetableAgent(riabAgent):
    """Extend the agent so that is has an optimal maximum trajectory length after which
    it resets to a random location

    default_params = {
        "dt": 0.01,  # time step, in seconds
        "trajectory_length": None,  # int or iterable of ints
        "num_trajectories": None,  # number of trajectory lengths to sample
        "exp_factors": None,  # exponential factors for trajectory_length (inv. scale, rate, minimum). Default is None.
        "random_max": None,  # max value for randomizing trajectory_length
        "start_position": None,  # position to start trajectories from
        "reset_position": None,  # position to reset trajectories from
        "target_position": None,  # position to use as target
        "wait_between_targets": 10,  # number of steps to wait between target reaching
        "reset_reached_within_tolerance_prop_to_speed_dt": 0.55,  # proportion of current speed to use as reset tolerance
        "target_reached_within_tolerance_prop_to_speed_dt": 0.55,  # proportion of current speed to use as target tolerance
        "fixed_direction": False,  # keep same direction (1D environment only)
    }
    """

    default_params = {
        "dt": 0.1,  # time step, in seconds
        "head_direction_smoothing_timescale": 0.2,  # higher than dt
        "trajectory_length": None,  # int or iterable of ints
        "num_trajectories": None,  # number of trajectory lengths to sample
        "exp_factors": None,  # exponential factors for trajectory_length (inv. scale, rate, minimum). Default is None.
        "random_max": None,  # max value for randomizing trajectory_length
        "start_position": None,  # position to start trajectories from
        "reset_position": None,  # position to reset trajectories from
        "target_position": None,  # position to use as target
        "wait_between_targets": 10,  # number of steps to wait between target reaching
        "reset_reached_within_tolerance_prop_to_speed_dt": 0.55,  # proportion of current speed * dt to use as reset tolerance
        "target_reached_within_tolerance_prop_to_speed_dt": 0.55,  # proportion of current speed * dt to use as target tolerance
        "fixed_direction": False,  # keep same direction (1D environment only)
    }

    def __init__(self, Env: "env.Environment", params: dict[str, Any] = dict()):
        """Initialise the agent.

        Args:
        - Env (Environment): The environment in which the agent is placed.
        - params (dict, optional): Parameters for the agent. Default is dict().

        Raises:
        - ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        with warnings.catch_warnings():
            if Env.dimensionality == "1D" and self.params["fixed_direction"]:  # type: ignore[attr-defined]
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Warning: You have solid 1D boundary",
                )
            super().__init__(Env, self.params)

        if self.Environment.dimensionality == "2D":
            self.fixed_direction = False

        self.set_trajectory_lengths()
        self.set_all_positions()
        self.set_target_df()
        self.set_trajectory_df()

        if self.Environment.dimensionality == "1D" and self.fixed_direction:
            if np.sign(self.reset_position - self.start_position) != np.sign(self.speed_mean):  # type: ignore[attr-defined]
                raise ValueError(
                    "If direction is fixed, speed must have the same sign."
                )

        self._last_target_reached_step = -1
        self._last_stop_step = -1

    def set_target_df(self):
        """Set the target dataframe, which records the target position and the
        step at which it was reached.
        """

        self.target_df = pd.DataFrame(columns=self.target_df_columns)
        self.set_new_target()

    @property
    def target_df_columns(self) -> list:
        """Get the target dataframe columns.

        Returns:
        - (list) List of column names.
        """

        if not hasattr(self, "_target_df_columns"):
            self._target_df_columns = [
                "position_x",
                "set_step",
                "reached_step",
            ]

            if self.Environment.D == 2:
                self._target_df_columns.append("position_x")

        return self._target_df_columns

    def set_trajectory_df(self):
        """Set the trajectory dataframe, which records the trajectory start and stop
        position, and the number of steps it lasted.
        """

        self.trajectory_df = pd.DataFrame(columns=self.trajectory_df_columns)
        self.set_new_trajectory()

    @property
    def trajectory_df_columns(self) -> list:
        """Get the target dataframe columns.

        Returns:
        - (list) List of column names.
        """

        if not hasattr(self, "_trajectory_df_columns"):
            trajectory_df_columns = ["start_position_x"]

            if self.Environment.D == 2:
                trajectory_df_columns.append("start_position_y")

            trajectory_df_columns.extend(["start_step", "stop_position_x"])

            if self.Environment.D == 2:
                trajectory_df_columns.append("stop_position_y")

            trajectory_df_columns.extend(["stop_step", "num_steps_total"])

            self._trajectory_df_columns = trajectory_df_columns

        return self._trajectory_df_columns

    def set_new_trajectory(self):
        """Add start information for a new trajectory."""

        trajectory_data = {
            "start_position_x": self.pos[0],
            "start_step": self.num_steps_total,
        }

        if self.Environment.D == 2:
            trajectory_data["start_position_y"] = self.pos[1]

        self.trajectory_df.loc[len(self.trajectory_df)] = trajectory_data  # type: ignore[assignment]

    def end_trajectory(self):
        """Add stop information for a trajectory that is ending."""

        idx = len(self.trajectory_df) - 1
        start_step = int(self.trajectory_df.loc[idx, "start_step"])  # type: ignore[assignment]

        self.trajectory_df.loc[idx, "stop_position_x"] = self.pos[0]
        self.trajectory_df.loc[idx, "stop_step"] = self.num_steps_total
        self.trajectory_df.loc[idx, "num_steps_total"] = (
            self.num_steps_total - start_step
        )

        if self.Environment.D == 2:
            self.trajectory_df["stop_position_y"] = self.pos[1]

    def set_new_target(self):
        if self.target_position is None:
            return

        target_data = {
            "position_x": self.target_position[0],
            "set_step": self.num_steps_total,
        }

        if self.Environment.D == 2:
            target_data["position_y"] = self.target_position[1]

        self.target_df.loc[len(self.target_df)] = target_data  # type: ignore[assignment]

    def set_all_positions(self):
        """Set all positions, checking that they are within the environment
        extent.
        """

        self.start_position = self.format_position(self.start_position)
        self.reset_position = self.format_position(self.reset_position)
        self.set_target_position(self.target_position)

        self.must_fix_record_after_manual_update = False
        if self.start_position is not None:
            self.set_position_and_velocity(position=self.start_position, velocity=0)

    def set_target_position(self, position):
        """Set the target position, checking that it is within the environment
        extent.
        """

        self.target_position = self.format_position(position)

        if self.target_position is None:
            return

        target_position = np.asarray(self.target_position).reshape(
            1, self.Environment.D
        )
        if hasattr(self, "_target_object_idx"):
            self.Environment.objects["objects"][
                self._target_object_idx
            ] = target_position
        else:
            self.Environment.add_object(position, "new")
            self._target_object_idx = len(self.Environment.objects["objects"]) - 1

        self.steps_before_checking_for_target = 0

    def reverse(self, reset=False):
        """Reverse the agent's start and reset positions."""

        new_reset_pos, new_start_pos = self.start_position, self.reset_position
        self.start_position = new_start_pos
        self.reset_position = new_reset_pos

        if self.Environment.D == 1:
            self.speed_mean = -self.speed_mean

        if reset:
            self.reset()

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
                exp_factors=self.exp_factors,  # type: ignore[attr-defined]
                random_max=self.random_max,  # type: ignore[attr-defined]
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
        position: (
            np.ndarray[tuple[int], np.dtype[np.float64]] | list[float] | None
        ) = None,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Formats positions, if applicable, and returns them.

        Args:
        - position (np.ndarray | list | None, optional): Position to format.
            Default is None.

        Raises:
        - ValueError: If position is not within the environment extent.

        Returns:
        - position (np.ndarray or None): Formatted position.
        """

        if position is not None:
            position = np.asarray(position).ravel()
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

    def fix_record_after_manual_update(self, dt: float | None = None):
        """Fix values computed and recorded based on the velocity, if applicable."""

        if not hasattr(self, "recorded_stats"):
            return

        self.recorded_stats = {
            "average_measured_speed": self.average_measured_speed,  # type: ignore[has-type]
            "distance_travelled": self.distance_travelled,  # type: ignore[has-type]
        }

        if dt is None:
            dt = float(self.dt)  # type: ignore[has-type]

        velocity = np.asarray(self.velocity).astype(np.float64)  # type: ignore[has-type]

        tau_speed = 10
        self.average_measured_speed = self.recorded_stats[
            "average_measured_speed"
        ] + dt / tau_speed * (  # type: ignore[has-type]
            np.linalg.norm(velocity, ord=2)
        )

        self.save_velocity = velocity

        self.distance_travelled = (
            self.recorded_stats["distance_travelled"]
            + np.linalg.norm(velocity, ord=2) * dt
        )

        if self.save_history is True and len(self.history["vel"]):  # type: ignore[attr-defined]
            self.history["vel"][-1] = self.save_velocity.tolist()
            if self.Environment.dimensionality == "2D":
                rotational_velocity = float(self.rotational_velocity)  # type: ignore[has-type]
                self.history["rot_vel"][-1] = rotational_velocity

    def check_and_fix_velocity_for_1D(
        self,
        prev_velocity: np.ndarray[tuple[int], np.dtype[np.float64]],
        dt: float | None = None,
    ):
        """Check if velocity is negative and fix if applicable."""

        if not (self.Environment.dimensionality == "1D" and self.fixed_direction):
            return

        if np.sign(self.reset_position - self.start_position) == np.sign(self.velocity):  # type: ignore[has-type]
            return

        if self.reset_position is not None:
            returning = np.sign(self.reset_position[0] - self.pos[0]) == np.sign(self.velocity[0])  # type: ignore[has-type]
            if returning:
                return

        if dt is None:
            dt = self.dt  # type: ignore[has-type]

        new_velocity = self.velocity  # type: ignore[has-type]
        speed_mean, speed_std = self.speed_mean, self.speed_std  # type: ignore[attr-defined]
        for _ in range(10):
            # resample velocity until it has the correct sign
            if np.sign(self.reset_position - self.start_position) != np.sign(new_velocity):  # type: ignore[has-type]
                new_velocity = prev_velocity + rutils.ornstein_uhlenbeck(
                    dt=dt,
                    x=prev_velocity,
                    drift=speed_mean,
                    noise_scale=speed_std,
                    coherence_time=self.speed_coherence_time,  # type: ignore[attr-defined]
                )
            else:
                break

        if np.sign(self.reset_position - self.start_position) != np.sign(new_velocity):  # type: ignore[has-type]
            new_velocity = prev_velocity * 0  # set to 0

        self.velocity = new_velocity
        self.fix_record_after_manual_update(dt=dt)

    def set_position_and_velocity(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        velocity: float | np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        rotational_velocity: float | None = 0.0,
        sample: bool = True,
    ):
        """Set the position and velocity of the agent.

        From Agent.__init__() in ratinabox/agent.py
        """

        # initialise starting positions and velocity

        speed_mean, speed_std = self.speed_mean, self.speed_std  # type: ignore[attr-defined]
        if self.Environment.dimensionality == "2D":
            if position is None:
                self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            elif sample:
                self.pos = self.sample_within_tolerance(position)
            else:
                self.pos = position
            if velocity is None or len(np.asarray(velocity).ravel()) == 1:
                direction = np.random.uniform(0, 2 * np.pi)
                velocity = speed_std * np.array([np.cos(direction), np.sin(direction)])
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

        self.must_fix_record_after_manual_update = True

        return

    def sample_within_tolerance(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]],
        sample_within_tolerance_prop_to_speed_dt: float = 0.5,
        max_attempts: int = 100,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Sample a position within the tolerance of the given position.

        Args:
        - position (np.ndarray): The position to sample around.
        - sample_within_tolerance_prop_to_speed_dt (float): The proportion of the tolerance to
            sample within. Default is None, in which case the agent's
            target_reached_within_tolerance_prop_to_speed_dt is used.

        Returns:
        - position (np.ndarray): The sampled position.
        """

        tolerance = np.absolute(self.speed_mean) * self.dt * sample_within_tolerance_prop_to_speed_dt  # type: ignore[has-type]

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
                f"{sample_within_tolerance_prop_to_speed_dt} of {position}."
            )

        return new_position

    def get_trajectory_lengths_to_date(self):
        """Return the trajectory lengths to date.

        Returns:
        - trajectory_lengths_to_date (list): Trajectory lengths to date.
        """

        trajectory_lengths_to_date = (
            self.trajectory_df["num_steps_total"].to_numpy().copy()
        )

        if np.isnan(trajectory_lengths_to_date[-1]):
            last_start = int(
                self.trajectory_df.loc[len(self.trajectory_df) - 1, "start_step"]  # type: ignore[assignment]
            )
            last_length = self.num_steps_total - last_start
            if last_length == 0:
                trajectory_lengths_to_date = trajectory_lengths_to_date[:-1]
            else:
                trajectory_lengths_to_date[-1] = last_length

        trajectory_lengths_to_date = trajectory_lengths_to_date.astype(int)

        return trajectory_lengths_to_date

    def log_trajectories_to_date(self):
        """Log the trajectory lengths to date."""
        trajectory_lengths_to_date = self.get_trajectory_lengths_to_date()
        print(
            f"Trajectory lengths ({len(trajectory_lengths_to_date)}) to date "
            f"(in steps): {trajectory_lengths_to_date}"
        )

    def log_trajectory_stats_to_date(self, log_as_time: bool = True):
        """Log the trajectory length statistics to date."""

        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        traj_length_unit = "steps"

        # get trajectory lengths in seconds
        if log_as_time:
            traj_leng_to_date = [leng * self.dt for leng in traj_leng_to_date]  # type: ignore[has-type]
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
    ) -> plt.Axes:
        """Plot the trajectory lengths to date.

        Args:
        - in_minutes (bool, optional): Whether to plot in minutes. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.
        """

        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        sub_ax, _ = plot_util.plot_trajectory_lengths(
            dt=self.dt, trajectory_lengths=traj_leng_to_date, in_minutes=in_minutes  # type: ignore[has-type]
        )

        fig = sub_ax.figure
        util.save_figure(fig, "trajectories_to_date", save=autosave)

        return sub_ax

    def get_reset_times(self):
        """Get the reset times.

        Returns:
        - reset_times (1D np.ndarray): Reset times.

        Raises:
        - ValueError: If agent does not have reset steps.
        """

        reset_steps = self.trajectory_df["stop_step"].to_numpy()
        if np.isnan(reset_steps[-1]):
            reset_steps = reset_steps[:-1]

        reset_times = reset_steps * self.dt

        return reset_times

    def check_if_position_reached(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        sample_within_tolerance_prop_to_speed_dt: float = 0.55,
    ) -> bool:
        """Check if the agent has reached a position.

        Args:
        - target_position (np.array): Target position.
        - sample_within_tolerance_prop_to_speed_dt (float):
            Tolerance proportion, wrt mean speed * dt.

        Returns:
        - position_reached (bool): Whether the agent has reached the target position.
        """

        position_reached = False

        if position is not None:
            # calculate the distance between the current position and the reset position
            dist = np.linalg.norm(self.pos - position, ord=2)

            # check if the distance is less than the tolerance
            speed = np.linalg.norm(self.velocity, ord=2)
            reached_dist = speed * self.dt * sample_within_tolerance_prop_to_speed_dt
            if dist < reached_dist:  # type: ignore[has-type]
                position_reached = True

        return position_reached

    def check_if_reset_position_reached(self) -> bool:
        """Check if the agent has reached the reset position.

        Returns:
        - position_reached (bool): Whether the agent has reached the reset position.
        """

        position_reached = self.check_if_position_reached(
            self.reset_position, self.reset_reached_within_tolerance_prop_to_speed_dt  # type: ignore[attr-defined]
        )

        return position_reached

    def check_if_target_position_reached(self) -> bool:
        """Check if the agent has reached the target position.

        Returns:
        - target_reached (bool): Whether the agent has reached the target position.
        """

        if self.target_position is None:
            target_reached = False

        elif self.steps_before_checking_for_target > 0:
            self.steps_before_checking_for_target -= 1
            target_reached = False

        else:
            target_reached = self.check_if_position_reached(
                self.target_position, self.target_reached_within_tolerance_prop_to_speed_dt  # type: ignore[attr-defined]
            )
            if target_reached:
                self.steps_before_checking_for_target = self.wait_between_targets  # type: ignore[attr-defined]

        return target_reached

    def check_if_trajectory_end_reached(self) -> bool:
        """Check if the agent has reached the end of its trajectory.

        Returns:
        - self.reached_end (bool): Whether the agent has reached the end of its
            trajectory.
        """

        self.reached_end = False
        if self.reset_position is not None and self.check_if_reset_position_reached():
            # record the time step at which the agent reached the reset position
            if self.num_steps_total == self._last_stop_step:
                self.reached_end = False
            else:
                self.reached_end = True

        if self.trajectory_length is not None:
            if self.current_trajectory_length >= self.trajectory_length:
                self.reached_end = True

        if self.reached_end:
            self._last_stop_step = self.num_steps_total

        return self.reached_end

    def check_and_record_target_reached(self) -> bool:
        """Check if the agent has reached the target in its trajectory.

        Returns:
        - self.reached_target (bool): Whether the agent has reached the target in its
            trajectory.
        """

        self.reached_target = False
        if self.target_position is not None and self.check_if_target_position_reached():
            # record the time step at which the agent reached the target position
            if self.num_steps_total == self._last_target_reached_step:
                self.reached_target = False
            else:
                self.reached_target = True

        if self.reached_target:
            df_idx = len(self.target_df) - 1
            self._last_target_reached_step = self.num_steps_total
            self.target_df.loc[df_idx, "reached_step"] = self.num_steps_total

            num_steps = self.num_steps_total - self.target_df.loc[df_idx, "set_step"]  # type: ignore[operator]
            self.target_df.loc[df_idx, "num_steps"] = num_steps
            self.set_new_target()

        return self.reached_target

    def reset(self):
        """Reset the agent."""

        self.end_trajectory()

        self.set_position_and_velocity(position=self.start_position, velocity=0)

        if self.trajectory_lengths is not None:
            i = (len(self.trajectory_df) - 1) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]

        self.current_trajectory_length = 0

        self.set_new_trajectory()

        return

    def update(
        self,
        dt: float | None = None,
        skip_checks: bool = False,
        new_pos: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        **kwargs,
    ):
        """Update the agent, optionally with a new position and velocity.

        See Agent.update() in ratinabox/agent.py for kwargs.
        """

        if not skip_checks:
            self.check_and_record_target_reached()
            if self.check_if_trajectory_end_reached():
                self.reset()

        self.recorded_stats = {
            "average_measured_speed": self.average_measured_speed,
            "distance_travelled": self.distance_travelled,
            "position": self.pos,
            "velocity": self.velocity,
        }

        if new_pos is not None:
            if dt is not None:
                self.dt = dt
            self.t += self.dt
            if not self.Environment.check_if_position_is_in_environment(new_pos):
                raise ValueError(
                    f"New position {new_pos} is not within the environment."
                )
            self.pos = new_pos
            self.must_fix_record_after_manual_update = True

            # write to history
            if self.save_history is True:  # type: ignore[attr-defined]
                self.save_to_history()
        else:
            super().update(dt=dt, **kwargs)

        if self.Environment.dimensionality == "1D" and self.fixed_direction:
            self.check_and_fix_velocity_for_1D(
                prev_velocity=self.recorded_stats["velocity"], dt=dt
            )

        elif self.must_fix_record_after_manual_update:
            self.fix_record_after_manual_update(dt=dt)

        self.must_fix_record_after_manual_update = False

        self.current_trajectory_length += 1
        self.num_steps_total += 1

    def plot_distance_to(
        self,
        position_name: str = "target",
        position: None | np.ndarray[tuple[int], np.dtype[np.float64]] = None,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        alpha: float = 0.8,
        color: str = "k",
        tolerance_prop_to_speed_dt: float | None = None,
        zoom_prop: float | None = None,
        mark_below_tolerance: bool = False,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """Plot the distance to a position across the agent's history."""

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)

        t = t[startid : endid + 1] / 60
        positions = np.asarray(self.history["pos"])[startid : endid + 1]

        if position is None:
            if position_name == "target":
                position = self.target_position
                if tolerance_prop_to_speed_dt is None:
                    tolerance_prop_to_speed_dt = self.target_reached_within_tolerance_prop_to_speed_dt  # type: ignore[attr-defined]
            elif position_name == "reset":
                position = self.reset_position
                if tolerance_prop_to_speed_dt is None:
                    tolerance_prop_to_speed_dt = self.reset_reached_within_tolerance_prop_to_speed_dt  # type: ignore[attr-defined]
            elif position_name == "start":
                position = self.start_position
            else:
                raise ValueError(
                    f"Can only infer `position` from `position_name` if the latter is "
                    f"'target', 'reset' or 'start', but got {position_name}."
                )
            if position is None:
                raise ValueError(f"{position_name} is set to None.")

        position = self.format_position(position)

        distances = np.linalg.norm(positions - position, ord=2, axis=1)

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(8, 2))

        sub_ax.plot(t, distances, alpha=alpha, lw=1, color=color)
        if tolerance_prop_to_speed_dt is not None:
            speed = np.linalg.norm(np.asarray(self.history["vel"]), ord=2, axis=1)
            speed = speed[startid : endid + 1]
            y = speed * self.dt * tolerance_prop_to_speed_dt  # type: ignore[attr-defined]
            sub_ax.plot(t, y, color=color, alpha=alpha / 2, lw=0.5)
            if zoom_prop is not None:
                sub_ax.set_ylim(0, y.max() * zoom_prop)
            if mark_below_tolerance:
                below = np.where(distances < y)[0]
                if len(below):
                    sub_ax.scatter(
                        t[below], distances[below], color=color, marker=".", s=12
                    )

        elif zoom_prop or mark_below_tolerance:
            raise ValueError(
                "Cannot zoom or mark below the tolerance if a tolerance is not provided."
            )

        sub_ax.set_xlabel("Time / min")
        sub_ax.set_ylabel(f"Distance to {position_name} / m")

        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["top"].set_visible(False)

        fig = sub_ax.figure
        util.save_figure(fig, "distance", save=autosave)

        return sub_ax

    def plot_trajectory_resets(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        framerate: int | float = 10,
        sub_ax: plt.Axes | None = None,
        alpha: float = 0.6,
        color: str = "k",
        ms: int | float = 45,
        plot_targets: bool = True,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """Plots the trajectory between t_start (seconds) and t_end (defaulting to the
        last time available)

        From Agent.plot_1D_trajectories() in ratinabox/agent.py. Modified to enable
        plotting of reset steps, and use of colormaps for trajectories.

        Args:
        - t_start: start time in seconds. (default = self.history["t"][0])
        - t_end: end time in seconds (default = self.history["t"][-1])
        - framerate: how many scatter points / per second of motion to display
        - sub_ax (plt.Axes): Subplot to plot on. If None, a new subplot is created.
        - alpha: trajectory point opaqness
        - color: trajectory point color
        - ms: plot point size
        - plot_targets: whether to plot the target
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectory resets plotted.
        """

        dt = self.dt
        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t_start, t_end = t[startid], t[endid]

        skiprate = max(1, int((1 / framerate) / dt))

        t = t / 60  # minutes
        time = t[startid : endid + 1][::skiprate]
        pos = pos[startid : endid + 1][::skiprate]

        # get reset step indices
        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")
        elif len(time) == 0:
            raise RuntimeError("Duration too short. No time points to plot.")

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(8, 5))

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

        reached_reset_position = self.trajectory_df["stop_step"].to_numpy()
        if np.isnan(reached_reset_position[-1]):
            reached_reset_position = reached_reset_position[:-1]
        reached_reset_position = reached_reset_position.astype(int)

        alpha /= self.Environment.D
        alpha_pts = 0.9 / self.Environment.D
        for i in range(self.Environment.D):
            sub_ax.scatter(
                time,
                pos,
                alpha=alpha,
                marker=markers.MarkerStyle("."),
                color=color,
                s=ms / 8,
            )

            if len(reached_reset_position):
                if self.start_position is not None:
                    x_start = [
                        t[x]
                        for x in reached_reset_position
                        if x >= startid and x < endid
                    ]
                    y_start = [self.start_position[i]] * len(x_start)
                    sub_ax.scatter(
                        x_start,
                        y_start,
                        marker=markers.MarkerStyle("^"),
                        color="gold",
                        alpha=alpha_pts,
                        s=ms / 3,
                    )

                if self.reset_position is not None:
                    x_reset = [
                        t[x - 1]
                        for x in reached_reset_position
                        if x >= startid and x < endid
                    ]
                    y_reset = [self.reset_position[i]] * len(x_reset)
                    sub_ax.scatter(
                        x_reset,
                        y_reset,
                        marker=markers.MarkerStyle("x"),
                        color="red",
                        alpha=alpha_pts,
                        s=ms / 2.5,
                    )

            if plot_targets and self.target_position is not None:
                target_reached_step = self.target_df["reached_step"].to_numpy()
                if np.isnan(target_reached_step[-1]):
                    target_reached_step = target_reached_step[:-1]

                if len(target_reached_step) != 0:
                    target_reached_step = target_reached_step.astype(int)

                    x_targ = [
                        t[x] for x in target_reached_step if x >= startid and x < endid
                    ]
                    y_targ = [self.target_position[i]] * len(x_targ)
                    sub_ax.scatter(
                        x_targ,
                        y_targ,
                        marker=markers.MarkerStyle("."),
                        color="blue",
                        alpha=alpha_pts,
                        s=ms,
                    )

        sub_ax.set_xlabel("Time / min")
        sub_ax.set_ylabel("Position / m")

        bottom = min_y - diff * 0.1  # type: ignore[operator]
        top = max_y + diff * 0.1  # type: ignore[operator]
        sub_ax.set_ylim(bottom=bottom, top=top)
        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["top"].set_visible(False)

        fig = sub_ax.figure
        util.save_figure(fig, "trajectory_resets", save=autosave)

        return sub_ax

    def plot_trajectory(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        framerate: int | float = 10,
        sub_ax: plt.Axes | None = None,
        decay_point_size: bool = False,
        plot_agent: bool = True,
        agent_color: str = "r",
        colormap: str | None | mpl_colors.Colormap = None,
        alpha: float = 0.7,
        xlim: float | None = None,
        background_color: str | None = None,
        plot_traj_ends: bool = True,
        target_alpha: float = 1.0,
        plot_target: bool = True,
        cmap_per: bool = False,
        scale_cmap_per: bool = False,
        ms_2D: int | float = 15,
        size_fact: float | None = None,
        autosave: bool | None = None,
        **kwargs,  # hacky catch-all...
    ) -> plt.Axes:
        """Plots the trajectory between t_start (seconds) and t_end (defaulting to the
        last time available)

        From Agent.plot_trajectory() in ratinabox/agent.py. Modified to enable plotting
        of reset steps, and use of colormaps for trajectories.

        Args:
        - t_start: start time in seconds (default = self.history["t"][0])
        - t_end: end time in seconds (default = self.history["t"][-1])
        - framerate: how many scatter points / per second of motion to display
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None,
            a new subplot is created using self.Environment.plot_environment().
            This can be used to plot trajectory on top of receptive fields etc.
        - decay_point_size: decay trajectory point size over time
           (recent times = largest)
        - plot_agent: dedicated point show agent current position
        - agent_color: color of agent point
        - colormap: colormap to use to plot trajectories
        - alpha: plot point opaqness
        - xlim: In 1D, forces the top (right) xlim to be a certain time (minutes)
           (useful if animating this function)
        - background_color: color of the background if not matplotlib default, only
            for 1D (probably white)
        - plot_traj_ends: plot a point at the end of each trajectory
        - target_alpha: transparency with which to plot target position
        - plot_targets: plot target position
        - cmap_per: if True, the colormap is used to set the color for each time
            point. Otherwise, each trajectory has its own color.
        - scale_cmap_per: if True, and cmap_per is True, the full range of the
            colormap is used for each trajectory, regardless of its length
        - ms_2D: the size of the points in the 2D plot is set to this value.
        - size_fact: if not None, the size of the points is multiplied by this value.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectory plotted.
        """

        dt = self.dt
        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t_start, t_end = t[startid], t[endid]

        if t_start is None or t_end is None:
            raise RuntimeError("t_start or t_end is None.")

        skiprate = max(1, int((1 / framerate) / dt))
        t = t / 60  # minutes

        if self.Environment.dimensionality == "2D":
            trajectory = pos[startid : endid + 1, :][::skiprate]
        elif self.Environment.dimensionality == "1D":
            trajectory = pos[startid : endid + 1][::skiprate]
        else:
            raise RuntimeError(f"Environment dimensionality must be either 1D or 2D.")
        time = t[startid : endid + 1][::skiprate]

        # get reset step indices
        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")
        elif len(time) == 0:
            raise RuntimeError("Duration too short. No time points to plot.")

        trajectory_lengths = self.get_trajectory_lengths_to_date()
        traj_idx = [np.full(steps, i) for i, steps in enumerate(trajectory_lengths)]
        if cmap_per:
            if scale_cmap_per:
                cmap_vals = [np.linspace(0, 1, steps) for steps in trajectory_lengths]
            else:
                cmap_vals = [
                    np.arange(steps, dtype=np.int64) for steps in trajectory_lengths
                ]
        else:
            cmap_vals = traj_idx[:]
        cmap_vals_np = np.concatenate(cmap_vals).astype(float)[startid : endid + 1][
            ::skiprate
        ]
        cmap_min, cmap_max = cmap_vals_np.min(), cmap_vals_np.max()
        if cmap_min == cmap_max:
            cmap_vals_np[:] = 0.5  # mid point of the colormap
        else:
            cmap_vals_np = (cmap_vals_np - cmap_min) / (cmap_max - cmap_min)

        traj_idx = np.concatenate(traj_idx).astype(int)[startid : endid + 1][::skiprate]

        if colormap is None:
            colormap = "crest"
        c = sns.color_palette(colormap, as_cmap=True)(cmap_vals_np)  # type: ignore[callable]
        ##############################

        if self.Environment.dimensionality == "2D":
            if size_fact is not None:
                extent = self.Environment.extent
                x_base = extent[1] - extent[0]
                y_base = extent[3] - extent[2]
                figsize = (size_fact * x_base, size_fact * y_base)
                _, sub_ax = plt.subplots(figsize=figsize)

            sub_ax = self.Environment.plot_environment(sub_ax=sub_ax)

            if sub_ax is None:
                raise RuntimeError("sub_ax is None.")

            if plot_target and self.target_position is not None:
                sub_ax.scatter(
                    *self.target_position,
                    marker=".",
                    color="blue",
                    s=18,
                    zorder=5,
                    alpha=target_alpha,
                    label="target",
                )

            s = ms_2D * np.ones_like(time)
            if decay_point_size == True:
                s = ms_2D * np.exp((time - time[-1]) / 10)
                s[(time[-1] - time) > ms_2D] *= 0

            if plot_traj_ends == True and len(self.trajectory_df) - 1 > 0:
                ends = np.where(np.diff(traj_idx) > 0)[0]
                ends = np.append(ends, len(trajectory) - 1)
                s[ends] = ms_2D * 2
                # set last colormap value to dark red
                c[ends] = mpl_colors.to_rgba("darkred")  # type: ignore[arg-type]

            if plot_agent == True:
                s[-1] = ms_2D * 2.75
                # set last colormap value to red
                c[-1] = mpl_colors.to_rgba(agent_color)  # type: ignore[arg-type]

            sub_ax.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                s=s,
                alpha=alpha,
                zorder=2,
                c=c,
                linewidth=0,
            )
        if self.Environment.dimensionality == "1D":
            if sub_ax is None:
                _, sub_ax = plt.subplots(figsize=(3, 1.5))
            sub_ax.scatter(time / 60, trajectory, alpha=alpha, linewidth=0, c=c, s=5)
            sub_ax.spines["left"].set_position(("data", t_start / 60))
            sub_ax.set_xlabel("Time / min")
            sub_ax.set_ylabel("Position / m")
            sub_ax.set_xlim(t_start / 60, t_end / 60)
            if xlim is not None:
                sub_ax.set_xlim(right=xlim)

            sub_ax.set_ylim(bottom=0, top=self.Environment.extent[1])
            sub_ax.spines["right"].set_visible(False)
            sub_ax.spines["top"].set_visible(False)
            sub_ax.set_xticks([t_start / 60, t_end / 60])
            ex = self.Environment.extent
            sub_ax.set_yticks([ex[1]])
            if background_color is not None:
                sub_ax.set_facecolor(background_color)
                sub_ax.figure.patch.set_facecolor(background_color)  # type: ignore[attr-defined]

        if sub_ax is None:
            raise RuntimeError("sub_ax is None.")

        fig = sub_ax.figure
        util.save_figure(fig, "trajectory", save=autosave)

        return sub_ax

    def plot_trajectory_edges(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        decay_point_size: bool = False,
        plot_agent: bool = True,
        colormap: str | None | mpl_colors.Colormap = None,
        alpha: float = 0.7,
        xlim: float | None = None,
        background_color: str | None = None,
        plot_starts: bool = True,
        plot_ends: bool = True,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """Plots the trajectory starts and ends between t_start (seconds) and t_end
        (defaulting to the last time available)

         Args:
         - t_start: start time in seconds
         - t_end: end time in seconds (default = self.history["t"][-1])
         - sub_ax (plt.Axes, optional): Subplot to plot on. If None,
             a new subplot is created using self.Environment.plot_environment().
             This can be used to plot trajectory on top of receptive fields etc.
         - decay_point_size: decay trajectory point size over time
            (recent times = largest)
         - plot_agent: dedicated point show agent current position
         - colormap: colormap to use to plot trajectories starts/ends
         - alpha: plot point opaqness
         - xlim: In 1D, forces the top (right) xlim to be a certain time (minutes)
            (useful if animating this function)
         - background_color: color of the background if not matplotlib default,
             only for 1D (probably white)
         - plot_starts: plot trajectory starts
         - plot_ends: plot trajectory ends
         - autosave (bool, optional): Whether to autosave the figure. Default is None.

         Returns:
         - sub_ax (plt.Axes): Subplot with trajectory edges plotted.
        """

        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t_start, t_end = t[startid], t[endid]

        if t_start is None or t_end is None:
            raise RuntimeError("t_start or t_end is None.")

        t = t / 60  # minutes

        if colormap is None:
            colormap = "crest"
        cmap = sns.color_palette(colormap, as_cmap=True)

        actual_trajectory_lengths = self.get_trajectory_lengths_to_date()
        all_ends = np.cumsum(actual_trajectory_lengths)
        start_c, end_c = None, None
        traj_starts, traj_ends = None, None
        if plot_starts:
            traj_starts = np.insert(all_ends, 0, 0)[:-1]
            start_c = cmap(np.linspace(0, 1, len(traj_starts)))  # type: ignore[callable]
        if plot_ends:
            traj_ends = all_ends - 1
            end_c = cmap(np.linspace(0, 1, len(traj_ends)))  # type: ignore[callable]
        if not (plot_starts or plot_ends):
            raise ValueError(
                "At least one of 'plot_starts' or 'plot_ends' must be True."
            )

        for c, traj_idx, marker in [
            (start_c, traj_starts, "^"),
            (end_c, traj_ends, "x"),
        ]:
            if c is None or traj_idx is None:
                continue
            lw = 2 if marker == "x" else 1
            traj_idx = traj_idx[(traj_idx >= startid) & (traj_idx <= endid)]
            trajectory = pos[traj_idx]
            time = t[traj_idx]

            if len(time) == 0:
                raise RuntimeError("Duration too short. No trajectory points to plot.")

            if self.Environment.dimensionality == "2D":
                sub_ax = self.Environment.plot_environment(sub_ax=sub_ax)
                if sub_ax is None:
                    raise RuntimeError("sub_ax is None.")
                if self.target_position is not None:
                    sub_ax.scatter(
                        *self.target_position,
                        marker=markers.MarkerStyle("^"),
                        color="gold",
                        s=20,
                        zorder=5,
                    )

                s = 15 * np.ones_like(time)
                if decay_point_size == True:
                    s = 15 * np.exp((time - time[-1]) / 10)
                    s[(time[-1] - time) > 15] *= 0

                if plot_agent == True:
                    s[-1] = 40
                    # set last colormap value to red
                    c[-1] = mpl_colors.to_rgba("r")  # type: ignore[arg-type]

                sub_ax.scatter(
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
                if sub_ax is None:
                    _, sub_ax = plt.subplots(figsize=(3, 1.5))
                sub_ax.scatter(
                    time / 60,
                    trajectory,
                    alpha=alpha,
                    linewidth=lw,
                    c=c,
                    s=5,
                    marker=marker,
                )
                sub_ax.spines["left"].set_position(("data", t_start / 60))
                sub_ax.set_xlabel("Time / min")
                sub_ax.set_ylabel("Position / m")
                sub_ax.set_xlim(t_start / 60, t_end / 60)
                if xlim is not None:
                    sub_ax.set_xlim(right=xlim)

                sub_ax.set_ylim(bottom=0, top=self.Environment.extent[1])
                sub_ax.spines["right"].set_visible(False)
                sub_ax.spines["top"].set_visible(False)
                sub_ax.set_xticks([t_start / 60, t_end / 60])
                ex = self.Environment.extent
                sub_ax.set_yticks([ex[1]])
                if background_color is not None:
                    sub_ax.set_facecolor(background_color)
                    sub_ax.figure.patch.set_facecolor(background_color)  # type: ignore[attr-defined]

        if sub_ax is None:
            raise RuntimeError("sub_ax is None.")

        fig = sub_ax.figure
        util.save_figure(fig, "trajectory_edges", save=autosave)

        return sub_ax


class TAgent(ResetableAgent, util.ParamsManagerMixin):
    """Extend the reset agent so that it operates in a T maze"""

    default_params = {
        "target_arm": "left",  # type: ignore[dict-item]
        "target_location_prop_to_arm": 0.75,  # proportion down arm at which to set target
        "left_arm_prop": 0.75,  # proportion of trajectories to target to left arm
    }

    ignored_param_keys = ["reset_position", "start_position", "target_position"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Env: env.TEnv, params: dict[str, Any] = dict()):
        """Initialise the agent.

        Args:
        - params (dict, optional): Parameters for the agent. Default is {}.

        Raises:
        - ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        if not isinstance(Env, env.TEnv):
            raise TypeError("Env must be a TEnv object.")

        super().__init__(Env, self.params)

        self.set_current_trajectory_arm()

    @property
    def near_branch(self) -> bool:
        return self.pos[1] > (self.Environment.branch_y * 0.98)

    @property
    def at_branch(self) -> bool:
        return self.pos[1] > self.Environment.branch_y

    @property
    def target_df_columns(self) -> list:
        """Get the target dataframe columns.

        Returns:
        - (list) List of column names.
        """

        if not hasattr(self, "_target_df_columns"):
            self._target_df_columns = [
                "target_arm",
                "position_x",
                "position_y",
                "set_step",
                "reached_step",
                "num_steps",
            ]

        return self._target_df_columns

    def set_current_trajectory_arm(self):
        # randomly choose a current target arm
        arms = ["left", "right"]
        arm_idx = int(np.random.rand() > self.left_arm_prop)  # type: ignore[attr-defined]
        self.current_arm = arms[arm_idx]

    def set_new_target(self):
        """Set a new target."""

        if self.target_position is None:
            raise RuntimeError("Target position is None.")

        target_data = {
            "target_arm": self.target_arm,  # type: ignore[attr-defined]
            "position_x": self.target_position[0],
            "position_y": self.target_position[1],
            "set_step": self.num_steps_total,
        }
        self.target_df.loc[len(self.target_df)] = target_data  # type: ignore[assignment]

    def get_direction(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the direction to the target."""

        if self.current_arm == "left":
            trajectory_end = self.Environment.left_T_end

        elif self.current_arm == "right":
            trajectory_end = self.Environment.right_T_end

        else:
            raise RuntimeError("Current arm must be 'left' or 'right'.")

        direction = np.asarray(trajectory_end) - self.pos

        return direction

    def update(  # type: ignore[override]
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
            * self.speed_mean  # type: ignore[attr-defined]
            * (direction / np.linalg.norm(direction, ord=2))
        )

        super().update(
            dt=dt,
            skip_checks=True,
            drift_velocity=drift_velocity,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
            **kwargs,
        )

    def reset(self):
        """Reset the agent."""

        super().reset()

        self.set_current_trajectory_arm()

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

        # set target position
        target_location_prop_to_arm = self.target_location_prop_to_arm  # type: ignore[attr-defined]
        T_split = self.Environment.T_split

        target_arm = self.target_arm  # type: ignore[attr-defined]
        if target_arm == "left":
            edge = self.Environment.left_T_end
        elif target_arm == "right":
            edge = self.Environment.right_T_end
        else:
            raise ValueError("target_arm must be 'left' or 'right'.")

        target_position = [
            T_split[i] + (edge[i] - T_split[i]) * target_location_prop_to_arm
            for i in [0, 1]
        ]
        self.target_position = self.format_position(target_position)
        self.Environment.add_object(self.target_position)

        self.steps_before_checking_for_target = 0

        # set initial position and velocity
        if self.start_position is not None:
            self.set_position_and_velocity(position=self.start_position, velocity=0)
            self.must_fix_record_after_manual_update = False

    def check_if_reset_position_reached(self, position: str = "both") -> bool:
        """Check if the agent has reached either of the reset positions.

        Returns:
        - reset_positon_reached (bool): Whether the agent has reached either of the
            reset positions.
        """

        # calculate the distance between the current position and the reset position
        if position == "both":
            distances = [
                np.linalg.norm(self.pos - reset_position, ord=2)  # type: ignore[operator]
                for reset_position in self.reset_position
            ]
            distance = min(distances)
        elif position == "left":
            distance = np.linalg.norm(self.pos - self.left_reset_position, ord=2)
        elif position == "right":
            distance = np.linalg.norm(self.pos - self.right_reset_position, ord=2)
        else:
            raise ValueError("pos must be 'both', 'left', or 'right'.")

        # check if the distance is less than the tolerance
        reset_positon_reached = False
        if distance < (np.absolute(self.speed_mean) * self.reset_reached_within_tolerance_prop_to_speed_dt):  # type: ignore[attr-defined]
            reset_positon_reached = True

        return reset_positon_reached

    def check_if_left_reset_position_reached(self) -> bool:
        return self.check_if_reset_position_reached(position="left")

    def check_if_right_reset_position_reached(self) -> bool:
        return self.check_if_reset_position_reached(position="right")


class OpenFieldAgent(ResetableAgent, util.ParamsManagerMixin):
    """Extend the reset agent so that it operates in an open field"""

    default_params = {
        "reward_factor": 5,  # factor for setting a reward object as a target for a trajectory
        "no_target_factor": 1,  # factor for not setting any target for a trajectory
        "trajectory_length": 2000,  # int or iterable of ints
        "num_trajectories": 10,  # number of trajectory lengths to sample
        "wait_between_targets": 10,  # number of steps to wait between target reaching
        "target_reached_within_tolerance_prop_to_speed_dt": 0.55,  # proportion of mean speed * dt to use as target tolerance
        "num_random_walk_steps": 100,  # number of steps to random walk, if target is not in sight
        "always_log_teleportation": False,  # whether to log teleportation events when they occur
    }

    ignored_param_keys = [
        "reset_position",
        "start_position",
        "target_position",
        "reset_reached_within_tolerance_prop_to_speed_dt",
        "fixed_direction",
    ]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Env: env.OpenField, params: dict[str, Any] = dict()):
        """Initialise the agent.

        Args:
        - params (dict, optional): Parameters for the agent. Default is {}.

        Raises:
        - ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        if not isinstance(Env, env.OpenField):
            raise TypeError("Env must be an OpenField object.")

        super().__init__(Env, self.params)

    @property
    def target_df_columns(self) -> list:
        """Get the target dataframe columns.

        Returns:
        - (list) List of column names.
        """

        if not hasattr(self, "_target_df_columns"):
            self._target_df_columns = [
                "object_idx",
                "object_type_name",
                "object_type_num",
                "position_x",
                "position_y",
                "set_step",
                "random_walk_periods",
                "reached_step",
                "num_steps",
                "num_random_walk_steps",
            ]

            # add a target in sight column with lists

        return self._target_df_columns

    def set_target_df(self):
        """Set the target dataframe, which records the target position and the
        step at which it was reached.
        """

        self.target_df = pd.DataFrame(columns=self.target_df_columns)

        self.target_df["random_walk_periods"] = self.target_df[
            "random_walk_periods"
        ].astype(object)

        self.set_new_target()
        self.set_random_walk()

    def get_target_probability_df(self) -> pd.DataFrame:
        """Get the target probability dataframe.

        Returns:
        - target_probability_df (pd.DataFrame): Dataframe with target probabilities.
        """
        target_probability_df = self.Environment.object_df.loc[
            ~self.Environment.object_df["object_type_name"].str.contains("teleport")
        ].copy()  # makes a copy

        target_probability_df.insert(0, "object_df_idx", target_probability_df.index)

        # reset index
        target_probability_df.reset_index(drop=True, inplace=True)

        # add a no target row
        target_probability_df.loc[len(target_probability_df), "object_type_name"] = (
            "no_target"
        )

        # add target probabilities
        target_probability_df.loc[:, "target_factor"] = 1
        target_probability_df.loc[
            target_probability_df["object_type_name"] == "reward", "target_factor"
        ] = self.reward_factor  # type: ignore[attr-defined]
        target_probability_df.loc[
            target_probability_df["object_type_name"] == "no_target", "target_factor"
        ] = self.no_target_factor  # type: ignore[attr-defined]

        target_probability_df.loc[:, "target_probability"] = (
            target_probability_df["target_factor"]
            / target_probability_df["target_factor"].sum()
        )

        return target_probability_df

    def _end_random_walk(self):
        """End the random walk."""

        if self.current_num_of_random_walk_steps == 0:
            return

        df_idx = len(self.target_df) - 1
        column = "random_walk_periods"
        self.target_df.loc[df_idx, column][-1].append(self.num_steps_total)
        self.current_num_of_random_walk_steps = 0

    def check_random_reached(self):
        """Check if the target was reached during a random walk."""

        if self.current_num_of_random_walk_steps != 0 and self.reached_target:
            self._end_random_walk()

    def end_trajectory(self):
        super().end_trajectory()

        self._end_random_walk()

    def set_new_target(self, target: str | None = None):
        """Set a new target."""

        # add random walk steps
        if len(self.target_df) != 0:
            self.check_random_reached()
            self._end_random_walk()
            random_walk_periods = np.asarray(
                self.target_df.loc[len(self.target_df) - 1, "random_walk_periods"]
            )
            num_random_walk_steps = 0
            if len(random_walk_periods) > 0:
                num_random_walk_steps = np.sum(np.diff(random_walk_periods, axis=1))
            self.target_df.loc[len(self.target_df) - 1, "num_random_walk_steps"] = (
                num_random_walk_steps
            )

        target_probability_df = self.get_target_probability_df()

        if target is None:
            object_weights = target_probability_df["target_probability"].values
            idx = np.random.choice(
                len(object_weights), 1, p=np.asarray(object_weights)
            )[0]
        else:
            rows = target_probability_df.loc[
                target_probability_df["object_type_name"] == target
            ]
            if len(rows) == 0:
                raise ValueError(f"No target of type {target} in the environment.")
            elif len(rows) > 1:
                raise RuntimeError(
                    f"More than one target of type {target} in the environment."
                )
            idx = rows.index[0]

        target_row = target_probability_df.loc[idx]

        if target_row["object_type_name"] == "no_target":
            target_data = {
                "object_type_name": "no_target",
            }
        else:
            target_data = {
                "object_idx": target_row["object_df_idx"],
                "object_type_name": target_row["object_type_name"],
                "object_type_num": target_row["object_type_num"],
                "position_x": target_row["position_x"],
                "position_y": target_row["position_y"],
            }

        target_data["set_step"] = self.num_steps_total  # type: ignore[assignment]
        target_data["random_walk_periods"] = list()  # type: ignore[assignment]

        new_idx = len(self.target_df)
        self.target_df.loc[new_idx] = target_data  # type: ignore[assignment]

        self.steps_before_checking_for_target = 0

        if target_row["object_type_name"] == "no_target":
            self.target_position = None
        else:
            self.target_position = np.asarray(
                [target_data["position_x"], target_data["position_y"]]
            )

    @property
    def teleportation_df(self):
        """Set the target dataframe, which records the target position and the
        step at which it was reached.
        """

        if not hasattr(self, "_teleportation_df"):
            teleportation_columns = [
                "teleport_pair_num",
                "step_num",
            ]

            for direction in ["in", "out"]:
                for key in [
                    "object_idx",
                    "object_type_num",
                    "object_type_name",
                    "position_x",
                    "position_y",
                    "vector_x",
                    "vector_y",
                    "velocity_x",
                    "velocity_y",
                ]:
                    teleportation_columns.append(f"{direction}_{key}")

            self._teleportation_df = pd.DataFrame(columns=teleportation_columns)

        return self._teleportation_df

    def check_if_target_is_in_sight(self) -> bool:
        """Check if the target is in sight.

        Returns:
        - target_in_sight (bool): Whether the target is in sight.
        """

        # check if the target is in the field of view
        dist = self.Environment.get_distances_between___accounting_for_environment(
            self.pos, self.target_position, wall_geometry="line_of_sight"
        )

        if dist == 1000:
            target_in_sight = False
        else:
            target_in_sight = True

        return target_in_sight

    def set_random_walk(self):
        """Set the random walk."""

        if self.target_position is None or not self.check_if_target_is_in_sight():
            self.current_num_of_random_walk_steps = int(self.num_random_walk_steps)  # type: ignore[attr-defined]
            df_idx = len(self.target_df) - 1
            column = "random_walk_periods"
            self.target_df.loc[df_idx, column].append([self.num_steps_total])  # type: ignore[has-method]
        else:
            self.current_num_of_random_walk_steps = 0

    def set_all_positions(self, first_setting: bool = True, target: str | None = None):
        """
        Set all the positions for the agent.

        Args:
        - first_setting (bool, optional): Whether this is the first setting.
            Default is True.
        - target (str, optional): The target to set. Ignore if first_setting is True.
            Default is None.
        """

        # set initial position and velocity
        self.start_position = self.Environment.sample_coords()
        self.set_position_and_velocity(position=self.start_position, velocity=0)
        if first_setting:
            self.must_fix_record_after_manual_update = False

        self.target_position = None
        if not first_setting:
            self.set_new_target(target=target)

        self.steps_before_checking_for_target = 0

        if not first_setting:
            self.set_random_walk()

    def log_teleportation(self, last=False):
        """Log the teleportation events.

        Args:
        - last (bool, optional): Whether to log only the last teleportation event.
            Default is False.
        """

        if len(self.teleportation_df) == 0:
            log_str = "No teleportation events."

        elif last:
            step_num = self.teleportation_df["step_num"].tolist()[-1]
            pair_num = self.teleportation_df["teleport_pair_num"].tolist()[-1]
            if step_num == len(self.history["t"]):
                seconds = self.history["t"][-1] + self.dt
            else:
                seconds = self.history["t"][step_num]
            log_str = (
                f"Teleported through pair {pair_num} at step {step_num} "
                f"({seconds:.2f} sec.)"
            )

        else:
            step_nums = self.teleportation_df["step_num"].tolist()
            pair_nums = self.teleportation_df["teleport_pair_num"].tolist()
            teleport_str = "    \n".join(
                [
                    f"through pair {pair_num} at step {step} ({self.history['t'][step]:.2f} sec.)"
                    for step, pair_num in zip(step_nums, pair_nums)
                ]
            )
            log_str = f"Teleportation events:\n    {teleport_str}"

        print(log_str)

    def sample_within_tolerance(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]],
        sample_within_tolerance_prop_to_speed_dt: float | None = None,
        max_attempts: int = 100,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Sample a position within the tolerance of the given position.

        Args:
        - position (np.ndarray): The position to sample around.
        - sample_within_tolerance_prop_to_speed_dt (float): The proportion of the tolerance to
            sample within. Default is None, in which case the agent's
            target_reached_within_tolerance_prop_to_speed_dt is used.

        Returns:
        - new_position (np.ndarray): The sampled position.
        """

        if len(position) != 2:
            raise ValueError(f"position must have length 2, but found {len(position)}.")

        if sample_within_tolerance_prop_to_speed_dt is None:
            prop_to_speed_dt = self.target_reached_within_tolerance_prop_to_speed_dt  # type: ignore[attr-defined]
        else:
            prop_to_speed_dt = sample_within_tolerance_prop_to_speed_dt

        new_position = super().sample_within_tolerance(
            position,
            sample_within_tolerance_prop_to_speed_dt=prop_to_speed_dt,
            max_attempts=max_attempts,
        )

        return new_position

    def get_teleport_vector(
        self, teleport_pair_num: int = 0, direction: str = "in"
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the teleport vector for the given teleport pair.

        Args:
        - teleport_pair_num (str): The teleport pair to get the vector for.

        Returns:
        - teleport_vector (np.ndarray): The teleport vector.
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
        - teleport_vector (np.ndarray): The teleport vector.

        Returns:
        - within_range (bool): Whether the agent is within the teleport vector.
        """

        velocity = False
        if isinstance(check_value, str):
            if check_value == "position":
                # in the right area wrt teleport location
                check_value = self.pos
            elif check_value == "velocity":
                # heading towards teleport
                check_value = -self.velocity
                velocity = True
            else:
                raise ValueError(f"Unrecognized check_value {check_value}.")

        check_value = np.array(check_value)  # copy

        if not velocity:
            if teleport_coords is None:
                raise ValueError(
                    "teleport_coords must be specified if check_value is not 'velocity'."
                )
            check_value -= teleport_coords

        norm_teleport_vector = teleport_vector / np.linalg.norm(teleport_vector)
        norm_check = np.asarray(check_value).astype(float) / np.linalg.norm(check_value)

        if np.dot(norm_teleport_vector, norm_check) > 0.707:  # 45 degrees, either side
            within_range = True
        else:
            within_range = False

        return within_range

    def check_if_teleport_in_should_activate(self, teleport_pair_num: int = 0) -> bool:
        """Check if the agent is in the right situation for the teleportation to activate.

        Args:
        - teleport_pair_num (int): The teleport pair to check.

        Returns:
        - teleport (bool): Whether the agent should teleport.
        """

        teleport_coords = self.Environment.get_teleport_coords(
            teleport_pair_num, direction="in"
        )

        tolerance_prop_to_speed_dt = self.target_reached_within_tolerance_prop_to_speed_dt  # type: ignore[attr-defined]

        teleport = False

        # check if close to teleport in
        near_teleport = self.check_if_position_reached(
            teleport_coords, tolerance_prop_to_speed_dt
        )
        if near_teleport:
            # check if agent is within 45 degrees, either side of the teleport in
            teleport_vector = self.get_teleport_vector(
                teleport_pair_num, direction="in"
            )
            teleport_angles = self.check_teleport_angles(
                teleport_vector, teleport_coords, check_value="position"
            )

            if teleport_angles:
                # check if agent is heading towards teleport in
                heading_teleport = self.check_teleport_angles(
                    teleport_vector, check_value="velocity"
                )

                if heading_teleport:
                    teleport = True

        return teleport

    def get_drift_velocity(
        self,
        pos: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        speed_fact: int | float = 3,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Get the drift velocity.

        Args:
        - pos (np.ndarray, optional): The position to use. Default is None.

        Returns:
        - drift_velocity (np.ndarray): The drift velocity.
        """

        if pos is None:
            pos = self.pos

        # calculate drift_velocity
        if self.current_num_of_random_walk_steps > 0:
            drift_velocity = None
        else:
            direction = np.asarray(self.target_position) - pos
            drift_velocity = (
                speed_fact
                * self.speed_mean  # type: ignore[attr-defined]
                * (direction / np.linalg.norm(direction, ord=2))
            )

        return drift_velocity

    def calculate_update(
        self,
        velocity: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        speed_fact: int | float = 3,
        drift_to_random_strength_ratio: float = 0.7,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Anticipate the agent's next update.

        Args:
        - velocity (np.ndarray, optional): The velocity to use. Default is None.

        Returns:
        - update_vector (np.ndarray): The update vector.
        """

        if velocity is None:
            velocity = self.velocity

        drift_velocity = self.get_drift_velocity(speed_fact=speed_fact)

        update_vector = util.get_update_vector(
            velocity,
            drift_velocity=drift_velocity,
            dt=self.dt,
            rotational_velocity=self.rotational_velocity,  # type: ignore[arg-type]
            rotational_velocity_coherence_time=self.rotational_velocity_coherence_time,  # type: ignore[attr-defined]
            speed_mean=self.speed_mean,  # type: ignore[attr-defined]
            speed_coherence_time=self.speed_coherence_time,  # type: ignore[attr-defined]
            rotational_velocity_std=self.rotational_velocity_std,  # type: ignore[attr-defined]
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
        )

        return update_vector

    def sample_teleport_out_position(
        self,
        teleport_pair_num: int = 0,
        max_attempts: int = 100,
        # adjust_backwards: bool = True,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Sample a position within the tolerance of the teleportation out coordinates.

        Args:
        - teleport_pair_num (int): The teleport pair to sample from.
        - max_attempts (int): The maximum number of attempts to make at sampling.
        - adjust_backwards (bool, optional): Whether to adjust the teleport out
            position to account for the agent's next update. Default is True.

        Returns:
        - out_coords (np.ndarray): The sampled position.
        """

        teleport_coords = self.Environment.get_teleport_coords(
            teleport_pair_num, direction="out"
        )

        teleport_vector = self.get_teleport_vector(teleport_pair_num, direction="out")

        tolerance_prop_to_speed_dt = self.target_reached_within_tolerance_prop_to_speed_dt  # type: ignore[attr-defined]

        i = 0
        out_coords = None
        while out_coords is None:
            sampled_out_coords = self.sample_within_tolerance(
                teleport_coords, tolerance_prop_to_speed_dt
            )
            for x, y in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                coords_diff = sampled_out_coords - teleport_coords
                check_coords = teleport_coords + coords_diff * np.asarray([x, y])
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
        self, teleport_pair_num: int = 0, adjust_backwards: bool = True
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the teleport out position based on the teleport in position and the agent's current position.

        Args:
        - teleport_pair_num (int, optional): The teleport pair to use. Default is 0.
        - adjust_backwards (bool, optional): Whether to adjust the teleport out
            position to account for the agent's next update. Default is True.

        Returns:
        - out_coords (np.ndarray): The teleport out position.
        """

        # get the teleport input info
        teleport_in_coords = self.Environment.get_teleport_coords(
            teleport_pair_num, direction="in"
        )
        teleport_in_vector = self.get_teleport_vector(teleport_pair_num, direction="in")

        # get the teleport output info
        teleport_out_coords = self.Environment.get_teleport_coords(
            teleport_pair_num, direction="out"
        )
        teleport_out_vector = self.get_teleport_vector(
            teleport_pair_num, direction="out"
        )

        # get the output vector
        out_vector = util.rotate_to(
            in_vector=self.pos - teleport_in_coords,
            in_basis=teleport_in_vector,  # type: ignore[arg-type]
            out_basis=teleport_out_vector,  # type: ignore[arg-type]
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
        - teleport_pair_num (int, optional): The teleport pair to use. Default is 0.

        Returns:
        - out_velocity (np.ndarray): The output velocity.
        """

        teleport_in_vector = self.get_teleport_vector(teleport_pair_num, direction="in")
        teleport_out_vector = self.get_teleport_vector(
            teleport_pair_num, direction="out"
        )

        out_velocity = -util.rotate_to(
            in_vector=self.velocity,
            in_basis=teleport_in_vector,  # type: ignore[arg-type]
            out_basis=teleport_out_vector,  # type: ignore[arg-type]
        )

        return out_velocity

    def teleport_coords_if_applicable(
        self,
        sample: bool = False,
        speed_fact: int | float = 3,
        drift_to_random_strength_ratio: float = 0.7,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Check if the agent should teleport."""

        out_coords = None
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

            in_coords = self.pos.copy()
            in_velocity = self.velocity.copy()

            # simulate update for records (as if the teleportation was a normal step)
            update = self.calculate_update(
                out_velocity,
                speed_fact=speed_fact,
                drift_to_random_strength_ratio=drift_to_random_strength_ratio,
            )

            # set a teleported position that will be used for calculating stats
            self.pos = out_coords - update

            if not self.Environment.check_if_position_is_in_environment(out_coords):
                raise RuntimeError("Teleport out position is not in the environment.")

            # record teleportation data
            teleportation_data = {
                "teleport_pair_num": teleport_pair_num,
                "step_num": self.num_steps_total,
            }

            for direction in ["in", "out"]:
                type_num = self.Environment.teleport_pairs_dict[teleport_pair_num][
                    direction
                ][0]
                object_idx = self.Environment.object_df.loc[
                    self.Environment.object_df["object_type_num"] == type_num
                ].index
                if len(object_idx) == 0:
                    raise RuntimeError(
                        f"Could not find object index for object type number {type_num}."
                    )
                elif len(object_idx) > 1:
                    raise RuntimeError(
                        f"Found multiple object indices for object type number {type_num}."
                    )
                teleportation_data[f"{direction}_object_idx"] = object_idx[0]
                teleportation_data[f"{direction}_object_type_num"] = type_num
                teleportation_data[f"{direction}_object_type_name"] = (
                    self.Environment.object_type_num_to_name_dict[type_num]
                )

                coords = in_coords if direction == "in" else out_coords
                velocity = in_velocity if direction == "in" else out_velocity
                teleport_coords = self.Environment.teleport_pairs_dict[
                    teleport_pair_num
                ][direction][1]
                vector = coords - teleport_coords

                teleportation_data[f"{direction}_position_x"] = coords[0]
                teleportation_data[f"{direction}_position_y"] = coords[1]
                teleportation_data[f"{direction}_vector_x"] = vector[0]
                teleportation_data[f"{direction}_vector_y"] = vector[1]
                teleportation_data[f"{direction}_velocity_x"] = velocity[0]
                teleportation_data[f"{direction}_velocity_y"] = velocity[1]

            self.teleportation_df.loc[len(self.teleportation_df)] = teleportation_data  # type: ignore[assignment]

            if self.always_log_teleportation:  # type: ignore[attr-defined]
                self.log_teleportation(last=True)

            break

        return out_coords

    def fix_record_after_manual_update(self, dt: float | None = None):
        """Fix values computed and recorded based on the velocity, if applicable.
        Applies to teleportation events."""

        if not hasattr(self, "recorded_stats"):
            return

        if dt is None:
            dt = self.dt

        prev_pos_recorded = np.asarray(self.history["pos"][-2])
        prev_pos_used = np.asarray(self.recorded_stats["position"])
        if not (prev_pos_recorded == prev_pos_used).any():
            self.velocity = (self.pos - prev_pos_used) / dt

        super().fix_record_after_manual_update(dt=dt)

    def reset(self):
        """Reset the agent to a random location."""

        self.end_trajectory()

        self.set_all_positions(first_setting=False)

        if self.trajectory_lengths is not None:
            i = (len(self.trajectory_df) - 1) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]

        self.current_trajectory_length = 0

        self.set_new_trajectory()

        return

    def update(  # type: ignore[override]
        self,
        dt: float | None = None,
        speed_fact: int | float = 3,
        drift_to_random_strength_ratio: float = 0.7,
        drift_velocity: (
            float | np.ndarray[tuple[int], np.dtype[np.float64]] | None
        ) = None,
        **kwargs,
    ):
        """Update the agent, optionally with a new position and velocity.

        See Agent.update() in ratinabox/agent.py for kwargs.

        Args:
        - dt (float): The time step to use.
        - speed_fact (float): The speed factor.
        - drift_to_random_strength_ratio (float): The ratio of the drift strength to
            the random walk strength.

        Keyword args:
        - **kwargs: Keyword arguments for Agent.update().
        """

        target_reached = self.check_and_record_target_reached()

        if self.check_if_trajectory_end_reached():
            self.reset()
        elif target_reached:
            self.set_random_walk()
        elif self.current_num_of_random_walk_steps == 0:
            if self.target_position is None:
                self.set_new_target()
            self.set_random_walk()

        teleport_coords = self.teleport_coords_if_applicable(
            speed_fact=speed_fact,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
        )

        # calculate drift_velocity
        if teleport_coords is None and drift_velocity is None:
            drift_velocity = self.get_drift_velocity(
                pos=self.pos, speed_fact=speed_fact
            )

        if self.current_num_of_random_walk_steps > 0:
            self.current_num_of_random_walk_steps -= 1
            if teleport_coords is not None:
                self.current_num_of_random_walk_steps = 0  # end random walk
            if self.current_num_of_random_walk_steps == 0:
                df_idx = len(self.target_df) - 1
                column = "random_walk_periods"
                self.target_df.loc[df_idx, column][-1].append(self.num_steps_total + 1)  # type: ignore[has-method]

        super().update(
            dt=dt,
            skip_checks=True,
            new_pos=teleport_coords,
            drift_velocity=drift_velocity,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
            **kwargs,
        )

    def get_agent_state_color(self, t: float | None = None) -> str:
        """Get the agent state.

        Args:
        - t (float, optional): The time to get the state color for. Default is None.

        Returns:
        - agent_color (str): The agent state color.
        """

        # get agent color based on whether the agent is aiming for a target
        endid = plot_util.get_plotting_times(np.array(self.history["t"]), t_end=t)[1]
        agent_color = "lavenderblush"
        past_df = self.target_df.loc[self.target_df["set_step"] <= endid]
        if len(past_df):
            row = past_df.loc[past_df.index[-1]]
            if row["object_type_name"] == "no_target":
                agent_color = "lavenderblush"
            elif np.isnan(row["reached_step"]) or endid < row["reached_step"]:
                agent_color = "red"
                # check if this was during random walk
                random_walk_periods = row["random_walk_periods"]
                if len(random_walk_periods) > 0:
                    for random_walk_period in random_walk_periods:
                        if len(random_walk_period) == 1:
                            if random_walk_period[0] <= endid:
                                agent_color = "violet"
                                break
                        elif random_walk_period[0] <= endid < random_walk_period[1]:
                            agent_color = "violet"
                            break

        return agent_color

    def plot_trajectory(  # type: ignore[override]
        self,
        t_end: float | None = None,
        target_alpha: float = 0.7,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the trajectory.

        Args:
        - target_alpha (float, optional): Alpha value of the targets.
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.
        - **kwargs: Additional keyword arguments.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectory plotted.
        """

        sub_ax = super().plot_trajectory(
            t_end=t_end,
            target_alpha=target_alpha,
            plot_target=False,
            agent_color=self.get_agent_state_color(t=t_end),
            autosave=False,
            **kwargs,
        )

        legend = sub_ax.get_legend()
        if no_legend and legend is not None:
            legend.remove()

        self.add_target_to_plot(sub_ax, t=t_end)

        fig = sub_ax.figure
        util.save_figure(fig, "trajectory", save=autosave)

        return sub_ax

    def plot_trajectory_targets(
        self,
        sub_ax: plt.Axes | None = None,
        alpha: float = 1.0,
        plot_env: bool = True,
        no_legend: bool = False,
        colormap: str | None | mpl_colors.Colormap = None,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """Plot the trajectory targets.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None,
            a new subplot is created using self.Environment.plot_environment().
            This can be used to plot trajectory on top of receptive fields etc.
        - alpha (float, optional): Alpha value of the targets.
        - plot_env (bool, optional): Whether to plot the environment.
            Default is True.
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - colormap (str, optional): Colormap to use. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectory targets plotted.
        """

        if sub_ax is None or plot_env:
            sub_ax = self.Environment.plot_environment(sub_ax=sub_ax)

        if sub_ax is None:
            raise RuntimeError("sub_ax is None.")

        if len(self.target_df) == 0:
            return sub_ax

        sub_target_df = self.target_df[
            self.target_df["object_type_name"] != "no_target"
        ]

        env_width = self.Environment.extent[1] - self.Environment.extent[0]
        env_height = self.Environment.extent[3] - self.Environment.extent[2]

        for object_idx in np.sort(sub_target_df["object_idx"].unique()):
            object_df = sub_target_df[sub_target_df["object_idx"] == object_idx]
            count = len(object_df)
            # write the number of times the target was visited
            for shift, color in [(1, "white"), (0, "black")]:
                shift_x = shift * 0.006 * env_width
                shift_y = shift * 0.006 * env_height
                sub_ax.text(
                    object_df.loc[object_df.index[0], "position_x"] + shift_x,
                    object_df.loc[object_df.index[0], "position_y"] + shift_y,
                    str(count),
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    color=color,
                    fontsize=10,
                    zorder=10,
                    fontweight="bold",
                )

        reached_df = sub_target_df[~sub_target_df["reached_step"].isna()]

        min_val, max_val = 0, 1
        if len(reached_df) != 0:
            reached_steps = reached_df["reached_step"].to_numpy()
            reached_steps = np.insert(reached_steps, 0, 0)

            # get linewidth calculation values
            min_val = np.min(np.diff(reached_steps))
            max_val = np.max(np.diff(reached_steps))

        if colormap is None:
            colormap = "crest"
        cmap = sns.color_palette(colormap, as_cmap=True)

        start_pos = self.history["pos"][0]
        start_step = 0
        for i, idx in enumerate(sub_target_df.index):
            target_row = sub_target_df.loc[idx]
            color = cmap(i / len(sub_target_df))  # type: ignore[callable]

            reached_step = target_row["reached_step"]
            if np.isfinite(reached_step):
                num_steps = reached_step - start_step
                lw = 1 + (num_steps - min_val) / (max_val - min_val)
                ls = None
            else:
                ls = "dashed"
                lw = None

            sub_ax.plot(
                [start_pos[0], target_row["position_x"]],
                [start_pos[1], target_row["position_y"]],
                color=color,
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                zorder=1,
            )

            start_pos = target_row[["position_x", "position_y"]].to_list()
            if np.isfinite(reached_step):
                start_step = reached_step

        legend = sub_ax.get_legend()
        if no_legend and legend is not None:
            legend.remove()

        fig = sub_ax.figure
        util.save_figure(fig, "trajectory_targets", save=autosave)

        return sub_ax

    def plot_trajectory_targets_over_time(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        no_legend: bool = False,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """Plot the trajectory targets over time.

        Args:
        - t_start (float, optional): Start time of the plot.
        - t_end (float, optional): End time of the plot.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created.
            This can be used to plot trajectory on top of receptive fields etc.
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectory targets plotted over time.
        """

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(8, 3))

        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        startid, endid = plot_util.get_plotting_times(t, t_start=t_start, t_end=t_end)
        t_start, t_end = t[startid], t[endid]

        if t_start is None or t_end is None:
            raise RuntimeError("t_start or t_end is None.")

        t = t[startid : endid + 1]  # keep in seconds
        pos = pos[startid : endid + 1]

        # plot reset points as vertical dashed lines
        reset_times = self.get_reset_times()
        for reset_time in reset_times:
            if reset_time >= t_start and reset_time <= t_end:
                sub_ax.axvline(
                    reset_time, color="black", ls="dashed", alpha=0.2, zorder=-1
                )

        # plot trajectory
        sub_ax.plot(t, pos[:, 0], color="lightgray", label="X")
        sub_ax.plot(t, pos[:, 1], color="darkgray", label="Y")

        sub_ax.set_title("Position over time")
        sub_ax.set_ylabel("Position")
        sub_ax.set_xlabel("Time (s)")

        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["top"].set_visible(False)
        sub_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        # plot teleportation points as vertical dashed lines
        for idx in self.teleportation_df.index:
            step = int(self.teleportation_df.loc[idx, "step_num"])  # type: ignore[assignment]
            if step < startid or step > endid:
                continue

            object_type = self.teleportation_df.loc[idx, "in_object_type_num"]
            color = self.Environment.type_num_to_plot_params_dict[object_type]["color"]
            sub_ax.axvline(t[step], color=color, ls="dashed", alpha=0.8, zorder=-1)

        # plot target objects
        type_num_to_plot_params_dict = copy.deepcopy(
            self.Environment.type_num_to_plot_params_dict
        )

        r = 0
        reset_times = np.append(reset_times, t[-1])
        for idx in self.target_df.index:
            row = self.target_df.loc[idx]
            if row["object_type_name"] == "no_target":
                continue

            plot_params = type_num_to_plot_params_dict[row["object_type_num"]]
            label = None
            if "name" in plot_params:
                label = plot_params.pop("name")
                plot_params["markersize"] = plot_params.pop("s") / 8

            if np.isnan(row["reached_step"]):
                target_reached_time = reset_times[r]
                alpha = 0.3
                r += 1
            else:
                target_reached_time = t[int(row["reached_step"])]
                alpha = 0.8

            if target_reached_time < t_start or target_reached_time > t_end:
                continue

            sub_ax.plot(
                [target_reached_time] * 2,
                [row["position_x"], row["position_y"]],
                **plot_params,
                label=label,
                lw=1.5,
                alpha=alpha,
            )

        sub_ax.legend(
            loc="center left", fontsize="small", bbox_to_anchor=(1, 0.5), frameon=False
        )

        legend = sub_ax.get_legend()
        if no_legend and legend is not None:
            legend.remove()

        # expand x limits a bit
        pad = (t_end - t_start) * 0.02
        sub_ax.set_xlim(t_start - pad, t_end + pad)
        plot_util.pad_axis(sub_ax, axis="y", pad_prop=0.04)

        fig = sub_ax.figure
        util.save_figure(fig, "trajectory_targets_over_time", save=autosave)

        return sub_ax

    def add_target_to_plot(
        self,
        sub_ax: plt.Axes,
        t: float | None = None,
    ):
        """
        Add the target for time t to the plot.

        Args:
        - sub_ax (plt.Axes): Subplot to plot on.
        - t (float, optional): The current time step. Default to None.
        """

        all_t = np.array(self.history["t"])
        if t is None:
            t = float(all_t[-1])
        endid = np.argmin(np.abs(all_t - (t))) + 1

        # get target
        past_df = self.target_df.loc[self.target_df["set_step"] <= endid]
        if len(past_df) == 0:  # no current
            return

        idx = past_df.index[-1]
        last_reached = past_df.loc[idx, "reached_step"]  # type: ignore[assignment]
        if not np.isnan(last_reached) and last_reached < endid:
            return

        sub_ax.scatter(
            past_df.loc[idx, "position_x"],  # type: ignore[assignment]
            past_df.loc[idx, "position_y"],  # type: ignore[assignment]
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
        - additional_plot_func: A function that is called after each frame of the
            animation is plotted. It takes sub_ax, t and **kwargs and returns sub_ax.
        - **kwargs: Additional keyword arguments passed to self.plot_trajectory().

        Returns:
            anim (matplotlib.animation.FuncAnimation): The animation object.
        """

        def run_all_additional_plot_funcs(
            sub_ax: plt.Axes,
            t: float | None = None,
            **kwargs,
        ) -> plt.Axes:
            """
            Run all additional plot functions.

            Args:
            - sub_ax (plt.Axes): Subplot to plot on.
            - t (float, optional): The current time step. Default to None.

            Returns:
            - sub_ax (plt.Axes): Subplot with updated trajectory plotted.
            """

            if additional_plot_func is not None:
                sub_ax = additional_plot_func(sub_ax=sub_ax, t=t, **kwargs)

            # self.add_target_to_plot(sub_ax, t=t)
            plot_util.remove_prev_handle_labels(sub_ax)

            return sub_ax

        anim = super().animate_trajectory(
            additional_plot_func=run_all_additional_plot_funcs, **kwargs
        )

        return anim
