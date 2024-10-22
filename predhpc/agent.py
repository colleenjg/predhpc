from typing import Any, TYPE_CHECKING, Callable, Sequence
import warnings

import copy
from matplotlib import pyplot as plt  # type: ignore[import]
from matplotlib import animation as mpl_animation
from matplotlib import markers as mpl_markers
from matplotlib import colors as mpl_colors
from matplotlib import figure as mpl_figure
import numpy as np
import pandas as pd  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from ratinabox import Agent as riabAgent  # type: ignore[import]
from ratinabox import utils as rutils

from predhpc import env, plot_fcts
from predhpc.util import gen_util, plot_util, ext_util


class ResetableAgent(riabAgent, ext_util.ParamsManagerMixin):
    """
    ResetableAgent()

    Class extending the agent so that is has trajectory lengths after which it resets
    to a random location.

    Must be initialised with an environment. A parameters dictionary can also be passed.

    default_params = {
        "dt": 0.03,  # time step, in seconds
        "trajectory_length": None,  # int or iterable of ints
        "num_trajectories": None,  # number of trajectory lengths to sample
        "exp_factors": None,  # exponential factors for trajectory_length (inv. scale, rate, minimum). Default is None.
        "random_max": None,  # max value for randomizing trajectory_length
        "start_position": None,  # position to start trajectories from
        "reset_position": None,  # position to reset trajectories from
        "target_position": None,  # position to use as target
        "wait_between_targets": 10,  # number of steps to wait between target reaching
        "reset_reached_within_tol_prop_to_speed_dt": 0.55,  # proportion of current speed to use as reset tolerance
        "target_reached_within_tol_prop_to_speed_dt": 0.55,  # proportion of current speed to use as target tolerance
        "fixed_direction": False,  # keep same direction (1D environment only)
    }

    List of properties (in addition to ratinabox.Agent properties):
        • self.target_df_columns
        • self.trajectory_df_columns

    List of methods (in addition to ratinabox.Agent methods):
        • self.format_position()
        • self.set_target_position
        • self.set_all_positions
        • self.reverse()
        • self.set_position_and_velocity()
        • self.sample_position_within_tolerance()
        • self.get_trajectory_lengths_to_date()
        • self.log_trajectories_to_date()
        • self.log_trajectory_stats_to_date()
        • self.check_if_position_reached()
        • self.check_if_reset_position_reached()
        • self.check_if_target_position_reached()
        • self.check_if_trajectory_end_reached()
        • self.check_and_record_target_reached()
        • self.get_reset_times()
        • self.get_reached_position_steps()
        • self.reset()
        • self.update()
        • self.plot_trajectories_to_date()
        • self.add_position_across_time_to_plot()
        • self.plot_trajectories_across_time()
        • self.plot_trajectories()
        • self.plot_trajectory_edges()
    """

    default_params = {
        "dt": 0.03,  # time step, in seconds
        "head_direction_smoothing_timescale": 0.2,  # higher than dt
        "trajectory_length": None,  # int or iterable of ints
        "num_trajectories": None,  # number of trajectory lengths to sample
        "exp_factors": None,  # exponential factors for trajectory_length (inv. scale, rate, minimum). Default is None.
        "random_max": None,  # max value for randomizing trajectory_length
        "start_position": None,  # position to start trajectories from
        "reset_position": None,  # position to reset trajectories from
        "target_position": None,  # position to use as target
        "wait_between_targets": 10,  # number of steps to wait between target reaching
        "reset_reached_within_tol_prop_to_speed_dt": 0.55,  # proportion of current speed * dt to use as reset tolerance
        "target_reached_within_tol_prop_to_speed_dt": 0.55,  # proportion of current speed * dt to use as target tolerance
        "fixed_direction": False,  # keep same direction (1D environment only)
    }

    ignored_param_keys = list()  # type: list[str]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Env: "env.Environment", params: dict[str, Any] = dict()):
        """
        ResetableAgent(Env)

        Initialise the agent.

        Attributes:
        - _last_stop_step (int): Last step at which the agent stopped.
        - _most_recent_target_reached_step (int): Last step at which the target was
            reached.

        Args:
        - Env (env.Environment): The environment in which the agent is placed.
        - params (dict, optional): Agent parameters. Default is dict().

        Raises:
        - ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.check_if_ignored_params(params)
        params = self.add_fixed_params(params)

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

        self.set_all_positions()
        self._init_trajectory_lengths()
        self._init_trajectory_df()
        self._init_target_df()

        if self.Environment.dimensionality == "1D" and self.fixed_direction:
            if np.sign(self.reset_position - self.start_position) != np.sign(self.speed_mean):  # type: ignore[attr-defined]
                raise ValueError(
                    "If direction is fixed, speed must have the same sign."
                )

        self._most_recent_target_reached_step = -1
        self._last_stop_step = -1

    @property
    def target_df_columns(self) -> list:
        """
        self.target_df_columns

        Obtain the target dataframe columns.

        Attributes:
        - _target_df_columns (list): List of column names.

        Returns:
        - (list) List of column names.
        """

        if not hasattr(self, "_target_df_columns"):
            self._target_df_columns = [
                "position_x",
                "set_step",
                "set_time",
                "reached_step",
                "reached_time",
                "num_steps_total",
                "time_total",
            ]

            if self.Environment.D == 2:
                self._target_df_columns.append("position_x")

        return self._target_df_columns

    @property
    def trajectory_df_columns(self) -> list:
        """
        self.trajectory_df_columns

        Obtain the trajectory dataframe columns.

        Returns:
        - (list): List of trajectory dataframe column names.
        """

        if not hasattr(self, "_trajectory_df_columns"):
            trajectory_df_columns = ["start_position_x"]

            if self.Environment.D == 2:
                trajectory_df_columns.append("start_position_y")

            trajectory_df_columns.extend(
                ["start_step", "start_time", "stop_position_x"]
            )

            if self.Environment.D == 2:
                trajectory_df_columns.append("stop_position_y")

            trajectory_df_columns.extend(
                ["stop_step", "stop_time", "num_steps_total", "time_total"]
            )

            self._trajectory_df_columns = trajectory_df_columns

        return self._trajectory_df_columns

    def _init_trajectory_df(self):
        """
        self._init_trajectory_df()

        Initialise the trajectory dataframe, which records the trajectory start and stop
        position, and the number of steps it lasted.

        Attributes:
        - trajectory_df (pd.DataFrame): Dataframe for tracking trajectories.
        """

        self.trajectory_df = pd.DataFrame(columns=self.trajectory_df_columns)
        self._add_new_trajectory_to_df()

    def _init_trajectory_lengths(self):
        """
        self._init_trajectory_lengths()

        Initialise the trajectory lengths, either from the passed value, or by sampling
        from the exponential distribution.

        Attributes:
        - current_trajectory_length (int): Current trajectory length.
        - num_steps_total (int): Total number of steps taken.
        - trajectory_length (int): Target length of the current trajectory.
        - trajectory_lengths (1D np.ndarray): All trajectory lengths.

        if self.trajectory length is not None:
        - exp (np.random.exponential): Exponential distribution parameters, set to None.
        - num_trajectories (int): Number of trajectories, set to None.
        - rand (np.random.uniform): Uniform distribution parameters, set to None.
        """

        if self.trajectory_length is not None:
            self.num_trajectories = None
            self.exp = None
            self.rand = None

        elif self.num_trajectories:
            self.trajectory_length = ext_util.get_trajectory_lengths(
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

    def _add_new_trajectory_to_df(self):
        """
        self._add_new_trajectory_to_df()

        Add start information for a new trajectory to trajectory dataframe.
        """

        trajectory_data = {
            "start_position_x": self.pos[0],
            "start_step": self.num_steps_total,
            "start_time": self.t,
        }

        if self.Environment.D == 2:
            trajectory_data["start_position_y"] = self.pos[1]

        self.trajectory_df.loc[len(self.trajectory_df)] = trajectory_data  # type: ignore[assignment]

    def _end_trajectory(self):
        """
        self._end_trajectory()

        Add stop information for a trajectory that is ending to trajectory dataframe.
        """

        idx = len(self.trajectory_df) - 1
        start_step = int(self.trajectory_df.loc[idx, "start_step"])  # type: ignore[assignment]
        start_time = self.trajectory_df.loc[idx, "start_time"]  # type: ignore[assignment]

        self.trajectory_df.loc[idx, "stop_position_x"] = self.pos[0]
        self.trajectory_df.loc[idx, "stop_step"] = self.num_steps_total
        self.trajectory_df.loc[idx, "stop_time"] = self.t
        self.trajectory_df.loc[idx, "num_steps_total"] = (
            self.num_steps_total - start_step
        )
        self.trajectory_df.loc[idx, "time_total"] = self.t - start_time

        if self.Environment.D == 2:
            self.trajectory_df["stop_position_y"] = self.pos[1]

    def _init_target_df(self):
        """
        self._init_target_df()

        Initialise the target dataframe, which records the target position and the
        step at which it was reached. Also adds a first row.

        Attributes:
        - target_df (pd.DataFrame): Dataframe for tracking targets.
        """

        self.target_df = pd.DataFrame(columns=self.target_df_columns)
        self._add_new_target_to_df()

    def _end_target_df_line(self):
        """
        self._end_target_df_line()

        End the current target dataframe line by adding the total number of steps and
        time taken (whether target was reached or not).
        """

        if len(self.target_df) == 0:
            return

        df_idx = len(self.target_df) - 1

        num_steps = self.num_steps_total - self.target_df.loc[df_idx, "set_step"]  # type: ignore[operator]
        self.target_df.loc[df_idx, "num_steps_total"] = num_steps
        time_total = self.t - self.target_df.loc[df_idx, "set_time"]  # type: ignore[operator]
        self.target_df.loc[df_idx, "time_total"] = time_total

    def _add_new_target_to_df(self):
        """
        self._add_new_target_to_df()

        Add a new line to the target dataframe.
        """

        if self.target_position is None:
            return

        self._end_target_df_line()

        target_data = {
            "position_x": self.target_position[0],
            "set_step": self.num_steps_total,
            "set_time": self.t,
        }

        if self.Environment.D == 2:
            target_data["position_y"] = self.target_position[1]

        self.target_df.loc[len(self.target_df)] = target_data  # type: ignore[assignment]

    def _check_and_adjust_current_velocity_for_1D(self, dt: float | None = None):
        """
        self._check_and_adjust_current_velocity_for_1D(prev_velocity)

        Check if velocity is opposite to the current fixed direction in a 1D
        environment. If so, since the current recorded velocity is used to determine
        the next position update, adjust it to better match the fixed direction.

        Attributes:
        - velocity (1D np.ndarray): Adjusted velocity.

        Args:
        - dt (float, optional): Time step. If None, agent time step is used.
            Default is None.
        """

        # check if velocity needs to be corrected
        if not (self.Environment.dimensionality == "1D" and self.fixed_direction):
            return

        # check if velocity matches target direction
        trajectory_sign = np.sign(self.reset_position - self.start_position)  # type: ignore[has-type]
        if np.sign(self.velocity) == trajectory_sign:
            return

        # check if velocity matches current direction
        current_sign = np.sign(self.reset_position - self.pos)  # type: ignore[has-type]
        if np.sign(self.velocity) == current_sign:
            return

        if dt is None:
            dt = self.dt  # type: ignore[has-type]

        # resample velocity until it has the correct sign
        new_velocity = self.velocity
        for _ in range(10):
            if np.sign(new_velocity) != trajectory_sign:  # type: ignore[has-type]
                new_velocity = self.velocity + rutils.ornstein_uhlenbeck(
                    dt=dt,
                    x=self.velocity,
                    drift=self.speed_mean,
                    noise_scale=self.speed_std,  # type: ignore[attr-defined]
                    coherence_time=self.speed_coherence_time,  # type: ignore[attr-defined]
                )
            else:
                break

        # if resampling failed, set to 0
        if np.sign(new_velocity) != trajectory_sign:  # type: ignore[has-type]
            new_velocity = self.velocity * 0  # set to 0

        self.velocity = new_velocity

    def get_plotting_times(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        raise_error: bool = True,
    ):
        """
        self.get_plotting_times()

        Obtain the times to plot.

        Args:
        - t_start (float, optional): Start time. Default is None.
        - t_end (float, optional): End time. Default is None.
        - raise_error (bool, optional): Whether to raise an error if the start and end
            times are not in the history. Default is True.

        Returns:
        - t (1D np.ndarray): Times to plot.
        - startid (int): Index of the start time point.
        - endid (int): Index of the end time point.
        """

        t = np.asarray(self.history["t"])
        startid, endid = plot_util.get_plotting_times(
            t, t_start=t_start, t_end=t_end, raise_error=raise_error
        )
        t = t[startid : endid + 1]

        return t, startid, endid

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
                    f"Got {self.Environment.D}."
                )

        return position

    def set_target_position(self, position):
        """
        self.set_target_position(position)

        Set a target position, checking that it is within the environment extent.
        Also move object in environment to the new target position.
        Reset the number of steps before checking for target to 0.

        Attributes:
        - _target_object_idx (int): Index of target object in environment objects.
        - steps_before_checking_for_target (int): Number of steps to wait before
            checking for target.
        - target_position (1D np.ndarray or None): Target position.

        Args:
        - position (1D np.ndarray or list or None): Target position.
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

    def set_all_positions(self):
        """
        self.set_all_positions()

        Set start, reset and target positions, checking that they are within the
        environment extent.

        If start_position is not None, set the agent's position and velocity.

        Attributes:
        - manual_pos (bool): Whether position was set manually.
        - start_position (1D np.ndarray): Start position.
        - reset_position (1D np.ndarray): Reset position.
        """

        self.start_position = self.format_position(self.start_position)
        self.reset_position = self.format_position(self.reset_position)
        self.set_target_position(self.target_position)

        self.manual_pos = False
        if self.start_position is not None:
            self.set_position_and_velocity(position=self.start_position, velocity=0)

    def reverse(self, reset=False):
        """
        self.reverse()

        Reverse the agent's start and reset positions. Also, optionally reset the
        current trajectory.

        Attributes:
        - reset_position (1D np.ndarray): Reset position.
        - speed_mean (float): Mean speed.
        - start_position (1D np.ndarray): Start position.

        Args:
        - reset (bool, optional): Whether to reset the agent. Default is False.
        """

        new_reset_pos, new_start_pos = self.start_position, self.reset_position
        self.start_position = new_start_pos
        self.reset_position = new_reset_pos

        if self.Environment.D == 1:
            self.speed_mean = -self.speed_mean

        if reset:
            self.reset()

    def set_position_and_velocity(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        velocity: float | np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        rotational_velocity: float | None = 0.0,
        sample: bool = True,
    ):
        """
        self.set_position_and_velocity()

        Set the position and velocity of the agent, and record that the velocity
        statistics should be corrected.

        Adapted from Agent.__init__() in ratinabox/agent.py

        Attributes:
        - manual_pos (bool): Whether position was set manually.
        - pos (1D np.ndarray): Position.
        - rotational_velocity (float): Rotational velocity.
        - velocity (1D np.ndarray): Velocity.

        Args:
        - position (1D np.ndarray, optional): Position to set. If None, a random
            position is sampled. Default is None.
        - velocity (float or 1D np.ndarray, optional): Velocity to set. If None, a
            random velocity is sampled. Default is None.
        - rotational_velocity (float, optional): Rotational velocity to set.
            Default is 0.0.
        - sample (bool, optional): Whether to sample within the tolerance of the given
            position. Ignored if position is None. Default is True.
        """

        # initialise starting positions and velocity
        if position is None:
            self.pos = self.Environment.sample_positions(n=1, method="random")[0]
        elif sample:
            self.pos = self.sample_position_within_tolerance(position)
        else:
            self.pos = position

        if self.Environment.dimensionality == "1D":
            if velocity is None:
                self.velocity = np.asarray([self.speed_mean]).reshape(1)
            else:
                self.velocity = np.asarray([velocity]).reshape(1)
            if self.Environment.boundary_conditions == "solid":
                if self.speed_mean != 0:
                    warnings.warn(
                        "solid 1D boundary conditions and non-zero speed mean."
                    )

        elif self.Environment.dimensionality == "2D":
            if velocity is None or len(np.asarray(velocity).ravel()) == 1:
                direction = np.random.uniform(0, 2 * np.pi)
                velocity = self.speed_std * np.asarray(  # type: ignore[attr-defined]
                    [np.cos(direction), np.sin(direction)]
                )
            self.velocity = np.asarray(velocity).reshape(2)
            self.rotational_velocity = rotational_velocity

        self.manual_pos = True

    def sample_position_within_tolerance(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]],
        sample_within_tol_prop_to_speed_dt: float = 0.5,
        max_attempts: int = 100,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        self.sample_position_within_tolerance(position)

        Sample a position within the tolerance of the given position.

        Args:
        - position (1D np.ndarray): The position to sample around.
        - sample_within_tol_prop_to_speed_dt (float): The proportion of the
            tolerance to sample within. Default is None, in which case the agent's
            target_reached_within_tol_prop_to_speed_dt is used.

        Returns:
        - new_position (1D np.ndarray): The sampled position.
        """

        tolerance = np.absolute(self.speed_mean) * self.dt * sample_within_tol_prop_to_speed_dt  # type: ignore[has-type]

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
                f"{sample_within_tol_prop_to_speed_dt} of {position}."
            )

        return new_position

    def get_trajectory_lengths_to_date(self):
        """
        self.get_trajectory_lengths_to_date()

        Obtain the lengths of all completed trajectories to date.

        Returns:
        - trajectory_lengths_to_date (list): Lengths of all completed trajectories
            to date.
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
        """
        self.log_trajectories_to_date()

        Log the trajectory lengths to date.
        """

        trajectory_lengths_to_date = self.get_trajectory_lengths_to_date()
        print(
            f"Trajectory lengths ({len(trajectory_lengths_to_date)}) to date "
            f"(in steps): {trajectory_lengths_to_date}"
        )

    def log_trajectory_stats_to_date(self, log_as_time: bool = True):
        """
        self.log_trajectory_stats_to_date()

        Log the trajectory length statistics to date.

        Args:
        - log_as_time (bool, optional): Whether to log trajectory lengths in time
            (sec/min). Otherwise, they are logged in steps. Default is True.
        """

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

    def check_if_position_reached(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        sample_within_tol_prop_to_speed_dt: float = 0.55,
    ) -> bool:
        """
        self.check_if_position_reached()

        Check if the agent has reached a specific position, within a specific
        tolerance. The tolerance is specified in proportion to speed and step size.

        Args:
        - position (1D np.array): Position to check if reached.
        - sample_within_tol_prop_to_speed_dt (float):
            Tolerance proportion, wrt mean speed * dt.

        Returns:
        - position_reached (bool): Whether the agent has reached the target position,
            within the specified tolerance.
        """

        position_reached = False

        if position is not None:
            # calculate the distance between the current position and the reset position
            dist = np.linalg.norm(self.pos - position, ord=2)

            # check if the distance is less than the tolerance
            speed = np.linalg.norm(self.velocity, ord=2)
            reached_dist = speed * self.dt * sample_within_tol_prop_to_speed_dt
            if dist < reached_dist:  # type: ignore[has-type]
                position_reached = True

        return position_reached

    def check_if_reset_position_reached(self) -> bool:
        """
        self.check_if_reset_position_reached()

        Check if the agent has reached the reset position.

        Returns:
        - reset_reached (bool): Whether the agent has reached the reset position,
            within the specified tolerance.
        """

        reset_reached = self.check_if_position_reached(
            self.reset_position, self.reset_reached_within_tol_prop_to_speed_dt  # type: ignore[attr-defined]
        )

        return reset_reached

    def check_if_target_position_reached(self) -> bool:
        """
        self.check_if_target_position_reached()

        Check if the agent has reached the target position, unless the agent is set to
        wait before checking for the target. If waiting, the number of steps to wait
        is decremented.

        Attributes:
        - steps_before_checking_for_target (int): Number of steps to wait before
            checking for target.

        Returns:
        - target_reached (bool): Whether the agent has reached the target position,
            within the specified tolerance.
        """

        if self.target_position is None:
            target_reached = False

        elif self.steps_before_checking_for_target > 0:
            self.steps_before_checking_for_target -= 1
            target_reached = False

        else:
            target_reached = self.check_if_position_reached(
                self.target_position, self.target_reached_within_tol_prop_to_speed_dt  # type: ignore[attr-defined]
            )
            if target_reached:
                self.steps_before_checking_for_target = self.wait_between_targets  # type: ignore[attr-defined]

        return target_reached

    def check_if_trajectory_end_reached(self) -> bool:
        """
        self.check_if_trajectory_end_reached()

        Check if the agent has reached the end of its current trajectory.

        Attributes:
        - reached_end (bool): Whether the agent has reached the end of its current
            trajectory.

        Returns:
        - (bool): Whether the agent has reached the end of its current trajectory.
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
        """
        self.check_and_record_target_reached()

        Check if the agent has reached the target, and if so, record in the target
        dataframe.

        Attributes:
        - _most_recent_target_reached_step (int): Most recent step at which a target
            was reached.
        - reached_target (bool): Whether the agent has reached the target.

        Returns:
        - (bool): Whether the agent has reached the target.
        """

        self.reached_target = False
        if self.target_position is not None and self.check_if_target_position_reached():
            if self.num_steps_total == self._most_recent_target_reached_step:
                self.reached_target = False
            else:
                self.reached_target = True

        if self.reached_target:
            df_idx = len(self.target_df) - 1
            self._most_recent_target_reached_step = self.num_steps_total
            self.target_df.loc[df_idx, "reached_step"] = self.num_steps_total
            self.target_df.loc[df_idx, "reached_time"] = self.t

            self._add_new_target_to_df()

        return self.reached_target

    def get_reset_times(self):
        """
        self.get_reset_times()

        Obtain the times at which positions were reset, marking the end of a trajectory.

        Returns:
        - reset_times (1D np.ndarray): Position reset times.
        """

        reset_steps = self.trajectory_df["stop_step"].to_numpy()
        if np.isnan(reset_steps[-1]):
            reset_steps = reset_steps[:-1]

        reset_times = reset_steps * self.dt

        return reset_times

    def get_reached_position_steps(self, position_name: str = "reset") -> np.ndarray:
        """
        self.get_reached_position_steps()

        Obtain the steps at which the agent reached the specified position.

        Returns:
        - reached_position_steps (1D np.ndarray): Steps at which the agent
            reached the specified position.
        """

        if position_name == "start":
            reached_position_steps = self.trajectory_df["stop_step"].to_numpy() - 1
        elif position_name == "reset":
            reached_position_steps = self.trajectory_df["stop_step"].to_numpy()
        elif position_name == "target":
            reached_position_steps = self.target_df["reached_step"].to_numpy()
        else:
            raise NotImplementedError(
                f"Position name must be 'start', 'reset', or 'target', but got "
                f"{position_name}."
            )

        if np.isnan(reached_position_steps[-1]):
            reached_position_steps = reached_position_steps[:-1]
        reached_position_steps = reached_position_steps.astype(int)

        return reached_position_steps

    def reset(self):
        """
        self.reset()

        Reset the agent. Specifically, end the current trajectory, set a new position
        and velocity, identify the new trajectory length, and reinitialise the
        new trajectory's step count. Also adds new trajectory to the trajectory
        dataframe.

        Attributes:
        - current_trajectory_length (int): Current trajectory length.
        - _end_trajectory (method): End the current trajectory.
        """

        self._end_trajectory()

        self.set_position_and_velocity(position=self.start_position, velocity=0)

        if self.trajectory_lengths is not None:
            i = (len(self.trajectory_df) - 1) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]

        self.current_trajectory_length = 0

        self._add_new_trajectory_to_df()

    def update(
        self,
        dt: float | None = None,
        skip_checks: bool = False,
        new_pos: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        **kwargs,
    ):
        """
        self.update()

        Update the agent's position and velocity statistics, time, current trajectory
        length, and history. If applicable, first checks whether target or trajectory
        end are reached, and if so, resets the agent.

        Attributes:
        - current_trajectory_length (int): Current trajectory length to date.
        - manual_pos (bool): Whether position was set manually.
        - num_steps_total (int): Total number of steps taken to date.
        - pos (1D np.ndarray): Position.
        - t (float): Current time.

        Args:
        - dt (float, optional): Time step. If None, agent time step is used.
            Default is None.
        - skip_checks (bool, optional): Whether to skip checks. Default is False.
        - new_pos (1D np.ndarray, optional): New position to use for update,
            if applicable. Default is None.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Agent.update().
        """

        if not skip_checks:
            self.check_and_record_target_reached()
            if self.check_if_trajectory_end_reached():
                self.reset()

        # check for a forced next position
        if new_pos is not None:
            kwargs["forced_next_position"] = new_pos
        elif self.manual_pos:
            kwargs["forced_next_position"] = self.pos

        super().update(dt=dt, **kwargs)

        self.manual_pos = False

        if self.Environment.dimensionality == "1D" and self.fixed_direction:
            self._check_and_adjust_current_velocity_for_1D(dt=dt)

        self.current_trajectory_length += 1
        self.num_steps_total += 1

    def plot_trajectory(self, return_traj_fig=False, **kwargs) -> plt.Axes:
        """
        self.plot_trajectory()

        Redirects to plot agent trajectories. Used to enable
        ratinabox.Agent.animate_trajectory() to work.

        Args:
        - return_traj_fig (bool, optional): Whether to return the figure as well.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Agent.plot_trajectory().

        Returns:
        - sub_ax (plt.Axes): Subplot axis with the agent's trajectory plotted.
        """

        kwargs = plot_util.organize_fig_ax_kwargs(**kwargs)

        if "additional_plot_func" in kwargs.keys():
            kwargs.pop("additional_plot_func")

        if "return_env_fig" in kwargs.keys():
            kwargs.pop("return_env_fig")

        if "ax" in kwargs.keys():
            kwargs["sub_ax"] = kwargs.pop("ax")
            kwargs.pop("fig")

        sub_ax = self.plot_trajectories(**kwargs)

        if return_traj_fig:
            fig = sub_ax.figure
            return fig, sub_ax
        else:
            return sub_ax

    def plot_trajectories_to_date(
        self, in_minutes: bool = True, autosave: bool | None = None
    ) -> plt.Axes:
        """
        self.plot_trajectories_to_date()

        Plot the trajectory lengths to date.

        Args:
        - in_minutes (bool, optional): Whether to plot time axis in minutes, instead
            of seconds. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot axis with the trajectory lengths to date plotted.
        """

        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        sub_ax, _ = plot_fcts.plot_trajectory_lengths(
            dt=self.dt, trajectory_lengths=traj_leng_to_date, in_minutes=in_minutes  # type: ignore[has-type]
        )

        fig = sub_ax.figure
        plot_util.save_figure(fig, "trajectories_to_date", save=autosave)

        return sub_ax

    def plot_distance_to(
        self,
        position: None | np.ndarray[tuple[int], np.dtype[np.float64]] = None,
        position_name: str = "target",
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        alpha: float = 0.8,
        color: str = "k",
        tol_prop_to_speed_dt: float | None = None,
        zoom_prop: float | None = None,
        mark_below_tolerance: bool = False,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """
        self.plot_distance_to()

        Plot the distance an agent was from a specific position across the agent's
        history.

        Args:
        - position (1D np.ndarray, optional): Position to plot distance to. If None,
            position name is used. Default is None.
        - position_name (str, optional): Position name to use in plot y axis label.
            Also, if position is None, named position to use
            (e.g., 'target', 'reset', 'start'). Default is 'target'.
        - t_start (float, optional): Start time for plotting, in seconds.
            Default is None.
        - t_end (float, optional): End time for plotting, in seconds. Default is None.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - alpha (float, optional): Trajectory point opaqueness. Default is 0.8.
        - color (str, optional): Trajectory point color. Default is 'k'.
        - tol_prop_to_speed_dt (float, optional): Proportion of the tolerance to
            speed * dt to mark on plot. Default is None.
        - zoom_prop (float, optional): Proportion of the maximum distance measured
            to zoom plot in on. Default is None.
        - mark_below_tolerance (bool, optional): Whether to mark points below the
            tolerance. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
            global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with distance to position plotted.
        """

        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)

        t = t / 60  # to minutes
        positions = np.asarray(self.history["pos"])[startid : endid + 1]

        if position is None:
            if position_name == "target":
                position = self.target_position
                if tol_prop_to_speed_dt is None:
                    tol_prop_to_speed_dt = self.target_reached_within_tol_prop_to_speed_dt  # type: ignore[attr-defined]
            elif position_name == "reset":
                position = self.reset_position
                if tol_prop_to_speed_dt is None:
                    tol_prop_to_speed_dt = self.reset_reached_within_tol_prop_to_speed_dt  # type: ignore[attr-defined]
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
        if tol_prop_to_speed_dt is not None:
            speed = np.linalg.norm(np.asarray(self.history["vel"]), ord=2, axis=1)
            speed = speed[startid : endid + 1]
            y = speed * self.dt * tol_prop_to_speed_dt  # type: ignore[attr-defined]
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
        sub_ax.spines[["right", "top"]].set_visible(False)

        fig = sub_ax.figure
        plot_util.save_figure(fig, "distance", save=autosave)

        return sub_ax

    def add_position_across_time_to_plot(
        self,
        sub_ax,
        position_name="start",
        base_s=15,
        t_start=None,
        t_end=None,
        in_minutes=True,
        dim_idx=0,
        raise_error=True,
        **kwargs,
    ):
        """
        self.add_position_across_time_to_plot()

        Add a position to a subplot across time.

        Args:
        - sub_ax (plt.Axes): Subplot to plot on.
        - position_name (str, optional): Position name to plot.
            Must be 'start', 'reset' or 'target'. Default is 'start'.
        - base_s (int, optional): Base marker size. Default is 15.
        - t_start (float, optional): Start time in seconds. Default is None.
        - t_end (float, optional): End time in seconds. Default is None.
        - in_minutes (bool, optional): Whether to plot time axis in minutes.
            Default is True.
        - dim_idx (int, optional): Dimension index. Default is 0.
        - raise_error (bool, optional): Whether to raise an error if no positions
            are found for the specified position name. Default is True.

        Keyword args:
        - **kwargs: Additional keyword arguments passed to plt.scatter().
        """

        indices = self.get_reached_position_steps(position_name)
        if len(indices) == 0:
            if raise_error:
                raise ValueError(f"No {position_name} positions reached.")
            else:
                return

        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)

        if in_minutes:
            t = t / 60

        if position_name == "start":
            position = self.start_position
        elif position_name == "reset":
            position = self.reset_position
        elif position_name == "target":
            position = self.target_position
        else:
            raise NotImplementedError(
                "Position name must be 'start', 'reset' or 'target', "
                f"but got {position_name}."
            )

        if position is None:
            if raise_error:
                raise ValueError(f"{position_name} is set to None.")
            return

        xs = [t[x] for x in indices if x >= startid and x < endid]
        ys = [position[dim_idx]] * len(xs)

        plot_kwargs = plot_util.get_plot_marker_kwargs(position_name, base_s=base_s)
        plot_kwargs.update(kwargs)
        sub_ax.scatter(xs, ys, **plot_kwargs)

    def plot_trajectories_across_time(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        framerate: int | float = 10,
        sub_ax: plt.Axes | None = None,
        alpha: float = 0.6,
        color: str = "k",
        s: int | float = 5,
        plot_targets: bool = True,
        xlim: float | None = None,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """
        self.plot_trajectories_across_time()

        Plot trajectory positions across time, marking start and reset positions.

        If environment is 2D, x and y positions are plotted over one another.

        From Agent.plot_1D_trajectories() in ratinabox/agent.py. Modified to enable
        plotting of reset steps, and use of colormaps for trajectories.

        Args:
        - t_start (float, optional): Start time in seconds. Default is None.
        - t_end (float, optional): End time in seconds. Default is None.
        - framerate (int or float, optional): How many scatter points / per second of
            motion to display. Default is 10.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - alpha (float, optional): Trajectory point transparency. Default is 0.6.
        - color (str, optional): Trajectory point color or colors. Default is 'k'.
        - s (float, optional): Size of scatterplot markers. Default is 5.
        - plot_targets (bool, optional): Whether to plot the target. Default is True.
        - xlim (float, optional): Upper x axis limit to set (in minutes).
            Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectory positions plotted across time.
        """

        dt = self.dt
        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)
        pos = np.asarray(self.history["pos"])

        skiprate = max(1, int((1 / framerate) / dt))

        time = t[::skiprate] / 60  # in minutes
        pos = pos[startid : endid + 1][::skiprate]

        # get reset step indices
        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")
        elif len(time) == 0:
            raise RuntimeError("Duration too short. No time points to plot.")

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(8, 5))

        positions_to_plot = ["start", "reset"]
        if plot_targets:
            positions_to_plot.append("target")

        alpha /= self.Environment.D
        alpha_pts = 0.9 / self.Environment.D
        for dim_idx in range(self.Environment.D):
            sub_ax.scatter(
                time,
                pos,
                alpha=alpha,
                marker=mpl_markers.MarkerStyle("."),
                color=color,
                s=s,
            )

            for position_name in positions_to_plot:
                self.add_position_across_time_to_plot(
                    sub_ax,
                    position_name=position_name,
                    base_s=s * 3,
                    alpha=alpha_pts,
                    t_start=t_start,
                    t_end=t_end,
                    in_minutes=True,
                    dim_idx=dim_idx,
                    raise_error=(position_name != "target"),
                )

        if self.Environment.D == 1:
            min_y, max_y = self.Environment.extent
            diff = max_y - min_y
        elif self.Environment.D == 2:
            left, right, bottom, top = self.Environment.extent
            diff = max(right - left, top - bottom)
            min_y, max_y = min(left, bottom), max(right, top)
        else:
            raise RuntimeError("Only 1D and 2D environments are supported.")

        # adjust y limits
        bottom = min_y - diff * 0.1  # type: ignore[operator]
        top = max_y + diff * 0.1  # type: ignore[operator]
        sub_ax.set_ylim(bottom=bottom, top=top)

        if xlim is not None:
            sub_ax.set_xlim(right=xlim)

        sub_ax.set_xlabel("Time / min")
        sub_ax.set_ylabel("Position / m")
        sub_ax.spines[["right", "top"]].set_visible(False)

        fig = sub_ax.figure
        plot_util.save_figure(fig, "trajectory_resets", save=autosave)

        return sub_ax

    def add_agent_to_plot(
        self,
        sub_ax,
        coords=None,
        agent_color="r",
        head_direction=None,
        zorder=None,
        s=40,
        alpha=1,
    ):
        """
        self.add_agent_to_plot()

        Add the agent to a subplot.

        Args:
        - sub_ax (plt.Axes): Subplot to plot on.
        - coords (1D np.ndarray, optional): Agent coordinates to plot. If None,
            current position is used. Default is None.
        - agent_color (str, optional): Agent color. Default is 'r'.
        - head_direction (1D np.ndarray, optional): Agent head direction.
            If None and coords is None, current head direction is plotted. Otherwise,
            head direction is not plotted. Default is None.
        - zorder (int, optional): Z order. Default is None.
        - s (int, optional): Marker size. Default is 40.
        - alpha (float, optional): Marker transparency. Default is 1.
        """

        if coords is None:
            coords = self.pos
            if head_direction is None:
                head_direction = self.head_direction
        else:
            coords = self.format_position(coords)

        sub_ax.scatter(
            *coords,
            s=s,
            zorder=3,
            c=agent_color,
            linewidth=0,
            marker="o",
            alpha=alpha,
        )

        # plot head direction
        if head_direction is not None:
            rotated_agent_marker = mpl_markers.MarkerStyle(
                marker=[(-1, 0), (1, 0), (0, 4)]
            )  # a triangle
            rotated_agent_marker._transform = (
                rotated_agent_marker.get_transform().rotate_deg(
                    -rutils.get_bearing(head_direction) * 180 / np.pi
                )
            )
            sub_ax.scatter(
                *coords,
                s=s * 5,
                alpha=alpha,
                zorder=zorder,
                c=agent_color,
                linewidth=0,
                marker=rotated_agent_marker,
            )

    def plot_trajectories(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        framerate: int | float = 10,
        sub_ax: plt.Axes | None = None,
        decay_point_size: bool = False,
        decay_point_timescale: float = 10,
        plot_agent: bool = True,
        agent_color: str = "r",
        plot_head_direction: bool = False,
        colormap: str | None | mpl_colors.Colormap = None,
        alpha: float = 0.7,
        xlim: float | None = None,
        plot_traj_ends: bool = True,
        target_alpha: float = 1.0,
        plot_target: bool = True,
        cmap_per: bool = False,
        scale_cmap_per: bool = False,
        s_2D: int | float = 15,
        size_fact: float | None = None,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """
        self.plot_trajectories()

        Plot the agent's trajectories. If environment is 1D, trajectories are
        plotted across time.

        From Agent.plot_trajectory() in ratinabox/agent.py. Modified to enable plotting
        of reset steps, and use of colormaps for trajectories.

        Args:
        - t_start (float, optional): Start time in seconds. Default is None.
        - t_end (float, optional): End time in seconds. Default is None.
        - framerate (int or float, optional): How many scatter points / per second of
            motion to display. Default is 10.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - decay_point_size (bool, optional): Whether to decay trajectory point size
            over time, with recent timepoints being plotted largest. Only applies to 2D
            environments. Default is False.
        - decay_point_timescale (float, optional): Time scale over which to decay
            point size. Default is 10.
        - plot_agent (bool, optional): Whether to plot agent's last position.
            Default is True.
        - agent_color (str, optional): Color for agent point, if applicable.
            Default is 'r'.
        - plot_head_direction (bool, optional): Whether to plot the agent's last head
            direction. Default is False.
        - colormap (str, optional): Colormap to use to plot trajectories.
            Default is None.
        - alpha (float, optional): Trajectory point transparency. Default is 0.6.
        - xlim (float, optional): Upper x axis limit to set (in minutes),
            if environment is 1D. Default is None.
        - plot_traj_ends (bool, optional): Whether to plot a point at the end of each
            trajectory, if environment is 2D. Default is True.
        - target_alpha (float, optional): Transparency to use for target position, if
            environment is 2D. Default is 1.0.
        - plot_targets (bool, optional): Whether to plot the target. Default is True.
        - cmap_per (bool, optional): if True, the colormap is used to set the color
            for each time point. Otherwise, each trajectory has its own color.
            Default is False.
        - scale_cmap_per (bool, optional): if True, and cmap_per is True, the full
            range of the colormap is used for each trajectory, regardless of its length.
            Default is False.
        - s_2D (float, optional): Size of trajectory points, if environment is 2D.
            Default is 15.
        - size_fact (float, optional): Size by which to multiply environment size,
            if environment is 2D. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectories plotted.
        """

        dt = self.dt
        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)
        pos = np.asarray(self.history["pos"])

        skiprate = max(1, int((1 / framerate) / dt))
        t = t[::skiprate]
        idx = np.arange(startid, endid + 1, skiprate)
        trajectory = pos[idx]

        time = t / 60  # in minutes

        # get reset step indices
        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")
        elif len(time) == 0:
            raise RuntimeError("Duration too short. No time points to plot.")

        trajectory_lengths = self.get_trajectory_lengths_to_date()
        colors = plot_util.get_trajectory_cmap_colors(
            trajectory_lengths,
            colormap,
            cmap_per=cmap_per,
            scale_cmap_per=scale_cmap_per,
            time_idx=idx,
        )

        full_traj_idx = [
            np.full(steps, i) for i, steps in enumerate(trajectory_lengths)
        ]
        traj_idx = np.concatenate(full_traj_idx).astype(int)[idx]

        if self.Environment.dimensionality == "2D":
            if size_fact is not None:
                figsize = self.Environment.get_environment_figsize(size_fact)
                _, sub_ax = plt.subplots(figsize=figsize)
            sub_ax = self.Environment.plot_environment(sub_ax=sub_ax)

            if plot_target and self.target_position is not None:
                sub_ax.scatter(
                    *self.target_position,
                    zorder=5,
                    alpha=target_alpha,
                    label="target",
                    **plot_util.get_plot_marker_kwargs("target"),
                )

            s = s_2D * np.ones_like(time)
            if decay_point_size == True:
                s = s_2D * np.exp((time - time[-1]) / decay_point_timescale)
                s[(time[-1] - time) > 1.5 * decay_point_timescale] *= 0

            if plot_traj_ends == True and len(self.trajectory_df) - 1 > 0:
                ends = np.where(np.diff(traj_idx) > 0)[0]
                ends = np.append(ends, len(trajectory) - 1)
                s[ends] = s_2D * 2
                # set last colormap value to dark red
                colors[ends] = mpl_colors.to_rgba("darkred")  # type: ignore[arg-type]

            sub_ax.scatter(
                *trajectory.T,
                s=s,
                alpha=alpha,
                zorder=2,
                c=colors,
                linewidth=0,
            )

            if plot_agent == True:
                head_direction = None
                if plot_head_direction:
                    head_direction = self.history["head_direction"][idx[-1]]

                self.add_agent_to_plot(
                    sub_ax,
                    coords=trajectory[-1],
                    agent_color=agent_color,
                    head_direction=head_direction,
                    zorder=3,
                    s=s_2D * 2.75,
                    alpha=0.9,
                )

        if self.Environment.dimensionality == "1D":
            sub_ax = self.plot_trajectories_across_time(
                t_start=t_start,
                t_end=t_end,
                framerate=framerate,
                sub_ax=sub_ax,
                alpha=alpha,
                color=colors,
                plot_targets=plot_target,
                xlim=xlim,
                autosave=False,
            )

        fig = sub_ax.figure
        plot_util.save_figure(fig, "trajectory", save=autosave)

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
        """
        self.plot_trajectory_edges()

        Plot trajectory starts and ends.

         Args:
         - t_start (float, optional): Start time in seconds. Default is None.
         - t_end (float, optional): End time in seconds. Default is None.
         - sub_ax (plt.Axes, optional): Subplot to plot on. If None,
             a new subplot is created using self.Environment.plot_environment().
             This can be used to plot trajectory on top of receptive fields etc.
             Default is None.
        - decay_point_size (bool, optional): Whether to decay trajectory point size
            over time, with recent timepoints being plotted largest. Only applies to 2D
            environments. Default is False.
        - plot_agent (bool, optional): Whether to plot agent's current position.
            Default is True.
        - colormap (str, optional): Colormap to use to plot trajectory starts and ends.
            Default is None.
        - alpha (float, optional): Trajectory point transparency. Default is 0.6.
        - xlim (float, optional): Upper x axis limit to set (in minutes), if
            environment is 1D. Default is None.
        - background_color (str, optional): Color of the background to use for a 1D
            environment plot. Default is None.
        - plot_starts (False, optional): Whether to plot trajectory starts.
            Default is True.
        - plot_ends (False, optional): Whether to plot trajectory ends. Default is True.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

         Returns:
         - sub_ax (plt.Axes): Subplot with trajectory edges plotted.
        """

        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)
        pos = np.asarray(self.history["pos"])

        t = t / 60  # minutes

        if colormap is None:
            colormap = "crest"
        cmap = sns.color_palette(colormap, as_cmap=True)

        actual_trajectory_lengths = self.get_trajectory_lengths_to_date()
        all_ends = np.cumsum(actual_trajectory_lengths)

        traj_plot_components = list()
        if plot_starts:
            kwargs = plot_util.get_plot_marker_kwargs("start")
            kwargs["lw"] = 1
            traj_plot_components.append((np.insert(all_ends, 0, 0)[:-1], kwargs))
        if plot_ends:
            kwargs = plot_util.get_plot_marker_kwargs("reset")
            kwargs["lw"] = 2
            traj_plot_components.append((all_ends - 1, kwargs))

        if not (plot_starts or plot_ends):
            raise ValueError("At least 'plot_starts' or 'plot_ends' must be True.")

        for traj_idxs, kwargs in traj_plot_components:
            kwargs["color"] = cmap(np.linspace(0, 1, len(traj_idxs)))  # type: ignore[callable]
            if "s" in kwargs.keys():
                kwargs.pop("s")

            traj_idxs = traj_idxs[(traj_idxs >= startid) & (traj_idxs <= endid)]
            trajectory = pos[traj_idxs]
            time = t[traj_idxs]

            if len(time) == 0:
                raise RuntimeError("Duration too short. No trajectory points to plot.")

            if self.Environment.dimensionality == "2D":
                sub_ax = self.Environment.plot_environment(sub_ax=sub_ax)
                if sub_ax is None:
                    raise RuntimeError("sub_ax is None.")
                if self.target_position is not None:
                    target_kwargs = plot_util.get_plot_marker_kwargs("target")
                    sub_ax.scatter(*self.target_position, zorder=5, **target_kwargs)

                s = 15 * np.ones_like(time)
                if decay_point_size == True:
                    s = 15 * np.exp((time - time[-1]) / 10)
                    s[(time[-1] - time) > 15] *= 0

                if plot_agent == True:
                    s[-1] = 40
                    # set last colormap value to red
                    kwargs["color"][-1] = mpl_colors.to_rgba("r")  # type: ignore[arg-type]

                sub_ax.scatter(*trajectory.T, s=s, alpha=alpha, zorder=2, **kwargs)

            elif self.Environment.dimensionality == "1D":
                if sub_ax is None:
                    _, sub_ax = plt.subplots(figsize=(3, 1.5))
                sub_ax.scatter(time / 60, trajectory, alpha=alpha, s=5, **kwargs)

                sub_ax.set_xlim(t[0] / 60, t[-1] / 60)
                sub_ax.set_xticks([t[0] / 60, t[-1] / 60])
                if xlim is not None:
                    sub_ax.set_xlim(right=xlim)

                sub_ax.set_yticks([self.Environment.extent[1]])
                sub_ax.set_ylim(bottom=0, top=self.Environment.extent[1])

                sub_ax.set_xlabel("Time / min")
                sub_ax.set_ylabel("Position / m")
                sub_ax.spines[["right", "top"]].set_visible(False)

                if background_color is not None:
                    sub_ax.set_facecolor(background_color)
                    sub_ax.figure.patch.set_facecolor(background_color)  # type: ignore[attr-defined]

        if sub_ax is None:
            raise RuntimeError("sub_ax is None.")

        fig = sub_ax.figure
        plot_util.save_figure(fig, "trajectory_edges", save=autosave)

        return sub_ax


class TAgent(ResetableAgent):
    """
    TAgent()

    Class extending the ResetAgent so that it operates in a T-maze.

    Must be initialised with an environment. A parameters dictionary can also be passed.

    default_params = {
        "target_arm": "left",  # type: ignore[dict-item]
        "target_location_prop_to_arm": 0.75,  # proportion down arm at which to set target
        "left_arm_prop": 0.75,  # proportion of trajectories to target to left arm
    }

    List of properties (in addition to ResetableAgent properties):
        • self.near_branch_point
        • self.at_branch_point
        • self.target_df_columns

    List of methods (in addition to ResetableAgent methods):
        • self.set_all_positions()
        • self.set_current_trajectory_arm()
        • self.get_direction_to_end()
        • self.check_if_reset_position_reached()
        • self.check_if_left_reset_position_reached()
        • self.check_if_right_reset_position_reached()
        • self.update()
        • self.reset()
    """

    default_params = {
        "target_arm": "left",  # type: ignore[dict-item]
        "target_location_prop_to_arm": 0.75,  # proportion down arm at which to set target
        "left_arm_prop": 0.75,  # proportion of trajectories to target to left arm
    }

    ignored_param_keys = ["reset_position", "start_position", "target_position"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Env: env.TEnv, params: dict[str, Any] = dict()):
        """
        TAgent(Env)

        Initialise the T-maze agent.

        Args:
        - params (dict, optional): Agent parameters. Default is dict().

        Raises:
        - ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        if not isinstance(Env, env.TEnv):
            raise TypeError("Env must be a TEnv object.")

        super().__init__(Env, self.params)

        self.set_current_trajectory_arm()

    @property
    def near_branch_point(self) -> bool:
        """
        self.near_branch_point

        Whether the agent is near the T-maze branch point.

        Returns:
        - (bool): Whether the agent is near the T-maze branch point.
        """

        return self.pos[1] > (self.Environment.branch_y * 0.98)

    @property
    def at_branch_point(self) -> bool:
        """
        self.at_branch_point

        Whether the agent has reached the T-maze branch point.

        Returns:
        - (bool): Whether the agent has reached the T-maze branch point.
        """

        return self.pos[1] > self.Environment.branch_y

    @property
    def target_df_columns(self) -> list:
        """
        self.target_df_columns

        Obtain the target dataframe column names.

        Returns:
        - (list) List of target dataframe column names.
        """

        if not hasattr(self, "_target_df_columns"):
            self._target_df_columns = [
                "target_arm",
                "position_x",
                "position_y",
                "set_step",
                "set_time",
                "reached_step",
                "reached_time",
                "num_steps_total",
                "time_total",
            ]

        return self._target_df_columns

    def _add_new_target_to_df(self):
        """
        self._add_new_target_to_df()

        Add a new line to the target dataframe.
        """

        if self.target_position is None:
            raise RuntimeError("Target position is None.")

        self._end_target_df_line()

        target_data = {
            "target_arm": self.target_arm,  # type: ignore[attr-defined]
            "position_x": self.target_position[0],
            "position_y": self.target_position[1],
            "set_step": self.num_steps_total,
            "set_time": self.t,
        }
        self.target_df.loc[len(self.target_df)] = target_data  # type: ignore[assignment]

    def set_all_positions(self):
        """
        self.set_all_positions()

        Set all the positions for the agent.

        Attributes:
        - _target_object_idx (int): Index of the target object in the environment.
        - left_reset_position (1D np.ndarray): Left arm reset position for the agent.
        - reset_position (list): List of reset positions.
        - right_reset_position (1D np.ndarray): Right arm reset position for the agent.
        - start_position (1D np.ndarray): Start position of the agent.
        - steps_before_checking_for_target (int): Number of steps to wait before
            checking for target.
        - target_position (1D np.ndarray): Target position of the agent.
        """

        self.start_position = self.format_position(self.Environment.T_start)

        # set reset positions
        self.left_reset_position = self.format_position(self.Environment.left_T_end)
        self.right_reset_position = self.format_position(self.Environment.right_T_end)
        self.reset_position = [self.left_reset_position, self.right_reset_position]

        # set target position
        target_arm = self.target_arm  # type: ignore[attr-defined]
        if target_arm == "left":
            edge = self.Environment.left_T_end
        elif target_arm == "right":
            edge = self.Environment.right_T_end
        else:
            raise ValueError("target_arm must be 'left' or 'right'.")

        T_split = self.Environment.T_split
        target_position = [
            T_split[i] + (edge[i] - T_split[i]) * self.target_location_prop_to_arm  # type: ignore[operator]
            for i in [0, 1]
        ]
        self.target_position = self.format_position(target_position)
        self.Environment.add_object(self.target_position)
        self._target_object_idx = len(self.Environment.objects["objects"]) - 1

        self.steps_before_checking_for_target = 0

        # set initial position and velocity
        if self.start_position is not None:
            self.set_position_and_velocity(position=self.start_position, velocity=0)

    def set_target_position(self, position):
        """
        self.set_target_position(position)

        Set the target position.

        Attributes:
        - target_arm (str): Arm of the target.
        - target_in_branch (bool): Whether the target is in the branch.
        - target_position (1D np.ndarray): Target position of the agent.

        Args:
        - position (1D np.ndarray): Position to set the target to.
        """

        super().set_target_position(position)

        if self.target_position is None:
            self.target_arm = None
            self.target_in_branch = False
        else:
            x, y = self.target_position
            if y < self.Environment.branch_y:
                self.target_arm = None
                self.target_in_branch = False
            else:
                self.target_in_branch = True
                if x > self.Environment.stim_right:
                    self.target_arm = "right"
                elif x < self.Environment.stim_left:
                    self.target_arm = "left"
                else:
                    self.target_arm = None

    def set_current_trajectory_arm(self, arm: str = "random"):
        """
        self.set_current_trajectory_arm()

        Set the current trajectory arm.

        Attributes:
        - current_arm (str): Current arm of the trajectory.

        Args:
        - arm (str, optional): Arm to Initialise the trajectory to. If random, arm is
            randomly set using self.left_arm_prop. Default is 'random'.

        Raises:
        - ValueError: If arm is not 'random', 'left' or 'right'.
        """

        arms = ["left", "right"]
        if arm == "random":
            arm_idx = int(np.random.rand() > self.left_arm_prop)  # type: ignore[attr-defined]
            self.current_arm = arms[arm_idx]
        elif arm in arms:
            self.current_arm = arm
        else:
            raise ValueError(f"Arm must be 'random', 'left' or 'right', but got {arm}.")

    def get_direction_to_end(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        self.get_direction_to_end()

        Obtain the direction from the current agent position to the trajectory end.

        Returns:
        - direction (1D np.ndarray): Direction from the current agent position to the
            end of the trajectory arm.
        """

        if self.current_arm == "left":
            trajectory_end = self.Environment.left_T_end

        elif self.current_arm == "right":
            trajectory_end = self.Environment.right_T_end

        else:
            raise RuntimeError("Current arm must be 'left' or 'right'.")

        direction = np.asarray(trajectory_end) - self.pos

        return direction

    def check_if_reset_position_reached(self, position: str = "both") -> bool:
        """
        self.check_if_reset_position_reached()

        Check if the agent has reached either of the reset positions.

        Args:
        - position (str, optional): Position to check. Must be 'both', 'left' or 'right'.
            Default is 'both'.

        Returns:
        - reset_position_reached (bool): Whether the agent has reached the specified
            reset position(s).
        """

        # calculate the distance between the current position and the reset position
        if position == "both":
            reset_positions_to_check = self.reset_position
        elif position == "left":
            reset_positions_to_check = [self.left_reset_position]
        elif position == "right":
            reset_positions_to_check = [self.right_reset_position]
        else:
            raise ValueError("pos must be 'both', 'left', or 'right'.")

        distances = [
            np.linalg.norm(self.pos - reset_position_to_check, ord=2)  # type: ignore[operator]
            for reset_position_to_check in reset_positions_to_check
        ]
        distance = min(distances)

        # check if the distance is less than the tolerance
        reset_position_reached = False
        if distance < (np.absolute(self.speed_mean) * self.reset_reached_within_tol_prop_to_speed_dt):  # type: ignore[attr-defined]
            reset_position_reached = True

        return reset_position_reached

    def check_if_left_reset_position_reached(self) -> bool:
        """
        self.check_if_left_reset_position_reached()

        Check if the agent has reached the left reset position.

        Returns:
        - (bool): Whether the agent has reached the left reset position.
        """

        return self.check_if_reset_position_reached(position="left")

    def check_if_right_reset_position_reached(self) -> bool:
        """
        self.check_if_right_reset_position_reached()

        Check if the agent has reached the right reset position.

        Returns:
        - (bool): Whether the agent has reached the right reset position.
        """

        return self.check_if_reset_position_reached(position="right")

    def reset(self):
        """
        self.reset()

        Reset the agent, setting a new trajectory arm.
        """

        super().reset()

        self.set_current_trajectory_arm()

    def update(  # type: ignore[override]
        self,
        dt: float | None = None,
        speed_fact: int | float = 3,
        drift_to_random_strength_ratio: float = 0.7,
        **kwargs,
    ):
        """
        self.update()

        Update the agent's position. Checks whether target is reached. Checks whether
        trajectory has ended, and if so, resets the agent. If agent is near the branch
        point and the target is in a branch, the agent drifts towards the target.
        Otherwise, the agent drifts towards the branch point.

        Args:
        - dt (float, optional): Time step. Default is None.
        - speed_fact (int or float, optional): Factor for speed. Default is 3.
        - drift_to_random_strength_ratio (float, optional): Ratio of drift to random
            strength. Default is 0.7.

        Keyword args:
            - **kwargs: Keyword arguments passed to ratinabox.Agent.update().
        """

        self.check_and_record_target_reached()
        if self.check_if_trajectory_end_reached():
            self.reset()

        # calculate drift_velocity
        if self.near_branch_point:
            direction = self.get_direction_to_end()
        else:
            direction = self.Environment.T_split - self.pos
        drift_velocity = (
            speed_fact
            * self.speed_mean  # type: ignore[attr-defined]
            * (direction / np.linalg.norm(direction, ord=2))
        )

        super().update(
            dt=dt,
            skip_checks=True,  # aleady done
            drift_velocity=drift_velocity,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
            **kwargs,
        )


class OpenFieldAgent(ResetableAgent):
    """
    OpenFieldAgent()

    Class extending the ResetAgent so that it operates in an open field.

    Must be initialised with an environment. A parameters dictionary can also be passed.

    default_params = {
        "reward_factor": 5,  # factor for setting a reward object as a target for a trajectory
        "no_target_factor": 1,  # factor for not setting any target for a trajectory
        "trajectory_length": 2000,  # int or iterable of ints
        "num_trajectories": 10,  # number of trajectory lengths to sample
        "wait_between_targets": 10,  # number of steps to wait between target reaching
        "target_reached_within_tol_prop_to_speed_dt": 0.55,  # proportion of mean speed * dt to use as target tolerance
        "num_random_walk_steps": 100,  # number of steps to random walk, if target is not in sight
        "always_log_teleportation": False,  # whether to log teleportation events when they occur
    }

    List of properties (in addition to ratinabox.Agent properties):
        • self.target_df_columns
        • self.teleportation_df

    List of methods (in addition to ratinabox.Agent methods):
        • self.set_all_positions()
        • self.sample_position_within_tolerance()
        • self.get_target_probability_df()
        • self.check_target_reached_during_random_walk()
        • self.check_if_target_is_in_sight()
        • self.check_if_teleport_angles_in_range()
        • self.check_if_teleport_in_should_activate()
        • self.get_teleport_vector()
        • self.get_drift_velocity()
        • self.anticipate_position_update()
        • self.sample_teleport_out_position()
        • self.get_matched_teleport_out_position()
        • self.get_teleport_rotated_velocity()
        • self.get_teleport_coords_if_applicable()
        • self.log_teleportation()
        • self.reset()
        • self.update()
        • self.get_agent_color_for_trajectory()
        • self.add_target_to_plot()
        • self.plot_trajectories()
        • self.plot_trajectory_targets()
        • self.plot_trajectory_target_coords_over_time()
        • self.animate_trajectories()
    """

    default_params = {
        "reward_factor": 5,  # factor for setting a reward object as a target for a trajectory
        "no_target_factor": 1,  # factor for not setting any target for a trajectory
        "trajectory_length": 2000,  # int or iterable of ints
        "num_trajectories": 10,  # number of trajectory lengths to sample
        "wait_between_targets": 10,  # number of steps to wait between target reaching
        "target_reached_within_tol_prop_to_speed_dt": 0.55,  # proportion of mean speed * dt to use as target tolerance
        "num_random_walk_steps": 100,  # number of steps to random walk, if target is not in sight
        "always_log_teleportation": False,  # whether to log teleportation events when they occur
    }

    ignored_param_keys = [
        "reset_position",
        "start_position",
        "target_position",
        "reset_reached_within_tol_prop_to_speed_dt",
        "fixed_direction",
    ]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()  # type: dict[str, Any]

    def __init__(self, Env: env.OpenField, params: dict[str, Any] = dict()):
        """
        OpenFieldAgent(Env)

        Initialise the open field agent.

        Args:
        - params (dict, optional): Agent parameters. Default is dict().

        Raises:
        - ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.params = copy.deepcopy(__class__.default_params)  # type: ignore[name-defined]
        self.params.update(params)

        if not isinstance(Env, env.OpenField):
            raise TypeError("Env must be an OpenField object.")

        super().__init__(Env, self.params)

    @property
    def target_df_columns(self) -> list:
        """
        self.target_df_columns

        Obtain the target dataframe column names.

        Returns:
        - (list) List of target dataframe column names.
        """

        if not hasattr(self, "_target_df_columns"):
            self._target_df_columns = [
                "object_idx",
                "object_type_name",
                "object_type_num",
                "position_x",
                "position_y",
                "set_step",
                "set_time",
                "random_walk_periods",
                "reached_step",
                "reached_time",
                "num_random_walk_steps",
                "random_walk_time",
                "num_steps_total",
                "time_total",
            ]

            # add a target in sight column with lists?

        return self._target_df_columns

    @property
    def teleportation_df(self):
        """
        self.teleportation_df

        Obtain the teleportation dataframe, which records variables around
        teleportation events.

        Returns:
        - teleportation_df (pd.DataFrame): Dataframe of teleportation events.
        """

        if not hasattr(self, "_teleportation_df"):
            teleportation_columns = [
                "teleport_pair_num",
                "step_num",
                "time",
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

    def _set_random_walk(self):
        """
        self._set_random_walk()

        If there is no current target or it is not in sight, starts a random walk and
        adds it to the current line of the target dataframe. Otherwise, sets the number
        of random walk steps to 0.

        Attributes:
        - current_num_of_random_walk_steps (int): Number of steps in the current
            random walk.
        """

        if self.target_position is None or not self.check_if_target_is_in_sight():
            self.current_num_of_random_walk_steps = int(self.num_random_walk_steps)  # type: ignore[attr-defined]
            df_idx = len(self.target_df) - 1
            column = "random_walk_periods"
            self.target_df.loc[df_idx, column].append([self.num_steps_total])  # type: ignore[has-method]
        else:
            self.current_num_of_random_walk_steps = 0

    def _end_random_walk(self):
        """
        self._end_random_walk()

        End a current random walk, if applicable.

        Attributes:
        - current_num_of_random_walk_steps (int): Number of steps in the current
            random walk.
        """

        if self.current_num_of_random_walk_steps == 0:
            return

        df_idx = len(self.target_df) - 1
        column = "random_walk_periods"
        self.target_df.loc[df_idx, column][-1].append(self.num_steps_total)
        self.current_num_of_random_walk_steps = 0

    def _end_trajectory(self):
        """
        self._end_trajectory()

        End the current trajectory. Also checks whether a current random walk should
        end.
        """

        super()._end_trajectory()

        self._end_random_walk()

    def _end_target_df_line(self):
        """
        self._end_target_df_line()

        End the current target dataframe line by adding the total number of steps and
        time taken (whether target was reached or not). Also checks whether a current
        random walk should end, and computes random walk values for the dataframe line.
        """

        if len(self.target_df) == 0:
            return

        super()._end_target_df_line()

        # add random walk steps
        self.check_target_reached_during_random_walk()
        self._end_random_walk()

        df_idx = len(self.target_df) - 1
        random_walk_periods = np.asarray(
            self.target_df.loc[df_idx, "random_walk_periods"]
        )
        num_random_walk_steps = 0
        if len(random_walk_periods) > 0:
            num_random_walk_steps = np.sum(np.diff(random_walk_periods, axis=1))
        self.target_df.loc[df_idx, "num_random_walk_steps"] = num_random_walk_steps
        self.target_df.loc[df_idx, "random_walk_time"] = num_random_walk_steps * self.dt

    def _add_new_target_to_df(self, target: str | None = None):
        """
        self._add_new_target_to_df()

        Add a new line to the target dataframe with new target. If target is not
        provided, selects a new target based on the target probability dataframe.

        Attributes:
        - steps_before_checking_for_target (int): Number of steps to wait before
            checking for target.
        - target_position (1D np.ndarray): Current target position for the agent.

        Args:
        - target (str, optional): The target to set. If None, new target is selected
            randomly. Default is None.
        """

        self._end_target_df_line()

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
        target_data["set_time"] = self.t  # type: ignore[assignment]
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

    def _init_target_df(self):
        """
        self._init_target_df()

        Initialise the target dataframe, which records the target position and the
        step at which it was reached. Adds a first row and determines whether a random
        walk will start.
        """

        self.target_df = pd.DataFrame(columns=self.target_df_columns)

        self.target_df["random_walk_periods"] = self.target_df[
            "random_walk_periods"
        ].astype(object)

        self._add_new_target_to_df()
        self._set_random_walk()

    def set_all_positions(self, first_setting: bool = True, target: str | None = None):
        """
        self.set_all_positions()

        Set all the positions for the agent.

        Attributes:
        - manual_pos (bool): Whether the agent's position was set manually.
        - start_position (1D np.ndarray): Start position of the agent.
        - steps_before_checking_for_target (int): Number of steps to wait before
        - target_position (1D np.ndarray): Target position of the agent.

        Args:
        - first_setting (bool, optional): Whether this is the first position setting.
            Default is True.
        - target (str, optional): The target to use. Ignored if first_setting is True.
            Default is None.
        """

        # set initial position and velocity
        self.start_position = self.Environment.sample_coords()
        self.set_position_and_velocity(position=self.start_position, velocity=0)

        self.target_position = None
        if not first_setting:
            self._add_new_target_to_df(target=target)

        self.steps_before_checking_for_target = 0

        if not first_setting:
            self._set_random_walk()

    def sample_position_within_tolerance(
        self,
        position: np.ndarray[tuple[int], np.dtype[np.float64]],
        sample_within_tol_prop_to_speed_dt: float | None = None,
        max_attempts: int = 100,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        self.sample_position_within_tolerance(position)

        Sample a position within the tolerance of the given position.

        Args:
        - position (1D np.ndarray): The position to sample around.
        - sample_within_tol_prop_to_speed_dt (float, optional): The proportion of the
            tolerance to sample within. Default is None, in which case the agent's
            target_reached_within_tol_prop_to_speed_dt is used.
        - max_attempts (int, optional): Maximum number of attempts for sampling within
            the tolerance. Default is 100.

        Returns:
        - sampled_position (1D np.ndarray): The sampled position.
        """

        if len(position) != 2:
            raise ValueError(f"position must have length 2, but found {len(position)}.")

        if sample_within_tol_prop_to_speed_dt is None:
            prop_to_speed_dt = self.target_reached_within_tol_prop_to_speed_dt  # type: ignore[attr-defined]
        else:
            prop_to_speed_dt = sample_within_tol_prop_to_speed_dt

        sampled_position = super().sample_position_within_tolerance(
            position,
            sample_within_tol_prop_to_speed_dt=prop_to_speed_dt,
            max_attempts=max_attempts,
        )

        return sampled_position

    def get_target_probability_df(self) -> pd.DataFrame:
        """
        self.get_target_probability_df()

        Obtain the target probability dataframe.

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

    def check_target_reached_during_random_walk(self):
        """
        self.check_target_reached_during_random_walk()

        Check whether the target was reached during a random walk, and if so, end
        random walk.
        """

        if self.current_num_of_random_walk_steps != 0 and self.reached_target:
            self._end_random_walk()

    def check_if_target_is_in_sight(self) -> bool:
        """
        self.check_if_target_is_in_sight()

        Check if the current target is in sight.

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

    def check_if_teleport_angles_in_range(
        self,
        teleport_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        teleport_coords: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        check_value: str | np.ndarray[tuple[int], np.dtype[np.float64]] = "position",
    ) -> bool:
        """
        self.check_if_teleport_angles_in_range(teleport_vector)

        Check whether the agent is within range for the teleportation to activate.
        Checks whether the agent is heading towards the teleport port, within 45
        degrees of the teleport vector, based on position or velocity.

        Args:
        - teleport_vector (1D np.ndarray): The teleport port vector.
        - teleport_coords (1D np.ndarray, optional): The teleport port coordinates.
            Not needed if 'check_value' is 'velocity'. Default is None.
        - check_value (str or 1D np.ndarray, optional): The value to check
            ('position' or 'velocity'). Default is 'position'.

        Returns:
        - within_range (bool): Whether the agent position or velocity matches the
            teleportation vector.
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

        check_value = copy.deepcopy(np.asarray(check_value))

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
        """
        self.check_if_teleport_in_should_activate()

        Check whether the agent should teleport through a specified port.

        Args:
        - teleport_pair_num (int): The teleport pair to check.

        Returns:
        - teleport (bool): Whether the agent should teleport through the specified port.
        """

        teleport_coords = self.Environment.get_teleport_coords(
            teleport_pair_num, direction="in"
        )

        tol_prop_to_speed_dt = self.target_reached_within_tol_prop_to_speed_dt  # type: ignore[attr-defined]

        teleport = False

        # check if close to teleport in
        near_teleport = self.check_if_position_reached(
            teleport_coords, tol_prop_to_speed_dt
        )
        if near_teleport:
            # check if agent is within 45 degrees, either side of the teleport in
            teleport_vector = self.get_teleport_vector(
                teleport_pair_num, direction="in"
            )
            teleport_angles = self.check_if_teleport_angles_in_range(
                teleport_vector, teleport_coords, check_value="position"
            )

            if teleport_angles:
                # check if agent is heading towards teleport in
                heading_teleport = self.check_if_teleport_angles_in_range(
                    teleport_vector, check_value="velocity"
                )

                if heading_teleport:
                    teleport = True

        return teleport

    def get_teleport_vector(
        self, teleport_pair_num: int = 0, direction: str = "in"
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        self.get_teleport_vector()

        Obtain the teleport vector for a specified teleport pair.

        Args:
        - teleport_pair_num (str): The teleport pair to get the vector for.

        Returns:
        - teleport_vector (1D np.ndarray): The teleport vector.
        """

        marker = self.Environment.get_teleport_plotting_marker(
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

    def get_drift_velocity(
        self,
        pos: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        speed_fact: int | float = 3,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """
        self.get_drift_velocity()

        Obtain the agent's drift velocity. If the agent is not currently doing a random
        walk, the drift velocity is towards the target. Otherwise, the drift velocity is
        None.

        Args:
        - pos (1D np.ndarray, optional): The position from which the agent should be
            drifting toward the current target, if the agent is not doing a random
            walk. If None, the agent's current position is used. Default is None.

        Returns:
        - drift_velocity (1D np.ndarray or None): The drift velocity.
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

    def anticipate_position_update(
        self,
        velocity: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        speed_fact: int | float = 3,
        drift_to_random_strength_ratio: float = 0.7,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        self.anticipate_position_update()

        Anticipate the agent's next position update.

        Args:
        - velocity (1D np.ndarray, optional): The velocity to use for the position
            update. If None, the agent's current velocity is used. Default is None.
        - speed_fact (int or float, optional): Factor to multiply the speed by for
            computing the  drift velocity. Default is 3.
        - drift_to_random_strength_ratio (float, optional): The ratio of drift to
            random strength. Default is 0.7.

        Returns:
        - update_vector (1D np.ndarray): The position update vector.
        """

        if velocity is None:
            velocity = self.velocity

        drift_velocity = self.get_drift_velocity(speed_fact=speed_fact)

        update_vector = ext_util.get_velocity_update_vector(
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
        """
        self.sample_teleport_out_position()

        Sample a position within the tolerance of the teleportation out coordinates.

        Args:
        - teleport_pair_num (int): The teleport pair to sample for. Default is 0.
        - max_attempts (int): The maximum number of attempts for sampling.
            Default is 100.

        Returns:
        - out_coords (1D np.ndarray): The sampled position.
        """

        teleport_coords = self.Environment.get_teleport_coords(
            teleport_pair_num, direction="out"
        )

        teleport_vector = self.get_teleport_vector(teleport_pair_num, direction="out")

        tol_prop_to_speed_dt = self.target_reached_within_tol_prop_to_speed_dt  # type: ignore[attr-defined]

        i = 0
        out_coords = None
        while out_coords is None:
            sampled_out_coords = self.sample_position_within_tolerance(
                teleport_coords, tol_prop_to_speed_dt
            )
            for x, y in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                coords_diff = sampled_out_coords - teleport_coords
                check_coords = teleport_coords + coords_diff * np.asarray([x, y])
                # position coordinates on the correct side of the teleport
                in_range = self.check_if_teleport_angles_in_range(
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

    def get_matched_teleport_out_position(
        self, teleport_pair_num: int = 0
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        self.get_matched_teleport_out_position()

        Obtain a teleport out position based on the teleport in position and the
        agent's current position. The angle and distance out are matched to the angle
        and distance in.

        Args:
        - teleport_pair_num (int, optional): The teleport pair to use. Default is 0.

        Returns:
        - out_coords (1D np.ndarray): The teleport out position.
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
        out_vector = gen_util.rotate_to(
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

    def get_teleport_rotated_velocity(
        self, teleport_pair_num: int = 0
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        self.get_teleport_rotated_velocity()

        Obtain the agent's current velocity, rotated to match the difference in angle
        between the in and out ports.

        Args:
        - teleport_pair_num (int, optional): The teleport pair to use. Default is 0.

        Returns:
        - out_velocity (1D np.ndarray): The output velocity.
        """

        teleport_in_vector = self.get_teleport_vector(teleport_pair_num, direction="in")
        teleport_out_vector = self.get_teleport_vector(
            teleport_pair_num, direction="out"
        )

        out_velocity = -gen_util.rotate_to(
            in_vector=self.velocity,
            in_basis=teleport_in_vector,  # type: ignore[arg-type]
            out_basis=teleport_out_vector,  # type: ignore[arg-type]
        )

        return out_velocity

    def get_teleport_coords_if_applicable(
        self,
        sample: bool = False,
        speed_fact: int | float = 3,
        drift_to_random_strength_ratio: float = 0.7,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """
        self.get_teleport_coords_if_applicable()

        Check whether agent should teleport. If so, obtains agent teleportation out
        coordinates and adds teleportation data to the teleportation dataframe.

        Attributes:
        - pos (1D np.ndarray): The agent's teleported position.

        Args:
        - sample (bool, optional): Whether to sample teleport out position instead of
            matching teleport in position. Default is False.
        - speed_fact (int or float, optional): The speed factor to use for the velocity
            drift. Default is 3.
        - drift_to_random_strength_ratio (float, optional): The ratio of drift to
            random strength. Default is 0.7.

        Returns:
        - out_coords (1D np.ndarray): The teleport out coordinates.
        """

        out_coords = None
        for teleport_pair_num in self.Environment.teleport_pairs_dict.keys():
            teleport = self.check_if_teleport_in_should_activate(teleport_pair_num)

            if not teleport:
                continue

            # teleport (sampling near out teleport coords)
            if sample:
                out_coords = self.sample_teleport_out_position(teleport_pair_num)
            else:
                out_coords = self.get_matched_teleport_out_position(teleport_pair_num)

            # rotate with respect to teleport in and out vectors
            out_velocity = self.get_teleport_rotated_velocity(teleport_pair_num)

            in_coords = self.pos.copy()
            in_velocity = self.velocity.copy()

            # simulate update for records (as if the teleportation was a normal step)
            update = self.anticipate_position_update(
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
                "time": self.t,
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

                add_data = [
                    ("position", coords),
                    ("vector", vector),
                    ("velocity", velocity),
                ]
                for key, data in add_data:
                    for i, axis in enumerate(["x", "y"]):
                        teleportation_data[f"{direction}_{key}_{axis}"] = data[i]

            self.teleportation_df.loc[len(self.teleportation_df)] = teleportation_data  # type: ignore[assignment]

            if self.always_log_teleportation:  # type: ignore[attr-defined]
                self.log_teleportation(last=True)

            break

        return out_coords

    def log_teleportation(self, last=False):
        """
        self.log_teleportation()

        Log the teleportation events.

        Args:
        - last (bool, optional): Whether to log only the last recorded teleportation
            event. Default is False.
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

    def reset(self, target: str | None = None):
        """
        self.reset()

        Reset the agent to a random location, end the current trajectory and updating
        all positions, including target.

        Attributes:
        - current_trajectory_length (int): The current trajectory length.
        - trajectory_lengths (list): List of trajectory lengths.

        Args:
        - target (str, optional): The target to use. Default is None.
        """

        self._end_trajectory()

        self.set_all_positions(first_setting=False, target=target)

        if self.trajectory_lengths is not None:
            i = (len(self.trajectory_df) - 1) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]

        self.current_trajectory_length = 0

        self._add_new_trajectory_to_df()

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
        """
        self.update()

        Update the agent's position. Checks whether target or trajectory end are
        reached, and if so, resets the agent. Also, checks whether a random walk
        should start or end.

        Args:
        - dt (float): The time step to use.
        - speed_fact (float): The speed factor.
        - drift_to_random_strength_ratio (float): The ratio of the drift strength to
            the random walk strength.
        - drift_velocity (float or 1D np.ndarray, optional): The drift velocity to use.

        Keyword args:
        - **kwargs: Keyword arguments passed to ratinabox.Agent.update().
        """

        target_reached = self.check_and_record_target_reached()

        if self.check_if_trajectory_end_reached():
            self.reset()
        elif target_reached:
            self._set_random_walk()
        elif self.current_num_of_random_walk_steps == 0:
            if self.target_position is None:
                self._add_new_target_to_df()
            self._set_random_walk()

        teleport_coords = self.get_teleport_coords_if_applicable(
            speed_fact=speed_fact,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
        )

        # calculate drift_velocity
        if teleport_coords is None and drift_velocity is None:
            drift_velocity = self.get_drift_velocity(
                pos=self.pos, speed_fact=speed_fact
            )

        # checks what happens to random walk
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

    def get_agent_color_for_trajectory(self, t: float | None = None) -> str:
        """
        self.get_agent_color_for_trajectory()

        Obtain the agent color for plotting based on what occurs during the trajectory
        specified by the specified time point.

        Colors:
        - lavender: Trajectory had no target or time corresponds exactly to the
            reach time.
        - red: Target not yet reached in the trajectory.
        - violet: Target was in a random walk.

        Args:
        - t (float, optional): The time to get the state color for. Default is None.

        Returns:
        - agent_color (str): The agent's state color.
        """

        endid = self.get_plotting_times(t_end=t)[-1]
        agent_color = "dodgerblue"
        past_df = self.target_df.loc[self.target_df["set_step"] <= endid]
        if len(past_df):
            row = past_df.loc[past_df.index[-1]]
            if row["object_type_name"] == "no_target":
                agent_color = "dodgerblue"
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

    def add_teleportation_markers_to_plots(
        self,
        ax: np.ndarray[Sequence[plt.Axes], np.dtype[np.object_]] | plt.Axes,
        t_start: float | None = None,
        t_end: float | None = None,
        timeseries: bool = False,
        plot_lines: bool = True,
        legend: bool = True,
    ):
        """
        self.add_teleportation_markers_to_plots()

        Adds teleportation markers to timeseries or environment plots.

        Args:
        - ax (np.ndarray or plt.Axes): Subplot or array of subplots on which to add
            teleportation markers.
        - t_start (float, optional): Start time of the plot. Default is None.
        - t_end (float, optional): End time. Default is None.
        - timeseries (bool, optional): Whether the plot is timeseries. Default is False.
        - plot_lines (bool, optional): Whether to plot lines for teleportation events,
            if timeseries is True. Default is True.
        - legend (bool, optional): Whether to add a legend. Default is True.
        """

        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)

        step_nums = self.teleportation_df["step_num"].to_numpy() - startid
        object_in_type_nums = self.teleportation_df["in_object_type_num"].to_numpy()

        ax1D = np.asarray(ax).ravel()

        for sub_ax in ax1D:
            plotted = list()
            if timeseries:
                y_min, y_max = sub_ax.get_ylim()
                y_pos = (y_max - y_min) * 0.98 + y_min

            for i in np.argsort(object_in_type_nums):
                step_num = step_nums[i]
                obj_num = object_in_type_nums[i]

                if step_num >= len(t):
                    continue
                elif step_num < 0:
                    alpha = 0.6
                else:
                    alpha = 1.0

                plot_params = copy.deepcopy(
                    self.Environment.object_type_num_to_plot_params_dict[obj_num]
                )
                name = plot_params.pop("name")
                if legend and name not in plotted:
                    label = name.replace("_", " ").replace(" in", "")
                    plotted.append(name)
                else:
                    label = None

                if timeseries:
                    if step_num < 0:
                        continue
                    x_pos = t[step_num] / 60
                    pos = [x_pos, y_pos]
                    plot_params["s"] /= 2
                    if plot_lines:
                        sub_ax.axvline(
                            x=x_pos,
                            color=plot_params["color"],
                            ls="dashed",
                            zorder=-1,
                            alpha=0.8,
                        )
                        sub_ax.scatter(
                            *pos,
                            alpha=0.9,
                            color="white",
                            s=plot_params["s"] * 6,
                            zorder=0,
                            marker="s",
                        )
                else:
                    pos = self.history["pos"][step_num + startid]

                sub_ax.scatter(*pos, alpha=alpha, label=label, **plot_params)

            if legend and len(plotted):
                sub_ax.legend()

    def add_target_to_plot(
        self,
        sub_ax: plt.Axes,
        t: float | None = None,
    ):
        """
        self.add_target_to_plot()

        Add the target for time t to the plot.

        Args:
        - sub_ax (plt.Axes): Subplot to add target to.
        - t (float, optional): The current time step. Default is None.
        """

        all_t = np.asarray(self.history["t"])
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
            marker=mpl_markers.MarkerStyle("x"),
            s=60,
            zorder=4,
            color="red",
            label="target",
            lw=3,
            alpha=0.6,
        )

    def plot_trajectories(  # type: ignore[override]
        self,
        t_end: float | None = None,
        target_alpha: float = 0.7,
        no_legend: bool = False,
        autosave: bool | None = None,
        **kwargs,
    ) -> plt.Axes:
        """
        self.plot_trajectories()

        Plot the agent's trajectories.

        Args:
        - t_end (float, optional): Time point to plot the trajectory until.
            Default is None.
        - target_alpha (float, optional): Alpha value for the target.
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.
        - **kwargs: Additional keyword arguments.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectory plotted.
        """

        sub_ax = super().plot_trajectories(
            t_end=t_end,
            target_alpha=target_alpha,
            plot_target=False,
            agent_color=self.get_agent_color_for_trajectory(t=t_end),
            autosave=False,
            **kwargs,
        )

        legend = sub_ax.get_legend()
        if no_legend and legend is not None:
            legend.remove()

        self.add_target_to_plot(sub_ax, t=t_end)

        fig = sub_ax.figure
        plot_util.save_figure(fig, "trajectory", save=autosave)

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
        """
        self.plot_trajectory_targets()

        Plot the trajectory targets.

        Args:
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None,
            a new subplot is created using self.Environment.plot_environment().
            This can be used to plot trajectory on top of receptive fields etc.
        - alpha (float, optional): Alpha value of the targets.
        - plot_env (bool, optional): Whether to plot the environment, if sub_ax is not
            None. Default is True.
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - colormap (str, optional): Colormap to use. Default is None.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

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
            shift_x = 1 * 0.006 * env_width
            shift_y = 1 * 0.006 * env_height

            t = sub_ax.text(
                object_df.loc[object_df.index[0], "position_x"] + shift_x,
                object_df.loc[object_df.index[0], "position_y"] + shift_y,
                str(count),
                horizontalalignment="left",
                verticalalignment="bottom",
                color="black",
                fontsize=10,
                zorder=2,
                fontweight="bold",
            )

            t.set_bbox(dict(facecolor="white", alpha=0.8, lw=0))

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
        plot_util.save_figure(fig, "trajectory_targets", save=autosave)

        return sub_ax

    def plot_trajectory_target_coords_over_time(
        self,
        t_start: float | None = None,
        t_end: float | None = None,
        sub_ax: plt.Axes | None = None,
        no_legend: bool = False,
        autosave: bool | None = None,
    ) -> plt.Axes:
        """
        self.plot_trajectory_target_coords_over_time()

        Plot the x and y coordinates of the trajectory targets over time.

        Args:
        - t_start (float, optional): Start time of the plot.
        - t_end (float, optional): End time of the plot.
        - sub_ax (plt.Axes, optional): Subplot to plot on. If None, a new subplot is
            created. Default is None.
        - no_legend (bool, optional): Whether to remove the legend. Default is False.
        - autosave (bool, optional): Whether to autosave the figure. If None, the
        global autosave setting for ratinabox is used. Default is None.

        Returns:
        - sub_ax (plt.Axes): Subplot with trajectory targets plotted over time.
        """

        if sub_ax is None:
            _, sub_ax = plt.subplots(figsize=(8, 3))

        t, startid, endid = self.get_plotting_times(t_start=t_start, t_end=t_end)

        pos = np.asarray(self.history["pos"])
        t_start, t_end = t[0], t[-1]

        if t_start is None or t_end is None:
            raise RuntimeError("t_start or t_end is None.")

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

        sub_ax.spines[["right", "top"]].set_visible(False)
        sub_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        # plot teleportation points as vertical dashed lines
        for idx in self.teleportation_df.index:
            step = int(self.teleportation_df.loc[idx, "step_num"])  # type: ignore[assignment]
            if step < startid or step > endid:
                continue

            object_type = self.teleportation_df.loc[idx, "in_object_type_num"]
            color = self.Environment.object_type_num_to_plot_params_dict[object_type][
                "color"
            ]
            sub_ax.axvline(t[step], color=color, ls="dashed", alpha=0.8, zorder=-1)

        # plot target objects
        object_type_num_to_plot_params_dict = copy.deepcopy(
            self.Environment.object_type_num_to_plot_params_dict
        )

        r = 0
        reset_times = np.append(reset_times, t[-1])
        for idx in self.target_df.index:
            row = self.target_df.loc[idx]
            if row["object_type_name"] == "no_target":
                continue

            plot_params = object_type_num_to_plot_params_dict[row["object_type_num"]]
            label = None
            if "name" in plot_params:
                label = plot_params.pop("name")
                plot_params["markersize"] = plot_params.pop("s") / 8

            if np.isnan(row["reached_step"]) or row["reached_step"] >= endid:
                if r >= len(reset_times):
                    continue
                target_reached_time = reset_times[r]
                alpha = 0.3
                r += 1
            else:
                target_reached_idx = int(row["reached_step"]) - startid
                if target_reached_idx < 0 or target_reached_idx >= len(t):
                    continue

                target_reached_time = t[target_reached_idx]
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
        plot_util.save_figure(fig, "trajectory_targets_over_time", save=autosave)

        return sub_ax

    def animate_trajectories(
        self,
        plot_head_direction=True,
        additional_plot_func: Callable | None = None,
        **kwargs,
    ) -> mpl_animation.FuncAnimation:
        """
        self.animate_trajectories()

        Animate the trajectory of the agent.

        Args:
        - plot_head_direction (bool, optional): Whether to plot the last head direction.
            Default is True.
        - additional_plot_func: A function that is called after each frame of the
            animation is plotted. It takes sub_ax, t and **kwargs and returns sub_ax.

        Keyword args:
        - **kwargs: Keyword arguments passed to self.plot_trajectories().

        Returns:
            anim (matplotlib.animation.FuncAnimation): The animation object.
        """

        def run_all_additional_plot_funcs(
            fig: mpl_figure.Figure,
            ax: plt.Axes,
            t: float | None = None,
            **kwargs,
        ) -> plt.Axes:
            """
            self.run_all_additional_plot_funcs()

            Run through all additional plot functions.

            Args:
            - fig (mpl_figure.Figure): Figure to plot on (for backward compatibility).
            - ax (plt.Axes): Subplot to plot on (called 'ax' for backward compatibility).
            - t (float, optional): The current time step. Defaut is None.

            Returns:
            - sub_ax (plt.Axes): Subplot with additional plot elements added.
            """

            sub_ax = ax
            if additional_plot_func is not None:
                sub_ax = additional_plot_func(sub_ax=sub_ax, t=t, **kwargs)

            # self.add_target_to_plot(sub_ax, t=t)
            plot_util.remove_prev_handle_labels(sub_ax)

            return fig, sub_ax

        # call ratinabox.Agent method
        anim = self.animate_trajectory(
            additional_plot_func=run_all_additional_plot_funcs,
            plot_head_direction=plot_head_direction,
            return_traj_fig=True,
            return_env_fig=True,
            **kwargs,
        )

        return anim
