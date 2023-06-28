
import warnings

import copy
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import seaborn as sns
from ratinabox import Agent
from ratinabox import utils as rutils

from predhpc import env, util, plot_util


class ResetAgent(Agent):
    """Extend the agent so that is has an optimal maximum trajectory length after which is resets to a random location
    
    default_params = {
        "trajectory_length": None, # int or iterable of ints
        "n_traj": None, # number of trajectory lengths to sample
        "exp": None, # exponential factors for trajectory_length (inv. scale, rate, minimum). Defaults to None.
        "rand": None, # max value for randomizing trajectory_length
        "reset_pos": (None, None), # position to reset (from, to)
    }
    """

    default_params = {
        "dt": 0.01, # time step, in seconds
        "trajectory_length": None, # int or iterable of ints
        "n_traj": None, # number of trajectory lengths to sample
        "exp": None, # exponential factors for trajectory_length (inv. scale, rate, minimum). Defaults to None.
        "rand": None, # max value for randomizing trajectory_length
        "start_pos": None, # position to start trajectories from
        "reset_pos": None, # position to reset trajectories from
        "target_pos": None, # position to use as target
        "target_wait": 100, # number of steps to wait between target reaching
        "reset_tolerance_prop": 0.5, # proportion of dt to use as reset tolerance
        "target_tolerance_prop": 0.5, # proportion of dt to use as target tolerance
        "fixed_direction": False, # keep same direction (1D environment only)
    }

    def __init__(self, Env, params={}):
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
            if self.speed_mean < 0:
                raise ValueError(
                    "Cannot have fixed direction with negative speed."
                    )
            if self.Environment.boundary_conditions == "periodic":
                raise NotImplementedError(
                    "Fixed direction not implemented for periodic boundary conditions."
                    )

        self.act_trajectory_lengths = []
        self.set_all_pos()
        self.set_trajectory_lengths()


    def set_all_pos(self):
        """Set all positions, checking that they are within the environment
        extent.
        """

        self.start_pos = self.set_pos(self.start_pos)
        self.reset_pos = self.set_pos(self.reset_pos)
        self.target_pos = self.set_pos(self.target_pos)

        self.reached_reset_pos = []
        self.reached_target_pos = []

        if self.start_pos is not None:
            self.set_pos_vel(pos=self.start_pos, velocity=0)
        self.target_waiting = 0


    def set_trajectory_lengths(self):
        """Set the trajectory lengths, either from the passed value, or by
        sampling from the exponential distribution.
        """

        if self.trajectory_length is not None:
            self.n_traj = None
            self.exp = None
            self.rand = None

        elif self.n_traj:
            self.trajectory_length = util.get_trajectory_lengths(
                n_traj=self.n_traj, exp=self.exp, rand=self.rand
                )
                
        self.trajectory_lengths = None
        self.curr_trajectory_length = 0
        if self.trajectory_length is not None:
            if not isinstance(self.trajectory_length, int):
                self.trajectory_length = np.maximum(self.trajectory_length, 1)
                if len(self.trajectory_length) == 0:
                    raise ValueError("If passing iterable, must have length > 0.")
                self.trajectory_lengths = self.trajectory_length
                self.trajectory_length = self.trajectory_lengths[0]
        
        self.num_steps_total = 0


    def set_pos(self, pos):
        """Check that start_pos and reset_pos are of the correct length, and 
        reshape if needed.
        """

        if pos is not None:
            pos = np.asarray(pos).reshape(-1)
            if len(pos) != self.Environment.D:
                raise ValueError(
                    f"Positions must comprise exactly {self.Environment.D} value(s)."
                    )
            pos = pos.reshape(self.Environment.D)

             # [min, max] or [left, right, bottom, top]
            extent = self.Environment.extent

            if self.Environment.D == 1:
                if pos < extent[0] or pos > extent[1]:
                    raise ValueError(
                        "Position must be within the environment extent: "
                        f"{extent}."
                        )
            elif self.Environment.D == 2:
                if pos[0] < extent[0] or pos[0] > extent[1]:
                    raise ValueError(
                        "First position dimension must be within the "
                        f"environment extent: {extent[:2]}."
                        )
                if pos[1] < extent[2] or pos[1] > extent[3]:
                    raise ValueError(
                        "Second position dimension must be within the "
                        f"environment extent: {extent[2:]}."
                        )
            
            else:
                raise ValueError(
                    "Expected environment dimensionality to be 1 or 2. "
                    f"Got {self.Environment.D}."
                    )
        
        return pos


    def fix_velocity_record(self, dt=None):
        """Fix values computed and recorded based on the velocity, if applicable.
        """

        if not hasattr(self, "prev_average_measured_speed"):
            return

        if dt is None:
            dt = self.dt

        tau_speed = 10
        self.average_measured_speed = (
            self.prev_average_measured_speed + dt / tau_speed * (
            np.linalg.norm(self.velocity, ord=2)
        )
        )

        self.save_velocity = self.velocity

        if self.save_history is True and len(self.history["vel"]):
            self.history["vel"][-1] = list(self.save_velocity)
            if self.Environment.dimensionality == "2D":
                self.history["rot_vel"][-1] = self.rotational_velocity


    def check_fix_velocity(self, prev_velocity, dt=None):
        """Check if velocity is negative and fix if applicable.
        """

        if not (self.Environment.dimensionality == "1D" and self.fixed_direction):
            return

        if self.velocity >= 0:
            return
        
        if self.reset_pos is not None and self.pos[0] > self.reset_pos[0]:
            return

        if dt is None:
            dt = self.dt

        new_velocity = self.velocity
        for _ in range(10):
            if new_velocity < 0: # resample velocity until it is positive
                new_velocity = prev_velocity + rutils.ornstein_uhlenbeck(
                    dt=dt,
                    x=prev_velocity,
                    drift=self.speed_mean,
                    noise_scale=self.speed_std,
                    coherence_time=self.speed_coherence_time,
                )
            else:
                break
        
        if new_velocity < 0:
            new_velocity = prev_velocity * 0 # set to 0

        self.velocity = new_velocity
        self.fix_velocity_record(dt=dt)


    def set_pos_vel(self, pos=None, velocity=None):
        """Set the position and velocity of the agent.
        
        From Agent.__init__() in ratinabox/agent.py
        """

        # initialise starting positions and velocity

        if pos is not None:
            self.pos = pos

        if self.Environment.dimensionality == "2D":
            if pos is None:
                self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            direction = np.random.uniform(0, 2 * np.pi)
            if velocity is None:
                velocity = self.speed_std              
            self.velocity = velocity * np.array(
                [np.cos(direction), np.sin(direction)]
            )
            self.rotational_velocity = 0

        if self.Environment.dimensionality == "1D":
            if pos is None:
                self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            if velocity is None:
                self.velocity = np.array([self.speed_mean]).reshape(1)
            else:
                self.velocity = np.array([velocity]).reshape(1)
            if self.Environment.boundary_conditions == "solid":
                if self.speed_mean != 0:
                    warnings.warn(
                        "solid 1D boundary conditions and non-zero speed mean."
                    )
    
        self.fix_velocity_record()

        return


    def reset(self):
        """Reset the agent to a random location.
        """

        self.set_pos_vel(pos=self.start_pos, velocity=0)
        
        self.act_trajectory_lengths.append(self.curr_trajectory_length)
        if self.trajectory_lengths is not None:
            i = len(self.act_trajectory_lengths) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]
       
        self.curr_trajectory_length = 0

        return
    
    
    def get_trajectory_lengths_to_date(self):
        """Return the trajectory lengths to date.

        Returns:
            list: Trajectory lengths to date.
        """
        traj_leng_to_date = self.act_trajectory_lengths
        if self.curr_trajectory_length > 0:
            traj_leng_to_date = self.act_trajectory_lengths + [self.curr_trajectory_length]
        return traj_leng_to_date


    def log_trajectories_to_date(self):
        """Log the trajectory lengths to date.        
        """
        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        print(f"Trajectory lengths ({len(traj_leng_to_date)}) to date (in steps): {traj_leng_to_date}")


    def log_trajectory_stats_to_date(self, time=True):
        """Log the trajectory length statistics to date.        
        """

        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        traj_length_unit = "steps"

        # get trajectory lengths in seconds
        if time:
            traj_leng_to_date = [leng * self.dt for leng in traj_leng_to_date]
            traj_length_unit = "sec"
            if np.mean(traj_leng_to_date) / 60 > 2:
                traj_leng_to_date = [leng / 60 for leng in traj_leng_to_date]
                traj_length_unit = "min"    

        # get trajectory length statistics
        traj_leng_to_date_mean = np.mean(traj_leng_to_date)
        traj_leng_to_date_std = np.std(traj_leng_to_date)

        print(f"Trajectory lengths ({len(traj_leng_to_date)}) to date: {traj_leng_to_date_mean:.2f} +/- {traj_leng_to_date_std:.2f} {traj_length_unit} each")


    def plot_trajectories_to_date(self, in_min=True):
        """Plot the trajectory lengths to date.
        
        Args:
            in_min (bool, optional): Whether to plot in minutes. Defaults to True.
        """

        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        plot_util.plot_trajectory_lengths(dt=self.dt, trajectory_lengths=traj_leng_to_date, in_min=in_min)

   
    def get_reset_times(self):
        """Get the reset times. 

        Returns:
            list: Reset times.

        Raises:
            ValueError: If agent does not have reset steps.
        """

        if hasattr(self, "act_trajectory_lengths"):
            if len(self.act_trajectory_lengths) == 0:
                reset_times = np.array([])
            else:
                reset_times = np.cumsum(self.act_trajectory_lengths) * self.dt
        else:
            raise ValueError("Agent does not have reset steps.")
        
        return reset_times
   

    def check_pos(self, target_pos=None, tolerance_prop=0.5):
        """Check if the agent has reached the target position.

        Args:
            target_pos (np.array): Target position.
            tolerance_prop (float): Tolerance proportion, wrt self.dt.
        
        Returns:
            bool: Whether the agent has reached the target position.
        """

        if target_pos is not None:
            # calculate the distance between the current position and the reset position
            dist = np.linalg.norm(self.pos - target_pos, ord=2)

            # check if the distance is less than the tolerance
            if dist < (self.dt * tolerance_prop):
                return True
    
        return False


    def check_reset_pos(self):
        """Check if the agent has reached the reset position.
        
        Returns: Whether the agent has reached the reset position.
        """

        return self.check_pos(self.reset_pos, self.reset_tolerance_prop)


    def check_target_pos(self):
        """Check if the agent has reached the target position.
        
        Returns: Whether the agent has reached the target position.
        """

        if self.target_pos is None:
            return 
        
        if self.target_waiting > 0:
            self.target_waiting -= 1
            return False

        else:
            target_reached = self.check_pos(self.target_pos, self.target_tolerance_prop)
            if target_reached:
                self.target_waiting = self.target_wait
            return target_reached


    def _check_end(self):
        """Check if the agent has reached the end of its trajectory.

        Returns:
            bool: Whether the agent has reached the end of its trajectory.
        """

        self.reached_end = False
        if self.reset_pos is not None and self.check_reset_pos():
            # record the time step at which the agent reached the reset position
            self.reached_end = True
            if len(self.reached_reset_pos):
                if self.num_steps_total == self.reached_reset_pos[-1]:
                    self.reached_end = False
            
            if self.reached_end:
                self.reached_reset_pos.append(self.num_steps_total)

        if self.trajectory_length is not None:
            if self.curr_trajectory_length >= self.trajectory_length:
                self.reached_end = True

        return self.reached_end


    def _check_target(self):
        """Check if the agent has reached the target in its trajectory.

        Returns:
            bool: Whether the agent has reached the target in its trajectory.
        """

        self.reached_target = False
        if self.target_pos is not None and self.check_target_pos():
            # record the time step at which the agent reached the target position
            self.reached_target = True
            if len(self.reached_target_pos):
                if self.num_steps_total == self.reached_target_pos[-1]:
                    self.reached_target = False

            if self.reached_target:
                self.reached_target_pos.append(self.num_steps_total)

        return self.reached_target


    def update(self, dt=None, skip_checks=False, **kwargs):
        """Update the agent, optionally with a new position and velocity.
        
        See Agent.update() in ratinabox/agent.py for kwargs.
        """

        if not skip_checks:
            self._check_target()
            if self._check_end():
                self.reset()
        
        self.prev_average_measured_speed = self.average_measured_speed
        prev_velocity = self.velocity
        
        super().update(dt=dt, **kwargs)

        if self.Environment.dimensionality == "1D" and self.fixed_direction:
            self.check_fix_velocity(prev_velocity=prev_velocity, dt=dt)

        self.curr_trajectory_length += 1
        self.num_steps_total += 1


    def plot_trajectory_resets(
        self,
        t_start=0,
        t_end=None,
        framerate=10,
        fig=None,
        ax=None,
        alpha=0.6,
        color="k",
        ms=50,
        plot_targets=True,
    ):

        """Plots the trajectory between t_start (seconds) and t_end (defaulting to the last time available)

        From Agent.plot_1D_trajectories() in ratinabox/agent.py. Modified to enable plotting of reset steps, and use of colormaps for trajectories.

        Args:
            • t_start: start time in seconds
            • t_end: end time in seconds (default = self.history["t"][-1])
            • framerate: how many scatter points / per second of motion to display
            • fig, ax: the fig, ax to plot on top of
            • alpha: plot point opaqness
            • xlim: In 1D, forces the xlim to be a certain time (minutes) (useful if animating this function)
        
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

        t = t / 60 # minutes
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

        n_reached = len(self.reached_reset_pos)
        n_targets = len(self.reached_target_pos)
        alpha /= self.Environment.D
        alpha_pts = 0.9 / self.Environment.D
        for i in range(self.Environment.D):
            ax.scatter(time, pos, alpha=alpha, marker=".", color=color, s=ms/10)
            
            if n_reached:
                if self.start_pos is not None:
                    x_start = [t[x] for x in self.reached_reset_pos if x >= startid and x < endid]
                    y_start = [self.start_pos[i]] * n_reached
                    ax.scatter(
                        x_start, y_start, marker=".", color="blue", alpha=alpha_pts, s=ms
                        )
                
                if self.reset_pos is not None:
                    x_reset = [t[x - 1] for x in self.reached_reset_pos if x >= startid and x < endid]
                    y_reset = [self.reset_pos[i]] * n_reached
                    ax.scatter(
                        x_reset, y_reset, marker="x", color="red", alpha=alpha_pts, s=ms/3
                        )
            
            if plot_targets and n_targets and self.target_pos is not None:
                x_targ = [t[x] for x in self.reached_target_pos if x >= startid and x < endid]
                y_targ = [self.target_pos[i]] * n_targets
                ax.scatter(
                    x_targ, y_targ, marker="d", color="gold", alpha=alpha_pts, s=ms/5
                    )

        ax.set_xlabel("Time / min")
        ax.set_ylabel("Position / m")
            
        bottom = min_y - diff * 0.1
        top = max_y + diff * 0.1
        ax.set_ylim(bottom=bottom, top=top)
        ax.spines["right"].set_color(None)
        ax.spines["top"].set_color(None)

        return fig, ax


    def plot_trajectory(
        self,
        t_start=0,
        t_end=None,
        framerate=10,
        fig=None,
        ax=None,
        decay_point_size=False,
        plot_agent=True,
        colormap=None,
        alpha=0.7,
        xlim=None,
        background_color=None,
        plot_traj_ends=True,
        target_alpha=1.0,
        cmap_per=False,
        scale_cmap_per=False,
        ms_2D=15,
        size_fact=None,
    ):

        """Plots the trajectory between t_start (seconds) and t_end (defaulting to the last time available)

        From Agent.plot_trajectory() in ratinabox/agent.py. Modified to enable plotting of reset steps, and use of colormaps for trajectories.

        Args:
            • t_start: start time in seconds
            • t_end: end time in seconds (default = self.history["t"][-1])
            • framerate: how many scatter points / per second of motion to display
            • fig, ax: the fig, ax to plot on top of, optional, if not provided used self.Environment.plot_Environment().
              This can be used to plot trajectory on top of receptive fields etc.
            • decay_point_size: decay trajectory point size over time (recent times = largest)
            • plot_agent: dedicated point show agent current position
            • colormap: colormap to use to plot trajectories
            • alpha: plot point opaqness
            • xlim: In 1D, forces the xlim to be a certain time (minutes) (useful if animating this function)
            • background_color: color of the background if not matplotlib default, only for 1D (probably white)
            • plot_traj_ends: plot a point at the end of each trajectory
            • target_alpha: transparency with which to plot target position
            • cmap_per: if True, the colormap is used to set the color for each time point. Otherwise, each trajectory has its own color.
            • scale_cmap_per: if True, and cmap_per is True, the full range of the colormap is used for each trajectory, regardless of its length
            • ms_2D: the size of the points in the 2D plot is set to this value.
        
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
        if self.Environment.dimensionality == "2D":
            trajectory = pos[startid:endid, :][::skiprate]
        if self.Environment.dimensionality == "1D":
            trajectory = pos[startid:endid][::skiprate]
        time = t[startid:endid][::skiprate]

        # get reset step indices
        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")
        elif len(time) == 0:
            raise RuntimeError("Duration too short. No time points to plot.")

        last_length = len(t) - sum(self.act_trajectory_lengths)
        trajectory_lengths = self.act_trajectory_lengths + [last_length]
        traj_idx = [np.full(steps, i) for i, steps in enumerate(trajectory_lengths)]
        if cmap_per:
            if scale_cmap_per:
                cmap_vals = [np.linspace(0, 1, steps) for steps in trajectory_lengths]
            else:
                cmap_vals = [np.arange(steps) for steps in trajectory_lengths]
        else:
            cmap_vals = traj_idx[:]
        cmap_vals = np.concatenate(cmap_vals).astype(float)
        cmap_vals = cmap_vals[startid : endid][::skiprate]
        cmap_min, cmap_max = cmap_vals.min(), cmap_vals.max()
        if cmap_min == cmap_max:
            cmap_vals[:] = 0.5 # mid point of the colormap
        else:
            cmap_vals = (cmap_vals - cmap_min) / (cmap_max - cmap_min)
        
        traj_idx = np.concatenate(traj_idx).astype(int)[startid : endid][::skiprate]
        
        if colormap is None:
            colormap = "crest"
        c = sns.color_palette(colormap, as_cmap=True)(cmap_vals)
        ##############################

        if self.Environment.dimensionality == "2D":
            if size_fact is not None:
                extent = self.Environment.extent
                x_base = extent[1] - extent[0]
                y_base = extent[3] - extent[2]
                figsize = (size_fact * x_base, size_fact * y_base)
                fig, ax = plt.subplots(figsize=figsize)

            fig, ax = self.Environment.plot_environment(fig=fig, ax=ax)
            if self.target_pos is not None:
                ax.scatter(*self.target_pos, marker="d", color="gold", s=20, zorder=5, edgecolors="darkgoldenrod", linewidth=0.5, alpha=target_alpha)

            s = ms_2D * np.ones_like(time)
            if decay_point_size == True:
                s = ms_2D * np.exp((time - time[-1]) / 10)
                s[(time[-1] - time) > ms_2D] *= 0

            if plot_traj_ends == True and len(self.act_trajectory_lengths):
                ends = np.where(np.diff(traj_idx) > 0)[0]
                ends = np.append(ends, len(trajectory) - 1)
                s[ends] = ms_2D * 2
                c[ends] = mcolors.to_rgba("darkred") ### set last colormap value to black

            if plot_agent == True:
                s[-1] = ms_2D * 2.75
                c[-1] = mcolors.to_rgba("r") ### set last colormap value to red

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

            if fig is None and ax is None:
                fig, ax = plt.subplots(figsize=(3, 1.5))
            ax.scatter(time / 60, trajectory, alpha=alpha, linewidth=0, c=c, s=5)
            ax.spines["left"].set_position(("data", t_start / 60))
            ax.set_xlabel("Time / min")
            ax.set_ylabel("Position / m")
            ax.set_xlim([t_start / 60, t_end / 60])
            if xlim is not None:
                ax.set_xlim(right=xlim)

            ax.set_ylim(bottom=0, top=self.Environment.extent[1])
            ax.spines["right"].set_color(None)
            ax.spines["top"].set_color(None)
            ax.set_xticks([t_start / 60, t_end / 60])
            ex = self.Environment.extent
            ax.set_yticks([ex[1]])
            if background_color is not None:
                ax.set_facecolor(background_color)
                fig.patch.set_facecolor(background_color)

        return fig, ax
    

    def plot_trajectory_edges(
        self,
        t_start=0,
        t_end=None,
        fig=None,
        ax=None,
        decay_point_size=False,
        plot_agent=True,
        colormap=None,
        alpha=0.7,
        xlim=None,
        background_color=None,
        plot_starts=True,
        plot_ends=True,
    ):

        """Plots the trajectory starts and ends between t_start (seconds) and t_end (defaulting to the last time available)

        Args:
            • t_start: start time in seconds
            • t_end: end time in seconds (default = self.history["t"][-1])
            • fig, ax: the fig, ax to plot on top of, optional, if not provided used self.Environment.plot_Environment().
              This can be used to plot trajectory ends on top of receptive fields etc.
            • decay_point_size: decay trajectory point size over time (recent times = largest)
            • plot_agent: dedicated point show agent current position
            • colormap: colormap to use to plot trajectories starts/ends
            • alpha: plot point opaqness
            • xlim: In 1D, forces the xlim to be a certain time (minutes) (useful if animating this function)
            • background_color: color of the background if not matplotlib default, only for 1D (probably white)
            • plot_starts: plot trajectory starts
            • plot_ends: plot trajectory ends
        
        Returns:
            fig, ax
        """        

        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        if t_end == None:
            t_end = t[-1]
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end))) + 1

        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")

        if colormap is None:
            colormap = "crest"
        cmap = sns.color_palette(colormap, as_cmap=True)

        all_ends = np.cumsum(self.act_trajectory_lengths)
        start_c, end_c = None, None
        if plot_starts:
            traj_starts = np.insert(all_ends, 0, 0)
            start_c = cmap(np.linspace(0, 1, len(traj_starts)))
        if plot_ends:
            traj_ends = np.append(all_ends - 1, len(t) - 1)
            end_c = cmap(np.linspace(0, 1, len(traj_ends)))
        if not (plot_starts or plot_ends):
            raise ValueError("At least one of 'plot_starts' or 'plot_ends' must be True.")

        for c, traj_idx, marker in [(start_c, traj_starts, "x"), (end_c, traj_ends, "o")]:
            if c is None:
                continue
            lw = 2 if marker == "x" else 0
            traj_idx = traj_idx[(traj_idx >= startid) & (traj_idx <= endid)]
            trajectory = pos[traj_idx]
            time = t[traj_idx]
    
            if len(time) == 0:
                raise RuntimeError("Duration too short. No trajectory points to plot.")

            if self.Environment.dimensionality == "2D":
                fig, ax = self.Environment.plot_environment(fig=fig, ax=ax)
                ax.scatter(*self.target_pos, marker="d", color="gold", s=20, zorder=5, edgecolors="black", linewidth=0.5)

                s = 15 * np.ones_like(time)
                if decay_point_size == True:
                    s = 15 * np.exp((time - time[-1]) / 10)
                    s[(time[-1] - time) > 15] *= 0

                if plot_agent == True:
                    s[-1] = 40
                    c[-1] = mcolors.to_rgba("r") ### set last colormap value to red

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
            if self.Environment.dimensionality == "1D":

                if fig is None and ax is None:
                    fig, ax = plt.subplots(figsize=(3, 1.5))
                ax.scatter(time / 60, trajectory, alpha=alpha, linewidth=lw, c=c, s=5, marker=marker)
                ax.spines["left"].set_position(("data", t_start / 60))
                ax.set_xlabel("Time / min")
                ax.set_ylabel("Position / m")
                ax.set_xlim([t_start / 60, t_end / 60])
                if xlim is not None:
                    ax.set_xlim(right=xlim)

                ax.set_ylim(bottom=0, top=self.Environment.extent[1])
                ax.spines["right"].set_color(None)
                ax.spines["top"].set_color(None)
                ax.set_xticks([t_start / 60, t_end / 60])
                ex = self.Environment.extent
                ax.set_yticks([ex[1]])
                if background_color is not None:
                    ax.set_facecolor(background_color)
                    fig.patch.set_facecolor(background_color)

        return fig, ax



class TAgent(ResetAgent, util.ParamsMixin):
    """Extend the reset agent so that it operates in a T maze    
    """

    default_params = {
        "target_arm": "left",
        "target_prop": 0.75, # proportion down arm at which to set target
        "left_arm_prop": 0.75, # proportion of trajectories to target to left arm
    }

    ignored_param_keys = ["reset_pos", "start_pos", "target_pos"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Env, params={}):
        """Initialise the agent.

        Args:
            params (dict, optional): Parameters for the agent. Defaults to {}.

        Raises:
            ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.check_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)     
        self.params.update(params)

        if not isinstance(Env, env.TEnv):
            raise TypeError("Env must be a TEnv object.")

        self.set_fixed_params()

        super().__init__(Env, self.params)

        self.set_current_arm()


    @property
    def near_branch(self):

        return self.pos[1] > (self.Environment.branch_y * 0.98)


    @property
    def at_branch(self):

        return self.pos[1] > self.Environment.branch_y
    
    def get_direction(self):
        """Get the direction to the target.
        """

        if self.target == "left":
            target = self.Environment.left_T_end
        
        elif self.target == "right":
            target = self.Environment.right_T_end

        else:
            raise RuntimeError("Target must be 'left' or 'right'.")

        direction = np.asarray(target) - self.pos

        return direction


    def update(self, dt=None, speed_fact=3, drift_to_random_strength_ratio=0.7, **kwargs):
        """Update the agent, optionally with a new position and velocity.
        
        See Agent.update() in ratinabox/agent.py for kwargs.
        """

        self._check_target()
        if self._check_end():
            self.reset()

        # calculate drift_velocity
        if self.near_branch:
            direction = self.get_direction()
        else:
            direction = self.Environment.T_split - self.pos
        drift_velocity = speed_fact * self.speed_mean * (direction / np.linalg.norm(direction, ord=2))

        super().update(
            dt=dt, 
            skip_checks=True,
            drift_velocity=drift_velocity,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
            **kwargs
            )


    def set_current_arm(self):
        """Sets which arm the agent will navigate to, this run.
        """

        # randomly choose a current target arm
        arms = ["left", "right"]
        self.target = arms[np.random.rand() > self.left_arm_prop]

        if not hasattr(self, "trajectory_targets"):
            self.trajectory_targets = []    
        self.trajectory_targets.append(self.target)

        return


    def reset(self):
        """Reset the agent to a random location.
        """

        super().reset()

        self.set_current_arm()

        return


    def set_all_pos(self):
        """Set all the positions for the agent.
        """
        
        self.start_pos = self.set_pos(self.Environment.T_start)

        # set reset positions
        self.left_reset_pos = self.Environment.left_T_end
        self.right_reset_pos = self.Environment.right_T_end
        self.reset_pos = [
            self.set_pos(reset_pos) 
            for reset_pos in [self.left_reset_pos, self.right_reset_pos]
            ]
        self.reached_reset_pos = []
        
        # set target position
        if self.target_arm == "left":
            edge = self.Environment.left_T_end
        elif self.target_arm == "right":
            edge = self.Environment.right_T_end
        else:
            raise RuntimeError("Target must be 'left' or 'right'.")
        
        T_split = self.Environment.T_split
        self.target_pos = [T_split[i] + (edge[i] - T_split[i]) * self.target_prop for i in [0, 1]]
        self.target_pos = self.set_pos(self.target_pos)
        
        self.reached_target_pos = []
        self.target_waiting = 0

        # set initial position and velocity
        if self.start_pos is not None:
            self.set_pos_vel(pos=self.start_pos, velocity=0)


    def check_reset_pos(self, pos="both"):
        """Check if the agent has reached either of the reset positions.
        
        Returns: Whether the agent has reached either of the reset positions.
        """

        # calculate the distance between the current position and the reset position
        if pos == "both":
            dist = min(
                [np.linalg.norm(self.pos - reset_pos, ord=2) for reset_pos in self.reset_pos]
            )
        elif pos == "left":
            dist = np.linalg.norm(self.pos - self.left_reset_pos, ord=2)
        elif pos == "right":
            dist = np.linalg.norm(self.pos - self.right_reset_pos, ord=2)
        else:
            raise ValueError("pos must be 'both', 'left', or 'right'.")

        # check if the distance is less than the tolerance
        if dist < (self.dt * self.reset_tolerance_prop):
            return True

        return False
    

    def check_left_reset_pos(self):
        return self.check_reset_pos(pos="left")

    def check_right_reset_pos(self):
        return self.check_reset_pos(pos="right")



class BoxAgent(ResetAgent, util.ParamsMixin):
    """Extend the reset agent so that it operates in an exploration box    
    """

    default_params = {
        "reward_fact": 5,
        "no_target_fact": 1,
        "trajectory_length": 2000, # int or iterable of ints
        "n_traj": 10, # number of trajectory lengths to sample
        "target_wait": 10, # number of steps to wait between target reaching
        "target_tolerance_prop": 0.5, # proportion of dt to use as target tolerance
        "num_random_walk_steps": 100, # number of steps to random walk, if target is not in sight
    }

    ignored_param_keys = [
        "reset_pos", "start_pos", "target_pos", "reset_tolerance_prop",
        "fixed_direction"
        ]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = dict()

    def __init__(self, Env, params={}):
        """Initialise the agent.

        Args:
            params (dict, optional): Parameters for the agent. Defaults to {}.

        Raises:
            ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.check_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)     
        self.params.update(params)

        if not isinstance(Env, env.ExploreBox):
            raise TypeError("Env must be an ExploreBox object.")

        self.set_fixed_params()

        super().__init__(Env, self.params)


    def get_targets_and_probs(self, skip_object=None):
        """Get the targets and the probabilities to use when sampling from them.
        """

        reward_num = self.Environment.type_name_to_num_dict["reward"]
        novel_num = self.Environment.type_name_to_num_dict["novel"]

        objects = [None] 
        obj_types = [-1]
        obj_weights = [self.no_target_fact]
        obj_names = ["no target"]
        for obj, obj_type in zip(
            self.Environment.objects["objects"],
            self.Environment.objects["object_types"]
        ):
            if obj_type not in [reward_num, novel_num]:
                continue
            elif (obj == skip_object).all():
                continue
            objects.append(obj)
            obj_types.append(obj_type)            
            obj_weight = self.reward_fact if obj_type == reward_num else 1
            obj_weights.append(obj_weight)
            obj_name = "reward" if obj_type == reward_num else "novel"
            obj_names.append(obj_name)

        div = sum(obj_weights)
        obj_weights = [obj_wei / div for obj_wei in obj_weights]
        
        return [objects, obj_types, obj_weights, obj_names]


    def set_curr_target(self):
        """Set the current target.
        """

        objects, obj_types, obj_weights, obj_names = self.get_targets_and_probs(self.target_pos)
        target_idx = np.random.choice(len(objects), 1, p=np.asarray(obj_weights))[0]

        # sample an object to go toward (check if in FOV, 5 attempts, otherwise no target for x steps)
        self.target = obj_names[target_idx]
        self.target_pos = objects[target_idx]
        self.target_type = obj_types[target_idx]

        if not hasattr(self, "trajectory_targets"):
            self.trajectory_targets = []    
        self.trajectory_targets.append((self.target, self.target_pos, self.target_type))

        self.target_waiting = 0


    def check_target_in_sight(self):
        """Check if the target is in sight.
        
        Returns:
            bool: Whether the target is in sight.
        """

        # check if the target is in the field of view
        dist = self.Environment.get_distances_between___accounting_for_environment(
            self.pos, self.target_pos, wall_geometry="line_of_sight"
        )

        if dist == 1000:
            return False
        else:
            return True


    def set_random_walk(self):
        """Set the random walk.
        """

        if self.target_pos is None or not self.check_target_in_sight():
            self.random_walk = self.num_random_walk_steps
        else:
            self.random_walk = 0


    def set_all_pos(self, first=True):
        """Set all the positions for the agent.
        """
        
        # set initial position and velocity
        self.start_pos = self.Environment.sample_coords()
        self.set_pos_vel(pos=self.start_pos, velocity=0)

        self.target_pos = None
        self.set_curr_target()

        self.target_waiting = 0
        self.set_random_walk()

        if first:
            self.reached_target_pos = []
            self.teleported = []
            self.teleport_pair = []
            self.start_positions = []
        
        self.start_positions.append(self.start_pos)


    def reset(self):
        """Reset the agent to a random location.
        """

        super().reset()

        self.set_all_pos(first=False)

        if len(self.reached_target_pos) == 0 or self.num_steps_total != self.reached_target_pos[-1]:
            self.reached_target_pos.append(-1)

        return
    

    def sample_within_tolerance(self, pos, tolerance_prop=None, max_attempts=100):
        """Sample a position within the tolerance of the given position.

        Args:
            pos (np.ndarray): The position to sample around.
            tolerance_prop (float): The proportion of the tolerance to sample within.
                Defaults to None, in which case the agent's target_tolerance_prop is used.
        
        Returns:
            pos (np.ndarray): The sampled position.
        """

        if len(pos) != 2:
            raise ValueError(f"pos must have length 2, but found {len(pos)}.")

        if tolerance_prop is None:
            tolerance_prop = self.target_tolerance_prop

        tolerance = self.dt * tolerance_prop
        
        new_pos = None
        for _ in range(max_attempts):
            x_jitter = np.random.uniform(-tolerance, tolerance)
            y_max = np.sqrt(tolerance ** 2 - x_jitter ** 2)
            y_jitter = np.random.uniform(-y_max, y_max)
            new_pos = pos + np.asarray([x_jitter, y_jitter])
            if self.Environment.check_if_position_is_in_environment(new_pos):
                break

        if new_pos is None:
            raise RuntimeError(
                f"Could not find a new position within tolerance proportion "
                f"{tolerance_prop} of {pos}. Check that the teleportation out "
                "coordinates are in a reasonable location."
                )

        return new_pos


    def get_shifted_teleport_center(self, teleport_pair, direction="in"):

        shift = self.dt * self.target_tolerance_prop / 2

        teleport_coords = self.Environment.teleport_pairs_dict[teleport_pair][direction][1]
        marker = self.Environment.get_teleport_pair_marker(teleport_pair, direction=direction)

        x_shift, y_shift = 0, 0
        if marker == "<": # towards right
            x_shift = shift
        elif marker == ">": # towards left
            x_shift = -shift
        elif marker == "^":
            y_shift = -shift # below
        elif marker == "v":
            y_shift = shift # above
        else:
            raise RuntimeError(f"Unrecognized marker {marker}.")

        shifted_center = teleport_coords + np.asarray([x_shift, y_shift])

        return shifted_center


    def check_teleport(self):
        """Check if the agent should teleport.
        """

        for teleport_pair in self.Environment.teleport_pairs_dict.keys():
            in_teleport_center = self.get_shifted_teleport_center(
                teleport_pair, direction="in"
                )
            teleport = self.check_pos(in_teleport_center, self.target_tolerance_prop)
            if not teleport:
                continue

            # teleport (sampling near out teleport coords)
            out_teleport_center = self.get_shifted_teleport_center(
                teleport_pair, direction="out"
                )
            # sample within tolerance prop of out teleport coords
            out_coords = self.sample_within_tolerance(out_teleport_center)

            self.set_pos_vel(pos=out_coords, velocity=0)

            self.teleported.append(self.num_steps_total)
            self.teleport_pair.append(teleport_pair)
            break

        return teleport
    

    def update(self, dt=None, speed_fact=3, drift_to_random_strength_ratio=0.7, **kwargs):
        """Update the agent, optionally with a new position and velocity.
        
        See Agent.update() in ratinabox/agent.py for kwargs.
        """

        target_reached = self._check_target()

        if self._check_end():
            self.reset()
        elif target_reached:
            self.set_curr_target()
            self.set_random_walk()

        self.check_teleport()

        if self.random_walk == 0:
            if self.target_pos is None:
                self.set_curr_target()
            self.set_random_walk()

        # calculate drift_velocity
        if self.random_walk > 0:
            drift_velocity = None
            self.random_walk -= 1
        else:
            direction = np.asarray(self.target_pos) - self.pos
            drift_velocity = speed_fact * self.speed_mean * (direction / np.linalg.norm(direction, ord=2))

        super().update(
            dt=dt, 
            skip_checks=True,
            drift_velocity=drift_velocity,
            drift_to_random_strength_ratio=drift_to_random_strength_ratio,
            **kwargs
            )

    def get_trajectory_nodes(self):
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
            target[1] for target in self.trajectory_targets
            if target[1] is not None
        ]
        targets = np.asarray(targets)
        traj_lengs = np.cumsum(self.get_trajectory_lengths_to_date())
        start_pos = np.asarray([self.start_positions[0]])

        reached_target_pos = np.asarray(self.reached_target_pos)
        reached_targets_idxs = np.where(reached_target_pos != -1)[0]
        reached_target_pos = reached_target_pos[reached_targets_idxs]

        reached_targets = np.zeros(len(targets)).astype(bool)
        reached_targets[reached_targets_idxs] = True
        unreached_targets = targets[~reached_targets]
        targets = targets[reached_targets]

        # get start and end nodes
        nodes = np.insert(targets, 0, start_pos[0], axis=0)
        values = np.insert(np.ones(len(nodes)-1), 0, 0)
        steps = np.insert(reached_target_pos, 0, 0)
        for l, leng in enumerate(traj_lengs):
            if leng in reached_target_pos:
                continue
            idx = np.where(leng < reached_target_pos)[0]
            if len(idx):
                # add the end node
                nodes = np.insert(nodes, idx[0], pos[leng-1:leng], axis=0)
                values = np.insert(values, idx[0], -1)
                steps = np.insert(steps, idx[0], leng)
                # add the start node
                nodes = np.insert(nodes, idx[0] + 1, start_pos[l+1:l+2], axis=0)
                values = np.insert(values, idx[0], 0)
                steps = np.insert(steps, idx[0], leng + 1)
            elif self.trajectory_targets[-1][0] != "no target":
                # add the end node
                nodes = np.append(nodes, pos[-2:-1], axis=0)
                values = np.append(values, -1)
                steps = np.append(steps, leng)
        
        if len(unreached_targets) != len(np.where(values == -1)[0]):
            raise RuntimeError("Wrong number of reset points found.")
        
        return nodes, values, steps, unreached_targets
    

    def plot_trajectory(self, target_alpha=0.7, **kwargs):
        
        fig, ax = super().plot_trajectory(
            target_alpha=target_alpha,
            **kwargs
        )

        return fig, ax

    def plot_trajectory_targets(self, fig=None, ax=None, alpha=0.8, plot_env=True, **kwargs):

        if fig is None or ax is None or plot_env:
            fig, ax = self.Environment.plot_environment(fig=fig, ax=ax)
        
        pos = np.asarray(self.history["pos"])
        
        targets = [
            target[1] for target in self.trajectory_targets
            if target[1] is not None
        ]

        unique_targets, counts = [], []
        for target in targets:
            present = [(target == unique_target).all() for unique_target in unique_targets]
            if sum(present):
                counts[present.index(True)] += 1
            else:
                unique_targets.append(target)
                counts.append(1)

        for target, count in zip(unique_targets, counts):
            # write the number of times the target was visited
            ax.text(
                target[0], target[1], str(count),
                horizontalalignment="left",
                verticalalignment="bottom",
                color="white",
                fontsize=10,
                zorder=10,
                fontweight="bold"
            )

        if len(targets) == 0:
            return fig, ax
        
        nodes, values, steps, unreached_targets = self.get_trajectory_nodes()

        # get linewidths
        step_diff = np.diff(steps)
        if len(step_diff) != 0:
            min_val, max_val = np.min(step_diff), np.max(step_diff)
            lws = np.around(
                (step_diff - min_val) / (max_val - min_val), 1
            ) + 1

        unreached = 0
        for n, node in enumerate(nodes[:-1]):
            ax.plot(
                [node[0], nodes[n+1][0]], 
                [node[1], nodes[n+1][1]], 
                color="black", 
                linewidth=lws[n],
                alpha=alpha,
                zorder=1
                )
        
            # add missed targets
            if values[n] == -1:
                ax.plot(
                    [nodes[n+1][0], unreached_targets[unreached][0]], 
                    [nodes[n+1][1], unreached_targets[unreached][1]], 
                    color="black", 
                    ls="dashed",
                    alpha=alpha,
                    zorder=1
                    )
                unreached += 1    

        return fig, ax


    def plot_trajectory_targets_over_time(self, t_start=0, t_end=None, fig=None, ax=None, **kwargs):

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
                ax.axvline(
                    reset_time, 
                    color="black", 
                    ls="dashed", 
                    alpha=0.2,
                    zorder=-1
                )
        
        # plot trajectory
        ax.plot(t, pos[:, 0], color="lightgray", label="X")
        ax.plot(t, pos[:, 1], color="darkgray", label="Y")

        ax.set_title("Position over time")
        ax.set_ylabel("Position")
        ax.set_xlabel("Time (s)")

        ax.spines["right"].set_color(None)
        ax.spines["top"].set_color(None)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        # plot teleportation points as vertical dashed lines
        for step, pair in zip(self.teleported, self.teleport_pair):
            if step < startid or step > endid:
                continue

            obj_type = self.Environment.teleport_pairs_dict[pair]["in"][0]
            color = self.Environment.type_num_to_plot_params_dict[obj_type]["color"]
            ax.axvline(
                t[step], color=color, ls="dashed", alpha=0.8, zorder=-1
            )

        # plot target objects
        targets, target_types = zip(*[
            (target[1], target[2]) for target in self.trajectory_targets
            if target[1] is not None
        ])

        # get the number of steps to the target
        num_steps = np.asarray(self.reached_target_pos)
        if len(num_steps) < len(targets):
            num_steps = np.append(num_steps, -1)
        if len(num_steps) != len(targets):
            raise RuntimeError("Cannot match targets to number of steps to reach.")

        type_num_to_plot_params_dict = copy.deepcopy(self.Environment.type_num_to_plot_params_dict)
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
            ax.plot([target_time] * 2, target, **plot_params, label=label, lw=1.5, alpha=alpha)

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        return fig, ax