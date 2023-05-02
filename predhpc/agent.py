
import warnings

import copy
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import seaborn as sns
from ratinabox import Agent
from ratinabox import utils as rat_utils

from predhpc import util


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
        "reset_tolerance_prop": 0.5, # proportion of dt to use as reset tolerance
        "fixed_direction": False, # keep same direction (1D environment only)
    }

    def __init__(self, Environment, params={}):
        """Initialise the agent.

        Args:
            Environment (Environment): The environment in which the agent is placed.
            params (dict, optional): Parameters for the agent. Defaults to {}.

        Raises:
            ValueError: If passing iterable for trajectory_length, must have length > 0.
        """

        self.params = copy.deepcopy(__class__.default_params)     
        self.params.update(params)

        super().__init__(Environment, self.params)

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

        self.reset_steps = []
        self.set_start_reset_pos()
        self.set_trajectory_lengths()


    def set_start_reset_pos(self):
        self.start_pos = self.set_pos(self.start_pos)
        self.reset_pos = self.set_pos(self.reset_pos)
        self.reached_reset_pos = []
        if self.start_pos is not None:
            self.set_pos_vel(pos=self.start_pos, velocity=0)


    def set_trajectory_lengths(self):
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
        """Check that start_pos and end_pos are of the correct length, and 
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
            np.linalg.norm(self.velocity)
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
                new_velocity = prev_velocity + rat_utils.ornstein_uhlenbeck(
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
                        "Warning: You have solid 1D boundary conditions and non-zero speed mean."
                    )
    
        self.fix_velocity_record()

        return


    def reset(self):
        """Reset the agent to a random location.
        """

        self.set_pos_vel(pos=self.start_pos, velocity=0)
        
        self.reset_steps.append(self.curr_trajectory_length)
        if self.trajectory_lengths is not None:
            i = len(self.reset_steps) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]
       
        self.curr_trajectory_length = 0

        return
    
    
    def get_trajectory_lengths_to_date(self):
        """Return the trajectory lengths to date.

        Returns:
            list: Trajectory lengths to date.
        """
        traj_leng_to_date = self.reset_steps
        if self.curr_trajectory_length > 0:
            traj_leng_to_date = self.reset_steps + [self.curr_trajectory_length]
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
        util.plot_trajectory_lengths(dt=self.dt, trajectory_lengths=traj_leng_to_date, in_min=in_min)

   
    def get_reset_times(self):
        """Get the reset times. 

        Returns:
            list: Reset times.

        Raises:
            ValueError: If agent does not have reset steps.
        """

        if hasattr(self, "reset_steps"):
            if len(self.reset_steps) == 0:
                reset_times = np.array([])
            else:
                reset_times = np.cumsum(self.reset_steps) * self.dt
        else:
            raise ValueError("Agent does not have reset steps.")
        
        return reset_times
   

    def check_reset_pos(self):
        """Check if the agent has reached the reset position.
        
        Returns: Whether the agent has reached the reset position.
        """

        if self.reset_pos is not None:
            # calculate the distance between the current position and the reset position
            dist = np.linalg.norm(self.pos - self.reset_pos)

            # check if the distance is less than the tolerance
            if dist < self.dt * self.reset_tolerance_prop:
                return True

        return False


    def check_end(self):
        """Check if the agent has reached the end of its trajectory.

        Returns:
            bool: Whether the agent has reached the end of its trajectory.
        """

        if self.reset_pos is not None and self.check_reset_pos():
            # record the time step at which the agent reached the reset position
            self.reached_reset_pos.append(self.num_steps_total)
            return True

        if self.trajectory_length is not None:
            if self.curr_trajectory_length >= self.trajectory_length:
                return True

        return False


    def update(self, dt=None, **kwargs):
        """Update the agent, optionally with a new position and velocity.
        
        See Agent.update() in ratinabox/agent.py for kwargs.
        """

        if self.check_end():
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
        endid = np.argmin(np.abs(t - (t_end)))
        skiprate = max(1, int((1 / framerate) / dt))
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
        alpha /= self.Environment.D
        alpha_pts = 0.9 / self.Environment.D
        for i in range(self.Environment.D):
            ax.scatter(time, pos, alpha=alpha, marker=".", color=color, s=ms/5)
            
            if n_reached:
                if self.start_pos is not None:
                    x_start = [t[x] for x in self.reached_reset_pos]
                    y_start = [self.start_pos[i]] * n_reached
                    ax.scatter(
                        x_start, y_start, marker=".", color="blue", alpha=alpha_pts, s=ms
                        )
                
                if self.reset_pos is not None:
                    x_reset = [t[x - 1] for x in self.reached_reset_pos]
                    y_reset = [self.reset_pos[i]] * n_reached
                    ax.scatter(
                        x_reset, y_reset, marker="x", color="red", alpha=alpha_pts, s=ms/3
                        )

        ax.set_xlabel("Time / sec")
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
        cmap_per=False,
        scale_cmap_per=False,
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
            • cmap_per: if True, the colormap is used to set the color for each time point. Otherwise, each trajectory has its own color.
            • scale_cmap_per: if True, and cmap_per is True, the full range of the colormap is used for each trajectory, regardless of its length
        
        Returns:
            fig, ax
        """        

        dt = self.dt
        t, pos = np.array(self.history["t"]), np.array(self.history["pos"])
        if t_end == None:
            t_end = t[-1]
        startid = np.argmin(np.abs(t - (t_start)))
        endid = np.argmin(np.abs(t - (t_end)))
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

        if len(self.reset_steps) > 0:
            last_length = len(t) - sum(self.reset_steps)
            trajectory_lengths = self.reset_steps + [last_length]
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
        else:
            cmap_vals = 0.5
            traj_idx = np.zeros(len(trajectory))
        
        if colormap is None:
            colormap = "crest"
        c = sns.color_palette(colormap, as_cmap=True)(cmap_vals)
        ##############################

        if self.Environment.dimensionality == "2D":
            fig, ax = self.Environment.plot_environment(fig=fig, ax=ax)
            s = 15 * np.ones_like(time)
            if decay_point_size == True:
                s = 15 * np.exp((time - time[-1]) / 10)
                s[(time[-1] - time) > 15] *= 0

            if plot_traj_ends == True:
                ends = np.where(np.diff(traj_idx) > 0)[0]
                ends = np.append(ends, len(trajectory) - 1)
                s[ends] = 30
                c[ends] = mcolors.to_rgba("darkred") ### set last colormap value to black

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
        endid = np.argmin(np.abs(t - (t_end)))

        if startid > endid:
            raise ValueError("'startid' must be lower than 'endid'.")

        if colormap is None:
            colormap = "crest"
        cmap = sns.color_palette(colormap, as_cmap=True)

        all_ends = np.cumsum(self.reset_steps)
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
    