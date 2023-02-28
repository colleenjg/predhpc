
import warnings

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import seaborn as sns
from ratinabox import Agent

from predhpc import util


class ResetAgent(Agent.Agent):
    ### Extend the agent so that is has an optimal maximum trajectory length after which is resets to a random location


    def __init__(self, Environment, params={}):
        """Initialise the agent.

        Args:
            Environment (Environment): The environment in which the agent is placed.
            params (dict, optional): Parameters for the agent. Defaults to {}.

        Raises:
            ValueError: If passing iterable for trajectory_length, must have length > 0.

        default_params = {
            "trajectory_length": None, # int or iterable of ints
            "n_traj": None, # number of trajectory lengths to sample
            "exp": None, # exponential factors for trajectory_length (inv. scale, rate, minimum). Defaults to None.
            "rand": None, # max value for randomizing trajectory_length
        }
        """

        default_params = {
            "trajectory_length": None, # int or iterable of ints
            "n_traj": None, # number of trajectory lengths to sample
            "exp": None, # exponential factors for trajectory_length (inv. scale, rate, minimum). Defaults to None.
            "rand": None, # max value for randomizing trajectory_length
        }

        self.params = default_params
        self.params.update(params)

        super().__init__(Environment, self.params)

        if self.trajectory_length is not None:
            self.n_traj = None
            self.exp = None
            self.rand = None

        elif self.n_traj:
            self.trajectory_length = util.get_trajectory_lengths(n_traj=self.n_traj, exp=self.exp, rand=self.rand)
                
        self.trajectory_lengths = None
        self.curr_trajectory_length = 0
        if self.trajectory_length is not None:
            self.reset_steps = []
            if not isinstance(self.trajectory_length, int):
                self.trajectory_length = np.maximum(self.trajectory_length, 1)
                if len(self.trajectory_length) == 0:
                    raise ValueError("If passing iterable, must have length > 0.")
                self.trajectory_lengths = self.trajectory_length
                self.trajectory_length = self.trajectory_lengths[0]



    def set_pos_vel(self):
        """Set the position and velocity of the agent.
        
        From Agent.__init__() in ratinabox/agent.py
        """

        # initialise starting positions and velocity
        if self.Environment.dimensionality == "2D":
            self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            direction = np.random.uniform(0, 2 * np.pi)
            self.velocity = self.speed_std * np.array(
                [np.cos(direction), np.sin(direction)]
            )
            self.rotational_velocity = 0

        if self.Environment.dimensionality == "1D":
            self.pos = self.Environment.sample_positions(n=1, method="random")[0]
            self.velocity = np.array([self.speed_mean])
            if self.Environment.boundary_conditions == "solid":
                if self.speed_mean != 0:
                    warnings.warn(
                        "Warning: You have solid 1D boundary conditions and non-zero speed mean. "
                    )
        return

    def reset(self):
        """Reset the agent to a random location.
        """

        self.set_pos_vel()
        
        if self.trajectory_lengths is not None:
            self.reset_steps.append(self.curr_trajectory_length)
            i = len(self.reset_steps) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]

        elif self.trajectory_length is not None:
            self.reset_steps.append(self.curr_trajectory_length)
        
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


    def plot_trajectories_to_date(self, in_min=True):
        """Plot the trajectory lengths to date.
        
        Args:
            in_min (bool, optional): Whether to plot in minutes. Defaults to True.
        """

        traj_leng_to_date = self.get_trajectory_lengths_to_date()
        util.plot_trajectory_lengths(dt=self.dt, trajectory_lengths=traj_leng_to_date, in_min=in_min)

    def update(self, **kwargs):
        """Update the agent, optionally with a new position and velocity.
        
        See Agent.update() in ratinabox/agent.py for kwargs.
        """

        if self.trajectory_length is not None:
            if self.curr_trajectory_length >= self.trajectory_length:
                self.reset()
        
        super().update(**kwargs)

        self.curr_trajectory_length += 1


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
        if self.Environment.dimensionality == "2D":
            skiprate = max(1, int((1 / framerate) / dt))
            trajectory = pos[startid:endid, :][::skiprate]
        if self.Environment.dimensionality == "1D":
            skiprate = max(1, int((1 / framerate) / dt))
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
    