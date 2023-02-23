
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
            i = (len(self.reset_steps) - 1) % len(self.trajectory_lengths)
            self.trajectory_length = self.trajectory_lengths[i]

        elif self.trajectory_length is not None:
            self.reset_steps.append(self.curr_trajectory_length)
        
        self.curr_trajectory_length = 0

        return
    
    def log_trajectories(self):
        print(f"Trajectory lengths ({len(self.reset_steps)}) to date (in steps): {self.reset_steps}")


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
        cmap_per=False,
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
            • color: plot point color
            • alpha: plot point opaqness
            • xlim: In 1D, forces the xlim to be a certain time (minutes) (useful if animating this function)
            • background_color: color of the background if not matplotlib default, only for 1D (probably white)
        Returns:
            fig, ax
        """

        if colormap is None:
            colormap = "crest"
        

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
        if len(self.reset_steps) > 0:
            last_step = len(t) - sum(self.reset_steps)
            if cmap_per:
                step_num = [np.linspace(0, 1, steps) for steps in self.reset_steps + [last_step]]
            else:
                step_num = [np.arange(steps) for steps in self.reset_steps + [last_step]]
            step_num = np.concatenate(step_num).astype(float)
            step_num /= step_num.max()
            step_num = step_num[startid : endid][::skiprate]

        c = sns.color_palette(colormap, as_cmap=True)(step_num)
        ##############################

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