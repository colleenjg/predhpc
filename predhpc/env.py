import copy
import itertools
import warnings

import numpy as np
from matplotlib import pyplot as plt

from ratinabox import Environment
from ratinabox import utils as rutils

from predhpc import util


class EnvironmentWarning(UserWarning):
    pass

warnings.simplefilter("once", EnvironmentWarning)


class TEnv(Environment, util.ParamsMixin):
    """T-shaped environment.   
    """

    default_params = {
        "prop": 0.2, # T-shape arms and stem width (prop of env dims)
        "scale_x": 1, # env width
        "scale_y": 1, # env height
        "prop_x": None, # T-shape stem width (prop of env width)
        "prop_y": None, # T-shape arms width (prop of env height)
    }
    
    ignored_param_keys = ["boundary", "scale", "aspect"]
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "holes": [],  # no holes
    }

    def __init__(self, params=dict()):
    
        self.check_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)     
        self.params.update(params)

        self.set_fixed_params()
                
        super().__init__(self.params)

        self.prop_x = self.prop if self.prop_x is None else self.prop_x
        self.prop_y = self.prop if self.prop_y is None else self.prop_y


    def set_fixed_params(self):
        """Sets fixed parameters.
        """
        all_fixed_params = self.get_all_fixed_params()
        for key, value in all_fixed_params.items():
            self.params[key] = value

        self.params["boundary"] = util.get_T_shape_boundaries(
            prop=self.params["prop"],
            scale_x=self.params["scale_x"],
            scale_y=self.params["scale_y"],
            prop_x=self.params["prop_x"],
            prop_y=self.params["prop_y"],
            )


    @property
    def branch_y(self):
        if not hasattr(self, "_branch_y"):
            self._branch_y = (1 - self.prop_y) * self.scale_y
        return self._branch_y


    @property
    def left_T_end(self):
        if not hasattr(self, "_left_T_end"):
            x_dim = self.prop_x / 2 * self.scale_x
            y_dim = (1 - self.prop_y / 2) * self.scale_y
            self._left_T_end = [x_dim, y_dim]
        return self._left_T_end

    @property
    def right_T_end(self):
        if not hasattr(self, "_right_T_end"):
            x_dim = (1 - self.prop_x / 2) * self.scale_x
            y_dim = (1 - self.prop_y / 2) * self.scale_y
            self._right_T_end = [x_dim, y_dim]
        return self._right_T_end

    @property
    def T_ends(self):
        """Get the coordinates of the ends of the T-shape arms.
        """

        return [self.left_T_end, self.right_T_end]


    @property
    def T_start(self):
        """Get the coordinates of the start of the T-shape.
        """

        if not hasattr(self, "_T_start"):
            x_dim = 0.5 * self.scale_x
            y_dim = (self.prop_y / 2) * self.scale_y
            self._T_start = [x_dim, y_dim]

        return self._T_start


    @property
    def T_split(self):
        """Get the coordinates of the split of the T branches.
        """

        if not hasattr(self, "_T_split"):
            x_dim = 0.5 * self.scale_x
            y_dim = (1 - self.prop_y / 2) * self.scale_y
            self._T_split = [x_dim, y_dim]

        return self._T_split



    def plot_environment(self, fig=None, ax=None, **kwargs):
        """Plot the environment.
        """

        fig, ax = super().plot_environment(fig=fig, ax=ax, **kwargs)

        ax.scatter(*self.T_start, marker="o", color="blue", s=20, zorder=5, label="start")
        ax.scatter(*self.left_T_end, marker="x", color="red", s=20, zorder=5, label="reset")
        ax.scatter(*self.right_T_end, marker="x", color="red", s=20, zorder=5)
        ax.legend(loc="lower right", frameon=False)

        return fig, ax



class ExploreBox(Environment, util.ParamsMixin):
    """Box-shaped environment to explore.   
    """

    default_params = {
        "init_random_reward_obj": 1,
        "init_random_novel_obj": 5,
        "init_random_walls": 5,
        "init_random_teleport_pairs": 2,
        "wall_lengths": [0.1, 0.2],
        "min_dist": 0.1,
    }
    
    ignored_param_keys = []
    ignored_params = {key: None for key in ignored_param_keys}

    fixed_params = {
        "dimensionality": "2D",  # 1D or 2D environment
        "boundary_conditions": "solid",  # solid vs periodic
        "holes": [],  # no holes,
        "boundary": None,
    }

    def __init__(self, params=dict()):
        """Initialize the environment.
        """
    
        self.check_ignored_params(params)

        self.params = copy.deepcopy(__class__.default_params)     
        self.params.update(params)

        self.set_fixed_params()
                
        super().__init__(self.params)

        if min(self.wall_lengths) <= 0:
            raise ValueError("Wall lengths must be positive.")

        self.num_teleport_pairs = 0

        self.add_reward_objects(self.init_random_reward_obj)
        self.add_novel_objects(self.init_random_novel_obj)
        self.add_teleport_pairs(self.init_random_teleport_pairs)
        self.add_walls(self.init_random_walls)


    def get_new_teleport_pair_obj_type_nums(self, first=None):
        """Get object type numbers for a new teleport pair.

        Args:
            first (int): First object type number to use. If None, use the next 
                available number. Defaults to None.
        
        Returns:
            obj_type_nums (dict): Dictionary of object type numbers for the 
                teleport pair.
        """

        if first is None:
            first = np.max(list(self.obj_type_num_to_name_dict.keys())) + 1

        obj_type_nums = {
            "in": first,
            "out": first + 1,
        }

        return obj_type_nums

    @property
    def obj_type_num_to_name_dict(self):
        """Dictionary for getting object type name from number.
        """

        obj_type_num_to_name_dict = {
            0: "reward",
            1: "novel",
        }

        for n in range(self.num_teleport_pairs):
            obj_type_nums = self.get_new_teleport_pair_obj_type_nums(
                first=np.max(list(obj_type_num_to_name_dict.keys())) + 1
                )
            for (direction, i) in obj_type_nums.items():
                obj_type_num_to_name_dict[i] = f"teleport_{n}_{direction}"

        return obj_type_num_to_name_dict

    @property
    def type_name_to_num_dict(self):
        """Dictionary for getting object type number from name.
        """

        obj_type_name_to_num_dict = {
            val: key for key, val in self.obj_type_num_to_name_dict.items()
        }

        return obj_type_name_to_num_dict

    @property
    def type_num_to_plot_params_dict(self):
        """Dictionary for getting object type number from name.
        """

        obj_type_num_to_name_dict = self.obj_type_num_to_name_dict
        teleport_nums = [
            val.replace("teleport_", "").replace("in_", "")
            for val in obj_type_num_to_name_dict.values()
            if val.startswith("teleport") and "_in" in val
        ]
        teleport_vals = np.linspace(0.5, 1, len(teleport_nums))
        teleport_colors = plt.get_cmap("Oranges")(teleport_vals)

        type_num_to_plot_params_dict = dict()
        for num, name in obj_type_num_to_name_dict.items():
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
                marker = "<" if direc == "in" else ">"
                color = teleport_colors[
                    int(name.replace("teleport_", "").replace(f"_{direc}", ""))
                    ]
                type_num_to_plot_params_dict[num] = {
                    "name": name,
                    "marker": marker,
                    "color": color,
                    "s": 20,
                    "zorder": 5,
                }
            else:
                raise ValueError(f"Unknown object type name: {name}")

        return type_num_to_plot_params_dict
    

    def set_fixed_params(self):
        """Sets fixed parameters.
        """
        all_fixed_params = self.get_all_fixed_params()
        for key, value in all_fixed_params.items():
            self.params[key] = value


    def get_dist_from_coords_to_closest_object(self, coords):
        """Get the distance from a set of coordinates to the closest objects.
        
        Args:
            coords (np.ndarray): coordinates to get distance from.
        
        Returns:
            float: closest distance.
        """

        if len(self.objects["objects"]) == 0:
            return np.inf

        closest_dist = min([
            np.linalg.norm(coords - obj_coords, ord=2) 
            for obj_coords in self.objects["objects"]
        ])

        return closest_dist


    def get_dist_from_coords_to_closest_wall(self, coords):
        """Get the distance from a set of coordinates to the closest wall.

        Args:
            coords (np.ndarray): coordinates to get distance from.

        Returns:
            float: closest distance.
        """

        if len(self.walls) == 0:
            return np.inf

        # returns points (1) x vectors x coords
        closest_dist = np.min(util.shortest_distances_from_points_to_lines(
            coords, self.walls
            ))

        return closest_dist


    def sample_coords(self, min_dist=None, max_attempts=1000):
        """Sample coordinates situated at least min_dist from the closest 
        object or wall.

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
            min_dist = self.min_dist
        
        i = 0
        while True:
            x = np.random.uniform(self.extent[0], self.extent[1])
            y = np.random.uniform(self.extent[2], self.extent[3])

            coords = np.array([x, y])

            # check distance to walls
            closest_dist = min([
                self.get_dist_from_coords_to_closest_object(coords),
                self.get_dist_from_coords_to_closest_wall(coords)
            ])
            if closest_dist >= min_dist:
                break
            if i > max_attempts:
                raise ValueError(
                    "Could not sample valid coordinates situated at least "
                    f"{min_dist} from the closest objects / walls."
                    )
            i += 1
        
        return coords


    def sample_wall_end(self, start_coords, min_dist=None):
        """Sample valid coordinates for the end of a wall given the start coordinates.

        Args:
            start_coords (1d array): start of wall.
            min_dist (float, optional): minimum distance to closest object or
                wall. Defaults to None.
        
        Returns:
            end_coords (1d array): sampled end of wall coordinates [x, y]. 
                Returns None if could not sample valid end coordinates.
        """

        if not self.check_if_position_is_in_environment(start_coords):
            return None

        if min_dist is None:
            min_dist = self.min_dist

        # sample wall length
        wall_length = np.random.uniform(*self.wall_lengths)

        # sample orientation + direction, then cycle through if needed, before 
        # abandoning each time check that the wall's max distance from another 
        # wall is reasonable and distance from objects.
        wall_orientations = ["x", "y"]
        wall_directions = [-1, 1]
        wall_ori_direcs = list(itertools.product(wall_orientations, wall_directions))
        np.random.shuffle(wall_ori_direcs)

        for wall_ori, wall_direc in wall_ori_direcs:
            c = 0 if wall_ori == "x" else 1
            end_coords = np.array(start_coords) # new array
            end_coords[c] += wall_length * wall_direc

            # check that end_coords are within bounds
            if not self.check_if_position_is_in_environment(end_coords):
                end_coords = None
                continue

            # check that end_coords are far enough from objects, if there are any
            if len(self.objects["objects"]) != 0:
                closest_dist = np.min(util.shortest_distances_from_points_to_lines(
                    self.objects["objects"], [start_coords, end_coords]
                    ))

                if closest_dist < min_dist:
                    end_coords = None
                    continue

        return end_coords


    def add_reward_objects(self, num=1):
        """Add reward objects.

        Args:
            num (int, optional): number of reward objects to add. Defaults to 1.
        """

        reward_type = self.type_name_to_num_dict["reward"]

        for _ in range(num):
            coords = self.sample_coords()
            self.add_object(coords, type=reward_type)


    def add_novel_objects(self, num=1):
        """Add novel objects.

        Args:
            num (int, optional): number of novel objects to add. Defaults to 1.
        """

        novel_type = self.type_name_to_num_dict["novel"]

        for _ in range(num):
            coords = self.sample_coords()
            self.add_object(coords, type=novel_type)


    def add_teleport_pairs(self, num=1):
        """Add teleport pairs (directional).

        Args:
            num (int, optional): number of teleport pairs to add. Defaults to 1.
        """

        for _ in range(num):
            obj_type_nums = self.get_new_teleport_pair_obj_type_nums()
            for (_, i) in obj_type_nums.items():
                coords = self.sample_coords()
                self.add_object(coords, type=i)
            self.num_teleport_pairs += 1


    def check_walls_ends_too_close(self, new_wall_coords, min_dist=None):
        """
        Checks whether a new wall's ends is too close to the ends of existing 
        walls.

        Specifically checks whether an end of the new wall intersects at less 
        than 45 degrees near the end of an existing wall, forming an V shape 
        with small ends sticking out. If so, returns True, else False.

        Does NOT check whether the new wall overlaps exactly with an existing 
        wall, or intersects near the middle of either wall. 
        """

        if len(self.walls) == 0:
            return False
        
        if min_dist is None:
            min_dist = self.min_dist
        
        new_wall = np.asarray(new_wall_coords)

        for wall in self.walls:
            # get angle between two vectors
            angle = util.get_angle_between_vectors(
                np.diff(new_wall_coords, axis=0)[0], 
                np.diff(wall, axis=0)[0]
                )
            
            if angle > 45:
                continue
            
            # if angle is less than 45 degrees, check any ends of the walls are too 
            # close together
            dists, coords = [], []
            for c1, c2 in itertools.product([0, 1], [0, 1]):
                coords.append([c1, c2])
                dists.append(np.linalg.norm(wall[c1] - new_wall[c2], ord=2))

            order = np.argsort(dists)

            if dists[order[0]] < self.min_dist:
                # farther must be at least as far as if the walls 
                # intersected only at their ends (no intersection) 
                farthest = dists[order[-1]]
                c1, c2 = coords[order[-1]]

                end1 = wall[c1] - wall[1 - c1]
                end2 = new_wall[c2] - new_wall[1 - c2]
                exp_dist = np.linalg.norm(end1 - end2, ord=2)

                if farthest < exp_dist:
                    return True
        
        return False


    def add_walls(self, num=1, max_attempts=1000):
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

        warnings.warn(
            "add_walls() does not check whether a new wall will create a hole "
            "in the environment. Be sure to check environment visually.", 
            category=EnvironmentWarning
            )
 
        for _ in range(num):
            i = 0
            while True:
                start_coords = self.sample_coords()
                end_coords = self.sample_wall_end(start_coords)
                if end_coords is not None:                    
                    # check that wall ends are not too close to another
                    if self.check_walls_ends_too_close(
                        [start_coords, end_coords]
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

    def plot_environment(self, fig=None, ax=None, **kwargs):
        """Plot the environment.

        Args:
            fig (matplotlib figure, optional): figure to plot on. Defaults to None.
            ax (matplotlib axis, optional): axis to plot on. Defaults to None.

            
        Returns:
            fig (matplotlib figure): figure with environment plotted.
            ax (matplotlib axis): axis with environment plotted.
        """

        class DontPlotObjects():
            def __init__(self, env):
                self.env = env
                self.plot_objects = env.plot_objects
            def __enter__(self):
                self.env.plot_objects = False
            def __exit__(self, type, value, traceback):
                self.env.plot_objects = self.plot_objects

        with DontPlotObjects(self):
            fig, ax = super().plot_environment(fig=fig, ax=ax, **kwargs)

        if self.plot_objects:
            type_num_to_plot_params_dict = copy.deepcopy(self.type_num_to_plot_params_dict)
            for coords, obj_type in zip(self.objects["objects"], self.objects["object_types"]):
                label = None
                if "name" in type_num_to_plot_params_dict[obj_type].keys():
                    label = type_num_to_plot_params_dict[obj_type].pop("name")
                ax.scatter(*coords, **type_num_to_plot_params_dict[obj_type], label=label)
            
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        return fig, ax


