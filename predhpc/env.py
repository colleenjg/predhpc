import copy
import itertools
import warnings

import numpy as np
from matplotlib import pyplot as plt

from ratinabox import Environment

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
        "min_dist": 0.1, # between objects (walls is half)
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


    def reset_obj_type_dicts(self):
        """Reset the object type dictionaries.
        """

        dict_attr_names = [
            "_obj_type_num_to_name_dict",
            "_type_num_to_plot_params_dict",
            "_teleport_pairs_dict",
        ]

        for dict_attr_name in dict_attr_names:
            if hasattr(self, dict_attr_name):
                delattr(self, dict_attr_name)


    @property
    def obj_type_num_to_name_dict(self):
        """Dictionary for getting object type name from number.
        """

        if not hasattr(self, "_obj_type_num_to_name_dict"):

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
            self._obj_type_num_to_name_dict = obj_type_num_to_name_dict


        return self._obj_type_num_to_name_dict

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

        if not hasattr(self, "_type_num_to_plot_params_dict"):

            teleport_nums = [
                val.replace("teleport_", "").replace("in_", "")
                for val in self.obj_type_num_to_name_dict.values()
                if val.startswith("teleport") and "_in" in val
            ]
            teleport_vals = np.linspace(0.5, 1, len(teleport_nums))
            teleport_colors = plt.get_cmap("Oranges")(teleport_vals)

            type_num_to_plot_params_dict = dict()
            for num, name in self.obj_type_num_to_name_dict.items():
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
                    teleport_num = int(name.replace("teleport_", "").replace(f"_{direc}", ""))
                    color = teleport_colors[teleport_num]
                    type_num_to_plot_params_dict[num] = {
                        "name": name,
                        "marker": self.get_teleport_pair_marker(teleport_num, direction=direc),
                        "color": color,
                        "s": 20,
                        "zorder": 5,
                    }
                else:
                    raise ValueError(f"Unknown object type name: {name}")
            
            self._type_num_to_plot_params_dict = type_num_to_plot_params_dict

        return self._type_num_to_plot_params_dict
    

    def get_teleport_pair_orientation(self, teleport_pair_num):
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
    

    def get_number_object_types_split(self):
        """Get the number of each object type.

        Returns:
            tuple: number of novel, reward, and teleport objects.
        """

        num_novel, num_reward, num_teleport = 0, 0, 0
        for object_type in self.object_types:
            object_name = self.obj_type_num_to_name_dict[object_type]
            if object_name == "novel":
                num_novel += 1
            elif object_name == "reward":
                num_reward += 1
            elif "teleport" in object_name:
                num_teleport += 1
        
        if num_teleport % 2:
            raise RuntimeError("Number of teleport pairs should be even.")
        
        return num_novel, num_reward, num_teleport


    def get_teleport_pair_marker(self, teleport_pair_num, direction="in"):
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
            min_dist = self.min_dist
        
        i = 0
        while True:
            x = np.random.uniform(self.extent[0], self.extent[1])
            y = np.random.uniform(self.extent[2], self.extent[3])

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


    def sample_wall_end(self, start_coords, min_dist=None):
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
            min_dist = self.min_dist / 2

        # sample wall length
        wall_length = np.random.uniform(*self.wall_lengths)

        # sample orientation + direction, then cycle through if needed, before 
        # abandoning each time check that the wall's max distance from another 
        # objects is reasonable.
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
        
        if num > 0:
            self.reset_obj_type_dicts()


    def add_novel_objects(self, num=1):
        """Add novel objects.

        Args:
            num (int, optional): number of novel objects to add. Defaults to 1.
        """

        novel_type = self.type_name_to_num_dict["novel"]

        for _ in range(num):
            coords = self.sample_coords()
            self.add_object(coords, type=novel_type)

        if num > 0:
            self.reset_obj_type_dicts()


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
            self.reset_obj_type_dicts() # within loop, so that teleport pair object types are not reused            


    @property
    def teleport_pairs_dict(self):
        """Returns dictionary of teleport pairs (directional).
        """

        if not hasattr(self, "_teleport_pairs_dict"):
            teleport_pairs_dict = dict()
            for name, obj_type in self.type_name_to_num_dict.items():
                if name.startswith("teleport_") and "in" in name:
                    obj_type_in = obj_type
                    teleport_pair = int(name.replace("teleport_", "").replace("_in", ""))
                    out_key = f"teleport_{teleport_pair}_out"
                    if out_key not in self.type_name_to_num_dict.keys():
                        raise RuntimeError(f"Teleport in {teleport_pair} does not have 'out' pair.")
                    obj_type_out = self.type_name_to_num_dict[out_key]

                    coords = []                    
                    for obj_type in [obj_type_in, obj_type_out]:
                        object_idxs = np.where(self.objects["object_types"] == obj_type)[0]
                        if len(object_idxs) != 1:
                            raise RuntimeError(f"Expected teleport in {teleport_pair} to correspond to exactly one object, but found {len(object_idxs)}.")
                        coords.append(self.objects["objects"][object_idxs[0]])
                    
                    teleport_pairs_dict[teleport_pair] = {
                        "in": (obj_type_in, coords[0]),
                        "out": (obj_type_out, coords[1])
                    }

            self._teleport_pairs_dict = teleport_pairs_dict

        return self._teleport_pairs_dict


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

    def plot_environment(self, fig=None, ax=None, add_labels=None, **kwargs):
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

        if fig is None and ax is None:
            env_width = self.extent[1] - self.extent[0]
            add_y = 0
            if self.plot_objects:
                add_x = 3 * env_width # for legend and labels
            fig, ax = plt.subplots(
                figsize=(3 * env_width + add_x, 3 * (self.extent[3] - self.extent[2]))
            )

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


