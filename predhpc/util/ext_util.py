import copy
from typing import Any
import pprint
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

import ratinabox
from ratinabox import utils as rutils  # type: ignore[import]
from predhpc.util import gen_util  # type: ignore[import]


class TemporarilyChangeAttribute:
    def __init__(self, obj, attr, temp_value):
        self.obj = obj
        self.attr = attr
        self.temp_value = temp_value

    def __enter__(self):
        self.prev_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.temp_value)

    def __exit__(self, *args):
        setattr(self.obj, self.attr, self.prev_value)


def extract_objects_from_Pyrs(Pyrs):
    """
    extract_objects_from_Pyrs(Pyrs)

    Extract objects from a Pyrs object.

    Args:
    - Pyrs (learning_neurons.BTSPLayer or two_comp_neurons.TwoCompLayer):
        Pyr. neuron layer

    Returns:
    - Env (env.Environment): Environment
    - Ag (agent.Agent): Agent
    - PCs (riab_neurons.PlaceCells): Place cells
    - Objs (object_neurons.ObjectCells or object_neurons.ObjectInstanceCells):
        Object cells (None if not applicable).
    """

    Env = Pyrs.Agent.Environment

    Ag = Pyrs.Agent

    if hasattr(Pyrs, "SomaCompartment"):  # two compartment
        Obj_key = list(Pyrs.DendriteCompartment.inputs.keys())[-1]
        Objs = Pyrs.DendriteCompartment.inputs[Obj_key]["layer"]

        PC_key = list(Pyrs.SomaCompartment.inputs.keys())[0]
        PCs = Pyrs.SomaCompartment.inputs[PC_key]["layer"]
    else:
        Objs = None
        PC_key = list(Pyrs.inputs.keys())[0]
        PCs = Pyrs.inputs[PC_key]["layer"]

    if Objs is not None:
        if not hasattr(Objs, "input_object_types"):
            raise RuntimeError(f"Objs incorrectly identified. Got {type(Objs)}.")

    if not hasattr(PCs, "place_cell_centres"):
        raise RuntimeError(f"PCs incorrectly identified. Got {type(PCs)}.")

    return Env, Ag, PCs, Objs


class ParamsManagerMixin:
    @property
    def have_ignored_params_been_checked(self):
        if not hasattr(self, "_have_ignored_params_been_checked"):
            self._have_ignored_params_been_checked = False
        return self._have_ignored_params_been_checked

    @classmethod
    def check_ignored_params_for_class(cls, params: dict[str, Any]):
        """
        cls.check_ignored_params_for_class()

        Check that specific parameters are ignored.

        Args:
        - params (dict): Parameters to check.

        Raises:
        - KeyError: If any of the parameters being checked are not ignored.
        """

        # collect all class parameters for the specified dictionary name
        all_ignored_params_dict = rutils.collect_all_params(
            cls, dict_name="ignored_params"
        )

        for key in all_ignored_params_dict:
            if key in params.keys():
                warnings.warn(
                    f"'{key}' should not be provided for {cls.__name__}. "
                    "Will be ignored."
                )

    def check_if_ignored_params(self, params: dict[str, Any]):
        """
        self.check_if_ignored_params()

        Check that specific parameters are ignored.

        Args:
        - params (dict): Parameters to check.

        Raises:
        - KeyError: If any of the parameters being checked are not ignored.
        """

        if hasattr(self, "_have_ignored_params_been_checked"):
            return

        self.check_ignored_params_for_class(params)
        self._have_ignored_params_been_checked = True

    @classmethod
    def get_all_fixed_params(cls, verbose: bool = False) -> dict[str, Any]:
        """
        cls.get_all_fixed_params()

        Obtain a dictionary of all the default parameters of the class, including
        those inherited from its parents.

        Args:
        - verbose (bool, optional): If True, prints the parameters. Default is False.

        Returns:
        - all_fixed_params (dict): Dictionary of all the default parameters of the
            class, including those inherited from its parent classes.
        """

        all_fixed_params: dict[str, Any] = dict()
        all_fixed_params.update(
            rutils.collect_all_params(cls, dict_name="fixed_params")
        )
        if verbose:
            pprint.pprint(all_fixed_params)
        return all_fixed_params

    def add_fixed_params(self, params: dict[str, Any] = dict()) -> dict[str, Any]:
        """
        self.add_fixed_params()

        Set fixed parameters for the class.

        Args:
        - params (dict, optional): Parameters to set. Default is dict().

        Returns:
        - params (dict): Parameters with fixed parameters added.
        """

        all_fixed_params = self.get_all_fixed_params()

        params = copy.copy(
            params
        )  # avoid deep copy to preserve reference to input layers
        for key, value in all_fixed_params.items():
            if key in params.keys() and value != params[key]:
                raise ValueError(
                    f"'{key}' parameter should not be passed, unless it is set to "
                    f"'{value}'."
                )
            params[key] = value

        return params


def repeat_to_fill(
    trajectory_lengths: np.ndarray[tuple[int], np.dtype[np.int64]],
    max_num_minutes: float | int = 2,
    dt: float = 0.03,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """
    repeat_to_fill(trajectory_lengths)

    Obtain an array of trajectory lengths that repeats to fill a specified
    duration in minutes. Values are repeated using np.repeat() (not np.tile()).

    Args:
    - trajectory_lengths (1D np.ndarray): Trajectory lengths, in steps
    - max_num_minutes (float, optional): Time to fill, in minutes. Default is 2.
    - dt (float, optional): Time step. Default is 0.03.

    Returns:
    - trajectory_lengths (1D np.ndarray): Trajectory lengths, in steps, repeated to fill
        the specified duration.
    """

    length_trajectory_cycle = np.sum(trajectory_lengths) * dt / 60  # in minutes
    num_repeats = int(np.ceil(max_num_minutes / length_trajectory_cycle))

    trajectory_lengths = np.repeat(trajectory_lengths, num_repeats)

    return trajectory_lengths


def get_sigma_in_steps(sigma=0.1, dt=0.03, mean_speed=0.25):
    """
    get_sigma_in_steps()

    Obtain Gaussian sigma value in steps from value in meters.

    Args:
    - sigma (float, optional): Sigma value in meters. Default is 0.1.
    - dt (float, optional): Time step in seconds. Default is 0.03.
    - mean_speed (float, optional): Mean speed in meters per second. Default is 0.25.

    Returns:
    - sigma_in_steps (float): Sigma value in steps.
    """

    sigma_in_steps = sigma / (mean_speed * dt)

    return sigma_in_steps


def get_oscillation_df(firingrates, window=5, amp_thr=0.1):
    """
    get_oscillation_df(firingrates)

    Obtain a DataFrame of single frame oscillations in firing rates. Useful for
    debugging network oscillation.

    Args:
    - firingrates (2D np.ndarray): Firing rates, with shape (frames, neurons).
    - window (int, optional): Window size for identifying oscillations. Default is 5.
    - amp_thr (float, optional): Threshold for mean and median amplitude of oscillation.
        Default is 0.1.

    Returns:
    - oscillation_df (pd.DataFrame): DataFrame of oscillations in firing rates
        with columns:
            - "neuron_idx" (int): Neuron index
            - "start_frame" (int): Start frame of oscillation
            - "stop_frame" (int): Stop frame of oscillation
            - "num_frames" (int): Number of frames in oscillation
            - "mean_amp" (float): Mean amplitude of oscillation
            - "median_amp" (float): Median amplitude of oscillation
    """

    firingrates = np.asarray(firingrates)
    num_fr, num_neurons = firingrates.shape
    fr_diff = np.diff(firingrates, axis=0)

    fr_diff_sign = np.sign(fr_diff)
    fr_diff_sign[::2] = -fr_diff_sign[::2]
    rolling_fr_diff_sign = (
        np.absolute(
            np.mean(sliding_window_view(fr_diff_sign, (window, 1)), axis=(2, 3))
        )
        == 1
    )
    flat_cumsum = gen_util.get_flattened_cumsum(rolling_fr_diff_sign.astype(int).T).T

    n_before = window // 2
    n_after = num_fr - len(flat_cumsum) - n_before
    flat_cumsum = np.concatenate(
        [
            np.zeros((n_before, num_neurons)),
            flat_cumsum,
            np.zeros((n_after, num_neurons)),
        ],
        axis=0,
    )

    oscillation_df = pd.DataFrame(
        columns=[
            "neuron_idx",
            "start_frame",
            "stop_frame",
            "num_frames",
            "mean_amp",
            "median_amp",
        ]
    )
    idx = 0
    for i in range(num_neurons):
        edges = gen_util.get_nonzero_edges(flat_cumsum[:, i], num_consec_thr=window)
        for start, stop in edges.T:
            start -= n_before
            stop += n_after
            oscillation = np.absolute(np.diff(firingrates[start:stop, i]))
            mean_amp = np.mean(oscillation)
            med_amp = np.median(oscillation)
            if mean_amp < amp_thr or med_amp < amp_thr:
                continue

            oscillation_df.loc[idx, "neuron_idx"] = i
            oscillation_df.loc[idx, "start_frame"] = start
            oscillation_df.loc[idx, "stop_frame"] = stop
            oscillation_df.loc[idx, "num_frames"] = stop - start
            oscillation_df.loc[idx, "mean_amp"] = mean_amp
            oscillation_df.loc[idx, "median_amp"] = med_amp

            idx += 1

    return oscillation_df


def get_velocity_update_vector(
    velocity: np.ndarray[tuple[int], np.dtype[np.float64]],
    drift_velocity: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    dt: float = 0.02,
    rotational_velocity: float = 0,
    rotational_velocity_coherence_time: float = 0.08,
    speed_mean: float = 0.08,
    speed_coherence_time: float = 0.08,
    rotational_velocity_std: float = 120 * (np.pi / 180),
    drift_to_random_strength_ratio: float = 1,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    get_velocity_update_vector(velocity)

    Obtain an update vector for a velocity, based on the velocity, rotational
    velocity, and speed.

    Args:
    - velocity (1D np.ndarray): Velocity vector
    - drift_velocity (1D np.ndarray, optional): Drift velocity vector. Default is None.
    - dt (float, optional): Time step. Default is 0.02.
    - rotational_velocity (float, optional): Rotational velocity. Default is 0.
    - rotational_velocity_coherence_time (float, optional): Coherence time for
        rotational velocity. Default is 0.08.
    - speed_mean (float, optional): Mean speed. Default is 0.08.
    - speed_coherence_time (float, optional): Coherence time for speed. Default is 0.08.
    - rotational_velocity_std (float, optional): Standard deviation of rotational
        velocity. Default is 120 * (np.pi / 180).
    - drift_to_random_strength_ratio (float, optional): Ratio of drift to random
        strength. Default is 1.

    Returns:
    - update_vector (1D np.ndarray): Update vector for velocity
    """

    # 1 Stochastically update the direction
    rotational_velocity = rotational_velocity + rutils.ornstein_uhlenbeck(
        dt=dt,
        x=rotational_velocity,
        drift=0,
        noise_scale=rotational_velocity_std,
        coherence_time=rotational_velocity_coherence_time,
    )
    dtheta = rotational_velocity * dt
    velocity = rutils.rotate(velocity, dtheta)

    # 2 Stochastically update the speed
    speed = np.linalg.norm(velocity)
    if speed == 0:  # add tiny velocity in [1,0] direction to avoid nans
        velocity = 1e-8 * np.asarray([1.0, 0.0])
        speed = 1e-8  # type: ignore[assignment]

    normal_variable = rutils.rayleigh_to_normal(speed, sigma=speed_mean)  # type: ignore[arg-type]
    new_normal_variable = normal_variable + rutils.ornstein_uhlenbeck(
        dt=dt,
        x=normal_variable,
        drift=0,
        noise_scale=1,
        coherence_time=speed_coherence_time,
    )
    speed_new = rutils.normal_to_rayleigh(new_normal_variable, sigma=speed_mean)  # type: ignore[arg-type]
    velocity = (speed_new / speed) * velocity

    # Deterministically drift velocity towards the drift_velocity which has been
    # passed into the update function
    if drift_velocity is not None:
        velocity = velocity + rutils.ornstein_uhlenbeck(
            dt=dt,
            x=velocity,
            drift=drift_velocity,  # type: ignore[arg-type]
            noise_scale=0,
            coherence_time=speed_coherence_time / drift_to_random_strength_ratio,
        )

    update_vector = velocity * dt

    return update_vector


def get_trajectory_lengths(
    num_trajectories: int = 100,
    exp_factors: tuple | None = None,
    random_max: int | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """
    get_trajectory_lengths()

    Obtain an array of trajectory lengths.

    Args:
    - num_trajectories (int, optional): Number of trajectory lengths to
        return. Default is 100.
    - exp_factors (tuple, optional): Exponential factors for trajectory
        lengths (inverse scale, rate, minimum). Default is None.
    - random_max (int, optional): Max value for randomizing trajectory
        lengths. Default is None.

    Raises:
    - ValueError: If specifying 'exp_factors', must be of length 3.
    - ValueError: If neither 'exp_factors' nor 'random_max' is specified.

    Returns:
    - trajectory_lengths (1D np.ndarray): Trajectory lengths
    """

    trajectory_lengths = None
    if exp_factors is not None:
        if len(exp_factors) != 3:
            raise ValueError(
                "If specifying 'exp_factors', must be of length 3 "
                "(inverse scale, rate, minimum)."
            )
        inverse_scale, rate, minimum = exp_factors
        shift = minimum - inverse_scale
        trajectory_lengths = (
            inverse_scale * np.exp(np.arange(num_trajectories) / rate) + shift
        ).astype(int)

    if random_max is not None:
        if trajectory_lengths is not None:
            raise ValueError("Cannot specify both 'exp_factors' and 'random_max'.")
        trajectory_lengths = np.random.randint(1, random_max, size=num_trajectories)

    if trajectory_lengths is None:
        raise ValueError("Must specify either 'exp_factors' or 'random_max'.")

    trajectory_lengths = np.maximum(trajectory_lengths, 1)

    return trajectory_lengths


def get_T_shape_env_boundaries(
    prop_env: float = 0.2,
    scale_x: float = 1.0,
    scale_y: float = 1,
    stem_width_as_prop_of_x: float | None = None,
    arm_height_as_prop_of_y: float | None = None,
) -> list[tuple[float, float]]:
    """
    get_T_shape_env_boundaries()

    Obtain boundaries for a T-shape environment.

    Args:
    - prop_env (float, optional): Proportion of the dims of the environment that the
        T-shape stem and arms should span.
    - scale_x (float, optional): Scale of the environment in the x-direction.
    - scale_y (float, optional): Scale of the environment in the y-direction.
    - stem_width_as_prop_of_x (float, optional): Proportion of the width of the
        environment that the stem of the T-shape should span in the x-direction.
        If None, defaults to prop.
    - arm_height_as_prop_of_y (float, optional): Proportion of the height of the
        environment that the T-shape arms should span in the y-direction. If None,
        defaults to prop.

    Returns:
    - boundaries (list): List of tuples of floats [(x1, y1), (x2, y2), ...]
    """

    stem_width_as_prop_of_x = (
        prop_env if stem_width_as_prop_of_x is None else stem_width_as_prop_of_x
    )
    arm_height_as_prop_of_y = (
        prop_env if arm_height_as_prop_of_y is None else arm_height_as_prop_of_y
    )

    for prop_env, dim in [
        (stem_width_as_prop_of_x, "x"),
        (arm_height_as_prop_of_y, "y"),
    ]:
        if prop_env >= 1:
            raise ValueError(
                f"{dim} proportion must be strictly smaller than 1, "
                f"but found {prop_env}."
            )

    # add diff and width
    left_edge_of_T_in_x = 0.5 - stem_width_as_prop_of_x / 2
    top_edge_of_T_in_y = 1 - arm_height_as_prop_of_y

    left_T_boundaries = [
        (0, 1),
        (0, top_edge_of_T_in_y),
        (left_edge_of_T_in_x, top_edge_of_T_in_y),
        (left_edge_of_T_in_x, 0),
    ]

    right_T_boundaries = [(1 - x, y) for x, y in left_T_boundaries]

    boundaries_unscaled = left_T_boundaries + right_T_boundaries[::-1]
    boundaries = [(x * scale_x, y * scale_y) for x, y in boundaries_unscaled]

    return boundaries


def get_num_samples_top_bottom_T_arms(n=10, area="both", top_arms_prop_of_area=0.5):
    """
    get_num_samples_top_bottom_T_arms()

    Obtain the number of samples to obtain from top and bottom arms of a T-shape
    environment.

    Args:
    - n (int, optional): Total number to sample. Default is 10.
    - area (str, optional): Area to sample arms from. Default is "both".
    - top_arms_prop_of_area (float, optional): Proportion of the area that the top
        arms span. Default is 0.5.

    Returns:
    - n_top (int): Number of items to sampletop arms.
    - n_bottom (int): Number of bottom arms.
    """

    if area == "both":
        n_top = int(np.around(top_arms_prop_of_area * n))
        n_bottom = n - n_top
    elif area == "top":
        n_top = n
        n_bottom = 0
    elif area == "bottom":
        n_top = 0
        n_bottom = n
    else:
        raise ValueError(f"Unknown area: {area}")

    return n_top, n_bottom


def sample_from_T_areas(
    n=10,
    extent_x=[0, 1],
    extent_y=[0, 1],
    method="random",
    top=True,
    adjusted_bottom_upper_limit=None,
):
    """
    sample_from_T_areas()

    Sample from the top or bottom sections of a T-shape environment.

    Args:
    - n (int, optional): Number of samples to take. Default is 10.
    - extent_x (list, optional): Extent in the x-direction. Default is [0, 1].
    - extent_y (list, optional): Extent in the y-direction. Default is [0, 1].
    - method (str, optional): Method for sampling. Default is "random".
    - top (bool, optional): Whether sampling from the top or bottom section.
        Default is True.
    - adjusted_bottom_upper_limit (float, optional): Adjusted upper limit computed from
        the top and used for sampling from the bottom section. Default is None.

    Returns:
    - area_positions (2D np.ndarray): Positions of the samples with shape (n, 2).
    - adjusted_bottom_upper_limit (float): Adjusted upper limit to use for the bottom
        section. None if top is False or method is "random".
    """

    if method == "random":
        area_positions = np.zeros((n, 2))
        area_positions[:, 0] = np.random.uniform(*extent_x, size=n)
        area_positions[:, 1] = np.random.uniform(*extent_y, size=n)

    elif method[:7] == "uniform":
        area_size = (extent_x[1] - extent_x[0]) * (extent_y[1] - extent_y[0])

        delta = np.sqrt(area_size / n)

        if top:
            num_y_vals = max(1, int(np.around((extent_y[1] - extent_y[0]) / delta)))
            num_x_vals = int(n // num_y_vals)
            delta_y = delta
            delta_x = (extent_x[1] - extent_x[0]) / num_x_vals

        else:
            num_x_vals = max(1, int(np.around((extent_x[1] - extent_x[0]) / delta)))
            num_y_vals = int(n // num_x_vals)
            delta_x = delta

            if adjusted_bottom_upper_limit is not None:
                delta_y = (adjusted_bottom_upper_limit - extent_y[0]) / (
                    num_y_vals + 0.5
                )
                extent_y[1] = adjusted_bottom_upper_limit - delta_y / 2
            else:
                delta_y = (extent_y[1] - extent_y[0]) / num_y_vals

        if num_x_vals < 2:
            x = np.asarray([extent_x[0] + (extent_x[1] - extent_x[0]) / 2])
        else:
            x = np.linspace(
                extent_x[0] + delta_x / 2, extent_x[1] - delta_x / 2, num_x_vals
            )

        if num_y_vals < 2:
            y = np.asarray([extent_y[0] + (extent_y[1] - extent_y[0]) / 2])
        else:
            y = np.linspace(
                extent_y[0] + delta_y / 2, extent_y[1] - delta_y / 2, num_y_vals
            )

        if top:
            adjusted_bottom_upper_limit = y[0]

        area_positions = np.asarray(np.meshgrid(x, y)).reshape(2, -1).T
        n_uniformly_distributed = area_positions.shape[0]
        if "jitter" in method:
            delta_x = x[0] - extent_x[0]
            area_positions[:, 0] += np.random.uniform(
                -0.45 * delta_x, 0.45 * delta_x, n_uniformly_distributed
            )
            delta_y = y[0] - extent_y[0]
            area_positions[:, 1] += np.random.uniform(
                -0.45 * delta_y, 0.45 * delta_y, n_uniformly_distributed
            )
        n_remaining = n - n_uniformly_distributed
        if n_remaining > 0:
            positions_remaining, _ = sample_from_T_areas(
                n=n_remaining, extent_x=extent_x, extent_y=extent_y, method="random"
            )
            area_positions = np.vstack((area_positions, positions_remaining))

    return area_positions, adjusted_bottom_upper_limit


def get_standard_sigmoid_params(
    min_fr: float = 0.0,
    max_fr: float = 1.0,
    center_0: bool = True,
    mid_x: float | None = None,
    width_x: float | None = None,
) -> dict[str, Any]:
    """
    get_standard_sigmoid_params()

    Obtain a dictionary of parameters for a sigmoid function.

    Args:
    - activation (str, optional): Activation function. Default is "sigmoid".
    - min_fr (float, optional): Minimum firing rate. Default is 0.0.
    - max_fr (float, optional): Maximum firing rate. Default is 1.0.
    - center_0 (bool, optional): Set mid_x if it is None to 0 if True, and
        midpoint otherwise. Default is True.
    - mid_x (float, optional): Midpoint of the sigmoid. Default is None.
    - width_x (float, optional): Width of the sigmoid. Default is None.

    Returns:
    - params (dict): Dictionary of parameters for a sigmoid function with keys:
        - "activation" (str): Activation function
        - "min_fr" (float): Minimum firing rate
        - "max_fr" (float): Maximum firing rate
        - "width_x" (float): Width of the sigmoid
        - "mid_x" (float): Midpoint of the sigmoid
    """

    if width_x is None:
        width_x = 2 * np.log(19)  # so that the sigmoid beta value is 1

    if mid_x is None:
        if center_0:
            mid_x = 0
        else:
            mid_x = min_fr + (max_fr - min_fr) / 2

    params = {
        "activation": "sigmoid",
        "min_fr": min_fr,
        "max_fr": max_fr,
        "width_x": width_x,
        "mid_x": mid_x,
    }

    return params


def get_weighted_object_types(weight_dict, n=10, allow_omit_object_types=False):
    """
    get_weighted_object_types(weight_dict)

    Obtain object types based on a dictionary of weights.

    Args:
    - weight_dict (dict): Dictionary of object types and their weights,
        with keys and values:
        - object_type (int): object weight (float)
    - n (int, optional): Number of object types to obtain. Default is 10.
    - allow_omit_object_types (bool, optional): Whether to allow object types to be
        omitted if not randomly selected. Default is False.

    Returns:
    - object_types (1D np.ndarray): Object types
    """

    object_types, object_weights = list(), list()
    for object_type, object_weight in weight_dict.items():
        if object_weight > 0:
            object_types.append(object_type)
            object_weights.append(object_weight)

    object_weights = np.asarray(object_weights).astype(float)
    object_weights /= object_weights.sum()

    rand_n = n
    fixed_object_types = list()
    if not allow_omit_object_types:
        if n < len(object_types):
            raise RuntimeError(
                "Not enough cells to represent all object types. Must increase "
                "n or set allow_omit_object_types to True."
            )
        rand_n = n - len(object_types)
        fixed_object_types = object_types

    rand_object_types = np.random.choice(
        object_types,
        replace=True,
        size=(rand_n,),
        p=object_weights,
    )

    object_types = np.concatenate((rand_object_types, fixed_object_types))
    np.random.shuffle(object_types)

    return object_types


def estimate_1D_place_cell_density(PCs):
    """
    estimate_1D_place_cell_density(PCs)

    Estimate 1D the place cell density based on the number of place cells and the
    environment the place cells are in.

    Args:
    - PCs (PlaceCells): Place cell object.

    Returns:
    - PC_density_1D (float): Place cell density (PC/m).
    """

    Env = PCs.Agent.Environment
    if Env.D == 1:
        env_scale = Env.scale
        PC_density_1D = PCs.n / env_scale
    else:
        if hasattr(Env, "get_area"):
            PC_density_1D = np.sqrt(PCs.n / Env.get_area())
        else:
            raise NotImplementedError(f"Env type {type(Env)} not supported.")

    return PC_density_1D


def create_weights_dict(weights, steps, t, steps_triggered=None):
    """
    create_weights_dict(weights, steps, t)

    Create a dictionary of weights, steps, and time.

    Args:
    - weights (list): Weights.
    - steps (list): Steps.
    - t (list): Full time array.
    - steps_triggered (int, optional): Steps at which BTSP was triggered. None for
        recorded steps that do not reflect BTSP updates. Default is None.

    Returns:
    - recorded_weights (dict): Dictionary with keys "weights", "steps", "time", and
        "steps_triggered" in which input weights from place cells are recorded,
        along with the step/time at which they were recorded and step at which they
        the BTSP update behind the recorded weight update was triggered, if applicable.
    """

    if len(steps) != len(weights):
        raise ValueError("Length of steps and weights must be the same.")

    steps = np.asarray(steps)
    if steps.max() >= len(t):
        raise ValueError("Steps must be within the range of time.")

    weights = np.asarray(weights)
    if len(steps):
        time = np.asarray(t)[steps]
    else:
        time = np.asarray([])

    weights_dict = {"weights": weights, "steps": steps, "time": time}

    if steps_triggered is not None:
        weights_dict["steps_triggered"] = steps_triggered

    return weights_dict


def assess_firingrate_CC_across_periods(
    firingrates, num_periods=8, plot=True, sub_ax=None
):
    """
    assess_firingrate_CC_across_periods(firingrates)

    Assess the correlation coefficient of firing rates across time periods.

    Args:
    - firingrates (2D np.ndarray): Firing rates, with shape (frames, neurons).
    - num_periods (int, optional): Number of time periods to assess. Default is 8.
    - plot (bool, optional): Whether to plot the correlation coefficient. Default is True.
    - sub_ax (plt.Axes, optional): Axes to plot on. Default is None.

    Returns:
    - gen_CC (2D np.ndarray): General correlation coefficient across time periods.
    if plot:
    - sub_ax (plt.Axes): Subplot on which correlation coefficient matrix is plotted.
    """

    fr = np.asarray(firingrates)

    # get a CC within each time period
    num_fr, num_neurons = fr.shape
    num_samples = num_fr // num_periods
    num_per = num_fr // num_samples
    reshaped_fr = fr[: int(num_samples * num_per)].reshape(
        num_periods, num_samples, num_neurons
    )

    CCs = list()
    for i in range(num_periods):
        CCs.append(np.corrcoef(reshaped_fr[i].T))
    CCs = np.asarray(CCs)

    gen_CC = np.corrcoef(CCs.reshape(num_periods, -1))

    if plot:
        from predhpc.util import plot_util

        sub_ax = plot_util.plot_CC_across_periods(gen_CC, sub_ax=sub_ax)
        return gen_CC, sub_ax
    else:
        return gen_CC
