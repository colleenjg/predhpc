import copy
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy  # type: ignore[import]
from scipy.optimize import curve_fit

import ratinabox
from ratinabox import utils as rutils  # type: ignore[import]


class TempFigureDirectory:
    def __init__(self, figure_directory):
        self.figure_directory = str(figure_directory)

    def __enter__(self):
        self.original_figure_directory = ratinabox.figure_directory
        ratinabox.figure_directory = self.figure_directory

    def __exit__(self, exc_type, exc_value, traceback):
        ratinabox.figure_directory = self.original_figure_directory


def trim_dict(data_dict):
    """
    trim_dict(data_dict)

    Trim a dictionary of data to remove empty levels.

    Args:
    - data_dict (dict): Dictionary of data.

    Returns:
    - new_dict (dict): Trimmed dictionary of data.
    """

    new_dict = dict()
    for key, data in data_dict.items():
        if isinstance(data, dict):
            trimmed_dict = trim_dict(data)
            if len(trimmed_dict):
                new_dict[key] = trimmed_dict
        elif isinstance(data, list):
            if len(data):
                new_dict[key] = data
        else:
            new_dict[key] = data

    return new_dict


def get_divisors(n: int) -> list[int]:
    """
    get_divisors(n)

    Obtain all divisors of n.

    Args:
    - n (int): Number for which to divisors.

    Returns:
    - divisors (list): List of divisors.
    """

    divisors = list()
    for i in range(1, n + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def get_index_of_closest(data, value=0, method="nearest"):

    if len(data.shape) != 1:
        raise NotImplementedError("Data must be 1D.")

    if method == "nearest":
        index = np.argmin(np.abs(data - value))
    elif method in ["above", "below"]:
        if method == "below":
            sub_indices = np.where(data <= value)[0]
        else:
            sub_indices = np.where(data >= value)[0]
        if len(sub_indices) == 0:
            raise RuntimeError(f"No values {method} {value} in data.")
        index = sub_indices[np.argmin(np.abs(data[sub_indices] - value))]
    else:
        raise NotImplementedError(f"Unknown method {method}.")

    return index


def get_minima_indices(data, min_pts_btw=30, minimum=None, single_direction=False):
    """
    get_minima_indices(data)

    Obtain the indices of the local minima in the data.

    Args:
    - data (1D np.ndarray): Data.
    - min_pts_btw (int, optional): Minimum number of points between minima.
        Default is 30.
    - minimum (float, optional): Minimum value to consider point a minimum.
        Default is None.
    - single_direction (bool, optional): If True, only counts minima in the provided
        direction. Default is False.

    Returns:
    - minimum_indices (1D np.ndarray): Indices of the local minima.
    """

    if minimum is None:
        minimum = np.inf

    curr_min_pt = minimum
    closest_step = None

    minimum_indices = list()
    for p, pt in enumerate(data):
        if pt < curr_min_pt:  # is current distance below previous minimum?, then update
            closest_step = p
            curr_min_pt = pt

        elif closest_step is not None:
            minimum_indices.append(closest_step)
            curr_min_pt = minimum
            closest_step = None

    if not single_direction:

        reverse_minimum_indices = (
            len(data)
            - 1
            - get_minima_indices(
                data[::-1],
                min_pts_btw=0,
                minimum=minimum,
                single_direction=True,
            )
        )

        minimum_indices = sorted(
            set(minimum_indices).intersection(set(reverse_minimum_indices))
        )

    minimum_indices = np.asarray(minimum_indices)

    if len(minimum_indices) and min_pts_btw > 1:
        keep = np.ones_like(minimum_indices, dtype=bool)
        for i in np.argsort(data[minimum_indices]):  # lowest to highest
            if keep[i]:
                keep[
                    np.absolute(minimum_indices - minimum_indices[i]) < min_pts_btw
                ] = False
                keep[i] = True
        minimum_indices = minimum_indices[keep]

    return minimum_indices


def get_flattened_cumsum(data):
    """
    get_flattened_cumsum(data)

    Obtain the cumulative sum of the data, reset whenever it is flat.

    Args:
    - data (np.ndarray): Data, with flattened cumulative sum calculated along the last
        dimension.

    Returns:
    - flattened_cumsum (1D np.ndarray): Cumulative sum of the data, after resetting
        when flat .
    """

    if len(data.shape) > 1:
        flattened_cumsum = list()
        for sub in data:
            flattened_cumsum.append(get_flattened_cumsum(sub))
        flattened_cumsum = np.asarray(flattened_cumsum)

    else:
        flattened_cumsum = np.cumsum(data)
        resets = np.where((data[1:] == 0) * (data[:-1] != 0))[0] + 1
        resets = np.append(resets, len(data))
        subtract = flattened_cumsum[resets[0] - 1]
        for i, reset in enumerate(resets[:-1]):
            subtract_next = flattened_cumsum[resets[i + 1] - 1]
            flattened_cumsum[reset : resets[i + 1]] -= subtract
            subtract = subtract_next

    return flattened_cumsum


def get_nonzero_edges(data, num_consec_thr=5):
    """
    get_nonzero_edges(data)

    Obtain the start and stop edges of nonzero values in the data.

    Args:
    - data (1D np.ndarray): Data.
    - num_consec_thr (int, optional): Number of consecutive values above threshold to
        consider as an edge. Default is 5.

    Returns:
    - edges (2D np.ndarray): Start and stop edges of nonzero values, with shape (2, n).
    """

    start_edges = np.where((data[:-1] == 0) * (data[1:] != 0))[0] + 1
    if data[0] != 0:
        start_edges = np.insert(start_edges, 0, 0)

    stop_edges = np.where((data[:-1] != 0) * (data[1:] == 0))[0] + 1
    if data[-1] != 0:
        stop_edges = np.append(stop_edges, len(data))

    if len(start_edges) != len(stop_edges):
        raise RuntimeError("'start_edges' is not the same length as 'stop_edges'.")

    edges = list()
    for start, stop in zip(start_edges, stop_edges):
        if data[stop - 1] >= num_consec_thr:
            edges.append([start, stop])

    edges = np.asarray(edges).T

    return edges


def get_rayleigh_sigma(mean):
    """
    get_rayleigh_sigma(mean)

    Obtain the sigma of a Rayleigh distribution from a target mean.

    Args:
    - mean (float): Target mean of the Rayleigh distribution.

    Returns:
    - sigma (float): Sigma of the Rayleigh distribution.
    """

    sigma = mean / np.sqrt(np.pi / 2)

    return sigma


def get_rayleigh_mean(sigma):
    """
    get_rayleigh_mean(sigma)

    Compute the mean of a Rayleigh distribution from its sigma parameter.

    Args:
    - sigma (float): Sigma of the Rayleigh distribution.

    Returns:
    - mean (float): Target mean of the Rayleigh distribution.
    """

    mean = sigma * np.sqrt(np.pi / 2)

    return mean


def fit_exp(xs, ys, log=True):
    """
    fit_exp(xs, ys)

    Fit an exponential function to data: f = A * exp(-B * x) + C.

    Args:
    - xs (1D np.ndarray): X values.
    - ys (1D np.ndarray): Y values.

    Returns:
    - parameters (1D np.ndarray): Three fitted parameters [A, B, C].
    - covariance (np.ndarray): Parameter covariance matrix.
    """

    if len(xs) != len(ys):
        raise ValueError("Length of xs and ys must match.")

    def exp_fct(x, A, B, C):
        y = A * np.exp(-B * x) + C
        return y

    parameters, covariance = curve_fit(exp_fct, xs, ys)

    if log:
        print("y = {:.2f} * exp(-{:.2f} * x) + {:.2f}".format(*parameters))

    return parameters, covariance


def sample_gaussian_clipped(n, seed=None, max_abs=2.3):
    """
    sample_gaussian_clipped(n)

    Sample values from a Gaussian distribution, with values clipped to be within
    [-max_abs, max_abs].

    Args:
    - n (int): Number of values to sample.
    - seed (int, optional): Random seed. If None, numpy.random is used. Default is None.
    - max_abs (float, optional): Maximum absolute value of the sampled values. If None,
        no clipping is done. Default is 2.3.

    Returns:
    - noise (1D np.ndarray): Array of values sampled from the Gaussian distribution,
        optionally clipped to be within [-max_abs, max_abs].
    """

    if seed is None:
        rng = np.random
    elif isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        rng = seed

    noise = rng.randn(n)

    if max_abs is not None:
        resample = np.abs(noise) > max_abs

        while resample.any():
            noise[resample] = rng.randn(resample.sum())
            resample = np.abs(noise) > max_abs

    return noise


def get_norm_data(data, axis=-1):
    """
    get_norm_data(data)

    Obtain data min-max normalized.

    Args:
    - data (np.ndarray): Data to normalize.
    - axis (int, optional): Axis along which to normalize. Default is -1.

    Returns:
    - norm_data (np.ndarray): Min-max normalized data.
    """

    data = np.asarray(data)

    data_min = data.min(axis=axis, keepdims=True)
    data_max = data.max(axis=axis, keepdims=True)

    norm_data = (data - data_min) / (data_max - data_min)

    return norm_data


def pad_throughout(indices, pad_prop=0.1, min_val=None, max_val=None):
    """
    pad_throughout(indices)

    Pad indices throughout.

    Args:
    - indices (1D np.ndarray): Indices to pad.
    - pad_prop (float, optional): Proportion of padding to add to the data. Default is 0.1.
    - min_val (int, optional): Minimum value for padding. Default is None.
    - max_val (int, optional): Maximum value for padding (excluded). Default is None.

    Returns:
    - padded_indices (1D np.ndarray): Padded indices.
    """

    if (np.sort(indices) != indices).all():
        raise ValueError("Indices must be sorted.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Indices must be unique.")

    if min_val is None:
        min_val = -np.inf
    if max_val is None:
        max_val = np.inf

    breaks = np.where(np.diff(indices) > 1)[0] + 1
    breaks = np.append(np.insert(breaks, 0, 0), len(indices))
    segments = list()
    for b, val in enumerate(breaks[:-1]):
        segments.append(indices[val : breaks[b + 1]])

    padded_indices = [indices]
    for s, seg in enumerate(segments):
        num_pad_each = int(np.ceil(np.around(len(seg) * pad_prop / 2)))
        if s == 0 and seg[0] > min_val:
            pre_start = max(min_val, seg[0] - num_pad_each)
            padded_indices.append(np.arange(pre_start, seg[0]))
        elif s > 0:
            pre_start = max(segments[s - 1][1], seg[0] - num_pad_each)
            padded_indices.append(np.arange(pre_start, seg[0]))

        if s == len(segments) - 1 and seg[-1] < max_val:
            post_end = min(max_val, seg[-1] + num_pad_each)
            padded_indices.append(np.arange(seg[-1], post_end))
        elif s < len(segments) - 1:
            post_end = min(segments[s + 1][0], seg[-1] + num_pad_each)
            padded_indices.append(np.arange(seg[-1], post_end))

    padded_indices = np.sort(np.unique(np.concatenate(padded_indices)))

    return padded_indices


def get_weights(num_in=10, num_out=10, distr="1to1", loc=1, scale=0):
    """
    get_weights()

    Obtain weights matrix.

    Args:
        num_in (int, optional): Number of input units. Default is 10.
        num_out (int, optional): Number of output units. Default is 10.
        distr (str, optional): Distribution from which to set weights.
            Default is "1to1".
        loc (float, optional): Mean of the distribution. Default is 1.
        scale (float, optional): Standard deviation of the distribution. Default is 0.

    Raises:
        ValueError: If num_in != num_out and distr is "1to1".
        NotImplementedError: If distr is not "1to1" or "randn".

    Returns:
        weights (2D np.ndarray): Weights matrix (out, in).
    """

    if distr == "1to1":
        if num_in != num_out:
            raise ValueError(
                f"If distribution is 1 to 1, num_in ({num_in}) must match num_out ({num_out})."
            )
        weights = np.eye(num_out) * (np.random.randn(num_out) * scale + loc)
    elif distr == "randn":
        weights = np.random.randn(num_out, num_in) * scale + loc
    else:
        raise NotImplementedError(f"Unknown distribution: {distr}.")

    return weights


def get_distance_between_targets_and_predictions(
    targets: np.ndarray[Any, np.dtype[np.float64]],
    predictions: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """
    get_distance_between_targets_and_predictions(targets, predictions)

    Calculate the mean distance between targets and predictions.

    Args:
    - targets (np.ndarray): Target values, with final axis of length 2.
    - predictions (np.ndarray): Predicted values, with same shape as targets.

    Returns:
    - distance (float): Mean distance between targets and predictions.

    Raises:
    - ValueError: If the final axis of targets is not of length 2.
    - ValueError: If the shapes of targets and predictions do not match.
    """

    if targets.shape[-1] != 2:
        raise ValueError("The final axis of `targets` should have length 2.")

    if targets.shape != predictions.shape:
        raise ValueError("The shapes of `targets` and `predictions` must match.")

    distances = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))

    distance = distances.mean()

    return distance


def get_angle_between_vectors(
    v1: np.ndarray[tuple[int], np.dtype[np.float64]],
    v2: np.ndarray[tuple[int], np.dtype[np.float64]],
    directional: bool = False,
) -> float:
    """
    get_angle_between_vectors(v1, v2)

    Obtain angle between two vectors.

    Args:
    - v1 (1D np.ndarray): First vector.
    - v2 (1D np.ndarray): Second vector.
    - directional (bool): Whether to return the directional angle
       (i.e., first vector to second, with same start points: 0 to 360 degrees)
        or non-directional (i.e., between 0 and 180 degrees). Default is False.

    Returns:
    - angle (float): Angle between vectors.
    """

    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    angle = np.rad2deg(np.arccos(np.dot(unit_v1, unit_v2))) % 360
    if not directional:
        angle = angle % 180
        angle = min(angle, 180 - angle)
    return angle


def get_vectors_to_target(positions, target, polar=False, radians=False):
    """
    get_vectors_to_target(positions, target)

    Args:
    - positions (2D np.ndarray): Positions of points, with shape (points, coords (2)):
        [(x1, y1), (x2, y2), ...]
    - target (tuple): Target position.
    - polar (bool, optional): If True, return vectors in polar coordinates.
        Default is False.
    - radians (bool, optional): If True and polar is True, return angles in radians.
        Default is False.

    Returns:
    - vectors (2D np.ndarray): Vectors from positions to target, with shape
        (points, coords (2)): [(x1, y1), (x2, y2),
    """

    vectors = np.asarray(target) - np.asarray(positions)
    if polar:
        rho = np.sqrt(vectors[:, 0] ** 2 + vectors[:, 1] ** 2)
        phi = np.arctan2(vectors[:, 1], vectors[:, 0])
        vectors = np.asarray([rho, phi]).T
        if not radians:
            vectors[:, 1] *= 180 / np.pi

    return vectors


def get_distance_to_target(positions, target):
    """
    get_distance_to_target(positions, target)

    Args:
    - positions (2D np.ndarray): Positions of points, with shape (points, coords (2)):
        [(x1, y1), (x2, y2), ...]
    - target (tuple): Target position.

    Returns:
    - distances (1D np.ndarray): Distances from positions to target.
    """

    vectors = get_vectors_to_target(positions, target)
    distances = np.linalg.norm(vectors, ord=2, axis=1)

    return distances


def get_binned_rates(
    rate,
    rel_pos,
    vel=None,
    num_bins=100,
    part_run=0.2,
    merge=True,
    new_trial=0.2,
    vel_sign_smooth=39,  # 5,
):
    """
    get_binned_rates(rate, rel_pos)

    Bin rates by relative position. Individual runs are identified by large changes
    in position and changes in velocity sign, if velocity is provided.

    Args:
    - rate (2D np.ndarray): Rates, with shape (time, neurons).
    - rel_pos (1D np.ndarray): Relative position.
    - vel (1D np.ndarray, optional): Velocity. Default is None.
    - num_bins (int, optional): Number of bins. Default is 100.
    - part_run (float, optional): Proportion of a run to consider for run merging
        or deletion. Default is 0.2.
    - merge (bool, optional): Whether to merge trials that are too short. If False,
        short trials are deleted. Default is True.
    - new_trial (float, optional): Minimum change in relative position to consider
        a new trial. Default is 0.3.
    - vel_sign_smooth (int, optional): Number of points to smooth velocity sign.
        Default is 5.

    Returns:
    - binned_rate_means (3D np.ndarray): Mean rates binned by relative position, with
        shape (trials, bins, neurons).
    - occupancy (2D np.ndarray): Number of occurrences in each bin, with shape
        (trials, bins).
    """

    if len(rel_pos.shape) != 1:
        raise ValueError("Relative position must be 1D.")
    if rel_pos.min() < 0 or rel_pos.max() > 1:
        raise ValueError("Relative position must be between 0 and 1.")
    if len(rate) != len(rel_pos):
        raise ValueError("Rate and relative position must have the same length.")

    num_neurons = rate.shape[1]

    # bin rate by
    rel_pos_bins = np.linspace(0, 1, num_bins + 1)
    binned_rel_pos = np.digitize(rel_pos, rel_pos_bins) - 1

    change_pts = np.where(np.abs(np.diff(rel_pos)) > new_trial)[0] + 1

    if vel is not None:
        if len(vel) != len(rel_pos):
            raise ValueError(
                "Velocity and relative position must have the same length."
            )
        vel_sign = np.sign(vel)
        if vel_sign_smooth > 1:
            if vel_sign_smooth % 2 == 0:
                raise ValueError("vel_sign_smooth must be odd.")
            vel_sign = np.sign(smooth_data(vel_sign.astype(float), vel_sign_smooth))
        vel_change_pts = np.where(np.diff(vel_sign))[0] + 1
        change_pts = np.concatenate([change_pts, vel_change_pts])

    change_pts = np.sort(np.unique(np.concatenate([[0], change_pts, [len(rel_pos)]])))

    binned_rate_sums = np.zeros((len(change_pts) - 1, num_bins, num_neurons))
    occupancy = np.zeros((len(change_pts) - 1, num_bins))
    prev_used = None
    for i, change_pt in enumerate(change_pts[1:]):
        start = change_pts[i]
        end = change_pt
        curr_num_vals = np.asarray(
            [sum(binned_rel_pos[start:end] == i) for i in range(num_bins)]
        )
        if len(np.where(curr_num_vals)[0]) >= num_bins * part_run:
            i = 0 if prev_used is None else i
            prev_used = i
        elif merge:
            i = 0 if prev_used is None else prev_used  # merge back
        else:
            continue

        occupancy[i] += curr_num_vals

        to_add = np.asarray(
            [
                np.sum(
                    rate[start:end][np.where(binned_rel_pos[start:end] == i)[0]], axis=0
                )
                for i in range(num_bins)
            ]
        )

        binned_rate_sums[i] += to_add

    empty = np.where(occupancy.sum(axis=1) == 0)
    occupancy = np.delete(occupancy, empty, axis=0)
    binned_rate_sums = np.delete(binned_rate_sums, empty, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        binned_rate_means = binned_rate_sums / occupancy.reshape(*occupancy.shape, 1)

    return binned_rate_means, occupancy


def get_filtered_signal(
    f_t1: np.ndarray,
    X_t: np.ndarray | None = None,
    T_t: np.ndarray | None = None,
    filter_tau: float | None = None,
    trend_tau: float | None = None,
    dt: float = 0.03,
    atol: float = 1e-08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    get_filtered_signal(f_t1)

    Obtain filtered signal.

    Args:
    - f_t1 (1D np.ndarray): Input signal to use to update filtered and trend
        signals. Can be all zeros if only filtering previous data (X_t) for an extra
        time step.
    - X_t (1D np.ndarray, optional): Filtered signal to update. If None, f_t1 is used.
        Default is None.
    - T_t (1D np.ndarray, optional): Trend signal to update with same length as X_t.
        If None, it is set to all zeros. Default is None.
    - filter_tau (float, optional): Filter time constant. Default is None.
    - trend_tau (float, optional): Trend time constant. Default is None.
    - dt (float, optional): Time step. Default is 0.03.
    - atol (float, optional): Absolute tolerance for comparing values to 0, to avoid
        overflow. Default is 1e-08.

    Raises:
    - ValueError: If 'tau' is smaller than 'dt'.

    Returns:
    - X_t1 (1D np.ndarray): Updated filtered signal.
    - T_t1 (1D np.ndarray): Updated trend signal.
    """

    def get_tau(tau: float | None, dt: float = 0.03) -> float:
        """
        get_tau()

        Obtain time constant. If None, set to dt.

        Args:
        - tau (float, optional): Time constant. Default is None.
        - dt (float, optional): Time step. Default is 0.03.

        Returns:
        - tau (float): Time constant.
        """

        tau = tau or float(dt)
        if tau < dt:
            raise ValueError(f"'tau' ({tau}) cannot be smaller than dt ({dt}).")

        return tau

    filter_tau = get_tau(filter_tau, dt)
    effective_filter_tau = filter_tau / dt
    filter_alpha = 1 - np.exp(-1 / effective_filter_tau)

    trend_tau = get_tau(trend_tau, dt)
    effective_trend_tau = trend_tau / dt
    trend_alpha = 1 - np.exp(-1 / effective_trend_tau)

    if T_t is None:
        T_t = np.zeros_like(f_t1)

    if X_t is None or not np.isfinite(X_t).all():
        X_t = f_t1

    X_t1 = filter_alpha * f_t1 + (1 - filter_alpha) * (X_t + T_t)
    if trend_tau > dt:
        T_t1 = trend_alpha * (X_t1 - X_t) + (1 - trend_alpha) * T_t
    else:
        T_t1 = T_t

    round_to_zero = np.isclose(X_t1, 0, atol=atol)
    X_t1[round_to_zero] = 0

    round_to_zero = np.isclose(T_t1, 0, atol=atol)
    T_t1[round_to_zero] = 0

    return X_t1, T_t1


def get_relative_filter_tau(filter_tau="half", base_filter_tau=4):
    """
    get_relative_filter_tau()

    Obtain a filter tau relative to a base value.

    Args:
    - filter_tau (str, optional): Filter tau to compute. Default is "half".
    - base_filter_tau (float, optional): Base filter tau. Default is 4.

    Returns:
    - filter_tau (float): Calculated BTSP filter tau
    """

    if isinstance(filter_tau, str):
        if filter_tau == "half":
            div = 2
        elif filter_tau == "third":
            div = 3
        elif filter_tau == "equal":
            div = 1
        else:
            raise ValueError(f"Invalid post_BTSP_filter_tau value: {filter_tau}")
        filter_tau = base_filter_tau / div

    return filter_tau


def get_exponential(
    filter_tau: float | None = None,
    trend_tau: float | None = None,
    dt: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    get_exponential()

    Obtain an exponential signal using specific filtering parameters.

    Args:
    - filter_tau (float, optional): Filter time constant. Default is None.
    - trend_tau (float, optional): Trend time constant. Default is None.
    - dt (float, optional): Time step. Default is 0.03.

    Returns:
    - exponential (np.ndarray): Exponential signal.
    """

    X_ts = [np.ones(1)]
    T_t = np.zeros(1)
    while not np.isclose(X_ts[-1], 0):
        X_t1, T_t = get_filtered_signal(
            f_t1=np.zeros(1),
            X_t=X_ts[-1],
            T_t=T_t,
            filter_tau=filter_tau,
            trend_tau=trend_tau,
            dt=dt,
        )
        X_ts.append(X_t1)

    X_ts = np.asarray(X_ts).reshape(-1)

    return X_ts


def get_pre_post_exponential(
    filter_tau: float | None = None,
    trend_tau: float | None = None,
    dt: float = 0.03,
    post_filter_tau: float | str | None = None,
    post_trend_tau: float | None = None,
):
    """
    get_pre_post_exponential()

    Obtain a pre and post combined exponential signal using specific filtering
    parameters.

    Args:
    - filter_tau (float, optional): Filter time constant. Default is None.
    - trend_tau (float, optional): Trend time constant. Default is None.
    - dt (float, optional): Time step. Default is 0.03.
    - post_filter_tau (float, str, optional): Post-filter time constant. Default is None.
    - post_trend_tau (float, optional): Post-trend time constant. Default is None.

    Returns:
    - pre_post_exp (np.ndarray): Pre and post combined exponential signal.
    """

    pre_exp = get_exponential(filter_tau, trend_tau, dt=dt)[::-1]

    if post_filter_tau is None:
        post_exp = np.asarray([pre_exp[-1]])
    else:
        post_filter_tau = get_relative_filter_tau(
            post_filter_tau, base_filter_tau=filter_tau
        )
        post_exp = get_exponential(post_filter_tau, post_trend_tau, dt=dt)

    pre_post_exp = np.concatenate([pre_exp, post_exp[1:]])
    peak_pt = len(pre_exp) - 1
    pre_post_exp[peak_pt] = np.mean([pre_exp[-1], post_exp[0]])
    pre_post_exp = pre_post_exp / pre_post_exp.max()

    return pre_post_exp


def get_exponential_AUC(
    filter_tau: float | None = None,
    trend_tau: float | None = None,
    dt: float = 0.03,
    post_filter_tau: float | str | None = None,
    post_trend_tau: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    get_exponential_AUC()

    Compute area under the curve for a signal starting at 1, using specific filtering
    parameters.

    Args:
    - filter_tau (float, optional): Filter time constant. Default is None.
    - trend_tau (float, optional): Trend time constant. Default is None.
    - dt (float, optional): Time step. Default is 0.03.
    - post_filter_tau (float, str, optional): Post-filter time constant. Default is None.
    - post_trend_tau (float, optional): Post-trend time constant. Default is None.

    Returns:
    - AUC (float): Area under the curve of the exponential signal
    """

    X_ts = get_pre_post_exponential(
        filter_tau=filter_tau,
        trend_tau=trend_tau,
        dt=dt,
        post_filter_tau=post_filter_tau,
        post_trend_tau=post_trend_tau,
    )
    AUC = np.sum(X_ts)

    return AUC


def shortest_distances_from_points_to_lines(
    positions: np.ndarray[tuple[int, int], np.dtype[np.float64]] | list,
    vectors: np.ndarray[tuple[int, int, int], np.dtype[np.float64]] | list,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    shortest_distances_from_points_to_lines(positions, vectors)

    Calculate the shortest distances between points and lines.

    Args:
    - positions (2D np.ndarray): Positions of points, with shape (points, coords (2)):
        [(x1, y1), (x2, y2), ...]
    - vectors (3D np.ndarray): Vectors defining lines, with shape (vectors, coords (2)):
        [[(x11, y11), (x12, y12)], [(x21, y21), (x22, y22)], ...]

    Returns:
    - closest_distances (2D np.ndarray): Shortest distances between each point and line
       (points, vectors)
    """

    positions = np.asarray(positions)
    if len(positions.shape) == 1:  # expand if only one point is provided
        positions = np.expand_dims(positions, axis=0)

    vectors = np.asarray(vectors)
    if len(vectors.shape) == 2:  # expand if only one vector is provided
        vectors = np.expand_dims(vectors, axis=0)

    # returns (points, vectors, coords)
    shortest_vectors = rutils.shortest_vectors_from_points_to_lines(positions, vectors)

    closest_distances = np.linalg.norm(shortest_vectors, ord=2, axis=-1)

    return closest_distances


def rotate_to(
    in_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    in_basis: tuple[int | float, int | float] = (1, 0),
    out_basis: tuple[int | float, int | float] = (-1, 0),
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    rotate_to(in_vector)

    Rotate a vector to a new basis.

    Args:
    - in_vector (1D np.ndarray): Vector to rotate.
    - in_basis (1D np.ndarray): Basis to rotate from.
    - out_basis (1D np.ndarray): Basis to rotate to.

    Returns:
    - out_vector (1D np.ndarray): Rotated vector.
    """

    in_vector = np.asarray(in_vector)

    # get angle wrt to basis
    in_angle = np.arctan2(in_vector[1], in_vector[0]) - np.arctan2(
        in_basis[1], in_basis[0]
    )
    in_norm = np.linalg.norm(in_vector, ord=2)

    # rotate to out basis
    out_angle = np.arctan2(out_basis[1], out_basis[0]) + in_angle
    out_vector = np.asarray([np.cos(out_angle) * in_norm, np.sin(out_angle) * in_norm])

    return out_vector


def rescale_data(data, target_range=None, pad_prop=0.1):
    """
    rescale_data(data)

    Rescale data to a target range, with padding.

    Args:
    - data (np.ndarray): Data to rescale.
    - target_range (tuple, optional): Target range for rescaling. If None, no rescaling
        is done. Default is None.
    - pad_prop (float, optional): Proportion of padding to add to the data. Default is 0.1.

    Raises:
    - ValueError: If target_range is not a 2-element tuple.

    Returns:
    - scaled_data (np.ndarray): Data rescaled, optionally to the target range, with padding.
    """

    min_val, max_val = data.min(), data.max()
    span = max_val - min_val
    scaled_data = (data - min_val) / span  # norm
    scaled_data = scaled_data * (1 - pad_prop) + pad_prop / 2  # pad

    if target_range is not None:  # rescale to target range
        if len(target_range) != 2:
            raise ValueError("Target range must be a 2-element tuple.")
        targ_min, targ_max = target_range
        scaled_data = scaled_data * (targ_max - targ_min) + targ_min

    return scaled_data


def get_rebin_factor(orig_num_bins=150, new_num_bins=None):
    """
    get_rebin_factor(orig_num_bins, new_num_bins)

    Obtain the factor by which to rebin data, given the original and new number of bins.

    Args:
    - orig_num_bins (int, optional): Original number of bins. Default is 150.
    - new_num_bins (int, optional): New number of bins. If None, the factor is set to 1.
        Default is None.

    Raises:
    - ValueError: If new_num_bins is greater than orig_num_bins or if new_num_bins is not a
        factor of orig_num_bins.

    Returns:
    - factor (int): Factor by which to rebin the data.
    """

    if new_num_bins is None:
        factor = 1
    else:
        if new_num_bins > orig_num_bins:
            raise ValueError(
                "Number of position bins requested cannot exceed original "
                f"number of position bins in data, i.e. {orig_num_bins}."
            )
        factor = orig_num_bins // new_num_bins
        if factor != (orig_num_bins / new_num_bins):
            breakpoint()
            raise ValueError(
                f"Number of position bins {new_num_bins} requested must "
                "be a factor of the original number of bins in the data, "
                f"i.e. {orig_num_bins}."
            )

    return factor


def index_array_2D(array_to_index, index_array):
    """
    index_array_2D(array_to_index, index_array)

    Index an array using another 2D array.

    Args:
    - array_to_index (np.ndarray): Array to index, with shape (d1, ...).
    - index_array (2D np.ndarray): Array of indices, with shape (x1, x2). Used to
        index the first axis, preserving any additional axes.

    Returns:
    - indexed_array (np.ndarray): Reindexed array with shape (x1, x2, ...).
    """

    nan_mask = np.zeros_like(index_array).astype(bool)
    nan_mask[np.isnan(index_array)] = True

    index_array = copy.deepcopy(index_array)
    index_array[nan_mask] = 0
    index_array = index_array.astype(int)

    indexed_array = array_to_index[np.arange(index_array.max() + 1)][index_array]

    indexed_array[nan_mask] = np.nan

    return indexed_array


def interpolate_data(data):
    """
    interpolate_data(data)

    Interpolate missing values in data.

    Args:
    - data (1 or 2D np.ndarray): Data to interpolate. Interpolation is done along the
        last axis.

    Returns:
    - interp_data (1 or 2D np.ndarray): Data with missing values interpolated.
    """

    def interpolate_1D(data_1D):
        nan_idx = np.isnan(data_1D)
        interp_vals = np.interp(
            nan_idx.nonzero()[0], (~nan_idx).nonzero()[0], data_1D[~nan_idx]
        )
        interp_data_1D = data_1D.copy()
        interp_data_1D[nan_idx] = interp_vals

        return interp_data_1D

    if len(data.shape) == 2:
        interp_data = np.asarray([interpolate_1D(sub_data) for sub_data in data])

    else:
        interp_data = interpolate_1D(data)

    return interp_data


def smooth_data(data, k=5, handle_nans=False):
    """
    smooth_data(data)

    Smooth data using a moving average filter.

    Args:
    - data (1, 2D or 3D np.ndarray): Data to smooth. Smoothing is done along the last
        axis.
    - k (int, optional): Number of frames over which to compute moving average filter.
        Default is 5.
    - handle_nans (bool, optional): If True, set NaN values to 0 before smoothing and
        then return to NaN, if they are still 0s after smoothing. Default is False.

    Returns:
    - smoothed_data (1 or 2D np.ndarray): Data smoothed using a moving average filter.
    """

    def convolve1D(data1D):
        return np.convolve(data1D, np.full(k, 1 / k), mode="same")

    if handle_nans:
        nan_mask = np.isnan(data)
        data[nan_mask] = 0

    if len(data.shape) == 1:
        smoothed_data = convolve1D(data)

    elif len(data.shape) == 2:
        smoothed_data = np.asarray([convolve1D(data1D) for data1D in data])

    elif len(data.shape) == 3:
        smoothed_data = np.asarray(
            [[convolve1D(data1D) for data1D in data2D] for data2D in data]
        )

    else:
        raise ValueError("Data must be 1, 2, or 3D.")

    if handle_nans:
        smoothed_data[nan_mask * (smoothed_data == 0)] = np.nan

    return smoothed_data


def get_2D_Gaussian_kernel(sigma, aperture_prop=8, max_aperture=np.inf):
    """
    Create a 2D Gaussian kernel.

    Parameters:
    - sigma (float): Standard deviation of the Gaussian.
    - aperture_prop (float): Aperture size to use relative to the standard
        deviation. When the Gaussian kernel is used with scipy's fft_convolve(),
        this aperture size appears to match the effective aperture size of
        scipy's gaussian_filter(). Default is 8.
    - max_aperture (float): Maximum aperture size to use. Default is np.inf.

    Returns:
    - kernel (2D np.ndarray): 2D Gaussian kernel.
    """

    aperture = int(np.ceil(min(sigma * aperture_prop, max_aperture)) // 2 * 2 + 1)

    x = np.linspace(-(aperture // 2), aperture // 2, aperture)
    xx, yy = np.meshgrid(x, x)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    return kernel


def half_width_proportion_to_kernel_skew(half_width_proportion: float = 1 / 4) -> float:
    """
    half_width_proportion_to_kernel_skew()

    Approximately calculate the skew of the Gaussian kernel skew given the proportion
    of the right half-width to the left half-width.

    Args:
    - half_width_proportion (float, optional): Right / left width half max.
        Default is 1/4.

    Returns:
    - kernel_skew (float): Skew of the Gaussian kernel
    """

    kernel_skew = -np.log(1 / half_width_proportion) * 6.5

    return kernel_skew


def get_skewed_Gaussian_kernel(
    width_at_half_max: float = 1.5,
    half_width_proportion: float = 1 / 4,
    atol: float = 1e-6,
    dt: float = 0.03,
    num_estimate_pts: int = 5000,
) -> tuple[np.ndarray[tuple[int], np.dtype[np.float64]], np.int64]:
    """
    get_skewed_Gaussian_kernel()

    Obtain a skewed Gaussian kernel.

    Args:
    - width_at_half_max (float, optional): Width at half max of the kernel.
        Default is 1.5.
    - half_width_proportion (float, optional): Right / left width half max.
        Default is 4.
    - atol (float, optional): Absolute tolerance for determining the edges of
        the distribution. Default is 1e-6.
    - dt (float, optional): Time step size in seconds. Default is 0.03.
    - num_estimate_pts (int, optional): Number of points to use for estimating the
        kernel parameters. Default is 5000.

    Returns:
    - skewed_Gaussian_kernel (1D np.ndarray): Skewed Gaussian kernel
    - max_value_idx (float): Index of the maximum value of the kernel
    """

    if width_at_half_max < 0:
        raise ValueError("Width at half max must be positive.")

    kernel_skew = half_width_proportion_to_kernel_skew(half_width_proportion)

    # create a basic kernel to estimate the target scale parameter
    base_scale = 1
    num_estimate_pts *= np.absolute(max(1, int(kernel_skew)))
    base_pts = np.linspace(-10, 10, num_estimate_pts)
    base_pdf = scipy.stats.skewnorm.pdf(base_pts, kernel_skew, 0, base_scale)

    max_value_idx = np.argmax(base_pdf)
    half_max = base_pdf[max_value_idx] / 2
    width_at_half_max_values = list()
    for i, sl in enumerate([slice(0, max_value_idx), slice(max_value_idx, None)]):
        width_at_half_max_idx = np.argmin(np.absolute(base_pdf[sl] - half_max))
        actual_width_at_half_max = base_pts[width_at_half_max_idx + i * max_value_idx]
        width_at_half_max_values.append(
            np.absolute(base_pts[max_value_idx] - actual_width_at_half_max)
        )
    max_width_at_half_max_value = np.max(width_at_half_max_values)
    ratio_to_requested_proportion = half_width_proportion / max_width_at_half_max_value

    # create the kernel
    scale = base_scale * ratio_to_requested_proportion
    num_pts = int(20 * scale / dt)
    pts = np.linspace(-10 * scale, 10 * scale, num_pts)
    pdf = scipy.stats.skewnorm.pdf(pts, kernel_skew, 0, scale)

    near_zero = np.isclose(pdf, 0, atol=atol)
    max_value_idx = np.argmax(pdf)
    kernel_edge_idxs = list()
    for i, sl in enumerate([slice(0, max_value_idx), slice(max_value_idx, None)]):
        half_zero_idxs = np.where(near_zero[sl])[0]
        if not len(half_zero_idxs):
            raise NotImplementedError(
                "Range values selected do not allow edges of the distribution to "
                "be identified."
            )
        if i == 0:
            kernel_edge_idx = half_zero_idxs[-1]
        elif i == 1:
            kernel_edge_idx = half_zero_idxs[0] + max_value_idx
        else:
            raise ValueError("`i` should be 0 or 1.")
        kernel_edge_idxs.append(kernel_edge_idx)

    skewed_Gaussian_kernel = pdf[kernel_edge_idxs[0] : kernel_edge_idxs[1]]
    max_value_idx = np.int64(np.argmax(skewed_Gaussian_kernel))

    skewed_Gaussian_kernel /= skewed_Gaussian_kernel.sum()

    return skewed_Gaussian_kernel, max_value_idx


def get_CC_sorter(CC, cut_off_thr=70, log=False):
    """
    get_CC_sorter(CC)

    Sorts items by correlation coefficient.

    Args:
        - CC (np.ndarray): correlation coefficients
        - cut_off_thr (float): threshold for each correlation group. If < 1,
            interpreted as a correlation coefficient value. Otherwise, interpreted as
            percentile across correlation coefficients. Default is 70.
        - log (bool): Whether to print the number of items in each group.
            Default is False.

    Returns:
        - sorter (np.ndarray): sorted indices
        - groups (dict): dictionary with group information
    """

    triu_idx = np.triu_indices(len(CC), k=1)
    CC_data = CC[triu_idx]

    sorter = list()
    n = 0
    groups = dict()
    for idx in np.argsort(CC_data)[::-1]:
        roi_idx = triu_idx[0][idx]
        pair_idx = triu_idx[1][idx]
        if roi_idx in sorter or pair_idx in sorter:
            continue

        sub_sorter = np.argsort(CC[roi_idx])[::-1]
        if cut_off_thr >= 1:
            thr = np.percentile(CC[roi_idx], q=cut_off_thr)
        else:
            thr = cut_off_thr
        for pair_idx in sub_sorter:
            if CC[roi_idx, pair_idx] < thr:
                break
            if pair_idx != roi_idx and pair_idx not in sorter:
                if roi_idx not in sorter:
                    sorter.append(roi_idx)
                sorter.append(pair_idx)
                last_thr = CC[roi_idx, pair_idx]

        num_in_grp = len(sorter) - n
        if num_in_grp > 0:
            groups[len(groups)] = (num_in_grp, last_thr)
        n = len(sorter)

    missing_idxs = [idx for idx in np.arange(len(CC)) if idx not in sorter]
    sorter = np.asarray(sorter + missing_idxs)

    if log:
        print_str = ", ".join(
            [
                f"{num_in_grp}/({len(CC)}) (>={last_thr:.2f})"
                for num_in_grp, last_thr in groups.values()
            ]
        )
        print(print_str)

    return sorter, groups
