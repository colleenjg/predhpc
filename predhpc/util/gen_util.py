import copy
from datetime import datetime
from pathlib import Path
import random
import time

import numpy as np

import ratinabox


class TempFigureDirectory:
    def __init__(self, figure_directory):
        self.figure_directory = str(figure_directory)

    def __enter__(self):
        self.original_figure_directory = ratinabox.figure_directory
        ratinabox.figure_directory = self.figure_directory

    def __exit__(self, exc_type, exc_value, traceback):
        ratinabox.figure_directory = self.original_figure_directory


def seed_all(seed=None):
    """
    seed_all()

    Set the random seed for all libraries.

    Args:
    - seed (int, optional): Random seed. If None, librairies are not seeded.
        Default is None.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def delete_np_dict(filepath):
    """
    delete_np_dict(filepath)

    Delete a saved numpy dictionary file.

    Args:
    - filepath (str or Path): Path to the numpy dictionary file to delete.
    """

    filepath = Path(filepath)
    if filepath.is_file():
        print(f"Deleting {filepath}.")
        filepath.unlink()


def load_np_dict(filepath):
    """
    load_np_dict(filepath)

    Load a dictionary saved  in numpy file.

    Args:
    - filepath (str or Path): Path to the numpy dictionary file to load.

    Returns:
    - data_dict (dict): Loaded numpy dictionary.
    """

    filepath = Path(filepath)
    data_dict = None
    if filepath.is_file():
        print(f"Loading data from {filepath}.")
        data_dict = dict(np.load(filepath))
    return data_dict


def save_np_dict(filepath, data_dict):
    """
    save_np_dict(filepath, data_dict)

    Save a dictionary to a numpy file.

    Args:
    - filepath (str or Path): Path to the numpy dictionary file to save.
    - data_dict (dict): Dictionary to save.
    """

    filepath = Path(filepath)
    if filepath.is_file():
        print(f"Overwriting {filepath}.")
    else:
        print(f"Saving data to {filepath}.")

    np.savez(filepath, **data_dict)


def get_filtered_np_data_dict(
    data_dict, filter_key, values=list(), skip_keys=list(), raise_missing=True
):
    """
    get_filtered_np_data_dict(data_dict, filter_key)

    Obtain a filtered version of a numpy data dictionary to retain for each key only
    the indices corresponding to specified values for a specified filtering key.

    Args:
    - data_dict (dict): Dictionary to filter.
    - key (str): Key to filter on.
    - values (list, optional): List of values to keep. Default is empty list.
    - skip_keys (list, optional): List of keys to skip. Default is empty list.
    - raise_missing (bool, optional): If True, raise an error if a value is not found
        in the filter key. Default is True.

    Returns:
    - filtered_dict (dict): Filtered dictionary.
    """

    if filter_key not in data_dict.keys():
        raise KeyError(f"Filter key '{filter_key}' not found in data dictionary.")

    indices = list()
    for val in values:
        if val not in data_dict[filter_key] and raise_missing:
            raise RuntimeError(
                f"{val} value not found in data dictionary under '{filter_key}'."
            )
        idxs = np.where(data_dict[filter_key] == val)[0]
        indices.extend(idxs.tolist())

    num = len(data_dict[filter_key])

    data_dict = data_dict.copy()
    for key in list(data_dict.keys()):
        if key in skip_keys:
            continue
        if len(data_dict[key]) != num:
            raise RuntimeError(
                f"Expected all keys to filter to have the same length as '{filter_key}' "
                f"({num}), but found {len(data_dict[key])} for '{key}'."
            )
        data_dict[key] = data_dict[key][indices]

    return data_dict


def get_proportion_edges(data, prop=0.5):
    """
    get_proportion_edges(data)

    Get the value at a specific proportion of the data's range.

    Args:
    - data (np.ndarray): 1D array of data.
    - prop (float, optional): Proportion of the data's range to return.
        Default is 0.5 (median).

    Returns:
    - prop_val (float): Value at the specified proportion of the data's range.
    """

    data = np.asarray(data)
    extent = data.max() - data.min()
    prop_val = data.min() + extent * prop

    return prop_val


def get_value_index_range(data, value, single_range_only=False):
    """
    get_value_index_range(data, value)

    Obtain the start and end indices of a specific value in the data.

    Args:
    - data (1D np.ndarray): Data.
    - value (float): Value for which to obtain the index range.
    - single_range_only (bool, optional): If True, only return the first contiguous
        range of indices. Default is False.

    Returns:
    - index_ranges (list): List of index ranges for the specified value, or single
        range if single_range_only is True. End of each range is exclusive.
    """

    data = np.asarray(data)
    indices = np.where(data == value)[0]

    if len(indices) == 0:
        index_ranges = list()

    else:
        index_ranges = list()
        start_idx = 0
        diffs = np.diff(indices)
        for i, diff in enumerate(diffs):
            if diff == 1:
                continue
            else:
                index_ranges.append((indices[start_idx], indices[i] + 1))
                start_idx = i + 1
        index_ranges.append((indices[start_idx], indices[-1] + 1))

    if single_range_only:
        if len(index_ranges) == 1:
            index_ranges = index_ranges[0]
        else:
            raise RuntimeError(
                f"Expected exactly one range, but found {len(index_ranges)} ranges."
            )

    return index_ranges


def get_duration_str(start_time, log=False):
    """
    get_duration_str(start_time)

    Report time elapsed since start time.

    Args:
    - start_time (float): Start time.
    - log (bool, optional): If True, print the time elapsed. Default is False.

    Returns:
    - time_str (str): Time elapsed as a string.
    """

    end_time = time.perf_counter()

    time_sec = end_time - start_time
    if time_sec / 3600 > 1.5:
        time_hour = int(time_sec // 3600)
        time_min = time_sec / 60 - time_hour * 60
        time_str = f"{time_hour}h {time_min:.2f}m"
    elif time_sec / 60 > 1.5:
        time_min = int(time_sec // 60)
        time_sec = time_sec - time_min * 60
        time_str = f"{time_min}m {time_sec:.2f}s"
    else:
        time_str = f"{time_sec:.2f} s"

    if log:
        print(f"Time elapsed: {time_str}.")

    return time_str


def get_short_time_str():
    """
    get_short_time_str()

    Obtain the time as a short string (HHMM).

    Returns:
    - short_time_str (str): Current time as a short string.
    """

    include_str = "%H%M"
    date_time_str = datetime.now().strftime(include_str)

    return date_time_str


def get_date_time_str(include_time=True):
    """
    get_date_time_str()

    Obtain the current date, and optionally time, as a string:
    - date only: "YY_MM_DD"
    - date and time: "YY-MM-DD_HH-MM-SS"

    Args:
    - include_time (bool, optional): If True, include time in the string.
        Default is True.

    Returns:
    - date_time_str (str): Current date and time as a string.
    """

    if include_time:
        include_str = "%y-%m-%d_%H-%M-%S"
    else:
        include_str = "%y_%m_%d"

    date_time_str = datetime.now().strftime(include_str)
    return date_time_str


def get_save_path(save_name, direc=None, create_dir=True, log=True):
    """
    get_save_path()

    Obtain the path to the save directory, which is the current working directory.

    Returns:
    - save_path (str): Path to the save directory.
    """

    today = get_date_time_str(include_time=False)
    now = get_short_time_str()

    stem = Path(save_name).stem
    suffix = Path(save_name).suffix
    save_name = f"{stem}_{now}{suffix}"

    save_path = Path(today, save_name)
    if direc is not None:
        save_path = Path(direc, save_path)

    if create_dir:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    if log:
        print(f"Saving to: {save_path}.")

    return save_path


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


def get_rounded_linspace(start, end, n, round=9):
    """
    get_rounded_linspace(start, end, n)

    Obtain a linearly spaced array of values, rounded to a specified number of decimal
    places.

    Args:
    - start (float): Start value.
    - end (float): End value.
    - n (int): Number of values to generate.
    - round (int, optional): Number of decimal places to round to. Default is 9.

    Returns:
    - values (np.ndarray): Linearly spaced array of values.
    """

    values = np.linspace(start, end, n)
    if round is not None:
        values = np.around(values, round)
    return values


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

    Obtain the start and end edges of nonzero values in the data.

    Args:
    - data (1D np.ndarray): Data.
    - num_consec_thr (int, optional): Number of consecutive values above threshold to
        consider as an edge. Default is 5.

    Returns:
    - edges (2D np.ndarray): Start and end edges of nonzero values, with shape (2, n).
    """

    start_edges = np.where((data[:-1] == 0) * (data[1:] != 0))[0] + 1
    if data[0] != 0:
        start_edges = np.insert(start_edges, 0, 0)

    end_edges = np.where((data[:-1] != 0) * (data[1:] == 0))[0] + 1
    if data[-1] != 0:
        end_edges = np.append(end_edges, len(data))

    if len(start_edges) != len(end_edges):
        raise RuntimeError("'start_edges' is not the same length as 'end_edges'.")

    edges = list()
    for start, end in zip(start_edges, end_edges):
        if data[end - 1] >= num_consec_thr:
            edges.append([start, end])

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


def spread_data(x, y, max_spread=0.5):
    """
    spread_data(x, y)

    Spread the data in y based on the duplicated values of x and y.

    Args:
    - x (1D np.ndarray): Array of x values.
    - y (1D np.ndarray): Array of y values.
    - max_spread (float, optional): Maximum spread of the y values. Default is 0.5.

    Returns:
    - y (1D np.ndarray): Array of y values with spread applied.
    """

    spread_y = np.zeros_like(y)
    for n in np.unique(x):
        x_mask = x == n
        if x_mask.sum() == 1:
            continue
        for y_val in np.unique(y[x_mask]):
            mask = (y == y_val) & x_mask
            if mask.sum() == 1:
                continue
            vals = np.arange(mask.sum()) - (mask.sum() - 1) / 2
            spread_y[mask] = vals

    max_val = np.absolute(spread_y).max()
    if max_val > 0:
        spread_y = spread_y / max_val * max_spread

    y = y + spread_y

    return y


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
