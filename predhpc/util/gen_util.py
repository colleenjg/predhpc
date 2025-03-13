import copy

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
