import numpy as np
import scipy  # type: ignore[import]
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


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


def smooth_circularly(data, k=3):
    """
    smooth_circularly(data)

    Perform circular smoothing on 1D data with a kernel of size k.
    The kernel is a uniform kernel of size k, i.e. [1/k, 1/k, ..., 1/k].

    Args:
    - data (1D np.ndarray): Data to convolve.
    - k (int, optional): Size of the kernel. Default is 3.

    Raises:
    - ValueError: If k is larger than the data length.
    - NotImplementedError: If data is not 1D.

    Returns:
    - smoothed (1D np.ndarray): Circularly smoothed data.
    """

    if k != int(k):
        raise ValueError("Kernel size must be an integer.")

    k = int(k)

    if k < 0:
        raise ValueError("Kernel size must be a positive integer.")
    if k == 0 or k == 1:
        return data

    if len(data.shape) != 1:
        raise NotImplementedError("Circular convolution only implemented for 1D data.")

    n = len(data)
    if k >= n:
        raise ValueError(f"Kernel size {k} cannot be larger than the data ({n}).")
    if k % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    kernel = np.ones(k) / k

    shift = int(k // 2)
    smoothed = np.convolve(np.tile(data, 3), kernel)[n + shift : 2 * n + shift]

    return smoothed


def get_partway_idx_of_half_max_crossing_on_right(signal):
    """
    get_partway_idx_of_half_max_crossing_on_right(signal)

    Find the index of the point at which the signal first crosses the half peak to on
    the right side. Index is a decimal value that is partway between the points around
    the half max crossing.

    Args:
    - signal (np.ndarray): 1D array of signal values.

    Returns:
    - partway_idx (float or None): Index of the point where the half max is first
        crossed, to the right of the peak. If no crossing is found, returns np.nan.
    """

    peak_idx = np.argmax(signal)
    half_max = get_half_max(signal)

    right_side = np.concatenate((signal[peak_idx:], signal[:peak_idx])) - half_max

    post_idxs = np.where(right_side < 0)[0]
    if len(post_idxs) == 0:
        return np.nan

    post_idx = post_idxs[0]
    pre_idx = post_idx - 1

    # Find a decimal index under linear interpolation between points
    pre_diff = right_side[pre_idx]
    post_diff = right_side[post_idx]

    pre_idx = (peak_idx + pre_idx) % len(signal)
    partway_idx = pre_idx + pre_diff / (pre_diff + np.absolute(post_diff))

    return partway_idx


def get_interp_x(x, partway_idx, max_x=None):
    """
    get_interp_x(x, partway_idx)

    Obtain the x value of the partway index interpolated from the x provided.

    Args:
    - x (1D np.ndarray): Sorted x.
    - partway_idx (float): Index of the point where the half max is first crossed.
    - max_x (float, optional): Maximum x value to consider for interpolation.

    Returns:
    - interp_x (float): Position corresponding to the partway index interpolated
        from the x values provided.
    """

    if partway_idx >= len(x):
        partway_idx = partway_idx % len(x)

    idx = int(partway_idx)

    left_pos = x[idx]
    if idx + 1 == len(x):
        if max_x is None:
            right_pos = left_pos + x[0]
        else:
            right_pos = max_x + x[0]

    else:
        right_pos = x[idx + 1]

    interp_x = left_pos + (right_pos - left_pos) * (partway_idx - idx)

    return interp_x


def get_half_max(signal):
    """
    get_half_max(signal)

    Compute the half maximum of a signal.

    Args:
    - signal (1D np.ndarray): Signal values.

    Returns:
    - half_max (float): Half maximum of the signal.
    """

    signal = np.asarray(signal)

    half_max = signal.min() + (signal.max() - signal.min()) / 2

    return half_max


def compute_FWHM(signal, x=None, k=1, max_x=None, return_edges=False):
    """
    compute_FWHM(signal)

    Compute the full width at half maximum (FWHM) of a signal.

    Args:
    - signal (1D np.ndarray): Signal values.
    - x (1D np.ndarray): X axis. If None, indices are used.
    - k (int, optional): Kernel size for circular smoothing. Default is 1.
    - max_x (float, optional): Maximum x value to consider for FWHM. If None,
        the maximum value of x is used. Default is None.
    - return_positions (bool, optional): If True, return the positions defining the
        FWHM.

    Returns:
    - FWHM (float): Full width at half maximum of the signal.
    if return_edges:
    - FWHM_edges (1D np.ndarray): Edges defining the FWHM [left, right].
    """

    if x is None:
        x = np.arange(len(signal))

    elif len(signal) != len(x):
        raise ValueError("Signal and x must have the same length.")

    sorter = np.argsort(x)
    x = x[sorter]
    signal = signal[sorter]

    smoothed_signal = smooth_circularly(signal, k=k)

    # check right side of the peak
    right_idx = get_partway_idx_of_half_max_crossing_on_right(smoothed_signal)
    left_idx = get_partway_idx_of_half_max_crossing_on_right(smoothed_signal[::-1])

    if max_x is None:
        max_x = x.max()

    FWHM_edges = np.full(2, np.nan)
    if np.isnan(right_idx) or np.isnan(left_idx):
        FWHM = max_x

    elif np.isclose(right_idx, len(x) - 1 - left_idx):
        FWHM = max_x

    else:
        right_pos = get_interp_x(x, right_idx, max_x=max_x)
        left_pos = get_interp_x(x[::-1], left_idx, max_x=max_x)

        if right_pos == left_pos:
            FWHM = max_x
        else:
            FWHM_edges = np.asarray([left_pos, right_pos])
            if right_pos > left_pos:
                FWHM = right_pos - left_pos
            else:
                FWHM = max_x + (right_pos - left_pos)

    if return_edges:
        return FWHM, FWHM_edges

    else:
        return FWHM


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


def get_exp_filtered_signal(
    f_t1: np.ndarray,
    X_t: np.ndarray | None = None,
    T_t: np.ndarray | None = None,
    filter_tau: float | None = None,
    trend_tau: float | None = None,
    dt: float = 0.03,
    atol: float = 1e-08,
) -> tuple[np.ndarray, np.ndarray]:
    """
    get_exp_filtered_signal(f_t1)

    Obtain an exponentially filtered signal.

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

    Obtain an exponential signal using specific filtering parameters. The exponential
    will start at 1.

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
        X_t1, T_t = get_exp_filtered_signal(
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


def smooth_kernel(kernel, sigma_in_steps, smooth_2D=False):
    """
    smooth_kernel(kernel, sigma_in_steps)

    Smooth a kernel using a Gaussian filter with an AUC of 1.

    Args:
    - kernel (1D np.ndarray): Kernel to smooth.
    - sigma_in_steps (int): Number of steps to smooth the kernel.
    - smooth_2D (bool, optional): If True, smooth the kernel in 2D. Default is False.

    Returns:
    - smoothed_kernel (1D or 2D np.ndarray): Smoothed kernel.
    """

    num_pts = len(kernel)
    kernel_2D = kernel.reshape(-1, 1)

    if smooth_2D:
        post = (num_pts - 1) // 2
        pre = num_pts - post - 1
        kernel_2D = np.pad(kernel_2D, pad_width=((0, 0), (pre, post)), mode="constant")

    gaussian_kernel = get_2D_Gaussian_kernel(
        sigma=sigma_in_steps, max_aperture=num_pts * 2
    )
    if sigma_in_steps > 4:
        smoothed_kernel = fftconvolve(kernel_2D, gaussian_kernel, mode="same")
    else:
        smoothed_kernel = gaussian_filter(
            kernel_2D, sigma_in_steps, mode="constant", cval=0
        )

    return smoothed_kernel


def pad_and_convolve_exp(pre_exp, post_exp):
    """
    pad_and_convolve_exp(pre_exp, post_exp)

    Pad and convolve two exponential signals.

    Args:
    - pre_exp (1D np.ndarray): Pre rising exponential signal.
    - post_exp (1D np.ndarray): Post falling exponential signal.

    Returns:
    - convolved (1D np.ndarray): Convolved signal.
    - align_pt (int): Alignment point for the convolved pre and post exponentials.
    """

    align_pt = len(pre_exp) - 1

    len_pre = len(pre_exp)
    len_post = len(post_exp)
    num_pad = max(len_pre, len_post) - 1
    pre_exp = np.pad(pre_exp, (0, num_pad), mode="constant")
    post_exp = np.pad(post_exp, (num_pad, 0), mode="constant")

    if len(pre_exp) < len(post_exp):
        shift = len(post_exp) - len(pre_exp)
        pre_exp = np.pad(pre_exp, (shift, 0), mode="constant")
        align_pt += shift  # only shift in this case
    if len(post_exp) < len(pre_exp):
        shift = len(pre_exp) - len(post_exp)
        post_exp = np.pad(post_exp, (0, len(pre_exp) - len(post_exp)), mode="constant")

    convolved = np.convolve(pre_exp, post_exp, mode="same")

    return convolved, align_pt


def get_pre_post_exponential(
    pre_filter_tau: float | None = None,
    pre_trend_tau: float | None = None,
    dt: float = 0.03,
    post_filter_tau: float | str | None = None,
    post_trend_tau: float | None = None,
    sigma_in_steps: int | None = None,
    concat: bool = False,
    return_AUC_only: bool = False,
):
    """
    get_pre_post_exponential()

    Obtain a pre and post combined exponential signal using specific filtering
    parameters.

    Args:
    - pre_filter_tau (float, optional): Time constant of the pre filter component.
        Default is None.
    - pre_trend_tau (float, optional): Time constant of the pre trend component.
        Default is None.
    - dt (float, optional): Time step. Default is 0.03.
    - post_filter_tau (float, optional): Time constant of the post filter component.
        If None, pre_filter_tau is used. Default is None.
    - post_trend_tau (float, optional): Time constant of the post trend component.
        If None, pre_trend_tau is used. Default is None.
    - sigma_in_steps (int, optional): Number of steps to smooth the signal.
        Default is None.
    - concat (bool, optional): If True, concatenate the pre and post exponential
        instead of convolving them. Default is False.
    - return_AUC_only (bool, optional): If True, return the area under the curve only.
        Default is False.

    Returns:
    if return_AUC_only:
    - AUC (float): Area under the curve.
    else:
    - pre_post_exp (np.ndarray): Pre and post combined exponential signal.
    - align_pt (int): Alignment point of the pre and post exponentials.
    """

    pre_exp = get_exponential(pre_filter_tau, pre_trend_tau, dt=dt)[::-1]

    if post_filter_tau is None:
        post_exp = np.asarray([pre_exp[-1]])
    else:
        post_filter_tau = get_relative_filter_tau(
            post_filter_tau, base_filter_tau=pre_filter_tau
        )
        post_exp = get_exponential(post_filter_tau, post_trend_tau, dt=dt)

    if concat:
        align_pt = len(pre_exp) - 1
        pre_post_exp = np.concatenate([pre_exp, post_exp[1:]])
    else:
        pre_post_exp, align_pt = pad_and_convolve_exp(pre_exp, post_exp)

    if sigma_in_steps is not None:
        pre_post_exp = smooth_kernel(pre_post_exp, sigma_in_steps)[:, 0]

    if return_AUC_only:
        AUC = np.sum(pre_post_exp)
        return AUC
    else:
        return pre_post_exp, align_pt


def get_exp_weighted_sum(pos_exp, neg_exp, neg_weight=0.5, align=None):
    """
    get_exp_weighted_sum(pos_exp, neg_exp)

    Obtain a weighted sum of two exponentially filtered signals.

    NOTE: If exponentials are passed instead of exponentially filtered signals, an error
    adjusted negative weight should be provided.

    Args:
    - pos_exp (float or np.ndarray): Positive exponential signal.
    - neg_exp (float or np.ndarray): Negative exponential signal.
    - neg_weight (float, optional): Weight to attribute to the negative filter
        component. Default is 0.5.
    - align (str, list, None, optional): End of points at which to align the two
        exponential signals ("start", "end" or [val1, val2]). Default is None.

    Returns:
    - weighted_sum (np.ndarray): Weighted sum of the two exponential signals.
    - align_pt (int): Index where the two exponential signals were aligned for summing.
        None if no alignment was done.
    """

    align_pt = None

    if align is not None:
        if isinstance(align, str):
            if align == "start":
                align_pts = [0, 0]
            elif align == "end":
                align_pts = [len(pos_exp), len(neg_exp)]
            else:
                raise ValueError(
                    "If provided as a string, align must be 'start' or 'end'."
                )
        elif len(align) == 2:
            align_pts = align
        else:
            raise ValueError("If provided, align_pts must comprise 2 values.")

        pos_pt, neg_pt = align_pts
        align_pt = max(pos_pt, neg_pt)

        diff_pre = pos_pt - neg_pt
        if diff_pre < 0:
            pos_exp = np.concatenate([np.zeros(np.absolute(diff_pre)), pos_exp])
        elif diff_pre > 0:
            neg_exp = np.concatenate([np.zeros(diff_pre), neg_exp])

        diff_post = len(pos_exp) - len(neg_exp)
        if diff_post < 0:
            pos_exp = np.concatenate([pos_exp, np.zeros(np.absolute(diff_post))])
        elif diff_post > 0:
            neg_exp = np.concatenate([neg_exp, np.zeros(diff_post)])

    if len(pos_exp) != len(neg_exp):
        raise ValueError(
            "Positive and negative exponential signals must have the same length."
        )

    if neg_weight == 1:
        raise ValueError(
            "'neg_weight' cannot be 1, as this will lead to division by 0."
        )

    div = np.absolute(1 - neg_weight)
    weighted_sum = pos_exp / div - neg_weight * neg_exp / div

    return weighted_sum, align_pt


def get_norm_adj_neg_weight(
    neg_weight=0.5, filter_tau_pos=1.0, filter_tau_neg=1.1, raise_one=True
):
    """
    get_norm_adj_neg_weight()

    Obtain normalization adjusted negative weight. Used to adjust the negative weight
    parameter used to subtract one exponential filter (neg) from another (pos) if they
    have been normalized to a max of 1.

    Args:
    - neg_weight (float, optional): Weight to attribute to the negative filter
        component. Default is 0.5.
    - filter_tau_pos (float, optional): Time constant of the positive filter component.
        Default is 1.0.
    - filter_tau_neg (float, optional): Time constant of the negative filter component.
        Default is 1.1.
    - raise_one (bool, optional): If True, raise an error if the adjusted negative
        weight is equal to 1. Default is True.

    Returns:
    - adj_neg_weight (float): Adjusted negative weight.
    """

    adj_neg_weight = neg_weight * filter_tau_pos / filter_tau_neg
    if adj_neg_weight == 1 and raise_one:
        raise RuntimeError(
            f"With pos tau={filter_tau_pos}, neg tau={filter_tau_neg} and "
            f"neg weight={neg_weight}, the adjusted negative weight is equal to 1."
        )

    return adj_neg_weight


def get_summed_exp(
    pre_filter_tau_pos=1,
    pre_filter_tau_neg=1.2,
    pre_neg_weight=0.5,
    dt=0.03,
    post_filter_tau_pos=None,
    post_filter_tau_neg=None,
    post_neg_weight=None,
    sigma_in_steps=None,
    norm_max=True,
    return_AUC_only=False,
    concat=False,
    **kwargs,
):
    """
    get_summed_exp()

    Obtain a summed exponential signal using specific filtering parameters.

    NOTE: Negative weights are adjusted, assuming exponentials are used as filters, and
    thus normalized based on their AUCs.

    Args:
    - pre_filter_tau_pos (float, optional): Tau of the pre, positive filter component.
        Default is 1.
    - pre_filter_tau_neg (float, optional): Tau of the pre, negative filter component.
        Default is 1.2.
    - dt (float, optional): Time step. Default is 0.03.
    - pre_neg_weight (float, optional): Weight to attribute to pre, negative filter
        component. Default is 0.5.
    - post_filter_tau_pos (float, str, optional): Post-filter time constant for positive
        exponential. If None, no post, positive component is included. Default is None.
    - post_filter_tau_neg (float, str, optional): Post-filter time constant for negative
        exponential. If None, no post, negative component is included. Default is None.
    - post_neg_weight (float, optional): Weight to attribute to pre, negative filter
        component. If None, pre_pos_prop is used. Default is None.
    - sigma_in_steps (float, optional): Standard deviation of the Gaussian noise in
        steps. If None, no smoothing is applied. Default is None.
    - norm_max (bool, optional): If True, normalize to the maximum value.
        Default is True.
    - return_AUC_only (bool, optional): If True, return the area under the curve.
        Default is False.
    - concat (bool, optional): If True, concatenate the pre and post exponential
        instead of convolving them. Default is False.

    Keyword Args:
    - kwargs: Additional keyword arguments ending with "_pos" or "_neg", passed to
        compute pos or neg exponential.

    Returns:
    if return_AUC_only:
    - AUC (float): Area under the curve.
    else:
    - summed_pre_post_exp (np.ndarray): Pre and post combined summed exponential signal.
    - align_pt (int): Alignment point of the pre and post exponentials.
    """

    pos_kwargs, neg_kwargs = dict(), dict()
    for key, item in kwargs.items():
        if key.endswith("_pos"):
            pos_kwargs[key.replace("_pos", "")] = item
        elif key.endswith("_neg"):
            neg_kwargs[key.replace("_neg", "")] = item
        else:
            raise ValueError(f"Invalid keyword argument: {key}")

    # positive exponentials
    if post_filter_tau_pos is None:
        post_filter_tau_pos = pre_filter_tau_pos
    else:
        post_filter_tau_pos = get_relative_filter_tau(
            post_filter_tau_pos, base_filter_tau=pre_filter_tau_pos
        )
    pos_exp, pos_align_pt = get_pre_post_exponential(
        pre_filter_tau_pos,
        dt=dt,
        post_filter_tau=post_filter_tau_pos,
        concat=True,
        **pos_kwargs,
    )

    # negative exponentials
    if post_filter_tau_neg is None:
        post_filter_tau_neg = pre_filter_tau_neg
    else:
        post_filter_tau_neg = get_relative_filter_tau(
            post_filter_tau_neg, base_filter_tau=pre_filter_tau_neg
        )
    neg_exp, neg_align_pt = get_pre_post_exponential(
        pre_filter_tau_neg,
        dt=dt,
        post_filter_tau=post_filter_tau_neg,
        concat=True,
        **neg_kwargs,
    )

    norm_adj_pre_neg_weight = get_norm_adj_neg_weight(
        pre_neg_weight,
        filter_tau_pos=pre_filter_tau_pos,
        filter_tau_neg=pre_filter_tau_neg,
    )
    summed_pre, _ = get_exp_weighted_sum(
        pos_exp[: pos_align_pt + 1],
        neg_exp[: neg_align_pt + 1],
        neg_weight=norm_adj_pre_neg_weight,
        align="end",
    )

    # obtain adjusted negative weight (post)
    post_neg_weight = post_neg_weight or pre_neg_weight
    norm_adj_post_neg_weight = get_norm_adj_neg_weight(
        post_neg_weight,
        filter_tau_pos=post_filter_tau_pos,
        filter_tau_neg=post_filter_tau_neg,
    )
    summed_post, _ = get_exp_weighted_sum(
        pos_exp[pos_align_pt:],
        neg_exp[neg_align_pt:],
        neg_weight=norm_adj_post_neg_weight,
        align="start",
    )

    if concat:
        summed_pre_post_exp = np.concatenate([summed_pre, summed_post[1:]])
        align_pt = max(pos_align_pt, neg_align_pt)
    else:
        summed_pre_post_exp, align_pt = pad_and_convolve_exp(summed_pre, summed_post)

    if sigma_in_steps is not None:
        summed_pre_post_exp = smooth_kernel(summed_pre_post_exp, sigma_in_steps)[:, 0]

    if norm_max:
        summed_pre_post_exp = summed_pre_post_exp / summed_pre_post_exp.max()

    if return_AUC_only:
        AUC = np.sum(summed_pre_post_exp)
        return AUC
    else:
        return summed_pre_post_exp, align_pt


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
