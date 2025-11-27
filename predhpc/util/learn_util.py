import copy
from typing import Any
import warnings

import numpy as np

from predhpc.util import ext_util, gen_util, signal_util, params_util, plot_util


def calculate_layer_output(
    Is: list[np.ndarray[tuple[int], np.dtype[np.float64]]],
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    b: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    calculate_layer_output(Is, ws)

    Calculate the output of a layer.

    Args:
    - Is (list of list of 1D np.ndarrays): List of 1D activation arrays for each input
        layer.
    - ws (list of 2D np.ndarrays): List of weights for each input layer i, each with shape
        (O, I_i).
    - b (1D np.ndarray, optional): Biases (one per output neuron). Default is None.

    Raises:
    - ValueError: If number of input layers does not match number of weights.

    Returns:
    - O (1D np.ndarray): Calculated layer output.
    """

    if len(Is) != len(ws):
        raise ValueError(
            f"Number of input layers ({len(Is)}) must match number of weights "
            f"({len(ws)})."
        )

    O = np.sum([np.dot(I, w.T) for I, w in zip(Is, ws)], axis=0)
    if b is not None:
        O += b
    return O


def calculate_mse_loss(
    targets: np.ndarray[Any, np.dtype[np.float64]],
    predictions: np.ndarray[Any, np.dtype[np.float64]],
    axis: int | None = None,
) -> float:
    """
    calculate_mse_loss(targets, predictions)

    Calculate the mean squared error between targets and predictions.

    Args:
    - targets (np.ndarray): Target values
    - predictions (np.ndarray): Predicted values, with same shape as targets
    - axis (int, optional): Axis along which to calculate the mean. Default is None.

    Raises:
    - ValueError: If targets and predictions do not have the same shape.

    Returns:
    - mse_loss (float or np.ndarray): MSE between targets and predictions. Number of
        axes depends on how many axes are specified by 'axis'.
    """

    if targets.shape != predictions.shape:
        raise ValueError("Targets and predictions must have the same shape.")

    mse_loss = ((targets - predictions) ** 2).mean(axis=axis)

    return mse_loss


def get_weight_norm(
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
) -> float:
    """
    get_weight_norm(ws)

    Calculate the L2 norm of the weights.

    Args:
    - ws (list of 2D np.ndarrays): List of weights for each input layer i, each
        with shape (O, I_i).

    Returns:
    - l2 (float): L2 norm of the weights.
    """

    l2 = float(np.sum([np.linalg.norm(w, ord=2) for w in ws]))

    return l2


def calculate_mse_loss_across_samples(
    all_Is: list[list[np.ndarray[tuple[int], np.dtype[np.float64]]]],
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    all_Os: list[np.ndarray[tuple[int], np.dtype[np.float64]]],
    b: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> float:
    """
    calculate_mse_loss_across_samples(all_Is, ws, all_Os)

    Calculate the MSE loss across all samples.

    Args:
    - all_Is (list of list of 1D np.ndarrays): For each sample, list of 1D activation
        arrays for each input layer.
    - ws (list of 2D np.ndarrays): List of weights for each input layer i, each with shape
        (O, I_i).
    - all_Os (list of 1D np.ndarrays): Target outputs for each sample
    - b (1D np.ndarray, optional): Biases (one per output neuron). Default is None.

    Raises:
    - ValueError: If number of input samples does not match number of output samples.

    Returns:
    - loss (float): Loss across all samples
    """

    if len(all_Is) != len(all_Os):
        raise ValueError(
            f"Number of input samples ({len(all_Is)}) must match number of output "
            f"samples ({len(all_Os)})."
        )

    loss = 0.0
    for Is, Os in zip(all_Is, all_Os):
        loss += calculate_mse_loss(Os, calculate_layer_output(Is, ws, b=b))

    return loss


def calculate_max_output_value(
    all_Is: list[list[np.ndarray[tuple[int], np.dtype[np.float64]]]],
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    b: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> float:
    """
    calculate_max_output_value(all_Is, ws)

    Calculate the maximum output value, based on the weights, and inputs, across
    all samples.

    Args:
    - all_Is (list of list of 1D np.ndarrays): For each sample, list of 1D activation
        arrays for each input layer.
    - ws (list of 2D np.ndarrays): List of weights for each input layer i, each with shape
        (O, I_i).
    - b (1D np.ndarray, optional): Biases (one per output neuron). Default is None.

    Returns:
    - max_output_value (float): Max output value, based on the weights, and inputs,
        across all samples.
    """

    max_output_value = max([calculate_layer_output(Is, ws, b=b).max() for Is in all_Is])

    return max_output_value


def perform_Hebbian_update_(
    Is: list[np.ndarray[tuple[int], np.dtype[np.float64]]],
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    O: np.ndarray[tuple[int], np.dtype[np.float64]],
    lr: np.ndarray[tuple[int], np.dtype[np.float64]] | float = 1e-4,
    b: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
):
    """
    perform_Hebbian_update_(Is, ws, O)

    Perform a Hebbian update on the weights and biases, in place.

    Args:
    - Is (list of 1D np.ndarrays): 1D activation arrays for each input layer.
    - ws (list of 2D np.ndarrays): List of weights for each input layer i, each with shape
        (O, I_i).
    - O (1D np.ndarray): Actual or target output.
    - lr (float or 1D np.ndarray, optional): Learning rate. Default is 1e-4.
    - b (1D np.ndarray, optional): Biases (one per output neuron). Default is None.

    Raises:
    - ValueError: If each w does not have the shape (O, I_i).
    - ValueError: If b does not have the length of O.
    """

    lr = np.asarray(lr).reshape(-1, 1)
    for i, I in enumerate(Is):
        if ws[i].shape != (len(O), len(I)):
            raise ValueError(
                f"w should have shape ({len(O)}, {len(I)}), "
                f"but found {ws[i].shape}."
            )
        incr = lr * np.outer(O, I)
        ws[i] += incr

    lr = np.asarray(lr).ravel()
    if b is not None:
        if len(b) != len(O):
            raise ValueError(
                f"b should have the length of O ({len(O)}), " f"but found {len(b)}."
            )
        incr = lr * O
        b += incr.reshape(b.shape)

    return


def calculate_Hebbian_norm(
    ws: list[np.ndarray[Any, np.dtype[np.float64]]],
    p: int = 2,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """
    calculate_Hebbian_norm(ws)

    Calculate the normalization factor across weights for Hebbian learning.

    Args:
    - ws (list of 1 or 2D np.ndarrays): Weights for each set of inputs i,
        with shape (O, I_i).
    - p (int, optional): Normalization factor. Default is 2.

    Returns:
    - div (1D np.ndarray): Hebbian normalization factor for each set of weights.
    """

    div = np.sum([np.sum(np.absolute(w**p), axis=-1) for w in ws], axis=0) ** (1 / p)

    return div


def perform_divisively_normalized_Hebbian_update_(
    Is: list[np.ndarray[tuple[int], np.dtype[np.float64]]],
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    O: np.ndarray[tuple[int], np.dtype[np.float64]],
    lr: np.ndarray[tuple[int], np.dtype[np.float64]] | float = 1e-4,
    p: int = 2,
    b: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    alpha: float | np.ndarray[tuple[int], np.dtype[np.float64]] = 1.0,
    only_if_above: bool = True,
):
    """
    perform_divisively_normalized_Hebbian_update_(Is, ws, O)

    Perform Hebbian learning, and divisively normalizes the weights after each update,
    in place.

    w_i(n + 1)' = w_i(n) + lr * (O * I_i)
    w_i(n + 1) = w_i(n + 1)' / ||w_i(n + 1)'||_p
    new weights = (old weights + Hebbian update) / p norm of updated weights

    b_i(n + 1)' = b_i(n) + lr * O
    b_i(n + 1) = b_i(n + 1)' / ||b_i(n + 1)'||_p
    new biases = (old biases + Hebbian update) / p norm of updated biases

    Args:
    - Is (list of 1D np.ndarrays): 1D activation arrays for each input layer.
    - ws (list of 2D np.ndarrays): List of weights for each input layer i,
        each with shape (O, I_i).
    - O (1D np.ndarray): Actual or target output
    - lr (float, optional): Learning rate Default is 1e-4.
    - p (int, optional): Normalization factor. Default is 2.
    - b (1D np.ndarray, optional): Biases (one per output neuron). Default is None.
    - alpha (float or 1D np.ndarray, optional): Regularization strength. Default is 1.0.
    - only_if_above (bool, optional): If True, only uses weight normalization to
        decrease weights. Default is True.

    Returns:
    - w_divs (np.ndarray): Divisive normalization factor for each set of weights.
    - b_div (float): Divisive normalization factor for biases. None, if b is None.
    """

    if alpha < 0 or lr < 0 or p < 0:
        raise ValueError(
            "Learning rate, alpha and normalization factor (p) must be non-negative."
        )

    # in-place update
    perform_Hebbian_update_(Is, ws, O, lr=lr, b=b)

    # in-place update
    w_divs = calculate_Hebbian_norm(ws, p=p) * alpha
    use_w_divs = copy.deepcopy(w_divs)
    if only_if_above:
        use_w_divs[use_w_divs <= 1] = 1

    # adjustment
    for i in range(len(ws)):
        ws[i] /= use_w_divs.reshape(-1, 1)

    # update biases, if provided
    if b is None:
        b_div = None
    else:
        b_div = calculate_Hebbian_norm([b], p=p)[0] * alpha
        use_b_div = b_div
        if only_if_above and b_div <= 1:
            use_b_div = 1
        b /= use_b_div  # adjust

    return w_divs, b_div


def calculate_Oja_subtrahend(
    ws: list[np.ndarray[Any, np.dtype[np.float64]]],
    O: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> list[float]:
    """
    calculate_Oja_subtrahend(ws, O)

    Calculate the subtrahend for Oja's rule.

    Subtrahend: O**2 * w_i(n)

    Args:
    - ws (list of 1 or 2D np.ndarrays): Weights for each set of inputs i,
        with shape (O, I_i).
    - O (1D np.ndarray): Actual or target output.

    Returns:
    - subtrahend (list): Subtrahend for Oja's rule.
    """

    subtrahend = [np.dot((O**2).reshape(1, -1), w) for w in ws]
    return subtrahend


def perform_Oja_update_(
    Is: list[np.ndarray[tuple[int], np.dtype[np.float64]]],
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    O: np.ndarray[tuple[int], np.dtype[np.float64]],
    lr: np.ndarray[tuple[int], np.dtype[np.float64]] | float = 1e-4,
    b: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    alpha: float = 0.1,
    normalize_on_predictions: bool = False,
):
    """
    perform_Oja_update_(Is, ws, O)

    Perform an update on the weights and biases, based on Oja's rule, in place.

    dw = lr * O * (I_i - alpha * O * w_i(n))

    w_i(n + 1) = w_i(n) + lr * O * (I_i - alpha * O * w_i(n))
    w_i(n + 1) = w_i(n) + lr * (O * I_i) - lr * alpha * (O**2 * w_i(n))
    new weights = old weights + Hebbian update - Oja subtrahend


    Proposed adaptation for use with biases (should be revised - biases not typically
        implemented with Oja's rule):
    db = lr * O - lr * alpha * O**2 * b_i(n)
    b_i(n + 1) = b_i(n) + lr * O - lr * alpha * O**2 * b_i(n)

    new weights = old weights + Hebbian update - Oja-like subtrahend

    Args:
    - Is (list of 1D np.ndarrays): 1D activation arrays for each input layer.
    - ws (list of 2D np.ndarrays): List of weights for each input layer i, each with shape
        (O, I_i).
    - O (1D np.ndarray): Actual or target output
    - lr (float or 1D np.ndarray, optional): Learning rate. Default is 1e-4.
    - b (1D np.ndarray, optional): Biases (one per output neuron). Default is None.
    - alpha (float or 1D np.ndarray, optional): Regularization strength. Default is 0.1.
    - normalize_on_predictions (bool, optional): If True, normalizes the weights
        based on the predictions, rather than the actual output (target) provided.
        Default is False.

    Returns:
    - w_subtrahends (list): Subtrahends for Oja's rule.
    - b_subtrahends (1D np.array): Subtrahends for biases. None, if b is None.
    """

    if alpha < 0 or lr < 0:
        raise ValueError("Learning rate and alpha must be non-negative.")

    # before in-place update
    O_for_normalization = O
    if normalize_on_predictions:  # calculate normalization on predictions
        O_for_normalization = calculate_layer_output(Is, ws, b=b)

    w_subtrahends = calculate_Oja_subtrahend(ws, O_for_normalization)
    w_subtrahends = [w_subtrahend * alpha * lr for w_subtrahend in w_subtrahends]

    b_subtrahends = None
    if b is not None:
        b_subtrahends = O**2 * b * alpha * lr  # Oja-like subtrahend for biases

    # in-place update
    perform_Hebbian_update_(Is, ws, O, lr=lr, b=b)

    # adjustment
    for i in range(len(ws)):
        ws[i] -= w_subtrahends[i]
    if b is not None:
        b -= b_subtrahends  # type: ignore[operator]

    return w_subtrahends, b_subtrahends


def perform_update_(
    Is: list[np.ndarray[tuple[int], np.dtype[np.float64]]],
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]],
    O: np.ndarray[tuple[int], np.dtype[np.float64]],
    lr: np.ndarray[tuple[int], np.dtype[np.float64]] | float = 1e-4,
    b: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    normalize_weights_divisively: bool = False,
    p: int = 2,
    alpha: float | np.ndarray[tuple[int], np.dtype[np.float64]] = 1.0,
    only_if_above: bool = True,
    apply_Ojas_rule: bool = False,
    normalize_on_predictions: bool = False,
):
    """
    perform_update_(Is, ws, O)

    Perform an update on the weights and biases, using Hebbian learning,
    Oja's rule or Hebbian learning with divisive normalization.

    Args:
    - Is (list of 1D np.ndarrays): 1D activation arrays for each input layer.
    - ws (list of 2D np.ndarrays): List of weights for each input layer i, each with shape
        (O, I_i).
    - O (1D np.ndarray): Actual or target output
    - lr (float or 1D np.ndarray, optional): Learning rate. Default is 1e-4.
    - b (1D np.ndarray, optional): Biases (one per output neuron). Default is None.
    - normalize_weights_divisively (bool, optional): If True, uses divisive
        normalization for weights. Default is False.
    - p (int, optional): Normalization factor, if using divisive normalization.
        Default is 2.
    - only_if_above (bool, optional): If True, only uses weight normalization to
        decrease weights, if using divisive normalization. Default is True.
    - alpha (float or 1D np.ndarray, optional): Regularization strength. Default is 0.1.
    - only_if_above (bool, optional): If True, only uses weight normalization to
        decrease weights. Default is True.
    - apply_Ojas_rule
    - normalize_on_predictions (bool, optional): If True and using Oja's rule,
        normalizes the weights based on the predictions, rather than the actual output
        (target) provided. Default is False.

    Raises:
    - ValueError: If both Oja's rule and divisive normalization are applied.

    Returns:
    if apply_Ojas_rule:
    - w_subs (list): Subtrahends for Oja's rule.
    - b_subs (np.ndarray): Subtrahends for biases. None, if b is None.

    if normalize_weights_divisively:
    - w_div (float): Divisive normalization factor for weights.
    - b_div (float): Divisive normalization factor for biases. None, if b is None.
    """

    if apply_Ojas_rule and normalize_weights_divisively:
        raise ValueError("Cannot use both Oja's rule and divisive normalization.")

    if apply_Ojas_rule:
        w_subs, b_subs = perform_Oja_update_(
            Is,
            ws,
            O,
            lr=lr,
            b=b,
            alpha=alpha,
            normalize_on_predictions=normalize_on_predictions,
        )
        return w_subs, b_subs

    elif normalize_weights_divisively:
        w_div, b_div = perform_divisively_normalized_Hebbian_update_(
            Is, ws, O, lr=lr, b=b, p=p, alpha=alpha, only_if_above=only_if_above
        )

        return w_div, b_div

    else:
        perform_Hebbian_update_(Is, ws, O, lr=lr, b=b)


def infer_spatial_learning_kernel(
    dt=0.03,
    speed_mean=0.08,
    PC_input_density_1D=11,
    PC_width=0.10,
    PC_max_fr=1,
    env_scale=1,
    env_1D=False,
    plot=False,
    summed_exp_kernel=True,
    kernel_kwargs=dict(),
):
    """
    infer_spatial_learning_kernel()

    Infer the spatial profile of a learning kernel applied to a set of uniformly
    arranged place cells, traversed by an agent at a constant speed.

    Args:
    - dt (float): Time step. Default is 0.03.
    - speed_mean (float): Mean speed of the agent. Default is 0.08.
    - PC_input_density_1D (int): Density of place cells to use (PCs/m). Default is 11.
    - PC_width (float): Width of place cell fields. Default is 0.10.
    - PC_max_fr (float, optional): Maximum firing rate of place cells. Default is 1.
    - env_scale (int): Scale of the environment. Default is 1.
    - env_1D (bool): If True, a 1D environment is modeled. Default is False.
    - plot (bool): If True, plots the inferred learning kernel. Default is False.
    - summed_exp_kernel (bool): If True, applies a summed exponential kernel to the
        input. Default is True.
    - kernel_kwargs (dict): Additional arguments for gen_util.get_summed_exp() or
        signal_util.get_pre_post_exponential().

    Returns:
    - Is (2D np.ndarray): Interpolated 2D learning kernel. The second dimension is of
        length 1, if env_1D is True.
    if plot:
    - sub_ax (plt.Axes): Subplot of the learning kernel.
    """

    # filtered input PC activity if crossing place fields in a perfect straight line
    if summed_exp_kernel:
        kernel, align_pt = signal_util.get_summed_exp(dt=dt, **kernel_kwargs)
    else:
        kernel, align_pt = signal_util.get_pre_post_exponential(dt=dt, **kernel_kwargs)

    kernel = kernel / kernel.max() * PC_max_fr

    # apply Gaussian filter to smooth the input
    sigma_in_steps = ext_util.get_sigma_in_steps(
        sigma=float(PC_width), dt=dt, mean_speed=speed_mean
    )

    # interpolate at PC field locations (evenly spaced)
    num_pts = int(np.around(2 * env_scale * PC_input_density_1D))
    coords_base = np.linspace(0, 2 * env_scale, num_pts)
    coords_base -= coords_base[(len(coords_base) - 1) // 2]

    smoothed_kernel = signal_util.gaussian_smooth_kernel(
        kernel,
        sigma_in_steps,
        smooth_2D=not (env_1D),
    )

    smoothed_kernel = smoothed_kernel / smoothed_kernel.max() * PC_max_fr

    # get x-coordinates
    distance_btw = dt * speed_mean
    xs = np.arange(len(kernel)) * distance_btw

    xs -= xs[align_pt]

    Is = np.asarray(
        [
            np.interp(coords_base, xs, smoothed_kernel[:, i])
            for i in range(smoothed_kernel.shape[1])
        ]
    ).T

    # center kernel to maximize AUC
    num_pts_keep = len(coords_base) // 2
    best = np.argmax(
        np.convolve(np.absolute(Is).sum(axis=-1), np.ones(num_pts_keep), mode="valid")
    )

    Is = Is[best : best + num_pts_keep]
    x_coords = coords_base[best : best + num_pts_keep]

    if not env_1D:
        # interpolate along y axis
        ys = np.arange(Is.shape[1]) * distance_btw
        ys -= ys[(len(ys) - 1) // 2]
        y_coords = coords_base[(num_pts_keep - 1) // 2 :][:num_pts_keep]
        Is = np.asarray([np.interp(y_coords, ys, Is[i]) for i in range(len(x_coords))])

    if plot:
        sub_ax = plot_util.plot_learning_kernel(Is, x_coords, smoothed_kernel, xs)
        return Is, sub_ax

    else:
        return Is


def assess_learning_rates_spatially(
    learning_kernel=None,
    lr=1e-4,
    dt=0.03,
    speed_mean=0.08,
    PC_input_density_1D=11,
    PC_width=0.10,
    PC_max_fr=10,
    env_scale=1,
    w_init_loc=0,
    incl_bias=False,
    apply_Ojas_rule=False,
    normalize_weights_divisively=False,
    p=1,
    regularization_alpha=None,
    activation_function=None,
    env_1D=False,
    output_fr=None,
    num_updates=1,
    flip=False,
    log_reg=False,
    plot=False,
    plot_input_kernel=False,
    summed_exp_kernel=True,
    kernel_kwargs=dict(),
):
    """
    assess_learning_rates_spatially()

    Assess learning rates spatially for a simple learning rule. Infers a spatial
    learning kernel applied to uniformly distributed Gaussian place cells, and
    simulates a series of weight updates using the provided or inferred kernel.

    Args:
    - learning_kernel (1D np.ndarray): Learning kernel. If None, a kernel is inferred.
        Default is None.
    - lr (float): Learning rate. Default is 1e-4.
    - dt (float): Time step. Default is 0.03.
    - speed_mean (float): Mean speed of the agent. Default is 0.08.
    - PC_input_density_1D (int): Density of place cells to use (PCs/m or PCs/m2).
        Default is 11.
    - PC_width (float): Width of place cell fields. Default is 0.10.
    - PC_max_fr (float): Maximum firing rate of place cells. Default is 10.
    - env_scale (int): Scale of the environment. Default is 1.
    - w_init_loc (float): Initial weight mean. Default is 0.
    - incl_bias (bool): If True, a bias is included. Default is False.
    - apply_Ojas_rule (bool): If True, Oja's rule is used in weight updates.
        Default is False.
    - normalize_weights_divisively (bool): If True, weights are normalized divisively.
        Default is False.
    - p (int): Divisive normalization factor. Default is 1.
    - regularization_alpha (float, optional): Regularization strength. Default is None.
    - activation_function (str, optional): Activation function. Default is None.
    - env_1D (bool): If True, a 1D environment is modeled. Default is False.
    - output_fr (float, optional): Output firing rate. Default is None.
    - num_updates (int): Number of weight updates to simulate. Default is 1.
    - flip (bool): If True, the input kernel is flipped for every other update to
        simulate updates in alternating locations. Default is False.
    - log_reg (bool): If regulartization is applied, logs the regularization used.
        Default is False.
    - plot (bool): If True, plots the learning rate assessment results. Default is False.
    - plot_input_kernel (bool): If True, plots the input kernel. Default is False.
    - summed_exp_kernel (bool): If True, applies a summed exponential kernel to the
        input. Default is True.
    - kernel_kwargs (dict): Additional arguments for signal_util.get_summed_exp() or
        signal_util.get_pre_post_exponential().

    Returns:
    - assessment_dict (dict): Learning rate assessment dictionary with initial and
        updated weights (under "ws"), computed output firing rates (under "Os"), and
        biases if applicable (under "bs").
    """

    if learning_kernel is None:
        outputs = infer_spatial_learning_kernel(
            dt=dt,
            speed_mean=speed_mean,
            PC_input_density_1D=PC_input_density_1D,
            PC_width=PC_width,
            PC_max_fr=PC_max_fr,
            env_scale=env_scale,
            env_1D=env_1D,
            plot=plot_input_kernel,
            summed_exp_kernel=summed_exp_kernel,
            kernel_kwargs=kernel_kwargs,
        )
        Is = outputs[0] if plot_input_kernel else outputs

    else:
        Is = learning_kernel.copy()
        if plot_input_kernel:
            raise NotImplementedError(
                "Plotting of the input kernel is not implemented if kernel is provided."
            )

    # ws
    ws = [np.full((1, Is.size), w_init_loc).astype(float)]
    assessment_dict = {
        "ws": [copy.deepcopy(ws[0]).reshape(Is.shape)],
    }

    # b
    if incl_bias:
        b = np.zeros(1).astype(float) if incl_bias else None
        assessment_dict["bs"] = [copy.deepcopy(b)[0]]
    else:
        b = None

    # O
    activation_function = params_util.get_activation_function(activation_function)
    V = np.dot(ws, Is.ravel())
    O = activation_function(V, deriv=False)
    assessment_dict["Os"] = [copy.deepcopy(O)[0][0]]

    if output_fr is not None:
        O = np.asarray([output_fr])

    for i in range(num_updates):
        if flip and i % 2:
            use_Is = Is.ravel()[::-1]
        else:
            use_Is = Is.ravel()

        outputs = perform_update_(
            [use_Is],
            ws,
            O,
            lr=lr,
            b=b,
            apply_Ojas_rule=apply_Ojas_rule,
            normalize_weights_divisively=normalize_weights_divisively,
            p=p,
            alpha=regularization_alpha,
        )

        if log_reg and (apply_Ojas_rule or normalize_weights_divisively):
            reg_str = f"Update {i + 1} regul.: {outputs[0][0]:.4f} (w)"
            if incl_bias:
                reg_str = f"{reg_str}, {outputs[1]:.4f} (b)"
            print(reg_str)
        assessment_dict["ws"].append(copy.deepcopy(ws[0]).reshape(Is.shape))

        add_b = b if incl_bias else 0
        if incl_bias:
            assessment_dict["bs"].append(copy.deepcopy(b[0]))

        O = activation_function(np.dot(ws, use_Is) + add_b, deriv=False)
        assessment_dict["Os"].append(copy.deepcopy(O[0][0]))

    if plot:
        plot_util.plot_learning_rate_assessment(assessment_dict)

    return assessment_dict


def assess_Pyrs_learning_rates_spatially(
    Pyrs,
    PCs_name="PCs",
    BTSP=True,
    output_fr=None,
    num_updates=1,
    flip=False,
    apply_Ojas_rule=None,
    normalize_weights_divisively=None,
    p=None,
    regularization_alpha=None,
    log_reg=False,
    plot=False,
    plot_input_kernel=False,
):
    """
    assess_Pyrs_learning_rates_spatially()

    Assess learning rates spatially for a Pyramidal neuron layer.

    Args:
    - Pyrs (Pyrs): Pyramidal neuron layer.
    - PCs_name (str): Name of the place cells. Default is "PCs".
    - BTSP (bool): If True, uses BTSP learning. Default is True.
    - output_fr (float, optional): Output firing rate. Default is None.
    - num_updates (int): Number of weight updates to simulate. Default is 1.
    - flip (bool): If True, the input kernel is flipped for every other update to
        simulate updates in alternating locations. Default is False.
    - apply_Ojas_rule (bool): If True, Oja's rule is used in weight updates. If None,
        setting is inferred from Pyrs. Default is None.
    - normalize_weights_divisively (bool): If True, weights are normalized divisively.
        If None, setting is inferred from Pyrs. Default is None.
    - p (int): Divisive normalization factor. If None, setting is inferred from Pyrs.
        Default is None.
    - regularization_alpha (float, optional): Regularization strength. Default is None.
    - log_reg (bool): If regulartization is applied, logs the regularization used.
        Default is False.
    - plot (bool): If True, plots the learning rate assessment results. Default is False.
    - plot_input_kernel (bool): If True, plots the input kernel. Default is False.

    Raises:
    - ValueError: If BTSP is True and Pyrs is not a BTSPLayer.
    - ValueError: If Pyrs is not a HebbianLayer.

    Returns:
    - assessment_dict (dict): Learning rate assessment dictionary with initial and
        updated weights (under "ws"), computed output firing rates (under "Os"), and
        biases if applicable (under "bs").
    """

    if BTSP:
        if not gen_util.attribute_type_checker(Pyrs, "BTSPLayer"):
            raise ValueError("BTSP cannot be set to True if Pyrs is not a BTSPLayer.")
        lr = Pyrs.BTSP_lr / Pyrs.BTSP_integral
        kernel_kwargs = Pyrs.get_BTSP_kernel_kwargs()

    else:
        if not gen_util.attribute_type_checker(Pyrs, "HebbianLayer"):
            raise ValueError("Pyrs must be a HebbianLayer.")
        lr = Pyrs.lr
        kernel_kwargs = Pyrs.get_learning_kernel_kwargs()

    if PCs_name not in Pyrs.inputs.keys():
        raise RuntimeError(f"{PCs_name} not found in inputs to Pyrs.")
    PCs = Pyrs.inputs[PCs_name]["layer"]
    PC_input_density_1D = ext_util.estimate_1D_place_cell_density(PCs)

    if PCs.description != "gaussian":
        if PCs.description == "gaussian_threshold":
            warnings.warn(
                "Actual place cells are modelled with a Gaussian threshold, "
                "but assessment will assume Gaussian place cells."
            )
        else:
            raise NotImplementedError(
                "Function only implemented for Gaussian place cells."
            )
    if PCs.wall_geometry != "geodesic":
        warnings.warn(
            "Assessment will assume geodesic wall geometry even though "
            f"place cell wall geometry is {PCs.wall_geometry}."
        )
    if PCs.place_cell_center_type != "uniform":
        warnings.warn(
            f"Actual place cell centers were initialized as {PCs.place_cell_center_type}, "
            "but assessment will assume uniformly arranged place cell centers."
        )

    Env = Pyrs.Environment
    env_1D = True if Env.D == 1 else False
    if env_1D:
        env_scale = Env.scale
    elif hasattr(Env, "get_area"):
        env_scale = np.sqrt(Env.get_area())
    else:
        raise NotImplementedError(f"Env type {type(Env)} not supported.")

    if apply_Ojas_rule is None:
        apply_Ojas_rule = Pyrs.apply_Ojas_rule
    if normalize_weights_divisively is None:
        normalize_weights_divisively = Pyrs.normalize_weights_divisively
    if p is None:
        p = Pyrs.p
    if regularization_alpha is None:
        regularization_alpha = Pyrs.regularization_alpha

    outputs = assess_learning_rates_spatially(
        lr=lr,
        dt=Pyrs.Agent.dt,
        speed_mean=Pyrs.Agent.speed_mean,
        PC_input_density_1D=PC_input_density_1D,
        PC_width=PCs.widths,
        PC_max_fr=PCs.max_fr,
        env_scale=env_scale,
        w_init_loc=Pyrs.inputs[PCs_name]["w_init"].mean(),
        incl_bias=Pyrs.trainable_biases,
        apply_Ojas_rule=apply_Ojas_rule,
        normalize_weights_divisively=normalize_weights_divisively,
        regularization_alpha=regularization_alpha,
        p=p,
        activation_function=Pyrs.activation_function,
        env_1D=env_1D,
        output_fr=output_fr,
        num_updates=num_updates,
        flip=flip,
        log_reg=log_reg,
        plot=plot,
        plot_input_kernel=plot_input_kernel,
        summed_exp_kernel=BTSP,
        kernel_kwargs=kernel_kwargs,
    )

    assessment_dict = outputs

    return assessment_dict
