from typing import Any

import numpy as np


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
    ws: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]]
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
        b += incr

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
    - ws (list of 2D np.ndarrays): List of weights for each input layer i, each with shape
        (O, I_i).
    - O (1D np.ndarray): Actual or target output
    - lr (float, optional): Learning rate Default is 1e-4.
    - p (int, optional): Normalization factor. Default is 2.
    - b (1D np.ndarray, optional): Biases (one per output neuron). Default is None.
    - alpha (float or 1D np.ndarray, optional): Regularization strength. Default is 1.0.
    """

    # in-place update
    perform_Hebbian_update_(Is, ws, O, lr=lr, b=b)

    # in-place update
    w_div = calculate_Hebbian_norm(ws, p=p)

    # adjustment
    for i in range(len(ws)):
        ws[i] /= w_div.reshape(-1, 1) * alpha

    # update biases, if provided
    if b is not None:
        b_div = calculate_Hebbian_norm([b], p=p)
        b /= b_div * alpha  # adjust

    return


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
    """

    # before in-place update
    O_for_normalization = O
    if normalize_on_predictions:  # calculate normalization on predictions
        O_for_normalization = calculate_layer_output(Is, ws, b=b)

    w_subtrahends = calculate_Oja_subtrahend(ws, O_for_normalization)
    b_subtrahend = None
    if b is not None:
        b_subtrahend = O**2 * b  # Oja-like subtrahend for biases

    # in-place update
    perform_Hebbian_update_(Is, ws, O, lr=lr, b=b)

    # adjustment
    for i in range(len(ws)):
        ws[i] -= lr * alpha * w_subtrahends[i]
    if b is not None:
        if b_subtrahend is None:
            raise NotImplementedError(
                "`b_subtrahend` should be specified if b is not None."
            )
        b -= lr * alpha * b_subtrahend  # type: ignore[operator]

    return
