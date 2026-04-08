from typing import Any

import numpy as np

from ratinabox import utils as rutils  # type: ignore[import]


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

    if (v1 == 0).all() or (v2 == 0).all():
        raise ValueError("Cannot calculate angle with a zero-length vector.")

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
