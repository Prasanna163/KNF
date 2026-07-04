"""
Low-level geometry utilities: distances, angles, unit vectors.

All functions operate on a ``(N, 3)`` NumPy coordinate array and use
integer atom indices.
"""

from __future__ import annotations

import numpy as np


def distance(coords: np.ndarray, i: int, j: int) -> float:
    """Euclidean distance between atoms *i* and *j*.

    Parameters
    ----------
    coords : np.ndarray
        Cartesian coordinate array with shape ``(N, 3)``.
    i, j : int
        Zero-based atom indices.

    Returns
    -------
    float
        Distance in the same length unit as *coords* (typically Å).

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    >>> distance(coords, 0, 1)
    1.0
    """
    diff = coords[j] - coords[i]
    return float(np.linalg.norm(diff))


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Full pairwise distance matrix.

    Parameters
    ----------
    coords : np.ndarray
        Cartesian coordinate array, shape ``(N, 3)``.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix of shape ``(N, N)`` with zeros on the
        diagonal.

    Notes
    -----
    Uses the identity
    ``||r_i - r_j||² = ||r_i||² + ||r_j||² - 2 r_i · r_j``
    for an efficient vectorised implementation.
    """
    # (N,)  squared norms
    sq_norms = np.einsum("ij,ij->i", coords, coords)
    # (N, N)  squared distances
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (coords @ coords.T)
    # Clamp tiny negatives that arise from floating-point noise
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def angle(coords: np.ndarray, i: int, j: int, k: int) -> float:
    """Valence angle i–j–k with *j* as the central (vertex) atom.

    Parameters
    ----------
    coords : np.ndarray
        Cartesian coordinate array, shape ``(N, 3)``.
    i, j, k : int
        Zero-based atom indices.  *j* is the vertex.

    Returns
    -------
    float
        Angle in **radians**, in the range [0, π].

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=float)
    >>> np.degrees(angle(coords, 0, 1, 2))  # 90°
    90.0
    """
    v1 = coords[i] - coords[j]
    v2 = coords[k] - coords[j]

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0  # degenerate — atoms on top of each other

    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp to [-1, 1] to guard against floating-point overshoots
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of *v*.

    Parameters
    ----------
    v : np.ndarray
        Input vector (any shape that ``np.linalg.norm`` accepts).

    Returns
    -------
    np.ndarray
        Normalised vector.  Returns a zero vector if ``||v|| ≈ 0``.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        return np.zeros_like(v)
    return v / norm
