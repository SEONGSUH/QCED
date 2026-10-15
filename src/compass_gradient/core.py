from __future__ import annotations

import numpy as np

from .operators import (
    DIRECTIONS,
    NEIGHBOR_OFFSETS,
    CompassOperator,
    get_operator,
)

__all__ = [
    "neighbor_stack",
    "compass_edges",
    "edge_magnitude",
    "edge_direction",
    "detect",
]

_M = 8


def _validate_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2-D grayscale image, got ndim={image.ndim}. "
            "Convert color images to grayscale first."
        )
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError("Image must be at least 3x3.")
    return image.astype(np.float64, copy=False)


def neighbor_stack(image: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Gather the eight neighbors p0..p7 of every pixel.

    Parameters
    ----------
    image:
        2-D grayscale image, shape ``(H, W)``.
    mode:
        Border handling, any mode accepted by :func:`numpy.pad`
        (e.g. ``"reflect"``, ``"edge"``, ``"constant"``).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(H, W, 8)``; ``[..., n]`` holds the intensity
        of the n-th neighbor, enumerated counterclockwise from the
        top-left pixel (see :data:`~compass_gradient.operators.NEIGHBOR_OFFSETS`).
    """
    image = _validate_image(image)
    padded = np.pad(image, 1, mode=mode)
    h, w = image.shape
    out = np.empty((h, w, _M), dtype=np.float64)
    for n, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
        out[:, :, n] = padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
    return out


def _edges_fft(
    image: np.ndarray, op: CompassOperator, mode: str
) -> np.ndarray:
    p = neighbor_stack(image, mode=mode)          # (H, W, 8)
    p_hat = np.fft.fft(p, axis=-1)                # W^{-1} P  (up to scaling)
    r = op.eigenvalues()                          # r(k) = FFT(h)
    eg = np.fft.ifft(r * p_hat, axis=-1).real     # W D W^{-1} P
    return np.moveaxis(eg, -1, 0)                 # (8, H, W)


def _edges_spatial(
    image: np.ndarray, op: CompassOperator, mode: str
) -> np.ndarray:
    p = neighbor_stack(image, mode=mode)          # (H, W, 8)
    H = op.circulant_matrix()                     # (8, 8)
    eg = np.einsum("mn,hwn->mhw", H, p)           # e(m) = sum_n H[m,n] p(n)
    return eg


def compass_edges(
    image: np.ndarray,
    operator: "str | CompassOperator" = "sobel",
    method: str = "spatial",
    mode: str = "reflect",
) -> np.ndarray:
    """Compute the eight directional edge values e(0)..e(7) of every pixel.

    Parameters
    ----------
    image:
        2-D grayscale image, shape ``(H, W)``.
    operator:
        ``"sobel"``, ``"prewitt"``, ``"kirsch"``, or a custom
        :class:`~compass_gradient.operators.CompassOperator`.
    method:
        ``"fft"`` for the transform-domain formulation of the paper,
        ``"spatial"`` for the equivalent direct mask correlation.
    mode:
        Border handling passed to :func:`numpy.pad`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(8, H, W)``. Index m corresponds to direction
        ``DIRECTIONS[m]`` = ("N", "NW", "W", "SW", "S", "SE", "E", "NE").
    """
    op = get_operator(operator)
    if method == "fft":
        return _edges_fft(image, op, mode)
    if method == "spatial":
        return _edges_spatial(image, op, mode)
    raise ValueError(f"Unknown method {method!r}; use 'fft' or 'spatial'.")


def edge_magnitude(edges: np.ndarray) -> np.ndarray:
    """Per-pixel edge magnitude: max_m e(m) (the classical compass rule).

    The signed maximum is used (the response of the winning mask).
    Because the eight directional responses of any zero-sum compass
    operator sum to zero at every pixel, this maximum is always
    non-negative, and — unlike max_m |e(m)| — it does not tie between
    opposite directions for symmetric operators such as Sobel/Prewitt
    (where e(m + 4) = -e(m)).
    """
    return np.max(edges, axis=0)


def edge_direction(edges: np.ndarray, labels: bool = False):
    """Per-pixel dominant direction: argmax_m e(m).

    The winning direction points toward the brighter side of the edge
    (the positive-weight side of the winning mask).

    Parameters
    ----------
    edges:
        Output of :func:`compass_edges`, shape ``(8, H, W)``.
    labels:
        If True, return an object array of direction names instead of
        integer indices.
    """
    idx = np.argmax(edges, axis=0)
    if labels:
        return np.asarray(DIRECTIONS, dtype=object)[idx]
    return idx


def detect(
    image: np.ndarray,
    operator: "str | CompassOperator" = "sobel",
    method: str = "fft",
    mode: str = "reflect",
) -> "tuple[np.ndarray, np.ndarray]":
    """Convenience wrapper returning ``(magnitude, direction_index)``."""
    edges = compass_edges(image, operator=operator, method=method, mode=mode)
    return edge_magnitude(edges), edge_direction(edges)
