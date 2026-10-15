from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CompassOperator",
    "SOBEL",
    "PREWITT",
    "KIRSCH",
    "get_operator",
    "from_eigenvalues",
    "DIRECTIONS",
    "NEIGHBOR_OFFSETS",
]

#: Direction names for e(0)..e(7).
DIRECTIONS: tuple[str, ...] = ("N", "NW", "W", "SW", "S", "SE", "E", "NE")

#: (row, col) offsets of the 8 neighbors relative to the center pixel,
#: enumerated counterclockwise starting from the top-left (see module
#: note on why counterclockwise).
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),  # n=0: top-left
    (0, -1),   # n=1: mid-left
    (1, -1),   # n=2: bottom-left
    (1, 0),    # n=3: bottom-mid
    (1, 1),    # n=4: bottom-right
    (0, 1),    # n=5: mid-right
    (-1, 1),   # n=6: top-right
    (-1, 0),   # n=7: top-mid
)

_M = 8  # period of the circulant structure


@dataclass(frozen=True)
class CompassOperator:
    """A 3x3 compass gradient operator defined by its N-mask weights h(i).

    Parameters
    ----------
    name:
        Human-readable operator name.
    h:
        Length-8 real weight vector ``[h(0), ..., h(7)]`` giving the
        weights of the north-directional mask laid out clockwise from
        the top-left position, as in Fig. 2a of the paper (center
        weight is zero by construction).
    """

    name: str
    h: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.h) != _M:
            raise ValueError(f"h must have length {_M}, got {len(self.h)}")
        if abs(sum(self.h)) > 1e-12:
            raise ValueError(
                "Compass gradient weights must sum to zero "
                f"(got sum={sum(self.h)!r})."
            )

    # -- circulant machinery -------------------------------------------------

    @property
    def h_array(self) -> np.ndarray:
        """Weights as a float64 array."""
        return np.asarray(self.h, dtype=np.float64)

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues r(k) of the circulant matrix H (Eq. 10).

        r(k) is the k-th DFT coefficient of ``[h(0), ..., h(7)]``.
        These are the transform-domain filtering coefficients; see
        Table 1 of the paper for Sobel / Prewitt / Kirsch.
        """
        return np.fft.fft(self.h_array)

    def circulant_matrix(self) -> np.ndarray:
        """The 8x8 circulant matrix H with H[m, n] = h((m - n) mod 8)."""
        idx = (np.arange(_M)[:, None] - np.arange(_M)[None, :]) % _M
        return self.h_array[idx]

    def masks(self) -> np.ndarray:
        """All eight 3x3 directional masks, shape ``(8, 3, 3)``.

        ``masks()[m]`` is the mask whose correlation with the image
        yields the directional edge value e(m); m follows
        :data:`DIRECTIONS`.
        """
        H = self.circulant_matrix()
        out = np.zeros((_M, 3, 3), dtype=np.float64)
        for m in range(_M):
            for n, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
                out[m, dr + 1, dc + 1] = H[m, n]
        return out


def from_eigenvalues(r: np.ndarray, name: str = "custom") -> CompassOperator:
    """Design a new compass operator from transform-domain coefficients.

    This is the design path suggested in Sec. 2.3 / Sec. 3 of the paper:
    a new edge operator can be obtained by choosing new eigenvalues r(k)
    in the transform domain.

    Parameters
    ----------
    r:
        Length-8 complex array of eigenvalues. For the resulting spatial
        weights h(i) to be real, ``r`` must be conjugate-symmetric
        (``r[k] == conj(r[(8 - k) % 8])``). For the weights to sum to
        zero (a gradient operator), ``r[0]`` must be 0.
    name:
        Name of the resulting operator.

    Returns
    -------
    CompassOperator
    """
    r = np.asarray(r, dtype=np.complex128)
    if r.shape != (_M,):
        raise ValueError(f"r must have shape ({_M},), got {r.shape}")
    if abs(r[0]) > 1e-9:
        raise ValueError("r[0] must be 0 for a gradient (zero-sum) operator.")
    conj_sym = np.conj(r[(-np.arange(_M)) % _M])
    if not np.allclose(r, conj_sym, atol=1e-9):
        raise ValueError(
            "r must be conjugate-symmetric so that the spatial weights are real."
        )
    h = np.fft.ifft(r).real
    return CompassOperator(name=name, h=tuple(h.tolist()))


SOBEL = CompassOperator("sobel", (1, 2, 1, 0, -1, -2, -1, 0))
PREWITT = CompassOperator("prewitt", (1, 1, 1, 0, -1, -1, -1, 0))
KIRSCH = CompassOperator("kirsch", (5, 5, 5, -3, -3, -3, -3, -3))

_REGISTRY: dict[str, CompassOperator] = {
    op.name: op for op in (SOBEL, PREWITT, KIRSCH)
}


def get_operator(operator: "str | CompassOperator") -> CompassOperator:
    """Resolve an operator name or pass a :class:`CompassOperator` through."""
    if isinstance(operator, CompassOperator):
        return operator
    try:
        return _REGISTRY[operator.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown operator {operator!r}; "
            f"available: {sorted(_REGISTRY)}"
        ) from None
