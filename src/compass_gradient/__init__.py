"""compass_gradient: 8-directional compass edge detection via circulant/FFT.

Implementation of:

    R.-H. Park and W. Y. Choi, "A new interpretation of the compass
    gradient edge operators," Computer Vision, Graphics, and Image
    Processing, vol. 47, pp. 259-265, 1989.
"""

from .core import (
    compass_edges,
    detect,
    edge_direction,
    edge_magnitude,
    neighbor_stack,
)
from .operators import (
    DIRECTIONS,
    KIRSCH,
    PREWITT,
    SOBEL,
    CompassOperator,
    from_eigenvalues,
    get_operator,
)

__version__ = "0.1.0"

__all__ = [
    "compass_edges",
    "detect",
    "edge_magnitude",
    "edge_direction",
    "neighbor_stack",
    "CompassOperator",
    "SOBEL",
    "PREWITT",
    "KIRSCH",
    "get_operator",
    "from_eigenvalues",
    "DIRECTIONS",
    "__version__",
]
