"""A set of simple vector utility functions.

All functions take constant arguments, and return a result; nothing
is modified in-place.
"""

from __future__ import division

__all__ = ['add', 'vfrom', 'dot', 'cross', 'mul', 'div', 'neg', 'mag2',
           'mag', 'dist2', 'dist', 'norm', 'avg', 'angle', 'rotate', 'perp',
           'proj', 'heading', 'from_heading']

from math import sqrt, acos, fsum, sin, cos, atan2
from itertools import zip_longest
from typing import Iterator, List, Tuple, Optional


epsilon = 1e-10


def float_equal(a: float, b: float) -> bool:
    return abs(a - b) < epsilon


Vec = List[float]  # A vector is a sequence of numbers.


def _zip(*vecs: Vec) -> Iterator[Tuple[float, float]]:
    return zip_longest(*vecs, fillvalue=0.0)


def add(*vecs: Vec) -> Vec:
    """Calculate the vector addition of two or more vectors."""
    return [ fsum(dim) for dim in _zip(*vecs) ]


def sub(v1: Vec, v2: Vec) -> Vec:
    """Subtract one vector from another"""
    return [ (n1 - n2) for (n1, n2) in _zip(v1, v2) ]


def vfrom(p1: Vec, p2: Vec) -> Vec:
    """Return the vector from p1 to p2."""
    return sub(p2, p1)


def dot(v1: Vec, v2: Vec) -> float:
    """Calculate the dot product of two vectors."""
    return sum((n1 * n2) for n1, n2 in zip(v1, v2))


def mul(v: Vec, c: float) -> Vec:
    """Multiply a vector by a scalar."""
    return [ n*c for n in v ]


def div(v: Vec, c: float) -> Vec:
    """Divide a vector by a scalar."""
    return [ n/c for n in v ]


def neg(v: Vec) -> Vec:
    """Invert a vector."""
    return [ -n for n in v ]


def mag2(v: Vec) -> float:
    """Calculate the squared magnitude of a vector."""
    return sum(n**2 for n in v)


def mag(v: Vec) -> float:
    """Calculate the magnitude of a vector."""
    return sqrt(mag2(v))


def dist2(p1: Vec, p2: Vec) -> float:
    """Find the squared distance between two points."""
    return mag2(vfrom(p1, p2))


def dist(p1: Vec, p2: Vec) -> float:
    """Find the distance between two points."""
    return mag(vfrom(p1, p2))


def norm(v: Vec, c: float = 1) -> Vec:
    """Return a vector in the same direction as v, with magnitude c."""
    return mul(v, c/mag(v))


def avg(*vecs: Vec) -> Vec:
    """Find the vector average of two or more points."""
    return div(add(*vecs), len(vecs))


def angle(v1: Vec, v2: Vec) -> float:
    """Find the angle in radians between two vectors."""
    ratio = dot(v1, v2) / (mag(v1) * mag(v2))
    ratio = _clamp(ratio, -1.0, 1.0)
    if abs(ratio - 1.0) < epsilon:
        return 0.0
    return acos(ratio)


def _clamp(value: float, lo: float, hi: float) -> float:
    value = max(lo, value)
    value = min(hi, value)
    return value


def rotate(v: Vec, theta: float) -> Vec:
    """Rotate a two-dimensional vector counter-clockwise by the given angle."""
    x, y = v
    sin_a = sin(theta)
    cos_a = cos(theta)
    return [
        x * cos_a - y * sin_a,
        x * sin_a + y * cos_a,
    ]


def cross(v1: Vec, v2: Vec) -> Vec:
    """Calculate the cross product of two three-dimensional vectors."""
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return [
        y1*z2 - z1*y2,
        z1*x2 - x1*z2,
        x1*y2 - y1*x2,
    ]


def perp(v: Vec) -> Vec:
    """
    Return a perpendicular to a two-dimensional vector.

    The direction of rotation is 90 degrees counterclockwise.
    """
    x, y = v
    return [-y, x]


def proj(v1: Vec, v2: Vec) -> Vec:
    """Calculate the vector projection of v1 onto v2."""
    return mul(v2, dot(v1, v2) / mag2(v2))


def heading(v: Vec) -> float:
    """
    Return the heading angle of the two-dimensional vector v.

    This is equivalent to the theta value of v in polar coordinates.

    Raises ValueError if passed a zero vector.
    """
    if list(v) == [0, 0]:
        raise ValueError('A zero vector has no heading.')
    x, y = v
    return atan2(y, x)


def from_heading(heading: float, c: float = 1) -> Vec:
    """
    Create a two-dimensional vector with the specified heading of the specified magnitude.
    """
    return rotate([c, 0], heading)


# Geometry

Line = Tuple[Vec, Vec]  # A line is defined by two points.


def bisector(a: Vec, b: Vec) -> Line:
    mid = avg(a, b)
    ab = vfrom(a, b)
    return (mid, mid + perp(ab))


def intersect_lines(line1: Line, line2: Line, segment: bool = False) -> Optional[Vec]:
    """
    Find the intersection of lines a-b and c-d.

    If the "segment" argument is true, treat the lines as segments, and check
    whether the intersection point is off the end of either segment.
    """
    a, b = line1
    c, d = line2

    # Reference:
    # http://geomalgorithms.com/a05-_intersect-1.html
    u = vfrom(a, b)
    v = vfrom(c, d)
    w = vfrom(c, a)

    u_perp_dot_v = dot(perp(u), v)
    if float_equal(u_perp_dot_v, 0):
        return None  # We have collinear segments, no single intersection.

    v_perp_dot_w = dot(perp(v), w)
    s = v_perp_dot_w / u_perp_dot_v
    if segment and (s < 0 or s > 1):
        return None

    u_perp_dot_w = dot(perp(u), w)
    t = u_perp_dot_w / u_perp_dot_v
    if segment and (t < 0 or t > 1):
        return None

    return add(a, mul(u, s))
