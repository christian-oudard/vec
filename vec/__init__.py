"""A set of simple vector utility functions.

All functions take constant arguments, and return a result; nothing
is modified in-place.
"""

from collections import namedtuple
from math import sqrt, acos, fsum, sin, cos, atan2
from itertools import zip_longest
from typing import Iterator, List, Sequence, Tuple, Optional


epsilon = 1e-10

Vec = Sequence[float]  # A vector is a sequence of numbers.
Line = Tuple[Vec, Vec]  # A line is defined by two points.
Circle = namedtuple('circle', 'c r')  # A circle is defined by a center and a radius.


def _float_equal(a: float, b: float) -> bool:
    return abs(a - b) < epsilon


def equal(a: Vec, b: Vec) -> bool:
    """Check if two vectors are equal, using a floating-point epsilon comparison."""
    return all( _float_equal(ai, bi) for (ai, bi) in zip(a, b) )


def unique(vecs: Iterator[Vec]) -> Iterator[Vec]:
    """Yield unique elements of an iterable, using a floating-point epsilon comparison."""
    seen = []
    for v in vecs:
        for seen_v in seen:
            if equal(v, seen_v):
                break
        else:
            seen.append(v)
            yield v


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

def bisector(a: Vec, b: Vec) -> Line:
    mid = avg(a, b)
    normal = perp(vfrom(mid, b))
    p = add(mid, normal)
    return (mid, p)


def intersect_lines(line1: Line, line2: Line, segment: bool = False, include_endpoints: bool = True) -> Optional[Vec]:
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
    if _float_equal(u_perp_dot_v, 0):
        return None  # We have collinear segments, no single intersection.

    v_perp_dot_w = dot(perp(v), w)
    s = v_perp_dot_w / u_perp_dot_v

    u_perp_dot_w = dot(perp(u), w)
    t = u_perp_dot_w / u_perp_dot_v

    result = add(a, mul(u, s))
    if not segment:
        return result

    # Handle segment endpoints.
    for m in [s, t]:
        if (
            _float_equal(m, 0) or
            _float_equal(m, 1)
        ):
            if include_endpoints:
                return result
            else:
                return None
        elif m < 0 or m > 1:
            return None

    return result


def circle_3_points(a: Vec, b: Vec, c: Vec) -> Optional[Circle]:
    center = intersect_lines(bisector(a, b), bisector(b, c))
    if center is None:
        return None
    radius = mag(vfrom(center, a))
    return Circle(center, radius)


def circle_2_points_radius(a: Vec, b: Vec, radius: float) -> Optional[Circle]:
    """
    Find the circle that intersects the two given points with the given radius.
    In general, there will be two solutions. The lefthand solution is given by specifying a positive radius,
    and the righthand solution is given by specifying a negative radius.
    """
    ccw = radius > 0
    radius = abs(radius)
    points = intersect_circles(Circle(a, radius), Circle(b, radius))
    if len(points) == 0:
        return None
    elif len(points) == 1:
        return Circle(points[0], radius)
    else:  # len(points) == 2
        i1, i2 = points
        assert side(a, b, i1) == 1
        assert side(a, b, i2) == -1
        if ccw:
            return Circle(i1, radius)
        else:
            return Circle(i2, radius)


def intersect_circles(circ1: Circle, circ2: Circle) -> List[Vec]:
    radius1 = abs(circ1.r)
    radius2 = abs(circ2.r)

    if radius2 > radius1:
        return intersect_circles(circ2, circ1)

    transverse = vfrom(circ1.c, circ2.c)
    dist = mag(transverse)

    # Check for identical or concentric circles. These will have either
    # no points in common or all points in common, and in either case, we
    # return an empty list.
    if equal(circ1.c, circ2.c):
        return []

    # Check for exterior or interior tangent.
    radius_sum = radius1 + radius2
    radius_difference = abs(radius1 - radius2)
    if (
        _float_equal(dist, radius_sum)
        or _float_equal(dist, radius_difference)
    ):
        return [
            add(
                circ1.c,
                norm(transverse, radius1)
            ),
        ]

    # Check for non intersecting circles.
    if dist > radius_sum or dist < radius_difference:
        return []

    # If we've reached this point, we know that the two circles intersect
    # in two distinct points.
    # Reference:
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html

    # Pretend that the circles are arranged along the x-axis.
    # Find the x-value of the intersection points, which is the same for both
    # points. Then find the chord length "a" between the two intersection
    # points, and use vector math to find the points.
    dist2 = mag2(transverse)
    x = (dist2 - radius2**2 + radius1**2) / (2 * dist)
    a = (
        (1 / dist)
        * sqrt(
            (-dist + radius1 - radius2)
            * (-dist - radius1 + radius2)
            * (-dist + radius1 + radius2)
            * (dist + radius1 + radius2)
        )
    )
    chord_middle = add(
        circ1.c,
        norm(transverse, x),
    )
    normal = perp(transverse)
    return [
        add(chord_middle, norm(normal, a / 2)),
        add(chord_middle, norm(normal, -a / 2)),
    ]


def side(a: Vec, b: Vec, c: Vec) -> int:
    """
    Determine whether the points a, b, and c are clockwise, counterclockwise, or collinear.
    Return -1 for clockwise, +1 for counterclockwise, and 0 for collinear.

    Reference:
    http://geomalgorithms.com/a01-_area.html
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    val = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(val) < epsilon:
        return 0
    elif val > 0:
        return 1
    else:  # val < 0
        return -1
