"""A set of simple vector utility functions.

All functions take constant arguments, and return a result; nothing
is modified in-place.
"""
from __future__ import division

__all__ = ['add', 'vfrom', 'dot', 'cross', 'mul', 'div', 'neg', 'mag2',
           'mag', 'dist2', 'dist', 'norm', 'avg', 'angle', 'rotate', 'perp',
           'proj', 'heading']

from math import sqrt, acos, fsum, sin, cos, atan2
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip, repeat, chain
    def zip_longest(*args, **kwds):
        fillvalue = kwds.get('fillvalue')
        def sentinel(counter=[fillvalue]*(len(args)-1)):
            yield counter.pop()     # raises IndexError when count hits zero
        iters = [chain(it, sentinel(), repeat(fillvalue)) for it in args]
        try:
            for tup in izip(*iters):
                yield tup
        except IndexError:
            pass

def vzip(*vecs):
    return zip_longest(*vecs, fillvalue=0)

def _iter_add(*vecs):
    for dim in vzip(*vecs):
        sum_func = fsum if any(isinstance(i, float) for i in dim) else sum
        yield sum_func(dim)

def add(*vecs):
    """Calculate the vector addition of two or more vectors."""
    return tuple(_iter_add(*vecs))

def sub(v1, v2):
    """Subtract one vector from another"""
    return vfrom(v2, v1)

def vfrom(p1, p2):
    """Return the vector from p1 to p2."""
    return tuple((n2 - n1) for n1, n2 in vzip(p1, p2))

def dot(v1, v2):
    """Calculate the dot product of two vectors."""
    return sum((n1 * n2) for n1, n2 in zip(v1, v2))

def mul(v, c):
    """Multiply a vector by a scalar."""
    return tuple(n*c for n in v)

def div(v, c):
    """Divide a vector by a scalar."""
    return tuple(n/c for n in v)

def neg(v):
    """Invert a vector."""
    return tuple(-n for n in v)

def mag2(v):
    """Calculate the squared magnitude of a vector."""
    return sum(n**2 for n in v)

def mag(v):
    """Calculate the magnitude of a vector."""
    return sqrt(mag2(v))

def dist2(p1, p2):
    """Find the squared distance between two points."""
    return mag2(vfrom(p1, p2))

def dist(p1, p2):
    """Find the distance between two points."""
    return mag(vfrom(p1, p2))

def norm(v, c=1):
    """Return a vector in the same direction as v, with magnitude c."""
    return mul(v, c/mag(v))

def avg(*args):
    """Find the vector average of two or more points."""
    return div(add(*args), len(args))

def angle(v1, v2):
    """Find the angle in radians between two vectors."""
    return acos(dot(v1, v2) / (mag(v1) * mag(v2)))

def rotate(v, theta):
    """Rotate a two-dimensional vector counter-clockwise by the given angle."""
    x, y = v
    sin_a = sin(theta)
    cos_a = cos(theta)
    return (
        x * cos_a - y * sin_a,
        x * sin_a + y * cos_a,
    )

def cross(v1, v2):
    """Calculate the cross product of two three-dimensional vectors."""
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return (y1*z2 - z1*y2,
            z1*x2 - x1*z2,
            x1*y2 - y1*x2)

def perp(v):
    """Return a perpendicular to a two-dimensional vector."""
    x, y = v
    return (y, -x)

def proj(v1, v2):
    """Calculate the vector projection of v1 onto v2."""
    return mul(v2, dot(v1, v2) / mag2(v2))

def heading(v):
    """
    Return the heading angle of the two-dimensional vector v.

    This is equivalent to the theta value of v in polar coordinates.
    """
    x, y = v
    return atan2(y, x)
