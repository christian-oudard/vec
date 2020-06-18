from pytest import approx, raises

from math import sqrt, pi, radians

from vec import (
    Circle, equal, add, vfrom, dot, cross, mul, div, neg, mag2, mag, dist2, dist, norm, avg, angle, rotate, perp, proj,
    heading, from_heading, bisector, intersect_lines, intersect_circles, circle_3_points, side
)

test_vectors = [
    [2, 4],
    [2.5, 53.342],
    [-2, -4],
    [1/3.0, 2.323423],
    [0, -3],
    [53, 0],
    [1, 2, 3],
]
specialvals = [
    [0, 0],
]


def assert_vec_almost_equal(v1, v2):
    assert len(v1) == len(v2)
    for f1, f2 in zip(v1, v2):
        assert f1 == approx(f2)


def test_mag():
    v = [3, -4]
    mag(v) == 5  # 3-4-5 triangle.


def test_norm():
    v = [3, -4]
    v = norm(v, 10)  # Set magnitude to 10.
    v == [6, -8]  # It's a 6-8-10 trangle now.
    for v in test_vectors:
        assert mag(norm(v)) == approx(1.0)  # Magnitude of a normalized vector is 1.
        assert v == mul(norm(v), mag(v))  # A vector is equal to its direction times its magnitude.


def test_mag2():
    for v in test_vectors + specialvals:
        assert mag2(v) == approx(mag(v)**2)


def test_neg():
    assert neg([-2, 5]) == [2, -5]
    assert neg([23, 53]) == [-23, -53]


def test_add():
    assert add([1, 1], [3, 5]) == [4, 6]
    assert add([1, 1, 1], [1, 1]) == [2, 2, 1]
    assert (
        add(*([.1, .2, .3] for i in range(10)))
        == [1.0, 2.0, 3.0]
    )
    assert str(add([1.0], [2])) == '[3.0]'
    assert str(add([1.0, 1], [2.0, 2])) == '[3.0, 3.0]'
    assert str(add([1], [2])) == '[3.0]'


def test_vfrom():
    assert vfrom([1, 1], [2, 2]) == [1, 1]
    assert vfrom([0, 0], [1, 1, 1]) == [1, 1, 1]


def test_mul():
    assert mul([2, 3], 2) == [4, 6]
    assert mul([5, -5], .5) == [2.5, -2.5]


def test_div():
    assert div([2, 3], 2) == [1, 1.5]
    assert div([5, -5], .5) == [10, -10]


def test_dot():
    assert dot([1, 2], [3, 4]) == 11
    assert dot([1], [3, 4, 0, 0]) == 3
    for v in test_vectors:
        assert dot(v, [0, 0]) == 0


def test_angle():
    test_angles = [
        ([0, 1], [1, 0], pi/2),
        ([7, -7], [1, 0], pi/4),
        ([2, 0], [-1, 0], pi),
        ([1, 0], [1, sqrt(3)], pi/3),
        ([3, 0], [sqrt(3), 1], pi/6),
        ([2, 2], [1, -1], pi/2),
        ([5, 5], [-sqrt(3), 1], pi/4 + pi/3),
        ([9.781476007338057, 2.0791169081775935], (9.781476007338057, 2.0791169081775935), 0.0),
    ]
    for (v1, v2, a) in test_angles:
        assert angle(v1, v2) == approx(a)
        assert angle(v2, v1) == approx(a)
        assert abs(heading(v1) - heading(v2)) == approx(a)
        assert abs(heading(v2) - heading(v1)) == approx(a)
    for v in test_vectors:
        print(v)
        assert angle(v, v) == approx(0)  # Same vector makes 0 radians.


def test_heading():
    for v in test_vectors:
        if len(v) != 2:
            continue
        h = heading(v)
        assert_vec_almost_equal(
            v,
            from_heading(h, mag(v)),
        )


def test_zero_heading_error():
    with raises(ValueError):
        heading([0, 0])


def test_perp():
    from cmath import phase

    two_pi = 2 * pi
    half_pi = pi / 2

    def wrap_angle(a):
        """Wrap an angle so that it is between -pi and pi."""
        while a > pi:
            a -= two_pi
        while a < -pi:
            a += two_pi
        return a

    assert perp([1, 2]) == [-2, 1]
    for v in test_vectors:
        if len(v) != 2:
            continue
        p = perp(v)
        assert angle(v, p) == approx(half_pi)
        assert dot(v, p) == 0

        av = phase(complex(*v))
        ap = phase(complex(*p))
        assert wrap_angle(av - ap) == approx(-half_pi)
        assert wrap_angle(ap - av) == approx(half_pi)


def test_proj():
    assert (
        proj([1, 1], [1, 0])
        == [1, 0]
    )
    assert (
        proj([1, 3], [4, 2])
        == [2, 1]
    )


def test_rotate():
    v = [1, 0]
    assert_vec_almost_equal(rotate(v, radians(0)), (1, 0))
    assert_vec_almost_equal(rotate(v, radians(90)), (0, 1))
    assert_vec_almost_equal(rotate(v, radians(180)), (-1, 0))
    assert_vec_almost_equal(rotate(v, radians(-90)), (0, -1))
    assert_vec_almost_equal(rotate(v, radians(30)), (0.8660254037844386, 0.5))


def test_avg():
    assert avg([1, 2], [2, 3], [3, 1]) == [2, 2]


def test_dist():
    assert dist2([3, 0], [0, 4]) == 25
    assert dist([3, 0], [0, 4]) == 5


def test_cross():
    assert cross([3, -3, 1], [4, 9, 2]) == [-15, -2, 39]


def test_intersect_lines():
    assert_vec_almost_equal(
        intersect_lines(
            ([0, 0], [10, 10]),
            ([0, 10], [10, 0]),
        ),
        [5, 5],
    )
    assert_vec_almost_equal(
        intersect_lines(
            ([0, 0], [10, 0]),
            ([5, 0], [15, 0.01]),
        ),
        [5, 0],
    )
    assert intersect_lines(
        ([0, 0], [1, 0]),
        ([0, 1], [1, 1]),
    ) is None
    assert intersect_lines(
        ([0, 0], [1, 0]),
        ([2, 1], [2, -1]),
        segment=True,
    ) is None
    assert intersect_lines(
        ([2, 1], [2, -1]),
        ([0, 0], [1, 0]),
        segment=True,
    ) is None


def test_bisector():
    assert bisector([0, 0], [1, 1]) == ([1/2, 1/2], [0, 1])


def test_circle_3_points():
    circle = circle_3_points([0, 0], [1, 0], [0, 1])
    assert circle.c == [1/2, 1/2]
    assert circle.r == approx(sqrt(2) / 2)


def test_side():
    assert side([0, 2], [3, 0], [4, 4]) == 1
    assert side([3, 0], [0, 2], [4, 4]) == -1
    assert side([3, 0], [0, 2], [-3, 4]) == 0


def test_intersect_circles():
    # Coincident circles, no single intersection point.
    assert intersect_circles(
        Circle([0, 0], 1),
        Circle([0, 0], 1),
    ) == []

    # No intersection, separated circles.
    assert intersect_circles(
        Circle([0, 0], 1),
        Circle([5, 0], 1),
    ) == []

    # No intersection, concentric circles.
    assert intersect_circles(
        Circle([0, 0], 1),
        Circle([0, 0], 2),
    ) == []

    # One point, exterior tangent.
    assert intersect_circles(
        Circle([0, 0], 1),
        Circle([2, 0], 1),
    ) == [[1, 0]]

    # One point, interior tangent.
    assert intersect_circles(
        Circle([0, 0], 2),
        Circle([1, 0], 1),
    ) == [[2, 0]]

    assert intersect_circles(
        Circle([0, 1], 1.5),
        Circle([0, 0], 2.5),
    ) == [[0, 2.5]]

    # Two points, same size circles.
    assert intersect_circles(
        Circle([-1, 0], sqrt(2)),
        Circle([1, 0], sqrt(2)),
    ) == [[0, 1], [0, -1]]

    # Two points, different size circles.
    p1, p2 = intersect_circles(
        Circle([0, 0], sqrt(2)),
        Circle([1, 0], 1),
    )
    assert equal(p1, [1, 1])
    assert equal(p2, [1, -1])


def test_intersect_circles_numerical():
    assert intersect_circles(
        Circle([-27.073924841728974, 65.92689560740814], -1.25),
        Circle([0.5, 0.5], -72.25000000000001),
    ) == [[-27.55938126499886, 67.07877757232733]]
