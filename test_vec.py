#! /usr/bin/env python

import unittest
import math

from vec import *

test_vectors = [
    (2, 4),
    (2.5, 53.342),
    (-2, -4),
    (1/3.0, 2.323423),
    (0, -3),
    (53, 0),
    (1, 2, 3),
]
specialvals = [
    (0,0),
]

class TestVec(unittest.TestCase):
    def test_mag(self):
        v = (3, -4)
        self.assertEqual(mag(v), 5) # 3-4-5 triangle

    def test_norm(self):
        v = (3, -4)
        v = norm(v, 10) # set magnitude to 10
        self.assertEqual(v, (6,-8)) # assert that it's a 6-8-10 trangle now
        for v in test_vectors:
            self.assertAlmostEqual(mag(norm(v)), 1.0) # magnitude of a normalized vector is 1
            self.assertEqual(v, mul(norm(v), mag(v))) # a vector is equal to its direction times its magnitude

    def test_mag2(self):
        for v in test_vectors + specialvals:
            self.assertAlmostEqual(mag2(v), mag(v)**2)

    def test_neg(self):
        self.assertEqual(neg((-2, 5)), (2, -5))
        self.assertEqual(neg((23, 53)), (-23, -53))

    def test_add(self):
        self.assertEqual(add((1, 1), (3, 5)), (4, 6))
        self.assertEqual(add((1, 1, 1), (1, 1)), (2, 2, 1))
        self.assertEqual(
            add(*((.1, .2, .3) for i in range(10))),
            (1.0, 2.0, 3.0),
        )
        # Don't use fsum unless there are floats.
        self.assertEqual(str(add((1.0,), (2,))), '(3.0,)')
        self.assertEqual(str(add((1.0, 1), (2.0, 2))), '(3.0, 3)')
        self.assertEqual(str(add((1,), (2,))), '(3,)')

    def test_vfrom(self):
        self.assertEqual(vfrom((1, 1), (2, 2)), (1, 1))
        self.assertEqual(vfrom((0, 0), (1, 1, 1)), (1, 1, 1))

    def test_mul(self):
        self.assertEqual(mul((2, 3), 2), (4, 6))
        self.assertEqual(mul((5, -5), .5), (2.5, -2.5))

    def test_div(self):
        self.assertEqual(div((2, 3), 2), (1, 1.5))
        self.assertEqual(div((5, -5), .5), (10, -10))

    def test_dot(self):
        self.assertEqual(dot((1, 2), (3, 4)), 11)
        self.assertEqual(dot((1,), (3, 4, 0, 0)), 3)
        for v in test_vectors:
            self.assertEqual(dot(v, (0, 0)), 0)

    def test_angle(self):
        test_angles = (((0,1), (1,0), math.pi/2),
                      ((7,-7), (1,0), math.pi/4),
                      ((2,0), (-1,0), math.pi),
                      ((1,0), (1,math.sqrt(3)), math.pi/3),
                      ((3,0), (math.sqrt(3),1), math.pi/6),
                      ((2,2), (1,-1), math.pi/2),
                      ((5,5), (-math.sqrt(3),1), math.pi/4 + math.pi/3))
        for (v1, v2, a) in test_angles:
            self.assertAlmostEqual(angle(v1, v2), a)
            self.assertAlmostEqual(angle(v2, v1), a) # both orders
        for v in test_vectors:
            self.assertAlmostEqual(angle(v, v), 0) # same vector makes 0 radians

    def test_perp(self):
        from cmath import phase

        two_pi = 2 * math.pi
        half_pi = math.pi / 2

        def wrap_angle(a):
            """Wrap an angle so that it is between -pi and pi."""
            while a > math.pi:
                a -= two_pi
            while a < -math.pi:
                a += two_pi
            return a

        self.assertEqual(perp((1, 2)), (2, -1))
        for v in test_vectors:
            if len(v) != 2:
                continue
            p = perp(v)
            self.assertEqual(dot(v, p), 0)
            self.assertEqual(angle(v, p), half_pi)

            av = phase(complex(*v))
            ap = phase(complex(*p))
            self.assertEqual(wrap_angle(av - ap), half_pi)
            self.assertEqual(wrap_angle(ap - av), -half_pi)

    def test_proj(self):
        self.assertEqual(
            proj((1, 1), (1, 0)),
            (1, 0),
        )
        self.assertEqual(
            proj((1, 3), (4, 2)),
            (2, 1),
        )


if __name__ == '__main__':
    unittest.main()
