# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2022  Andrey Vasilik
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
Calculator for placing a camera in the field.
'''

import logging
from math import asin, atan2, cos, degrees, nan, radians, sin
from typing import Iterable

import numpy as np
import videof2b.core.geometry as geom
from scipy.optimize import fsolve
from videof2b.core.common import QUART_PI

log = logging.getLogger(__name__)


class CamDirector:
    '''The core calculator of cam placement geometry.'''

    DEFAULT_R = 21.0  # default sphere radius, in m.
    DEFAULT_C = -1.0  # default camera height relative to equator, in m.
    DEFAULT_G = -1.5  # default ground level relative to equator, in m.
    DEFAULT_A = 71.75  # default maximum vertical viewing angle of the camera, in degrees

    def __init__(self) -> None:
        self._R = CamDirector.DEFAULT_R
        self._C = CamDirector.DEFAULT_C
        self._G = CamDirector.DEFAULT_G
        self._A = CamDirector.DEFAULT_A
        self._d_min = nan
        self._d_max = nan
        self._alpha_at_min = nan
        self._alpha_at_max = nan
        self._tangent_elev_at_min = nan
        self._tangent_elev_at_max = nan
        self.solve()

    @property
    def R(self) -> float:
        '''Sphere radius. Must be positive and nonzero.'''
        return self._R

    @R.setter
    def R(self, val) -> None:
        if val > 0.:
            self._R = val
            self.solve()

    @property
    def C(self) -> float:
        '''Signed camera height relative to flight equator. Up is positive.'''
        return self._C

    @C.setter
    def C(self, val) -> None:
        self._C = val
        self.solve()

    @property
    def G(self) -> float:
        '''Signed ground level relative to flight equator. Up is positive.'''
        return self._G

    @G.setter
    def G(self, val) -> None:
        self._G = val
        self.solve()

    @property
    def A(self) -> float:
        '''Maximum vertical viewing angle of the camera, in degrees.'''
        return self._A

    @A.setter
    def A(self, val) -> None:
        if val > 0.:
            self._A = val
            self.solve()

    @property
    def cam_distance_limits(self) -> Iterable:
        '''
        The limits (min, max) of camera distance from flight center.
        '''
        return (self._d_min, self._d_max)

    @property
    def cam_view_limits(self) -> Iterable:
        '''
        The limits (min, max) of camera vertical viewing angle, in degrees,
        at the respective `cam_distance_limits`.
        '''
        return (self._alpha_at_min, self._alpha_at_max)

    @property
    def cam_tangent_elev_limits(self) -> Iterable:
        '''
        The limits (min, max) of the elevation of the tangent point from camera to sphere
        at the respective `cam_distance_limits`.
        The point's elevation is measured from flight center in degrees.
        '''
        return (self._tangent_elev_at_min, self._tangent_elev_at_max)

    @staticmethod
    def _get_3point_angle(m):
        '''
        Return the angle between u and v, where
                u = m[1] - m[0]
                v = m[2] - m[0]
        '''
        u = m[1] - m[0]
        v = m[2] - m[0]
        return geom.angle(u, v)

    @staticmethod
    def _get_elevation_angle(p):
        '''Given point p in XY space, determine its elevation angle above x-axis.'''
        return np.arctan2(abs(p[1]), abs(p[0]))

    def _get_tangent_point(self, p):
        '''Given point p, find coordinates of a point q
        such that [pq] is tangent to circle with radius R whose center is at the origin.'''
        # circle is at origin, so `d = c - p` is just `-p`
        d = -np.atleast_1d(p)
        d_norm = np.linalg.norm(d)
        a = asin(self._R / d_norm)
        b = atan2(d[1], d[0])
        t = b - a
        ta = self._R * np.array([sin(t), -cos(t)])
        t = b + a
        tb = self._R * np.array([-sin(t), cos(t)])
        # print(f'tangent points: {ta}, {tb}')
        # Pick the tangent point with the higher y coord
        if ta[1] > tb[1]:
            return ta
        return tb

    def _solve_closest_d(self):
        '''Solve the current state for the closest distance to flight circle
        where the vertical view angle approaches the maximum allowed by the camera.'''

        target_alpha = radians(self._A)

        def func(x):
            p0 = np.r_[x, self._C]
            try:
                p1 = self._get_tangent_point(p0)
            except:
                return 1e6
            p2 = [0, self._G]
            P = np.vstack((p0, p1, p2))
            alpha = CamDirector._get_3point_angle(P)
            return target_alpha - alpha

        ret = fsolve(func, self._R + self._C)
        return ret[0]

    def solve(self):
        '''
        For the current state, solve for the following:
            `d_min` and `d_max` such that `alpha <= max_alpha`
        Subject to the constraint:
            * maximum vertical view angle must include the 45-degree latitude
              and the center field marker on the ground.
        '''
        R = self._R
        h = self._C
        p2 = [0, self._G]
        # 45deg tangent line equation
        b = R * (sin(QUART_PI) + cos(QUART_PI))
        # Find `x_max` (where tangent line intersects the camera height)
        x_max = h - b
        p0_max = [x_max, h]
        p1_max = self._get_tangent_point(p0_max)
        p_max = np.vstack((p0_max, p1_max, p2))
        self._alpha_at_max = degrees(CamDirector._get_3point_angle(p_max))
        # `x_min` is limited by sphere radius at one extreme
        # and by self.A on the other.
        x_min_for_alpha = self._solve_closest_d()
        log.debug('x_min_for_alpha (initial) = %s' % x_min_for_alpha)
        if x_min_for_alpha > 0.:
            x_min_for_alpha *= -1.0
            log.debug('x_min_for_alpha (flipped) = %s' % x_min_for_alpha)
        # Both x_min candidate values are negative, so take the min of the two.
        x_min = min(-R, x_min_for_alpha)
        # Use absolute value for the user because it's just a distance.
        self._d_min = abs(x_min)
        self._d_max = abs(x_max)
        # Output
        if self._d_min > self._d_max:
            self._d_min, self._d_max = self._d_max, self._d_min
            x_min, x_max = x_max, x_min
        #
        p0_min = [x_min, h]
        p1_min = self._get_tangent_point(p0_min)
        p_min = np.vstack((p0_min, p1_min, p2))
        self._alpha_at_min = degrees(CamDirector._get_3point_angle(p_min))
        self._tangent_elev_at_min = degrees(CamDirector._get_elevation_angle(p1_min))
        p0_max = [x_max, h]
        p1_max = self._get_tangent_point(p0_max)
        p_max = np.vstack((p0_max, p1_max, p2))
        self._alpha_at_max = degrees(CamDirector._get_3point_angle(p_max))
        self._tangent_elev_at_max = degrees(CamDirector._get_elevation_angle(p1_max))
        #
        log.debug('Solved CamDirector with inputs:\n'
                  '    R=%.3f\n'
                  '    C=%.3f\n'
                  '    G=%.3f\n'
                  '    A=%.2f\n' %
                  (self._R, self._C, self._G, self._A))
        log.debug('outputs:\n'
                  '    D limits=%s\n'
                  '    V limits=%s\n'
                  '    T limits=%s\n' %
                  (self.cam_distance_limits, self.cam_view_limits, self.cam_tangent_elev_limits))
