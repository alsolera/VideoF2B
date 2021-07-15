# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021  Andrey Vasilik - basil96@users.noreply.github.com
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

'''General geometry related to F2B figures.'''

from math import acos, asin, atan, cos, degrees, pi, radians, sqrt

import numpy as np
import numpy.linalg as LA
from scipy.optimize import fsolve

QUART_PI = 0.25 * pi


class Fillet:
    '''On a sphere of radius `R`, a fillet is defined as an arc of a minor circle of radius `r`
    between two great arcs of the sphere with angle `psi` between them.  This class can
    calculate the parameters of the fillet.
    If degrees is True, `psi` is in degrees, otherwise it is in radians (default).'''

    def __init__(self, R, r, psi, degrees=False):
        # Values that define the fillet
        self.R = R
        self.r = r
        self.psi = psi
        self.degrees = degrees
        if self.degrees:
            self.psi = radians(psi)
        # Values we calculate based on the defining values above
        self.alpha = None
        self.theta = None
        self.x_p = None
        self.y_p = None
        self.d = None
        self.beta = None
        self.calculate()

    def calculate(self):
        '''Calculate the fillet as follows:

        Given two intersecting planes with angle `psi` between them, a cone of slant height `R`
        and base radius `r` whose apex rests on the intersection line of the planes
        will rest tangent to both planes when the cone's axis makes an angle `theta`
        with the intersection line of the planes in the bisecting plane.

        If we set up a coordinate system on the cone's base such that:
            * the origin is at the cone's apex,
            * -Z axis is along the cone's axis toward the cone's base,
            * +X axis is toward the intersection line,
        then the coordinates of the points of tangency between the cone and the planes are:
            * (x_p, y_p, -d) and
            * (x_p, -y_p, -d)
        Let `beta` be the central angle of the arc along the cone's base that joins
        the two tangency points on the side of the intersection (shorter of the two possible arcs).

        Perform the following:
            * Ensure that tangency is possible. This requires that
                2 * alpha <= psi <= pi
                where `alpha` = asin(r/R) is the half-angle of the cone's apex.
            * Find angle `theta` (Gorjanc solution)
            * Find `x_p`, `y_p`, and `d`
            * Find angle `beta`.
        '''
        self.alpha = asin(self.r / self.R)
        if self.psi < 2. * self.alpha or self.psi > pi:
            # Tangency is not possible
            return False
        self.theta = Fillet.get_fillet_theta(self.R, self.r, self.psi)
        self.d = sqrt(self.R**2 - self.r**2)
        self.x_p = self.r / self.d * sqrt(
            (self.d**2 - self.r**2 + self.R**2 * cos(pi - self.psi)) / 2.0
        )
        self.y_p = sqrt(self.r**2 - self.x_p**2)
        self.beta = 2.0 * acos(self.x_p / self.r)
        return True

    @staticmethod
    def get_fillet_theta(R, r, psi):
        '''Calculate the angle between cone axis and intersection line.
        Gorjanc solution. Values of `x_p` and `y_p` follow from this as well.
        This method is provided static for optimizer use in addition to Fillet class use.'''
        theta = atan(
            r * sqrt(
                2.0 / (R**2 * (1 + cos(pi - psi)) - 2.0 * r**2)
            )
        )
        return theta


def get_equilateral_phi(sigma):
    '''Angle between sides of an equilateral spherical triangle
    as a function of side angle `sigma`.
    See https://www.av8n.com/physics/spherical-triangle.htm'''
    phi = acos(cos(sigma) / (cos(sigma)+1))
    return phi


def get_equilateral_height(sigma):
    '''Height of an equilateral spherical triangle
    as a function of side angle `sigma`.
    See https://math.stackexchange.com/a/3139032/10521'''
    h = acos(cos(sigma) / cos(0.5*sigma))
    return h


def get_cone_alpha(R, r):
    '''Half-angle of the apex of a cone with slant height `R` and base radius `r`.'''
    alpha = asin(r / R)
    return alpha


def calc_equilateral_sigma(height=QUART_PI):
    '''Calculate the side of an equilateral triangle whose target height is given, in radians.
    This is essentially the inverse of the `get_equilateral_height` function.
    Returns
        sigma (in radians).'''
    def root_finder(sigma):
        return get_equilateral_height(sigma) - height
    ret = fsolve(root_finder, QUART_PI)
    sigma = ret[0]
    return sigma


def calc_tri_loop_params(R, r, target_elev=QUART_PI):
    '''Calculate the basic parameters of a triangular loop on a sphere of radius `R`.
    Loop has corner turns of radius `r`, and the tops of those turns are tangent
    to an imaginary circle at `target_elev` on the sphere.

        Returns
            sigma, phi (all in radians)
    '''

    phi = 0.0

    def root_finder(sigma):
        '''Given an equilateral triangle on the surface of a sphere of radius R
        such that the top of a corner turn of radius `r` is located
        at `target_elev` on the sphere, calculate:
            * The central angle `sigma` of the side of the triangle,
            * The angle `phi` between adjacent sides of the triangle.
        '''
        nonlocal phi
        # Angle between adjacent sides of the triangle
        phi = get_equilateral_phi(sigma)
        # Angle between axis of corner cone and intersection line of the two planes where the cone falls
        theta = Fillet.get_fillet_theta(R, r, phi)
        # Half-angle of corner cone
        alpha = get_cone_alpha(R, r)
        # Height of the triangle as a function of sigma
        h = get_equilateral_height(sigma)
        # Finally, the objective function:
        # We want `sigma` such that `target_elev - alpha + theta = h`
        result = target_elev - alpha + theta - h
        # print('===== calc_tri_loop_params: root_finder ====================')
        # print(f'     R = {R}')
        # print(f'     r = {r}')
        # print(f'  elev = {target_elev} [{degrees(target_elev)} deg]')
        # print(f' sigma = {sigma} [{degrees(sigma)} deg]')
        # print(f'   phi = {phi} [{degrees(phi)} deg]')
        # print(f' theta = {theta} [{degrees(theta)} deg]')
        # print(f' alpha = {alpha} [{degrees(alpha)} deg]')
        # print(f'     h = {h} [{degrees(h)} deg]')
        # print(f'result = {result}')
        # print('=' * 40)
        return result

    ret = fsolve(root_finder, QUART_PI)
    sigma = ret[0]
    return sigma, phi


def angle(a, b):
    '''Angle between two vectors.'''
    inner_prod = np.inner(a, b)
    norms_prod = LA.norm(a) * LA.norm(b)
    if abs(norms_prod) < 1e-16:
        return np.NaN
    cos_theta = inner_prod / norms_prod
    theta = acos(np.clip(cos_theta, -1., 1.))
    return theta
