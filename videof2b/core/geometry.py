# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021 - 2022  Andrey Vasilik - basil96
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

from math import (acos, asin, atan, atan2, cos, degrees, pi, radians, sin,
                  sqrt, tan)

import numpy as np
import numpy.linalg as LA
from scipy.optimize import fsolve
from videof2b.core.common import QUART_PI, TWO_PI


class ArgumentError(Exception):
    '''Thrown when the provided combination of arguments is invalid.'''


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
        self.is_valid = False
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
        self.alpha = get_cone_alpha(self.R, self.r)
        if self.psi < 2. * self.alpha or self.psi > pi:
            # Tangency is not possible
            self.is_valid = False
            return
        self.theta = Fillet.get_fillet_theta(self.R, self.r, self.psi)
        self.d = get_cone_d(self.R, self.r)
        self.x_p = self.r / self.d * sqrt(
            (self.d**2 - self.r**2 + self.R**2 * cos(pi - self.psi)) / 2.0
        )
        self.y_p = sqrt(self.r**2 - self.x_p**2)
        self.beta = 2.0 * acos(self.x_p / self.r)
        self.is_valid = True

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


def get_arc(r, alpha, rho=100):
    # TODO: change meaning of `rho` from angular density to circumferential (linear) density
    # for more consistent point spacing on arcs of different radius.
    '''Return 3D points for an arc of radius `r` and included angle `alpha`
    with point density `rho`, where `rho` is number of points per 2*pi.
    Arc center is (0, 0, 0).  The arc lies in the XY plane.
    Arc starts at zero angle, i.e., at (r, 0, 0) coordinate, and ends CCW at `alpha`.
    Angle measurements are in radians.
    Endpoint is always included.
    '''
    nom_step = TWO_PI / rho
    num_pts = int(alpha / nom_step)
    if num_pts < 3:
        # Prevent degenerate arcs
        num_pts = 3
    act_step = alpha / num_pts
    if act_step > nom_step:
        num_pts += 1
        act_step = alpha / num_pts
    pts = np.array(
        [
            (r*cos(act_step*t),
                r*sin(act_step*t),
                0.0)
            for t in range(num_pts + 1)
        ]
    )
    return pts


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


def get_cone_d(R, r):
    '''Perpendicular height of a cone with slant height `R` and base radius `r`.'''
    d = sqrt(R**2 - r**2)
    return d


def get_cone_delta(alpha, theta=None, beta=None):
    '''Consider a base cone whose axis lies in the XY plane, and whose
    ruled surface contains the Y-axis.
    Rotate this cone around the X-axis by an angle `beta` such that
    the elevation of the cone's axis is at angle `theta`.
    This result is important because it preserves the cone's tangency point
    with the YZ plane correctly after the rotation.  In the cone's base plane,
    a line segment from the cone axis to this point of tangency lies in the XY
    plane when the cone is unrotated (the base cone). After rotation of the
    cone by `beta`, this same line segment is no longer parallel to the XY
    plane.  It can be viewed as having effectively been rotated around the
    cone's axis.  That angle is what we call `delta` here.
    Inputs:
        `alpha` is the cone's half-angle, and is always known.
        `theta` is the elevation of cone axis.
        `beta` is the rotation of the cone around the x-axis from base.
    There are two possible cases:
        `theta` is known (as in top corners of square loops)
        -> return (delta, beta)
            OR
        `beta` is known (as in clover loops).
        -> return (delta, theta)
    '''
    delta = np.nan
    aux = np.nan
    known_str = 'UNDEFINED'
    aux_str = 'UNDEFINED'
    known_theta = theta is not None and beta is None
    known_beta = theta is None and beta is not None
    if known_theta:
        known_str = 'theta'
        aux_str = 'beta'
        # Rotation angle from equator around x-axis such that elevation of cone's axis = `theta`
        beta = aux = asin(sin(theta)/cos(alpha))
    elif known_beta:
        # Elevation angle of cone's axis when rotated by `beta` around x-axis
        theta = aux = asin(sin(beta)*cos(alpha))
        known_str = 'beta'
        aux_str = 'theta'
    else:
        raise ArgumentError('Invalid combination of input arguments')
    # Azimuth of cone axis when cone is at final elevation
    phi = atan2(tan(alpha), cos(beta))
    # Components of unit vector of the axis in Cartesian coords
    ux, uy, uz = spherical_to_cartesian((1.0, theta, phi))
    uxx_uyy_sqrt = cos(theta)  # nice simplification of ux^2+uy^2 due to spherical coords
    uxx_uyy = uxx_uyy_sqrt**2
    # Result
    delta = acos((sin(beta)*ux*uz + cos(beta)*uxx_uyy) / uxx_uyy_sqrt)
    # print(f'delta={degrees(delta)} deg, {aux_str}={degrees(aux)} deg  [known angle: {known_str}]')
    return delta, aux


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
    '''Angle between two vectors in radians.'''
    inner_prod = np.inner(a, b)
    norms_prod = LA.norm(a) * LA.norm(b)
    if abs(norms_prod) < 1e-16:
        return np.NaN
    cos_theta = inner_prod / norms_prod
    theta = acos(np.clip(cos_theta, -1., 1.))
    return theta


def spherical_to_cartesian(p):
    '''Convert a given point p from elevation-based spherical coordinates
    to Cartesian coordinates.
    Input must be an array or sequence like (`r`, `theta`, `phi`).
    Returns an array like (`x`, `y`, `z`).
    All angles are in radians.
    '''
    r, theta, phi = p[0], p[1], p[2]
    x = r * cos(theta) * cos(phi)
    y = r * cos(theta) * sin(phi)
    z = r * sin(theta)
    return np.array([x, y, z])


def cartesian_to_spherical(p):
    '''Convert a given XYZ point `p` from Cartesian coordinates to
    elevation-based spherical coordinates.
    Returns an array like (`r`, `theta`, `phi`) where
        `r` = radius,
        `theta` = elevation angle,
        `phi` = azimuth angle.
    All angles are in radians.
    '''
    x, y, z = p[0], p[1], p[2]
    r = sqrt(x*x + y*y + z*z)
    theta = atan2(z, sqrt(x*x + y*y))
    phi = atan2(y, x)
    return np.array([r, theta, phi])
