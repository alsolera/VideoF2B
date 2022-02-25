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
from typing import Optional, Tuple

import numpy as np
import numpy.linalg as LA
import numpy.typing as npt
from scipy.optimize import fsolve
from videof2b.core.common import QUART_PI, TWO_PI


class ArgumentError(Exception):
    '''Thrown when the provided combination of arguments is invalid.'''


class Fillet:
    r'''
    Represents a spherical fillet.

    On a sphere of radius ``R``, a fillet is defined as an arc of a small circle
    of radius ``r`` between two great arcs of the sphere with angle ``psi``
    (:math:`\psi`) between the arcs such that the small circle is tangent to
    both arcs. The constructor of this class tries to calculate the parameters
    of the fillet via the :meth:`calculate` method.

    :param R: radius of the sphere.
    :param r: radius of the fillet.
    :param psi: angle between two great arcs that define the fillet.
    :param is_degrees: If True, ``psi`` is given in degrees, otherwise it is
        given in radians.
    '''

    def __init__(self, R: float, r: float, psi: float, is_degrees: Optional[bool] = False):
        # Values that define the fillet
        self.R = R
        self.r = r
        self.psi = psi
        self.is_degrees = is_degrees
        if self.is_degrees:
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

    def calculate(self) -> None:
        r'''
        Calculate the fillet as follows:

        Given two intersecting planes with angle :math:`\psi` between them, a
        cone of slant height :math:`R` and base radius :math:`r` whose apex
        rests on the intersection line of the planes will rest tangent to both
        planes when the cone's axis makes an angle :math:`\theta` with the
        intersection line of the planes in the bisecting plane.

        If we set up a coordinate system on the cone's base such that:

        - the origin is at the cone's apex;
        - the :math:`-z` axis is along the cone's axis toward the cone's base;
          and
        - the :math:`+x` axis is toward the intersection line,

        then the coordinates of the points of tangency between the cone and the
        planes are :math:`(x_p, y_p, -d)` and :math:`(x_p, -y_p, -d)`.

        Let :math:`\beta` be the central angle of the arc along the cone's base
        that joins the two tangency points on the side of the intersection (the
        shorter of the two possible arcs).

        Perform the following:

        - Ensure that tangency is possible. This requires that

            .. math::
                2 \alpha \leqslant \psi \leqslant \pi

          where :math:`\alpha = \arcsin\left(\dfrac{r}{R}\right)` is the
          half-angle of the cone's apex.

          If this condition fails, the instance attribute ``is_valid`` is set to
          False and this method returns early.

        - Find angle :math:`\theta` (Gorjanc solution). Store it in the instance
          attribute ``theta``.
        - Find :math:`x_p`, :math:`y_p`, and :math:`d`. Store them in instance
          attributes ``x_p``, ``y_p``, and ``d``, respectively.
        - Find angle :math:`\beta`. Store it in the instance attribute ``beta``.
        - Set ``is_valid`` to True and return.

        .. seealso:: method :meth:`get_fillet_theta`.
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
    def get_fillet_theta(R: float, r: float, psi: float) -> float:
        r'''
        Calculate the angle :math:`\theta` between cone axis and intersection
        line of the planes that are tangent to a :class:`Fillet`.

        Implements the |stosci|_ solution. Values of :math:`x_p` and :math:`y_p`
        follow from this as well.

        :param R: radius of the sphere on which the fillet is defined.
        :param r: radius of the fillet, which is also equal to the base radius
            of the fillet's cone.
        :param psi: angle between the planes that are tangent to the fillet's
            cone.

        .. note:: This method is provided as a static method for optimizer use
            in addition to use by :class:`Fillet` here.

        .. seealso:: The :meth:`calculate` method.

        .. |stosci| replace:: Gorjanc
        .. _stosci: https://www.grad.hr/geomteh3d/Plohe/plohe5_eng.html
        '''
        # TODO: update Gorjanc (_stosci) link to the actual solution when (if?) it becomes available.
        theta = atan(
            r * sqrt(
                2.0 / (R**2 * (1 + cos(pi - psi)) - 2.0 * r**2)
            )
        )
        return theta


def get_arc(r: float, alpha: float, rho: Optional[int] = 100) -> np.ndarray:
    # TODO: change meaning of `rho` from angular density to circumferential (linear) density
    # for more consistent point spacing on arcs of different radius.
    r'''
    Create an array of 3D points that represent a circular arc.

    Return 3D points for an arc of radius ``r`` and included angle ``alpha``
    with point density ``rho``, where ``rho`` is the number of points per
    :math:`2\pi`. Arc center is ``(0, 0, 0)``.  The arc lies in the :math:`xy`
    plane. The arc starts at zero angle, i.e., at ``(r, 0, 0)``, and proceeds
    counterclockwise until it ends at ``alpha``. Angle measurements are in
    radians. The endpoint is always included.

    :param r: radius of the arc.
    :param alpha: included angle of the arc in radians.
    :param rho: angular density of generated points. Defaults to 100.

    :return: ``(N, 3)`` array of points, where ``N >= 3`` and is proportional to
        ``alpha`` and ``rho``.

    .. warning:: The meaning of the ``rho`` parameter may change in the future
        from angular density to circumferential (linear) density to provide more
        consistent point spacing on arcs of different radii in the same scene.
    '''
    nom_step = TWO_PI / rho
    num_pts = int(alpha / nom_step)
    # Prevent degenerate arcs
    num_pts = max(num_pts, 3)
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


def get_equilateral_phi(sigma: float) -> float:
    r'''
    Calculate the angle between sides of an equilateral spherical triangle as a
    function of side angle `sigma`. See the derivation |equilaterals|_.

    :param sigma: the side angle :math:`\sigma` of the spherical triangle.

    :return: the angle :math:`\phi` between the sides of the equilateral
        spherical triangle.

    .. |equilaterals| replace:: here
    .. _equilaterals: https://www.av8n.com/physics/spherical-triangle.htm
    '''
    phi = acos(cos(sigma) / (cos(sigma)+1))
    return phi


def get_equilateral_height(sigma: float) -> float:
    r'''
    Calculate the height of an equilateral spherical triangle as a function of
    its side angle :math:`\sigma`. Takes advantage of the |cosine-rule|_ in
    spherical trigonometry.

    This is the companion to the :func:`calc_equilateral_sigma` function.

    :param sigma: the side angle :math:`\sigma` of the spherical triangle.
    :return: the height of the equilateral spherical triangle in radians.

    .. |cosine-rule| replace:: cosine rule
    .. _cosine-rule: https://math.stackexchange.com/a/3139032/10521
    '''
    h = acos(cos(sigma) / cos(0.5*sigma))
    return h


def get_cone_alpha(R: float, r: float) -> float:
    r'''
    Calculates the half-aperture of a cone with slant height ``R`` and base
    radius ``r``.

    :param R: slant height of cone.
    :param r: base radius of cone.
    :return: angle :math:`\alpha` in radians.
    '''
    alpha = asin(r / R)
    return alpha


def get_cone_d(R: float, r: float) -> float:
    '''
    Perpendicular height of a cone with slant height ``R`` and base radius ``r``.

    :param R: slant height of cone.
    :param r: base radius of cone.
    :raises: ValueError if ``R < r``
    :return: height ``d`` of the cone.
    '''
    d = sqrt(R**2 - r**2)
    return d


def get_cone_delta(alpha: float, theta: Optional[float] = None, beta: Optional[float] = None) -> Tuple[float]:
    r'''
    Calculate the properties of a cone rotated from the flight base to a certain
    elevation.

    Consider a base cone whose axis lies in the :math:`xy` plane, and whose
    ruled surface contains the :math:`y` axis. Rotate this cone around the
    :math:`x` axis by an angle :math:`\beta` such that the elevation of the
    cone's axis is at angle :math:`\theta`. This result is important because it
    preserves the cone's tangency point with the :math:`yz` plane after the
    rotation. In the cone's base plane, a line segment from the cone axis to
    this point of tangency lies in the :math:`xy` plane when the cone is
    unrotated (the "base" cone). After rotation of the cone by :math:`\beta`,
    this same line segment is no longer parallel to the :math:`xy` plane.
    Effectively, it has been rotated around the cone's axis by an angle that we
    hereby call :math:`\delta`. The goal is to calculate :math:`\delta` and one
    of the other angles such that the caller has all three angles :math:`\delta`,
    :math:`\theta`, and :math:`\beta` at its disposal.

    :param alpha: the cone's half-aperture, in radians. This is required.
    :param theta: the elevation of the cone's axis, in radians.
    :param beta: the rotation of the cone around the :math:`x` axis from the
        base, in radians.

    :raises ArgumentError: when ``theta`` and ``beta`` arguments are supplied
        inconsistently, i.e., when both are given or neither is given.
    :return: angle :math:`\delta` and one of the missing angles :math:`\theta`
        or :math:`\beta`. Two mutually exclusive cases are possible:

        - ``theta`` is known (as in top corners of square loops)
            --> return ``(delta, beta)``
        - ``beta`` is known (as in clover loops).
            --> return ``(delta, theta)``
    '''
    delta = np.nan
    aux = np.nan
    known_str = 'UNDEFINED'  # pylint: disable=unused-variable
    aux_str = 'UNDEFINED'  # pylint: disable=unused-variable
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
    ux, _, uz = spherical_to_cartesian((1.0, theta, phi))
    uxx_uyy_sqrt = cos(theta)  # nice simplification of ux^2+uy^2 due to spherical coords
    uxx_uyy = uxx_uyy_sqrt**2
    # Result
    delta = acos((sin(beta)*ux*uz + cos(beta)*uxx_uyy) / uxx_uyy_sqrt)
    # print(f'delta={degrees(delta)} deg, {aux_str}={degrees(aux)} deg  [known angle: {known_str}]')
    return delta, aux


def calc_equilateral_sigma(height: Optional[float] = QUART_PI) -> float:
    r'''
    Solve for the side of an equilateral spherical triangle whose ``height`` is
    given.

    This is the companion to the :func:`get_equilateral_height` function.

    :param height: height of the equilateral spherical triangle, in radians.
        Defaults to :math:`\dfrac{\pi}{4}`.
    :return: side angle :math:`\sigma`, in radians.
    '''
    def root_finder(sigma):
        return get_equilateral_height(sigma) - height
    ret = fsolve(root_finder, QUART_PI)
    sigma = ret[0]
    return sigma


def calc_tri_loop_params(R: float, r: float, target_elev: Optional[float] = QUART_PI) -> Tuple[float]:
    r'''
    Calculate the basic parameters of a triangular loop figure on a sphere.

    Given an equilateral spherical triangle on the surface of a sphere of radius
    ``R`` such that the top of a corner turn of radius ``r`` is located at
    ``target_elev`` on the sphere, calculate:

        - The central angle :math:`\sigma` of the side of the triangle,
        - The angle :math:`\phi` between adjacent sides of the triangle.

    :param R: radius of the sphere.
    :param r: radius of the loop's corner turns.
    :param target_elev: elevation of the highest point in the top turn. Defaults
        to :math:`\dfrac{\pi}{4}`.
    :return: sigma, phi

    All angles are in radians.
    '''

    phi = 0.0

    def root_finder(sigma: float) -> float:
        r'''
        Solve the triangular loop including corners.

        :param sigma: the side of the triangular loop without the corners.
        :return: the objective function

        Called by an optimizer in the enclosing function.
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


def angle(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    '''
    Calculate the angle between two vectors in radians.

    :param a: any vector in 2D or 3D.
    :param b: any vector in 2D or 3D.
    :return: the angle between ``a`` and ``b`` in radians.
    '''
    inner_prod = np.inner(a, b)
    norms_prod = LA.norm(a) * LA.norm(b)
    if abs(norms_prod) < 1e-16:
        return np.NaN
    cos_theta = inner_prod / norms_prod
    theta = acos(np.clip(cos_theta, -1., 1.))
    return theta


def spherical_to_cartesian(p: npt.ArrayLike) -> np.ndarray:
    r'''
    Convert a point from elevation-based spherical coordinates to Cartesian
    coordinates.

    :param p: an array or sequence like ``(r, theta, phi)`` where

        - ``r`` = radius,
        - ``theta`` = elevation angle :math:`\theta`,
        - ``phi`` = azimuth angle :math:`\phi`.

    :return: an array containing ``(x, y, z)``.
    :seealso: the inverse conversion function is :func:`cartesian_to_spherical`.

    All angles are in radians.
    '''
    r, theta, phi = p[0], p[1], p[2]
    x = r * cos(theta) * cos(phi)
    y = r * cos(theta) * sin(phi)
    z = r * sin(theta)
    return np.array([x, y, z])


def cartesian_to_spherical(p: npt.ArrayLike) -> np.ndarray:
    r'''
    Convert a point from Cartesian coordinates to elevation-based spherical
    coordinates.

    :param p: an array or sequence representing a point ``(x, y, z)`` in
        Cartesian space.
    :return: an array containing ``(r, theta, phi)`` where

        - ``r`` = radius,
        - ``theta`` = elevation angle :math:`\theta`,
        - ``phi`` = azimuth angle :math:`\phi`.

    :seealso: the inverse conversion function is :func:`spherical_to_cartesian`.

    All angles are in radians.
    '''
    x, y, z = p[0], p[1], p[2]
    r = sqrt(x*x + y*y + z*z)
    theta = atan2(z, sqrt(x*x + y*y))
    phi = atan2(y, x)
    return np.array([r, theta, phi])
