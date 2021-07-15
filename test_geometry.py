#!/usr/bin/env python
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

'''Unit tests for video_f2b.geometry'''

from math import acos, atan, atan2, cos, degrees, pi, radians, sin, sqrt

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ROT

import geometry as geom
from Drawing import Drawing

HALF_PI = 0.5 * pi
QUART_PI = 0.25 * pi
EIGHTH_PI = 0.125 * pi
TWO_PI = 2.0 * pi


class TestGeometry:
    '''Unit tests of the geometry module.'''

    @staticmethod
    def print_fillet_data(sigma, psi, f):
        '''Helper method for fillet test cases.'''
        print(f' sigma = {sigma} [{degrees(sigma)} deg]')
        print(f'   psi = {psi} [{degrees(psi)} deg]')
        print(f'  beta = {f.beta} [{degrees(f.beta)} deg]')
        print(f' theta = {f.theta} [{degrees(f.theta)} deg]')
        print(f'     d = {f.d}')
        print(f'   x_p = {f.x_p}')
        print(f'   y_p = {f.y_p}')

    def test_60_tri_loop_fillet(self):
        '''Verify parameters of an empirical triangular loop where corner radii
        are slightly larger than F2B spec (2.1 m on a 21.0 m sphere).
        Inputs:
            * Unit sphere, R = 1.0
            * r = 0.1
            * Target elevation of fillet top = such that sides of the construction triangle are 60 deg
        Expected values created and verified in CAD.'''
        R = 1.
        r = 0.1
        target = radians(50.50055900011077981)
        sigma, psi = geom.calc_tri_loop_params(R, r, target_elev=target)
        f = geom.Fillet(R, r, psi)
        TestGeometry.print_fillet_data(sigma, psi, f)
        # assert False  # uncomment this to dump print() outputs for debugging
        '''
        ===== calc_tri_loop_params: root_finder ====================
             R = 1.0
             r = 0.1
          elev = 0.8814010286495886 [50.50055900011078 deg]
         sigma = [1.04719755] [60.00000000000001 deg]
           phi = 1.2309594173407747 [70.52877936550931 deg]
         theta = 0.1740830106364804 [9.974221794401346 deg]
         alpha = 0.1001674211615598 [5.739170477266787 deg]
             h = 0.9553166181245094 [54.73561031724535 deg]
        result = -2.220446049250313e-16
        ========================================
         sigma = 1.0471975511965979 [60.00000000000001 deg]
           psi = 1.2309594173407747 [70.52877936550931 deg]
          beta = 1.9249549697264348 [110.29179551805787 deg]
         theta = 0.1740830106364804 [9.974221794401346 deg]
             d = 0.99498743710662
           x_p = 0.057148869332588434
           y_p = 0.08206099398622183
        '''
        assert 1.0471975511965979 == sigma
        assert 1.2309594173407747 == psi
        assert 1.0 == f.R
        assert 0.1 == f.r
        assert 0.1001674211615598 == f.alpha
        assert 1.9249549697264348 == f.beta
        assert 0.1740830106364804 == f.theta
        assert 0.99498743710662 == f.d
        assert f.psi == psi
        assert 0.057148869332588434 == f.x_p
        assert 0.082060993986221830 == f.y_p

    def test_45_tri_loop_fillet(self):
        '''Verify parameters of an empirical triangular loop where corner radii
        are slightly larger than F2B spec (2.1 m on a 21.0 m sphere).
        Inputs:
            * Unit sphere, R = 1.0
            * r = 0.1
            * Target elevation of fillet top = 45 deg
        Expected values created and verified in CAD.'''
        R = 1.
        r = 0.1
        target = QUART_PI
        sigma, psi = geom.calc_tri_loop_params(R, r, target_elev=target)
        f = geom.Fillet(R, r, psi)
        TestGeometry.print_fillet_data(sigma, psi, f)
        # assert False  # uncomment this to dump print() outputs for debugging
        '''
        ===== calc_tri_loop_params: root_finder ====================
             R = 1.0
             r = 0.1
          elev = 0.7853981633974483 [45.0 deg]
         sigma = [0.95602413] [54.77614796131095 deg]
           phi = 1.1963115345797664 [68.54360191423945 deg]
         theta = 0.17852908633383857 [10.228963167255653 deg]
         alpha = 0.1001674211615598 [5.739170477266787 deg]
             h = 0.863759828569727 [49.48979268998886 deg]
        result = 0.0
        ========================================
         sigma = 0.9560241334844555 [54.77614796131095 deg]
           psi = 1.1963115345797664 [68.54360191423945 deg]
          beta = 1.9601482955632028 [112.30822455553339 deg]
         theta = 0.17852908633383857 [10.228963167255653 deg]
             d = 0.99498743710662
           x_p = 0.05569609656974007
           y_p = 0.08305386701950844
        '''
        assert 0.9560241334844555 == sigma
        assert 1.1963115345797664 == psi
        assert 1.0 == f.R
        assert 0.1 == f.r
        assert 0.1001674211615598 == f.alpha
        assert 1.9601482955632028 == f.beta
        assert 0.17852908633383857 == f.theta
        assert 0.99498743710662 == f.d
        assert f.psi == psi
        assert 0.05569609656974007 == f.x_p
        assert 0.08305386701950844 == f.y_p

    def test_f2b_tri_loop_fillet(self):
        '''Verify parameters of an empirical triangular loop where corner radii
        are per F2B spec (1.5 m on a 21.0 m sphere).
        Inputs:
            * Sphere, R = 21.0
            * r = 1.5
            * Target elevation of fillet top = 45 deg
        Expected values created and verified in CAD.'''
        R = 21.
        r = 1.5
        target = QUART_PI
        sigma, psi = geom.calc_tri_loop_params(R, r, target_elev=target)
        f = geom.Fillet(R, r, psi)
        TestGeometry.print_fillet_data(sigma, psi, f)
        # assert False  # uncomment this to dump print() outputs for debugging
        '''
        ===== calc_tri_loop_params: root_finder ====================
             R = 21.0
             r = 1.5
          elev = 0.7853981633974483 [45.0 deg]
         sigma = [0.93376229] [53.500638402250615 deg]
           phi = 1.1885910110968365 [68.10124850303593 deg]
         theta = 0.12791661970232607 [7.329082438523277 deg]
         alpha = 0.07148944988552053 [4.096043758152333 deg]
             h = 0.8418253332142539 [48.233038680370946 deg]
        result = -1.1102230246251565e-16
        ========================================
         sigma = 0.9337622920381917 [53.500638402250615 deg]
           psi = 1.1885910110968365 [68.10124850303593 deg]
          beta = 1.9606017302158762 [112.33420444741655 deg]
         theta = 0.12791661970232607 [7.329082438523277 deg]
             d = 20.946360065653412
           x_p = 0.835158980817275
           y_p = 1.2459973823247987
        '''
        assert 0.9337622920381917 == sigma
        assert 1.1885910110968365 == psi
        assert 21. == f.R
        assert 1.5 == f.r
        assert 0.07148944988552053454 == f.alpha
        assert 1.9606017302158762 == f.beta
        assert 0.12791661970232607 == f.theta
        assert 20.946360065653412 == f.d
        assert f.psi == psi
        assert 0.835158980817275 == f.x_p
        assert 1.2459973823247987 == f.y_p

    def test_real_triangular_loop(self):
        '''Create a real-world triangular loop and verify its basic parameters,
        such as location and orientation of all its pieces in world space.'''
        R = 21.0
        r = 1.5
        sigma, phi = geom.calc_tri_loop_params(R, r)
        h = geom.get_equilateral_height(sigma)
        f = geom.Fillet(R, r, phi)
        # Raw template points
        pts = Drawing.get_arc(r, f.beta, rho=27)
        pts_ctr = np.zeros((3,))
        # Starting template arc in the middle of the bottom leg
        pts = ROT.from_euler('zxy', [0.5*(pi - f.beta), -HALF_PI, HALF_PI]
                             ).apply(pts) + [0, f.d, 0]
        # 1st corner turn
        corner1 = ROT.from_rotvec(-0.5*phi*np.array([-sin(0.5*sigma), cos(0.5*sigma), 0.])).apply(
            ROT.from_euler('z', 0.5*sigma-f.theta).apply(pts)
        )
        corner1_norms = np.linalg.norm(corner1, axis=1)
        # 2nd corner turn
        corner2 = ROT.from_euler('x', h - f.theta).apply(
            ROT.from_euler('y', HALF_PI).apply(pts)
        )
        corner2_norms = np.linalg.norm(corner2, axis=1)
        # 3rd corner turn
        corner3 = ROT.from_rotvec(0.5*phi*np.array([sin(0.5*sigma), cos(0.5*sigma), 0.])).apply(
            ROT.from_euler('yz', [pi, -(0.5*sigma-f.theta)]).apply(pts)
        )
        corner3_norms = np.linalg.norm(corner3, axis=1)
        # Leg angles between the tangencies
        leg_sigmas = (
            geom.angle(corner1[-1], corner2[0]),
            geom.angle(corner2[-1], corner3[0]),
            geom.angle(corner3[-1], corner1[0]),
        )
        # --- Expected data
        # Expected first and last points of the each corner arc wrt world
        exp_c1_pts = np.array([
            [-7.4118195040371475, 19.648535101619522, 0.0],
            [-8.6578168863619480, 19.020492013929108, 2.064725238448216]
        ])
        exp_c2_pts = np.array([
            [-1.2459973823247994, 15.28457565266493, 14.34675007244506],
            [+1.2459973823247994, 15.28457565266493, 14.34675007244506]
        ])
        exp_c3_pts = np.array([
            [8.657816886361946, 19.020492013929108, 2.0647252384482173],
            [7.4118195040371484, 19.648535101619522, 0.]
        ])
        # Expected angles of all leg arcs between tangencies
        exp_leg_sigma = 0.721431018560276
        # Output stuff if we fail
        np.set_printoptions(precision=16)
        print(f'  beta = {f.beta} [{degrees(f.beta)} deg]')
        print(f'Corner 1: tangency point 1 = {corner1[0]}')
        print(f'          tangency point 2 = {corner1[-1]}')
        print(f'Corner 2: tangency point 1 = {corner2[0]}')
        print(f'          tangency point 2 = {corner2[-1]}')
        print(f'Corner 3: tangency point 1 = {corner3[0]}')
        print(f'          tangency point 2 = {corner3[-1]}')
        print(f'exp_leg_sigma =  {exp_leg_sigma}')
        print(f'   leg_sigmas = {leg_sigmas}')
        # --- Verifications
        # Verify the number of points in the template arc
        assert (10, 3) == pts.shape
        # Verify that all corner arcs indeed lie on the sphere
        assert np.allclose(R, corner1_norms)
        assert np.allclose(R, corner2_norms)
        assert np.allclose(R, corner3_norms)
        # Verify coords of first & last points of corner arc wrt World
        assert np.allclose(exp_c1_pts[0], corner1[0])
        assert np.allclose(exp_c1_pts[-1], corner1[-1])
        assert np.allclose(exp_c2_pts[0], corner2[0])
        assert np.allclose(exp_c2_pts[-1], corner2[-1])
        assert np.allclose(exp_c3_pts[0], corner3[0])
        assert np.allclose(exp_c3_pts[-1], corner3[-1])
        # Verify angles of leg segments
        assert np.allclose(exp_leg_sigma, leg_sigmas, rtol=1e-16, atol=1e-15)
        assert False  # uncomment this to dump print() outputs for debugging

    def test_equilateral_height_and_sigma(self):
        '''Test the relationship between sigma and height angles in equilateral spherical triangles.'''
        sigmas = (
            EIGHTH_PI,
            QUART_PI,
            HALF_PI,
            radians(48.23303868),
            radians(100.)
        )
        for sigma in sigmas:
            h = geom.get_equilateral_height(sigma)
            calc_sigma = geom.calc_equilateral_sigma(height=h)
            calc_h = geom.get_equilateral_height(calc_sigma)
            assert h == pytest.approx(calc_h, abs=1e-15)
            assert sigma == pytest.approx(calc_sigma, abs=1e-15)
