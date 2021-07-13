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

from math import degrees, pi, radians

import geometry as geom

QUART_PI = 0.25 * pi


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
