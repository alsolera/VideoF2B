# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021-2022  Andrey Vasilik - basil96
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
Tests for CamDirector.
'''

import pytest
from videof2b.core.camera_director import CamDirector


@pytest.fixture
def cd():
    cam_director = CamDirector()
    yield cam_director
    del cam_director


def print_cd_state(cd_instance: CamDirector):
    '''Output the state of the given CamDirector.'''
    print('CamDirector state: =====================')
    print(f'                      R = {cd_instance.R}')
    print(f'                      C = {cd_instance.C}')
    print(f'                      G = {cd_instance.G}')
    print(f'                      A = {cd_instance.A}')
    print(f'    cam_distance_limits = {cd_instance.cam_distance_limits}')
    print(f'        cam_view_limits = {cd_instance.cam_view_limits}')
    print(f'cam_tangent_elev_limits = {cd_instance.cam_tangent_elev_limits}')


def tol(expected, tol=1e-13):
    return pytest.approx(expected, abs=tol)


def test_default(cd: CamDirector):
    '''Test the state properties of a default instance of CamDirector.'''
    print_cd_state(cd)
    assert cd.R == 21.0
    assert cd.C == -1.0
    assert cd.G == -1.5
    assert cd.A == 71.75
    assert cd.cam_distance_limits == tol((22.63437695528075, 30.698484809834998))
    assert cd.cam_view_limits == tol((71.75, 45.9331195689391))
    assert cd.cam_tangent_elev_limits == tol((19.51547469372101, 45.0))


def test_adjust_r(cd: CamDirector):
    '''Test the outputs when we adjust sphere radius R.'''
    cd.R = 19.51
    print_cd_state(cd)
    assert cd.cam_distance_limits == tol((21.067492322478817, 28.59130660189909))
    assert cd.cam_view_limits == tol((71.75, 46.00187690909413))
    assert cd.cam_tangent_elev_limits == tol((19.609559663362617, 45.0))


def test_adjust_c(cd: CamDirector):
    '''Test the outputs when we adjust camera height C.'''
    cd.C = -0.5
    print_cd_state(cd)
    assert cd.cam_distance_limits == tol((22.650506217157492, 30.198484809834998))
    assert cd.cam_view_limits == tol((71.75, 46.896613390338736))
    assert cd.cam_tangent_elev_limits == tol((20.777916882261618, 45.0))


def test_adjust_g(cd: CamDirector):
    '''Test the outputs when we adjust ground level G.'''
    cd.G = -1.0
    print_cd_state(cd)
    assert cd.cam_distance_limits == (22.442016864562575, 30.698484809834998)
    assert cd.cam_view_limits == (71.75000000000635, 45.0)
    assert cd.cam_tangent_elev_limits == (18.24999999999365, 45.0)


def test_adjust_a(cd: CamDirector):
    '''Test the outputs when we adjust max view angle A.'''
    cd.A = 60.0
    print_cd_state(cd)
    assert cd.cam_distance_limits == tol((25.13942141576743, 30.698484809834998))
    assert cd.cam_view_limits == tol((60.0, 45.9331195689391))
    assert cd.cam_tangent_elev_limits == tol((31.139410200050825, 45.0))


def test_narrow_fov(cd: CamDirector):
    '''Test the outputs when the camera FOV is too narrow.'''
    cd.A = 40.0
    print_cd_state(cd)
    assert cd.cam_distance_limits == tol((30.698484809834998, 34.47576935937541))
    assert cd.cam_view_limits == tol((45.9331195689391, 40.0))
    assert cd.cam_tangent_elev_limits == tol((45.0, 50.83089897596292))


def test_high_camera(cd: CamDirector):
    '''Test the outputs when the camera is placed higher than usual.'''
    cd.C = 5.0
    print_cd_state(cd)
    assert cd.cam_distance_limits == tol((22.07332389352903, 24.698484809834998))
    assert cd.cam_view_limits == tol((71.75, 59.74442791202347))
    assert cd.cam_tangent_elev_limits == tol((34.658282717315934, 45.0))
