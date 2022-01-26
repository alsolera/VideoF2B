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

'''Unit tests for video_f2b.core.drawing'''

import random
from math import degrees
from pathlib import Path

import numpy as np
import pytest
import videof2b.core.geometry as geom
from videof2b.core.camera import CalCamera
from videof2b.core.detection import Detector
from videof2b.core.drawing import Drawing
from videof2b.core.flight import Flight


def tol(expected, tol=1e-13):
    return pytest.approx(expected, abs=tol)


@pytest.fixture
def caloc_artist():
    '''A calibrated and located artist instance.'''
    detector = Detector(100, 1.0)
    img_size = (1920, 1080)
    # TODO: is it possible to mock a video stream that just contains a single black frame?
    # test_frame = np.zeros((img_size[1], img_size[0], 3))

    # TODO: utilize Flight.read() method from the "serialize flight" branch later.
    # flight_path = Path('tests/data/01_markers.flight')

    vid_path = Path('tests/data/01_markers.mp4')
    cal_path = Path('tests/data/CamCalibration_SONY_A7C_14mm.npz')
    loc_pts = [
        (910, 934),
        (1088, 909),
        (745, 907),
        (1484, 904),
    ]
    test_flight = Flight(
        vid_path=vid_path,
        is_live=False,
        cal_path=cal_path,
        # randomize R within the typical range: just short of F2D up to just longer than F2B.
        flight_radius=random.uniform(15.85, 21.95),
        marker_radius=28.04,
        marker_height=1.3462,
        sphere_offset=[0.0, 0.0, 0.0, ],
        loc_pts=loc_pts
    )
    test_cam = CalCamera(img_size, test_flight)
    test_cam.locate(test_flight)
    artist = Drawing(detector, cam=test_cam)
    artist.locate(test_cam, test_flight)
    yield artist
    del artist


def test_figure_connections(caloc_artist):
    '''Test for gaps between components of figures.'''
    # Verify connections in all figures
    # Plain loop
    loop_pts = caloc_artist._get_loop_pts()
    _verify_figure_gaps(loop_pts, is_closed=True)
    # Square loop
    sq_loop_pts = caloc_artist._get_square_loop_pts()
    _verify_figure_gaps(sq_loop_pts, is_closed=True)
    # Triangular loop
    tri_loop_pts = caloc_artist._get_tri_loop_pts()
    _verify_figure_gaps(tri_loop_pts, is_closed=True)
    # Horizontal eight
    hor_eight_pts = caloc_artist._get_hor_eight_pts()
    _verify_figure_gaps(hor_eight_pts, is_closed=True)
    # Square eight
    sq_eight_pts = caloc_artist._get_sq_eight_pts()
    _verify_figure_gaps(sq_eight_pts, is_closed=True)
    # Vertical eight
    ver_eight_pts = caloc_artist._get_ver_eight_pts()
    _verify_figure_gaps(ver_eight_pts, is_closed=True)
    # Hourglass
    hourglass_pts = caloc_artist._get_hourglass_pts()
    _verify_figure_gaps(hourglass_pts, is_closed=True)
    ovr_eight_pts = caloc_artist._get_ovr_eight_pts()
    _verify_figure_gaps(ovr_eight_pts, is_closed=True)
    # Four-leaf clover
    clover_pts = caloc_artist._get_clover_pts()
    _verify_figure_gaps(clover_pts)


def test_clover_tangencies(caloc_artist):
    '''Verify the elevation and azimuth angles of the four main
    tangency points of the four-leaf clover.'''
    clover_pts = caloc_artist._get_clover_pts()
    # Point 1: at bottom of vertical, tangent to vertical and to bottom loops.
    pt1 = clover_pts[0][0]
    _, pt1_theta, pt1_phi = geom.cartesian_to_spherical(pt1)
    pt1_theta = degrees(pt1_theta)
    pt1_phi = degrees(pt1_phi)
    # Point 2: at top of vertical, tangent to vertical and to top loops.
    pt2 = clover_pts[0][-1]
    _, pt2_theta, pt2_phi = geom.cartesian_to_spherical(pt2)
    pt2_theta = degrees(pt2_theta)
    pt2_phi = degrees(pt2_phi)
    # Point 3: tangency between right loops.
    pt3 = clover_pts[1][-1]
    _, pt3_theta, pt3_phi = geom.cartesian_to_spherical(pt3)
    pt3_theta = degrees(pt3_theta)
    pt3_phi = degrees(pt3_phi)
    # Point 4: tangency between left loops.
    pt4 = clover_pts[2][-1]
    _, pt4_theta, pt4_phi = geom.cartesian_to_spherical(pt4)
    pt4_theta = degrees(pt4_theta)
    pt4_phi = degrees(pt4_phi)

    print(f'pt1 tangency: elev={pt1_theta} deg, phi={pt1_phi} deg')
    print(f'pt2 tangency: elev={pt2_theta} deg, phi={pt2_phi} deg')
    print(f'pt3 tangency: elev={pt3_theta} deg, phi={pt3_phi} deg')
    print(f'pt4 tangency: elev={pt4_theta} deg, phi={pt4_phi} deg')

    assert pt1_theta == tol(22.5)
    assert pt1_phi == tol(90.0)
    assert pt2_theta == tol(67.5)
    assert pt2_phi == tol(90.0)
    assert pt3_theta == tol(40.789470940925284)
    assert pt3_phi == tol(90.0 - 30.36119340482174)
    assert pt4_theta == tol(40.789470940925284)
    assert pt4_phi == tol(90.0 + 30.36119340482174)


def _verify_figure_gaps(pieces, is_closed=False):
    '''Helper function.
    Verify that the given pieces of a drawn figure have acceptably small gaps
    from one connection to the next.  If a figure is closed, verify that the
    start point of the first piece is acceptably close to the last point of
    the last piece.

    :param pieces: array of XYZ point arrays that represent figure components in 3D.
    :param is_closed: True if start and end of the figure are supposed to be the same point.
    '''
    for i, piece in enumerate(pieces):
        if i == 0:
            continue
        prev_piece = pieces[i-1]
        gap_vector = piece[0] - prev_piece[-1]
        gap_dist = np.linalg.norm(gap_vector)
        assert 0.0 == pytest.approx(gap_dist, abs=1e-13), \
            f'Unexpected gap between components id {i-1} and {i}: {gap_dist}.'
    if is_closed:
        endp_gap_vector = pieces[0][0] - pieces[-1][-1]
        endp_dist = np.linalg.norm(endp_gap_vector)
        assert 0.0 == pytest.approx(endp_dist, abs=1e-13), \
            f'Unexpected gap between start and end points of a closed figure: {endp_dist}.'
