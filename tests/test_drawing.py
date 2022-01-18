# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021  Andrey Vasilik - basil96
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
from math import (acos, atan, atan2, cos, degrees, isclose, pi, radians, sin,
                  sqrt)
from pathlib import Path

import numpy as np
import pytest
import videof2b.core.geometry as geom
from videof2b.core.camera import CalCamera
from videof2b.core.detection import Detector
from videof2b.core.drawing import Drawing
from videof2b.core.flight import Flight


class TestDrawing:
    '''Unit tests of the drawing module.'''

    def test_figure_connections(self):
        '''Test for gaps between components of figures.'''
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
        # Verify connections in all figures
        # Plain loop
        loop_pts = artist._get_loop_pts()
        self._verify_figure_gaps(loop_pts, is_closed=True)
        # Square loop
        sq_loop_pts = artist._get_square_loop_pts()
        self._verify_figure_gaps(sq_loop_pts, is_closed=True)
        # Triangular loop
        tri_loop_pts = artist._get_tri_loop_pts()
        self._verify_figure_gaps(tri_loop_pts, is_closed=True)
        # Horizontal eight
        hor_eight_pts = artist._get_hor_eight_pts()
        self._verify_figure_gaps(hor_eight_pts, is_closed=True)
        # Square eight
        sq_eight_pts = artist._get_sq_eight_pts()
        self._verify_figure_gaps(sq_eight_pts, is_closed=True)
        # Vertical eight
        ver_eight_pts = artist._get_ver_eight_pts()
        self._verify_figure_gaps(ver_eight_pts, is_closed=True)
        # Hourglass
        hourglass_pts = artist._get_hourglass_pts()
        self._verify_figure_gaps(hourglass_pts, is_closed=True)
        ovr_eight_pts = artist._get_ovr_eight_pts()
        self._verify_figure_gaps(ovr_eight_pts, is_closed=True)
        # Four-leaf clover
        clover_pts = artist._get_clover_pts()
        self._verify_figure_gaps(clover_pts)

    def _verify_figure_gaps(self, pieces, is_closed=False):
        '''Helper method.
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
