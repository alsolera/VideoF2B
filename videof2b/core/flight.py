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

'''Defines a recorded flight.'''

import logging
from pathlib import Path

import videof2b.core.common as common
from imutils.video import FileVideoStream
from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class Flight(QObject):
    '''Contains information about a flight to be processed.'''

    # Signals
    locator_points_changed = Signal(object)

    point_names = (
        'circle center',
        'front marker',
        'left marker',
        'right marker'
    )

    def __init__(self, vid_path: Path, is_live=False, cal_path=None, **kwargs) -> None:
        '''Provide a flight video with a path to the video file, plus these options:
            `is_live`: True if using a live video stream, False if loading from a file.
            `cal_path`: path to the calibration file.
            `kwargs` are additional parameters when cal_path is specified.
            These are as follows:
            `flight_radius`
            `marker_radius`
            `marker_height`
            `sphere_offset`
            All dimensions are in meters.
        '''
        super().__init__()
        self.is_ready = False
        self.video_path = vid_path
        self.is_live = is_live
        self.calibration_path = cal_path
        self.is_calibrated = cal_path is not None and cal_path.exists()
        self.flight_radius = kwargs.pop('flight_radius', common.DEFAULT_FLIGHT_RADIUS)
        self.marker_radius = kwargs.pop('marker_radius', common.DEFAULT_MARKER_RADIUS)
        self.marker_height = kwargs.pop('marker_height', common.DEFAULT_MARKER_HEIGHT)
        self.sphere_offset = kwargs.pop('sphere_offset', common.DEFAULT_CENTER)
        # Load the video
        self.cap = FileVideoStream(str(self.video_path)).start()
        # Check if we succeeded.
        if self.cap.isOpened():
            self.is_ready = True
            # TODO: Using filename only for now. Maybe use a sanitized path?
            log.info(f'Loaded video source {self.video_path.name}')
        else:
            self.is_ready = False
            # TODO: Using filename only for now. Maybe use a sanitized path?
            log.error(f'Failed to load video source {self.video_path.name}')
            self.cap.release()
            self.cap = None

    def add_locator_point(self, point) -> None:
        '''Add a potential locator point.'''
        if len(self._loc_pts) < len(Flight.point_names):
            self._loc_pts.append(point)
            self.on_locator_points_changed()

    def pop_locator_point(self) -> None:
        '''Remove the last added locator point, if it exists.'''
        if self._loc_pts:
            _ = self._loc_pts.pop()
            self.on_locator_points_changed()

    def on_locator_points_changed(self):
        '''Emits the `locator_points_changed` signal.'''
        self.locator_points_changed.emit(self._loc_pts)
