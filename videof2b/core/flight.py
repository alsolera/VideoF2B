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

'''Defines a recorded flight.'''

import logging
from pathlib import Path

from imutils.video import FileVideoStream
from PySide6.QtCore import QObject, Signal
from videof2b.core import common

log = logging.getLogger(__name__)


class Flight(QObject):
    '''Contains information about a flight to be processed.'''

    # --- Signals
    # Indicates that the current locator points changed during the locating procedure.
    locator_points_changed = Signal(object, str)
    # Indicates that all required points have been defined during the locating procedure.
    locator_points_defined = Signal()

    obj_point_names = (
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
            `skip_locate`: indicates whether to skip the camera locating procedure
                           in a calibrated Flight. Defaults to False.
            `flight_radius`: radius of the flight hemisphere.
            `marker_radius`: radius of the circle where the markers are located.
            `marker_height`: elevation of the marker circle above the center marker.
            `sphere_offset`: XYZ sequence of coordinates of the AR flight sphere
                             with respect to the center of the marker circle.
                             Defaults to origin (0., 0., 0.)
            `loc_pts`: locator points (in pixels). Use when you know what you are doing.
            All dimensions are in meters unless otherwise specified.
        '''
        super().__init__()
        self.num_obj_pts = len(Flight.obj_point_names)
        # Indicates whether the underlying video stream is ready to roll.
        self.is_ready = False
        self.video_path = vid_path
        self.is_live = is_live
        self.calibration_path = cal_path
        self.is_calibrated = cal_path is not None and cal_path.exists()
        self.is_located = False
        self.is_ar_enabled = True
        self.skip_locate = kwargs.pop('skip_locate', False)
        self.flight_radius = kwargs.pop('flight_radius', common.DEFAULT_FLIGHT_RADIUS)
        self.marker_radius = kwargs.pop('marker_radius', common.DEFAULT_MARKER_RADIUS)
        self.marker_height = kwargs.pop('marker_height', common.DEFAULT_MARKER_HEIGHT)
        self.sphere_offset = kwargs.pop('sphere_offset', common.DEFAULT_CENTER)
        rcos45 = self.marker_radius * common.COS_45
        # --- Object points for pose estimation.
        # NOTE: This sequence must reflect the representation of these points in `Flight.obj_point_names`
        self.obj_pts = (
            [0, 0, -self.marker_height],  # Center marker
            [0, self.marker_radius, 0],  # Front marker
            [-rcos45, rcos45, 0],  # Left marker
            [rcos45, rcos45, 0]  # Right marker
        )
        # Locator points. This list grows and shrinks during the locating procedure.
        self.loc_pts = kwargs.pop('loc_pts', [])
        # Log some basic info.
        log.info('Creating a new flight =========')
        log.info(f'      Video path: {self.video_path.name}')
        if self.is_calibrated:
            log.info(f'Calibration path: {self.calibration_path.name}')
        log.info(f'  flight radius = {self.flight_radius} m')
        log.info(f'    mark radius = {self.marker_radius} m')
        log.info(f'    mark height = {self.marker_height} m')
        log.info(f'  sphere offset = {self.sphere_offset} m')
        # Load the video stream
        self._load_stream()
        # Log some more to report the stream's status.
        if self.is_ready:
            log.info('Video stream is ready.')
        else:
            log.warning('Video stream IS NOT ready.')

    def _load_stream(self):
        '''Load this flight's video stream.'''
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
            self.cap.stop()
            self.cap = None

    def restart(self):
        '''Restart this flight's video stream.'''
        self._load_stream()

    def add_locator_point(self, point) -> None:
        '''Add a potential locator point.'''
        if len(self.loc_pts) < self.num_obj_pts:
            self.loc_pts.append(point)
            self.on_locator_points_changed()
        # If we just reached the limit, then emit
        # that all required points have been defined.
        if len(self.loc_pts) == self.num_obj_pts:
            self.locator_points_defined.emit()

    def pop_locator_point(self) -> None:
        '''Remove the last added locator point, if it exists.'''
        if self.loc_pts:
            _ = self.loc_pts.pop()
            self.on_locator_points_changed()

    def clear_locator_points(self):
        '''Clear all locator points.'''
        self.loc_pts.clear()
        self.on_locator_points_changed()

    def on_locator_points_changed(self):
        '''Signals that locator points changed, and the new instruction message.'''
        idx = len(self.loc_pts)
        if idx < len(self.obj_point_names):
            msg = f'Click {self.obj_point_names[idx]}.'
        else:
            msg = 'All points defined.'
        self.locator_points_changed.emit(self.loc_pts, msg)
