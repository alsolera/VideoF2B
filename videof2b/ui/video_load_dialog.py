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

'''
The dialog that loads the input video.
'''

from videof2b.core.common.path import path_to_str, str_to_path

from PySide6 import QtCore, QtGui, QtWidgets
from videof2b.core.common import (DEFAULT_FLIGHT_RADIUS, DEFAULT_MARKER_HEIGHT,
                                  DEFAULT_MARKER_RADIUS)
from videof2b.core.common.store import StoreProperties
from videof2b.core.flight import Flight
from videof2b.ui import EXTENSIONS_VIDEO
from videof2b.ui.widgets import PathEdit, PathEditType


class LoadVideoDialog(QtWidgets.QDialog, StoreProperties):
    def __init__(self, parent) -> None:
        super().__init__(
            parent,
            QtCore.Qt.WindowSystemMenuHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.WindowCloseButtonHint
        )
        # TODO: try to implement this UI with a QAbstractListModel or a QStandardItemModel for proper cohesion.
        # see https://doc.qt.io/qtforpython/overviews/model-view-programming.html#models
        self.flight = None
        self.start_processing = False
        self.setup_ui()
        self.setWindowTitle('Load Video')
        self.start_proc_chk.clicked.connect(self.on_start_proc_checked)
        self.cancel_btn.clicked.connect(self.reject)
        self.load_btn.clicked.connect(self.accept)

    def setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setObjectName('main_layout')
        self.live_chk = QtWidgets.QCheckBox('&Live video', self)
        self.video_path_lbl = QtWidgets.QLabel('Video source:', self)
        self.video_path_txt = PathEdit(self, PathEditType.Files,
                                       'Select video file',
                                       str_to_path(self.settings.value('mru/video_dir')))  # TODO: there must be a cleaner way to convert path<->str in settings!
        self.video_path_txt.filters = f'Video files ({" ".join(EXTENSIONS_VIDEO)});;All files (*)'
        self.cal_path_lbl = QtWidgets.QLabel('Calibration file:', self)
        self.cal_path_txt = PathEdit(self, PathEditType.Files,
                                     'Select calibration file',
                                     str_to_path(self.settings.value('mru/cal_dir')))  # TODO: there must be a cleaner way to convert path<->str in settings!
        self.cal_path_txt.filters = 'Calibration files (*.npz);;All files (*)'
        # Measurement inputs
        self.flight_radius_lbl = QtWidgets.QLabel(
            'Flight radius (m)', self)
        self.marker_radius_lbl = QtWidgets.QLabel(
            'Height markers: distance to center (m)', self)
        self.marker_height_lbl = QtWidgets.QLabel(
            'Height markers: height above center of circle (m)', self)
        self.flight_radius_txt = QtWidgets.QLineEdit(f'{DEFAULT_FLIGHT_RADIUS:.1f}', self)
        self.marker_radius_txt = QtWidgets.QLineEdit(f'{DEFAULT_MARKER_RADIUS:.1f}', self)
        self.marker_height_txt = QtWidgets.QLineEdit(f'{DEFAULT_MARKER_HEIGHT:.1f}', self)
        self.meas_grid = QtWidgets.QGridLayout()
        self.meas_grid.setObjectName('meas_grid')
        self.meas_grid.addWidget(self.flight_radius_lbl, 1, 1, 1, 1)
        self.meas_grid.addWidget(self.flight_radius_txt, 1, 2, 1, 1)
        self.meas_grid.addWidget(self.marker_radius_lbl, 2, 1, 1, 1)
        self.meas_grid.addWidget(self.marker_radius_txt, 2, 2, 1, 1)
        self.meas_grid.addWidget(self.marker_height_lbl, 3, 1, 1, 1)
        self.meas_grid.addWidget(self.marker_height_txt, 3, 2, 1, 1)
        #
        self.start_proc_chk = QtWidgets.QCheckBox('Start processing immediately', self)
        self.load_btn = QtWidgets.QPushButton('Load', self)
        self.load_btn.setDefault(True)
        self.cancel_btn = QtWidgets.QPushButton('Cancel', self)
        self.bottom_layout = QtWidgets.QHBoxLayout()
        self.bottom_layout.addWidget(self.load_btn)
        self.bottom_layout.addWidget(self.cancel_btn)
        self.main_layout.addWidget(self.live_chk)
        self.main_layout.addWidget(self.video_path_lbl)
        self.main_layout.addWidget(self.video_path_txt)
        self.main_layout.addWidget(self.cal_path_lbl)
        self.main_layout.addWidget(self.cal_path_txt)
        self.main_layout.addLayout(self.meas_grid)
        self.main_layout.addWidget(self.start_proc_chk)
        self.main_layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        self.main_layout.addLayout(self.bottom_layout)
        # Actions
        # action = QtGui.QAction(self)
        # action.setChecked(False)
        # action.setCheckable(True)
        # self.act_live_chk = action
        # self.live_chk.addAction(self.act_live_chk)

    def on_start_proc_checked(self, val):
        self.start_processing = val

    def accept(self) -> None:
        if not self.validate():
            return
        self.flight = Flight(
            self.video_path_txt.path,
            cal_path=self.cal_path_txt.path,
            is_live=self.live_chk.isChecked(),
            # TODO: use proper validation for these numeric fields!
            flight_radius=float(self.flight_radius_txt.text()),
            marker_radius=float(self.marker_radius_txt.text()),
            marker_height=float(self.marker_height_txt.text())
        )
        # TODO: there must be a cleaner way to convert path<->str in settings!
        self.settings.setValue('mru/video_dir', path_to_str(self.video_path_txt.path.parent))
        # Cal path is optional, so check it first
        cal_path = self.cal_path_txt.path
        if path_to_str(cal_path) and cal_path.exists():
            self.settings.setValue('mru/cal_dir', path_to_str(cal_path.parent))
        # Do not proceed if flight failed to load
        if not self.flight.is_ready:
            QtWidgets.QMessageBox.critical(
                self, 'Error',
                'Failed to load video source.',
                QtWidgets.QMessageBox.Ok
            )
            return None
        return super().accept()

    def validate(self) -> bool:
        '''Validate user inputs.'''
        result = True
        # TODO: this is rudimentary for now. Using a Qt Model with field validators would be more ideal.
        vid_path = self.video_path_txt.path
        if vid_path is None or not vid_path.exists():
            QtWidgets.QMessageBox.critical(
                self, 'Error',
                'Please specify a valid video source.',
                QtWidgets.QMessageBox.Ok
            )
            result = False
        try:
            float(self.flight_radius_txt.text())
            float(self.marker_radius_txt.text())
            float(self.marker_height_txt.text())
        except ValueError as err:
            QtWidgets.QMessageBox.critical(
                self, 'Error',
                f'ERROR: {err}\n'
                'Please verify numeric inputs.',
                QtWidgets.QMessageBox.Ok
            )
            result = False
        return result
