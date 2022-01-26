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

'''
The dialog window for camera calibration.
'''

from pathlib import Path

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QImage
from videof2b.core.common import get_bundle_dir, launch_document
from videof2b.core.common.store import StoreProperties
from videof2b.ui import EXTENSIONS_VIDEO
from videof2b.ui.video_window import VideoWindow
from videof2b.ui.widgets import PathEdit, PathEditType


class CameraCalibrationDialog(QtWidgets.QDialog, StoreProperties):
    '''Camera calibration UI.'''

    def __init__(self, parent) -> None:
        super().__init__(
            parent,
            QtCore.Qt.WindowSystemMenuHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.WindowCloseButtonHint
        )
        self.cal_path: Path = None
        self.is_fisheye: bool = False
        self.assets_path = get_bundle_dir() / 'resources' / 'cal'
        self.cal_img_path = self.assets_path / 'pattern_10x7.svg'
        self.cal_doc_path = self.assets_path / 'pattern_10x7.pdf'
        self.cal_vid_th_path = self.assets_path / 'sample_vid_th.png'
        self.setup_ui()
        self.cal_path_txt.path_changed.connect(self.on_path_changed)
        self.fisheye_chk.stateChanged.connect(self.on_fisheye_changed)
        self.cancel_btn.clicked.connect(self.reject)
        self.start_btn.clicked.connect(self.accept)

    def setup_ui(self):
        '''Designs the UI.'''
        self.setWindowTitle('Calibrate your camera')
        self.setContentsMargins(5, 5, 5, 5)
        self.content_lbl_1 = QtWidgets.QLabel(self)
        self.content_lbl_1.setText(f"""
        <h1>Camera Calibration Procedure</h1>

        <h2>1. Obtain calibration pattern</h2>

        <p style="text-align:center;">The following chessboard pattern is used for calibration:</p>
        <p style="text-align:center;"><img src="{self.cal_img_path}" width="170"></img></p>

        <p style="text-align:center;">Click one of the following: display the pattern on screen, or print it on paper.</p>
        """)
        self.content_lbl_2 = QtWidgets.QLabel(self)
        self.content_lbl_2.setText(f"""
        <h2>2. Record calibration video</h2>
        <p style="text-align:center;">Use the result pattern and your camera to record a video like this example:</p>
        <p style="text-align:center;"><a href="https://youtu.be/DeNIlietn9E"><img src="{self.cal_vid_th_path}" width="170"></img></a></p>
        <p style="text-align:center;">(Click to view in your browser.)</p>

        <h2>3. Load calibration video and press Start</h2>
        """)
        for label in (self.content_lbl_1, self.content_lbl_2):
            label.setTextFormat(QtCore.Qt.RichText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
            label.setOpenExternalLinks(True)
        self.image_btn = QtWidgets.QPushButton("Display", self)
        self.image_btn.clicked.connect(self.on_image_display)
        self.doc_btn = QtWidgets.QPushButton("Print", self)
        self.doc_btn.clicked.connect(self.on_doc_open)
        #
        self.cal_path_lbl = QtWidgets.QLabel('Choose video file:', self)
        self.cal_path_txt = PathEdit(
            self, PathEditType.Files,
            'Choose a calibration video file',
            self.settings.value('mru/cal_dir')
        )
        self.cal_path_txt.filters = f'Video files ({" ".join(EXTENSIONS_VIDEO)});;All files (*)'
        self.fisheye_chk = QtWidgets.QCheckBox('Fisheye model', self)
        self.fisheye_chk.setChecked(self.is_fisheye)
        #
        self.start_btn = QtWidgets.QPushButton('Start', self)
        self.start_btn.setDefault(True)
        self.start_btn.setEnabled(False)
        self.cancel_btn = QtWidgets.QPushButton('Cancel', self)
        #
        self.path_layout = QtWidgets.QHBoxLayout()
        self.path_layout.addWidget(self.cal_path_lbl)
        self.path_layout.addWidget(self.cal_path_txt)
        #
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.buttons_layout.addStretch(1)
        self.buttons_layout.addWidget(self.start_btn)
        self.buttons_layout.addWidget(self.cancel_btn)
        self.buttons_layout.addStretch(1)
        #
        self.options_layout = QtWidgets.QHBoxLayout()
        self.options_layout.addStretch(1)
        self.options_layout.addWidget(self.image_btn)
        self.options_layout.addSpacing(20)
        self.options_layout.addWidget(self.doc_btn)
        self.options_layout.addStretch(1)
        #
        self.scroll_widget = QtWidgets.QWidget(self)
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(5)
        self.scroll_layout.addWidget(self.content_lbl_1)
        self.scroll_layout.addLayout(self.options_layout)
        self.scroll_layout.addWidget(self.content_lbl_2)
        self.scroll_layout.addLayout(self.path_layout)
        self.scroll_layout.addWidget(self.fisheye_chk, alignment=QtCore.Qt.AlignRight)
        self.fisheye_chk.setVisible(self.settings.value('core/enable_fisheye'))
        self.scroll_layout.addStretch(1)
        #
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setObjectName('main_layout')
        self.scroll_area = QtWidgets.QScrollArea(widgetResizable=True)
        self.scroll_area.setMinimumSize(460, 580)
        self.scroll_area.setWidget(self.scroll_widget)
        #
        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(self.buttons_layout)
        # Finalize
        self.setLayout(self.main_layout)

    def on_doc_open(self):
        '''Open the calibration pattern PDF file.'''
        launch_document(self.cal_doc_path)

    def on_image_display(self):
        '''Show a lightweight window in full screen with the
        calibration pattern. Esc key closes this window by default.'''
        # Using the SplashScreen window flag allows full-screen display
        # on top of all windows including the OS taskbar while reacting
        # to Esc and Alt-Tab in the expected manner.
        img_window = QtWidgets.QDialog(self, QtCore.Qt.SplashScreen)
        img = QImage(str(self.cal_img_path))
        frame = VideoWindow(self)
        frame.update_frame(img)
        layout = QtWidgets.QVBoxLayout(img_window)
        layout.addWidget(frame)
        img_window.showFullScreen()

    def on_path_changed(self, new_path):
        '''Update UI when the calibration path changes.'''
        if new_path is not None and new_path.exists():
            self.cal_path = new_path
            self.start_btn.setEnabled(True)
        else:
            self.cal_path = None
            self.start_btn.setEnabled(False)

    def on_fisheye_changed(self, state):
        '''Update fisheye flag when checkbox state changes.'''
        self.is_fisheye = state == QtCore.Qt.Checked
