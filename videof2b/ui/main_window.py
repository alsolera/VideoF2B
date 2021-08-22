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
The main GUI window of VideoF2B application.
'''

import logging
from datetime import datetime, timedelta

from PySide6 import QtCore, QtGui, QtWidgets
from videof2b.core.common.store import StoreProperties
from videof2b.core.processor import VideoProcessor
from videof2b.ui.video_load_dialog import LoadVideoDialog
from videof2b.ui.video_window import VideoWindow

log = logging.getLogger(__name__)


class Ui_MainWindow:
    '''Define the UI layout.'''

    def setup_ui(self, main_window):
        '''Create the UI here.'''
        main_window.setObjectName('MainWindow')
        main_window.setDockNestingEnabled(True)
        main_window.setWindowTitle(
            f'{self.application.applicationName()} '
            f'{self.application.applicationVersion()}'
        )
        self.main_widget = QtWidgets.QWidget(main_window)
        self.main_widget.setObjectName('main_widget')
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(3, 3, 3, 3)
        self.main_layout.setObjectName('main_layout')
        main_window.setCentralWidget(self.main_widget)
        # Figure checkboxes
        # TODO: add button(s) to advance the current checkbox to the next figure
        self.pnl_chk = QtWidgets.QVBoxLayout()
        self.chk_loops = QtWidgets.QCheckBox('Loops', main_window)
        self.chk_sq_loops = QtWidgets.QCheckBox('Square loops', main_window)
        self.chk_tri_loops = QtWidgets.QCheckBox('Triangular loops', main_window)
        self.chk_hor_eight = QtWidgets.QCheckBox('Horizontal eight', main_window)
        self.chk_sq_hor_eight = QtWidgets.QCheckBox('Square horizontal eight', main_window)
        self.chk_ver_eight = QtWidgets.QCheckBox('Vertical eight', main_window)
        self.chk_hourglass = QtWidgets.QCheckBox('Hourglass', main_window)
        self.chk_over_eight = QtWidgets.QCheckBox('Overhead eight', main_window)
        self.chk_clover = QtWidgets.QCheckBox('Four-leaf clover', main_window)
        self.chk_diag = QtWidgets.QCheckBox('Draw Diagnostics', main_window)
        self.pnl_chk.addWidget(self.chk_loops)
        self.pnl_chk.addWidget(self.chk_sq_loops)
        self.pnl_chk.addWidget(self.chk_tri_loops)
        self.pnl_chk.addWidget(self.chk_hor_eight)
        self.pnl_chk.addWidget(self.chk_sq_hor_eight)
        self.pnl_chk.addWidget(self.chk_ver_eight)
        self.pnl_chk.addWidget(self.chk_hourglass)
        self.pnl_chk.addWidget(self.chk_over_eight)
        self.pnl_chk.addWidget(self.chk_clover)
        self.pnl_chk.addWidget(self.chk_diag)
        self.pnl_chk.addItem(QtWidgets.QSpacerItem(20, 40, vData=QtWidgets.QSizePolicy.Expanding))

        self.pnl_top = QtWidgets.QHBoxLayout()
        self.pnl_top.addLayout(self.pnl_chk)
        # Video window
        self.video_window = VideoWindow(main_window)
        self.video_window.setObjectName('video_window')
        self.video_window.setMinimumSize(640, 360)
        self.pnl_top.addWidget(self.video_window, stretch=1)
        self.main_layout.addLayout(self.pnl_top, stretch=1)
        # Message panel
        self.output_txt = QtWidgets.QTextEdit(main_window)
        self.output_txt.setFontFamily('Consolas')
        self.output_txt.setReadOnly(True)
        self.output_txt.setMinimumHeight(40)
        self.output_txt.setMaximumHeight(100)
        self.main_layout.addWidget(self.output_txt)
        # Menu
        self.menu_bar = QtWidgets.QMenuBar(main_window)
        self.menu_bar.setObjectName('menu_bar')
        self.file_menu = QtWidgets.QMenu(self.menu_bar)
        self.tools_menu = QtWidgets.QMenu(self.menu_bar)
        self.help_menu = QtWidgets.QMenu(self.menu_bar)
        main_window.setMenuBar(self.menu_bar)
        # Status bar
        self.status_bar = QtWidgets.QStatusBar(main_window)
        self.status_bar.setObjectName('status_bar')
        # Status bar: video filename
        self.filename_label = QtWidgets.QLabel(self.status_bar)
        self.filename_label.setObjectName('filename_label')
        self.status_bar.addPermanentWidget(self.filename_label, stretch=2)
        # Status bar: processing progress bar
        self.proc_progress_bar = QtWidgets.QProgressBar(self.status_bar)
        self.proc_progress_bar.setObjectName('proc_progress_bar')
        self.status_bar.addPermanentWidget(self.proc_progress_bar, stretch=1)
        self.proc_progress_bar.hide()
        self.proc_progress_bar.setValue(0)
        main_window.setStatusBar(self.status_bar)
        # Actions
        action = QtGui.QAction(main_window)
        action.setText('&Load')
        action.setStatusTip('Load a video source')
        action.setToolTip('Load a video source')
        action.setShortcut(QtGui.QKeySequence.Open)
        self.act_file_load = action
        self.file_menu.addAction(self.act_file_load)
        self.file_menu.setTitle("&File")
        action = QtGui.QAction(main_window)
        action.setText('E&xit')
        action.setShortcut(QtCore.Qt.CTRL | QtCore.Qt.Key_Q)
        self.act_file_exit = action
        self.file_menu.addAction(self.act_file_exit)
        #
        action = QtGui.QAction(main_window)
        action.setText('&Calibrate camera..')
        self.act_tools_cal_cam = action
        self.tools_menu.addAction(self.act_tools_cal_cam)
        self.tools_menu.setTitle('&Tools')
        action = QtGui.QAction(main_window)
        action.setText('&Place camera..')
        self.act_tools_place_cam = action
        self.tools_menu.addAction(self.act_tools_place_cam)
        #
        action = QtGui.QAction(main_window)
        action.setText('&About')
        self.act_help_about = action
        self.help_menu.addAction(self.act_help_about)
        self.help_menu.setTitle('&Help')
        self.menu_bar.addAction(self.file_menu.menuAction())
        self.menu_bar.addAction(self.tools_menu.menuAction())
        self.menu_bar.addAction(self.help_menu.menuAction())
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_Space)
        self.act_clear_track = action
        main_window.addAction(self.act_clear_track)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_Left)
        self.act_rotate_left = action
        main_window.addAction(self.act_rotate_left)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_Right)
        self.act_rotate_right = action
        main_window.addAction(self.act_rotate_right)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_W)
        self.act_move_north = action
        main_window.addAction(self.act_move_north)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_S)
        self.act_move_south = action
        main_window.addAction(self.act_move_south)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_A)
        self.act_move_west = action
        main_window.addAction(self.act_move_west)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_D)
        self.act_move_east = action
        main_window.addAction(self.act_move_east)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_X)
        self.act_move_reset = action
        main_window.addAction(self.act_move_reset)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_C)
        self.act_relocate_cam = action
        main_window.addAction(self.act_relocate_cam)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_P)
        self.act_pause = action
        main_window.addAction(self.act_pause)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_BracketLeft)
        self.act_figure_start = action
        main_window.addAction(self.act_figure_start)
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.Key_BracketRight)
        self.act_figure_end = action
        main_window.addAction(self.act_figure_end)
        # Finish
        self.setLayout(self.main_layout)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow, StoreProperties):
    '''Our main window.'''

    def __init__(self):
        super().__init__()
        log.debug('Creating MainWindow')
        # Set up the interface
        self.setup_ui(self)
        # Set up internal objects
        self._proc = VideoProcessor()
        # Set up signals and slots
        self.act_file_load.triggered.connect(self.on_load_video)
        self.act_file_exit.triggered.connect(self.on_close)
        self.video_window.point_added.connect(self._proc.add_locator_point)
        self.video_window.point_removed.connect(self._proc.pop_locator_point)
        self._proc.locator_points_changed.connect(self.on_loc_pts_changed)
        self._proc.progress_updated.connect(self.on_progress_updated)
        self.act_clear_track.triggered.connect(self.on_clear_track)
        self.act_rotate_left.triggered.connect(self.on_rotate_left)
        self.act_rotate_right.triggered.connect(self.on_rotate_right)
        self.act_move_north.triggered.connect(self.on_move_north)
        self.act_move_south.triggered.connect(self.on_move_south)
        self.act_move_west.triggered.connect(self.on_move_west)
        self.act_move_east.triggered.connect(self.on_move_east)
        self.act_move_reset.triggered.connect(self.on_move_reset)
        self.act_relocate_cam.triggered.connect(self.on_relocate_cam)
        self.act_pause.triggered.connect(self.on_pause_unpause)
        self.act_figure_start.triggered.connect(self.on_figure_start)
        self.act_figure_end.triggered.connect(self.on_figure_end)
        self.move(5, 5)  # TODO: center on main screen

    def on_close(self, event):
        # TODO: need to handle all exit/close/quit requests here, ensuring all threads are stopped before we close.
        log.debug(event)

    def on_loc_pts_changed(self):
        '''Handler to echo changes in the VideoProcessor's locator points.'''
        print('in MainWindow.on_loc_pts_changed()')
        # self._print_pts(points)

    def _output_msg(self, msg):
        '''Append a message to the output text box.'''
        self.output_txt.append(f'{datetime.now()} : {msg}')
        # log.info(msg)

    def _print_pts(self, points):
        if points:
            q = ['Image points:']
            for i, p in enumerate(points):
                q += [f'  {i+1:2d} : ({p.x():4d}, {p.y():4d})']
            self._output_msg('\n'.join(q))

    def on_load_video(self):
        diag = LoadVideoDialog(self)
        if diag.exec() == QtWidgets.QDialog.Accepted:
            # At this point, the flight data is validated. Load it into the processor.
            self._proc.load_flight(diag.flight)
            # Enable mouse if cam locating is necessary
            self.video_window.is_mouse_enabled = diag.flight.is_calibrated
            self.proc_progress_bar.show()
            self._output_msg(f'Video loaded successfully from {self._proc.flight.video_path}')
            # Using Qt.QueuedConnection here in signal connect() based on
            # https://stackoverflow.com/questions/2823112/communication-between-threads-in-pyside
            self._proc.new_frame_available.connect(
                self.video_window.update_frame,
                QtCore.Qt.QueuedConnection
            )
            self._proc.start()

    # @QtCore.Slot(tuple) #TODO: is this decorator necessary anymore?
    def on_progress_updated(self, data):
        '''Display video processing progress.'''
        frame_time, progress = data
        # TODO: This is a temporary output here, it's simpler to just update the progress bar..
        # TODO: It would be nice to use `frame_time` to update the current video time in the status bar!
        # self._output_msg(f'video time: {timedelta(seconds=frame_time)}, progress: {progress:3d}%')
        # Update the progress bar
        self.proc_progress_bar.setValue(progress)

    # ====================================================
    # TODO: connect all these handlers to VideoProcessor.
    # ====================================================

    def on_clear_track(self):
        self._output_msg('CLEAR TRACK')

    def on_rotate_left(self):
        self._output_msg('rotate LEFT')

    def on_rotate_right(self):
        self._output_msg('rotate RIGHT')

    def on_move_north(self):
        self._output_msg('move NORTH')

    def on_move_south(self):
        self._output_msg('move SOUTH')

    def on_move_west(self):
        self._output_msg('move WEST')

    def on_move_east(self):
        self._output_msg('move EAST')

    def on_move_reset(self):
        self._output_msg('move RESET to center')

    def on_relocate_cam(self):
        self._output_msg('RELOCATE CAM')

    def on_figure_start(self):
        self._output_msg('FIGURE START')

    def on_figure_end(self):
        self._output_msg('FIGURE END')

    def on_pause_unpause(self):
        self._output_msg('PAUSE / UNPAUSE')
