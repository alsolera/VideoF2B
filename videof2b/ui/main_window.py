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
from videof2b.core.common import FigureTypes, SphereManipulations
from videof2b.core.common.store import StoreProperties
from videof2b.core.processor import ProcessorReturnCodes, VideoProcessor
from videof2b.ui.load_flight_dialog import LoadFlightDialog
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
        # TODO: group the checkboxes together for simpler programmatic access.
        self.pnl_chk = QtWidgets.QVBoxLayout()
        self.chk_loops = QtWidgets.QCheckBox('Loops', main_window)
        self.chk_loops.setObjectName('chk_loops')
        self.chk_sq_loops = QtWidgets.QCheckBox('Square loops', main_window)
        self.chk_sq_loops.setObjectName('chk_sq_loops')
        self.chk_tri_loops = QtWidgets.QCheckBox('Triangular loops', main_window)
        self.chk_tri_loops.setObjectName('chk_tri_loops')
        self.chk_hor_eight = QtWidgets.QCheckBox('Horizontal eight', main_window)
        self.chk_hor_eight.setObjectName('chk_hor_eight')
        self.chk_sq_hor_eight = QtWidgets.QCheckBox('Square horizontal eight', main_window)
        self.chk_sq_hor_eight.setObjectName('chk_sq_hor_eight')
        self.chk_ver_eight = QtWidgets.QCheckBox('Vertical eight', main_window)
        self.chk_ver_eight.setObjectName('chk_ver_eight')
        self.chk_hourglass = QtWidgets.QCheckBox('Hourglass', main_window)
        self.chk_hourglass.setObjectName('chk_hourglass')
        self.chk_over_eight = QtWidgets.QCheckBox('Overhead eight', main_window)
        self.chk_over_eight.setObjectName('chk_over_eight')
        self.chk_clover = QtWidgets.QCheckBox('Four-leaf clover', main_window)
        self.chk_clover.setObjectName('chk_clover')
        # All figure checkboxes, excluding the diag checkbox.
        self.fig_chk_boxes = (
            self.chk_loops,
            self.chk_sq_loops,
            self.chk_tri_loops,
            self.chk_hor_eight,
            self.chk_sq_hor_eight,
            self.chk_ver_eight,
            self.chk_hourglass,
            self.chk_over_eight,
            self.chk_clover
        )
        for fig_chk in self.fig_chk_boxes:
            self.pnl_chk.addWidget(fig_chk)
        self.chk_diag = QtWidgets.QCheckBox('Draw Diagnostics', main_window)
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
        # Status bar: instruction label
        self.instruct_lbl = QtWidgets.QLabel(self.status_bar)
        self.instruct_lbl.setObjectName('instruct_lbl')
        self.status_bar.addPermanentWidget(self.instruct_lbl, stretch=1)
        # Status bar: processing progress bar
        self.proc_progress_bar = QtWidgets.QProgressBar(self.status_bar)
        self.proc_progress_bar.setObjectName('proc_progress_bar')
        self.status_bar.addPermanentWidget(self.proc_progress_bar, stretch=1)
        self.proc_progress_bar.hide()
        self.proc_progress_bar.setValue(0)
        main_window.setStatusBar(self.status_bar)
        # Actions
        # TODO: refactor all this repetitive creation of actions into helper functions.
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
        action.setShortcut(QtCore.Qt.Key_Escape)
        self.act_stop_proc = action
        main_window.addAction(self.act_stop_proc)
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
        self.act_pause_resume = action
        main_window.addAction(self.act_pause_resume)
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
        #
        action = QtGui.QAction(main_window)
        action.setShortcut(QtCore.Qt.CTRL | QtCore.Qt.Key_R)
        self.act_restart_flight = action
        main_window.addAction(self.act_restart_flight)
        # Finish
        self.setLayout(self.main_layout)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow, StoreProperties):
    '''The main UI window of the VideoF2B application.'''

    # Signals
    stop_processor = QtCore.Signal()
    locating_completed = QtCore.Signal()
    clear_track = QtCore.Signal()
    figure_state_changed = QtCore.Signal(FigureTypes, bool)
    figure_diags_changed = QtCore.Signal(bool)
    relocate_cam = QtCore.Signal()
    manipulate_sphere = QtCore.Signal(SphereManipulations)
    figure_mark = QtCore.Signal(bool)
    pause_resume = QtCore.Signal()

    # Mapping of processor retcodes to user-friendly messages.
    # Keep this dict in sync with all current definitions in ProcessorReturnCodes.
    _retcodes_msgs = {
        ProcessorReturnCodes.ExceptionOccurred: 'Critical unhandled error occurred.',
        ProcessorReturnCodes.Undefined: 'Video processing never started.',
        ProcessorReturnCodes.Normal: 'Video processing finished normally.',
        ProcessorReturnCodes.UserCanceled: 'Video processing was cancelled by user.',
        ProcessorReturnCodes.PoseEstimationFailed: 'Failed to locate the camera!',
        ProcessorReturnCodes.TooManyEmptyFrames: 'Encountered too many consecutive empty frames.',
    }

    # Maps object names of figure checkboxes to the common.FigureType required by core.Drawing.
    _fig_chk_names_types = {
        'chk_loops': FigureTypes.INSIDE_LOOPS,
        'chk_sq_loops': FigureTypes.INSIDE_SQUARE_LOOPS,
        'chk_tri_loops': FigureTypes.INSIDE_TRIANGULAR_LOOPS,
        'chk_hor_eight': FigureTypes.HORIZONTAL_EIGHTS,
        'chk_sq_hor_eight': FigureTypes.HORIZONTAL_SQUARE_EIGHTS,
        'chk_ver_eight': FigureTypes.VERTICAL_EIGHTS,
        'chk_hourglass': FigureTypes.HOURGLASS,
        'chk_over_eight': FigureTypes.OVERHEAD_EIGHTS,
        'chk_clover': FigureTypes.FOUR_LEAF_CLOVER
    }

    def __init__(self):
        '''Constructor.'''
        super().__init__()
        log.debug('Creating MainWindow')
        # Set up the interface
        self.setup_ui(self)
        self._is_window_closing = False
        # Set up internal objects
        self._proc = None
        self._proc_thread = None
        self._last_flight = None
        # Set up signals and slots that are NOT related to VideoProcessor
        self.act_file_load.triggered.connect(self.on_load_flight)
        # TODO: The restart action is currently hidden because it's a nice feature for developer convenience. Should it become visible?
        # self.act_restart_flight.triggered.connect(self.on_restart_flight)
        self.act_file_exit.triggered.connect(self.close)
        # TODO: center on main screen on first use, or use saved Settings if available.
        self.move(5, 5)

    def closeEvent(self, event):
        '''Overridden to handle the closing of the main window in a safe manner.
        Handles all exit/close/quit requests here, ensuring all threads are stopped
        before we close.
        '''
        log.debug('Entering MainWindow.closeEvent()')
        self._is_window_closing = True
        if hasattr(self, '_proc_thread'):
            log.debug(f'self._proc_thread object: {self._proc_thread}')
        else:
            # This is hopefully a debugging aid for future..just in case.
            log.warning('self._proc_thread does not exist!')
        if self._proc_thread is not None and self._proc_thread.isRunning():
            # NOTE: we currently do not pause the processing loop
            # while we display this dialog to the user, but that
            # is an option to consider.
            ret = QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                'Flight processing is still running. '
                'Do you want to stop it and quit?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if ret == QtWidgets.QMessageBox.Yes:
                log.debug('Requesting VideoProcessor to stop...')
                self.on_stop_proc()
                log.debug('Request sent. Waiting for VideoProcessorThread to exit...')
            else:
                log.debug('User changed mind about exiting early.')
                self._is_window_closing = False
            # IMPORTANT: ignore the event here to allow the proc thread's
            # `finished` signal to do its work in its own time.  When the
            # proc thread exits, this closeEvent() method will be called
            # again and then the `else` will execute.
            event.ignore()
        else:
            log.debug('VideoProcessor thread is not running. MainWindow is closing...')
            event.accept()

    def on_locating_started(self):
        '''Prepare UI for the camera locating procedure.'''
        self._output_msg(
            'Locating AR geometry. Follow the instructions in the status bar. '
            'Left-click to add a point, right-click to remove last point.')
        self.instruct_lbl.show()
        self.video_window.is_mouse_enabled = True

    def on_loc_pts_changed(self, points, msg):
        '''Echoes changes in the VideoProcessor's locator points
        and updates the instruction message.'''
        self._output_pts(points)
        self.instruct_lbl.setStyleSheet('QLabel { color : red; }')
        self.instruct_lbl.setText(msg)

    def on_loc_pts_defined(self):
        '''Present the user with a confirm/redefine choice via messagebox.'''
        self.instruct_lbl.setStyleSheet('QLabel { color : green; }')
        self.instruct_lbl.setText('Fully defined.')
        ret = QtWidgets.QMessageBox.information(
            self,
            'Locating completed',
            'Locating points are fully defined. Proceed with processing?\n'
            'Press Yes to proceed, No to redefine the points.',
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        )
        if ret == QtWidgets.QMessageBox.Yes:
            # Done locating. Disable the mouse reactions in video window.
            self.video_window.is_mouse_enabled = False
            self.instruct_lbl.setText('')
            self.instruct_lbl.hide()
            # Signal to the processor that we're done.
            self.locating_completed.emit()
            self._pre_processing()
        else:
            self._last_flight.clear_locator_points()

    def _output_msg(self, msg):
        '''Append a message to the output text box and log it.'''
        self.output_txt.append(f'{datetime.now()} : {msg}')
        log.info(msg)

    def _output_pts(self, points):
        '''Output locator points to log.'''
        q = ['Image points:']
        for i, p in enumerate(points):
            q += [f'  {i+1:1d} : ({",".join(f"{pp:4d}" for pp in p)})']
        if not points:
            q += ['  <empty>']
        log.info('\n'.join(q))

    def on_chk_figure_changed(self):
        '''Tell the processor to toggle the drawn state of a figure.'''
        # Leverage each checkbox's object name to send the appropriate params.
        sender = self.sender()
        name = sender.objectName()
        self.figure_state_changed.emit(
            self._fig_chk_names_types[name],
            sender.isChecked()
        )

    def on_chk_diag_changed(self):
        '''Tell the processor to toggle the drawn state of figure diagnostics.'''
        self.figure_diags_changed.emit(self.sender().isChecked())

    def _init_proc(self):
        '''Create a new instance of the video processor and connect all its signals.'''
        self._proc = VideoProcessor()

        # Connect signals that communicate between threads.
        # DO NOT call the processor's API directly from
        # the main UI thread, or you might have a bad day.
        # NOTE: the connections in this section must use
        # QtCore.Qt.QueuedConnection for cross-thread safety.
        self.stop_processor.connect(self._proc.stop, QtCore.Qt.QueuedConnection)
        self.locating_completed.connect(self._proc.stop_locating, QtCore.Qt.QueuedConnection)
        self.clear_track.connect(self._proc.clear_track, QtCore.Qt.QueuedConnection)
        self._proc.track_cleared.connect(self.on_track_cleared, QtCore.Qt.QueuedConnection)
        self._proc.progress_updated.connect(self.on_progress_updated, QtCore.Qt.QueuedConnection)
        self._proc.ar_geometry_available.connect(self.on_ar_geometry_available, QtCore.Qt.QueuedConnection)
        self._proc.new_frame_available.connect(self.video_window.update_frame, QtCore.Qt.QueuedConnection)
        self._proc.locating_started.connect(self.on_locating_started, QtCore.Qt.QueuedConnection)
        self._proc.locator_points_changed.connect(self.on_loc_pts_changed, QtCore.Qt.QueuedConnection)
        self._proc.locator_points_defined.connect(self.on_loc_pts_defined, QtCore.Qt.QueuedConnection)
        self.video_window.point_added.connect(self._proc.add_locator_point, QtCore.Qt.QueuedConnection)
        self.video_window.point_removed.connect(self._proc.pop_locator_point, QtCore.Qt.QueuedConnection)
        self.figure_state_changed.connect(self._proc.update_figure_state, QtCore.Qt.QueuedConnection)
        self.figure_diags_changed.connect(self._proc.update_figure_diags, QtCore.Qt.QueuedConnection)
        self.pause_resume.connect(self._proc.pause_resume, QtCore.Qt.QueuedConnection)
        self._proc.paused.connect(self.on_paused_resumed, QtCore.Qt.QueuedConnection)
        self.relocate_cam.connect(self._proc.relocate, QtCore.Qt.QueuedConnection)
        self.manipulate_sphere.connect(self._proc.manipulate_sphere, QtCore.Qt.QueuedConnection)
        self.figure_mark.connect(self._proc.mark_figure, QtCore.Qt.QueuedConnection)

        # Actions and signals that are within the UI thread.
        # NOTE: all of these must be disconnected in self._deinit_proc(),
        # or we will end up with multiple connections after more than one
        # flight is loaded during one application session.
        self.act_stop_proc.triggered.connect(self.on_stop_proc)
        self.act_clear_track.triggered.connect(self.on_clear_track)
        for fig_chk in self.fig_chk_boxes:
            fig_chk.stateChanged.connect(self.on_chk_figure_changed)
        self.chk_diag.stateChanged.connect(self.on_chk_diag_changed)
        self.act_rotate_left.triggered.connect(self.on_rotate_ccw)
        self.act_rotate_right.triggered.connect(self.on_rotate_cw)
        self.act_move_north.triggered.connect(self.on_move_north)
        self.act_move_south.triggered.connect(self.on_move_south)
        self.act_move_west.triggered.connect(self.on_move_west)
        self.act_move_east.triggered.connect(self.on_move_east)
        self.act_move_reset.triggered.connect(self.on_move_reset)
        self.act_relocate_cam.triggered.connect(self.on_relocate_cam)
        self.act_pause_resume.triggered.connect(self.on_pause_resume)
        self.act_figure_start.triggered.connect(self.on_figure_start)
        self.act_figure_end.triggered.connect(self.on_figure_end)

    def _deinit_proc(self):
        '''Disconnect all of the video processor's signals and dereference the processor.'''
        # This is a companion method to `_init_proc`. Make sure that any signal connections
        # that are made there are disconnected here, except for those directly referencing
        # `self._proc`. Just disconnect the UI thread's actions here.
        # ==================================================================================
        # NOTE: I don't know if there is any better way to manage the creation/destruction
        # of the VideoProcessor instance. When we start the processing thread, we move that
        # instance to that thread, so strictly speaking, it is no longer accessible from the
        # main thread here in the UI.  Accessing it for any reason from here is likely to
        # cause cross-thread exceptions at runtime.  Also, the typical pattern is to connect
        # the processor thread's `finished` signal to the instance's `deleteLater` method,
        # which means we are not guaranteed to have a processor instance at all once any
        # processing loop is finished.  Therefore, we create a new processor on demand and
        # destroy it when the processing thread exits.  This should not be a performance hit
        # for users since the UI currently allows one video to be processed at a time anyway.
        # ==================================================================================
        self.act_stop_proc.triggered.disconnect(self.on_stop_proc)
        self.act_clear_track.triggered.disconnect(self.on_clear_track)
        self.act_rotate_left.triggered.disconnect(self.on_rotate_ccw)
        self.act_rotate_right.triggered.disconnect(self.on_rotate_cw)
        self.act_move_north.triggered.disconnect(self.on_move_north)
        self.act_move_south.triggered.disconnect(self.on_move_south)
        self.act_move_west.triggered.disconnect(self.on_move_west)
        self.act_move_east.triggered.disconnect(self.on_move_east)
        self.act_move_reset.triggered.disconnect(self.on_move_reset)
        self.act_relocate_cam.triggered.disconnect(self.on_relocate_cam)
        self.act_pause_resume.triggered.disconnect(self.on_pause_resume)
        self.act_figure_start.triggered.disconnect(self.on_figure_start)
        self.act_figure_end.triggered.disconnect(self.on_figure_end)
        self._proc = None

    def on_load_flight(self):
        '''Loads a flight via LoadFlightDialog and starts processing it.'''
        # TODO: is there a scenario when we would NOT want to start processing a Flight immediately?
        diag = LoadFlightDialog(self)
        if diag.exec() == QtWidgets.QDialog.Accepted:
            self._init_proc()
            # At this point, the flight data is validated. Load it into the processor.
            self._last_flight = diag.flight
            self._load_flight(diag.flight)

    def _load_flight(self, flight):
        '''Load a specified flight.'''
        if flight is None:
            return
        self._proc.load_flight(flight)
        self._pre_processing()
        self.start_proc_thread()

    def _pre_processing(self):
        '''Prepare the UI before we start processing.'''
        self.proc_progress_bar.show()
        self._output_msg(f'Video loaded successfully from {self._proc.flight.video_path.name}')
        self.filename_label.setText(self._proc.flight.video_path.name)
        for chk in self.fig_chk_boxes:
            chk.setChecked(False)
        self.chk_diag.setChecked(False)

    def on_restart_flight(self):
        '''Reload the current flight and restart it.'''
        # TODO: needs attention. processor exits, but processing thread does not join.
        if self._last_flight is None:
            return
        self.on_stop_proc()
        # Block until proc thread joins us.
        if hasattr(self, '_proc_thread') and self._proc_thread is not None:
            self._proc_thread.wait()
        self._init_proc()
        self._last_flight.restart()
        self._load_flight(self._last_flight)

    def on_proc_starting(self):
        '''Handle required preparations when the processing thread starts.'''
        # Do not allow loading of a Flight while we are processing a loaded one.
        self.act_file_load.setEnabled(False)

    def on_proc_finished(self, retcode: ProcessorReturnCodes):
        '''Let the user know that the video processing finished, and indicate the reason.'''
        # BEWARE: `_proc` is available here only because we arrange
        # the order of signal connections in `start_proc_thread()`
        # so that this runs before the processor instance is deleted.
        # Process any unhandled exceptions from the processing thread:
        if self._proc.exc is not None:
            proc_exc = self._proc.exc
            QtWidgets.QMessageBox.critical(
                self,
                'Critical Error',
                'A surprise error occurred.\n'
                'It may be a bug. Please report this:\n'
                f'{proc_exc}'
            )
            # Re-raise the exception here on the main thread so that excepthook has a chance.
            raise RuntimeError('Problem in VideoProcessor.') from proc_exc

        self._output_msg(self._retcodes_msgs.get(retcode, f'Unknown return code: {retcode}'))
        # TODO: what else do we want to do here?
        # Options:
        #   * blank out the video window
        self.video_window.clear()
        #   * hide the progress bar
        self.proc_progress_bar.hide()
        #   * reset filename_label
        self.filename_label.setText('')
        #   * All of the above?
        #   * Anything else?

    def on_proc_thread_finished(self):
        '''Handles cleanup when the processing thread finishes.'''
        self._deinit_proc()
        # Re-enable the File > Load action.
        self.act_file_load.setEnabled(True)
        # If main window close was requested, close it now.
        if self._is_window_closing:
            self.close()

    def _deinit_proc_thread(self):
        '''Explicitly dereference the processing thread so that
        we don't end up referencing a deleted C++ object in PySide.'''
        log.debug(f'Dereferencing _proc_thread ... self._proc = {self._proc}')
        self._proc_thread = None

    def start_proc_thread(self):
        '''Starts the video processor on a worker thread.'''
        self._proc_thread = QtCore.QThread()
        # A note on QThread naming: =======================================================
        # Sadly, this currently does not work.
        # PySide6 names the threads `Dummy-#` instead.
        # According to PySide6 docs:
        # "Note that this is currently not available with release builds on Windows."
        # See https://doc.qt.io/qtforpython/PySide6/QtCore/QThread.html#managing-threads
        self._proc_thread.setObjectName('VideoProcessorThread')
        self._proc.moveToThread(self._proc_thread)
        self._proc_thread.started.connect(self._proc.run)
        self._proc.finished.connect(self.on_proc_finished)
        self._proc.finished.connect(self._proc_thread.exit)
        self._proc.finished.connect(self._proc.deleteLater)
        self._proc_thread.finished.connect(self._proc_thread.deleteLater)
        self._proc_thread.finished.connect(self.on_proc_thread_finished)
        self._proc_thread.destroyed.connect(self._deinit_proc_thread)
        # --- Final preparations
        self.on_proc_starting()
        # Start the thread
        self._proc_thread.start()

    def on_progress_updated(self, data):
        '''Display video processing progress.'''
        frame_time, progress = data
        # Update the progress bar
        self.proc_progress_bar.setValue(progress)
        # TODO: It would be nice to use `frame_time` to update the current video time in the status bar!
        # For best results, the progress update signal would need
        # to fire when frame_time is near a whole second of the input.
        log.debug(f'video time: {timedelta(seconds=frame_time)}, progress: {progress:3d}%')

    def on_stop_proc(self):
        '''Request to stop the video processor.'''
        log.debug('User says: CANCEL PROCESSING')
        # Clean up the UI
        self.instruct_lbl.setText('')
        self.instruct_lbl.hide()
        # Politely ask the processing loop to stop.
        self.stop_processor.emit()

    def _enable_figure_controls(self, enable: bool) -> None:
        '''Enables/disables all the widgets that control drawn figure state.'''
        for fig_chk in self.fig_chk_boxes:
            fig_chk.setEnabled(enable)
        self.chk_diag.setEnabled(enable)

    def on_ar_geometry_available(self, is_available: bool):
        '''Update UI controls based on availability of AR geometry.'''
        self._enable_figure_controls(is_available)
        msg_bit = 'is' if is_available else 'is not'
        self._output_msg(f'AR geometry {msg_bit} available.')

    def on_clear_track(self):
        '''Clear the aircraft's existing flight track.'''
        log.debug('User says: CLEAR TRACK')
        self.clear_track.emit()

    def on_track_cleared(self):
        '''Slot that responds to the processor's signal.'''
        self._output_msg('Track cleared.')

    def on_relocate_cam(self):
        '''Relocate the camera.'''
        log.debug('User says: RELOCATE CAM')
        self.relocate_cam.emit()

    def on_pause_resume(self):
        '''Pause/resume processing at the current frame.'''
        log.debug('User says: PAUSE / RESUME')
        self.pause_resume.emit()

    def on_paused_resumed(self, is_paused: bool):
        '''
        Slot that responds to VideoProcessor's signal.

            :param is_paused: True if paused, False if resumed.
        '''
        if is_paused:
            self._output_msg('Paused processing.')
        else:
            self._output_msg('Resumed processing.')

    def on_rotate_ccw(self):
        '''Rotate AR sphere CCW.'''
        log.debug('User says: rotate CCW')
        self.manipulate_sphere.emit(SphereManipulations.RotateCCW)

    def on_rotate_cw(self):
        '''Rotate AR sphere CW.'''
        log.debug('User says: rotate CW')
        self.manipulate_sphere.emit(SphereManipulations.RotateCW)

    def on_move_north(self):
        '''Move AR sphere North.'''
        log.debug('User says: move NORTH')
        self.manipulate_sphere.emit(SphereManipulations.MoveNorth)

    def on_move_south(self):
        '''Move AR sphere South.'''
        log.debug('User says: move SOUTH')
        self.manipulate_sphere.emit(SphereManipulations.MoveSouth)

    def on_move_west(self):
        '''Move AR sphere West.'''
        log.debug('User says: move WEST')
        self.manipulate_sphere.emit(SphereManipulations.MoveWest)

    def on_move_east(self):
        '''Move AR sphere East.'''
        log.debug('User says: move EAST')
        self.manipulate_sphere.emit(SphereManipulations.MoveEast)

    def on_move_reset(self):
        '''Reset AR sphere's center to world origin.'''
        log.debug('User says: RESET to center')
        self.manipulate_sphere.emit(SphereManipulations.ResetCenter)

    def on_figure_start(self):
        '''Mark the start of a figure in 3D.'''
        log.debug('User says: START FIGURE')
        self.figure_mark.emit(True)

    def on_figure_end(self):
        '''Mark the end of a figure in 3D. '''
        log.debug('User says: END FIGURE')
        self.figure_mark.emit(False)
