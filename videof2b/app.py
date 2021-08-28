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

'''Main VideoF2B application.'''

import argparse
import logging
import logging.handlers
import sys
from datetime import datetime
from traceback import format_exception

from PySide6 import QtCore, QtWidgets

from videof2b.core.common import PD
from videof2b.core.common.settings import Settings
from videof2b.core.common.store import Store
from videof2b.ui.exception_form import ExceptionDialog
from videof2b.ui.main_window import MainWindow

__all__ = ['VideoF2B', 'start']

log = logging.getLogger()


class VideoF2B(QtCore.QObject):

    def run(self, app):
        app.aboutToQuit.connect(self._quitting)
        self.main_window = MainWindow()
        self.main_window.show()
        return app.exec()

    def _quitting(self):
        '''Exit hook.'''
        log.info('Exiting application.')

    def hook_exception(self, exc_type, value, traceback):
        """
        Add an exception hook so that any uncaught exceptions are displayed in this window rather than somewhere where
        users cannot see it and cannot report when we encounter these problems.

        :param exc_type: The class of exception.
        :param value: The actual exception object.
        :param traceback: A traceback object with the details of where the exception occurred.
        """
        # We can't log.exception here because the last exception no longer exists, we're actually busy handling it.
        exc_msg = ''.join(format_exception(exc_type, value, traceback))
        log.critical(exc_msg)
        # TODO: temporarily here until ExceptionDialog is ready.
        raise Exception(exc_msg)

        # TODO: enable the exception form here once its design is ready
        # if not hasattr(self, 'exception_form'):
        #     self.exception_form = ExceptionDialog()
        # self.exception_form.exception_text_edit.setPlainText(
        #     ''.join(format_exception(exc_type, value, traceback)))
        # self.set_normal_cursor()
        # self.exception_form.exec()

    # @staticmethod
    # def process_events():
    #     '''Wrapper to make ProcessEvents visible and named correctly.'''
    #     QtWidgets.QApplication.processEvents()

    # @staticmethod
    # def set_busy_cursor():
    #     '''Sets the Busy Cursor for the Application.'''
    #     QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.BusyCursor)
    #     QtWidgets.QApplication.processEvents()

    # @staticmethod
    # def set_normal_cursor():
    #     '''Sets the Normal Cursor for the Application.'''
    #     QtWidgets.QApplication.restoreOverrideCursor()
    #     QtWidgets.QApplication.processEvents()


def parse_options():
    '''Parse the command line arguments'''
    parser = argparse.ArgumentParser(prog='VideoF2B')
    parser.add_argument('-l', '--log-level', dest='loglevel', default='info', metavar='LEVEL',
                        help='Set logging to LEVEL level. Valid values are "debug", "info", "warning".')
    return parser.parse_args()


def set_up_logging(log_path, level=logging.DEBUG):
    '''Set up logging to the given path using the given level.
    '''
    # Get rid of the default StreamHandler.
    # See https://stackoverflow.com/questions/11820338/replace-default-handler-of-python-logger/11821510
    log.handlers.clear()
    # Create the log path on demand
    if not log_path.exists():
        log_path.mkdir(parents=True)
    file_path = log_path / 'videof2b.log'
    # Maximum of 5 rotating logs, up to 10MB each
    file_handler = logging.handlers.RotatingFileHandler(
        file_path,
        maxBytes=10485760,
        backupCount=5,
        encoding='utf8'
    )
    formatter = logging.Formatter(
        '%(asctime)s - '
        '%(name)-26s - '
        '%(threadName)-12s - '
        '%(levelname)-8s - '
        '%(message)s'
    )
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.setLevel(level)
    log.info('Logger started.')


def start():
    args = parse_options()
    if args and args.loglevel.lower() in ['d', 'debug']:
        level = logging.DEBUG
    elif args and args.loglevel.lower() in ['w', 'warning']:
        level = logging.WARNING
    else:
        level = logging.INFO
    set_up_logging(PD.user_data_path, level=level)
    log.debug('Initializing application')
    qt_app = QtWidgets.QApplication()
    qt_app.setOrganizationName('VideoF2B')
    qt_app.setApplicationName('VideoF2B')
    qt_app.setApplicationVersion('0.6 BETA')  # TODO: need a single point of access for app version
    # Create the shared store
    Store().create()
    # TODO: supposedly, QSettings uses these names from the QApplication object...how to use that mechanism?
    settings = Settings(qt_app.organizationName(), qt_app.applicationName())
    Store().add('settings', settings)
    Store().add('application', qt_app)
    # Create the application
    app = VideoF2B()
    # Register the sys-level exception hook
    sys.excepthook = app.hook_exception
    # Start the application
    log.info(f'Starting {qt_app.applicationName()} {qt_app.applicationVersion()}')
    sys.exit(app.run(qt_app))
