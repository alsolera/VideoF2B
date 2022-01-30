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

'''The entry point for VideoF2B.'''

import atexit
import faulthandler
import logging

from videof2b.app import start
from videof2b.core.common import PD

log = logging.getLogger(__name__)
ERROR_LOG_FILE = None


def tear_down_fault_handling():
    '''On exit, release the file we were using for the faulthandler.'''
    global ERROR_LOG_FILE  # pylint: disable=global-statement,global-variable-not-assigned
    ERROR_LOG_FILE.close()


def set_up_fault_handling():
    '''Set up the Python fault handler.'''
    global ERROR_LOG_FILE  # pylint: disable=global-statement
    # Create the user data directory if it doesn't exist.
    # Set the fault handler to log to an error log file there.
    try:
        PD.user_data_path.mkdir(parents=True, exist_ok=True)
        ERROR_LOG_FILE = (PD.user_data_path / 'error.log').open('ab')
        atexit.register(tear_down_fault_handling)
        faulthandler.enable(ERROR_LOG_FILE)
    except OSError:
        log.exception('An exception occurred when enabling the fault handler')
        atexit.unregister(tear_down_fault_handling)
        if ERROR_LOG_FILE:
            ERROR_LOG_FILE.close()


def main():
    '''Start the app.'''
    set_up_fault_handling()
    # TODO add support for multiprocessing from frozen EXE (built using PyInstaller).
    # see https://docs.python.org/3/library/multiprocessing.html#multiprocessing.freeze_support
    # if is_win():
    #     multiprocessing.freeze_support()
    start()


if __name__ == '__main__':
    main()
