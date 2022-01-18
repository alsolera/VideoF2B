# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2020 - 2022  Andrey Vasilik - basil96
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

'''Common definitions and constants for VideoF2B.'''

import enum
import os
import platform
import subprocess
import sys
from importlib.metadata import metadata
from math import cos, pi
from pathlib import Path
from typing import Tuple

import numpy as np
import platformdirs
import videof2b

# Some default lengths, in meters
DEFAULT_FLIGHT_RADIUS = 21.0
DEFAULT_MARKER_RADIUS = 25.0
DEFAULT_MARKER_HEIGHT = 1.5
# Default radius in all figure turns, in meters
DEFAULT_TURN_RADIUS = 1.5
# Default sphere center, in meters
DEFAULT_CENTER = np.float32([0., 0., 0.])
# Frequently used chunks of pi
HALF_PI = 0.5 * pi
QUART_PI = 0.25 * pi
EIGHTH_PI = 0.125 * pi
TWO_PI = 2.0 * pi
# Trig functions of common angles
COS_45 = cos(QUART_PI)


@enum.unique
class FigureTypes(enum.Enum):
    '''All F2B figures in sequence.'''
    TAKEOFF = 1
    REVERSE_WINGOVER = 2
    INSIDE_LOOPS = 3
    INVERTED_FLIGHT = 4
    OUTSIDE_LOOPS = 5
    INSIDE_SQUARE_LOOPS = 6
    OUTSIDE_SQUARE_LOOPS = 7
    INSIDE_TRIANGULAR_LOOPS = 8
    HORIZONTAL_EIGHTS = 9
    HORIZONTAL_SQUARE_EIGHTS = 10
    VERTICAL_EIGHTS = 11
    HOURGLASS = 12
    OVERHEAD_EIGHTS = 13
    FOUR_LEAF_CLOVER = 14
    LANDING = 15
    FOUR_LEAF_CLOVER_ALT = 16


@enum.unique
class SphereManipulations(enum.Enum):
    '''Possible manipulations of the AR sphere during processing.'''
    ResetCenter = 0
    RotateCCW = 1
    RotateCW = 2
    MoveWest = 3
    MoveEast = 4
    MoveNorth = 5
    MoveSouth = 6


# Common instance of PlatformDir that helps us with various platform-specific paths
PD = platformdirs.PlatformDirs('VideoF2B', roaming=True)


def get_frozen_path(path_when_frozen, path_when_non_frozen):
    '''Return one of the given paths based on status of `sys.frozen`.'''
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return path_when_frozen
    return path_when_non_frozen


def get_bundle_dir():
    '''Return the path of the bundle directory.
    When frozen as a one-file app, this is the _MEI### dir in temp.
    When frozen as a one-dir app, this is that dir.
    When running as a script, this is the project's root dir.
    '''
    bundle_path = Path(videof2b.__file__).parent.parent
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_path = Path(sys._MEIPASS)
    return bundle_path


def get_app_metadata() -> Tuple:
    '''
    Get basic app information.
    Returns (name, version) as a tuple of strings.
    '''
    result = metadata('videof2b')
    return result['Name'], result['Version']


def get_lib_versions():
    '''
    Get User-friendly names and versions of libraries
    that we care about for bug reports. This is just a sub-list
    of `install_requires` items in our `setup.cfg`.
    '''
    libs = (
        'PySide6',
        'cv2',
        'Numpy',
        'scipy',
    )
    versions = {'Python': platform.python_version()}
    for name in libs:
        if name == 'cv2':
            pkg = __import__(name)
            ver = pkg.__version__
        else:
            ver = metadata(name)['Version']
        versions[name] = ver
    return versions


def is_win() -> bool:
    '''
    Returns True if running on a Windows OS.

    :return: True if running on a Windows OS.
    '''
    return platform.system() == 'Windows'


def is_linux() -> bool:
    '''
    Returns True if running on a Linux OS.

    :return: True if running on a Linux OS.
    '''
    return platform.system() == 'Linux'


def launch_document(path) -> None:
    '''
    Open the specified document using the default application.
    '''
    if is_win():
        os.startfile(str(path))
    elif is_linux():
        subprocess.run(['xdg-open', str(path)])
