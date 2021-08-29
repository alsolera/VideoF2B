# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2020 - 2021  Andrey Vasilik - basil96@users.noreply.github.com
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
from math import cos, pi

import numpy as np
import platformdirs

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