# VideoF2B - Draw F2B figures from video
# Copyright (C) 2020  Andrey Vasilik - basil96@users.noreply.github.com
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

'''F2B Figure tracker.'''

import logging

import numpy as np


class UserError(Exception):
    '''Class for exception that occur during Figure tracking due to user errors.'''
    pass


class FigureTracker:
    '''Container that tracks F2B figures.
    May be used for fitting the actual flight path to the nominal figure to determine a score.'''

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.actuals = []
        self._figure_actuals = None
        self.is_figure_in_progress = False
        self.figure_idx = 0

    def start_figure(self):
        if self.is_figure_in_progress:
            raise UserError(
                f"Cannot start new figure as figure {self.figure_idx + 1} is in progress.")
        self.is_figure_in_progress = True
        self._figure_actuals = np.empty((3,), float)
        self.logger.debug(f'Started tracking Figure {self.figure_idx}')

    def finish_figure(self):
        if not self.is_figure_in_progress:
            raise UserError(f"Cannot finish a figure as no figure is in progress.")
        self.is_figure_in_progress = False
        self.actuals.append(self._figure_actuals)
        self._figure_actuals = None
        self.logger.debug(f'Finished tracking Figure {self.figure_idx}')
        self.logger.debug(
            f'Figure {self.figure_idx} points: {self.actuals[self.figure_idx]} shape = {self.actuals[self.figure_idx].shape}')
        self.figure_idx = len(self.actuals)

    def add_actual_point(self, point):
        if not self.is_figure_in_progress:
            # no effect when we're not actively tracking any figure
            return False
        self.logger.debug(f'add_actual_point: point = {point} {point.shape}')
        self._figure_actuals = np.append(self._figure_actuals, point, axis=0)
        return True
