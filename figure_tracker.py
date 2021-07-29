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

'''F2B Figure tracker.'''

import logging

import numpy as np

import figures
from common import FigureTypes

logger = logging.getLogger(__name__)


class UserError(Exception):
    '''Class for exception that occur during Figure tracking due to user errors.'''
    pass


class FigureTracker:
    '''Container that tracks F2B figures.
    May be used for fitting the actual flight path to the nominal figure to determine a score.

    All FAI figures, per "F2, Annex 4J -- F2B Manoeuvre Diagrams" for reference.
    Not all may be easy to track:
        * 4.J.1  Take-off (Rule 4.2.15.3)
        * 4.J.2  Reverse wingover (Rule 4.2.15.4)
        * 4.J.3  Three consecutive inside loops (Rule 4.2.15.5)
        * 4.J.4  Two consecutive laps of inverted level flight (Rule 4.2.15.6)
        * 4.J.5  Three consecutive outside loops (Rule 4.2.15.7)
        * 4.J.6  Two consecutive inside square loops (Rule 4.2.15.8)
        * 4.J.7  Two consecutive outside square loops (Rule 4.2.15.9)
        * 4.J.8  Two consecutive inside triangular loops (Rule 4.2.15.10)
        * 4.J.9  Two consecutive horizontal eight (Rule 4.2.15.11)
        * 4.J.10 Two consecutive horizontal square eight (Rule 4.2.15.12)
        * 4.J.11 Two consecutive vertical eight (Rule 4.2.15.13)
        * 4.J.12 Hourglass (Rule 4.2.15.14)
        * 4.J.13 Two consecutive overhead eight (Rule 4.2.15.15)
        * 4.J.14 Four-leaf clover manoeuvre (Rule 4.2.15.16)
        * 4.J.15 Landing manoeuvre (Rule 4.2.15.17)
    '''

    # Maps figure index to figure type
    FIGURE_MAP = {
        0: FigureTypes.INSIDE_LOOPS,
    }

    def __init__(self, callback=lambda x: None, **kwargs):
        '''Create a new FigureTracker.'''
        self.enable_diags = kwargs.pop('enable_diags', False)
        self.actuals = []  # TODO: convert to dict
        self.indexes = []  # TODO: convert to dict
        self.is_figure_in_progress = False
        self.figure_idx = 0
        self.figure_params = []
        self.R = None
        self._callback = callback
        self._curr_figure_fitter = None
        self._figure_actuals = None
        self._figure_indexes = None

    def start_figure(self):
        '''Start tracking a new figure.'''
        if self.is_figure_in_progress:
            self._callback(
                f"Cannot start new figure as figure {self.figure_idx + 1} is in progress.")
            return
        self.is_figure_in_progress = True
        self._figure_actuals = []
        self._figure_indexes = []
        logger.debug(f'Started tracking Figure {self.figure_idx}')

    def finish_figure(self):
        '''Finish trfig_type_constructorhe currently tracked figure.'''
        if not self.is_figure_in_progress:
            self._callback(f"Cannot finish a figure as no figure is in progress.")
            return
        self.is_figure_in_progress = False
        self.actuals.append(np.asarray(self._figure_actuals))
        self.indexes.append(self._figure_indexes)
        logger.debug(
            f'Finished tracking Figure {self.figure_idx} '
            f'({len(self._figure_actuals)} points)')
        # logger.debug(
        #     f'Figure {self.figure_idx} points:\n{self.actuals[self.figure_idx]} shape = {self.actuals[self.figure_idx].shape}')
        self._figure_actuals = None
        self._figure_indexes = None

        fig_type = FigureTracker.FIGURE_MAP.get(self.figure_idx)
        if fig_type is not None and self.R is not None:
            self._curr_figure_fitter = figures.Figure.create(
                fig_type, R=self.R, actuals=self.actuals[self.figure_idx], enable_diags=self.enable_diags)
            self.figure_params.append(
                # tuple of (initial, final) fit parameters
                (self._curr_figure_fitter.p0, self._curr_figure_fitter.fit())
            )

        self.figure_idx = len(self.actuals)

    def add_actual_point(self, idx, point):
        '''Add a measured (actual) point to the currently tracked figure at a given index.
        If no figure is currently being tracked, this call has no effect.'''
        if not self.is_figure_in_progress:
            # no effect when we're not actively tracking any figure
            return False
        logger.debug(f'add_actual_point: point = {point} {point.shape}')
        if self.R is None:
            self.R = np.linalg.norm(point)
        self._figure_actuals.append(point)
        self._figure_indexes.append(idx)
        return True

    def export(self, path):
        '''Export all tracked figures as numpy arrays to the specified file.
        Arrays are labeled "fig0", "fig1", etc.'''
        d = {
            f'fig{i}':
            np.asarray([np.hstack(x) for x in zip(self.indexes[i], act)]) for i, act in enumerate(self.actuals)
        }
        np.savez(path, **d)
        logger.debug(f'Exported {len(self.actuals)} figure(s) to "{path}".')

    def finish_all(self):
        '''Clean-up method. Finish current figure, if any figure is in progress.'''
        if self.is_figure_in_progress:
            self.finish_figure()
