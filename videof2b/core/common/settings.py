# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021-2022  Andrey Vasilik - basil96
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

'''Module for handling persistent settings.'''

from pathlib import Path

from PySide6.QtCore import QByteArray, QPoint, QSettings


class Settings(QSettings):
    '''Simple wrapper around QSettings.
    Contains core definitions of all known keys and their default values.
    Does not contain a strategy for versioning of settings.
    Handles lookup of default values in the most basic manner.
    '''

    __defaults__ = {
        'core/enable_fisheye': False,
        'mru/video_dir': Path('..'),
        'mru/cal_dir': Path('..'),
        'mru/crashreport_dir': Path('..'),
        'ui/main_window_position': QPoint(0, 0),
        'ui/main_window_geometry': QByteArray(),
        'ui/main_window_state': QByteArray(),
    }

    def __init__(self, *args, **kwargs):
        '''Initialize settings.'''
        super().__init__(*args, **kwargs)
        # Store default settings if they don't already exist.
        for k, v in Settings.__defaults__.items():
            if not self.contains(k):
                self.setValue(k, v)

    def value(self, key):
        '''Return the value for the given key.
        The key must exist in `Settings.__defaults__`.
        If value not found, return the value from `Settings.__defaults__`
        '''
        default_value = Settings.__defaults__[key]
        try:
            setting = super().value(key, default_value)
        except TypeError:
            setting = default_value
        return self._convert_value(setting, default_value)

    def _convert_value(self, value, default_value):
        '''Convert the given `value` to the type of `default_value`.
        Expand this logic for missing types as needed.
        '''
        # --- Empty values return None. Handle them appropriately.
        if value is None:
            if isinstance(default_value, str):
                return ''
            if isinstance(default_value, list):
                return []
            if isinstance(default_value, dict):
                return {}
            if isinstance(default_value, bool):
                return False
            if isinstance(default_value, int):
                return 0
            if isinstance(default_value, float):
                return 0.
        # --- Handle specific types
        if isinstance(default_value, bool):
            if isinstance(value, bool):
                return value
            # In some environments bools are stored as strings.
            return value == 'true'
        if isinstance(default_value, int):
            return int(value)
        if isinstance(default_value, float):
            return float(value)
        return value
