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
Provides icons for the VideoF2B application.
'''

import logging

# import qtawesome as qta
from PySide6.QtGui import QIcon
from videof2b.core.common import get_bundle_dir
from videof2b.core.common.singleton import Singleton

log = logging.getLogger(__name__)


class MyIcons(metaclass=Singleton):
    '''Provide application-wide icons.'''

    def __init__(self):
        '''Icons for use across the VideoF2B GUI.'''
        icons = {
            # TODO: it would be easier to use qta, but its abstraction layer (QtPy) currently lacks support for PySide6.
            # Check https://github.com/spyder-ide/qtpy/issues/233
            # and https://github.com/spyder-ide/qtpy/pull/225 for updates.
            # 'browse': {'icon': 'fa5.folder-open'},
            # 'pause': {'icon': 'fa5.pause-circle'},  # and 'fa5s.pause-circle'
            # 'play': {'icon': 'fa5.play-circle'},  # and 'fa5s.play-circle'
            # # or 'fa5.arrow-alt-circle-down','fa5s.arrow-alt-circle-down','fa5s.arrow-circle-down' ?
            # 'advance': {'icon': 'fa5s.arrow-down'},

            # These should work until QtPy supports PySide6..
            'browse': {'icon': 'folder-open-line.svg'},
            'pause': {'icon': 'pause-circle-line.svg'},
            'play': {'icon': 'play-circle-line.svg'},
            'advance': {'icon': 'arrow-down-line.svg'},
        }
        self._init_icons(icons)

    def _init_icons(self, icons):
        '''Adds icons as attributes to the singleton instance.'''
        # Try to stick with this API regardless of the underlying icon provider.
        bundle_path = get_bundle_dir()
        log.debug(f'bundle_path: {bundle_path.resolve().absolute()}')
        for k, v in icons.items():
            res_path = (bundle_path / 'resources' / 'art' / v['icon']).absolute()
            setattr(self, k, QIcon(str(res_path)))
            #
            # TODO: this call is for when/if we can use qtawesome.
            # setattr(self, k, qta.icon(v['icon']))
