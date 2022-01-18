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

'''
Provides icons for the VideoF2B application.
'''

import logging

import qtawesome as qta
from PySide6.QtGui import QIcon, QPixmap
from videof2b.core.common import get_bundle_dir
from videof2b.core.common.singleton import Singleton

log = logging.getLogger(__name__)


class MyIcons(metaclass=Singleton):
    '''Provide application-wide icons.'''

    def __init__(self):
        '''Icons for use across the VideoF2B GUI.'''
        icons = {
            'advance': {'icon': 'ri.arrow-down-line'},
            'exception': {'icon': 'ph.smiley-x-eyes'},
            'browse': {'icon': 'ri.folder-open-line'},
            'pause': {'icon': 'ri.pause-circle-line'},
            'play': {'icon': 'ri.play-circle-line'},
            'save': {'icon': 'ri.save-3-line'},
        }
        self.videof2b_icon = QIcon()
        self._init_icons(icons)

    def _init_icons(self, icons):
        '''Adds icons as attributes to the singleton instance.'''
        # Try to stick with this API regardless of the underlying icon provider.
        bundle_path = get_bundle_dir()
        log.debug(f'bundle_path: {bundle_path.resolve().absolute()}')
        for k, v in icons.items():
            setattr(self, k, qta.icon(v['icon']))
        # Main app icon
        app_icon_path = (bundle_path / 'resources' / 'art' / 'videof2b.svg').absolute()
        app_icon_pixmap = QPixmap(str(app_icon_path))
        self.videof2b_icon.addPixmap(app_icon_pixmap, QIcon.Normal, QIcon.Off)
