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

import logging

from .singleton import Singleton

log = logging.getLogger(__name__)


class Store(metaclass=Singleton):
    '''An object store.  This is a singleton that provides access
    to object references that are shared within a process.'''

    log.debug('Store loading')

    @classmethod
    def create(cls):
        '''The constructor for the Store.'''
        log.debug('Store initializing')
        store = cls()
        store._items = {}
        return store

    def get(self, key):
        '''Get the specified object from the store.'''
        if key in self._items:
            return self._items[key]

    def add(self, key, item):
        '''Add an item to the store.'''
        if key not in self._items:
            self._items[key] = item

    def remove(self, key):
        '''Remove an item from the store.'''
        if key in self._items:
            self._items.pop(key)


class StoreProperties:
    '''Adds shared components to classes for use at run time.'''

    _application = None
    _settings = None

    @property
    def application(self):
        '''Dynamically-added application object.'''
        if not hasattr(self, '_application') or not self._application:
            self._application = Store().get('application')
        return self._application

    @property
    def settings(self):
        '''Dynamically-added settings object.'''
        if not hasattr(self, '_settings') or not self._settings:
            self._settings = Store().get('settings')
        return self._settings
