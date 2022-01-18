#!/usr/bin/env python3
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

'''Package setup script just for legacy support of editable installs.'''

import distutils.command.build

import setuptools

# Custom command extension for building the main EXE.
# TODO: this is here temporarily until `setuptools` provides a more friendly API for it.
# See https://github.com/pypa/setuptools/issues/2591 for details.
distutils.command.build.build.sub_commands.append(('build_exe', None))

# Compatibility call for editable installs.
setuptools.setup()
