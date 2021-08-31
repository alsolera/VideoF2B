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
Module for handling application crashes.
'''

import faulthandler
import json
import locale
import os
import os.path
import platform
import tempfile
import traceback
import uuid
from typing import Any, cast

# pylint: disable=import-error
from PySide6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QGroupBox,
                               QLabel, QPushButton, QTextEdit, QVBoxLayout)

HOME_DIR = os.path.expanduser("~")

# TODO: complete this module using Cura's CrashHandler as a guide?
