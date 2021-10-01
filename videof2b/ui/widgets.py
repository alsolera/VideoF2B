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

'''
This module contains various custom widgets for the UI.
'''

import enum
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from videof2b.core.common.path import path_to_str, replace_params, str_to_path
from videof2b.ui.icons import MyIcons


@enum.unique
class PathEditType(enum.Enum):
    '''Specifies the type of browser in a PathEdit.'''
    Files = 1
    Directories = 2


class FileDialog(QtWidgets.QFileDialog):
    '''A wrapped QFileDialog compatible with Path objects.'''

    @classmethod
    def getExistingDirectory(cls, *args, **kwargs):
        '''Thin wrapper of `getExistingDirectory` compatible with Path objects as args.

        :type parent: QtWidgets.QWidget | None
        :type caption: str
        :type directory: pathlib.Path
        :type options: QtWidgets.QFileDialog.Options
        :rtype: pathlib.Path
        '''
        args, kwargs = replace_params(args, kwargs, ((2, 'directory', path_to_str),))
        return_value = super().getExistingDirectory(*args, **kwargs)
        return str_to_path(return_value)

    @classmethod
    def getOpenFileName(cls, *args, **kwargs):
        '''Thin wrapper of `getOpenFileName` compatible with Path objects as args.

        :type parent: QtWidgets.QWidget | None
        :type caption: str
        :type directory: pathlib.Path
        :type filter: str
        :type initialFilter: str
        :type options: QtWidgets.QFileDialog.Options
        :rtype: tuple[pathlib.Path, str]
        '''
        args, kwargs = replace_params(args, kwargs, ((2, 'directory', path_to_str),))
        file_name, selected_filter = super().getOpenFileName(*args, **kwargs)
        return str_to_path(file_name), selected_filter

    @classmethod
    def getOpenFileNames(cls, *args, **kwargs):
        '''Thin wrapper of `getOpenFileNames` compatible with Path objects as args.

        :type parent: QtWidgets.QWidget | None
        :type caption: str
        :type directory: pathlib.Path
        :type filter: str
        :type initialFilter: str
        :type options: QtWidgets.QFileDialog.Options
        :rtype: tuple[list[pathlib.Path], str]
        '''
        args, kwargs = replace_params(args, kwargs, ((2, 'directory', path_to_str),))
        file_names, selected_filter = super().getOpenFileNames(*args, **kwargs)
        paths = [str_to_path(path) for path in file_names]
        return paths, selected_filter

    @classmethod
    def getSaveFileName(cls, *args, **kwargs):
        '''Thin wrapper of `getSaveFileName` compatible with Path objects as args.

        :type parent: QtWidgets.QWidget | None
        :type caption: str
        :type directory: pathlib.Path
        :type filter: str
        :type initialFilter: str
        :type options: QtWidgets.QFileDialog.Options
        :rtype: tuple[pathlib.Path | None, str]
        '''
        args, kwargs = replace_params(args, kwargs, ((2, 'directory', path_to_str),))
        file_name, selected_filter = super().getSaveFileName(*args, **kwargs)
        return str_to_path(file_name), selected_filter


class PathEdit(QtWidgets.QWidget):
    '''Custom QWidget subclass for selecting a file or directory.'''

    # Signals
    path_changed = QtCore.Signal(Path)

    def __init__(self, parent=None,
                 path_type=PathEditType.Files,
                 caption=None,
                 initial_path=None):
        '''Create the PathEdit widget.

        :param QtWidget.QWidget | None parent: The parent of this widget.
        :param PathEditType path_type: the type 
        :param str caption: Used to customise the caption in the QFileDialog.
        :rtype: None
        '''
        super().__init__(parent)
        self.caption = caption
        self._path_type = path_type
        self._path = None
        self._initial_path = initial_path
        self.filters = 'All files (*)'
        self._setup()

    def _setup(self):
        '''Set up the widget.
        :rtype: None
        '''
        widget_layout = QtWidgets.QHBoxLayout()
        widget_layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QtWidgets.QLineEdit(self)
        widget_layout.addWidget(self.line_edit)
        self.browse_button = QtWidgets.QToolButton(self)
        self.browse_button.setIcon(MyIcons().browse)
        widget_layout.addWidget(self.browse_button)
        self.setLayout(widget_layout)
        # Signals and Slots
        self.browse_button.clicked.connect(self.on_browse_button_clicked)
        self.line_edit.editingFinished.connect(self.on_line_edit_editing_finished)
        self.update_button_tool_tips()

    @QtCore.Property('QVariant')
    def path(self):
        '''Returns the selected path.

        :return: The selected path
        :rtype: Path
        '''
        return self._path

    @path.setter
    def path(self, path):
        '''Sets the selected path.

        :param Path path: The path to set the widget to
        :rtype: None
        '''
        self._path = path
        text = path_to_str(path)
        self.line_edit.setText(text)
        self.line_edit.setToolTip(text)

    @property
    def path_type(self):
        '''Returns the path type. Path type specifies selecting a file or directory.

        :return: The type selected
        :rtype: PathType
        '''
        return self._path_type

    @path_type.setter
    def path_type(self, path_type):
        '''Set the path type.

        :param PathType path_type: The type of path to select
        :rtype: None
        '''
        self._path_type = path_type
        self.update_button_tool_tips()

    def update_button_tool_tips(self):
        '''Updates the button tooltips during init and when `path_type` changes.

        :rtype: None
        '''
        if self._path_type == PathEditType.Directories:
            self.browse_button.setToolTip('Browse for directory')
        else:
            self.browse_button.setToolTip('Browse for file')

    def on_browse_button_clicked(self) -> None:
        '''Shows the QFileDialog when the browse button is clicked.
        Emits `path_changed` if appropriate.

        :rtype: None
        '''
        caption = self.caption
        path = None
        if self._path_type == PathEditType.Directories:
            if not caption:
                caption = 'Select Directory'
            path = FileDialog.getExistingDirectory(
                self, caption, self._path, FileDialog.ShowDirsOnly)
        elif self._path_type == PathEditType.Files:
            if not caption:
                caption = self.caption = 'Select File'
            path, _ = FileDialog.getOpenFileName(self, caption, self._initial_path, self.filters)
        if path:
            self.on_new_path(path)

    def on_line_edit_editing_finished(self):
        '''Updates `path` and emits `path_changed` when the line edit has finished being edited.

        :rtype: None
        '''
        path = Path(self.line_edit.text())
        self.on_new_path(path)

    def on_new_path(self, path):
        '''If the given path is different from current `path`,
        updates `path` and emits the `path_changed` Signal.

        :param Path path: The new path
        :rtype: None
        '''
        if self._path != path:
            self.path = path
            self.path_changed.emit(path)


class QHLine(QtWidgets.QFrame):
    '''A horizontal line widget.'''

    def __init__(self):
        '''Create a new horizontal line widget.'''
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class QVLine(QtWidgets.QFrame):
    '''A vertical line widget.'''

    def __init__(self):
        '''Create a new vertical line widget.'''
        super().__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
