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
Interactive tool for placing a camera in the field.
'''

import logging
from typing import Any, Union

from PySide6 import QtCore, QtGui, QtWidgets
from videof2b.core.camera_director import CamDirector
from videof2b.core.common import get_bundle_dir
from videof2b.ui.widgets import QHLine

log = logging.getLogger(__name__)


class CamDirectorDialog(QtWidgets.QDialog):
    '''
    Interactive camera placement aid.
    '''

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setup_model()
        self.setup_ui()
        # pylint: disable=no-member
        # Signals
        self.inputs_model.dataChanged.connect(self.on_new_solution)

    def setup_model(self):
        '''Create the models.'''
        self.domain_obj = CamDirector()
        self.inputs_model = CamDirectorInputsModel(self.domain_obj)
        self.results_model = CamDirectorResultsModel(self.domain_obj)

    def setup_ui(self):
        '''Create the UI.'''
        self.setWindowTitle('Camera placement calculator')
        # Components
        self.inputs_label = QtWidgets.QLabel('<strong>Inputs</strong>', self)
        self.inputs_view = QtWidgets.QTableView(self)
        self.inputs_view.setModel(self.inputs_model)
        self._size_table(self.inputs_view)
        url = 'https://www.scantips.com/lights/fieldofview.html#top'
        self.fov_link_label = QtWidgets.QLabel(
            f'Calculate your camera\'s FOV <a href="{url}">here</a>.', self)
        self.fov_link_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        self.fov_link_label.setOpenExternalLinks(True)
        self.results_label = QtWidgets.QLabel('<strong>Results</strong>', self)
        self.results_view = QtWidgets.QTableView(self)
        self.results_view.setModel(self.results_model)
        self._size_table(self.results_view)
        self.diag_widget = QtWidgets.QWidget(self)
        self.diag_layout = QtWidgets.QVBoxLayout(self.diag_widget)
        self.diagram = QtWidgets.QLabel()
        bundle_path = get_bundle_dir()
        diagram_img_path = (bundle_path / 'resources' / 'art' / 'cam-director-side-view.png').absolute()
        self.diagram.setPixmap(QtGui.QPixmap(str(diagram_img_path)))
        self.diag_layout.addWidget(self.diagram)
        self.diag_layout.addStretch(1)
        # Layout
        self.data_widget = QtWidgets.QWidget(self)
        self.data_layout = QtWidgets.QVBoxLayout(self.data_widget)
        self.data_layout.addWidget(self.inputs_label)
        self.data_layout.addWidget(self.inputs_view, alignment=QtCore.Qt.AlignHCenter)
        self.data_layout.addWidget(self.fov_link_label, alignment=QtCore.Qt.AlignHCenter)
        self.data_layout.addWidget(QHLine())
        self.data_layout.addWidget(self.results_label)
        self.data_layout.addWidget(self.results_view, alignment=QtCore.Qt.AlignHCenter)
        self.data_layout.addStretch(1)
        self.data_widget.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
                                       QtWidgets.QSizePolicy.MinimumExpanding)
        # Main layout
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.addWidget(self.data_widget, 1)
        self.main_layout.addWidget(self.diag_widget, 1)
        self.setLayout(self.main_layout)

    def _size_table(self, table: QtWidgets.QTableView):
        table.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table.resizeColumnsToContents()
        table.setFixedSize(
            table.horizontalHeader().length() + table.verticalHeader().width(),
            table.verticalHeader().length() + table.horizontalHeader().height()
        )

    def on_new_solution(self, _):
        '''Update results view when a new solution is available.'''
        self._size_table(self.inputs_view)
        self._size_table(self.results_view)
        self.results_view.reset()


class CamDirectorInputsModel(QtCore.QAbstractListModel):
    '''Model that represents the cam director's inputs.'''

    _row_names = (
        'Flight radius [R]',
        'Camera height [C]',
        'Ground level [G]',
        'Camera FOV angle [A]',
    )
    _col_names = (
        'Value',
    )
    _biz_props = (
        'R',
        'C',
        'G',
        'A',
    )
    _row_formats = (
        '%.1f',
        '%+.2f',
        '%+.2f',
        '%.2f',
    )
    _tooltips = (
        'Radius of the flight hemisphere.',

        'Height of the camera relative to the flight base.\n'
        'Above base is positive, below is negative.',

        'Height of ground level at pilot\'s feet relative to the flight base.\n'
        'Above base is positive, below is negative.',

        'Maximum vertical angle of the camera system\'s Field Of View, in degrees.\n'
        'Some typical examples:\n'
        '* A 10mm lens on APS-C 1.5x crop sensor with 16:9 crop in a 3:2 camera\n'
        'results in 68.04° vertical angle of view.\n'
        '* A 14mm lens on full-frame sensor with 16:9 crop in a 3:2 camera\n'
        'results in 71.75° vertical angle of view.',
    )

    def __init__(self, biz_obj: CamDirector, parent=None) -> None:
        super().__init__(parent)
        self.biz_obj = biz_obj

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = ...) -> Any:
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Vertical:
                return self._row_names[section]
            if orientation == QtCore.Qt.Horizontal:
                return self._col_names[section]
        return None

    def rowCount(self, parent: Union[QtCore.QModelIndex, QtCore.QPersistentModelIndex] = ...) -> int:
        return len(self._row_names)

    def data(self, index: Union[QtCore.QModelIndex, QtCore.QPersistentModelIndex], role: int = ...) -> Any:
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            data = self.biz_obj.__getattribute__(self._biz_props[row])
            return self._row_formats[row] % data
        if role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignRight + QtCore.Qt.AlignVCenter
        if role == QtCore.Qt.ToolTipRole:
            return self._tooltips[index.row()]
        return None

    def flags(self, index: Union[QtCore.QModelIndex, QtCore.QPersistentModelIndex]) -> QtCore.Qt.ItemFlags:
        return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable

    def setData(self,
                index: Union[QtCore.QModelIndex, QtCore.QPersistentModelIndex],
                value: Any, role: int = ...) -> bool:
        if role == QtCore.Qt.EditRole:
            if value == '':
                return False
            try:
                log.debug('attr=\'%s\' value=%s', self._biz_props[index.row()], value)
                self.biz_obj.__setattr__(self._biz_props[index.row()], float(value))
                # pylint: disable=no-member
                self.dataChanged.emit(index, index)
                return True
            except Exception as exc:
                log.critical('Failed to set data on %s: %s', index.row(), exc)
        return False


class CamDirectorResultsModel(QtCore.QAbstractTableModel):
    '''Model that represents the cam director's results.'''

    _row_names = (
        'Camera distance',
        'View angle',
        'Tangent elevation',
    )
    _col_names = (
        'Nearest',
        'Farthest',
    )
    _biz_props = (
        'cam_distance_limits',
        'cam_view_limits',
        'cam_tangent_elev_limits'
    )
    _row_formats = (
        '%.3f',
        '%.2f',
        '%.2f',
    )
    _col_tooltips = (
        'Values when camera is located safely nearest to the flight circle\n'
        'such that the top of the visible edge of the sphere and the pilot\'s\n'
        'feet still fit vertically within the video frame.',

        'Values when camera is located at the farthest practical distance\n'
        'from the flight circle such that the 45° latitude touches the\n'
        'top of the visible edge of the sphere in the video frame.',
    )
    _tooltips = (
        'Horizontal distance from flight center to camera.',

        'Vertical FOV angle in the resulting video from\n'
        'the pilot\'s feet to the top of the visible edge\n'
        'of the flight sphere, in degrees.',

        'Elevation of the tangent point where\n'
        'a straight line from the camera touches\n'
        'the visible edge of the sphere.\n'
        'Elevation is relative to flight center, in degrees.',
    )

    def __init__(self, biz_obj: CamDirector, parent=None) -> None:
        super().__init__(parent)
        self.biz_obj = biz_obj

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = ...) -> Any:
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Vertical:
                return self._row_names[section]
            if orientation == QtCore.Qt.Horizontal:
                return self._col_names[section]
        if role == QtCore.Qt.ToolTipRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._col_tooltips[section]
        return None

    def rowCount(self, parent: Union[QtCore.QModelIndex, QtCore.QPersistentModelIndex] = ...) -> int:
        return len(self._row_names)

    def columnCount(self, parent: Union[QtCore.QModelIndex, QtCore.QPersistentModelIndex] = ...) -> int:
        return len(self._col_names)

    def data(self, index: Union[QtCore.QModelIndex, QtCore.QPersistentModelIndex], role: int = ...) -> Any:
        row = index.row()
        data = self.biz_obj.__getattribute__(self._biz_props[row])[index.column()]
        if role == QtCore.Qt.DisplayRole:
            return self._row_formats[row] % data
        if role == QtCore.Qt.TextAlignmentRole:
            return QtCore.Qt.AlignRight + QtCore.Qt.AlignVCenter
        if role == QtCore.Qt.BackgroundRole:
            if (row == 1 and (data - self.biz_obj.A) > 1e-6) or (row == 2 and (data - 45.0) > 1e-6):
                return QtGui.QColor('pink')
        if role == QtCore.Qt.ToolTipRole:
            tooltip_text = [self._tooltips[index.row()]]
            if row == 1 and (data - self.biz_obj.A) > 1e-6:
                tooltip_text.append('*** WARNING: View angle is larger than camera\'s FOV angle. ***')
            if row == 2 and (data - 45.0) > 1e-6:
                tooltip_text.append('*** WARNING: Tangent elevation is higher than 45 degrees. ***')
            return '\n'.join(tooltip_text)
        return None

    def flags(self, index: Union[QtCore.QModelIndex, QtCore.QPersistentModelIndex]) -> QtCore.Qt.ItemFlags:
        return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
