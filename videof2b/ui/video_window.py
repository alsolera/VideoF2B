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
The video window of VideoF2B application.
'''
import logging

from PySide6 import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)


class VideoWindow(QtWidgets.QLabel):
    '''The window that displays video frames during processing.'''

    # Our signals
    point_added = QtCore.Signal(tuple)
    point_removed = QtCore.Signal(tuple)

    def __init__(self, parent: QtWidgets.QWidget, **kwargs) -> None:
        '''
        Create a new empty video window.

        kwargs:
            `enable_mouse` : Whether to enable mouse reactions.
                Also available via the `MouseEnabled` property.
                Default is False.
        '''
        super().__init__(parent, **kwargs)
        # The current image, original size
        self._pixmap = QtGui.QPixmap()
        # The currently displayed image in the UI
        self._scaled_pix_map = None
        # Internal flag for the is_mouse_enabled property
        self._is_mouse_enabled = kwargs.pop('enable_mouse', False)

    def update_frame(self, frame) -> None:
        '''Set a new video frame in the window.'''
        # log.debug('Entering VideoWindow.update_frame()')
        # log.debug(f'  source={repr(frame)}')
        self._pixmap = QtGui.QPixmap(QtGui.QPixmap.fromImage(frame))
        # log.debug(f'  self._pixmap = {self._pixmap}')
        self._update_pixmap(self.size())
        # log.debug('Leaving  VideoWindow.update_frame()')

    @property
    def is_mouse_enabled(self):
        '''Indicates whether the video window reacts to mouse events.'''
        return self._is_mouse_enabled

    @is_mouse_enabled.setter
    def is_mouse_enabled(self, val: bool) -> None:
        '''Setter for `is_mouse_enabled`'''
        self._is_mouse_enabled = val

    def _update_pixmap(self, size: QtCore.QSize) -> None:
        '''Rescale our pixmap to the given size.
        Always create the scaled pixmap from the original provided pixmap.'''
        if self._pixmap.isNull():
            # A null QPixmap has zero width, zero height and no contents.
            # You cannot draw in a null pixmap.
            return
        self._scaled_pix_map = self._pixmap.scaled(
            size,
            aspectMode=QtCore.Qt.KeepAspectRatio,
            mode=QtCore.Qt.SmoothTransformation)
        # TODO: center horizontally
        self.setPixmap(self._scaled_pix_map)

    def clear(self) -> None:
        '''Overridden method to clear our custom pixmap.'''
        self._pixmap = QtGui.QPixmap()
        return super().clear()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # pylint:disable=invalid-name
        '''Overridden event so that our window resizes with its parent
        while maintaining the loaded image's original aspect ratio.'''
        self._update_pixmap(event.size())
        self.update()
        event.accept()
        # return super().resizeEvent(event) # possible stack overflow?

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # pylint:disable=invalid-name
        '''Overridden event so that we react to mouse clicks as needed.'''
        if not self._is_mouse_enabled:
            return None
        options = {
            QtCore.Qt.LeftButton: self.point_added,
            QtCore.Qt.RightButton: self.point_removed
        }
        button = event.button()
        signal = options.get(button)
        return self._point_reaction(event.pos(), signal)

    def _get_image_point(self, pos: QtCore.QPoint) -> QtCore.QPoint:
        '''Determine the image coordinate of a clicked point
        with respect to the originally loaded image. If the determined
        coordinate is outside the original image bounds, return None.'''
        scaled_w, scaled_h = self._scaled_pix_map.size().toTuple()
        orig_w, orig_h = self._pixmap.size().toTuple()
        w_ratio = orig_w / scaled_w
        h_ratio = orig_h / scaled_h
        displayed_pixmap = self.pixmap()
        local_offset_y = 0.5 * (self.height() - displayed_pixmap.height())
        img_x = round(w_ratio * pos.x())
        img_y = round(h_ratio * (pos.y() - local_offset_y))
        if (img_x < 0 or img_x >= self._pixmap.width() or
                img_y < 0 or img_y >= self._pixmap.height()):
            return None
        return QtCore.QPoint(img_x, img_y)

    def _point_reaction(self, pos: QtCore.QPoint, signal: QtCore.Signal) -> None:
        '''React to a clicked point by emitting the specified signal.'''
        if signal is None:
            return
        img_point = self._get_image_point(pos)
        if img_point is not None:
            signal.emit(img_point.toTuple())
