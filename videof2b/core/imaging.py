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
Imaging functions.
'''

from PySide6.QtGui import QImage
import numpy as np


def cv_img_to_qimg(cv_img: np.ndarray) -> QImage:
    '''Convert a cv2 image to a QImage for display in QPixmap objects.'''
    # This is an adaptation of this simple idea:
    # https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713
    #
    # One way to do it, maybe there are others:
    # https://stackoverflow.com/questions/57204782/show-an-opencv-image-with-pyqt5
    #
    # When cropping cv2 images, we end up with non-contiguous arrays.
    # First, `strides` is required:
    # https://stackoverflow.com/questions/52869400/how-to-show-image-to-pyqt-with-opencv/52869969#52869969
    # Second, it must be contiguous:
    # https://github.com/almarklein/pyelastix/issues/14
    #
    # These extra requirements manifest themselves when we process calibrated flights.
    # This can be verified by uncommenting the log messages around `np.ascontiguousarray()` call below,
    # but leave them commented for production!
    # Note that this extra step is not necessary for cv2 processing, only for converting to QImage for display.

    # TODO: profile this whole method for an idea of the performance hit involved here.
    # log.debug(f'Is `cv_img` C-contiguous before? {cv_img.flags["C_CONTIGUOUS"]}')
    cv_img = np.ascontiguousarray(cv_img)
    # log.debug(f'Is `cv_img` C-contiguous  after? {cv_img.flags["C_CONTIGUOUS"]}')
    image = QImage(
        cv_img.data,
        cv_img.shape[1],
        cv_img.shape[0],
        cv_img.strides[0],
        QImage.Format_RGB888
    ).rgbSwapped()
    return image
