# VideoF2B - Draw F2B figures from video
# Copyright (C) 2018  Alberto Solera Rico - albertoavion(a)gmail.com
# Copyright (C) 2020  Andrey Vasilik - basil96@users.noreply.github.com
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

'''Module for handling video.'''

import os
import sys
import tkinter as Tkinter
from tkinter import filedialog as tkFileDialog

import cv2


def LoadVideo(path=None):
    root = Tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    live = False

    try:
        CF = open('vid.conf', 'r')
        initialdir = CF.read()
        print(initialdir)
        CF.close()
    except:
        initialdir = '../'

    if path is None:
        path = tkFileDialog.askopenfilename(parent=root, initialdir=initialdir,
                                            title='Select video file')
    if not path:  # If file is not provided, ask for URL
        path = Tkinter.simpledialog.askstring('Input', 'Input URL')
        if path:
            live = True
    else:
        CF = open('vid.conf', 'w')
        CF.write(os.path.dirname(path))
        CF.close()

    print(f'Video path: {path}')
    if not path:
        return None, None, live

    CF = open('vid.conf', 'w')
    CF.write(os.path.dirname(path))
    CF.close()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():  # check if we succeeded
        print('Error loading video file')
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    del root
    return cap, path, live


def Size(Camera, cap, im_width):
    if Camera.Calibrated:
        inp_width = Camera.roi[2]
        inp_height = Camera.roi[3]
    else:
        inp_width = cap.get(3)
        inp_height = cap.get(4)

    scale = float(inp_width)/float(im_width)

    print(f"FPS: {cap.get(5)}")
    print(f"Res: {inp_width} x {inp_height}")

    return scale, inp_width, inp_height
