# VideoF2B v0.4 - Draw F2B figures from video
# Copyright (C) 2018  Alberto Solera Rico - albertoavion(a)gmail.com
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


import cv2
import tkinter
import tkinter.filedialog
import sys
from os import path


def LoadVideo():
    root = tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    live = False
    
    try:
        CF = open('vid.conf', 'r')
        initialdir = CF.read()
        print(initialdir)
        CF.close()
    except:
        initialdir = '../'
    
    Path = tkinter.filedialog.askopenfilename(parent=root, initialdir=initialdir,
                                        title='Select video file')
    
    if Path == (): #If file is not provided, ask for URL
        Path = tkinter.simpledialog.askstring('Input', 'Input URL')
        if Path != ():
            live = True
    else:
        CF = open('vid.conf', 'w')
        CF.write(path.dirname(Path)) 
        CF.close()
        
    print(Path)
    assert Path != ()
    
    cap = cv2.VideoCapture(Path)
    if not cap.isOpened():  # check if we succeeded
        print('Error loading video file')
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
        
    del root
    return cap, Path, live

def Size(Camera, cap, ImWidth):
    if Camera.Calibrated:
        inp_width = Camera.roi[2]
        inp_height = Camera.roi[3]
    else:
        inp_width = cap.get(3)
        inp_height = cap.get(4)

    scale = float(inp_width)/float(ImWidth)

    print("FPS: ", cap.get(5))
    print("Res: ", inp_height, 'x', inp_width)

    return scale, inp_width, inp_height
