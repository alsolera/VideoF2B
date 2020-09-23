# VideoF2B v0.4 - Draw F2B figures from video
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


import cv2
import imutils
from imutils.video import FPS
import os
import platform
import Drawing
import Camera
import Video
import Detection
import numpy as np
import time

import tkinter
master = tkinter.Tk()
loops_chk = tkinter.BooleanVar()
chk1 = tkinter.Checkbutton(master, text="Loops", variable=loops_chk).grid(row=0, sticky ='w')
hor_eight_chk = tkinter.BooleanVar()
chk2 = tkinter.Checkbutton(master, text="Horizontal eight", variable=hor_eight_chk).grid(row=1, sticky ='w')
ver_eight_chk = tkinter.BooleanVar()
chk3 = tkinter.Checkbutton(master, text="Vertical eight", variable=ver_eight_chk).grid(row=2, sticky ='w')
over_eight_chk = tkinter.BooleanVar()
chk4 = tkinter.Checkbutton(master, text="Overhead eight", variable=over_eight_chk).grid(row=3, sticky ='w')

maxTime = 15 # seconds
ImWidth = 600
Window_name = 'VideoF2B v0.5 - A. Solera'

liveVideos = '../VideoF2B_videos'

# Video loading
cap, videoPath, live = Video.LoadVideo()
FULL_FRAME_SIZE = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

# Camera calibration loading
Camera = Camera.CalCamera(frame_size=FULL_FRAME_SIZE)
if not Camera.Calibrated:
    master.withdraw()

# Video size
scale, inp_width, inp_height = Video.Size(Camera, cap, ImWidth)

#What depends on running environment
if platform.system() == 'Windows':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    fourcc = cv2.VideoWriter_fourcc(*'H264')

cv2.namedWindow(Window_name, cv2.WINDOW_NORMAL )

# Output file
if not live:
    out = cv2.VideoWriter(os.path.splitext(videoPath)[0]+'_out.MP4',fourcc, cap.get(5),\
                         ( int(inp_width),int(inp_height) ))
else:
    if not os.path.exists(liveVideos):
        os.makedirs(liveVideos)
    timestr = time.strftime("%Y%m%d-%H%M")
    out = cv2.VideoWriter(liveVideos+'/out_'+timestr+'.MP4',fourcc, cap.get(5),\
                         ( int(inp_width),int(inp_height) ))

# Track lenght
maxlen = int(maxTime * cap.get(5))

# Detector
Detector = Detection.Detector(maxlen, scale)

# Speed meter
fps = FPS().start()

offsettAngle = 0
emptys = 0

while True:
    _, frame_or = cap.read()

    if frame_or is None:
        emptys += 1
        if emptys > 256:  # GoPro videos show empty frames, quick fix
            break
        continue
    emptys = 0

    if Camera.Calibrated:
        master.update()
        
        frame_or = Camera.Undistort(frame_or)
        if Camera.Located == False and Camera.AR == True:
            Camera.Locate(frame_or)

    frame = imutils.resize(frame_or, width=ImWidth)

    Detector.process(frame)
    if Camera.Located:
        Drawing.draw_all_geometry(frame_or, Camera, offsettAngle, axis=False)
        
        distZero = np.zeros_like(Camera.dist)
        if loops_chk.get():
            Drawing.draw_loop(frame_or, offsettAngle, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero, Camera.cableLenght, color=(255,255,255))
        if ver_eight_chk.get():
            Drawing.drawVerEight(frame_or, offsettAngle, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero, Camera.cableLenght, color=(255,255,255))
        if hor_eight_chk.get():
            Drawing.drawHorEight(frame_or, offsettAngle, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero, Camera.cableLenght, color=(255,255,255))
        if over_eight_chk.get():
            Drawing.drawOverheadEight(frame_or, offsettAngle, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero, Camera.cableLenght, color=(255,255,255))
        
    Drawing.draw_track(frame_or, Detector.pts_scaled, maxlen)

    # Write text
    cv2.putText(frame_or, "VideoF2B - v0.5", (10, 15),  cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 255), 1)

    # Show output
    cv2.imshow(Window_name, frame_or)

    # Save frame
    out.write(frame_or)

    key = cv2.waitKeyEx(1)  # & 0xFF
    if key == 27 or key == 1048603 or cv2.getWindowProperty(Window_name, 1) < 0:  # ESC
        break
    elif key == 32 or key == 1048608:  # Space
        Detector.clear()
    elif key == 1113939 or key == 2555904 or key == 65363:  # Arrow
        offsettAngle += 0.5
    elif key == 1113937 or key == 2424832 or key == 65361:  # Arrow
        offsettAngle -= 0.5
    elif key == 99 or key == 1048675:  # c
        Camera.Located = False
        Camera.AR = True

    fps.update()

fps.stop()
print("Elapsed time: {:.1f}".format(fps.elapsed()))
print("Approx. FPS: {:.1f}".format(fps.fps()))

# Clean
cap.release()
out.release()
cv2.destroyAllWindows()
