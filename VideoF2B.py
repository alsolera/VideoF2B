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


import logging
import logging.handlers
import os
import platform
import sys
import time
import tkinter

import cv2
import imutils
import numpy as np
from imutils.video import FPS

import Camera
import Detection
import Drawing
import projection
import Video

master = tkinter.Tk()
loops_chk = tkinter.BooleanVar()
chk1 = tkinter.Checkbutton(master, text="Loops", variable=loops_chk).grid(row=0, sticky='w')
hor_eight_chk = tkinter.BooleanVar()
chk2 = tkinter.Checkbutton(master, text="Horizontal eight",
                           variable=hor_eight_chk).grid(row=1, sticky='w')
ver_eight_chk = tkinter.BooleanVar()
chk3 = tkinter.Checkbutton(master, text="Vertical eight",
                           variable=ver_eight_chk).grid(row=2, sticky='w')
over_eight_chk = tkinter.BooleanVar()
chk4 = tkinter.Checkbutton(master, text="Overhead eight",
                           variable=over_eight_chk).grid(row=3, sticky='w')

MAX_TRACK_TIME = 15  # seconds
IM_WIDTH = 960
WINDOW_NAME = 'VideoF2B v0.6 - A. Solera, A. Vasilik'
CALIBRATION_PATH = None
VIDEO_PATH = None
LOG_PATH = r'sphere_calcs.log'

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=10485760,
                                               backupCount=5, encoding='utf8')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)7s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Logger started.')

liveVideos = '../VideoF2B_videos'

# Load input video
cap, videoPath, live = Video.LoadVideo(path=VIDEO_PATH)
if cap is None:
    sys.exit(1)
FULL_FRAME_SIZE = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

# Load camera calibration
cam = Camera.CalCamera(frame_size=FULL_FRAME_SIZE, calibrationPath=CALIBRATION_PATH)
if not cam.Calibrated:
    master.withdraw()

# Determine input video size and the scale wrt detector's frame size
scale, inp_width, inp_height = Video.Size(cam, cap, IM_WIDTH)
logger.debug(f'scale = {scale}')

# Platform-dependent stuff
if platform.system() == 'Windows':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
else:
    fourcc = cv2.VideoWriter_fourcc(*'H264')
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Output video file
VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
if not live:
    out = cv2.VideoWriter(os.path.splitext(videoPath)[0]+'_out.mp4',
                          fourcc, VIDEO_FPS, (int(inp_width), int(inp_height)))
else:
    if not os.path.exists(liveVideos):
        os.makedirs(liveVideos)
    timestr = time.strftime("%Y%m%d-%H%M")
    out = cv2.VideoWriter(os.path.join(liveVideos, f'out_{timestr}.mp4'),
                          fourcc, VIDEO_FPS, (int(inp_width), int(inp_height)))

# Track length
MAX_TRACK_LEN = int(MAX_TRACK_TIME * VIDEO_FPS)
# Detector
detector = Detection.Detector(MAX_TRACK_LEN, scale)

# Speed meter
fps = FPS().start()

# Angle offset of current AR hemisphere wrt world coordinate system
azimuth_delta = 0
# Number of empty frames at beginning of capture
num_empty_frames = 0
MAX_EMPTY_FRAMES = 256

data_path = f'{os.path.splitext(videoPath)[0]}_out_data.csv'
data_writer = open(data_path, 'w', encoding='utf8')
# data_writer.write('p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,root1,root2\n')
frame_idx = 0
while True:
    _, frame_or = cap.read()

    # frame_idx += 1
    # if frame_idx < int(51*VIDEO_FPS):
    #     continue
    # if frame_idx > int(64*VIDEO_FPS):
    #     break

    if frame_or is None:
        num_empty_frames += 1
        if num_empty_frames > MAX_EMPTY_FRAMES:  # GoPro videos show empty frames, quick fix
            break
        continue
    num_empty_frames = 0

    if cam.Calibrated:
        master.update()

        frame_or = cam.Undistort(frame_or)
        if cam.Located == False and cam.AR == True:
            cam.Locate(frame_or)

    frame = imutils.resize(frame_or, width=IM_WIDTH)

    detector.process(frame)

    if cam.Located:
        Drawing.draw_all_geometry(frame_or, cam, azimuth_delta, axis=False)

        distZero = np.zeros_like(cam.dist)
        if loops_chk.get():
            Drawing.draw_loop(frame_or, azimuth_delta, cam.rvec, cam.tvec,
                              cam.newcameramtx, distZero, cam.cableLength, color=(255, 255, 255))
        if ver_eight_chk.get():
            Drawing.drawVerEight(frame_or, azimuth_delta, cam.rvec, cam.tvec,
                                 cam.newcameramtx, distZero, cam.cableLength, color=(255, 255, 255))
        if hor_eight_chk.get():
            Drawing.drawHorEight(frame_or, azimuth_delta, cam.rvec, cam.tvec,
                                 cam.newcameramtx, distZero, cam.cableLength, color=(255, 255, 255))
        if over_eight_chk.get():
            Drawing.drawOverheadEight(frame_or, azimuth_delta, cam.rvec, cam.tvec,
                                      cam.newcameramtx, distZero, cam.cableLength, color=(255, 255, 255))

        # try to track the aircraft in world coordinates
        if detector.pts_scaled[0] is not None:
            projection.projectImagePointToSphere(cam, detector.pts_scaled[0], frame_or, data_writer)

    Drawing.draw_track(frame_or, detector.pts_scaled, MAX_TRACK_LEN)

    # Write text
    cv2.putText(frame_or, "VideoF2B - v0.6", (10, 15),
                cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 255), 1)

    # Show output
    cv2.imshow(WINDOW_NAME, frame_or)

    # Save frame
    out.write(frame_or)

    key = cv2.waitKeyEx(1)  # & 0xFF
    # if key != -1:
    #     print(key, key & 0xff)
    if key % 256 == 27 or key == 1048603 or cv2.getWindowProperty(WINDOW_NAME, 1) < 0:  # ESC
        break
    elif key % 256 == 32 or key == 1048608:  # Space
        detector.clear()
    elif key == 1113939 or key == 2555904 or key == 65363:  # Arrow
        azimuth_delta += 0.5
    elif key == 1113937 or key == 2424832 or key == 65361:  # Arrow
        azimuth_delta -= 0.5
    elif key % 256 == 99 or key == 1048675:  # c
        cam.Located = False
        cam.AR = True

    fps.update()

fps.stop()
print(f"Elapsed time: {fps.elapsed():.1f}")
print(f"Approx. FPS: {fps.fps():.1f}")

# Clean
cap.release()
out.release()
cv2.destroyAllWindows()
data_writer.close()
