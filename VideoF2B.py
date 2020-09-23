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
from datetime import timedelta

import cv2
import imutils
import numpy as np
from imutils.video import FPS

import Camera
import Detection
import Drawing
import figure_tracker as figtrack
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
LOG_PATH = 'sphere_calcs.log'
CALIBRATION_PATH = None  # Default: ask
VIDEO_PATH = None  # Default: ask
FLIGHT_RADIUS = None  # Default: ask
MARKER_RADIUS = None  # Default: ask
MARKER_HEIGHT = None  # Default: ask

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
if cap is None or videoPath is None:
    print('ERROR: no input specified.')
    sys.exit(1)
FULL_FRAME_SIZE = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

# Load camera calibration
cam = Camera.CalCamera(frame_size=FULL_FRAME_SIZE,
                       calibrationPath=CALIBRATION_PATH, logger=logger,
                       flight_radius=FLIGHT_RADIUS,
                       marker_radius=MARKER_RADIUS,
                       marker_height=MARKER_HEIGHT)
if not cam.Calibrated:
    master.withdraw()

# Determine input video size and the scale wrt detector's frame size
scale, inp_width, inp_height = Video.Size(cam, cap, IM_WIDTH)
logger.debug(f'scale = {scale}')

# Platform-dependent stuff
WINDOW_FLAGS = cv2.WINDOW_NORMAL
fourcc = cv2.VideoWriter_fourcc(*'H264')
if platform.system() is 'Windows':
    WINDOW_FLAGS = cv2.WINDOW_KEEPRATIO
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

input_base_name = os.path.splitext(videoPath)[0]
# Output video file
VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
if not live:
    out = cv2.VideoWriter(f'{input_base_name}_out.mp4',
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
# Number of total empty frames in the input.
num_empty_frames = 0
# Number of consecutive empty frames at beginning of capture
num_consecutive_empty_frames = 0
# Maximum allowed number of consecutive empty frames. If we find more, we quit.
MAX_CONSECUTIVE_EMPTY_FRAMES = 256

if cam.Calibrated:
    data_path = f'{input_base_name}_out_data.csv'
    data_writer = open(data_path, 'w', encoding='utf8')
    # data_writer.write('p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,root1,root2\n')

fig_tracker = None
if cam.Calibrated:
    fig_tracker = figtrack.FigureTracker(logger=logger, callback=sys.stdout.write)

# aids for drawing figure start/end points over track
is_fig_in_progress = False
fig_img_pts = []
# aids for drawing nominal paths
MAX_DRAW_FIT_FRAMES = int(VIDEO_FPS * 2.)
fit_img_pts = []
draw_fit = False
num_draw_fit_frames = 0
# misc
frame_idx = 0
frame_delta = 0
num_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
MAX_FRAME_DELTA = int(num_input_frames / 100)
cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
while True:
    _, frame_or = cap.read()

    if frame_or is None:
        num_empty_frames += 1
        num_consecutive_empty_frames += 1
        if num_consecutive_empty_frames > MAX_CONSECUTIVE_EMPTY_FRAMES:  # GoPro videos show empty frames, quick fix
            break
        continue
    num_consecutive_empty_frames = 0

    frame_idx += 1
    # if frame_idx < int(51*VIDEO_FPS):
    #     continue
    # if frame_idx > int(64*VIDEO_FPS):
    #     break

    if frame_or is None:
        num_empty_frames += 1
        if num_empty_frames > MAX_CONSECUTIVE_EMPTY_FRAMES:  # GoPro videos show empty frames, quick fix
            break
        continue
    num_empty_frames = 0

    if cam.Calibrated:
        master.update()

        frame_or = cam.Undistort(frame_or)
        if not cam.Located and cam.AR:
            cam.Locate(frame_or)

        '''
        # This section maps the entire image space to world, effectively meshing the sphere
        # to enable the approximation of real-world error at a given pixel.
        if frame_idx == 1:
            import time
            # first dimension: near/far point index
            # second dimension: image width
            # third dimension: image height
            # fourth dimension: XYZ point on sphere
            world_map = np.zeros((2, inp_width, inp_height, 3), dtype=np.float32)
            world_map[:, :, :, :] = np.nan
            num_pts_collected = 0
            for v in range(inp_height):  # NOTE: top of sphere starts at row index 48
                t1 = time.process_time()
                for u in range(inp_width):
                    world_pts = projection.projectImagePointToSphere(cam, (u, v), frame_or, data_writer)
                    if world_pts is not None:
                        # print(u)
                        # print(world_pts)
                        world_map[0, u, v] = world_pts[0]
                        world_map[1, u, v] = world_pts[1]
                        num_pts_collected += 1
                t2 = time.process_time()
                t_diff = t2 - t1
                print(f'Number of world points in row index {v}: {num_pts_collected}, collected in {t_diff} s')
                num_pts_collected = 0
            print('saving world_map array to file...')
            np.save('world_map.npy', world_map)
            print('...done.')
        # '''

    frame = imutils.resize(frame_or, width=IM_WIDTH)

    detector.process(frame)
    Drawing.draw_track(frame_or, detector.pts_scaled, MAX_TRACK_LEN)

    if cam.Located:
        Drawing.draw_all_geometry(frame_or, cam, azimuth_delta, axis=False)

        distZero = np.zeros_like(cam.dist)
        if loops_chk.get():
            Drawing.draw_loop(frame_or, azimuth_delta, cam.rvec, cam.tvec,
                              cam.newcameramtx, distZero, cam.flightRadius, color=(255, 255, 255))
        if ver_eight_chk.get():
            Drawing.drawVerEight(frame_or, azimuth_delta, cam.rvec, cam.tvec,
                                 cam.newcameramtx, distZero, cam.flightRadius, color=(255, 255, 255))
        if hor_eight_chk.get():
            Drawing.drawHorEight(frame_or, azimuth_delta, cam.rvec, cam.tvec,
                                 cam.newcameramtx, distZero, cam.flightRadius, color=(255, 255, 255))
        if over_eight_chk.get():
            Drawing.drawOverheadEight(frame_or, azimuth_delta, cam.rvec, cam.tvec,
                                      cam.newcameramtx, distZero, cam.flightRadius, color=(255, 255, 255))

        # try to track the aircraft in world coordinates
        if detector.pts_scaled[0] is not None:
            if is_fig_in_progress:
                fig_img_pts.append(detector.pts_scaled[0])
            act_pts = projection.projectImagePointToSphere(
                cam, detector.pts_scaled[0], frame_or, data_writer)
            if act_pts is not None:  # and act_pts.shape[0] == 2:
                # fig_tracker.add_actual_point(act_pts)
                # Typically the first point is the "far" point on the sphere...Good enough for most figures that are on the far side of the camera.
                # TODO: Make this smarter so that we track the correct path point at all times.
                fig_tracker.add_actual_point(act_pts[0])
                # fig_tracker.add_actual_point(act_pts[1])

        # ========== Draw the fitted figures ==========
        if draw_fit and num_draw_fit_frames < MAX_DRAW_FIT_FRAMES:
            # Draw the nominal paths: initial and best-fit
            for i, f_pts in enumerate(fit_img_pts):
                # Draw the initial guess (the defined nominal path) in GREEN
                # Draw the optimized fit path in CYAN
                f_color = (0, 255, 0) if i == 0 else (255, 255, 0)
                for j in range(1, len(f_pts)):
                    cv2.line(frame_or, f_pts[j], f_pts[j-1], f_color, 2)
                    # Draw the error whiskers. TODO: verify the arrays are correct after figure.u is trimmed.
                    if i == 1:
                        cv2.line(frame_or, fig_img_pts[j], f_pts[j], (255, 255, 255), 1)
            num_draw_fit_frames += 1
        if num_draw_fit_frames == MAX_DRAW_FIT_FRAMES:
            draw_fit = False
            fit_img_pts = []
            fig_img_pts = []
            num_draw_fit_frames = 0

        # Draw the start/end points of the figure's actual tracked path (as marked by user)
        if fig_img_pts:
            cv2.circle(frame_or, fig_img_pts[0], 6, (0, 255, 255), -1)
            if not is_fig_in_progress:
                cv2.circle(frame_or, fig_img_pts[-1], 6, (255, 0, 255), -1)

    # Write text
    cv2.putText(frame_or, "VideoF2B - v0.6", (10, 15),
                cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 255), 1)

    # Show output
    cv2.imshow(WINDOW_NAME, frame_or)

    # Save frame
    out.write(frame_or)

    frame_time = frame_idx / VIDEO_FPS
    progress = int(frame_idx / (num_input_frames) * 100)
    if (frame_idx == 1) or (progress % 5 == 0 and frame_delta >= MAX_FRAME_DELTA) or (frame_time % 1 < 0.05):
        print(
            f'time: {timedelta(seconds=frame_time)}, '
            f'progress: {progress:3d}% ({frame_idx:6d}/{num_input_frames:6d} frames), '
            f'empty frames = {num_empty_frames}',
            end='\r')
        frame_delta = 0
    frame_delta += 1

    key = cv2.waitKeyEx(1)  # & 0xFF
    # if key != -1:
    #     print(key, key & 0xff)
    if key % 256 == 27 or key == 1048603 or cv2.getWindowProperty(WINDOW_NAME, 1) < 0:  # ESC
        # Exit
        break
    elif key % 256 == 32 or key == 1048608:  # Space
        # Clear the current track
        detector.clear()
    elif key == 1113939 or key == 2555904 or key == 65363:  # Right Arrow
        azimuth_delta += 0.5
    elif key == 1113937 or key == 2424832 or key == 65361:  # Left Arrow
        azimuth_delta -= 0.5
    elif key % 256 == 99 or key == 1048675:  # c
        # Relocate AR sphere
        cam.Located = False
        cam.AR = True
    elif key % 256 == 91:  # [
        # Mark the beginning of a new figure
        if fig_tracker is not None:
            detector.clear()
            fig_tracker.start_figure()
            is_fig_in_progress = True
    elif key % 256 == 93:  # ]
        # Mark the end of the current figure
        if fig_tracker is not None:
            fig_tracker.finish_figure()
            is_fig_in_progress = False
            # Uniform t along the figure path
            # t = np.linspace(0., 1., fig_tracker._curr_figure_fitter.num_nominal_pts)
            # The figure's chosen t distribution (initial, final)
            t = (fig_tracker._curr_figure_fitter.diag.u0,
                 fig_tracker._curr_figure_fitter.u)
            # Trim our detected points according to the fit
            trims = fig_tracker._curr_figure_fitter.diag.trim_indexes
            fig_img_pts = list(tuple(pair) for pair in np.array(fig_img_pts)[trims].tolist())
            # print(fig_img_pts)
            # The last item is the tuple of (initial, final) fit params
            for i, fp in enumerate(fig_tracker.figure_params[-1]):
                nom_pts = projection.projectSpherePointsToImage(
                    cam, fig_tracker._curr_figure_fitter.get_nom_point(*fp, *t[i]))
                fit_img_pts.append(tuple(map(tuple, nom_pts.tolist())))
            # print(f't ({len(t)} pts)')
            print(f'fit_img_pts[initial] ({len(fit_img_pts[0])} pts)')
            # # print(fit_img_pts[0])
            print(f'fit_img_pts[final] ({len(fit_img_pts[1])} pts)')
            # # print(fit_img_pts[1])
            print(f'fig_img_pts ({len(fig_img_pts)} pts)')
            # # print()
            # Set the flag for drawing the fit figures, diags, etc.
            draw_fit = True

    fps.update()

fps.stop()
print()
print(f'frame_idx={frame_idx}, num_input_frames={num_input_frames}, num_empty_frames={num_empty_frames}, progress={progress}%')
print(f"Elapsed time: {fps.elapsed():.1f}")
print(f"Approx. FPS: {fps.fps():.1f}")

if fig_tracker is not None:
    fig_tracker.finish_all()
    fig_tracker.export(f'{input_base_name}_out_figures.npz')

# Clean
cap.release()
out.release()
cv2.destroyAllWindows()
if cam.Located:
    data_writer.close()
