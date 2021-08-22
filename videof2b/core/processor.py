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
The main video processor in VideoF2B.
'''

import logging
import platform
import sys
import time
# from datetime import timedelta
from pathlib import Path

import cv2
import imutils
import numpy as np
import videof2b.core.figure_tracker as figtrack
import videof2b.core.projection as projection
from imutils.video import FPS
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from videof2b.core import common
from videof2b.core.camera import CalCamera
# from videof2b.core.common import FigureTypes
from videof2b.core.common.store import StoreProperties
from videof2b.core.detection import Detector
from videof2b.core.drawing import Drawing
from videof2b.core.flight import Flight

log = logging.getLogger(__name__)


class ProcessorSettings:
    '''Stores persistable user settings.'''
    # TODO: implement all these in the shared Settings object.
    perform_3d_tracking = False
    max_track_time = 15  # seconds
    # width of detector frame
    im_width = 960
    sphere_xy_delta = 0.1  # XY offset delta in m
    sphere_rot_delta = 0.5  # Rotation offset delta in degrees
    live_videos = Path('../VideoF2B_videos')


class VideoProcessor(QThread, StoreProperties):
    '''Main video processor. Handles processing
    of a video input from start to finish.'''

    # Signals
    locator_points_changed = Signal(object)
    new_frame_available = Signal(QImage)
    progress_updated = Signal(tuple)

    def __init__(self) -> None:
        '''Create a new video processor.'''
        super().__init__()
        # A note on QThread naming: =======================================================
        # Sadly, this currently does not work. PySide6 names the threads `Dummy-#` instead.
        # According to PySide6 docs:
        # "Note that this is currently not available with release builds on Windows."
        # See https://doc.qt.io/qtforpython/PySide6/QtCore/QThread.html#managing-threads
        self.setObjectName('VideoProcessorThread')
        #
        self._full_frame_size = None
        self._fourcc = None
        self.flight = None

    def load_flight(self, flight: Flight) -> None:
        '''Load a Flight and prepare for processing.

        :param Flight flight: a fully instantiated Flight instance.
        '''
        self.flight = flight
        self.flight.locator_points_changed.connect(self.on_locator_points_changed)
        self._full_frame_size = (
            int(self.flight.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.flight.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        # TODO: move the setup code from run() to here. The code in run() should just contain the loop.

    def add_locator_point(self, point):
        '''Add a potential locator point.'''
        # Thin wrapper for the same method in Flight.
        if self.flight is not None:
            self.flight.add_locator_point(point)

    def pop_locator_point(self, _p):
        '''Remove the last locator point, if one exists.'''
        # Thin wrapper for the same method in Flight.
        if self.flight is not None:
            self.flight.pop_locator_point()

    def on_locator_points_changed(self, points):
        ''''''
        log.info(f'new locator points:\n{points}')
        self.locator_points_changed.emit(points)

    @staticmethod
    def _cv_img_to_qimg(cv_img: np.ndarray) -> QImage:
        '''Convert a cv2 image to a QImage for display in QPixmap objects.'''
        # One way to do it, maybe there are others:
        # https://stackoverflow.com/questions/57204782/show-an-opencv-image-with-pyqt5
        image = QImage(
            cv_img.data,
            cv_img.shape[1],
            cv_img.shape[0],
            QImage.Format_RGB888).rgbSwapped()
        return image

    def get_size(self, camera, cap, im_width):
        '''Calculate sizing information.'''
        # TODO: refactor this later
        if camera.Calibrated:
            inp_width = camera.roi[2]
            inp_height = camera.roi[3]
        else:
            inp_width = cap.get(3)
            inp_height = cap.get(4)
        scale = float(inp_width) / float(im_width)
        log.info(f"Input FPS : {cap.get(5)}")
        log.info(f"Input size: {inp_width} x {inp_height}")
        return scale, inp_width, inp_height

    def run(self):
        '''The main processing loop.'''
        log.debug('VideoProcessor thread start.')

        # This is an adaptation of this simple idea:
        # https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713

        # Prepare for processing -----------------
        cap = self.flight.cap

        # Load camera calibration
        cam = CalCamera(
            frame_size=self._full_frame_size,
            calibrationPath=self.flight.calibration_path,
            flight_radius=self.flight.flight_radius,
            marker_radius=self.flight.marker_radius,
            marker_height=self.flight.marker_height
        )
        if not cam.Calibrated:
            pass  # TODO: disable/hide the figure checkboxes in UI before processing.

        # Determine input video size and the scale wrt detector's frame size
        scale, inp_width, inp_height = self.get_size(cam, cap, ProcessorSettings.im_width)
        log.info(f'processing size: {inp_width} x {inp_height} px')
        log.debug(f'detector im_width = {ProcessorSettings.im_width} px')
        log.debug(f'scale = {scale:.4f}')

        # Platform-dependent stuff
        self._fourcc = cv2.VideoWriter_fourcc(*'H264')
        if platform.system() == 'Windows':
            self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        parent_path = self.flight.video_path.parent
        video_name = self.flight.video_path.stem

        # Output video file
        VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
        OUT_VIDEO_PATH = parent_path / f'{video_name}_out.mp4'
        w_ratio = inp_width / self._full_frame_size[0]
        h_ratio = inp_height / self._full_frame_size[1]
        RESTORE_SIZE = w_ratio > 0.95 and h_ratio > 0.95
        log.info(f'full frame size  = {self._full_frame_size}')
        log.debug(f'input ratios w,h = {w_ratio:.4f}, {h_ratio:.4f}')
        if not self.flight.is_live:
            if RESTORE_SIZE:
                log.info(f'Output size: {self._full_frame_size}')
                out = cv2.VideoWriter(
                    str(OUT_VIDEO_PATH),
                    self._fourcc, VIDEO_FPS,
                    self._full_frame_size
                )
                # The resized width if we resize height to full size
                w_final = int(self._full_frame_size[1] / inp_height * inp_width)
                resize_kwarg = {'height': self._full_frame_size[1]}
                crop_offset = (int(0.5*(w_final - self._full_frame_size[0])), 0)
                if w_final < self._full_frame_size[0]:
                    # The resized height if we resize width to full size
                    h_final = int(self._full_frame_size[0] / inp_width * inp_height)
                    resize_kwarg = {'width': self._full_frame_size[0]}
                    crop_offset = (0, int(0.5*(h_final - self._full_frame_size[1])))
                crop_idx_y = crop_offset[1] + self._full_frame_size[1]
                crop_idx_x = crop_offset[0] + self._full_frame_size[0]
            else:
                out = cv2.VideoWriter(
                    str(OUT_VIDEO_PATH),
                    self._fourcc, VIDEO_FPS,
                    (int(inp_width), int(inp_height))
                )
        else:
            # TODO: clean up this path repetition
            if not ProcessorSettings.live_videos.exists():
                ProcessorSettings.live_videos.mkdir(parents=True)
            timestr = time.strftime("%Y%m%d-%H%M")
            out = cv2.VideoWriter(
                ProcessorSettings.live_videos / f'out_{timestr}.mp4',
                self._fourcc, VIDEO_FPS,
                (int(inp_width), int(inp_height))
            )

        # Track length
        max_track_len = int(ProcessorSettings.max_track_time * VIDEO_FPS)
        # Detector
        detector = Detector(max_track_len, scale)
        # Drawing artist
        artist = Drawing(detector, cam=cam, axis=False)
        # Angle offset of current AR hemisphere wrt world coordinate system
        azimuth_delta = 0.0
        # Number of total empty frames in the input.
        num_empty_frames = 0
        # Number of consecutive empty frames at beginning of capture
        num_consecutive_empty_frames = 0
        # Maximum allowed number of consecutive empty frames. If we find more, we quit.
        MAX_CONSECUTIVE_EMPTY_FRAMES = 256
        log.info(f'3D tracking is {"ON" if ProcessorSettings.perform_3d_tracking else "OFF"}')

        fig_tracker = None
        if cam.Calibrated and ProcessorSettings.perform_3d_tracking:
            data_path = parent_path / f'{video_name}_out_data.csv'
            data_writer = data_path.open('w', encoding='utf8')
            # data_writer.write('frame_idx,p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,root1,root2\n')
            fig_tracker = figtrack.FigureTracker(
                callback=sys.stdout.write, enable_diags=True)  # FIXME: callback to a log file. callback=log.debug ?

        # aids for drawing figure start/end points over track
        is_fig_in_progress = False
        fig_img_pts = []
        # aids for drawing nominal paths
        MAX_DRAW_FIT_FRAMES = int(VIDEO_FPS * 2.)  # multiplier is the number of seconds
        fit_img_pts = []
        draw_fit = False
        num_draw_fit_frames = 0
        # Misc
        frame_idx = 0
        frame_delta = 0
        num_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        MAX_FRAME_DELTA = int(num_input_frames / 100)
        is_paused = False
        # Speed meter
        fps = FPS().start()
        sphere_offset = common.DEFAULT_CENTER  # TODO: maybe get this from UI?
        watermark_text = f'{self.application.applicationName()} - v{self.application.applicationVersion()}'

        while cap.more():
            frame_or = cap.read()

            if frame_or is None:
                num_empty_frames += 1
                num_consecutive_empty_frames += 1
                if num_consecutive_empty_frames > MAX_CONSECUTIVE_EMPTY_FRAMES:
                    # GoPro videos show empty frames, quick fix
                    log.warning(
                        f'Failed to read frame from input! '
                        f'frame_idx={frame_idx}/{num_input_frames}, '
                        f'num_empty_frames={num_empty_frames}, '
                        f'num_consecutive_empty_frames={num_consecutive_empty_frames}'
                    )
                    break
                time.sleep(0.001)
                continue
            num_consecutive_empty_frames = 0

            frame_idx += 1

            if cam.Calibrated:
                frame_or = cam.Undistort(frame_or)
                if not cam.Located and cam.AR:
                    # TODO: this is broken right now because Camera needs rework
                    log.debug('Locating camera...')
                    cam.Locate(frame_or)
                    artist.Locate(cam, center=sphere_offset)
                    # The above two calls, especially cam.Locate(), take a long time.
                    # Restart FPS meter to be fair
                    fps.start()

                # TODO: refactor this into a private method here.
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
                                world_map[:, u, v] = world_pts
                                num_pts_collected += 1
                        t2 = time.process_time()
                        t_diff = t2 - t1
                        print(f'Number of world points in row index {v}: {num_pts_collected}, collected in {t_diff} s')
                        num_pts_collected = 0
                    print('saving world_map array to file...')
                    np.save('world_map.npy', world_map)
                    print('...done.')
                # '''

            # TODO: should we make `im_width` variable so that it's always less than input video width?
            frame = imutils.resize(frame_or, width=ProcessorSettings.im_width)

            detector.process(frame)

            if cam.Located:
                # TODO: connect `artist.figure_state` to figure checkboxes in UI
                log.warning('TODO: connect `artist.figure_state` to figure checkboxes in UI')
                # OLD CODE HERE JUST FOR REFERENCE, DELETE IT WHEN FINISHED ===========================
                # artist.figure_state[FigureTypes.INSIDE_LOOPS] = loops_chk.get()
                # artist.figure_state[FigureTypes.INSIDE_SQUARE_LOOPS] = sq_loops_chk.get()
                # artist.figure_state[FigureTypes.INSIDE_TRIANGULAR_LOOPS] = tri_loops_chk.get()
                # artist.figure_state[FigureTypes.HORIZONTAL_EIGHTS] = hor_eight_chk.get()
                # artist.figure_state[FigureTypes.HORIZONTAL_SQUARE_EIGHTS] = sq_hor_eight_chk.get()
                # artist.figure_state[FigureTypes.VERTICAL_EIGHTS] = ver_eight_chk.get()
                # artist.figure_state[FigureTypes.HOURGLASS] = hourglass_chk.get()
                # artist.figure_state[FigureTypes.OVERHEAD_EIGHTS] = over_eight_chk.get()
                # artist.figure_state[FigureTypes.FOUR_LEAF_CLOVER] = clover_chk.get()
                # artist.DrawDiags = diag_chk.get()
                # END OF OLD CODE =====================================================================

            # TODO: add a `set_azimuth` method to Drawing class and connect UI events to it. The `Drawing.draw` method should just take an image frame as input.
            artist.draw(frame_or, azimuth_delta)

            if cam.Located:
                if ProcessorSettings.perform_3d_tracking:
                    # try to track the aircraft in world coordinates
                    if detector.pts_scaled[0] is not None:
                        act_pts = projection.projectImagePointToSphere(
                            cam, artist.center, detector.pts_scaled[0], frame_or, data_writer)
                        if act_pts is not None:  # and act_pts.shape[0] == 2:
                            # fig_tracker.add_actual_point(act_pts)
                            # Typically the first point is the "far" point on the sphere...Good enough for most figures that are on the far side of the camera.
                            # TODO: Make this smarter so that we track the correct path point at all times.
                            fig_tracker.add_actual_point(frame_idx, act_pts[0])
                            # fig_tracker.add_actual_point(act_pts[1])
                            if is_fig_in_progress:
                                fig_img_pts.append(detector.pts_scaled[0])

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

            if not self.flight.is_live and RESTORE_SIZE:
                # Size us back up to original. preserve aspect, crop to middle
                frame_or = imutils.resize(frame_or, **resize_kwarg)[
                    crop_offset[1]:crop_idx_y,
                    crop_offset[0]:crop_idx_x]

            # Write text
            cv2.putText(frame_or, watermark_text, (10, 15),
                        cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 255), 1)

            # Show output
            self.new_frame_available.emit(self._cv_img_to_qimg(frame_or))

            # Save frame
            out.write(frame_or)
            fps.update()

            # Display processing progress to user
            frame_time = frame_idx / VIDEO_FPS
            progress = int(frame_idx / (num_input_frames) * 100)
            if (frame_idx == 1) or (progress % 5 == 0 and frame_delta >= MAX_FRAME_DELTA) or (frame_time % 1 < 0.05):
                self.progress_updated.emit((frame_time, progress))
                frame_delta = 0
            frame_delta += 1

            # azimuth_delta += 0.4  # For quick visualization of sphere outline

            # TODO: refactor this huge if block into handlers for signals from UI.
            # =============================================================================================================
            # key = cv2.waitKeyEx(1)  # & 0xFF
            # LSB = key & 0xff

            # if key != -1:
            #     print(f'\nKey: {key}  LSB: {key & 0xff}')

            # if LSB == 27 or key == 1048603 or cv2.getWindowProperty(WINDOW_NAME, 1) < 0:  # ESC
            #     # Exit
            #     break
            # elif LSB == 112:  # p
            #     # Pause/Resume with ability to quit while paused
            #     # TODO: allow sphere rotation and movement while paused!
            #     fps.pause()
            #     is_paused = True
            #     pause_str = f'pausing at frame {frame_idx}/{num_input_frames} (time={timedelta(seconds=frame_time)})'
            #     print(f'\n{pause_str}')
            #     log.info(pause_str)
            #     get_out = False
            #     while is_paused:
            #         key = cv2.waitKeyEx(100)
            #         LSB = key & 0xff
            #         if LSB == 27 or cv2.getWindowProperty(WINDOW_NAME, 1) < 0:  # ESC
            #             get_out = True
            #             break
            #         is_paused = not (LSB == 112)
            #     if get_out:
            #         log.info(f'quitting from pause at frame {frame_idx}')
            #         break
            #     fps.resume()
            #     resume_str = f'resuming from frame {frame_idx}'
            #     print(resume_str)
            #     log.info(resume_str)
            # elif LSB == 32 or key == 1048608:  # Space
            #     # Clear the current track
            #     detector.clear()
            # elif key == 1113939 or key == 2555904 or key == 65363:  # Right Arrow
            #     azimuth_delta -= 0.5
            # elif key == 1113937 or key == 2424832 or key == 65361:  # Left Arrow
            #     # Positive CCW around Z, per normal convention
            #     azimuth_delta += 0.5
            # elif LSB == 99 or key == 1048675:  # c
            #     # Relocate AR sphere
            #     cam.Located = False
            #     cam.AR = True
            # elif ProcessorSettings.perform_3d_tracking and LSB == 91:  # [ key
            #     # Mark the beginning of a new figure
            #     if fig_tracker is not None:
            #         detector.clear()
            #         fig_tracker.start_figure()
            #         is_fig_in_progress = True
            # elif ProcessorSettings.perform_3d_tracking and LSB == 93:  # ] key
            #     # Mark the end of the current figure
            #     if fig_tracker is not None:
            #         print(f'before finishing figure: fig_img_pts size ={len(fig_img_pts)}')
            #         fig_tracker.finish_figure()
            #         is_fig_in_progress = False
            #         # Uniform t along the figure path
            #         # t = np.linspace(0., 1., fig_tracker._curr_figure_fitter.num_nominal_pts)
            #         # The figure's chosen t distribution (initial, final)
            #         t = (fig_tracker._curr_figure_fitter.diag.u0,
            #              fig_tracker._curr_figure_fitter.u)
            #         print(f'figure finish: shape of u0 vs. u final: {t[0].shape, t[1].shape}')
            #         # Trim our detected points according to the fit
            #         trims = fig_tracker._curr_figure_fitter.diag.trim_indexes
            #         print(f'fig_img_pts before: size = {len(fig_img_pts)}')
            #         print(f'trims: shape = {trims.shape}')
            #         fig_img_pts = list(tuple(pair)
            #                            for pair in np.array(fig_img_pts)[trims].tolist())
            #         print(f'fig_img_pts after: size = {len(fig_img_pts)}')
            #         # The last item is the tuple of (initial, final) fit params
            #         for i, fp in enumerate(fig_tracker.figure_params[-1]):
            #             nom_pts = projection.projectSpherePointsToImage(
            #                 cam, fig_tracker._curr_figure_fitter.get_nom_point(*fp, *t[i]))
            #             fit_img_pts.append(tuple(map(tuple, nom_pts.tolist())))
            #         # print(f't ({len(t)} pts)')
            #         print(f'fit_img_pts[initial] ({len(fit_img_pts[0])} pts)')
            #         # # print(fit_img_pts[0])
            #         print(f'fit_img_pts[final] ({len(fit_img_pts[1])} pts)')
            #         # # print(fit_img_pts[1])
            #         print(f'fig_img_pts ({len(fig_img_pts)} pts)')
            #         # # print()
            #         # Set the flag for drawing the fit figures, diags, etc.
            #         draw_fit = True
            # elif LSB == ord('d'):
            #     # offset sphere right (+X)
            #     log.info(
            #         f'User moves sphere center X by {ProcessorSettings.sphere_xy_delta} after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.MoveCenterX(ProcessorSettings.sphere_xy_delta)
            # elif LSB == ord('a'):
            #     # offset sphere left (-X)
            #     log.info(
            #         f'User moves sphere center X by {-ProcessorSettings.sphere_xy_delta} after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.MoveCenterX(-ProcessorSettings.sphere_xy_delta)
            # elif LSB == ord('w'):
            #     # offset sphere away from cam (+Y)
            #     log.info(
            #         f'User moves sphere center Y by {ProcessorSettings.sphere_xy_delta} after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.MoveCenterY(ProcessorSettings.sphere_xy_delta)
            # elif LSB == ord('s'):
            #     # offset sphere toward cam (-Y)
            #     log.info(
            #         f'User moves sphere center Y by {-ProcessorSettings.sphere_xy_delta} after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.MoveCenterY(-ProcessorSettings.sphere_xy_delta)
            # elif LSB == ord('x'):
            #     # reset sphere offset to world origin
            #     log.info(
            #         f'User resets sphere center after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.ResetCenter()

        fps.stop()
        self.progress_updated.emit((frame_time, progress))
        final_progress_str = f'frame_idx={frame_idx}, num_input_frames={num_input_frames}, num_empty_frames={num_empty_frames}, progress={progress}%'
        elapsed_time_str = f'Elapsed time: {fps.elapsed():.1f}'
        mean_fps_str = f'Approx. FPS: {fps.fps():.1f}'
        log.info(f'Finished processing {self.flight.video_path}')
        log.info(f'Result video written to {OUT_VIDEO_PATH}')
        log.info(final_progress_str)
        log.info(elapsed_time_str)
        log.info(mean_fps_str)

        if fig_tracker is not None:
            fig_tracker.finish_all()
            fig_tracker.export(parent_path / f'{video_name}_out_figures.npz')

        # Clean up
        out.release()
        cv2.destroyAllWindows()
        if cam.Located and ProcessorSettings.perform_3d_tracking:
            data_writer.close()
        log.debug('VideoProcessor thread end.')
