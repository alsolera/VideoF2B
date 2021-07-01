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

'''Module for drawing flight track and Augmented Reality in video.'''

import logging
import math
from collections import defaultdict

import cv2
import numpy as np
import scipy.spatial.transform as xform

import common
from common import FigureTypes

ROT = xform.Rotation


class Scatter:
    '''Defines a collection of scattered points.'''
    pass


class Polyline:
    '''Defines a polyline.'''
    pass


class Drawing:
    '''Container that performs all the drawing of AR sphere, track, figures, etc. in any given image frame.'''
    # Default location of drawn sphere wrt world center.
    DEFAULT_CENTER = np.float32([0., 0., 0.])
    # Default point density per pi (180 degrees) of arc.
    DEFAULT_N = 100

    def __init__(self, detector, **kwargs):
        '''Initialize the Drawing instance.
        If only track drawing is required, provide a Detector instance.
        If drawing AR geometry is also required, supply the following kwargs:
            `cam`: instance of CalCamera.
            `R`: radius of the Augmented Reality sphere. Default is 21.0 m.
            `marker_radius`: radius of the world markers around the sphere. Default is 25.0 m.
            `center`: 3-tuple or ndarray of (x, y, z) location of drawn sphere with respect to the world center defined by markers.
                        Default is the origin (0, 0, 0).
            Optional:
                `point_density`: number of arc points per 180 degrees of arc. Default is 100 points.
        '''
        # Defines the visibility state of all drawn figures.
        self.figure_state = defaultdict(bool)
        # Map figure types to their drawing functions. Not all figure types are used.
        self._figure_funcs = {
            FigureTypes.INSIDE_LOOPS: Drawing.draw_loop,
            FigureTypes.INSIDE_SQUARE_LOOPS: Drawing.draw_square_loop,
            FigureTypes.INSIDE_TRIANGULAR_LOOPS: Drawing.draw_triangular_loop,
            FigureTypes.HORIZONTAL_EIGHTS: Drawing.drawHorEight,
            FigureTypes.HORIZONTAL_SQUARE_EIGHTS: Drawing.drawSquareEight,
            FigureTypes.VERTICAL_EIGHTS: Drawing.drawVerEight,
            FigureTypes.HOURGLASS: Drawing.drawHourglass,
            FigureTypes.OVERHEAD_EIGHTS: Drawing.drawOverheadEight,
            FigureTypes.FOUR_LEAF_CLOVER: Drawing.drawFourLeafClover
        }
        self.logger = kwargs.pop('logger', logging.getLogger(__name__))
        self._is_center_at_origin = False
        self._detector = detector
        self.axis = kwargs.pop('axis', False)
        self._cam = kwargs.pop('cam', None)
        if self._cam is not None:
            self.R = kwargs.pop('R', common.DEFAULT_FLIGHT_RADIUS)
            self.marker_radius = kwargs.pop('marker_radius', common.DEFAULT_MARKER_RADIUS)
            center = kwargs.pop('center', Drawing.DEFAULT_CENTER.copy()
                                ) or Drawing.DEFAULT_CENTER.copy()
            # Ensure it's a numpy array (allow tuple, list as input)
            self.center = np.float32(center)
            self._evaluate_center()
            self._point_density = kwargs.pop('point_density', Drawing.DEFAULT_N)
            Drawing.square_loop_pts = Drawing.calc_square_loop(self.R)

        # Cache of drawing point collections, keyed on the variables that affect a unique AR scene.
        self._cache = {}

    def MoveCenterX(self, delta):
        self.center[0] += delta
        self._evaluate_center()

    def MoveCenterY(self, delta):
        self.center[1] += delta
        self._evaluate_center()

    def ResetCenter(self):
        '''Reset sphere center to default.'''
        self.center = Drawing.DEFAULT_CENTER.copy()
        self._evaluate_center()

    def _evaluate_center(self):
        self.center = self.center.round(1)
        self._is_center_at_origin = (abs(self.center) < 0.01).all()
        self.logger.info(f'Sphere center: {self.center}')

    @staticmethod
    def get_arc(r, alpha, rho=100):
        '''Return 3D points for an arc of radius `r` and included angle `alpha`
        with point density `rho`, where `rho` is number of points per 2*pi.
        Arc center is (0, 0, 0).  The arc lies in the XY plane.
        Arc starts at zero angle, i.e., at (r, 0, 0) coordinate, and ends at `alpha`.
        Angle measurements are in radians.
        Endpoint is always included.
        '''
        nom_step = 2 * math.pi / rho
        num_pts = int(alpha / nom_step)
        act_step = alpha / num_pts
        if act_step > nom_step:
            num_pts += 1
            act_step = alpha / num_pts
        pts = np.array(
            [
                (r*math.cos(act_step*t),
                 r*math.sin(act_step*t),
                 0.0)
                for t in range(num_pts + 1)
            ]
        )
        return pts

    @staticmethod
    def PointsInCircum(r, n=100):
        two_pi_over_n = 2 * math.pi / n
        return [(math.cos(two_pi_over_n*x)*r, math.sin(two_pi_over_n*x)*r) for x in range(0, n+1)]

    @staticmethod
    def PointsInHalfCircum(r, n=100):
        pi_over_n = math.pi / n
        return [(math.cos(pi_over_n*x)*r, math.sin(pi_over_n*x)*r) for x in range(0, n+1)]

    @staticmethod
    def _get_track_color(x, x_max):
        color = int(255 * float(x) / float(x_max))
        return (0, min(255, color*2), min(255, (255-color)*2))

    def _draw_track(self, img):
        '''Draw the track behind the aircraft.'''
        pts = self._detector.pts_scaled
        maxlen = self._detector.maxlen
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # If either of the tracked points are None, draw the available point
            f1 = pts[i] is None
            f2 = pts[i-1] is None
            if f1 or f2:
                if not (f1 and f2):
                    pt = pts[i-1] if f1 else pts[i]
                    cv2.circle(img, pt, 1, Drawing._get_track_color(i, maxlen), -1)
                continue
            # draw the lines
            cv2.line(img, pts[i - 1], pts[i], Drawing._get_track_color(i, maxlen), 1)
        return img

    def draw(self, img, azimuth_delta=0, figures=None):
        '''Draw all relevant geometry in the given image frame.'''
        if self._cam.Located:
            self._draw_all_geometry(img, azimuth_delta)
            self._draw_figures(img, azimuth_delta, figures)
        self._draw_track(img)

    def _draw_all_geometry(self, img, azimuth_delta=0):
        '''Draw all AR geometry according to the current location and rotation of AR sphere.'''
        # Local names to minimize dot-name lookups
        rvec = self._cam.rvec
        tvec = self._cam.tvec
        newcameramtx = self._cam.newcameramtx
        distZero = np.zeros_like(self._cam.dist)
        cam_pos = self._cam.cam_pos
        r = self._cam.flightRadius
        center = self.center
        # Begin drawing
        if not self._is_center_at_origin:
            cv2.putText(
                img,
                f'C=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})',
                (120, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(
            img,
            f'R={self.R:.2f}',
            (30, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        Drawing._draw_level(img, rvec, tvec, newcameramtx, distZero, center, r)
        Drawing._draw_all_merid(img, rvec, tvec, newcameramtx, distZero, center, r, azimuth_delta)
        if self.axis:
            Drawing._draw_axis(img, rvec, tvec, newcameramtx, distZero, center)
        Drawing._draw_45(img, rvec, tvec, newcameramtx, distZero, center, r, color=(0, 255, 0))
        Drawing._draw_base_tol(img, rvec, tvec, newcameramtx, distZero, center, r)
        Drawing._draw_edge(img, cam_pos, rvec, tvec, newcameramtx, distZero, center, r)
        Drawing._draw_points(img, self._cam, rvec, tvec, newcameramtx, distZero, center)
        return img

    def _draw_figures(self, img, azimuth_delta=0., figures=None):
        '''Draw currently selected nominal figures.'''
        if any(self.figure_state.values()):
            rvec = self._cam.rvec
            tvec = self._cam.tvec
            newcameramtx = self._cam.newcameramtx
            distZero = np.zeros_like(self._cam.dist)
            center = self.center
            r = self._cam.flightRadius
            for ftype, fflag in self.figure_state.items():
                if fflag:
                    self._figure_funcs[ftype](
                        img, azimuth_delta, rvec, tvec, newcameramtx, distZero, center, r)

    @staticmethod
    def _draw_level(img, rvec, tvec, cameramtx, dist, center, r):
        '''Draw the base level (aka the equator) of flight hemisphere.'''
        # unit is m
        n = 100
        coords = np.asarray(Drawing.PointsInCircum(r, n), np.float32)
        points = np.c_[coords, np.zeros(1 + n)] + center
        twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
        twoDPoints = twoDPoints.astype(int).reshape(-1, 2)
        for i in range(np.shape(twoDPoints)[0] - 1):
            img = cv2.line(
                img, tuple(twoDPoints[i]), tuple(twoDPoints[i+1]), (255, 255, 255), 1)
        return img

    @staticmethod
    def _draw_all_merid(img, rvec, tvec, cameramtx, dist, center, r, azimuth_delta):
        '''Draw all the meridians of the flight hemisphere.'''
        for angle in range(0, 180, 45):
            gray = 255 - angle*2
            color = (gray, gray, gray)
            Drawing._draw_merid(img, angle + azimuth_delta, rvec, tvec,
                                cameramtx, dist, center, r, color)
        return img

    @staticmethod
    def _draw_merid(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        '''Draw one meridian of the flight hemisphere, defined by an angle in degrees CCW from the reference azimuth.'''
        # unit is m
        n = 100
        pi = math.pi
        angle = angle * pi/180
        c = math.cos(angle)
        s = math.sin(angle)
        RotMatrix = [[c, s, 0],
                     [s, c, 0],
                     [0, 0, 1]]

        coords = np.asarray(Drawing.PointsInHalfCircum(r=r, n=n), np.float32)
        points = np.c_[np.zeros(1+n), coords]

        points = np.matmul(points, RotMatrix) + center
        twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
        twoDPoints = twoDPoints.astype(int).reshape(-1, 2)

        for i in range(np.shape(twoDPoints)[0] - 1):
            img = cv2.line(img, tuple(twoDPoints[i]),
                           tuple(twoDPoints[i+1]), color, 1)
        return img

    @staticmethod
    def _draw_axis(img, rvec, tvec, cameramtx, dist, center):
        # unit is m
        points = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, 5], [0, 0, 0]]) + center
        axisPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
        img = cv2.line(img,
                       tuple(axisPoints[3]),
                       tuple(axisPoints[0]), (0, 0, 255), 1)
        img = cv2.line(img,
                       tuple(axisPoints[3]),
                       tuple(axisPoints[1]), (0, 255, 0), 1)
        img = cv2.line(img,
                       tuple(axisPoints[3]),
                       tuple(axisPoints[2]), (255, 0, 0), 1)
        return img

    @staticmethod
    def _draw_45(img, rvec, tvec, cameramtx, dist, center, r=20, color=(255, 255, 255)):
        # unit is m
        n = 100
        pi = math.pi
        r45 = math.cos(pi/4) * r
        # TODO: use np.float32() constructor
        coords = np.asarray(Drawing.PointsInCircum(r=r45, n=n), np.float32)
        points = np.c_[coords, np.ones(1+n)*r45] + center
        twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
        twoDPoints = twoDPoints.astype(int).reshape(-1, 2)
        for i in range(np.shape(twoDPoints)[0] - 1):
            img = cv2.line(img,
                           tuple(twoDPoints[i]),
                           tuple(twoDPoints[i+1]), color, 1)
        return img

    @staticmethod
    def _draw_base_tol(img, rvec, tvec, cameramtx, dist, center, R=21.):
        '''Draw the upper & lower limits of base flight envelope. Nominally these are 0.30m above and below the equator.'''
        n = 200
        tol = 0.3
        r = math.sqrt(R**2 - tol**2)
        coords = np.asarray(Drawing.PointsInCircum(r, n))
        pts_lower = np.c_[coords, np.ones(1 + n) * (-tol)] + center
        pts_upper = np.c_[coords, np.ones(1 + n) * tol] + center
        img_pts_lower = cv2.projectPoints(
            pts_lower, rvec, tvec, cameramtx, dist)[0].astype(int).reshape(-1, 2)
        img_pts_upper = cv2.projectPoints(
            pts_upper, rvec, tvec, cameramtx, dist)[0].astype(int).reshape(-1, 2)
        color = (204, 204, 204)
        # Draw dashed lines
        num_on = 3
        num_off = 2
        counter = 0
        is_visible = True
        for i in range(n):
            counter += 1
            if is_visible:
                img = cv2.line(img,
                               tuple(img_pts_lower[i]),
                               tuple(img_pts_lower[i + 1]), color, 1)
                img = cv2.line(img,
                               tuple(img_pts_upper[i]),
                               tuple(img_pts_upper[i + 1]), color, 1)
                if counter == num_on:
                    counter = 0
                    is_visible = False
            else:
                if counter == num_off:
                    counter = 0
                    is_visible = True
        return img

    @staticmethod
    def _draw_edge(img, cam_pos, rvec, tvec, cameramtx, dist, center, R):
        '''Draw the edge outline of the flight sphere as seen from camera's perspective.'''
        # normal vector of circle: points from sphere center to camera
        p = cam_pos.reshape((3,)) - center
        d = np.linalg.norm(p)
        if d <= R:
            # We are on the surface of the sphere, or inside its volume.
            return
        n = p / d
        # TODO: use math instead of np for trig functions of scalars to speed things up
        # TODO: use np.float32() constructor where possible
        phi = np.arctan2(n[1], n[0])
        rot_mat = np.array([
            [np.cos(phi), -np.sin(phi), 0.],
            [np.sin(phi), np.cos(phi), 0.],
            [0., 0., 1.]])
        u = np.array([0., 1., 0.])
        u = rot_mat.dot(u)
        v = np.cross(n, u)
        t = np.linspace(0., 1., 100)
        k = np.pi  # semi-circle
        c = (R**2 / d) * n + center
        det = d**2 - R**2
        if det < 0.:
            # guard against negative arg of sqrt
            r = 0.0
        else:
            r = R / d * np.sqrt(det)
        world_pts = np.array([c + r * (np.cos(k * t_i) * u + np.sin(k * t_i) * v) for t_i in t])
        img_pts, _ = cv2.projectPoints(world_pts, rvec, tvec, cameramtx, dist)
        img_pts = img_pts.astype(int).reshape(-1, 2)
        for i in range(img_pts.shape[0] - 1):
            img = cv2.line(
                img, tuple(img_pts[i]), tuple(img_pts[i+1]), (255, 0, 255), 1)
        return img

    @staticmethod
    def _draw_points(img, cam, rvec, tvec, newcameramtx, dist, center):
        r = cam.flightRadius
        rcos45 = cam.markRadius * 0.70710678
        marker_size_x = 0.20  # marker width, in m
        marker_size_z = 0.60  # marker height, in m
        world_points = np.array([
            # Points on sphere centerline: sphere center, pilot's feet, top of sphere.
            [0, 0, 0],
            [0, 0, -cam.markHeight],
            [0, 0, r],
            # Points on equator: bottom of right & left marker, right & left antipodes, front & rear antipodes
            [rcos45, rcos45, 0],
            [-rcos45, rcos45, 0],
            [cam.markRadius, 0, 0],
            [-cam.markRadius, 0, 0],
            [0, -cam.markRadius, 0],
            [0, cam.markRadius, 0],
            # Points on corners of an imaginary marker at center of sphere (optional)
            [0.5 * marker_size_x, 0., 0.5 * marker_size_z],
            [-0.5 * marker_size_x, 0., 0.5 * marker_size_z],
            [-0.5 * marker_size_x, 0., -0.5 * marker_size_z],
            [0.5 * marker_size_x, 0., -0.5 * marker_size_z],
            # Points on corners of front marker
            [0.5 * marker_size_x, cam.markRadius, 0.5 * marker_size_z],
            [-0.5 * marker_size_x, cam.markRadius, 0.5 * marker_size_z],
            [-0.5 * marker_size_x, cam.markRadius, -0.5 * marker_size_z],
            [0.5 * marker_size_x, cam.markRadius, -0.5 * marker_size_z],
            # Points on corners of right marker
            [rcos45 + 0.5 * marker_size_x, rcos45, 0.5 * marker_size_z],
            [rcos45 - 0.5 * marker_size_x, rcos45, 0.5 * marker_size_z],
            [rcos45 - 0.5 * marker_size_x, rcos45, -0.5 * marker_size_z],
            [rcos45 + 0.5 * marker_size_x, rcos45, -0.5 * marker_size_z],
            # Points on corners of left marker
            [-rcos45 + 0.5 * marker_size_x, rcos45, 0.5 * marker_size_z],
            [-rcos45 - 0.5 * marker_size_x, rcos45, 0.5 * marker_size_z],
            [-rcos45 - 0.5 * marker_size_x, rcos45, -0.5 * marker_size_z],
            [-rcos45 + 0.5 * marker_size_x, rcos45, -0.5 * marker_size_z],
        ],
            dtype=np.float32)
        # Adjust locations of the center marker points. Leave all other references as-is.
        world_points[9:13, ] += center

        imgpts, _ = cv2.projectPoints(world_points, rvec, tvec, newcameramtx, dist)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Draw the world points in the video according to our color scheme
        for i, img_pt in enumerate(imgpts):
            if i < 9:
                # RED: points on centerline and equator
                pt_color = (0, 0, 255)
            elif 8 < i < 13:
                # CYAN: corners of imaginary marker at center of sphere
                pt_color = (255, 255, 0)
                # print(f'center marker: i={i}, img_pt={img_pt}')
            else:
                # GREEN: corners of the three outside markers
                pt_color = (0, 255, 0)
                # print(f'perimeter markers: i={i}, img_pt={img_pt}')
            cv2.circle(img, (img_pt[0], img_pt[1]), 1, pt_color, -1)
        return img

    @staticmethod
    def draw_loop(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        # unit is m
        n = 50
        pi = math.pi
        YawAngle = angle * pi/180
        c = math.cos(YawAngle)
        s = math.sin(YawAngle)

        center = [0,
                  0.85356*r,    # cos(theta)*cos(theta)  where theta = 22.5 degrees
                  0.35355*r]    # cos(theta)*sin(theta)

        # Rotation around the world X-axis. CCW positive.
        TiltMatrix = [[1,       0,        0],
                      [0,  0.92388, 0.38268],  # cos(theta), sin(theta)
                      [0, -0.38268, 0.92388]]  # -sin(theta), cos(theta)

        # Rotation around the world Z-axis. NOTE: CW positive.
        YawMatrix = [[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]]

        rLoop = r*0.382683  # sin(theta)

        coords = np.asarray(Drawing.PointsInCircum(r=rLoop, n=n), np.float32)
        points = np.c_[np.zeros(1+n), coords]
        points = np.matmul(points, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        points = np.matmul(points, TiltMatrix)+center
        points = np.matmul(points, YawMatrix)

        coords_in = coords * 0.99
        points_in = np.c_[np.zeros(1+n), coords_in]
        points_in = np.matmul(points_in, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        points_in = np.matmul(points_in, TiltMatrix)+center
        points_in = np.matmul(points_in, YawMatrix)

        coords_out = coords * 1.01
        points_out = np.c_[np.zeros(1+n), coords_out]
        points_out = np.matmul(points_out, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        points_out = np.matmul(points_out, TiltMatrix)+center
        points_out = np.matmul(points_out, YawMatrix)

        twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
        twoDPoints = twoDPoints.astype(int).reshape(-1, 2)

        twoDPoints_in, _ = cv2.projectPoints(points_in, rvec, tvec, cameramtx, dist)
        twoDPoints_in = twoDPoints_in.astype(int).reshape(-1, 2)

        twoDPoints_out, _ = cv2.projectPoints(points_out, rvec, tvec, cameramtx, dist)
        twoDPoints_out = twoDPoints_out.astype(int).reshape(-1, 2)

        border_color = (100, 100, 100)
        for i in range(np.shape(twoDPoints)[0] - 1):
            img = cv2.line(
                img, tuple(twoDPoints[i]), tuple(twoDPoints[i+1]), color, 2)
            img = cv2.line(
                img, tuple(twoDPoints_in[i]), tuple(twoDPoints_in[i+1]), border_color, 1)
            img = cv2.line(
                img, tuple(twoDPoints_out[i]), tuple(twoDPoints_out[i+1]), border_color, 1)
        return img

    @staticmethod
    def draw_top_loop(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        # unit is m
        n = 100
        pi = math.pi
        YawAngle = angle * pi/180
        c = math.cos(YawAngle)
        s = math.sin(YawAngle)

        center = [0,
                  0.35355*r,
                  0.85356*r]

        TiltMatrix = [[1,       0,        0],
                      [0,  0.38268, 0.92388],
                      [0, -0.92388, 0.38268]]

        YawMatrix = [[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]]

        rLoop = r*0.382683

        coords = np.asarray(Drawing.PointsInCircum(r=rLoop, n=n), np.float32)
        points = np.c_[np.zeros(1+n), coords]
        points = np.matmul(points, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        points = np.matmul(points, TiltMatrix)+center
        points = np.matmul(points, YawMatrix)

        twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
        twoDPoints = twoDPoints.astype(int).reshape(-1, 2)

        for i in range(np.shape(twoDPoints)[0] - 1):
            img = cv2.line(
                img, tuple(twoDPoints[i]), tuple(twoDPoints[i+1]), color, 1)
        return img

    @staticmethod
    def drawHorEight(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        Drawing.draw_loop(img, angle+24.47, rvec, tvec, cameramtx,
                          dist, center, r, color=(255, 255, 255))
        Drawing.draw_loop(img, angle-24.47, rvec, tvec, cameramtx,
                          dist, center, r, color=(255, 255, 255))
        return img

    @staticmethod
    def drawVerEight(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        Drawing.draw_loop(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255))
        Drawing.draw_top_loop(img, angle, rvec, tvec, cameramtx,
                              dist, center, r, color=(255, 255, 255))
        return img

    @staticmethod
    def drawOverheadEight(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255)):
        Drawing.draw_top_loop(img, angle+90, rvec, tvec, cameramtx,
                              dist, center, r, color=(255, 255, 255))
        Drawing.draw_top_loop(img, angle-90, rvec, tvec, cameramtx,
                              dist, center, r, color=(255, 255, 255))
        return img

    @staticmethod
    def calc_square_loop(r):
        # Radius of minor arc at 45deg elevation. Also the height of the 45deg elevation arc.
        r45 = r * 0.7071067811865476
        # Helper arc for the bottom flat and both laterals
        major_arc = Drawing.get_arc(r, 0.25*math.pi)
        # Helper arc for the top flat
        minor_arc = Drawing.get_arc(r45, 0.25*math.pi)
        points = np.vstack((
            # Arc 1: bottom flat
            ROT.from_euler('z', 0.375*math.pi).apply(major_arc),
            # Arc 2: left lateral
            ROT.from_euler('xz', [0.5*math.pi, 0.625*math.pi]).apply(major_arc),
            # Arc 3: top flat
            ROT.from_euler('zy', [0.375*math.pi, math.pi]
                           ).apply(minor_arc) + np.array((0., 0., r45)),
            # Arc 4: right lateral
            ROT.from_euler('xyz', [-0.5*math.pi, -0.25*math.pi, 0.375*math.pi]).apply(major_arc)
        ))
        return points

    @staticmethod
    def draw_square_loop(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        # World rotation of the whole figure
        rot_yaw = ROT.from_euler('z', -math.radians(angle))
        points = rot_yaw.apply(Drawing.square_loop_pts) + center

        twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
        twoDPoints = twoDPoints.astype(int).reshape(-1, 2)

        for i in range(np.shape(twoDPoints)[0] - 1):
            img = cv2.line(img, tuple(twoDPoints[i]),
                           tuple(twoDPoints[i+1]), color, 3)
        return img

    @staticmethod
    def draw_triangular_loop(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        # TODO: draw triangular loop
        pass

    @staticmethod
    def drawSquareEight(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        # TODO: draw square eight
        pass

    @staticmethod
    def drawHourglass(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        # TODO: draw hourglass
        pass

    @staticmethod
    def drawFourLeafClover(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
        # TODO: draw four-leaf clover
        pass
