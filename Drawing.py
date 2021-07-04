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
from math import cos, pi, sin, sqrt

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ROT

import common
from common import FigureTypes

logger = logging.getLogger(__name__)
HALF_PI = 0.5 * pi
QUART_PI = 0.25 * pi
EIGHTH_PI = 0.125 * pi
TWO_PI = 2.0 * pi


class Colors:
    '''Shortcuts for OpenCV-compatible colors.'''
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)


class Plot:
    '''Base class for plotting primitives.
    * Call `draw()` to draw the Plot instance in your image.
    kwargs:
        size: the line thickness or point radius.
        color: the color of the primitives in this Plot.
        is_fixed: bool indicating whether this Plot is fixed in object space or not.
                  If True, world transforms do not affect the object coordinates.
                  If False (default), then world transforms will
                    rotate, scale, and translate the object coordinates.
    '''

    def __init__(self, obj_pts, **kwargs):
        # The object points (world points)
        self._obj_pts = obj_pts
        self._img_pts = None
        # Attributes of the primitive
        self._size = kwargs.pop('size', 1)
        self._color = kwargs.pop('color', Colors.BLACK)
        self._is_fixed = kwargs.pop('is_fixed', False)
        # Optimization cache
        self._cache = {}

    def _calculate(self, key, rvec, tvec, cameraMatrix, distCoeffs):
        '''Calculate image points for the given optimization key.
        The key is a tuple of (z_angle, translation) such that object points are
        transformed according to:
        `ROT.from_euler('z', z_angle, degrees=True).apply(self._obj_pts) + translation`
        where `z_angle` is in degrees.
        If the key is not yet in the cache, transform object points according to the key
        and calculate the corresponding image points according to the camera parameters.
        '''
        if key not in self._cache:
            z_angle, translation = key
            if not self._is_fixed:
                obj_pts = ROT.from_euler('z', z_angle, degrees=True).apply(
                    self._obj_pts) + translation
            else:
                obj_pts = self._obj_pts
            tmp, _ = cv2.projectPoints(obj_pts, rvec, tvec, cameraMatrix, distCoeffs)
            self._cache[key] = tmp.astype(int).reshape(-1, 2)
        self._img_pts = self._cache[key]

    def draw(self, key, **kwargs):
        '''Call this method from derived classes before drawing the image points.'''
        rvec = kwargs.pop('rvec')
        tvec = kwargs.pop('tvec')
        cameraMatrix = kwargs.pop('cameraMatrix')
        distCoeffs = kwargs.pop('distCoeffs')
        self._calculate(key, rvec, tvec, cameraMatrix, distCoeffs)


class Scatter(Plot):
    '''Defines a collection of scattered points.'''

    def __init__(self, obj_pts, **kwargs):
        super().__init__(obj_pts, **kwargs)

    def draw(self, img, key, **kwargs):
        '''Draw scatter points as solid-filled circles using their attributes.'''
        super().draw(key, **kwargs)
        for point in self._img_pts:
            img = cv2.circle(img,
                             (point[0], point[1]),
                             self._size, self._color, -1)
        return img


class Polyline(Plot):
    '''Defines a polyline.'''

    def __init__(self, obj_pts, **kwargs):
        super().__init__(obj_pts, **kwargs)

    def draw(self, img, key, **kwargs):
        '''Draw this polyline using its attributes.'''
        super().draw(key, **kwargs)
        for i in range(1, self._img_pts.shape[0]):
            img = cv2.line(img,
                           tuple(self._img_pts[i]),
                           tuple(self._img_pts[i-1]),
                           self._color, self._size)
        return img


class Scene:
    '''A scene consists of a collection of Plot-like objects.'''

    def __init__(self, *items):
        self._items = list(items)

    def add(self, item):
        '''Add an item to this scene.'''
        self._items.append(item)

    def draw(self, img, key, **kwargs):
        '''Draw this scene in the given image.'''
        for item in self._items:
            img = item.draw(img, key, **kwargs)
        return img


class DummyScene:
    '''Placeholder object for an empty scene.'''

    def draw(self, *args, **kwargs):
        pass


DEFAULT_SCENE = DummyScene()


class Drawing:
    '''Container that performs all the drawing of AR sphere, track, figures, etc., in any given image frame.'''
    # Default location of drawn sphere wrt world center.
    DEFAULT_CENTER = np.float32([0., 0., 0.])
    # Default point density per pi (180 degrees) of arc.
    DEFAULT_N = 100

    def __init__(self, detector, **kwargs):
        '''Initialize the Drawing artist instance.
        If only track drawing is required, provide a Detector instance.
        If drawing AR geometry is also required, supply the following kwargs:
            `cam`: instance of CalCamera.
            `center`: 3-tuple or ndarray of (x, y, z) location of drawn sphere
                      with respect to the world center defined by markers.
                      Default is Drawing.DEFAULT_CENTER, which is the origin (0, 0, 0).
            Optional:
                `axis`: True to draw vertical axis, False otherwise.
                `point_density`: number of arc points per full circle.
                                 Default is Drawing.DEFAULT_N, which is 100.
        '''
        # Defines the visibility state of all drawn figures.
        self.figure_state = defaultdict(bool)
        self._is_center_at_origin = False
        self._detector = detector
        self._cam = kwargs.pop('cam', None)
        self.R = None
        self.marker_radius = None
        self.center = None
        self.axis = kwargs.pop('axis', False)
        self._point_density = None
        self.Locate(self._cam, **kwargs)
        self._init_scenes()

    def Locate(self, cam, **kwargs):
        '''Locate a new camera or relocate an existing one.'''
        self._cam = cam
        if self._cam is not None and self._cam.Located:
            self.R = self._cam.flightRadius
            self.marker_radius = self._cam.markRadius
            center = kwargs.pop('center', Drawing.DEFAULT_CENTER.copy())
            # Ensure it's a numpy array (allow tuple, list as input)
            self.center = np.float32(center)
            self._evaluate_center()
            self._point_density = kwargs.pop('point_density', Drawing.DEFAULT_N)
            self._init_scenes()

    def _init_scenes(self):
        if not self._cam.Located:
            self._scenes = {}
            return
        # When the cam is located, we define lots of reference geometry.
        # =============== Base scene: main hemisphere and markers =============
        sc_base = Scene()
        # --- The base level (aka the equator) of flight hemisphere.
        sc_base.add(Polyline(Drawing.get_arc(self.R, TWO_PI), size=1, color=Colors.WHITE))
        # --- All the meridians of the flight hemisphere.
        # rot for initial meridian in YZ plane
        rot0 = ROT.from_euler('xz', [HALF_PI, HALF_PI])
        for angle in range(0, 180, 45):
            gray = 255 - 2*angle
            color = (gray, gray, gray)
            # rot for orienting each meridian
            rot1 = ROT.from_euler('z', angle, degrees=True)
            pts = rot1.apply(rot0.apply(self.get_arc(self.R, pi)))
            sc_base.add(Polyline(pts, size=1, color=color))
        # --- The coordinate axis reference at sphere center
        if self.axis:
            # XYZ sphere axes
            p = np.float32([
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 5]
            ])
            sc_base.add(Polyline((p[0], p[1]), size=1, color=Colors.RED))
            sc_base.add(Polyline((p[0], p[2]), size=1, color=Colors.GREEN))
            sc_base.add(Polyline((p[0], p[3]), size=1, color=Colors.BLUE))
        # --- 45-degree elevation circle
        r45 = self.R * cos(QUART_PI)
        elev_circle = Drawing.get_arc(r45, TWO_PI) + (0, 0, r45)
        sc_base.add(Polyline(elev_circle, size=1, color=Colors.GREEN))
        # --- The upper & lower limits of base flight envelope. Nominally these are 0.30m above and below the equator.

        # --- The edge outline of the flight sphere as seen from camera's perspective.

        # --- Reference points and markers (fixed in world space)

        # =============== END OF base scene ===================================

        # =============== Figures =============================================
        # Loop
        sc_loop = self._init_loop()
        # Square loop
        sc_sq_loop = self._init_square_loop()
        # Triangular loop
        sc_tri_loop = self._init_tri_loop()
        # Horizontal eight
        sc_hor_eight = self._init_hor_eight()
        # Horizontal square eight
        sc_sq_eight = self._init_sq_eight()
        # Vertical eight
        sc_ver_eight = self._init_ver_eight()
        # Hourglass
        sc_hourglass = self._init_hourglass()
        # Overhead eight
        sc_ovr_eight = self._init_ovr_eight()
        # Clover
        sc_clover = self._init_clover()
        # =============== END OF Figures ======================================

        # Put them all together
        self._scenes = {
            'base': sc_base,
            FigureTypes.INSIDE_LOOPS: sc_loop,
            FigureTypes.INSIDE_SQUARE_LOOPS: sc_sq_loop,
            FigureTypes.INSIDE_TRIANGULAR_LOOPS: sc_tri_loop,
            FigureTypes.HORIZONTAL_EIGHTS: sc_hor_eight,
            FigureTypes.HORIZONTAL_SQUARE_EIGHTS: sc_sq_eight,
            FigureTypes.VERTICAL_EIGHTS: sc_ver_eight,
            FigureTypes.HOURGLASS: sc_hourglass,
            FigureTypes.OVERHEAD_EIGHTS: sc_ovr_eight,
            FigureTypes.FOUR_LEAF_CLOVER: sc_clover
        }

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
        logger.info(f'Sphere center: {self.center}')

    @staticmethod
    def get_arc(r, alpha, rho=100):
        '''Return 3D points for an arc of radius `r` and included angle `alpha`
        with point density `rho`, where `rho` is number of points per 2*pi.
        Arc center is (0, 0, 0).  The arc lies in the XY plane.
        Arc starts at zero angle, i.e., at (r, 0, 0) coordinate, and ends CCW at `alpha`.
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
                (r*cos(act_step*t),
                 r*sin(act_step*t),
                 0.0)
                for t in range(num_pts + 1)
            ]
        )
        return pts

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

    def draw(self, img, azimuth_delta=0.):
        '''Draw all relevant geometry in the given image frame.'''
        if self._cam.Located:
            self._draw_located_geometry(img, azimuth_delta)
        self._draw_track(img)

    def _draw_located_geometry(self, img, azimuth_delta=0.):
        '''Draw the geometry that is relevant when the camera is located.'''
        center = tuple(self.center)
        # Display flight radius at all times
        cv2.putText(
            img,
            f'R={self.R:.2f}',
            (30, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.GREEN, 2)
        # Display world offset of sphere when it is not at world origin
        if not self._is_center_at_origin:
            cv2.putText(
                img,
                f'C=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})',
                (120, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.GREEN, 2)
        rvec = self._cam.rvec
        tvec = self._cam.tvec
        newcameramtx = self._cam.newcameramtx
        dist_zero = self._cam.dist_zero
        # Draw the base scene
        self._scenes.get('base', DEFAULT_SCENE).draw(
            img,
            (azimuth_delta, center),
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=newcameramtx,
            distCoeffs=dist_zero
        )
        # Draw selected figures
        for ftype, fflag in self.figure_state.items():
            if fflag:
                # Draw the scene that is defined for this figure type
                self._scenes.get(ftype, DEFAULT_SCENE).draw(
                    img,
                    (azimuth_delta, center),
                    rvec=rvec,
                    tvec=tvec,
                    cameraMatrix=newcameramtx,
                    distCoeffs=dist_zero
                )

    # def _draw_all_geometry(self, img, azimuth_delta=0):
    #     '''Draw all AR geometry according to the current location and rotation of AR sphere.'''
    #     # Local names to minimize dot-name lookups
    #     rvec = self._cam.rvec
    #     tvec = self._cam.tvec
    #     newcameramtx = self._cam.newcameramtx
    #     distZero = np.zeros_like(self._cam.dist)
    #     cam_pos = self._cam.cam_pos
    #     r = self.R
    #     center = self.center
    #     # Begin drawing
    #     if not self._is_center_at_origin:
    #         cv2.putText(
    #             img,
    #             f'C=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})',
    #             (120, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     cv2.putText(
    #         img,
    #         f'R={r:.2f}',
    #         (30, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     Drawing._draw_level(img, rvec, tvec, newcameramtx, distZero, center, r)
    #     Drawing._draw_all_merid(img, rvec, tvec, newcameramtx, distZero, center, r, azimuth_delta)
    #     if self.axis:
    #         Drawing._draw_axis(img, rvec, tvec, newcameramtx, distZero, center)
    #     Drawing._draw_45(img, rvec, tvec, newcameramtx, distZero, center, r, color=(0, 255, 0))
    #     Drawing._draw_base_tol(img, rvec, tvec, newcameramtx, distZero, center, r)
    #     Drawing._draw_edge(img, cam_pos, rvec, tvec, newcameramtx, distZero, center, r)
    #     Drawing._draw_points(img, self._cam, rvec, tvec, newcameramtx, distZero, center)
    #     return img

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

    # @staticmethod
    # def drawHorEight(img, angle, rvec, tvec, cameramtx, dist, center, r, color=(255, 255, 255)):
    #     Drawing.draw_loop(img, angle+24.47, rvec, tvec, cameramtx,
    #                       dist, center, r, color=(255, 255, 255))
    #     Drawing.draw_loop(img, angle-24.47, rvec, tvec, cameramtx,
    #                       dist, center, r, color=(255, 255, 255))
    #     return img

    def _init_loop(self):
        points, n = self._get_loop_pts()
        border_color = (100, 100, 100)
        result = Scene()
        # Wide white band with narrom gray outline
        result.add(Polyline(points[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points[n:2*n], size=1, color=border_color))
        result.add(Polyline(points[2*n:], size=1, color=border_color))
        return result

    def _get_loop_pts(self):
        '''Helper method. Returns the points for a basic loop.'''
        # Loop radius
        rLoop = self.R * sin(EIGHTH_PI)
        # Loop template
        points = self.get_arc(rLoop, TWO_PI)
        n = points.shape[0]
        # Rotate template to XZ plane
        rot = ROT.from_euler('x', HALF_PI+EIGHTH_PI)
        points = rot.apply(np.vstack([
            points,             # main body of loop
            points * 0.99,      # inner border
            points * 1.01       # outer border
        ]))
        # Translate template to sphere surface
        d = sqrt(self.R**2 - rLoop**2)
        points += [0., d*cos(EIGHTH_PI), d*sin(EIGHTH_PI)]
        return points, n

    def _init_square_loop(self):
        r = self.R
        # Radius of minor arc at 45deg elevation. Also the height of the 45deg elevation arc.
        r45 = r * 0.7071067811865476
        # Helper arc for the bottom flat and both laterals
        major_arc = Drawing.get_arc(r, 0.25*math.pi)
        # Helper arc for the top flat
        minor_arc = Drawing.get_arc(r45, 0.25*math.pi)
        result = Scene(Polyline(np.vstack([
            # Arc 1: bottom flat
            ROT.from_euler(
                'z', 0.375*math.pi).apply(major_arc),
            # Arc 2: left lateral
            ROT.from_euler(
                'xz', [0.5*math.pi, 0.625*math.pi]).apply(major_arc),
            # Arc 3: top flat
            ROT.from_euler(
                'zy', [0.375*math.pi, math.pi]).apply(minor_arc) + np.array((0., 0., r45)),
            # Arc 4: right lateral
            ROT.from_euler(
                'xyz', [-0.5*math.pi, -0.25*math.pi, 0.375*math.pi]).apply(major_arc)
        ]), size=3, color=Colors.WHITE))
        return result

    def _init_tri_loop(self):
        # TODO
        result = Scene()
        return result

    def _init_hor_eight(self):
        points, n = self._get_loop_pts()
        points_right = ROT.from_euler('z', -24.47, degrees=True).apply(points)
        points_left = ROT.from_euler('z', 24.47, degrees=True).apply(points)
        border_color = (100, 100, 100)
        result = Scene()
        result.add(Polyline(points_right[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_right[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_right[2*n:], size=1, color=border_color))
        result.add(Polyline(points_left[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_left[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_left[2*n:], size=1, color=border_color))
        return result

    def _init_sq_eight(self):
        # TODO
        result = Scene()
        return result

    def _init_ver_eight(self):
        points_bot, n = self._get_loop_pts()
        points_top = ROT.from_euler('x', QUART_PI).apply(points_bot)
        border_color = (100, 100, 100)
        result = Scene()
        result.add(Polyline(points_bot[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_bot[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_bot[2*n:], size=1, color=border_color))
        result.add(Polyline(points_top[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_top[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_top[2*n:], size=1, color=border_color))
        return result

    def _init_hourglass(self):
        # TODO
        result = Scene()
        return result

    def _init_ovr_eight(self):
        points, n = self._get_loop_pts()
        points_right = ROT.from_euler('xy', (HALF_PI-EIGHTH_PI, EIGHTH_PI)).apply(points)
        points_left = ROT.from_euler('y', -QUART_PI).apply(points_right)
        border_color = (100, 100, 100)
        result = Scene()
        result.add(Polyline(points_right[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_right[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_right[2*n:], size=1, color=border_color))
        result.add(Polyline(points_left[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_left[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_left[2*n:], size=1, color=border_color))
        return result

    def _init_clover(self):
        # TODO
        result = Scene()
        return result
