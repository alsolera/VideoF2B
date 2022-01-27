# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2018  Alberto Solera Rico - videof2b.dev@gmail.com
# Copyright (C) 2020 - 2022  Andrey Vasilik - basil96
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
from math import asin, atan, atan2, cos, pi, sin, sqrt, tan

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as ROT
from videof2b.core import common
from videof2b.core import geometry as geom
from videof2b.core.camera import CalCamera
from videof2b.core.common import (DEFAULT_TURN_RADIUS, EIGHTH_PI, HALF_PI,
                                  QUART_PI, TWO_PI, FigureTypes)
from videof2b.core.flight import Flight

log = logging.getLogger(__name__)


class Colors:
    '''Shortcuts for OpenCV-compatible colors.'''
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY20 = (50, 50, 50)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)


class Plot:
    '''Base class for plotting primitives.
    * Call `draw()` to draw the Plot instance in your image.
    kwargs:
        size: the line thickness or point radius.
        color: the color of the primitives in this Plot.
        is_fixed: bool indicating whether this Plot is fixed in object space or not.
                  If True, world transforms do not affect the object coordinates.
                  If False (default), then world transforms will
                    rotate, scale, and translate the object coordinates according
                    to the rules in the `_calculate()` method.
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

    def _calculate(self, key, rvec, tvec, camera_matrix, dist_coeffs):
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
            tmp, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
            self._cache[key] = tmp.astype(int).reshape(-1, 2)
        self._img_pts = self._cache[key]

    def draw(self, key, **kwargs):
        '''Call this method from derived classes before drawing the image points.'''
        rvec = kwargs.pop('rvec')
        tvec = kwargs.pop('tvec')
        camera_matrix = kwargs.pop('cameraMatrix')
        dist_coeffs = kwargs.pop('distCoeffs')
        self._calculate(key, rvec, tvec, camera_matrix, dist_coeffs)


class Scatter(Plot):
    '''Defines a collection of scattered points.'''

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

    def draw(self, img, key, **kwargs):
        '''Draw this polyline using its attributes.'''
        super().draw(key, **kwargs)
        for i in range(1, self._img_pts.shape[0]):
            img = cv2.line(img,
                           tuple(self._img_pts[i]),
                           tuple(self._img_pts[i-1]),
                           self._color, self._size)
        return img


class DashedPolyline(Plot):
    '''Defines a polyline that is drawn dashed.'''

    def draw(self, img, key, **kwargs):
        '''Draw this polyline using its attributes.'''
        super().draw(key, **kwargs)
        # Draw dashed lines
        num_on = 3
        num_off = 2
        counter = 0
        is_visible = True
        for i in range(1, self._img_pts.shape[0]):
            counter += 1
            if is_visible:
                img = cv2.line(img,
                               tuple(self._img_pts[i]),
                               tuple(self._img_pts[i-1]),
                               self._color, self._size)
                if counter == num_on:
                    counter = 0
                    is_visible = False
            else:
                if counter == num_off:
                    counter = 0
                    is_visible = True
        return img


class EdgePolyline(Polyline):
    '''Defines a special polyline that represents the visible edge of the sphere.
    This polyline is aware of the camera's position.'''

    def __init__(self, R, cam_pos, **kwargs):
        super().__init__(None, **kwargs)
        self.R = R
        self.cam_pos = cam_pos

    def _calculate(self, key, rvec, tvec, camera_matrix, dist_coeffs):
        '''This class must provide a custom `_calculate()` method
        because the calculation is more complex than that for ordinary world points.'''
        # The visible edge is independent of sphere rotation.
        # Hence this calculation is keyed on translation only.
        _, translation = key
        key = translation
        if key not in self._cache:
            R = self.R
            # Vector from current sphere center to cam
            p = self.cam_pos.reshape((3,)) - translation
            # Length of p
            d = np.linalg.norm(p)
            if d <= R or d < 0.001:
                # We are on the surface of the sphere, or inside its volume.
                # Also guard against DBZ errors.
                return
            d_inv = 1. / d
            # Unit vector of p: points from sphere center back to cam.
            # This is the normal of the "visible edge" arc.
            n = p * d_inv
            # Azimuth angle of the normal.
            # Accounts correctly for the sign (CCW positive around +Z from +X axis)
            phi = atan2(n[1], n[0])
            # Choose two linearly independent arbitrary vectors `u` and `v` that span the arc.
            # `u` is chosen such that it is perpendicular to `n` and lies in the world XY plane.
            u = np.array((-sin(phi), cos(phi), 0.))
            # `v` follows from `n` and `u`.
            v = np.cross(n, u)
            # Helper quantity
            R_sq = R * R
            # Vector from sphere center to center of the "visible edge" arc.
            c = (R_sq * d_inv) * n + translation
            # Determinant for calculating the arc's radius
            det = d*d - R_sq
            if det < 0.:
                # guard against negative arg of sqrt
                r = 0.0
            else:
                r = R * d_inv * np.sqrt(det)
            # `u`, `v`, and `n` form the basis vectors for the arc.
            # Arrange them as columns in a matrix to orient the arc.
            # Add `c` to place its center correctly.
            # Also, this is always a 180-degree arc, and 35 points (rho=70) looks smooth enough.
            obj_pts = ROT.from_matrix(np.array([u, v, n]).T).apply(
                geom.get_arc(r, pi, rho=70)) + c
            tmp, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
            self._cache[key] = tmp.astype(int).reshape(-1, 2)
        self._img_pts = self._cache[key]


class Scene:
    '''A scene consists of a collection of Plot-like objects.'''

    def __init__(self, *items):
        self._items = list(items)
        self._diags = []
        self._diags_on = False

    @property
    def diags_on(self):
        '''Boolean flag that controls drawing of diagnostics.'''
        return self._diags_on

    @diags_on.setter
    def diags_on(self, value):
        self._diags_on = value

    def add(self, item):
        '''Add an item to this scene.'''
        self._items.append(item)

    def add_diag(self, item):
        '''Add a diagnostic item to this scene.'''
        self._diags.append(item)

    def draw(self, img, key, **kwargs):
        '''Draw this scene in the given image.'''
        for item in self._items:
            img = item.draw(img, key, **kwargs)
        if self._diags_on:
            for diag in self._diags:
                img = diag.draw(img, key, **kwargs)
        return img


class DummyScene:
    '''Placeholder object for an empty scene.'''

    def draw(self, *args, **kwargs):
        '''No-op method.'''


DEFAULT_SCENE = DummyScene()


class Drawing:
    '''Container that performs all the drawing
    of AR sphere, track, figures, etc., in any given image frame.'''
    # Default point density per 2*pi (360 degrees) of arc.
    DEFAULT_N = 100

    def __init__(self, detector, **kwargs):
        '''Initialize the Drawing artist instance.
        If only track drawing is required, provide a Detector instance.
        If drawing AR geometry is also required, supply the following kwargs:
            `cam`: instance of CalCamera.
            `flight`: instance of Flight.
            `center`: 3-tuple or ndarray of (x, y, z) location of drawn sphere
                      with respect to the world center defined by markers.
                      Default is `videof2b.core.common.DEFAULT_CENTER`, which is the origin (0, 0, 0).
            Optional:
                `axis`: True to draw vertical axis, False otherwise.
                `point_density`: number of arc points per full circle.
                                 Default is Drawing.DEFAULT_N, which is 100.
                `turn_r`: the radius of turns in all figures, in meters.
        '''
        # Defines the visibility state of all drawn figures.
        self.figure_state = defaultdict(bool)
        # Indicates whether the sphere center is sufficiently close to world origin.
        self._is_center_at_origin = False
        # Instance of motion detector.
        self._detector = detector
        # Instance of the camera.
        self._cam: CalCamera = kwargs.pop('cam', None)
        # Instance of the recorded Flight.
        flight: Flight = kwargs.pop('flight', None)
        # Internal flag. Indicates whether located geometry can be drawn.
        self._is_located = False
        # Current azimuth of the AR sphere. Zero points "North", or from sphere center to front marker.
        self._azimuth = 0.
        # Radius of flight hemisphere, in m.
        self.R = None
        # Radius of the circle in which the markers are located, in m.
        self.marker_radius = None
        # Coordinates of flight sphere relative to world coordinates, in m.
        self.center = None
        # Indicates whether we draw the axis (CSys marker at center)
        self.axis = kwargs.pop('axis', False)
        # Number of points per 360 degrees of any arc.
        self._point_density = kwargs.pop('point_density', Drawing.DEFAULT_N)
        # Defines the turn radius of all applicable figures.
        self.turn_r = kwargs.pop('turn_r', DEFAULT_TURN_RADIUS)
        # Indicates whether to draw diagnostics or not.
        self._draw_diags = False
        # Collection of all necessary scenes, keyed on type.
        self._scenes = {}
        self.locate(self._cam, flight, **kwargs)

    @property
    def draw_diags(self):
        '''Controls the drawing of diagnostics.'''
        return self._draw_diags

    @draw_diags.setter
    def draw_diags(self, val):
        if val == self._draw_diags:
            # Don't waste our time
            return
        self._draw_diags = val
        self._sync_scene_diags()

    def locate(self, cam: CalCamera, flight: Flight = None, **kwargs) -> None:
        '''Locate a new Flight or relocate an existing one using the given camera.'''
        log.debug('Entering Drawing.locate()')
        self._cam = cam
        self.figure_state['base'] = False
        self._is_located = cam is not None and flight is not None and flight.is_located
        if self._is_located:
            self.R = flight.flight_radius
            self.marker_radius = flight.marker_radius
            self.marker_height = flight.marker_height
            center = kwargs.pop('center', common.DEFAULT_CENTER.copy())
            # Ensure it's a numpy array (allow tuple, list as input).
            # TODO: maybe it is safer to use `np.atleast_1d(center)` ?
            self.center = np.float32(center)
            self._evaluate_center()
            self.figure_state['base'] = True
            self._init_scenes()

    def _init_scenes(self):
        # When the cam is located, we define lots of reference geometry:
        # ============ Base scene: main hemisphere and markers ==========
        sc_base = self._get_base_scene()
        # =============== Figures =======================================
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
        # =============== END OF Figures ================================
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
            FigureTypes.FOUR_LEAF_CLOVER: sc_clover,
        }
        self._sync_scene_diags()

    def _sync_scene_diags(self):
        '''Sync the scene diagnostics flag with our flag.'''
        for scene in self._scenes.values():
            scene.diags_on = self._draw_diags

    def _get_base_scene(self):
        '''Base scene: main hemisphere and markers.'''
        result = Scene()
        # --- The base level (aka the equator) of flight hemisphere.
        result.add(Polyline(geom.get_arc(self.R, TWO_PI, rho=self._point_density),
                            size=1, color=Colors.WHITE))
        # --- All the meridians of the flight hemisphere.
        # rot for initial meridian in YZ plane
        rot0 = ROT.from_euler('xz', [HALF_PI, HALF_PI])
        for angle in range(0, 180, 45):
            gray = 255 - 2*angle
            color = (gray, gray, gray)
            # rot for orienting each meridian
            rot1 = ROT.from_euler('z', angle, degrees=True)
            pts = rot1.apply(rot0.apply(geom.get_arc(self.R, pi, rho=self._point_density)))
            result.add(Polyline(pts, size=1, color=color))
        # --- The coordinate axis reference at sphere center
        if self.axis:
            # XYZ sphere axes
            p = np.array([
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 5]
            ], dtype=np.float32)
            result.add(Polyline((p[0], p[1]), size=1, color=Colors.RED))
            result.add(Polyline((p[0], p[2]), size=1, color=Colors.GREEN))
            result.add(Polyline((p[0], p[3]), size=1, color=Colors.BLUE))
        # --- 45-degree elevation circle
        r45 = self.R * cos(QUART_PI)
        elev_circle = geom.get_arc(r45, TWO_PI, rho=self._point_density) + (0, 0, r45)
        result.add(Polyline(elev_circle, size=1, color=Colors.GREEN))
        # --- The upper & lower limits of base flight envelope.
        # Nominally these are 0.30m above and below the equator.
        tol = 0.3
        # Radius of the tolerance circles is equivalent to the axis height of a cone whose radius is `tol`.
        r_tol = geom.get_cone_d(self.R, tol)
        # Use 2X point density to account for dashes
        tol_pts = geom.get_arc(r_tol, TWO_PI, rho=2*self._point_density)
        pts_lower = tol_pts - [0., 0., tol]
        pts_upper = tol_pts + [0., 0., tol]
        color = (214, 214, 214)
        result.add(DashedPolyline(pts_lower, size=1, color=color))
        result.add(DashedPolyline(pts_upper, size=1, color=color))
        # --- The edge outline of the flight sphere as seen from camera's perspective.
        result.add(EdgePolyline(self.R, self._cam.cam_pos, size=1, color=Colors.MAGENTA))
        # --- Reference points and markers (fixed in world space)
        r45 = self.marker_radius * cos(QUART_PI)
        marker_size_x = 0.20  # marker width, in m
        marker_size_z = 0.60  # marker height, in m
        # Sphere reference points (fixed in object space)
        pts_ref = Scatter(
            np.array([
                # --- Points on sphere centerline:
                # sphere center
                [0, 0, 0],
                # pilot's feet
                [0, 0, -self.marker_height],
                # top of sphere
                [0, 0, self.R],
                # --- Points on equator:
                # bottom of right & left marker
                [r45, r45, 0],
                [-r45, r45, 0],
                # right & left antipodes
                [self.marker_radius, 0, 0],
                [-self.marker_radius, 0, 0],
                # front & rear antipodes
                [0, -self.marker_radius, 0],
                [0, self.marker_radius, 0]]),
            size=1, color=Colors.RED, is_fixed=True)
        # Points on corners of an imaginary marker at center of sphere (optional)
        mrk_center = Scatter(
            np.array([[0.5 * marker_size_x, 0., 0.5 * marker_size_z],
                      [-0.5 * marker_size_x, 0., 0.5 * marker_size_z],
                      [-0.5 * marker_size_x, 0., -0.5 * marker_size_z],
                      [0.5 * marker_size_x, 0., -0.5 * marker_size_z]]),
            size=1, color=Colors.CYAN)
        # Perimeter markers (fixed in object space)
        mrk_perimeter = Scatter(
            np.array([
                # Points on corners of front marker
                [0.5 * marker_size_x, self.marker_radius, 0.5 * marker_size_z],
                [-0.5 * marker_size_x, self.marker_radius, 0.5 * marker_size_z],
                [-0.5 * marker_size_x, self.marker_radius, -0.5 * marker_size_z],
                [0.5 * marker_size_x, self.marker_radius, -0.5 * marker_size_z],
                # Points on corners of right marker
                [r45 + 0.5 * marker_size_x, r45, 0.5 * marker_size_z],
                [r45 - 0.5 * marker_size_x, r45, 0.5 * marker_size_z],
                [r45 - 0.5 * marker_size_x, r45, -0.5 * marker_size_z],
                [r45 + 0.5 * marker_size_x, r45, -0.5 * marker_size_z],
                # Points on corners of left marker
                [-r45 + 0.5 * marker_size_x, r45, 0.5 * marker_size_z],
                [-r45 - 0.5 * marker_size_x, r45, 0.5 * marker_size_z],
                [-r45 - 0.5 * marker_size_x, r45, -0.5 * marker_size_z],
                [-r45 + 0.5 * marker_size_x, r45, -0.5 * marker_size_z],
            ]),
            size=1, color=Colors.GREEN, is_fixed=True)
        result.add(pts_ref)
        result.add(mrk_center)
        result.add(mrk_perimeter)
        return result

    def set_azimuth(self, azimuth):
        '''Set the aximuth of the AR sphere, in degrees.'''
        self._azimuth = azimuth

    def move_center_x(self, delta):
        '''Move sphere center by `delta` along world X direction, in meters.'''
        self.center[0] += delta
        self._evaluate_center()

    def move_center_y(self, delta):
        '''Move sphere center by `delta` along world Y direction, in meters.'''
        self.center[1] += delta
        self._evaluate_center()

    def reset_center(self):
        '''Reset sphere center to default.'''
        self.center = common.DEFAULT_CENTER.copy()
        self._evaluate_center()

    def _evaluate_center(self):
        self.center = self.center.round(1)
        self._is_center_at_origin = (abs(self.center) < 0.01).all()
        log.info(f'Sphere center: {self.center}')

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

    def draw(self, img):
        '''Draw all relevant geometry in the given image frame.'''
        if self._is_located:
            self._draw_located_geometry(img)
        self._draw_track(img)

    def _draw_located_geometry(self, img):
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
        # Draw selected figures
        for ftype, fflag in self.figure_state.items():
            if fflag:
                # Draw the scene that is defined for this figure type
                self._scenes.get(ftype, DEFAULT_SCENE).draw(
                    img,
                    (self._azimuth, center),
                    rvec=rvec,
                    tvec=tvec,
                    cameraMatrix=newcameramtx,
                    distCoeffs=dist_zero
                )

    def _get_base_loop_pts(self, half_angle=None, cw=False):
        '''Helper method for generating a basic loop of specified cone half-angle on the sphere.
        If the angle is not specified, a basic 45-degree loop is generated.
        Start & end points are at the bottom.
        If `cw` is True, generate a clockwise loop. Default is counterclockwise.'''
        # Loop radius: default is a basic 45-deg loop
        if half_angle is None:
            half_angle = EIGHTH_PI
        r_loop = self.R * sin(half_angle)
        # Loop template
        points = geom.get_arc(r_loop, TWO_PI, rho=self._point_density)
        if cw:
            points = ROT.from_euler('x', pi).apply(points)
        # Rotate template to XZ plane
        points = ROT.from_euler('zx', [-HALF_PI, HALF_PI]).apply(points)
        # Translate template to sphere surface
        d = geom.get_cone_d(self.R, r_loop)
        points += [0., d, 0.]
        return points

    @staticmethod
    def _add_diag_connections(scene, pieces):
        '''This results in "donuts" of green/red color at each connection.
        If points at a connection do not meet, then that donut's ID/OD
        will not appear concentric. Use for visual diagnostics.'''
        # Grab first and last point of each piece
        conns = np.vstack([(egg[0], egg[-1]) for egg in pieces])
        if len(pieces) == 1:
            scene.add_diag(Scatter(conns[1], size=7, color=Colors.GREEN))
            scene.add_diag(Scatter(conns[0], size=3, color=Colors.RED))
        else:
            # Connection points at traversals: direction is red -> green
            scene.add_diag(Scatter(conns[2::4], size=7, color=Colors.RED))
            scene.add_diag(Scatter(conns[3::4], size=7, color=Colors.GREEN))
            # Connection points at corners: direction is red -> green
            scene.add_diag(Scatter(conns[0::4], size=3, color=Colors.RED))
        if len(pieces) > 2:
            scene.add_diag(Scatter(conns[1::4], size=3, color=Colors.GREEN))

    @staticmethod
    def _get_figure_scene(pieces):
        '''Helper method for creating any figure from the given contiguous pieces.'''
        course = np.vstack(pieces)
        result = Scene(
            Polyline(course, size=7, color=Colors.GRAY20),  # outline border
            Polyline(course, size=3, color=Colors.WHITE)
        )
        # Diagnostics: tangency points
        Drawing._add_diag_connections(result, pieces)
        return result

    def _init_loop(self):
        '''Initialize the scene of the basic loop.'''
        return Drawing._get_figure_scene(self._get_loop_pts())

    def _get_loop_pts(self):
        '''Helper method for generating the basic loop.'''
        points = ROT.from_euler('x', EIGHTH_PI).apply(self._get_base_loop_pts())
        return (points,)

    def _init_square_loop(self):
        '''Initialize the scene of the square loop.'''
        return Drawing._get_figure_scene(self._get_square_loop_pts())

    def _get_square_loop_pts(self):
        '''Helper method for generating the square loop and square eight.'''
        # This is the most complicated of all F2B figures due to lack of overall symmetry.
        # There is only symmetry about the vertical plane.
        #
        # Corner radius in m
        r = self.turn_r
        # Half-angle of apex of all corner cones
        alpha = geom.get_cone_alpha(self.R, r)
        # Distance from sphere center to centers of all corners
        d = geom.get_cone_d(self.R, r)
        # ---- Parameters of bottom corners
        # Fillet representing the bottom corners
        f = geom.Fillet(self.R, r, HALF_PI)
        # Template arc for bottom corners
        bot_corner_arc = geom.get_arc(r, f.beta, rho=27)
        bot_corner = ROT.from_euler('zxy', [0.5*(pi - f.beta), -HALF_PI, HALF_PI]
                                    ).apply(bot_corner_arc) + [0, f.d, 0]
        # ---- Parameters of top corners
        # Elevation of centers of the top corners
        theta = QUART_PI - alpha
        # `delta`: Helper angle for determining the included angle of top corners
        # `omega`: Rotation of corner base cone from equator around x-axis
        #          by `omega` such that elevation of C_r = theta
        delta, omega = geom.get_cone_delta(alpha, theta=theta)
        # Included angle of top corners
        beta = HALF_PI - delta
        # Template arc for top corners
        top_corner_arc = geom.get_arc(r, beta, rho=27)
        # Center it on y-axis at the equator as a start
        top_corner = ROT.from_euler('xz', [HALF_PI, pi]).apply(top_corner_arc) + [0., d, 0.]
        # Radius of minor arc at 45deg elevation. Also the height of same arc above equator.
        R45 = self.R * 0.5 * sqrt(2.0)
        # ---- Corners
        corner1 = ROT.from_rotvec(-QUART_PI*np.array([-sin(EIGHTH_PI), cos(EIGHTH_PI), 0.])).apply(
            ROT.from_euler('z', EIGHTH_PI-f.theta).apply(bot_corner)
        )
        corner2 = ROT.from_euler('xz', [omega, EIGHTH_PI]).apply(
            ROT.from_euler('z', -alpha).apply(top_corner)
        )
        # Corner 3: mirror of Corner 2 about YZ plane, but still CW direction
        corner3 = ROT.from_euler('xz', [omega, -EIGHTH_PI]).apply(
            ROT.from_euler('yz', [HALF_PI+delta, alpha]).apply(top_corner)
        )
        # Corner 4
        corner4 = ROT.from_rotvec(QUART_PI*np.array([sin(EIGHTH_PI), cos(EIGHTH_PI), 0.])).apply(
            ROT.from_euler('yz', [pi, -(EIGHTH_PI-f.theta)]).apply(bot_corner)
        )
        # ---- Parameters of connecting arcs
        # Helper points to determine the central angle of each connecting arc
        p0 = [0., 0., R45]
        p1 = corner2[-1]
        p2 = corner3[0]
        # Central angle of top arc (arc 2)
        arc2_angle = geom.angle(p1-p0, p2-p0)
        p0 = [0., 0., 0.]
        p1 = corner1[-1]
        p2 = corner2[0]
        # While we're here: calculate elevation of the lower tangency point of the laterals
        lateral_elev = geom.cartesian_to_spherical(p1)[1]
        # Central angle of ascending & descending laterals (arcs 1 & 3)
        arc13_angle = geom.angle(p1-p0, p2-p0)
        p1 = corner4[-1]
        p2 = corner1[0]
        # Central angle of bottom arc (arc 4)
        arc4_angle = geom.angle(p1-p0, p2-p0)
        # Template arc for the top flat
        top_arc = geom.get_arc(R45, arc2_angle, rho=self._point_density)
        # Template arc for the bottom flat
        arc4 = geom.get_arc(self.R, arc4_angle, rho=self._point_density)
        # Template arc for the laterals
        arc13 = geom.get_arc(self.R, arc13_angle, rho=self._point_density)
        # ---- All the points
        points = (
            # Corner 1
            corner1,
            # Arc 1: ascending (left) lateral
            ROT.from_euler('xz', [HALF_PI, 0.625*pi]).apply(
                ROT.from_euler('z', lateral_elev).apply(arc13)
            ),
            # Corner 2
            corner2,
            # Arc2: top flat
            ROT.from_euler('zy', [0.5*(pi-arc2_angle), pi]).apply(top_arc) + [0., 0., R45],
            # Corner 3
            corner3,
            # Arc 3: descending (right) lateral
            ROT.from_euler(
                'xyz', [-HALF_PI, -(arc13_angle+lateral_elev), 0.375*math.pi]).apply(arc13),
            # Corner 4
            corner4,
            # Arc4: bottom flat
            ROT.from_euler('z', 0.5*(pi - arc4_angle)).apply(arc4)
        )
        return points

    def _init_tri_loop(self):
        '''Initialize the scene of the triangular loop.'''
        return Drawing._get_figure_scene(self._get_tri_loop_pts())

    def _get_tri_loop_pts(self):
        '''Helper method for generating the triangular loop.'''
        # Corner radius
        r = self.turn_r
        # Central angle of each arc and angle between adjacent arcs
        sigma, phi = geom.calc_tri_loop_params(self.R, r)
        # Height of the full equilateral triangle (without the radii in corners)
        h = geom.get_equilateral_height(sigma)
        # The fillet that represents each corner with radius `r`
        f = geom.Fillet(self.R, r, phi)
        # Template arc for corners
        corner_pts = geom.get_arc(r, f.beta, rho=27)
        # Corners: template arc in the middle of the bottom leg
        corner_pts = ROT.from_euler('zxy', [0.5*(pi - f.beta), -HALF_PI, HALF_PI]
                                    ).apply(corner_pts) + [0, f.d, 0]
        # 1st corner arc
        corner1 = ROT.from_rotvec(-0.5*phi*np.array([-sin(0.5*sigma), cos(0.5*sigma), 0.])).apply(
            ROT.from_euler('z', 0.5*sigma-f.theta).apply(corner_pts)
        )
        # Top corner arc
        corner2 = ROT.from_euler('x', h - f.theta).apply(
            ROT.from_euler('y', HALF_PI).apply(corner_pts)
        )
        # Last corner arc
        corner3 = ROT.from_rotvec(0.5*phi*np.array([sin(0.5*sigma), cos(0.5*sigma), 0.])).apply(
            ROT.from_euler('yz', [pi, -(0.5*sigma-f.theta)]).apply(corner_pts)
        )
        # Main arcs are actually this long to meet tangency
        leg_sigma = geom.angle(corner3[-1], corner1[0])
        # Legs: template arc
        leg_points = geom.get_arc(self.R, leg_sigma, rho=self._point_density)
        leg_points = ROT.from_euler('z', 0.5*(pi - leg_sigma)).apply(leg_points)
        # Helper values for construction of the corner axes
        caxis_s = sin(0.5*sigma)
        caxis_c = cos(0.5*sigma)
        points = (
            # first corner
            corner1,
            # ascending leg
            ROT.from_rotvec(-phi*np.array([-caxis_s, caxis_c, 0.])).apply(
                ROT.from_euler('y', pi).apply(leg_points)
            ),
            # top corner
            corner2,
            # descending leg
            ROT.from_rotvec(phi*np.array([caxis_s, caxis_c, 0.])).apply(
                ROT.from_euler('y', pi).apply(leg_points)
            ),
            # last corner
            corner3,
            # bottom leg
            leg_points
        )
        return points

    def _init_hor_eight(self):
        '''Initialize the scene of the horizontal eight.'''
        return Drawing._get_figure_scene(self._get_hor_eight_pts())

    def _get_hor_eight_pts(self):
        '''Helper method for generating the horizontal eight.'''
        # Start with a 45-deg loop whose axis is the y-axis, start/end at left
        loop_pts = ROT.from_euler('y', HALF_PI).apply(self._get_base_loop_pts())
        # Loop radius and distance
        # r_loop = self.R * sin(EIGHTH_PI)
        # d = geom.get_cone_d(self.R, r_loop)
        # The elevation angle of each loop to satisfy tangency for a horizontal eight.
        theta = asin(tan(EIGHTH_PI))
        points = (
            # Right loop (CCW, starts & ends at tangency point between loops).
            # Should be CW but let's not bother just for drawing.
            ROT.from_euler('zx', [-EIGHTH_PI, theta]).apply(loop_pts),
            # Left loop (CCW, also starts & ends at tangency point between loops)
            ROT.from_euler('yzx', [pi, EIGHTH_PI, theta]).apply(loop_pts)
        )
        return points

    def _init_sq_eight(self):
        '''Initialize the scene of the horizontal square eight.'''
        return Drawing._get_figure_scene(self._get_sq_eight_pts())

    def _get_sq_eight_pts(self):
        '''Helper method for generating the horizontal square eight.'''
        points = self._get_square_loop_pts()
        # For this figure, rearrange the order of the paths a bit:
        # start at center vertical and finish at the bottom left corner.
        points = (*points[1:], points[0])
        points_right = [ROT.from_euler('z', -EIGHTH_PI).apply(arr) for arr in points]
        # Left side is a mirror about the YZ plane
        points_left = [arr*[-1, 1, 1] for arr in points_right]
        return tuple((*points_right, *points_left))

    def _init_ver_eight(self):
        '''Initialize the scene of the vertical eight.'''
        return Drawing._get_figure_scene(self._get_ver_eight_pts())

    def _get_ver_eight_pts(self):
        '''Helper method for generating the vertical eight.'''
        # Base loop: axis on equator, start/end on bottom of loop
        base_pts = self._get_base_loop_pts()
        # Direction of bottom loop here is opposite (ccw), but let's not force it for drawing
        points_bot = ROT.from_euler('yx', [pi, EIGHTH_PI]).apply(base_pts)
        points_top = ROT.from_euler('x', 1.5*QUART_PI).apply(base_pts)
        return (points_bot, points_top)

    def _init_hourglass(self):
        '''Initialize the scene of the hourglass.'''
        return Drawing._get_figure_scene(self._get_hourglass_pts())

    def _get_hourglass_pts(self):
        '''Helper method for generating the hourglass.
        The hourglass is just two identical equilateral spherical triangles mirroring
        each other above/below the 45-degree latitude.  Therefore the angle subtended
        by the ascending & descending arcs is twice the angle subtended by the top and
        bottom arcs of the figure.  After constructing the above, we add the corner
        radii at the bottom and top turns.
        '''
        # Radius of corner turns
        r = self.turn_r
        # Angle subtended by one leg of the equilateral spherical triangle
        sigma = geom.calc_equilateral_sigma()
        # Angle between legs of the triangle
        phi = geom.get_equilateral_phi(sigma)
        # The fillet that represents all four corners of the hourglass
        f = geom.Fillet(self.R, r, phi)
        # Template arc for corners
        corner_pts = geom.get_arc(r, f.beta, rho=27)
        # Bottom corners (1 & 4, CW): template arc in the middle of the bottom leg
        corner_pts_bot = ROT.from_euler('zxy', [0.5*(pi-f.beta), -HALF_PI, HALF_PI]
                                        ).apply(corner_pts) + [0, f.d, 0]
        # Top corners (2 & 3, CCW): template arc in the middle of the top leg
        corner_pts_top = ROT.from_euler('zx', [-HALF_PI+0.5*(pi-f.beta), pi]
                                        ).apply(corner_pts) + [0, 0, f.d]
        half_sigma = 0.5*sigma
        # Corner arc 1
        corner1 = ROT.from_rotvec(-0.5*phi*np.array([-sin(half_sigma), cos(half_sigma), 0.])).apply(
            ROT.from_euler('z', half_sigma-f.theta).apply(corner_pts_bot)
        )
        # Corner arc 2
        corner2 = ROT.from_rotvec(-0.5*phi*np.array([sin(half_sigma), 0., cos(half_sigma)])).apply(
            ROT.from_euler('y', half_sigma-f.theta).apply(corner_pts_top)
        )
        # Corner arc 3
        corner3 = ROT.from_rotvec(0.5*phi*np.array([-sin(half_sigma), 0., cos(half_sigma)])).apply(
            ROT.from_euler('zy', [pi, -(half_sigma-f.theta)]).apply(corner_pts_top)
        )
        # Corner arc 4
        corner4 = ROT.from_rotvec(0.5*phi*np.array([sin(half_sigma), cos(half_sigma), 0.])).apply(
            ROT.from_euler('yz', [pi, -(half_sigma-f.theta)]).apply(corner_pts_bot)
        )
        # Angle of a diagonal arc between tangency points
        diag_sigma = geom.angle(corner1[-1], corner2[0])
        # Template arc for the diagonal arcs
        diag_pts = ROT.from_euler('z', 0.5*(pi-diag_sigma)).apply(
            geom.get_arc(self.R, diag_sigma, rho=self._point_density)
        )
        # Ascending arc
        arc_ascend = ROT.from_rotvec((pi-phi)*np.array([-sin(half_sigma), cos(half_sigma), 0])).apply(
            ROT.from_euler('z', 1.5*sigma).apply(diag_pts)
        )
        # Angle of top/bottom arcs between tangency points
        hor_sigma = geom.angle(corner2[-1], corner3[0])
        # Bottom arc
        arc_bot = ROT.from_euler('z', 0.5*(pi-hor_sigma)
                                 ).apply(geom.get_arc(self.R, hor_sigma, rho=self._point_density))
        # Top arc
        arc_top = ROT.from_euler('x', HALF_PI).apply(arc_bot)
        # Descending arc
        arc_descend = ROT.from_rotvec(phi*np.array([sin(half_sigma), cos(half_sigma), 0])).apply(
            ROT.from_euler('yz', [pi, 0.5*sigma]).apply(diag_pts)
        )
        # The full course
        points = (
            corner1,
            arc_ascend,
            corner2,
            arc_top,
            corner3,
            arc_descend,
            corner4,
            arc_bot
        )
        return points

    def _init_ovr_eight(self):
        '''Initialize the scene of the overhead eight.'''
        return Drawing._get_figure_scene(self._get_ovr_eight_pts())

    def _get_ovr_eight_pts(self):
        '''Helper method for generating the overhead eight.'''
        # Template loop: overhead loop whose axis is Z-axis, start/end on right
        base_pts = ROT.from_euler('yx', [-HALF_PI, HALF_PI]).apply(self._get_base_loop_pts())
        points_right = ROT.from_euler('zy', (pi, EIGHTH_PI)).apply(base_pts)
        points_left = ROT.from_euler('y', -EIGHTH_PI).apply(base_pts)
        return (points_right, points_left)

    def _init_clover(self):
        '''Initialize the scene of the four-leaf clover.'''
        return Drawing._get_figure_scene(self._get_clover_pts())

    def _get_clover_pts(self):
        '''Helper method for generating the four-leaf clover.'''
        # Rotation of base cone to achieve its tangency to equator
        beta = EIGHTH_PI
        # Each cone's half-angle
        alpha = atan(sin(beta))
        # Helper angles
        delta, _ = geom.get_cone_delta(alpha, beta=beta)
        # # Radius of each loop
        r = self.R * sin(alpha)
        # # Distance from sphere center to center of each loop
        d = self.R * cos(alpha)
        d_offset = [0., d, 0.]
        tilt_angle_top = beta + QUART_PI
        # Template arc: almost 3/4 loop
        base_arc = geom.get_arc(r, 1.5*pi-delta, rho=self._point_density)
        # Template for loops 2 & 3, CCW. Axis on y-axis. Start pt at right.
        arc_ccw = ROT.from_euler('x', HALF_PI).apply(base_arc) + d_offset
        # Template for loops 1 & 4, CW. Axis on y-axis. Start pt at left, rotated by `delta`.
        arc_cw = ROT.from_euler('zyx', [delta+HALF_PI, pi, HALF_PI]).apply(base_arc) + d_offset
        # Template arc for the connecting paths: centered azimuthally, lying in the XY plane, CCW
        conn_arc = ROT.from_euler(
            'z', QUART_PI+EIGHTH_PI).apply(geom.get_arc(self.R, QUART_PI, rho=self._point_density))
        # clover loops in the order they are performed
        points = (
            # Connector: Start -> 1 (or Loop 2 -> 3)
            ROT.from_euler('yx', [HALF_PI, QUART_PI]).apply(conn_arc),
            # Loop 1
            ROT.from_euler('yzx', [-delta - HALF_PI, -alpha, tilt_angle_top]).apply(arc_cw),
            # Connector: Loop 1 -> 2
            ROT.from_euler('x', QUART_PI).apply(conn_arc),
            # Loop 2
            ROT.from_euler('yzx', [-delta - HALF_PI, alpha, beta]).apply(arc_ccw),
            # Connector: Loop 2 -> 3
            ROT.from_euler('yx', [HALF_PI, QUART_PI]).apply(conn_arc),
            # Loop 3
            ROT.from_euler('zx', [alpha, tilt_angle_top]).apply(arc_ccw),
            # Connector: Loop 3 -> 4
            ROT.from_euler('yx', [pi, QUART_PI]).apply(conn_arc),
            # Loop 4
            ROT.from_euler('zx', [-alpha, beta]).apply(arc_cw)
        )
        return points
