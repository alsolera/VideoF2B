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
from math import acos, atan, atan2, cos, degrees, pi, radians, sin, sqrt

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


class DashedPolyline(Polyline):
    '''Defines a polyline that is drawn dashed.'''

    def __init__(self, obj_pts, **kwargs):
        super().__init__(obj_pts, **kwargs)

    def draw(self, img, key, **kwargs):
        '''Draw this polyline using its attributes.'''
        super().draw(img, key, **kwargs)
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

    def _calculate(self, key, rvec, tvec, cameraMatrix, distCoeffs):
        '''This class must provide a custom `_calculate()` method
        because the calculation is more complex than that for ordinary world points.'''
        # The visible edge is independent of sphere rotation. Hence this calculation is keyed on translation only.
        _, translation = key
        key = translation
        if key not in self._cache:
            R = self.R
            p = self.cam_pos.reshape((3,)) - translation
            d = np.linalg.norm(p)
            if d <= R:
                # We are on the surface of the sphere, or inside its volume.
                return
            n = p / d
            # TODO: optimize the creation of this arc using ROT or....?
            phi = atan2(n[1], n[0])
            rot_mat = np.array([
                [cos(phi), -sin(phi), 0.],
                [sin(phi), cos(phi), 0.],
                [0., 0., 1.]])
            u = np.array([0., 1., 0.])
            u = rot_mat.dot(u)
            v = np.cross(n, u)
            t = np.linspace(0., 1., 100)
            k = np.pi  # semi-circle
            c = (R**2 / d) * n + translation
            det = d**2 - R**2
            if det < 0.:
                # guard against negative arg of sqrt
                r = 0.0
            else:
                r = R / d * np.sqrt(det)
            obj_pts = np.array([c + r * (np.cos(k * t_i) * u + np.sin(k * t_i) * v) for t_i in t])
            tmp, _ = cv2.projectPoints(obj_pts, rvec, tvec, cameraMatrix, distCoeffs)
            self._cache[key] = tmp.astype(int).reshape(-1, 2)
        self._img_pts = self._cache[key]

    def draw(self, img, key, **kwargs):
        return super().draw(img, key, **kwargs)


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
        tol = 0.3
        r_tol = sqrt(self.R**2 - tol**2)
        tol_pts = Drawing.get_arc(r_tol, TWO_PI, 200)
        pts_lower = tol_pts - [0., 0., tol]
        pts_upper = tol_pts + [0., 0., tol]
        color = (214, 214, 214)
        sc_base.add(DashedPolyline(pts_lower, size=1, color=color))
        sc_base.add(DashedPolyline(pts_upper, size=1, color=color))
        # --- The edge outline of the flight sphere as seen from camera's perspective.
        sc_base.add(EdgePolyline(self.R, self._cam.cam_pos, size=1, color=Colors.MAGENTA))
        # --- Reference points and markers (fixed in world space)
        r45 = self._cam.markRadius * cos(QUART_PI)
        marker_size_x = 0.20  # marker width, in m
        marker_size_z = 0.60  # marker height, in m
        # Sphere reference points (fixed in object space)
        pts_ref = Scatter(
            np.array([
                # Points on sphere centerline: sphere center, pilot's feet, top of sphere.
                [0, 0, 0],
                [0, 0, -self._cam.markHeight],
                [0, 0, self.R],
                # Points on equator: bottom of right & left marker, right & left antipodes, front & rear antipodes
                [r45, r45, 0],
                [-r45, r45, 0],
                [self._cam.markRadius, 0, 0],
                [-self._cam.markRadius, 0, 0],
                [0, -self._cam.markRadius, 0],
                [0, self._cam.markRadius, 0]]),
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
                [0.5 * marker_size_x, self._cam.markRadius, 0.5 * marker_size_z],
                [-0.5 * marker_size_x, self._cam.markRadius, 0.5 * marker_size_z],
                [-0.5 * marker_size_x, self._cam.markRadius, -0.5 * marker_size_z],
                [0.5 * marker_size_x, self._cam.markRadius, -0.5 * marker_size_z],
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
        sc_base.add(pts_ref)
        sc_base.add(mrk_center)
        sc_base.add(mrk_perimeter)
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

    def _init_loop(self):
        points, n = self._get_loop_pts()
        border_color = Colors.GRAY20
        result = Scene()
        # Wide white band with narrom gray outline
        result.add(Polyline(points[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points[n:2*n], size=1, color=border_color))
        result.add(Polyline(points[2*n:], size=1, color=border_color))
        return result

    def _get_loop_pts(self, half_angle=None):
        '''Helper method. Returns the points for a loop of specified cone half-angle on the sphere.
        If the angle is not specified, a basic 45-degree loop is generated.'''
        # Loop radius: default is a basic 45-deg loop
        if half_angle is None:
            half_angle = EIGHTH_PI
        r_loop = self.R * sin(half_angle)
        # Loop template
        points = self.get_arc(r_loop, TWO_PI)
        n = points.shape[0]
        # Rotate template to XZ plane plus loop's tilt
        rot = ROT.from_euler('x', HALF_PI+half_angle)
        points = rot.apply(np.vstack([
            points,             # main body of loop
            points * 0.99,      # inner border
            points * 1.01       # outer border
        ]))
        # Translate template to sphere surface
        d = sqrt(self.R**2 - r_loop**2)
        points += [0., d*cos(half_angle), d*sin(half_angle)]
        return points, n

    def _init_square_loop(self):
        points = self._get_square_loop_pts()
        result = Scene(Polyline(points, size=3, color=Colors.WHITE))
        return result

    def _get_square_loop_pts(self):
        r = self.R
        # Radius of minor arc at 45deg elevation. Also the height of the 45deg elevation arc.
        r45 = r * 0.7071067811865476
        # Helper arc for the bottom flat and both laterals
        major_arc = Drawing.get_arc(r, 0.25*math.pi)
        # Helper arc for the top flat
        minor_arc = Drawing.get_arc(r45, 0.25*math.pi)
        points = np.vstack([
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
        ])
        return points

    def _init_tri_loop(self):
        # corner radius
        r = 1.5
        # central angle (length) of each arc
        # TODO: figure this out as a function of the triangle's height
        s = radians(53.7)
        # angle between adjacent arcs: see https://www.av8n.com/physics/spherical-triangle.htm [Eq. 1]
        alpha = acos(cos(s)/(cos(s)+1.))
        # main arcs are actually this long to meet tangency
        a = radians(40)  # TODO: figure this out
        main_offset = 0.5*(s - a)
        # distance from sphere center to corner centers
        d_corner = sqrt(self.R**2 - r**2)
        # corner arc is short of 180 by this much to meet tangency
        corner_short = alpha  # TODO: determine correct value
        corner_offset = 0.5*a+0.0125    # TODO: determine the correct value
        # template arc for corners
        corner_points = Drawing.get_arc(r, pi-corner_short)
        # first corner arc
        corner1 = ROT.from_euler('z', corner_offset).apply(
            ROT.from_euler('x', r/self.R).apply(
                ROT.from_euler('x', HALF_PI).apply(
                    ROT.from_euler('z', 1.5*pi).apply(
                        ROT.from_euler('x', pi).apply(corner_points)
                    )
                ) + (0., d_corner, 0.)
            )
        )
        # top corner arc
        corner2 = ROT.from_euler('x', QUART_PI - r/self.R).apply(
            ROT.from_euler('x', HALF_PI).apply(
                ROT.from_euler('y', pi).apply(
                    ROT.from_euler('z', 0.5*corner_short).apply(corner_points)
                )
            ) + (0., d_corner, 0.)
        )
        # last corner arc
        corner3 = ROT.from_euler('z', -corner_offset).apply(
            ROT.from_euler('x', r/self.R).apply(
                ROT.from_euler('x', HALF_PI).apply(
                    ROT.from_euler('z', HALF_PI-corner_short).apply(
                        ROT.from_euler('x', pi).apply(corner_points)
                    )
                ) + (0., d_corner, 0.)
            )
        )
        # template arc
        points = Drawing.get_arc(self.R, a)
        course = np.vstack((
            # bottom leg
            ROT.from_euler('z', HALF_PI - 0.5*a).apply(points),
            # first corner
            corner1,
            # ascending leg
            ROT.from_euler('z', HALF_PI + 0.5*s).apply(
                ROT.from_euler('x', pi - alpha).apply(
                    ROT.from_euler('z', main_offset).apply(points))),
            # top corner
            corner2,
            # descending leg
            ROT.from_euler('z', HALF_PI - 0.5*s).apply(
                ROT.from_euler('x', alpha).apply(
                    ROT.from_euler('z', a + main_offset).apply(
                        ROT.from_euler('x', pi).apply(points)
                    )
                )
            ),
            # last corner
            corner3
        ))
        result = Scene()
        result.add(Polyline(course, size=7, color=Colors.GRAY20))  # outline border
        result.add(Polyline(course, size=3, color=Colors.WHITE))
        return result

    def _init_hor_eight(self):
        points, n = self._get_loop_pts()
        points_right = ROT.from_euler('z', -24.47, degrees=True).apply(points)
        points_left = ROT.from_euler('z', 24.47, degrees=True).apply(points)
        border_color = Colors.GRAY20
        result = Scene()
        result.add(Polyline(points_right[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_right[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_right[2*n:], size=1, color=border_color))
        result.add(Polyline(points_left[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_left[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_left[2*n:], size=1, color=border_color))
        return result

    def _init_sq_eight(self):
        points = self._get_square_loop_pts()
        points_right = ROT.from_euler('z', -EIGHTH_PI).apply(points)
        points_left = ROT.from_euler('z', EIGHTH_PI).apply(points)
        result = Scene()
        result.add(Polyline(points_right, size=3, color=Colors.WHITE))
        result.add(Polyline(points_left, size=3, color=Colors.WHITE))
        return result

    def _init_ver_eight(self):
        points_bot, n = self._get_loop_pts()
        points_top = ROT.from_euler('x', QUART_PI).apply(points_bot)
        border_color = Colors.GRAY20
        result = Scene()
        result.add(Polyline(points_bot[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_bot[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_bot[2*n:], size=1, color=border_color))
        result.add(Polyline(points_top[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_top[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_top[2*n:], size=1, color=border_color))
        return result

    def _init_hourglass(self):
        result = Scene()
        # Rough first attempt based on empirical drawing in CAD.
        # TODO: Improve this after I figure out the actual formulas.
        # Angle subtended by one leg of the equilateral spherical triangle
        alpha = radians(50.165)
        # Angle by which the tilted arcs are rotated from "flat" around the 45-degree axis through the crossover point
        beta = radians(56.5)
        # First rotations of template arcs to get them centered on the crossover later
        rotA = ROT.from_euler('z', 0.5*(pi-alpha))
        rotB = ROT.from_euler('z', 0.5*(pi-2*alpha))
        # Second rotation for the climb/descend arcs: 45deg around world X
        rot2 = ROT.from_euler('x', QUART_PI)
        # Template arcs
        # Defines top & bottom traversals
        arc1 = rotA.apply(Drawing.get_arc(self.R, alpha))
        # Defines the climb & descend arcs
        arc2 = rotB.apply(Drawing.get_arc(self.R, 2*alpha))
        # Pieces of the final hourglass
        arc_bottom = arc1
        arc_climb = rot2.apply(ROT.from_euler('y', -(pi+beta)).apply(arc2))
        arc_top = ROT.from_euler('x', HALF_PI).apply(arc1)
        arc_descend = rot2.apply(ROT.from_euler('y', pi+beta).apply(arc2))
        # The full course
        course = Polyline(
            np.vstack((
                arc_bottom,
                arc_climb,
                arc_top,
                arc_descend
            )), size=3, color=Colors.WHITE)
        result.add(course)
        return result

    def _init_ovr_eight(self):
        points, n = self._get_loop_pts()
        points_right = ROT.from_euler('xy', (HALF_PI-EIGHTH_PI, EIGHTH_PI)).apply(points)
        points_left = ROT.from_euler('y', -QUART_PI).apply(points_right)
        border_color = Colors.GRAY20
        result = Scene()
        result.add(Polyline(points_right[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_right[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_right[2*n:], size=1, color=border_color))
        result.add(Polyline(points_left[:n], size=3, color=Colors.WHITE))
        result.add(Polyline(points_left[n:2*n], size=1, color=border_color))
        result.add(Polyline(points_left[2*n:], size=1, color=border_color))
        return result

    def _init_clover(self):
        cone_half_angle = atan(sin(EIGHTH_PI))
        tilt_angle_top = EIGHTH_PI + QUART_PI
        tilt_angle_bot = EIGHTH_PI
        # radius of each loop
        r = self.R * sin(cone_half_angle)
        # distance from sphere center to center of each loop
        d = self.R * cos(cone_half_angle)
        # template loop with center at equator
        points, n = self._get_loop_pts(cone_half_angle)
        points = ROT.from_euler('x', -cone_half_angle).apply(points)
        # clover loops in the order they are performed
        loops = (
            ROT.from_euler('x', tilt_angle_top).apply(
                ROT.from_euler('z', -cone_half_angle).apply(points)),
            ROT.from_euler('x', tilt_angle_bot).apply(
                ROT.from_euler('z', cone_half_angle).apply(points)),
            ROT.from_euler('x', tilt_angle_top).apply(
                ROT.from_euler('z', cone_half_angle).apply(points)),
            ROT.from_euler('x', tilt_angle_bot).apply(
                ROT.from_euler('z', -cone_half_angle).apply(points))
        )
        # template arc for the connecting paths: centered azimuthally, lying in the XY plane
        arc = ROT.from_euler('z', QUART_PI+EIGHTH_PI).apply(Drawing.get_arc(self.R, QUART_PI))
        # arc that connects left and right halves
        arc_horz = ROT.from_euler('x', QUART_PI).apply(arc)
        # arc that connects top and bottom halves
        arc_vert = ROT.from_euler('x', QUART_PI).apply(ROT.from_euler('y', HALF_PI).apply(arc))
        # Create the scene
        border_color = Colors.GRAY20
        result = Scene()
        for loop in loops:
            result.add(Polyline(loop[:n], size=3, color=Colors.WHITE))
            result.add(Polyline(loop[n:2*n], size=1, color=border_color))
            result.add(Polyline(loop[2*n:], size=1, color=border_color))
        result.add(Polyline(arc_horz, size=3, color=Colors.WHITE))
        result.add(Polyline(arc_vert, size=3, color=Colors.WHITE))
        return result
