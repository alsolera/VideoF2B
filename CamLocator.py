# -*- coding: utf-8 -*-
# CamLocator - calculate the best location for a camera.
# Copyright (C) 2020 - 2021  Andrey Vasilik - basil96@users.noreply.github.com
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

import argparse as ap
from math import cos, degrees, pi, radians, sin, sqrt, tan

import numpy as np

import geometry as geom


def get_angle_arr(m):
    '''Return the angle between u and v, where
            u = m[1] - m[0]
            v = m[2] - m[0]'''
    u = m[1] - m[0]
    v = m[2] - m[0]
    return geom.angle(u, v)


def get_tangent_point(p, R):
    '''Given point p, find coordinates of a point q
    such that [pq] is tangent to circle at origin or radius R.'''
    d = np.linalg.norm(p)
    a = R**2 / d
    det = d**2 - R**2
    if det < 0.:
        raise Exception('Point is inside circle, tangent is undefined')
    r = R / d * sqrt(det)
    n = p / d
    n_perp = np.array([n[1], -n[0]])  # rotation of n by -90deg
    q = a * n + r * n_perp
    return q


def get_elevation_angle(p):
    '''Given point p in XY space, determine its elevation angle above x-axis.'''
    return np.arctan(abs(p[1]) / abs(p[0]))


def str_vec(u):
    '''Pretty-formatter of a 1-d vector.'''
    return f"({','.join(f'{s:.3f}' for s in u)})"


def solve_x(R, h, target_alpha, ground_level):
    from scipy.optimize import fsolve

    def func(x):
        p0 = [x[0], h]  # use x[0] because fsolve supplies an array, which throws deprecation warnings here
        try:
            p1 = get_tangent_point(p0, R)
        except:
            return 1e6
        p2 = [0, ground_level]
        P = np.vstack((p0, p1, p2))
        alpha = get_angle_arr(P)
        return target_alpha - alpha

    ret = fsolve(func, R+h)
    return ret[0]


def main(R, h, max_alpha, ground_level):
    '''For a given `h` (signed height relative to equator),
        a given sphere radius `R`,
        and a given vertical view angle `max_alpha` in degrees, find:
            `d_min` and `d_max` such that `alpha <= max_alpha`.
        Center marker is at `ground_level` relative to equator.'''
    p2 = [0, ground_level]
    max_alpha = radians(max_alpha)
    # 45deg tangent line equation
    quart_pi = 0.25 * pi
    b = R * (sin(quart_pi)+cos(quart_pi))
    # Find `x_max` (where tangent line intersects the camera height)
    x_max = h - b
    p0_max = [x_max, h]
    p1_max = get_tangent_point(p0_max, R)
    P_max = np.vstack((p0_max, p1_max, p2))
    alpha_at_max = get_angle_arr(P_max)
    # `x_min` is at first limited by sphere radius
    x_min = -R
    # The other limit for `x_min` is `max_alpha`
    x_min_for_alpha = solve_x(R, h, max_alpha, ground_level)
    # Both x_min values are negative, so take the min of the two.
    x_min = min(x_min, x_min_for_alpha)
    # Use absolute value for the user because it's just a distance.
    d_min = abs(x_min)
    d_max = abs(x_max)
    # Output
    d_max_str = f'{d_max:.2f}'
    alpha_at_max_str = f'{degrees(alpha_at_max):.2f}°'
    elev_max_str = f'{degrees(get_elevation_angle(p1_max)):.2f}°'
    if d_min <= d_max:
        p0_min = [x_min, h]
        p1_min = get_tangent_point(p0_min, R)
        P_min = np.vstack((p0_min, p1_min, p2))
        alpha_at_min = get_angle_arr(P_min)
        d_min_str = f'{d_min:.2f}'
        alpha_at_min_str = f'{degrees(alpha_at_min):.2f}°'
        elev_min_str = f'{degrees(get_elevation_angle(p1_min)):.2f}°'
    else:
        d_min_str = alpha_at_min_str = elev_min_str = 'N/A'
    w = 8
    print('=' * 80)
    print('--- INPUTS ----')
    print(f'                Sphere radius R = {R:.2f}')
    print(f'      Camera height wrt equator = {h:.2f}')
    print(f'       Ground level wrt equator = {ground_level:.2f}')
    print(f'    Maximum vertical view angle = {degrees(max_alpha):.2f}°')
    print('--- RESULTS ---')
    print(f'                   |---------------------|')
    print(f'                   | Nearest  | Farthest |')
    print(f'    |--------------+---------------------|')
    print(f'    | Cam distance | {d_min_str.center(w)} | {d_max_str.center(w)} |')
    print(f'    | View angle   | {alpha_at_min_str.center(8)} | {alpha_at_max_str.center(w)} |')
    print(f'    | Tangent elev | {elev_min_str.center(w)} | {elev_max_str.center(w)} |')
    print(f'    |--------------+---------------------|')
    print('=' * 80)
    print('''Camera max distance is such that the 45° elevation at the visible edge of the
sphere fits exactly at the top of the video frame. Beyond that distance the 45°
elevation will appear below the visible edge of the sphere in video.

Camera min distance is such that the entire scene still fits vertically within
the frame while the camera is safely outside the flight sphere.

For reference:
    * A 10mm lens on APS-C 1.5x crop sensor with 16:9 crop in a 3:2 camera
      results in 68.04° vertical angle of view.
    * A 14mm lens on full-frame sensor with 16:9 crop in a 3:2 camera
      results in 71.75° vertical angle of view.''')
    print('=' * 80)


if __name__ == '__main__':
    DEFAULT_R = 21.0  # default sphere radius
    DEFAULT_C = -1.0  # default camera height relative to equator
    DEFAULT_G = -1.5  # default ground level relative to equator
    DEFAULT_A = 71.75  # default maximum vertical viewing angle of the camera, in degrees
    AP = ap.ArgumentParser(description='Camera position calculator for VideoF2B application.')
    AP.add_argument('-r', required=False, default=DEFAULT_R, type=float,
                    help=f'Radius of sphere [default = {DEFAULT_R}]')
    AP.add_argument('-c', required=False, default=DEFAULT_C, type=float,
                    help=f'Height of camera relative to equator (up is positive) [default = {DEFAULT_C}]')
    AP.add_argument('-g', required=False, default=DEFAULT_G, type=float,
                    help=f'Height of ground level relative to equator (up is positive) [default = {DEFAULT_G}]')
    AP.add_argument('-a', required=False, default=DEFAULT_A, type=float,
                    help=f'Maximum vertical view angle of camera, in degrees [default = {DEFAULT_A}°]')
    args = AP.parse_args()
    main(R=args.r, h=args.c, max_alpha=args.a, ground_level=args.g)
