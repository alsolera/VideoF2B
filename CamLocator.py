# CamLocator - calculate the best location for a camera.
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

import numpy as np


def get_angle_arr(m):
    '''Return the angle between u and v, where
            u = m[1] - m[0]
            v = m[2] - m[0]'''
    u = m[1] - m[0]
    v = m[2] - m[0]
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm * v_norm == 0.0:
        return np.nan
    return np.arccos(u.dot(v) / (u_norm * v_norm))


def get_tangent_point(p, R):
    '''Given point p, find coordinates of a point q
    such that [pq] is tangent to circle at origin or radius R.'''
    d = np.linalg.norm(p)
    a = R**2 / d
    det = d**2 - R**2
    if det < 0.:
        raise Exception('Point is inside circle, tangent is undefined')
    r = R / d * np.sqrt(det)
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


def main():
    # R = 15.92  # F2D lines
    # R = 62. * 0.3048  # 60ft lines C-C plus 2ft of arm
    R = 66. * 0.3048    # 64ft lines C-C plus 2ft of arm
    # x coord is at intersection of equator and 45° tangent line
    # p0 = np.array([-R * np.sqrt(2) + 2, -1.])
    # p0 = np.array([-21.9, -1.])
    p0 = np.array([-24.0, -0.5])
    # p0 = np.array([-25, -1.])
    # p0 = np.array([-100.*.3048, -1.])
    p1 = get_tangent_point(p0, R)
    p2 = np.array([0., -1.55])
    P = np.vstack((p0, p1, p2))
    view_angle = get_angle_arr(P)
    print('=' * 80)
    print(f'              Sphere radius R = {R}')
    print(f'              Camera location = {str_vec(p0)}')
    print(f'   Tangent from cam to sphere = {str_vec(p1)}')
    print(f"Sphere center at pilot's feet = {str_vec(p2)}")
    print(f'           Minimum view angle = {np.degrees(view_angle):.2f}°')
    print(f'   Elevation angle at tangent = {np.degrees(get_elevation_angle(p1)):.2f}°')
    print('=' * 80)
    print('''For reference:
    * A 10mm lens on APS-C 1.5x crop sensor with 16:9 crop in a 3:2 camera
      results in 68.04° vertical angle of view.
    * A 14mm lens on full-frame sensor with 16:9 crop in a 3:2 camera
      results in 71.75° vertical angle of view.''')
    print('=' * 80)


if __name__ == '__main__':
    main()
