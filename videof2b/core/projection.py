# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2020 - 2021  Andrey Vasilik - basil96
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
Project image points onto world sphere.
'''

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def projectSpherePointsToImage(cam, world_pts, frame=None):
    img_pts, _ = cv2.projectPoints(world_pts, cam.rvec, cam.tvec, cam.newcameramtx, cam.dist)
    img_pts = np.int32(img_pts).reshape(-1, 2)
    return img_pts


def projectImagePointToSphere(frame, cam, radius, center, img_point, data_writer):
    '''Project image point to world sphere given a calibrated camera and frame.

    :param frame: the image frame of interest.
    :param cam: instance of CalCamera.
    :param radius: radius of sphere.
    :param center: XYZ location of sphere center wrt world pose estimation.
    :param img_point: the image point that we wish to project onto the sphere.
    :param data_writer: a handle to a file-like object.
    '''
    logger.debug(f'img_point = {img_point}')
    # Augment point for transform. This is the {u v 1} vector.
    pt_px = np.vstack((img_point[0], img_point[1], 1))  # TODO: I think there's a faster syntax for this..
    # logger.debug(f'pt_px.T = {pt_px.T} [{len(detector.pts_scaled)} points.]')
    avec = cam.qmat.dot(pt_px)
    # logger.debug(f'avec =\n{avec}')
    a1, a2, a3 = avec[:, 0]
    t1, t2, t3 = cam.rtvec[:, 0]
    cx, cy, cz = center
    # logger.debug(f'a1 = {a1}')
    # logger.debug(f'a2 = {a2}')
    # logger.debug(f'a3 = {a3}')
    # logger.debug(f't1 = {t1}')
    # logger.debug(f't2 = {t2}')
    # logger.debug(f't3 = {t3}')
    logger.debug(f'cx = {cx}')
    logger.debug(f'cy = {cy}')
    logger.debug(f'cz = {cz}')
    coeffs = (
        -(a1**2) - (a2**2) - (a3**2),

        2.*(a1*(t1+cx) + a2*(t2+cy) + a3*(t3+cz)),

        radius**2 - 2.*(t1*cx + t2*cy + t3*cz) -
        (t1**2 + t2**2 + t3**2 + cx**2 + cy**2 + cz**2)
    )
    logger.debug(f'coeffs = {coeffs}')
    # logger.debug(f'coeffs = {coeffs}')
    # Value of determinant (b^2 - 4ac) determines type of solution (no intersection, tangent point, or two points)
    # determinant = coeffs[1]**2 - 4.*coeffs[0]*coeffs[2]
    roots = np.roots(coeffs)
    logger.debug(f'roots = {roots}')
    pts_world = None
    if np.all(abs(roots.imag) < 1e-7):
        pts_world = np.vstack(
            [root * cam.qmat.dot(pt_px) - cam.rtvec for root in roots]).reshape(-1, 3)
        # logger.debug(pts_world.shape)
        # logger.debug(pts_world)

        # sanity check 1: norm of each point in pts_world should be close to sphere radius
        norms_world = np.linalg.norm(pts_world, axis=1)
        logger.debug(f'norms_world = {norms_world}')

        # sanity check 2: project pts_world back into video and compare against img_point
        imgPtsCheck, _ = cv2.projectPoints(pts_world, cam.rvec, cam.tvec,
                                           cam.newcameramtx, cam.dist)
        imgPtsCheck = np.int32(imgPtsCheck).reshape(-1, 2)
        # logger.debug(f'imgPtsCheck = {imgPtsCheck}')
        logger.debug(f'imgPtsCheck - img_point =\n{imgPtsCheck - img_point}')
        for ipt, img_pt_check in enumerate(imgPtsCheck):
            # logger.debug(f'img_pt_check - img_point = {img_pt_check - img_point}')
            if ipt == 0:
                p_rad = 9
                p_col = (255, 0, 0)
            else:
                p_rad = 3
                p_col = (0, 0, 255)
            cv2.circle(frame, (img_pt_check[0], img_pt_check[1]), p_rad, p_col, -1)

        logger.debug(f'pts_world =  shape={pts_world.shape}')
        # logger.debug(pts_world)
        for pw in pts_world:
            pstr = ','.join(f'{coord:+.6f}' for coord in pw)
            logger.debug(f"  ({pstr})")
            data_writer.write(f'{pstr},')
        for root in roots:
            data_writer.write(f'{root:.6f},')
        data_writer.write('\n')
        # print()
        # break
    else:
        # We have complex roots, so there is no solution.
        # data_writer.write(','.join(str(x) for x in [np.nan]*6))
        # for root in roots:
        #     data_writer.write(f'{root:.6f},')
        # data_writer.write('\n')
        logger.debug(
            '=====#################### roots are complex, no solution ####################=====')
    return pts_world
