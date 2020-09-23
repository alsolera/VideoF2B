# VideoF2B - Draw F2B figures from video
# Copyright (C) 2018  Alberto Solera Rico - albertoavion(a)gmail.com
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

from collections import deque

import cv2
import imutils
from numpy import asarray
from numpy.linalg import norm


class Detector(object):
    def __init__(self, maxlen, scale):
        # Create Background subtractor object
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=4, varThreshold=15, detectShadows=False)
        self.pts = deque(maxlen=maxlen)
        self.pts_scaled = deque(maxlen=maxlen)
        self.scale = scale

    def process(self, img):
        blurred = cv2.GaussianBlur(img, (7, 7), 0)
        mask = self.subtractor.apply(blurred)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)

        # find contours in the mask and initialize the current (x, y) center
        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        center = None
        center_scaled = None

        # only proceed if at least one contour was found
        if cnts is not None and len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute the centroid
            c = max(cnts, key=cv2.contourArea)
            #((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                center_scaled = (int(self.scale * M["m10"] / M["m00"]),
                                 int(self.scale * M["m01"] / M["m00"]))
            else:
                center = None
                center_scaled = None

            if len(self.pts) > 0 and self.pts[0] is not None:
                # print(pts[0], center)
                if norm(asarray(self.pts[0]) - asarray(center)) > 30:
                    center = None
                    center_scaled = None

        # update the points queue
        self.pts.appendleft(center)
        self.pts_scaled.appendleft(center_scaled)

    def clear(self):
        self.pts.clear()
        self.pts_scaled.clear()
