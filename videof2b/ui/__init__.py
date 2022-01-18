# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021  Andrey Vasilik - basil96
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

'''The :mod:`ui` module provides the core user interface for VideoF2B'''


# Video file extensions obtained from `EXTENSIONS_VIDEO` in
# 'include/vlc_interface.h' from vlc 3.0.x source at
# https://github.com/videolan/vlc/blob/master/include/vlc_interface.h
# NOTE: There is no guarantee that OpenCV can open all of these, as the
# answer to this question is not entirely clear. For some details, see
# https://stackoverflow.com/questions/20852514/list-file-extensions-supported-by-opencv
# tl;dr: Use `cv2.isOpened()` to validate input.
EXTENSIONS_VIDEO = (
    "*.3g2", "*.3gp", "*.3gp2", "*.3gpp",
    "*.amv", "*.asf", "*.avi",
    "*.bik", "*.bin",
    "*.crf",
    "*.dav", "*.divx", "*.drc", "*.dv", "*.dvr-ms",
    "*.evo",
    "*.f4v", "*.flv",
    "*.gvi", "*.gxf",
    "*.iso",
    "*.m1v", "*.m2v", "*.m2t", "*.m2ts", "*.m4v", "*.mkv", "*.mov", "*.mp2",
    "*.mp2v", "*.mp4", "*.mp4v", "*.mpe", "*.mpeg", "*.mpeg1", "*.mpeg2",
    "*.mpeg4", "*.mpg", "*.mpv2", "*.mts", "*.mtv", "*.mxf", "*.mxg",
    "*.nsv", "*.nuv",
    "*.ogg", "*.ogm", "*.ogv", "*.ogx",
    "*.ps",
    "*.rec", "*.rm", "*.rmvb", "*.rpl",
    "*.thp", "*.tod", "*.tp", "*.ts", "*.tts", "*.txd",
    "*.vob", "*.vro",
    "*.webm", "*.wm", "*.wmv", "*.wtv",
    "*.xesc"
)
