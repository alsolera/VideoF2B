#!/usr/bin/env python3
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

'''Package build/setup script.'''

from setuptools import setup, find_packages

VERSION = '0.6'

setup(
    name='VideoF2B',
    version=VERSION,
    description="Draw F2B figures from video.",
    long_description='''\
VideoF2B is an open-source desktop application for tracing F2B Stunt competition flights in video.''',
    # Classifiers from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Microsoft :: Windows :: Windows 7',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Desktop Environment :: Gnome',
        'Topic :: Desktop Environment :: K Desktop Environment (KDE)',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'Topic :: Multimedia :: Video',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    keywords='open source video hobby aeromodeling f2b stunt flight tracing competition judging',
    author='Alberto Solera',
    author_email='albertoavion(a)gmail.com',
    url='http://videof2b.blogspot.com/',
    license='GPL-3.0-or-later',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.7',
    dependency_links=['https://github.com/basil96/imutils/tarball/master'],
    install_requires=[
        'numpy>=1.21.1',
        'platformdirs>=2.2.0',
        'opencv-python >= 4.5.3; platform_system=="Windows"',
        'PySide6 >= 6.1.2',
        'scipy>=1.7.1'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
            # 'pytest-qt'
        ]
    },
    setup_requires=['pytest-runner'],
    entry_points={'gui_scripts': ['videof2b = videof2b.__main__:main']}
)
